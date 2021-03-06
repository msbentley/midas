#!/usr/bin/python
"""dds_utils.py - module to interact with the ESA DDS"""

import pandas as pd
import os, time, socket
from midas import common, ros_tm, socks
from dateutil import parser

import logging
log = logging.getLogger(__name__)

# Global module level definitions
dds_req_params = ['request-id', 'filename', 'directory', 'apid', 'start_time', 'end_time']
dds_targets = ['home', 'iwf_ftp', 'pisa']

# these hosts do not need to tunnel through a socks server
nosocks_hosts = ['phys-mab', 'midas']

# servers need a different pathname
servers = ['comp-l05', 'comp-l06', 'comp-l08', 'comp-l09']

# ISO date format
isofmt = "%Y-%m-%dT%H:%M:%SZ"

dds_wait_time=1*60. # 1 minute

# Set default paths

template_file = 'midas_dds_request.xml'
template_file = os.path.join(common.config_path, template_file)

tcp_template = os.path.join(common.config_path,'tcp_dds_request.xml')
single_template = os.path.join(common.config_path,'generic_dds_req.xml')
cmh_template = os.path.join(common.config_path,'midas_cmh_request.xml')

schema_file = 'GDDSRequest.xsd'
schema_file = os.path.join(common.config_path, schema_file)

obs_list_file = 'observations.csv'
obs_list_file = os.path.join(common.config_path, obs_list_file)

pkt_file = 'sc_packets.csv'
pkt_file = os.path.join(common.config_path, pkt_file)

def validate_xml(xml, schema_file, isfile=False):
    """Validate a DDS request (or other) XML file against a given schema"""

    # LXML validates XML on parse
    # http://lxml.de/validation.html

    from lxml import etree

    schema_doc = etree.parse(schema_file)

    current_dir = os.getcwd()
    schema_dir =  os.path.dirname(os.path.abspath(schema_file))

    # the top level schema often calls others - they are assumed
    # to be in the same directory, so first move there
    os.chdir(schema_dir)

    schema = etree.XMLSchema(schema_doc)
    # xmlparser = etree.XMLParser(schema=schema)

    os.chdir(current_dir)

    try:
        if isfile:
            with open(xml, 'r') as f:
                schema.validate(etree.parse(f))
        else:
            schema.validate(etree.fromstring(xml))
            log.debug('XML validation against schema %s successful' % (schema_file))
        return True
    except etree.XMLSyntaxError as e:
        log.error('XML parse error: %s' % e)
        log.info('error occured at line %d column %d' % e.position)
        return False
    except IOError as e:
        print "ERROR: I/O error({0}): {1}".format(e.errno, e.strerror)


def generate_request(start_time, end_time, apid, sid, pkt_type=None, pkt_subtype=None, template_file=template_file, target=False):
    """Generates and returns a DDS XML request based on a template,
    start and end time and an APID."""

    import re, datetime

    if not os.path.isfile(template_file):
        log.error('template file %s not found' % (template_file))
        return 0

    # validate inputs
    # if apid is not None:
    #     if apid not in ros_tm.midas_apids:
    #         log.warning('APID %i invalid for MIDAS' % apid)

    if type(start_time)!=pd.Timestamp:
        start_time = pd.Timestamp(start_time)

    if type(end_time)!=pd.Timestamp:
        end_time = pd.Timestamp(end_time)

    if (type(start_time) or type(end_time)) is not pd.Timestamp:
        log.error('start and end times must be given as a Timestamp!')
        return False

    # start_time = start_time.isoformat()
    # end_time = end_time.isoformat()

    start_time = start_time.strftime(isofmt)
    end_time = end_time.strftime(isofmt)

    if os.path.basename(template_file)=='midas_cmh_request.xml':
        request = 'MD_CMH_%s--%s' % (start_time, end_time)
    else:
        if sid is None:
            request = 'MD_TLM_%i_%s--%s' % (apid, start_time, end_time)
        else:
            request = 'MD_TLM_%i_%i_%s--%s' % (apid, sid, start_time, end_time)

    request = request.replace(":","")

    if target and target not in dds_targets:
        log.error('invalid DDS target server %s' % target)
        return False
    elif not target:
        target = 'home'

    # create a dictionary of parameter/value entries for the XML template
    request_params = { \
                'request-id': request,
                'filename' : request+'.DAT',
                'directory' : './',
                'target' : target,
                'apid' : apid,
                'start_time' : start_time,
                'end_time' : end_time }

    if pkt_type is not None and pkt_subtype is not None and sid is not None:
        request_params.update( {
            'pkt_type': pkt_type,
            'pkt_subtype': pkt_subtype,
            'sid': sid })

    f = open(template_file, 'r')
    template = f.read()
    f.close()

    # find positions of opening and closing tags ('[', ']')
    opentag = [tag.start() for tag in re.finditer('\[',template)]
    closetag = [tag.start() for tag in re.finditer('\]',template)]

    # check we have an equal number of open and close tags
    if len(opentag) != len(closetag):
        log.error('template file %s has %i open and %i closing tags - must be equal!' \
            % (template_file, len(opentag), len(closetag)))

    # get the list of unique tags
    tags = [template[opentag[n]+1:closetag[n]] for n in range(len(opentag))]
    tags = list(set(tags))

    log.debug('XML template file %s opened with %i unique tags' % (template_file, len(tags)))

    # check that all tags are included in the params dictionary
    matches = sum([True for key in tags if key in request_params])
    if matches < len(tags):
        log.error('%i tags in template %s but only %i parameters given' % \
            (len(tags), template_file, len(['params'])))
        return False

    # open the file again and search and replace all tags
    f = open(template_file, 'r')
    template = f.read()
    f.close()

    for tag in tags:
        template = template.replace('['+tag+']',str(request_params[tag]))

    return template, request


def submit_request(template, request, socks):
    """Submit a DDS request via FTP. Since access is only allowed
    through certain IPs, and to maintain portability, SocksiPi is
    used to tunnel via SOCKS: https://pypi.python.org/pypi/SocksiPy"""

    request_dir = '/home/roreq/incoming/'

    import ftplib, StringIO

    # TODO: overriding socks flag for now to automatically switch according to host
    if socket.gethostname() not in nosocks_hosts: socks = True

    if socks:

        tunnel = open_tunnel()
        if tunnel.returncode:
            print('Tunnel open return code: %i' % (tunnel.returncode))

        time.sleep(20) # tunnel stays open, hence no return code and just need to wait

        import socks as proxy
        proxy.setdefaultproxy(proxy.PROXY_TYPE_SOCKS5, 'localhost', 1080)
        proxy.wrapmodule(ftplib)

    try:
        ftp = ftplib.FTP('rodda.esoc.ops.esa.int', timeout=30)
        ftp.login('roreq', 'rod6$')
    except Exception as e:
        log.error('exception: %s' % e)
        return None, None


    # check we're in the right directory (should be by default)
    cwd = ftp.pwd()
    if cwd != request_dir: ftp.cwd(request_dir)

    buf = StringIO.StringIO(template)
    put_result = ftp.storlines('STOR %s.tmp' % request, buf)
    rename_result = ftp.rename('%s.tmp' % request ,'%s.XML' % request)

    log.debug('FTP results (put, rename): %s %s' % (put_result, rename_result))

    if socks:
        tunnel.kill()
        proxy.setdefaultproxy()

    return (put_result, rename_result)


def request_data_by_apid(start_time, end_time, apid=False, target=False, socks=False, template_file=template_file):
    """General XML, validate and submit to the DDS in a given time frame. All standard
    date/time strings are accapted for the start/end times. If the optional apid= keyword is set,
    only this APID will be requested, otherwise all MIDAS APIDs are used.

    The optional keyword target= can be used to specify an alternate server configured with the DDS,
    otherwise the default is used.

    If socks=True then a socks5 tunnel to the appropriate server must already be open and listening
    on port 1080."""

    from datetime import datetime

    st = parser.parse(start_time) if (type(start_time) != pd.tslib.Timestamp and type(start_time) != datetime) else start_time
    et = parser.parse(end_time) if type(end_time) != pd.tslib.Timestamp else end_time

    filenames = []

    if type(apid) != list and type(apid) != bool: apid = [apid]
    aplist = apid if apid else ros_tm.midas_apids

    for ap in aplist:
        # (start_time, end_time, apid, sid, pkt_type=None, pkt_subtype=None, template_file=template_file, target=False)
        log.info('building request for APID %d' % ap)
        xml, request_id = generate_request(start_time=st, end_time=et, apid=ap, sid=None, pkt_type=None,
            pkt_subtype=None, template_file=template_file, target=target)
        filenames.append(request_id+'.DAT')

        if validate_xml(xml, schema_file):
            submit_request(xml, request_id, socks)
            pass
        else:
            log.error('problem validating XML against schema, request not submitted')
            return None, None

    log.info('%d requests submitted to the DDS' % len(filenames))

    return filenames, aplist


def get_data(start_time, end_time, outputfile=False, outputpath='.', apid=False, socks=False, delfiles=True, max_retry=5, retry_delay=2):
    """Pull data from the DDS. The time interval should be given via start_time and end_time in any sane format.

    If outputfile=False, separate files are created for each APID. If true, one file is created with the name given by outputfile.
    All files are places in outputpath ('.' by default)
    If apid=False, ALL MIDAS APIDs are requested, otherwise APIDs can be passed.
    If delfiles=True, intermediate files are deleted after a combined outputfile has been created.
    max_retry= and retry_delay= control how often and how many times the MIDAS server is polled for returned DDS files"""

    filenames, aplist = request_data_by_apid(start_time, end_time, apid=apid, socks=socks)

    log.info('waiting for DDS to service requests before starting retrieval...')
    time.sleep(dds_wait_time) # wait a few minutes before accessing the data via SFTP

    filelist = get_files(filenames, outputpath=outputpath, apid=aplist, outputfile=outputfile, delfiles=delfiles, max_retry=max_retry, retry_delay=retry_delay)

    return filelist, apid


def request_data(start_time, end_time, apid, sid, pkt_type, pkt_subtype, target=False, socks=False, template_file=single_template):
    """Requests data for a given packet type, sub-type and APIDs, between start_time and end_time."""

    from datetime import datetime

    st = parser.parse(start_time) if (type(start_time) != pd.tslib.Timestamp and type(start_time) != datetime) else start_time
    et = parser.parse(end_time) if type(end_time) != pd.tslib.Timestamp else end_time

    if apid is not None:
        packet = ros_tm.pid[ (ros_tm.pid.apid==apid) & (ros_tm.pid.type==pkt_type) & (ros_tm.pid.subtype==pkt_subtype) & (ros_tm.pid.sid==sid) ]
        if len(packet)==0:
            log.error('no packet found in the database for APID %i, SID %i, type %i, subtype %i' % (apid, sid, pkt_type, pkt_subtype))
            return False
        elif len(packet)>1:
            log.error('more than one packet found matching APID %i, SID %i, type %i, subtype %i' % (apid, sid, pkt_type, pkt_subtype))
        else:
            log.info('building request for packet %s' % packet.description.squeeze())

    # start_time, end_time, apid, pkt_type=None, pkt_subtype=None, template_file=template_file, target=False
    xml, request_id = generate_request(start_time=st, end_time=et, apid=apid, sid=sid, pkt_type=pkt_type, pkt_subtype=pkt_subtype, template_file=template_file, target=target)
    filename = request_id + '.DAT'

    if validate_xml(xml, schema_file):
        submit_request(xml, request_id, socks)
        pass
    else:
        log.error('problem validating XML against schema. Request not submitted')
        return None

    log.info('request submitted to the DDS')

    return filename


def get_data_since(start_time, outputfile, outputpath='.', apid=False, socks=False, max_retry=5, retry_delay=2):
    """Calls request_data_by_apid_since() and retrieves packets from remote SFTP site"""

    filenames, aplist = request_data_by_apid_since(start_time, apid=apid, socks=socks)

    log.info('waiting for DDS to service requests before starting retrieval...')
    time.sleep(dds_wait_time) # wait a few minutes before accessing the data via SFTP

    filelist = get_files(filenames, outputpath=outputpath, apid=aplist, outputfile=outputfile, max_retry=max_retry, retry_delay=retry_delay)

    return filelist, apid


def get_new_data(since, outputfile, outputpath, apid=False, socks=False, max_retry=5, retry_delay=2):

    filenames, aplist = request_new_data(since, apid=apid, socks=socks)
    get_files(filenames, outputpath, outputfile=outputfile, apid=aplist, max_retry=max_retry, retry_delay=retry_delay)


def request_data_by_apid_since(start_time, apid=False, socks=False):
    """Calls request_data_by_apid() from a given date until the present"""

    from datetime import datetime

    end_time = datetime.now().strftime(isofmt)
    filenames, aplist = request_data_by_apid(start_time, end_time, apid, socks=socks)


    return filenames, aplist


def request_new_data(since, apid=False, socks=False):
    """Calls request_data_by_apid() for a period specified by the current time and since, in hours"""

    from datetime import datetime, timedelta, date

    end_time = datetime.now()
    start_time = (end_time - timedelta(hours=since) )

    start_time = date.strftime(start_time, '%c')
    end_time = date.strftime(end_time, '%c')

    filenames, aplist = request_data_by_apid(start_time, end_time, apid, socks=socks)

    return filenames, aplist



def get_files(filenames, outputpath, apid=False, outputfile=False, delfiles=True, strip_dds=False, max_retry=5, retry_delay=2):
    """Retrieves requested files from SFTP site. If outputfile= is set the
    frames are combined into a single packet with the given name given by
    outputpath/outputfile.

    If delfiles=True the individual source files are locally deleted."""

    import shutil

    # retrieve data from the SFTP server, skipping missing or zero byte files
    retrieved = retrieve_data(filenames, outputpath, max_retry=max_retry, retry_delay=retry_delay)

    # return an empty filelist if no matching files are on server
    if len(retrieved)==0: return []

    if type(filenames)!=list:
        filenames=[filenames]

    if apid:
        if type(apid)!=list:
            apid=[apid]

    if outputfile:

        if len(retrieved)==1: #simple copy
            shutil.copy(retrieved[0], os.path.join(outputpath, outputfile))
        else:
            tm = ros_tm.tm()
            for localfile in retrieved:
                # tm.get_pkts(localfile, apid=apid[filenames.index(os.path.basename(localfile))], append=True)
                tm.get_pkts(localfile, append=True, simple=True)

            tm.write_pkts(outputfile, outputpath=outputpath, strip_dds=strip_dds)
            log.info('combined TM written to file %s' % (outputfile))

    if delfiles and outputfile: [os.remove(localfile) for localfile in retrieved]

    return retrieved if not (outputfile and delfiles) else os.path.join(outputpath,outputfile)



def retrieve_data(filelist, localpath='.', max_retry=5, retry_delay=2):
    """Checks for a given file on a remote SFTP server and retrieves it if
    it exists. Default are set for the MIMAS server and current working directory.

    Returns a list of files successfully retreived.

    If an expected file is not found, another attempted will be made after a
    delay of retry_delay= minutes, for a maximum of max_retry= attempts."""

    if type(filelist) is not list: filelist = [filelist] # deal with single files

    # Open an SSH connection to the MIMAS server
    ssh = sftp()
    ssh.open()

    retrieved = []
    remaining = list(filelist)

    retry = 1 # first attempt

    while (retry < max_retry) and (len(remaining) > 0):

        # refresh list of files available on the server
        files = ssh.sftp.listdir()
        log.debug('list of files on server: %s' % " ".join(files))

        for filename in remaining:

            log.debug('processing file %s' % (filename))

            if filename not in files:
                log.warning('file %s not found on the server' % (filename))
                continue

            stat = ssh.sftp.stat(filename)
            if stat.st_size==0:
                log.warning('file %s has zero size, skipping...' % (filename))
                remaining.remove(filename)
                ssh.sftp.remove(filename)
                continue

            try:
                ssh.sftp.get(filename, os.path.join(localpath,filename))
                ssh.sftp.remove(filename)
            except Exception as e:
                log.error('error getting/deleting file %s' % filename)
                log.error('exception: %s' % e)
                continue
            remaining.remove(filename)
            retrieved.append( os.path.join(localpath,filename) )
            log.info('file %s retrieved and removed from the server' % (filename))

        if len(remaining)==0: break

        log.info('%i files remaining, waiting %i minutes to retry (attempt %i/%i)' %
            (len(remaining), retry_delay, retry+1, max_retry))

        retry += 1

        time.sleep(retry_delay*60)

    ssh.close()

    if len(remaining) > 0:
        log.warning('unable to retrieve %i files' % (len(remaining)))
        print(remaining)

    return retrieved


def get_timecorr(outputpath='.', socks=False, max_retry=5, retry_delay=5):

    from datetime import datetime

    # type                                            190
    # subtype                                          40
    # apid                                           1966
    # sid                                               0
    # spid                                            140
    # description    Time Correlator Coefficients Packets

    apid = 1966
    sid = 0
    pkt_type = 190
    subtype = 40

    start_time = parser.parse('2 March 2004  07:17').strftime(isofmt) # Rosetta launch date
    end_time = datetime.utcnow().strftime(isofmt)

    # filenames, aplist = request_data_by_apid(start_time, end_time, apid=1966, socks=socks, template_file=tcp_template)
    filename = request_data(start_time, end_time, apid=apid, sid=sid, pkt_type=pkt_type, pkt_subtype=subtype, socks=socks, template_file=single_template)

    log.info('waiting for DDS to service requests before starting retrieval...')
    time.sleep(dds_wait_time) # wait a few minutes before accessing the data via SFTP

    filelist = get_files(filename, outputpath=outputpath, apid=apid, outputfile='TLM__MD_TIMECORR.DAT', max_retry=max_retry, retry_delay=retry_delay)

    return filelist



def get_tc_history(start_time, end_time, outputpath='.', outputfile='CMH.DAT', socks=False, max_retry=5, retry_delay=5):

    from datetime import datetime

    # st = parser.parse(start_time) if (type(start_time) != pd.tslib.Timestamp and type(start_time) != datetime) else start_time
    # et = parser.parse(end_time) if type(end_time) != pd.tslib.Timestamp else end_time

    filename = request_data(start_time, end_time, apid=None, sid=None, pkt_type=None, pkt_subtype=None, socks=socks, template_file=cmh_template)

    if filename is None:
        return None

    log.info('waiting for DDS to service requests before starting retrieval...')

    time.sleep(dds_wait_time) # wait a few minutes before accessing the data via SFTP

    filelist = get_files(filename, outputpath=outputpath, apid=None, outputfile=outputfile, max_retry=max_retry, retry_delay=retry_delay)

    return filelist



def get_tc_hist_per_obs(outputdir='.', socks=False, max_retry=10, retry_delay=5):

    observations = read_obs_file()

    obs_filenames = []

    for idx, obs in observations.iterrows():

        start = obs.start.isoformat()
        end = obs.end.isoformat()

        padded_name = obs.observation + (20-len(obs.observation)) * '_'
        cmh_filename = 'CMH__MD_M%03d_S%03d_%s_COUNT_%02d.DAT' % (obs.mtp, obs.stp, padded_name, obs.cnt)

        filename = get_tc_history(start, end, outputpath=outputdir, outputfile=cmh_filename, max_retry=max_retry, retry_delay=retry_delay)
        obs_filenames.append(os.path.join(outputdir, cmh_filename))

    return



def get_single_pkt(start_time, end_time, apid, sid, pkt_type, pkt_subtype, outputfile=False, outputpath='.', socks=False, delfiles=True, max_retry=5, retry_delay=2):

    # request_data(start_time, end_time, apid, pkt_type, pkt_subtype, target=False, socks=False, template_file=single_template):
    filename = request_data(start_time, end_time, apid, sid, pkt_type, pkt_subtype, socks=socks)

    if filename is None:
        return None

    log.info('waiting for DDS to service requests before starting retrieval...')
    time.sleep(dds_wait_time) # wait a few minutes before accessing the data via SFTP

    filelist = get_files(filename, outputpath=outputpath, apid=apid, outputfile=outputfile, delfiles=delfiles, max_retry=max_retry, retry_delay=retry_delay)

    return filelist



def get_pkts_from_list(start_time, end_time, filename, outputfile=False, outputpath='.', socks=False, delfiles=True, max_retry=10, retry_delay=10):

    filelist = []

    packets = pd.read_csv(filename, skipinitialspace=True, na_values='None')

    for idx, packet in packets.iterrows():

        # def request_data_by_apid(start_time, end_time, apid=False, target=False, socks=False, template_file=template_file):
        # def request_data(start_time, end_time, apid, sid, pkt_type, pkt_subtype, target=False, socks=False, template_file=single_template):

        if pd.isnull(packet.type):

            # request_data_by_apid(start_time, end_time, apid=False, target=False, socks=False, template_file=template_file)
            # returns apids, filelist
            filename, apid = request_data_by_apid(start_time=start_time, end_time=end_time, apid=int(packet.apid))
            if filename is None:
                log.error('something went wrong building the request, skipping')
            else:
                filelist.append(filename[0])

        else:

            # request_data(start_time, end_time, apid, sid, pkt_type, pkt_subtype, target=False, socks=False, template_file=single_template)
            # returns filename
            filename = request_data(start_time, end_time, apid=int(packet.apid), sid=int(packet.sid), pkt_type=int(packet.type),
                pkt_subtype=int(packet.subtype))
            filelist.append(filename)

        time.sleep(10) # try to avoid problems at the DDS end

    log.info('waiting for DDS to service requests before starting retrieval...')
    time.sleep(dds_wait_time) # wait a few minutes before accessing the data via SFTP

    gotfiles = get_files(filelist, outputpath=outputpath, apid=False,
        outputfile=outputfile, delfiles=delfiles, max_retry=max_retry, retry_delay=retry_delay)

    return filelist, gotfiles



def add_observations(evf_file):
    """Adds all observations (events) from a given EVF file to the MIDAS event
    list (which is polled daily to retrieve data from complete observations)"""

    import planning
    from datetime import datetime

    evffile = os.path.basename(evf_file)

    # Extract MTP and STP cycle from filename
    # e.g. EVF__MD_M004_S005_01_A_RSM0PIM1.evf
    if evffile[0:4] != 'EVF_':
        log.error('EVF files must start with prefix EVF_')
        return False

    if evffile[5:7] != 'MD':
        log.error('EVF file is not for the MIDAS instrument!')
        return False

    mtp = int(evffile[9:12])
    stp = int(evffile[14:17])
    case = evffile[21]
    if case not in planning.activity_case_types:
        log.error('activity case %s invalid' % (case))
        return False
    rorl = int(evffile[18:20])

    evf_start, evf_end, events = planning.read_evf(evf_file)
    events = pd.DataFrame(events)

    start = events[events.start==True]
    start = start.drop(['start','event_id'],1)
    start = start.rename(columns={"time": "start_time"})

    end = events[events.start==False]
    end = end.drop(['start','event_id'],1)
    end = end.rename(columns={"time": "end_time"})

    events = pd.merge(left=start,right=end,how='inner')

    observations = read_obs_file()

    # Loop through observations, add validated date to the df

    for idx,obs in events.iterrows():

        obs_data = {}
        obs_data['mtp'] = mtp
        obs_data['stp'] = stp
        obs_data['observation'] = obs.obs_type
        obs_data['cnt'] = obs.obs_id
        obs_data['start'] = obs.start_time # datetime.strftime(obs.start_time,isofmt)
        obs_data['end'] = obs.end_time # datetime.strftime(obs.end_time,isofmt)
        obs_data['retrieved'] = False

        observations = observations.append(obs_data, ignore_index=True)

    observations = observations.sort_values(by='start')

    # Check for and remove duplicates
    observations.drop_duplicates(subset=['mtp','stp','observation','cnt'], inplace=True)

    # write the df back to a file (overwrite the old)
    observations.to_csv(obs_list_file, index=False, date_format=isofmt)

    return observations


def read_obs_file(filename=obs_list_file):
    """Open and returns the observation status file. This lists the MIDAS
    observations and their status wrt the DDS. New observations are added
    by calling add_observations(evf). This file is polled to see if new
    completed observations are available to be downloaded from the DDS."""

    # File format:
    # MTP, STP, OBSERVATION, COUNT, START, END, RETRIEVED
    #
    # RETRIEVED can be True or False

    observations = pd.read_csv(filename, skipinitialspace=True, comment='#',
        parse_dates=['start','end'])

    return observations

def write_obs_file(obs_list):
    """Accepts a df containing the observation status list and simply
    writes it to a CSV file."""

    obs_list.to_csv(obs_list_file, index=False, date_format=isofmt)

    return


def get_aux_observations(outputdir='.', pkt_file=pkt_file, max_retry=20, retry_delay=5):
    """Get all auxiliary observations using get_pkts_from_list()"""

    observations = read_obs_file()

    for idx,obs in observations.iterrows():

        start = obs.start.isoformat()
        end = obs.end.isoformat()

        padded_name = obs.observation + (20-len(obs.observation)) * '_'
        aux_filename = 'TLM__SC_M%03d_S%03d_%s_COUNT_%02d.DAT' % (obs.mtp, obs.stp, padded_name, obs.cnt)

        log.info('requesting data for auxiliary file %s' % aux_filename)

        reqfiles, gotfiles = get_pkts_from_list(start, end, pkt_file, outputfile=aux_filename, outputpath=outputdir,
            delfiles=True, max_retry=max_retry, retry_delay=retry_delay)

        if len(gotfiles)<len(reqfiles):
            log.warning('only %d of %d requested files retrieved for aux file %s' % (len(gotfiles), len(reqfiles), aux_filename))

    return


def get_new_observations(outputdir='.', mtpstp_dir=True, get_aux=True, max_retry=10, retry_delay=5):
    """Based on the current date and time, searches the event list for recently
    finished observations and retrieves data from the DDS.

    outputdir= specifies the location of the retrieved files, and mtpstp_dir
    is set to false to place all files in outputdir and True to create sub-
    directories based on the MTP/STP (e.g. MTP006/STP013)"""

    from datetime import datetime, timedelta

    # dl_time is a time an offset from the end of the observation to take into
    # account that we don't know exactly when the data will be retrieved
    dl_time = timedelta(hours=12)

    observations = read_obs_file()
    now = datetime.utcnow()

    # filter observations by those that have ended > a day ago and have not yet been retrieved
    # new_obs = observations[ (observations.retrieved==False) & ((observations.end+dl_time) < now) ]
    # Retrieving ongoing observations as well, but not flagging them as retrieved
    new_obs = observations[ (observations.retrieved==False) & (observations.start < now) ]

    if len(new_obs)==0: # nothing to see here

        log.info('no new observations available for retrieval')
        return []
    else:
        log.info('%i new observation(s) available for retrieval' % (len(new_obs)))

    # Now do the following for each observation:
    # -- build DDS requests
    # -- submit request to the DDS
    # -- wait a few minutes
    # -- retrieve requested files from SFTP server
    # -- remove the file from the server
    # -- save the combined packet set with the appropriate name
    # -- remove the local source files
    # -- update the status in the obs list

    obs_filenames = []

    for idx,obs in new_obs.iterrows():

        start = obs.start.isoformat()
        end = obs.end.isoformat()

        # Data go into subdirectories of the data root, e.g. ./MTP003/STP004/
        if mtpstp_dir:
            stp_dir = os.path.join(outputdir,'MTP%03d/STP%03d/'%(obs.mtp,obs.stp))
        else:
            stp_dir = outputdir

        # This is also used to name the observations when retrieved:
        # TLM__MD_Mnnn_Smmm_zzzzzzzzzzzzzzzzzzzz_COUNT_cc.DAT
        # TLM__MD_M004_S005_TARGET_EXPOSE________COUNT_01.DAT

        padded_name = obs.observation + (20-len(obs.observation)) * '_'
        obs_filename = 'TLM__MD_M%03d_S%03d_%s_COUNT_%02d.DAT' % (obs.mtp, obs.stp, padded_name, obs.cnt)
        aux_filename = 'TLM__SC_M%03d_S%03d_%s_COUNT_%02d.DAT' % (obs.mtp, obs.stp, padded_name, obs.cnt)

        obs_filenames.append(os.path.join(stp_dir,obs_filename))

        # Build and submit DDS requests for the observation times, all APIDs
        filelist, aplist = request_data_by_apid(start, end)

        if get_aux:
            aux_pkt = { 'type': 3, 'subtype': 25, 'apid':84, 'sid':102 }
            aux_file = get_single_pkt(start, end, apid=aux_pkt['apid'], sid=aux_pkt['sid'], pkt_type=aux_pkt['type'], pkt_subtype=aux_pkt['subtype'],
                outputfile=aux_filename, outputpath=stp_dir, delfiles=True, max_retry=max_retry, retry_delay=retry_delay)

        time.sleep(dds_wait_time) # wait n minutes before accessing the data via SFTP


        # Retrieve the files from the server - delete them at the remote and local side
        # Files are also combined into a single TM
        filelist = get_files(filelist, stp_dir, apid=aplist, outputfile=obs_filename, delfiles=True, strip_dds=False, max_retry=max_retry, retry_delay=retry_delay)

        if len(filelist)==0:
            # If no files are returned, remove the observation from the list and flag a warning
            log.warning('no files retrieved for observation %s' % obs_filename)
            obs_filenames.remove(os.path.join(stp_dir,obs_filename))

        # Update the status of this observation - only if the end time is >12 hours in the past
        if (obs.end + dl_time) < now:
            observations.retrieved.loc[idx]=True

    write_obs_file(observations)

    return obs_filenames


class sftp():

    def open(self, sftpURL = 'mimas.iwf.oeaw.ac.at', sftpUser = 'midas', sftpPass = 'middoc'):

        import paramiko

        try:

            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.client.connect(sftpURL, username=sftpUser, password=sftpPass)
            log.info('opening connection to %s@%s' % (sftpUser,sftpURL))
            self.sftp = self.client.open_sftp()
            return 0

        except Exception as e:
            print e
            return False

    def close(self):
        self.sftp.close()
        log.info('closing SSH connection')


def open_tunnel():

    import subprocess
    # tunnel = subprocess.Popen( ['ssh', '-N', 'mbentley@193.170.92.8', '-D1080', '-p222'] )
    if socket.gethostname() in ['laptop-mark', 'desktop-home']:
        user = 'mbentley'
    else:
        user = os.getlogin()
    server = 'midas.iwf.oeaw.ac.at'
    address = user+'@'+server
    tunnel = subprocess.Popen( ['ssh', '-N', address, '-D1080'] )

    return tunnel



def get_fdyn_files(directory='.'):
    """Log onto the MIMAS server and retrieve any new Flight Dynamics data files"""

    ssh = sftp()
    try:
        ssh.open()
    except Exception as e:
        log.error('could not connect to server')
        log.error('%s' % e)
        return None

    retrieved = []

    files = ssh.sftp.listdir()
    ros_files = [f for f in files if f.split('.')[-1]=='ROS']

    for f in ros_files:

        try:
            ssh.sftp.get(f, os.path.join(directory,f))
            ssh.sftp.remove(f)
        except Exception as e:
            log.error('could not retrieve file %s' % f)
            log.error('%s' % e)
            continue

        retrieved.append(f)

    ssh.close()

    log.info('%d FDyn files retrieved and removed from the server' % len(retrieved))

    return retrieved


def get_sgs_files(directory=common.ops_path):
    """Log onto the MIDAS server and retrieve any new SGS-pushed files"""

    ssh = sftp()
    ssh.open()
    retrieved = []

    ssh.sftp.chdir('INTRAY')
    files = ssh.sftp.listdir()

    # Process the following files:
    # 0         1         2         3
    # 01234567890123456789012345678901234
    # PTRM_PL_Mmmm______01_c_RSMiPIM0.ROS where mmm is the MTP, c is the case and i is the index
    # EVF__MD_Mmmm_Ssss_01_c_RSMiPIMj.evf where sss is the STP, j is the PI counter
    # ITLS_MD_Mmmm_Ssss_01_c_RSMiPIMj.itl

    allowed_extensions = ['ROS','itl','evf']
    ok_files = [f for f in files if f.split('.')[-1] in allowed_extensions]

    if len(ok_files)<len(files):
        log.warning('%i unknown files found, skipping these' % (len(files)-len(ok_files)))

    retrieve = []

    for f in ok_files:
        mtp = int(f[9:12])
        case = f[21]
        mtpdir = os.path.join(directory,'MTP%03i%c' % (mtp, case))
        if not os.path.exists(mtpdir):
            os.makedirs(mtpdir)
            log.info('creating new directory %s' % mtpdir)
        try:
            ssh.sftp.get(f, os.path.join(mtpdir,f))
            ssh.sftp.remove(f)
        except Exception as e:
            log.error('could not retrieve file %s' % f)
            log.error('%s' % e)
            continue

        retrieved.append(f)

    ssh.close()

    log.info('%i OFPM files retrieved from the server' % len(retrieved))


    return retrieved


if __name__ == "__main__":

    log.warning('this module cannot be called directly')
