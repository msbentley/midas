#!/usr/bin/python
"""dds_utils.py - module to interact with the ESA DDS"""

debug = False

import pandas as pd
import os, time, socket
from midas import ros_tm

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

template_path = os.path.expanduser('~/MIDAS/software') if socket.gethostname() in servers else os.path.expanduser('~/Dropbox/work/midas/software')
template_file = 'midas_dds_request.xml'
template_file = os.path.join(template_path,template_file)

schema_path = os.path.expanduser('~/MIDAS/software') if socket.gethostname() in servers else os.path.expanduser('~/Dropbox/work/midas/software')
schema_file = 'GDDSRequest.xsd'
schema_file = os.path.join(schema_path,schema_file)

obs_list_path = os.path.expanduser('~/MIDAS/operations') if socket.gethostname() in servers else os.path.expanduser('~/Dropbox/work/midas/operations')
obs_list_file = 'observations.csv'
obs_list_file = os.path.join(obs_list_path,obs_list_file)

data_path  = os.path.expanduser('~/Copy/midas/data') if socket.gethostname() in servers else os.path.expanduser('~/Copy/midas/data')
fdyn_path = os.path.expanduser('~/Copy/midas/fdyn')


def validate_xml(xml, schema_file, isfile=False):
    """Validate a DDS request (or other) XML file against a given schema"""

    # LXML validates XML on parse
    # http://lxml.de/validation.html

    from lxml import etree

    with open(schema_file, 'r') as f:
        schema_root = etree.XML(f.read())

    current_dir = os.getcwd()
    schema_dir =  os.path.dirname(os.path.abspath(schema_file))

    # the top level schema often calls others - they are assumed
    # to be in the same directory, so first move there
    os.chdir(schema_dir)

    schema = etree.XMLSchema(schema_root)
    xmlparser = etree.XMLParser(schema=schema)

    os.chdir(current_dir)

    try:
        if isfile:
            with open(xml, 'r') as f:
                etree.fromstring(f.read(), xmlparser)
        else:
            etree.fromstring(xml, xmlparser)
            if debug: print('DEBUG: XML validation against schema %s successful' % (schema_file))
        return True
    except:
        print('ERROR: file or parse error')
        return False

def generate_request(start_time, end_time, apid, template_file, target=False):
    """Generates and returns a DDS XML request based on a template,
    start and end time and an APID."""

    import re, datetime

    if not os.path.isfile(template_file):
        print('ERROR: template file %s not found' % (template_file))
        return 0

    # validate inputs
    if apid not in ros_tm.midas_apids:
        print('WARNING: APID %i invalid for MIDAS' % apid)
        # return 0

    if (type(start_time) or type(end_time)) is not datetime.datetime:
        print('ERROR: start and end times must be given as a datetime!')
        return 0

    start_time = datetime.datetime.strftime(start_time,isofmt)
    end_time = datetime.datetime.strftime(end_time,isofmt)

    request = 'MD_TLM_%i_%s--%s' % (apid, start_time, end_time)
    request = request.replace(":","")

    if target and target not in dds_targets:
        print('ERROR: invalid DDS target server %s' % target)
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

    f = open(template_file, 'r')
    template = f.read()
    f.close()

    # find positions of opening and closing tags ('[', ']')
    opentag = [tag.start() for tag in re.finditer('\[',template)]
    closetag = [tag.start() for tag in re.finditer('\]',template)]

    # check we have an equal number of open and close tags
    if len(opentag) != len(closetag):
        print('ERROR: template file %s has %i open and %i closing tags - must be equal!' \
            % (template_file, len(opentag), len(closetag)))

    # get the list of unique tags
    tags = [template[opentag[n]+1:closetag[n]] for n in range(len(opentag))]
    tags = list(set(tags))

    if debug: print('DEBUG: XML template file %s opened with %i unique tags' % (template_file, len(tags)))

    # check that all tags are included in the params dictionary
    matches = sum([True for key in tags if key in request_params])
    if matches < len(tags):
        print('ERROR: %i tags in template %s but only %i parameters given' % \
            (len(tags), template_file, len(['params'])))
        return 0

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

        time.sleep(5) # tunnel stays open, hence no return code and just need to wait

        import socks as proxy
        proxy.setdefaultproxy(proxy.PROXY_TYPE_SOCKS5, 'localhost', 1080)
        proxy.wrapmodule(ftplib)

    ftp = ftplib.FTP('rodda.esoc.ops.esa.int')
    ftp.login('roreq', 'rod6$')

    # check we're in the right directory (should be by default)
    cwd = ftp.pwd()
    if cwd != request_dir: ftp.cwd(request_dir)

    buf = StringIO.StringIO(template)
    put_result = ftp.storlines('STOR %s.tmp' % request, buf)
    rename_result = ftp.rename('%s.tmp' % request ,'%s.XML' % request)

    if debug: print 'DEBUG: FTP results (put, rename): %s %s' % (put_result, rename_result)

    if socks:
        tunnel.kill()
        proxy.setdefaultproxy()

    return (put_result, rename_result)


def request_data(start_time, end_time, apid=False, target=False, socks=False):
    """General XML, validate and submit to the DDS in a given time frame. All standard
    date/time strings are accapted for the start/end times. If the optional apid= keyword is set,
    only this APID will be requested, otherwise all MIDAS APIDs are used.

    The optional keyword target= can be used to specify an alternate server configured with the DDS,
    otherwise the default is used.

    If socks=True then a socks5 tunnel to the appropriate server must already be open and listening
    on port 1080."""

    from dateutil import parser
    from datetime import datetime

    st = parser.parse(start_time) if (type(start_time) != pd.tslib.Timestamp and type(start_time) != datetime) else start_time
    et = parser.parse(end_time) if type(end_time) != pd.tslib.Timestamp else end_time

    filenames = []

    if type(apid) != list and type(apid) != bool: apid = [apid]
    aplist = apid if apid else ros_tm.midas_apids

    for ap in aplist:
        xml, request_id = generate_request(st, et, ap, template_file, target=target)
        filenames.append(request_id+'.DAT')

        if validate_xml(xml, schema_file):
            submit_request(xml, request_id, socks)
            pass
        else:
            print('ERROR: problem validating XML against schema')

    print('INFO: %d requests submitted to the DDS' % len(filenames))

    return filenames, aplist


def get_data(start_time, end_time, outputfile=False, outputpath='.', apid=False, socks=False, delfiles=True, max_retry=5, retry_delay=2):

    filenames, aplist = request_data(start_time, end_time, apid=apid, socks=socks)

    print('INFO: waiting for DDS to service requests before starting retrieval...')
    time.sleep(dds_wait_time) # wait a few minutes before accessing the data via SFTP

    filelist = get_files(filenames, outputpath=outputpath, apid=aplist, outputfile=outputfile, delfiles=delfiles, max_retry=max_retry, retry_delay=retry_delay)

    return filelist, apid


def get_data_since(start_time, outputfile, outputpath='.', apid=False, socks=False, max_retry=5, retry_delay=2):
    """Calls request_data_since() and retrieves packets from remote SFTP site"""

    filenames, aplist = request_data_since(start_time, apid=apid, socks=socks)

    print('INFO: waiting for DDS to service requests before starting retrieval...')
    time.sleep(dds_wait_time) # wait a few minutes before accessing the data via SFTP

    filelist = get_files(filenames, outputpath=outputpath, apid=aplist, outputfile=outputfile, max_retry=max_retry, retry_delay=retry_delay)

    return filelist, apid


def get_new_data(since, outputfile, outputpath, apid=False, socks=False, max_retry=5, retry_delay=2):

    filenames, aplist = request_new_data(since, apid=apid, socks=socks)
    get_files(filenames, outputpath, outputfile=outputfile, apid=aplist, max_retry=max_retry, retry_delay=retry_delay)


def request_data_since(start_time, apid=False, socks=False):
    """Calls request_data() from a given date until the present"""

    from datetime import datetime

    end_time = datetime.now().isoformat()
    filenames, aplist = request_data(start_time, end_time, apid, socks=socks)


    return filenames, aplist


def request_new_data(since, apid=False, socks=False):
    """Calls request_data() for a period specified by the current time and since, in hours"""

    from datetime import datetime, timedelta, date

    end_time = datetime.now()
    start_time = (end_time - timedelta(hours=since) )

    start_time = date.strftime(start_time, '%c')
    end_time = date.strftime(end_time, '%c')

    filenames, aplist = request_data(start_time, end_time, apid, socks=socks)

    return filenames, aplist



def get_files(filenames, outputpath, apid=False, outputfile=False, delfiles=True, strip_dds=False, max_retry=5, retry_delay=2):
    """Retrieves requested files from SFTP site. If outputfile= is set the
    frames are combined into a single packet with the given name given by
    outputpath/outputfile.

    If delfiles=True the individual source files are locally deleted."""


    # retrieve data from the SFTP server, skipping missing or zero byte files
    retrieved = retrieve_data(filenames, outputpath, max_retry=max_retry, retry_delay=retry_delay)

    # return an empty filelist if no matching files are on server
    if len(retrieved)==0: return []

    if outputfile:
        tm = ros_tm.tm()
        for localfile in retrieved:
            tm.get_pkts(localfile, apid=apid[filenames.index(os.path.basename(localfile))], append=True)

        tm.write_pkts(outputfile, outputpath=outputpath, strip_dds=strip_dds)
        print('INFO: combined TM written to file %s' % (outputfile))

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

    retry = 0 # first attempt

    while (retry < max_retry) and (len(remaining) > 0):

        # refresh list of files available on the server
        files = ssh.sftp.listdir()

        for filename in remaining:

            if debug: print('DEBUG: processing file %s' % (filename))

            if filename not in files:
                # print('WARNING: file %s not found on the server' % (filename))
                continue

            stat = ssh.sftp.stat(filename)
            if stat.st_size==0:
                print('WARNING: file %s has zero size, skipping...' % (filename))
                remaining.remove(filename)
                ssh.sftp.remove(filename)
                continue

            ssh.sftp.get(filename, os.path.join(localpath,filename))
            ssh.sftp.remove(filename)
            remaining.remove(filename)
            retrieved.append( os.path.join(localpath,filename) )
            print('INFO: file %s retrieved and removed from the server' % (filename))

        if len(remaining)==0: break

        retry += 1

        print('INFO: %i files remaining, waiting %i minutes to retry (attempt %i/%i)' %
            (len(remaining), retry_delay, retry+1, max_retry))

        time.sleep(retry_delay*60)

    ssh.close()

    if len(remaining) > 0:
        print('WARNING: unable to retrieve %i files' % (len(remaining)))

    return retrieved



def add_observations(evf_file):
    """Adds all observations (events) from a given EVF file to the MIDAS event
    list (which is polled daily to retrieve data from complete observations)"""

    import planning
    from datetime import datetime

    evffile = os.path.basename(evf_file)

    # Extract MTP and STP cycle from filename
    # e.g. EVF__MD_M004_S005_01_A_RSM0PIM1.evf
    if evffile[0:4] != 'EVF_':
        print('ERROR: EVF files must start with prefix EVF_')
        return False

    if evffile[5:7] != 'MD':
        print('ERROR: EVF file is not for the MIDAS instrument!')
        return False

    mtp = int(evffile[9:12])
    stp = int(evffile[14:17])
    case = evffile[21]
    if case not in planning.activity_case_types:
        print('ERROR: activity case %s invalid' % (case))
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
        obs_data['start'] = datetime.strftime(obs.start_time,isofmt)
        obs_data['end'] = datetime.strftime(obs.end_time,isofmt)
        obs_data['retrieved'] = False

        observations = observations.append(obs_data, ignore_index=True)

    observations = observations.sort('start')

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

    observations = pd.read_csv(filename, skipinitialspace=True, parse_dates=['start','end'])

    return observations

def write_obs_file(obs_list):
    """Accepts a df containing the observation status list and simply
    writes it to a CSV file."""

    obs_list.to_csv(obs_list_file, index=False, date_format=isofmt)

    return





def get_new_observations(outputdir=data_path, mtpstp_dir=True, max_retry=10, retry_delay=5):
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

        print('INFO: no new observations available for retrieval')
        return []
    else:
        print('INFO: %i new observation(s) available for retrieval' % (len(new_obs)))

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

        # Build and submit DDS requests for the observation times, all APIDs
        filelist, aplist = request_data(obs.start.isoformat(), obs.end.isoformat())
        time.sleep(dds_wait_time) # wait n minutes before accessing the data via SFTP

        # Data go into subdirectories of the data root, e.g. ./MTP003/STP004/
        if mtpstp_dir:
            stp_dir = os.path.join(outputdir,'MTP%03d/STP%03d/'%(obs.mtp,obs.stp))
        else:
            stp_dir = outputdir

        # This is also used to name the observations when retrieved:
        # TLM__MD_Mnnn_Smmm_zzzzzzzzzzzzzzzzzzzz_COUNT_cc.DAT
        # TLM__MD_M004_S005_TARGET_EXPOSE________COUNT_01.DAT

        padded_name = obs.observation+(20-len(obs.observation))*'_'
        obs_filename = 'TLM__MD_M%03d_S%03d_%s_COUNT_%02d.DAT' % (obs.mtp, obs.stp, padded_name, obs.cnt)
        obs_filenames.append(os.path.join(stp_dir,obs_filename))

        # Retrieve the files from the server - delete them at the remote and local side
        # Files are also combined into a single TM
        get_files(filelist, stp_dir, apid=aplist, outputfile=obs_filename, delfiles=True, strip_dds=False, max_retry=max_retry, retry_delay=retry_delay)

        # Update the status of this observation - only if the end time is >12 hours in the past
        if (obs.end + dl_time) < now:
            observations.retrieved.ix[idx]=True

    write_obs_file(observations)

    return obs_filenames


class sftp():

    def open(self, sftpURL = 'mimas.iwf.oeaw.ac.at', sftpUser = 'midas', sftpPass = 'middoc'):

        import paramiko

        try:

            self.client = paramiko.SSHClient()
            self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.client.connect(sftpURL, username=sftpUser, password=sftpPass)
            print('INFO: opening connection to %s@%s' % (sftpUser,sftpURL))
            self.sftp = self.client.open_sftp()
            return 0

        except Exception as e:
            print e
            return -1

    def close(self):
        self.sftp.close()
        print('INFO: closing SSH connection')


def open_tunnel():

    import subprocess
    tunnel = subprocess.Popen( ['ssh', '-N', 'mbentley@193.170.92.8', '-D1080', '-p222'] )

    return tunnel



def get_fdyn_files(directory=fdyn_path):
    """Log onto the MIMAS server and retrieve any new Flight Dynamics data files"""

    ssh = sftp()
    ssh.open()
    retrieved = []

    files = ssh.sftp.listdir()
    ros_files = [f for f in files if f.split('.')[-1]=='ROS']

    for f in ros_files:
        ssh.sftp.get(f, os.path.join(directory,f))
        ssh.sftp.remove(f)
        retrieved.append(f)

    ssh.close()

    print('INFO: file %s retrieved and removed from the server' % (f))

    return retrieved


def get_sgs_files(directory=obs_list_path):
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
        print('WARNING: %i unknown files found, skipping these' % (len(files)-len(ok_files)))

    retrieve = []

    for f in ok_files:
        mtp = int(f[9:12])
        case = f[21]
        mtpdir = os.path.join(directory,'MTP%03i%c' % (mtp, case))
        if not os.path.exists(mtpdir):
            os.makedirs(mtpdir)
            print('INFO: creating new directory %s' % mtpdir)
        ssh.sftp.get(f, os.path.join(mtpdir,f))
        ssh.sftp.remove(f)
        retrieved.append(f)

    ssh.close()

    print('INFO: %i OFPM files retrieved from the server' % len(retrieved))


    return retrieved


if __name__ == "__main__":

    # Called as a file (e.g. from a cron job) not as a library
    #
    # Poll the observation file and look for new observations - request and download data for
    # those that have completed but have not yet been downloaded from the DDS.

    get_new_observations(outputdir=data_path)
