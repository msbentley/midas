#!/usr/bin/python
"""
planning.py

Mark S. Bentley (mark@lunartech.org), 2013

This module reads and processes the various planning files used in the
Rosetta mission. Specifically it reads ITLM data files for MIDAS and
finds the duration of scanning and exposure slots. These are checked
against constraints, and if acceptable appropriate STP level commanding
is inserted into the relevant ITLS planning file.

"""
# Module level constants and definitions

from __future__ import print_function
import pandas as pd
import numpy as np
import logging, os, sys
from datetime import datetime, timedelta
from dateutil import parser

from midas import common, ros_tm

debug = False

# Date format example:  28-September-2014_10:00:00
dateformat = "%d-%b-%Y_%H:%M:%S"
timeformat = "%H:%M:%S"

# Set of valid observation names
valid_obs_names = [ 'TARGET_SCAN', 'TARGET_EXPOSE', 'MIDAS__SCI__MTP003', 'MD_TARGET_SCAN', 'MIDAS__SCI',
    'TARGET_EXPOSE_SCAN', 'DUMMY','TARGET_EXPOSE_SCAN_SO', 'TARGET_EXPOSE_SCAN_EO']
valid_event_types = [ 'SO', 'EO' ]

# Set of valid facet status types (in status file)
facet_status = ['NEW', 'PRESCANNED', 'EXPOSED', 'SCANNED']

# Set of valid observation template types and filenames
expose_templates = {
    'EXPOSE': 'ITLS_MD_TARGET_EXPOSE.itl',
    'OPEN_SHUTTER': 'ITLS_MD_OPEN_SHUTTER.itl',
    'CLOSE_SHUTTER': 'ITLS_MD_CLOSE_SHUTTER.itl' }
scan_templates = {
    'SCAN': 'ITLS_MD_SCAN.itl',
    'SCAN_AT_SURFACE': 'ITLS_MD_SCAN_AT_SURFACE.itl',
    'LINE_SCAN': 'ITLS_MD_LINE_SCAN.itl',
    'LINE_SURF': 'ITLS_MD_LINE_SCAN_AT_SURFACE.itl',
    'LINE_CONTACT': 'ITLS_MD_LINE_SCAN_CONTACT.itl',
    'LINE_CON_SURF': 'ITLS_MD_LINE_SCAN_CONTACT_AT_SURFACE.itl',
    'AUTOZOOM': 'ITLS_MD_AUTO_ZOOM.itl',
    'SCAN_NOSETUP': 'ITLS_MD_SCAN_NOSETUP.itl',
    'CONTACT_SCAN': 'ITLS_MD_SCAN_CONTACT.itl',
    'CONTACT_SCAN_AT_SURFACE': 'ITLS_MD_SCAN_CONTACT_AT_SURFACE.itl'}
pstp_templates = {
    'PSTP_PREP': 'ITLS_MD_PSTP_PREP.itl',
    'PSTP_SETUP': 'ITLS_MD_PSTP_SETUP.itl',
    'PSTP_SCAN': 'ITLS_MD_PSTP_SCAN.itl',
    'PSTP_FEATURE': 'ITLS_MD_PSTP_FEATURE.itl' }
other_templates = {
    'FEATURE': 'ITLS_MD_FEATURES.itl',
    'POWER_ON': 'ITLS_MD_POWER_ON.itl',
    'POWER_OFF': 'ITLS_MD_POWER_OFF.itl',
    'SUBSYS_ONOFF': 'ITLS_MD_SUBSYS_ON_OFF.itl',
    'FSCAN': 'ITLS_MD_FSCAN.itl',
    'CANT_SURVEY': 'ITLS_MD_FSCAN_SURVEY_CANT.itl',
    'ABORT': 'ITLS_MD_ABORT.itl',
    'APP_MIN': 'ITLS_MD_APPROACH_MIN.itl',
    'APP_ABS': 'ITLS_MD_APPROACH_ABS.itl',
    'PSTP': 'ITLS_MD_PSTP_PLACEHOLDER.itl',
    'XYZ_MOVE': 'ITLS_MD_XYZ_MOVE.itl',
    'LINEAR_MAX': 'ITLS_MD_LINEAR_MAX.itl',
    'LINEAR_MIN': 'ITLS_MD_LINEAR_MIN.itl',
    'TECH_CMD': 'ITLS_MD_TECH_CMD.itl',
    'LIN_ABS': 'ITLS_MD_LINEAR_ABS.itl',
    'SETUP': 'ITLS_MD_INSTRUMENT_SETUP.itl',
    'COMMENT': 'ITLS_MD_COMMENT.itl',
    'WHEEL': 'ITLS_MD_WHEEL_MOVE.itl' }

scan_status_url = 'https://docs.google.com/spreadsheets/d/1tfgKcdYqeNnCtOAEl2_QOQZZK8p004EsLV-xLYD2BWI/export?format=csv&id=1tfgKcdYqeNnCtOAEl2_QOQZZK8p004EsLV-xLYD2BWI&gid=645126966'

all_templates = dict(expose_templates.items() + scan_templates.items() + other_templates.items() + pstp_templates.items())

# Set of valid scan types and statuses
scan_types = ['PRESCAN', 'SURVEY', 'FOLLOW', 'FOLLOWZOOM', 'CALIBRATION', 'PSTP']
scan_statuses = ['WAITING', 'SCHEDULED', 'SUCCESS', 'FAIL']

planning_cycle_types = ['STP', 'MTP', 'LTP']
activity_case_types = ['A','B','C','H','P']
orfa_seq_types = ['BUPL', 'ITLM', 'ITLS', 'PDOP']

allowed_mtp=['EVF_','ITLS', 'PTRM','MTP_'] # allowed file prefixes in MTP delivery

# Two types of pointing "slots" are defined, those that are acceptable
# (OK) for MIDAS operation and those that are not (NOK)
pointing_ok = ['MNAV', 'MWAC', 'MSLW', 'SLEW', 'OBS','MWNV', 'MWOL']
pointing_nok = ['MOCM']


# Some configurable limits - these should be set in a separate file! TODO
min_prescanned = 3 # minimum number of pre-scanned facets to maintain

# File and directory paths
template_dir = os.path.expanduser('~/Dropbox/work/midas/operations/ITL_templates/')

facet_file = 'facet_status.csv'
facet_file = os.path.join(common.config_path, facet_file)
scan_file = 'scan_list.csv'
scan_file = os.path.join(common.config_path, scan_file)
tip_file = 'tip_status.csv'
tip_file = os.path.join(common.config_path, tip_file)
log_file = 'midas_planner.log'
log_file = os.path.join(common.config_path, log_file)

ros_config_path = os.path.join(common.ros_sgs_path, 'CONFIG')
fecs_evf_file = os.path.join(common.ros_sgs_path,'PLANNING/RMOC/FCT/FECS____________RSGS_XXXXX.evf')

# Power and data thresholds
exposure_on_pwr = 13.0 # W - TODO update to correct value

#---------------------------------------------------------------

# Helper functions

def date_doy(date=False):
    """Returns the DOY corresponding to the input date, or else the current date
    if no argument is specified"""

    if date:
        date = parser.parse(date)
        return date.timetuple().tm_yday
    else:
        return datetime.now().timetuple().tm_yday

def doy_date(doy):
    """Returns the date corresponding to a given DOY - the current year is assumed"""

    from datetime import date
    first_day = date(date.today().year, 1, 1)
    date = first_day + timedelta(days=doy-1)

    return date

def find_sq(text):
    """Searches the descriptions of all MIDAS TC SQs in the database"""

    csf = ros_tm.read_csf()
    csf = csf[csf.sequence.str.startswith('AMD')]
    csf = csf[csf.description.str.contains(text, case=False)]

    return csf

def get_scan_status(url=scan_status_url):
    """Downloads the latest copy of the scan status spreadsheet from Google
    Sheets and loads it into a DataFrame"""

    import requests
    from StringIO import StringIO
    r = requests.get(url)
    data = r.content

    scan_status = pd.read_csv(StringIO(data), skiprows=0,)

    return scan_status

def date_to_obs(date):
    """Accepts a datetime and looks in the observations.csv file to see which
    observation this corresponds to"""

    import dds_utils

    if type(date) != datetime:
        date = parser.parse(date)

    obs = dds_utils.read_obs_file()

    matching = obs[ (obs.start<=date) & (obs.end>=date) ]

    if len(matching)!=1:
        print('ERROR: could not find an observation matching date %s' % date)
        return False

    return matching.squeeze()


def get_cycles(ltp, case='P', suffix=''):
    """Uses the ros_mtp_stp_vstp.def file from the ROS_SGS repo to look
    up the date corresponding to a given M/S/VS-TP start. Note that
    these are now produced per LTP so this must be specified, as well
    as the case."""

    # if not case:
    #     cycle_file = os.path.join(ros_config_path, 'ros_mtp_stp_vstp.def')
    # else:

    if case.lower() not in ['a', 'b', 'c', 'p', 'h']:
        print('ERROR: case must be A, B, C, P or H!')
        return False

    case = case.upper()

    ltpdir = os.path.join(common.ros_sgs_path, 'PLANNING/LTP%03i/LTP%03i%c%s' % (ltp, ltp, case, suffix))
    cycle_file = os.path.join(ltpdir, 'ros_mtp_stp_vstp_LTP%03i%c%s.def' % (ltp, case, suffix))

    # cycle_files = {
    #     'a': os.path.join(ros_config_path, 'ros_mtp_stp_vstp_a.def'),
    #     'b': os.path.join(ros_config_path, 'ros_mtp_stp_vstp_b.def'),
    #     'c': os.path.join(ros_config_path, 'ros_mtp_stp_vstp_c.def') }

    #    if case.lower() not in cycle_files.keys():
    #        print('ERROR: case must be A, B or C!')
    #        return False

    #    cycle_file = cycle_files[case.lower()]

    # Strip comment lines and empty lines and pack into a StringIO object
    from StringIO import StringIO
    stripped = StringIO()

    with open(cycle_file, 'r') as f:
        for line in f.readlines():
            if line.startswith('#') or line.startswith('\n') or line.startswith('   '):
                continue
            else:
                stripped.write(line)

    stripped.seek(0)

    col_names = ['STP','MTP','VSTP_start','VSTP_end','skip','LTP','VSTP1','VSTP2','VSTP3','VSTP4','VSTP5']
    cycles = pd.read_table(stripped, skiprows=0, header=None, sep=' ', skipinitialspace=True, na_filter=False,
        names=col_names, parse_dates=['VSTP1','VSTP2','VSTP3','VSTP4','VSTP5'])
    cycles = cycles.drop('skip', axis=1)
    cycles = cycles.convert_objects(convert_numeric=True)

    return cycles


def get_stp(ltp, stp, case='p', suffix=''):
    """Returns the start and end date/time of the specified STP."""

    cycles = get_cycles(ltp, case, suffix=suffix)
    stp_start = cycles[cycles.STP==stp].VSTP1.squeeze()
    stp_end = cycles[cycles.STP==stp+1].VSTP1.squeeze()

    if type(stp_start) !=pd.tslib.Timestamp or type(stp_end) != pd.tslib.Timestamp:
        print('ERROR: data for STP %i invalid' % stp)
        return None, None

    return stp_start,stp_end


def which_stp(date, ltp=6, case='p'):
    """Uses the LTP (specify latest!) definition file, along with case, to resolve
    a given date to a given STP. Date can be given either as a datetime object
    or as a string in any sensible date/time format."""

    if type(date)!=datetime and type(date)!=pd.Timestamp:
        date = parser.parse(date)

    cycles = get_cycles(ltp=ltp, case=case)

    return cycles[ cycles.VSTP1>date ].STP.iloc[0]-1



def get_vstp(ltp, vstp, case='p', suffix=''):
    """Returns the start and end date/time of the VSTP"""

    cycles = get_cycles(ltp, case, suffix=suffix)

    stp = cycles[ (cycles.VSTP_start<=vstp) & (cycles.VSTP_end>=vstp) ]
    vstp_in_stp = int(vstp-stp.VSTP_start.squeeze()+1)
    vstp_start = stp['VSTP%i' % vstp_in_stp].squeeze()

    next_vstp = vstp+1
    stp = cycles[ (cycles.VSTP_start<=next_vstp) & (cycles.VSTP_end>=next_vstp) ]
    vstp_in_stp = int(next_vstp-stp.VSTP_start.squeeze()+1)
    vstp_end = stp['VSTP%i' % vstp_in_stp].squeeze()

    return vstp_start, vstp_end


def get_mtp(ltp, mtp, case='p', suffix=''):
    """Returns the start and end date/time of the MTP"""

    cycles = get_cycles(ltp, case, suffix=suffix)
    current_mtp = cycles[cycles.MTP==mtp].sort('STP')

    mtp_start = current_mtp.sort('VSTP1').VSTP1.iloc[0] # first VSTP in this MTP

    # Check for subsequent MTP (at end of an LTP these are not always present)
    if mtp==cycles.MTP.max():
        last_stp = current_mtp.iloc[-1]
        vstp = last_stp.VSTP1
        mtp_end = [ last_stp[ 'VSTP%i' % num ] for num in range(1,6) if last_stp[ 'VSTP%i' % num ] >= vstp ]
        mtp_end = mtp_end[0]
        print('WARNING: no data for MTP %i, start of last VSTP in MTP %i is at %s' % (mtp+1, mtp, mtp_end))
    else:
        mtp_end = cycles[cycles.MTP==mtp+1].sort('VSTP1').VSTP1.iloc[0]

    return mtp_start,mtp_end



def stps_in_mtp(ltp, mtp, case='p', suffix=''):
    """Returns a list of the STPs within the specified MTP"""

    cycles = get_cycles(ltp, case, suffix=suffix)
    stps = cycles[cycles.MTP==mtp].STP.values.astype(int).tolist()

    return stps


def vstps_in_stp(ltp, stp, case='p', suffix=''):
    """Returns a list of the VSTPs within the specified STP"""

    cycles = get_cycles(ltp, case, suffix=suffix)
    vstp_start = cycles[cycles.STP==stp].VSTP_start.squeeze()
    vstp_end = cycles[cycles.STP==stp].VSTP_end.squeeze()

    return range(vstp_start,vstp_end+1)


def ltp_from_mtp(mtp):
    """Return the LTP for a given MTP - this is useful to search for the correct
    long term SPICE kernels (CORB, RORB) for a given MTP"""

    ltps = {
        1: range(1,6+1),
        2: range(7,8+1),
        3: range(9,10+1),
        4: range(11,13+1),
        5: range(14,18+1),
        6: range(19,21+1), }

    return [ltp for (ltp,mtps) in ltps.iteritems() if mtp in mtps][0]


def pstp_itl(itl_file, evf_file):
    """Returns an ITL object for the PSTP sequence already reserved in the ITL file."""

    import eps_utils

    placeholder = 'AMDC030A'

    # Read the ITL and check for the placeholder SQ
    stp_itl = eps_utils.parse_itl(itl_file)
    sqs = [sq.name for sq in stp_itl.timeline]
    if placeholder not in sqs:
        print('ERROR: no PSTP high res scan placeholder found in this ITL')
        return False

    # Find the sequence containiner the placeholder
    pstp_idx = sqs.index(placeholder)
    pstp_sq = stp_itl.timeline[pstp_idx]

    # Read the corresponding EVF in order to resolve the time to absolute
    evf_start, evf_end, event_list = read_evf(evf_file)
    event_list = pd.DataFrame(event_list)
    event = pstp_sq.label
    count = int(pstp_sq.options.val)
    event_time = event_list[ (event_list.event_id==event) & (event_list.obs_id==count) ]
    if len(event_time)==0:
        print('ERROR: no event matching ID %s and count %i found in EVF' % (event, count))
        return False
    event_time = event_time.time.squeeze()
    pstp_time = event_time + pstp_sq.time

    print('INFO: PSTP placeholder sequence starts at absolute time %s' % pstp_time)

    # Find the details of the subsequent SQ to calculate the PSTP duration
    next_sq = stp_itl.timeline[pstp_idx+1]
    next_count = int(next_sq.options.val)
    next_event = next_sq.label
    next_time = event_list[ (event_list.event_id==next_event) & (event_list.obs_id==next_count) ].time.squeeze()
    next_time += next_sq.time
    duration = next_time - pstp_time
    print('INFO: following sequence starts at %s, PSTP duration %s' % (next_time, duration))

    # Create an ITL containing the scan sequence

    # We don't have the usual observation table or pointing block list, so need to manually
    # initialise the ITL class with both (only the elements needed)

    # obs_type	obs_id	power	drate	duration	start_time	end_time	start_id	end_id
    # TARGET_EXPOSE_SCAN	5	18.3	333	3 days	2014-11-18 23:25:00	2014-11-21 23:25:00	MD_TARGET_EXPOSE_SCAN______SO	MD_TARGET_EXPOSE_SCAN______EO
    obs = pd.Series( [count, event_time, pstp_sq.label], index=['obs_id','start_time','start_id'])

    # start	                end	                duration	        offset
    # 2014-11-19 02:35:00	2014-11-21 23:25:00	2 days, 20:50:00	03:10:00
    block = pd.Series( [0, event_time, next_time, duration, pstp_sq.time], index=['index', 'start', 'end', 'duration', 'offset'] )

    pstp_itl = itl(obs=obs)
    pstp_itl.current_block = block
    pstp_itl.time = pstp_itl.current_block['offset']
    pstp_itl.abs_time = pstp_itl.current_obs.start_time + pstp_itl.time

    return pstp_itl


def generate_pdor(itl_file, pdor_file=None, evf_file=None, comments=True):
    """Generate a PDOR given a corresponding ITLS (and if needed EVF) file. If
    relative=True then relative timing is used, otherwise absolute timing is used."""

    import eps_utils, ros_tm

    # Get the current time in CCSDS B format
    ccsds_b_fmt = '%y-%jT%H:%M:%SZ' # YY-DDDThh:mm:ss[.mmm]Z
    now = datetime.now()
    filetime = datetime.strftime(now,ccsds_b_fmt)

    # Load and parse the ITLS into a timeline
    itl = eps_utils.parse_itl(itl_file)

    if evf_file is None:
        for header in itl.header:
            # if header.key == 'Start_time': itl_start = datetime.strptime(header.val,dateformat)
            if header.key == 'Start_time': itl_start = parser.parse(header.val.replace('_', ' '))
    else:
        evf_start, evf_end, event_list = read_evf(evf_file)
        event_list = pd.DataFrame(event_list)

    csf=ros_tm.read_csf()
    csf = csf[csf.sequence.str.startswith('AMD')] # filter by MIDAS sequences

    # Load the command parameter details
    csp = ros_tm.read_csp()

    # PDOP template
    # PDOP_PI5RSO_DAPPTEST________00018.ROS
    if pdor_file is None:
        sq_num = get_orfa_seq('PDOP')
        pdor_file = 'PDOP_PI5RSO_D_______________%05d.ROS' % (sq_num)

    # Open the PDOR file for output
    f = open(pdor_file, 'w')

    # For now set the version number to a fixed value of 1
    version = 1

    # Write the primary header
    # DOR_ 00001 11-067T11:10:23.
    f.write('C Primary header (PDOR, version %d, generated at %s)\n' % (version, filetime))
    f.write('C Generated from ITL file %s\n' % (os.path.basename(itl_file) )) # and EVF %s\n' % , os.path.basename(evf_file)))
    f.write('DOR_ %05d %s\n\n' % (version, '{:>20}'.format(filetime)))


    # Write the secondary header
    # start_time (20) end_time (20) operation_count (I4)
    # 11-064T00:00:00.000Z 11-064T00:06:00.000Z 0006
    f.write('C Secondary header (HTA record: start and end in absolute time, sequence count)\n')

    # Find the resolved time of the first and last sequences
    first_sequence = itl.timeline[0]
    event = first_sequence.label
    count = int(first_sequence.options.val)
    if evf_file is None:
        event_time = itl_start
    else:
        event_time = event_list[ (event_list.event_id==event) & (event_list.obs_id==count) ]
        event_time = event_time.time.squeeze()
    first_request_time = event_time + first_sequence.time
    first_request_time = datetime.strftime(first_request_time,ccsds_b_fmt)

    last_sequence = itl.timeline[-1]
    event = last_sequence.label
    count = int(last_sequence.options.val)
    last_request_time = event_time + last_sequence.time
    last_request_time = datetime.strftime(last_request_time,ccsds_b_fmt)

    num_seqs = len(itl.timeline)

    f.write('%s %s %04d\n\n' % ('{:<20}'.format(first_request_time), '{:<20}'.format(last_request_time), num_seqs))

    # Loop through the timeline entries and generate Operation Request Header
    # and data records for each

    for idx, sequence in enumerate(itl.timeline):

        # Look up the sequence description and use as a comment
        descript = csf[csf.sequence==sequence.name].description.squeeze()
        if comments: f.write('C %s\n' % descript)

        # Write the 5 ORS header records, H1-H5
        # H1: operation name and type (for ITL conversion always SQ name and 'S')
        f.write('H1%s S\n' % (sequence.name.strip()))

        # H2: ins/del (always insert here), time ref (opt), dest (T for timeline DAF),
        # source (D for DOR), number of params (I3)
        # NB. when dest = T, EXECUTION TIME (H4) must contain a value

        # Example:
        # H2I UTC    P D 004
        # this was for an in-pass interactive op - so dest = P!
        num_params = len(sequence.params)
        f.write('H2I UTC    T D %03d\n' % (num_params))

        # H3: not used if we specify H4 execution time
        f.write('H3\n')

        # H4: absolute time of execution - any time-tagged requests in a SQ are
        # executed relative to this
        event = sequence.label
        count = int(sequence.options.val)
        if evf_file is not None:
            event_time = event_list[ (event_list.event_id==event) & (event_list.obs_id==count) ]
            if len(event_time) != 1:
                print('ERROR: cannot find event %s (COUNT=%d) in EVF file' % (event, count))
                return False
                event_time = event_time.time.squeeze()
        sq_time = event_time + sequence.time

        f.write('H4%s\n' % (datetime.strftime(sq_time,ccsds_b_fmt)))

        # H5: not used
        f.write('H5\n')

        # Sub-header records
        # S1: unique ID (allowing deletion) - 10 chars long
        f.write('S1SEQ%07d\n' % (idx))

        # Data records
        # If parameter field is left blank, default value (FP) in dbase id used
        # Number of data records must match H2
        # 'P' type = parameter
        for param in sequence.params:
            if param:
                param_descrip = csp[(csp.sequence==sequence.name) & (csp.param==param.name) ].param_descrip.squeeze()
                if comments: f.write('C %s\n' % (param_descrip))
                val_type = csp[(csp.sequence==sequence.name) & (csp.param==param.name)].val_type.squeeze()
                if param.unit=='ENG':
                    unit = '    '
                else:
                    unit_len = len(param.unit)
                    unit = (4-unit_len)*' '+param.unit
                if val_type=='R': # radix must be given for all raw parameters
                    radix = csp[(csp.sequence==sequence.name) & (csp.param==param.name)].radix.squeeze()
                else:
                    radix = ' '
                f.write('P%s %c %s %c %s\n' % (param.name.strip(),val_type, unit, radix, param.value))

        f.write('\n')

    return pdor_file



def load_stp(directory, stp=False, load_obs=True):
    """Accepts a directory as argument, searches for ITL and EVF files and
    loads the latest MTP level files, returning a set of validated observations"""

    import glob

    # Search for MIDAS MTP level EVF and ITL files
    evf=glob.glob(os.path.join(directory,'EVF__MD_M0??_S???_01_?_RSM?PIM?.evf'))
    if stp:
        itl=glob.glob(os.path.join(directory, 'ITLS_MD_M0??_S???_01_?_RSM?PIS?.itl'))
    else:
        itl=glob.glob(os.path.join(directory, 'ITLS_MD_M0??_S???_01_?_RSM?PIM?.itl'))

    if (len(evf)==0) or (len(itl)==0):
        print('ERROR: no matching EVF/ITL files found in folder %s' % (directory))
        return False

    # Find the latest file - sort first by RS counter, then PI counter
    evf = sorted(evf, key=lambda x: ( int(os.path.basename(x)[26]), int(os.path.basename(x)[30])) )[-1]
    itl = sorted(itl, key=lambda x: ( int(os.path.basename(x)[26]), int(os.path.basename(x)[30])) )[-1]

    print('INFO: using input files:\n    EVF %s\n    ITL %s' % (evf, itl))

    itl_start, itl_end, version, timeline = read_itlm(itl)
    evf_start, evf_end, events = read_evf(evf)

    if load_obs:
        # Load the ITL and EVF files and validate them to a set of observations
        observations = validate_obs(evf_start, evf_end, events, itl_start, itl_end, timeline)
    else:
        observations = None

    return os.path.basename(itl), os.path.basename(evf), itl_start, itl_end, observations



def read_evf(filename):

    """Reads an ITLM event file and returns a list of dictionaries. Each entry
    contains the date/time of an event, the event label, and the event count"""

    event_list = []

    # First check if filename is valid
    if not os.path.isfile(filename):
        logging.error('EVF file with name %s not found' % (filename))
        return(None)

    mtp_start = None; mtp_end = None

    for line in open(filename): # read the EVF line by line
        line = line.strip() # remove carriage return etc.

        if line == '' or line[0] == '#': continue  # skip comments and blank lines

        if len(line.split()) == 2: # start or stop time
            if line.split()[0] == 'Start_time:':
                mtp_start = line.split()[1]
                # mtp_start = datetime.strptime(mtp_start,dateformat)
                mtp_start = parser.parse(mtp_start.replace('_',' '))
            elif line.split()[0] == 'End_time:':
                mtp_end = line.split()[1]
                # mtp_end = datetime.strptime(mtp_end,dateformat)
                mtp_end = parser.parse(mtp_end.replace('_',' '))
            else:
                logging.error('invalid syntax in EVF file %s' % (filename))
                return(None)
        elif len(line.split()) == 5: # standard event line

            # read the time (into python datetime format), label and count
            event = {}

            # Time follows the format defined at the top of this module
            event['time'] = parser.parse(line.split()[0].replace('_', ' '))

            # Event label follows the following definition:
            # 123456789012345678901234567890123
            # PP_zzzzzzzzzzzzzzzzzzzz_IIIIII_EE
            #
            # where PP = MD, EE = SO or EO, IIIIII is an internal ID
            # and zzzzzzzzzzzzzzzzzzzz is the description

            label = line.split()[1]

            if label[0:2] != 'MD':
                logging.error( 'event found for instrument %s instead of MD' % (label[0:2]))
                return None

            event['event_id'] = label

            event['obs_type'] = label[3:-3].strip('_') # free form name in the ICD

            if event['obs_type'] not in valid_obs_names:
                logging.error( 'observation type %s not valid for MIDAS' % (event['obs_type']))
                return(None, None, None)

            suffix = label[-2:]
            if suffix not in valid_event_types:
                logging.error( 'event label must end in SO or EO')
                return None
            event['start'] = True if suffix == 'SO' else False
            event['obs_id'] = int(line.split()[4][:-1])

            # add this to a list of events to be returned
            event_list.append(event)

        else:
            logging.error('invalid syntax in EVF file %s' % (filename))
            return(None)

    print('INFO: %i events read from EVF file %s' % (len(event_list), os.path.basename(filename)))

    return( mtp_start, mtp_end, event_list )


def read_itlm(filename):
    """"Read an ITLM file and extract the Z record (power and data) specification
    for each observation. This can then be combined with data from read_evf to
    produce a complete observation envelope."""

    from eps_utils import parse_itl

    itl = parse_itl(filename)

    for header in itl.header:
        if header.key == 'Version': itl_ver = header.val
        if header.key == 'Start_time': itl_start = parser.parse(header.val.replace('_', ' '))
        if header.key == 'End_time': itl_end = parser.parse(header.val.replace('_', ' '))


    seq_list = []

    for entry in itl.timeline:

        seq = {}

        seq['obs_type'] = entry.label[3:-3].strip('_') # free form name in the ICD
        if seq['obs_type'] not in valid_obs_names:
            logging.error('event %s not valid for MIDAS' % (seq['obs_type']))
            return None

        suffix = entry.label[-2:]
        if suffix not in valid_event_types:
            logging.error('event label must end in SO or EO')
            return None
        seq['start'] = True if suffix == 'SO' else False

        if entry.options.key != 'COUNT':
            logging.error('event must contain a COUNT')
            return None
        else:
            seq['obs_id'] = int(entry.options.val)

        if entry.instrument != 'MIDAS':
                logging.error('ITL command references an instrument other than MIDAS')
                return None

        seq['reltime'] = entry.time

        # Extract power and data specs from Z records
        for zrec in entry.zrec:
            if zrec.name == 'DATA_RATE_PROFILE':
                seq['drate'] = float(zrec.value)
            elif zrec.name == 'POWER_PROFILE':
                seq['power'] = float(zrec.value)

        # entries are not always given for switch off - ensure that we zero out
        # the data and power if the switch-off sequence is found
        if entry.sequence.name == 'AMDF041A':
            seq['drate'] = 0.0
            seq['power'] = 0.0

        seq_list.append(seq)

    return( itl_start, itl_end, itl_ver, seq_list )



def resolve_time(itl_file, evf_file, html=False, expand_params=False):
    """Reads an ITLS and EVF file and resolves events into absolute times.

    If the html= keyword is set to a filename, an html summary is produced.

    If expand_params= is True then sequence parameters are expanded and included
    in the summary."""

    import eps_utils, ros_tm

    itl = eps_utils.parse_itl(itl_file)
    evf_start, evf_end, event_list = read_evf(evf_file)

    if event_list is None:
        return None

    event_list = pd.DataFrame(event_list)
    event_list = event_list.drop( ['obs_type', 'start'], axis=1 )
    event_list = event_list.rename(columns={"event_id": "event"})
    event_list = event_list.rename(columns={"obs_id": "cnt"})
    event_list = event_list.rename(columns={"time": "event_time"})

    seqs = []
    seqs= pd.DataFrame( [(seq.name,seq.time,seq.label,int(seq.options.val)) for seq in itl.timeline], \
        columns=['sequence', 'reltime', 'event', 'cnt'] )

    # Get the description of each sequence from the command sequence file (csf.dat)
    csf_file = os.path.join(common.s2k_path, 'csf.dat')
    cols = ('sequence', 'description')
    csf = pd.read_table(csf_file,header=None,names=cols,usecols=[0,1],na_filter=False)
    csf = csf[csf.sequence.str.startswith('AMD')] # filter by MIDAS sequences

    # Load the command parameter details
    csp = ros_tm.read_csp()

    # merge frames to add sequence names
    seqs = pd.merge(left=seqs,right=csf,on='sequence',how='left').sort( ['cnt','reltime'] )

    # add the absolute time for each sequence according to event and count
    seqs = pd.merge(left=seqs,right=event_list,on=['event','cnt'],how='left')

    # absolute time is simply the event time plus the ITL relative time
    seqs['abs_time']=seqs.reltime+seqs.event_time

    # RMOC often uses "Day of Year", so also calculate the DOY
    seqs['doy']= seqs.abs_time.apply( lambda x: x.dayofyear )

    # Only return the useful content for now
    seqs = seqs[ ['abs_time', 'doy', 'sequence','description','event','cnt','reltime'] ]

    if expand_params:
        param_list = []
        for seq in itl.timeline:
            param_list.append ( [ (seq.time, seq.label,int(seq.options.val),seq.name,param.name,param.value,param.unit) for param in seq.params if len(seq.params)>0] )
        param_list = [item for sublist in param_list for item in sublist] # flatten the list of lists
        param_list=pd.DataFrame(param_list,columns=['reltime','event','cnt','sequence','param','value','unit'])
        seqs = pd.merge(left=param_list,right=seqs,on=('event','cnt','reltime','sequence'),how='outer')
        seqs.fillna('')

        # Get the description of each TC param from the command sequence file (csp.dat)
        csp = ros_tm.read_csp()
        seqs = pd.merge(left=seqs,right=csp,on=['sequence','param'],how='left').sort( ['abs_time','param_num'] )

    if expand_params:
        seqs = seqs[ ['abs_time','doy','sequence','description','param','param_descrip','value','unit'] ]
    else:
        seqs = seqs[ ['abs_time','doy','sequence','description'] ]

    if html:
        seqs_html = seqs.to_html(classes='alt_table', index=False, na_rep='')
        ros_tm.css_write(html=seqs_html, filename=html)

    return seqs


def locate_mtp(mtp, case):
    """Locates the directory corresponding to a given MTP and case in the
    ROS_SGS repository. If multiple folders are found, the most recent
    by name index is returned."""

    import glob

    ltp = 'LTP%03i' % ltp_from_mtp(mtp)
    planning_dir = os.path.join(common.ros_sgs_path,'PLANNING')
    ltp_root = os.path.join(planning_dir,ltp)

    # check for directories matching LTPnnnX[_aa]
    # where nnn is the LTP, X is case and _aa is an optional count

    # get folders in the LTP directory
    ltp_dirs = [folder for folder in os.listdir(ltp_root) if os.path.isdir(os.path.join(ltp_root,folder))]
    ltp_case = ltp + case.upper()
    ltp_dir = [folder for folder in ltp_dirs if folder.startswith(ltp_case)]

    if len(ltp_dir)==0:
        print('ERROR: no LTP folder found in ROS_SGS for MTP %i, case %c' % (mtp, case.upper()))
        return None
    elif len(ltp_dir)==1:
        ltp_dir = ltp_dir[0]
    else:
        new_cases = [folder for folder in ltp_dir if '_' in folder]
        ltp_dir = sorted(new_cases, key=lambda x: ( int(x[-2:])) )[0]

    ltp_dir = os.path.join(ltp_root, ltp_dir)

    # now we have found the correct LTP folder, check for the MTP directory
    mtp_dir = os.path.join(ltp_dir,'MTP%03i%c' % (mtp, case.upper()))
    if not os.path.isdir(mtp_dir):
        print('ERROR: no MTP folder found in ROS_SGS for MTP %i, case %c' % (mtp, case.upper()))
        return None

    return os.path.join(ltp_dir,mtp_dir)



def list_scans(seqs, printall=True):
    """Accepts a timeline of sequences and parameters generated by
    resolve_time() and returns a list of scan parameters - essentially
    all of the parameters to AMDF035A, plus the linear stage and wheel
    position from preceding calls to those SQs."""

    # the logic for extracting scans from ITLS is fairly straightforward, but
    # needs some thought - if we assume that we are only looking at a scan
    # observation things are simplified.

    # Full scan SQ: AMDF028A (VMDD21F2 = data type, VMDD2142 = main scan dir)
    #
    full_scans = seqs[ seqs.sequence=='AMDF028A' ].abs_time.unique()

    # TODO: complete!

    # filter for scan setup TCs
    setup_times = seqs[seqs.sequence=='AMDF035A'].abs_time.unique()

    # wheel movememnts = AMDF020A, segment number = param VMDD4032
    segment = seqs[ (seqs.sequence=='AMDF020A') & (seqs.param=='VMDD4032') ].value.values
    segment = segment.astype(int)

    # linear stage posn = AMDF025A, voltage = VMDD1072
    linear = seqs[ (seqs.sequence=='AMDF025A') & (seqs.param=='VMDD1072') ].value.values
    linear = linear.astype(float)

    # frequency scan = AMDF026A, VMDDC002 = block, VMDDC102 = cantilever
    # filter by fscans with unique times (need more than one param from each)
    fscans = []
    fscan_times = seqs[ seqs.sequence=='AMDF026A' ].abs_time.unique()
    for obt in fscan_times:
        fscan_cmd = seqs[seqs.abs_time==obt][ ['param', 'param_descrip', 'value', 'unit'] ]
        fscans.append(fscan_cmd)


    if (len(segment) != len(linear) != len(setup_times) !=len(fscans)):
        print('ERROR - cannot identify unique scan parameters from this ITLS')
        return False

    scans = []

    for obt in setup_times:
        setup_cmd = seqs[seqs.abs_time==obt][ ['param', 'param_descrip', 'value', 'unit'] ]
        scans.append(setup_cmd)

    if printall:
        for idx, scan in enumerate(scans):
            x_orig = int(scan[scan.param=='VMDD2012'].value.squeeze())
            y_orig = int(scan[scan.param=='VMDD2022'].value.squeeze())
            xsteps = int(scan[scan.param=='VMDD2032'].value.squeeze())
            ysteps = int(scan[scan.param=='VMDD2042'].value.squeeze())
            xstep = int(scan[scan.param=='VMDD2052'].value.squeeze())
            ystep = int(scan[scan.param=='VMDD2062'].value.squeeze())
            zretract = int(scan[scan.param=='VMDD2172'].value.squeeze())
            cantilever = int(fscans[idx][fscans[idx].param=='VMDDC002'].value.squeeze())*8+\
                int(fscans[idx][fscans[idx].param=='VMDDC102'].value.squeeze())+1

            print('OBT: %s. Segment %i, linear position %3.3f, cantilever #%i. %ix%i scan from (%i,%i) with step (%i,%i) and retract %i' % \
                (setup_times[idx], segment[idx],linear[idx], cantilever, xsteps, ysteps, x_orig, y_orig, xstep, ystep, zretract))

    return scans


def validate_obs(evf_start, evf_end, event_list, itl_start, itl_end, seq_list):

    """Takes a list of events from an EVF file and POWER and DRATE specs
    from the corresponding ITL fragment and creates a validated observation
    definition"""

    # check if start and end times match
    if (evf_start != itl_start) or (evf_end != itl_end):
        logging.warning('EVF start time %s different from ITL start time %s' % (evf_start, itl_start))

    # combine EVF and ITL information into one dataset

    evf = pd.DataFrame(event_list)
    itl = pd.DataFrame(seq_list)

    # combine frames
    observations = pd.merge(itl,evf,on=('obs_type','obs_id','start'),how='left')

    # remove relative time for now
    observations = observations.drop('reltime',1)

    # match observation start and end times to calculate duration
    start=observations[observations.start==True]
    start=start.rename(columns={"time": "start_time"})
    start=start.rename(columns={"event_id": "start_id"})
    start=start.drop('start',1)

    end=observations[observations.start==False]
    end=end.rename(columns={"time": "end_time"})
    end=end.rename(columns={"event_id": "end_id"})
    end=end.drop('start',1)

    if 'power' in end.columns: end.drop(['power'],1, inplace=True)
    if 'drate' in end.columns: end.drop(['drate'],1, inplace=True)

    observations=pd.merge(start,end, how='inner', on=['obs_id','obs_type'])
    observations.sort(['start_time'],inplace=True)

    # calculate duration
    observations['duration']=observations['end_time']-observations['start_time']

    # check for overlaps
    # TODO

    # re-order (just to make inspection easier)
    observations = observations[['obs_type','obs_id','power','drate','duration','start_time','end_time','start_id','end_id']]

    return observations


def read_scanlist(filename):
    """Reads a CSV file containing a list of proposed scans of different types
    (prescan, survey, follow, follow_zoom, follow_pstp) and validates the
    file (checks for syntax errors). Returns a priority sorted list."""

    #TODO check file exists
    scan_list = pd.read_table(filename,sep=',')

    # validate the entries

    if not set(scan_list['scan_type']).issubset(scan_types):
        raise ValueError, "inavlid scan type"
        logging.error('invalid scan type in file %s' % (filename))
        return None

    if not set(scan_list['status']).issubset(scan_statuses):
        logging.error('invalid scan status in file %s' % (filename))
        return None

    if (scan_list['priority'].values > 5).any() or \
       (scan_list['priority'].values < 1).any():
        logging.error('priority  must be between 1 and 5')
        return None

    if (scan_list['segment'].values > 1023).any() or \
       (scan_list['segment'].values < 0).any():
        logging.error('segment number must be between 0 and 1023')
        return None

    if (scan_list['orig_tip'].values < 0).any() or \
       (scan_list['orig_tip'].values > 15).any():
        logging.error('tip number must be between 0 and 15')
        return None

    if scan_list['magnetic'].dtype != 'bool':
        logging.error('magnetic scan must be True or False')
        return None

    #TODO finish these tedious checks

    # sort by priority (1=highest)
    scan_list.sort(['priority'],inplace=True)

    return scan_list


def read_facetlist(filename):
    """Reads a CSV file containing the status of the facets"""

    facet_list = pd.read_table(filename,sep=',')
    facet_list = facet_list.set_index('facet_id')

    if len(facet_list) != 64:
        logging.error('facet status file must contain exactly 64 entries!')
        return None

    if not facet_list['status'].isin(facet_status).all():
        logging.error('facet file contains invalid STATUS entries!')
        return None

    new = len(facet_list[facet_list.status=='NEW'])
    prescanned = len(facet_list[facet_list.status=='PRESCANNED'])
    exposed = len(facet_list[facet_list.status=='EXPOSED'])
    scanned = len(facet_list[facet_list.status=='SCANNED'])

    logging.info('%i new, %i prescanned, %i exposed and %i scanned facets' % (new, prescanned, exposed, scanned))

    return facet_list


def update_facetlist(filename, facet_num, new_status):
    """Updates the status of a single facet in the facet status file"""

    if facet_num < 0 or facet_num > 63:
        logging.error('facet number must be between 0 and 63')
        return None

    if new_status not in facet_status:
        logging.error('new status %s is not valid' % (new_status))
        return None

    facet_list = read_facetlist(filename)
    # if facet_list==0: return None

    # save the old status, set the new one
    old_status = facet_list.loc[facet_num,'status']

    #check for sane transition (must go in order):
    # 'NEW' -> 'PRESCANNED' -> 'EXPOSED' -> 'SCANNED'
    old_index = facet_status.index(old_status)
    new_index = facet_status.index(new_status)
    if new_index != (old_index+1):
        logging.error('invalid facet transition %s to %s' % (old_status, new_status))
        return None

    facet_list.loc[facet_num,'status'] = new_status

    # re-write the CSV file
    facet_list.to_csv(filename)

    logging.info('status of facet %i changed from %s to %s' % (facet_num, old_status, new_status))

    return(filename, facet_num, old_status, new_status)

#------------------ ptrm class for re4ading and maipulating PTRM data -----------------

class ptrm:
    """The PTRM class reads and manipulates pointing timeline (PTRM) files.

    If no parameters are given, the currently directory is searched for the latest
    PTRM file, otherwise the latest file is read from the given directory.

    Otherwise a filename can be optionally specified; in this case the full path
    to the file should be given.

    The show() and merge() functions can then be used to display the PTRM and
    to produce a merged timeline consisting of OK and NOK blocks of time for
    MIDAS operations."""

    def __init__(self, directory='.', filename=None):

        self.filename = None
        if filename:
            self.filename = os.path.join(directory, filename)
        else:
            self.filename = self.latest(directory)

        if self.filename: self.ptrm = self.read(self.filename)


    def latest(self, directory):
        """Scans the given directory and returns the PTRM files with
        the highest index"""

        import glob

        # Search for PTRM files, filenames:
        # PTRM_PL_M???______01_?_RSM?PIM0.ROS

        ptrm=glob.glob(os.path.join(directory,'PTRM_PL_M???______01_?_RSM?PIM0.ROS'))

        if (len(ptrm)==0):
            print('ERROR: no PTRM files found in folder %s' % (directory))
            return False

        # Find the latest file - sort first by RS counter, then PI counter
        ptrm = sorted(ptrm, key=lambda x: ( int(os.path.basename(x)[26]) )) [-1]
        print('INFO: using PTRM file %s' % ptrm)

        return ptrm



    def read(self, filename):
        """Read a PTSL or PTRM file and return a dataframe containing the
        pointing type, start and end time."""

        # import xml.etree.ElementTree as etree
        from lxml import etree

        if not os.path.isfile(filename):
            print('ERROR: file %s cannot be found or opened' % (filename))
            return None

        # open and parse the XML file
        ptr = etree.parse(filename).getroot()

        # each MTP is one "segment" and each pointing is a timeline block
        blocks = ptr.findall('body/segment/data/timeline/block')
        block_types = [block.get('ref') for block in blocks]
        slew_index = [i for i, type in enumerate(block_types) if type=='SLEW']

        isofmt1 = "%Y-%m-%dT%H:%M:%S.%fZ"
        isofmt2 = "%Y-%m-%dT%H:%M:%SZ"

        mtp = []; start_time = []; end_time = []; block_type = []

        # print 'Number of blocks: %i' % (len(blocks))

        for i in range(len(blocks)):

            block_type.append(block_types[i])
            mtp.append(int(blocks[i].getparent().getparent().getparent().get('name').split('_')[1]))

            if i not in slew_index:
                start = blocks[i].find('startTime').text
                end = blocks[i].find('endTime').text
                # print 'Block type %s, start %s, end %s' % (block_types[i],start,end)
            else:
                start = blocks[i-1].find('endTime').text
                end = blocks[i+1].find('startTime').text
                # print 'Block type %s' % (block_types[i])

            start = start.strip()
            end = end.strip()

            if start[-1] != 'Z': start += 'Z'
            isofmt = isofmt1 if '.' in start.split(':')[-1] else isofmt2
            start_time.append(datetime.strptime(start,isofmt))

            if end[-1] != 'Z': end += 'Z'
            isofmt = isofmt1 if '.' in end.split(':')[-1] else isofmt2
            end_time.append(datetime.strptime(end,isofmt))

        pointing = pd.DataFrame(zip(mtp, block_type, start_time, end_time))
        pointing.columns = ['MTP', 'Activity', 'Start', 'End']

        pointing.filename = filename

        return pointing



    def merge(self, observations=None, selected=None, avoid_wols=False):
        """Accepts a list of PTRM entries and merges consecutive OK types to create
        a set of pointing blocks compatible with MIDAS operations"""

        ptrm = self.ptrm.sort('Start')
        blocks = pointing_ok[:]

        if avoid_wols:
            blocks.remove('MWOL')
            blocks.remove('MWNV')

        ok_blocks = ptrm[ptrm.Activity.isin(blocks)]
        num_blocks = len(ok_blocks)

        if selected:
            num_obs = len(observations)
            if (selected < 0) or (selected > num_obs):
                print('ERROR: selected observation is out of range')
                return False

        midas_blocks = []
        block_start = ok_blocks.irow(0).Start

        for idx in range(1,num_blocks):
            if ok_blocks.irow(idx).Start == ok_blocks.irow(idx-1).End:
                next
            else:
                block_end = ok_blocks.irow(idx-1).End
                midas_blocks.append([block_start,block_end])
                block_start = ok_blocks.irow(idx).Start

        block_end = ok_blocks.irow(-1).End
        midas_blocks.append([block_start,block_end])
        midas_blocks = pd.DataFrame(midas_blocks,columns=['start','end'])

        # Now we have a list of the valid time periods - we need to check which of these overlap
        # the selected observation. This defines the true periods into which we can slot ops.
        valid_blocks = []

        if selected != None:
            obs_start = observations.start_time[selected]
            obs_end = observations.end_time[selected]
        else:
            obs_start = ptrm.Start.iloc[0]
            obs_end = ptrm.End.iloc[-1]

        for idx in range(len(midas_blocks)):
            block = midas_blocks.irow(idx)
            if (obs_start <= block.end) and (block.start <= obs_end):
                valid_blocks.append(idx) # overlapping blocks

        midas_blocks = midas_blocks.irow(valid_blocks)

        # truncate the first and last blocks to the observation window if necessary
        if midas_blocks.irow(0).start < obs_start:
           midas_blocks.iloc[0,0] = obs_start

        if midas_blocks.irow(-1).end > obs_end:
           midas_blocks.iloc[-1,1] =  obs_end

        midas_blocks['duration'] = midas_blocks.apply(lambda row: row.end-row.start,axis=1)
        midas_blocks['offset'] = midas_blocks.start - obs_start

        self.merged = midas_blocks[midas_blocks.offset>0]
        self.merged = self.merged.reset_index()

        return



    def show(self, observations=False, selected=False):
        """Displays a timeline showing pointing blocks and, optionally,
        MIDAS operations."""

        import matplotlib.pyplot as plt
        import matplotlib.dates as md

        fig = plt.figure()

        ptrm = self.ptrm.copy()

        if type(selected) != bool:
            if type(observations)==bool:
                print('WARNING: if selected is set, observations must be given!')
                selected = False
            elif (type(selected) != int):
                print('WARNING: keyword selected must be an integer')
            elif (selected > len(observations)):
                print('WARNING: keyword selected must be an integer less than %i' % (len(observations)))
                selected = False
            else:
                observations = observations.iloc[[selected]]

        # If observations are passed, create a second plot showing obs blocks
        if (type(observations) != bool) and type(selected) == bool and not selected:
            # remove vertical space between the two subplots
            fig.subplots_adjust(hspace=0.00)
            ax = fig.add_subplot(2,1,1)
            obs_ax = fig.add_subplot(2,1,2, sharex=ax)
            obs_ax.hold(True)
        else:
            ax = fig.add_subplot(1,1,1)

        pt_ok = [block for block in pointing_ok if block != 'OBS']

        plot_blocks = []

        ok_hatch = [ "/" , "." , "|", "\\",'-', '+' ]
        nok_hatch = ['x']

        for index,pointing in enumerate(pt_ok):
            block = ptrm[ptrm.Activity==pointing]
            for idx,item in block.iterrows():
                span = ax.axvspan(item.Start, item.End, facecolor='g',label=pointing,hatch=ok_hatch[index])
            plot_blocks.append(span)

        for index,pointing in enumerate(pointing_nok):
            block = ptrm[ptrm.Activity==pointing]
            for idx,item in block.iterrows():
                span = ax.axvspan(item.Start, item.End, facecolor='r',label=pointing,hatch=nok_hatch[index])
            plot_blocks.append(span)

        plt.suptitle('Pointing visualisation for PTRM: %s' % (os.path.basename(self.filename)))

        legend_title = pt_ok + pointing_nok
        pt_legend = ax.legend(plot_blocks, legend_title, loc=0, fancybox=True)
        pt_legend.get_frame().set_alpha(0.7)
        ax.get_yaxis().set_ticks([])

        if (type(observations) != bool) and type(selected) == bool and not selected:

            ax.xaxis.set_visible(False)

            obs_span = []

            obs_types = observations.obs_type.unique()
            for obs_type in obs_types:
                blocks = observations[observations.obs_type==obs_type]
                for idx,block in blocks.iterrows():
                    span = obs_ax.axvspan(block.start_time, block.end_time,label=obs_type)
                obs_span.append(span)

            obs_legend = obs_ax.legend(obs_span, obs_types.tolist(), loc=0, fancybox=True)
            obs_legend.get_frame().set_alpha(0.7)
            obs_ax.get_yaxis().set_ticks([])

        if type(selected) != bool:
            ax.set_xlim(observations.start_time.values, observations.end_time.values)
            hours = observations.duration.values / np.timedelta64(1, 'h')

            ax.set_title('Selected observation: %s (COUNT=%i), duration %i hours' % \
                (observations.obs_type.values[0],observations.obs_id.values,hours))

        fig.autofmt_xdate()
        xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
        ax.xaxis.set_major_formatter(xfmt)

        plt.draw()

#------------------ ITL class for generating ITLs -----------------

class itl:
    """The ITL class is used to generate timeline files. The various MIDAS operations
    are represented by methods in this class, e.g. power_on() and scan(). Each call
    generates an ITL fragment and when complete itl.write() will create the final file.

    itl.time always represents the current end of the timeline sequence. itl.abs_time
    holds the equivalent absolute time.

    To generate an ITL referencing multiple observations/events, set itl.current_obs to
    the observation of interest (returned by validate_observations()) before any calls"""


    def __init__(self, obs=None, model='FM', start=None, end=None):
        """Create a new ITL object - an observation should be passed, as generated by
        validate_observations()."""

        self.itl = []
        self.time = timedelta(0)
        self.current_obs = obs
        self.cantilever = 1
        self.current_block = {}
        self.abs_time = None

        model_list = ['QM', 'FS', 'FM']
        model = model.upper()
        if model not in model_list:
            print('WARNING: invalid MIDAS model specified, defaulting to the FM')
            model = 'FM'

        self.model = model

        if obs is None: # create a dummy observation for testing or FS operations
            self.current_obs = pd.Series({
                'obs_type': 'DUMMY_OBSERVATION',
                'obs_id': 1,
                'start_time': pd.Timestamp('2015-01-01 00:00') if start is None else pd.Timestamp(start),
                'end_time': pd.Timestamp('2015-12-31 00:00') if end is None else pd.Timestamp(end),
                'duration': pd.Timestamp('2015-12-31 00:00') - pd.Timestamp('2015-01-01 00:00'),
                'start_id': 'MD_DUMMY___________________SO',
                'end_id': 'MD_DUMMY___________________EO' })

            self.current_block = {
                'duration': self.current_obs.duration,
                'end': self.current_obs.end_time,
                'index': 0,
                'offset': timedelta(0),
                'start': self.current_obs.start_time }


    def get_actions(self, outputdir='/tmp/'):
        """Run the EPS against the current ITL and produce the actions.out file needed to run the FS"""

        import eps_utils

        itl_file = os.path.join('/tmp', 'fs.itl')
        evf_file = os.path.expanduser('~/Dropbox/work/midas/operations/config/EVF__MD_M000_S000_01_A_RSM0PIM0.evf')

        self.write(itl_file)
        eps_utils.run_eps(itl_file, evf_file, fs=True, outputdir=outputdir)

        return os.path.abspath(os.path.join(outputdir, 'actions.out'))


    def generate(self, procedure, duration=timedelta(minutes=1)):
        """Generates an ITL fragment based on a template. The template has a set of default parameters
        and variable parameters. A dictionary containing all variable parameters must be passed.

        A search and replace is performed on the template file. stp_info contains the information
        needed to generate the correct filename. This can then be validated with the EPS."""

        import re, os

        obs = self.current_obs
        start = self.abs_time

        if procedure['template'] not in all_templates:
            logging.error('observation template %s not recognised' % (procedure['template']))
            return None

        template_file = all_templates[procedure['template']]

        # Read template and extract a list of parameters (tags) that need to be replaced
        template_file = os.path.join(template_dir, template_file)

        #TODO check file exists etc.
        templatefile = open(template_file, 'r')
        template = templatefile.read()
        templatefile.close()

        if procedure['template'] != 'COMMENT':

            # Add observation specific event label and count information
            procedure['params']['event_start'] = obs.start_id
            # procedure['params']['event_end'] = obs.end_id
            procedure['params']['event_count'] = obs.obs_id

        # find positions of opening and closing tags ('<', '>')
        opentag = [tag.start() for tag in re.finditer('<',template)]
        closetag = [tag.start() for tag in re.finditer('>',template)]

        # check we have an equal number of open and close tags
        if len(opentag) != len(closetag):
            logging.error('template file %s has %i open and %i closing tags - must be equal!' \
                % (template_file, len(opentag), len(closetag)))

        # get the list of unique tags
        tags = [template[opentag[n]+1:closetag[n]] for n in range(len(opentag))]
        tags = list(set(tags))

        logging.info('ITL template file %s opened with %i unique tags' % (template_file, len(tags)))
        logging.info('tag list for template: %s' % (tags))

        # check that all tags are included in the params dictionary
        matches = sum([True for key in tags if key in procedure['params']])
        if matches < len(tags):
            logging.error('%i tags in template %s but only %i parameters given' % \
                (len(tags), template_file, len(procedure['params'])))

        # open the file again and search and replace all tags
        templatefile = open(template_file, 'r')
        template = templatefile.read()
        templatefile.close()
        for tag in tags:
            template = template.replace('<'+tag+'>',str(procedure['params'][tag]))

        if procedure['template'] != 'COMMENT':

            # deal with the relative times (adding an offset if necessary)
            rel_times = re.findall(r'\{(.+?)\}',template)
            new_times = []
            for time in rel_times:
                # ignore pos/neg times for now - templates should only have positive times
                if( time.startswith('+') or time.startswith('-') ): time = time[1:]
                days = int(time.split('_')[0]) if time.find('_') >= 0 else 0
                hours,minutes,seconds = map(int,(time.split('_')[1].split(':'))) if time.find('_') >= 0 else map(int,time.split(':'))
                reltime = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
                shifted = reltime + self.time
                new_times.append("+%03d_%02d:%02d:%02d" % (shifted.days, shifted.seconds//3600, (shifted.seconds//60)%60, shifted.seconds%60))

            # Now search and replace each rel_times entry for the corresponding new_times entry
            for idx, time in enumerate(rel_times):
                template = template.replace('{'+time+'}',new_times[idx])

            end_time = shifted+duration
            real_duration = end_time - self.time
            remaining = self.current_block['offset'] + self.current_block['duration'] - end_time

            self.time = end_time
            self.abs_time = obs.start_time + self.time

            print('INFO: %s: %s (duration %s)' % (start, procedure['template'], real_duration))

            if end_time > self.current_block['offset'] + self.current_block['duration']:
                exceeded = -remaining # end_time - self.current_block['offset'] + self.current_block['duration']
                print('WARNING: observation is %s too long for block %i' % (exceeded, self.current_block['index']))
            else:
                print('INFO: %s left in block %i' % (remaining,self.current_block['index']))

        # Add the ITL regardless (some time over-runs are "soft")
        self.itl.append(template)


        return

    def wait(self, duration):
        """Simple insert a pause in the ITL and update the time and abs_time pointers"""

        self.time += duration
        self.abs_time = self.current_obs.start_time + self.time

        remaining = self.current_block['offset'] + self.current_block['duration'] - self.time

        print('INFO: duration of activity WAIT: %s' % (duration))
        print('INFO: %s left in block %i' % (remaining,self.current_block['index']))

    def wait_until(self, abs_time):
        """Insert a pause until the abs_time (if it's in the future!), update time pointers"""

        if type(abs_time) not in [pd.Timestamp, datetime]:
            abs_time = parser.parse(abs_time)
        if abs_time < self.abs_time:
            print('ERROR: specified time (%s) is before current time (%s)!' % (abs_time, self.abs_time))
            return None

        self.wait(abs_time-self.abs_time)


    def write(self, filename, itl_start=None, itl_end=None):
        """Accepts a list of ITL fragments and combines them into a single file, adding the
        necessary header information"""

        f=open(filename,'w')

        if itl_start is None:
            itl_start = self.abs_time-self.time

        if itl_end is None:
            itl_end = itl_start + self.time

        # write the minimum header
        print('Version:    00001', file=f)
        print('Start_time: %s' % (datetime.strftime(itl_start,dateformat)), file=f)
        print('End_time:   %s' % (datetime.strftime(itl_end,dateformat)), file=f)

        [f.write(fragment) for fragment in self.itl]
        print('INFO: written %i ITL fragments to %s' % (len(self.itl), filename))
        f.close()


    def set_current_block(self, blocks, index, offset=timedelta(0)):
        """Accepts a list of pointing-compatible time blocks generated from merge_ptrm()
        and an index to the current block. The ITL time pointer will be set to the start
        of this block and subsequent ITL fragments will be generated from here. An
        additional offset can be provided using the offset= keyword, which takes a
        timedelta as argument."""

        if index>=len(blocks):
            print('ERROR: requested block does not exist')
            return

        current_block = blocks.iloc[index].squeeze()

        self.current_block.update( {
            'index': index,
            'start': current_block.start,
            'end': current_block.end,
            'duration': timedelta(seconds=current_block.duration/np.timedelta64(1, 's')),
            'offset': timedelta(seconds=current_block.offset/np.timedelta64(1, 's'))
        })

        print('INFO: block %i has duration %s' % (index, self. current_block['duration']))

        self.time = self.current_block['offset'] + offset
        self.abs_time = self.current_obs.start_time + self.time


    #-------------------------------------
    # Below are python functions that create ITL fragments

    def comment(self, comment):
        """Inserts a string comment into an ITL template"""

        # Insert the current ITL time to the string
        comment = '%s : ' % self.abs_time + comment

        params = { 'template': 'COMMENT', 'params': { 'comment': comment} }
        self.generate(params, duration=timedelta(seconds=0))

        return


    def pstp_placeholder(self, duration, drate=333.0):
        """Insert a PSTP placeholder (test command)"""

        params = { 'template': 'PSTP', 'params': { 'data_rate': drate } }
        # duration = timedelta(minutes=1)
        self.generate(params, duration)

        return

    def pstp_prep(self):

        proc = {}
        proc['template'] = 'PSTP_PREP'
        duration = timedelta(seconds=30)

        self.generate(proc, duration)

        return

    def tech_cmd(self, cmds):
        """Sends one or more TCMDs to MIDAS. Input should be a list of commands (up to 20)"""

        proc = {}
        proc['template'] = 'TECH_CMD'
        duration = timedelta(minutes=1)

        if len(cmds)>20:
            print('ERROR: max 20 TCMDs can be given')
            return None

        # Remaining TCMDs are filled with 0x8F35 (NOOP)
        noop = 0x8F35
        num_noops = 20 - len(cmds)
        tcmds = [noop] * num_noops

        tcmds.extend(cmds)
        keys = ['tech_%i'%i for i in range(1,21)]
        proc['params'] = dict(zip(keys, tcmds))

        self.generate(proc, duration)

        return





    def power_on(self):
        """Power on via OBCP"""

        params = { 'template': 'POWER_ON', 'params': {} }
        duration = timedelta(minutes=2)
        self.generate(params, duration)

        return


    def power_off(self):
        """Power off via OBCP"""

        params = { 'template': 'POWER_OFF', 'params': {} }
        self.generate(params, timedelta(seconds=0))


    def power_subsys(self, cssc_on=False, app_on=False, lin_on=False, preamp_on=False, block1_on=False, block2_on=False,
        motor_on=False, wheel_on=False, hv_on=False, zstrain_on=False):
        """Command individual MIDAS subsystems ON or OFF"""

        proc = {}
        proc['template'] = 'SUBSYS_ONOFF'

        # if block1_on and block2_on:
        #     print('WARNING: both cantilever blocks should not be powered simultaneously')
        #     return False

        block1 = True # if self.cantilever <= 8 else False
        block2 = True # if self.cantilever > 8 else False

        # if (block1 and block2_on) or (block2 and block1_on):
        #     print('WARNING: this will change the powered cantilever block')


        proc['params'] = {
            'cssc_power': 'REL_ON' if cssc_on else 'REL_OFF',
            'app_lvdt_pwr': 'REL_ON' if app_on else 'REL_OFF',
            'lin_lvdt_pwr': 'REL_ON' if lin_on else 'REL_OFF',
            'preamp_pwr': 'REL_ON' if preamp_on else 'REL_OFF',
            'block1_pwr': 'REL_ON' if block1_on else 'REL_OFF',
            'block2_pwr': 'REL_ON' if block2_on else 'REL_OFF',
            'motor_drv_pwr': 'REL_ON' if motor_on else 'REL_OFF',
            'wheel_enc_pwr': 'REL_ON' if wheel_on else 'REL_OFF',
            'piezo_hv_pwr': 'REL_ON' if hv_on else 'REL_OFF',
            'z_strain_pwr': 'REL_ON' if zstrain_on else 'REL_OFF',
            }
        self.generate(proc, timedelta(minutes=1))

    def linear_abs(self, lin_pos):

        proc = {}

        if (lin_pos<-10 or lin_pos>10):
            print('ERROR: specified linear position out of range!')

        proc['template'] = 'LIN_ABS'
        proc['params'] = {
            'linear_posn': lin_pos }

        self.generate(proc, duration=timedelta(minutes=10))

        return


    def expose(self, facet, duration=None):

        proc = {}

        if (facet<0) or (facet>63):
            print('ERROR: facet number must be between 0 and 63')
            return False

        proc['template'] = 'EXPOSE'
        proc['params'] = {
            'segment': common.facet_to_seg(common.opposite_facet(facet)) }

        self.generate(proc)

        if duration is None:
            print('WARNING: no exposure duration specified, remember to issue CLOSE_SHUTTER sequence!')
        else:
            self.wait(duration)
            self.close_shutter()

        return


    def open_shutter(self):
        self.generate({'template': 'OPEN_SHUTTER', 'params': {}}, timedelta(minutes=2))
        return


    def close_shutter(self):
        self.generate({'template': 'CLOSE_SHUTTER', 'params': {}}, timedelta(minutes=2))
        return

    def linear_min(self):
        self.generate({'template': 'LINEAR_MIN', 'params': {}}, timedelta(minutes=5))
        return

    def linear_max(self):
        self.generate({'template': 'LINEAR_MAX', 'params': {}}, timedelta(minutes=5))
        return


    def wheel_move(self, segment, pwidth=147):

        proc = {}
        proc['template'] = 'WHEEL'

        if ((segment<0) or (segment>1023)):
            print('ERROR: invalid segment number')
            return None

        # eng = 21 + raw*42
        # RAW 4 = 189 us FS - 3 mins
        # RAW 3 = 147 us FM - 3 mins
        # RAW 2 = 105 us - 6 mins
        # RAW 1 = 63 us - 15 mins?

        if ((pwidth - 21) % 42) != 0:
            print('ERROR: pulse width cannot be converted to raw value')
            return None

        duration = {
            63:  15*60,
            105:  6*60,
            147:  3*60,
            189:  3*60 }

        proc['params'] = {
            'segment': segment,
            'pwidth' : pwidth,
            'tout': duration[pwidth] }

        self.generate(proc, duration=timedelta(seconds=duration[pwidth]))


    def xyz_move(self, x_dac=0, y_dac=0, z_dac=0, scan_mode='DYN', fsynth=False, zout=False):
        """Run SQ AMDF035B to manually set the XYZ stage to a given position"""

        proc = {}
        proc['template'] = 'XYZ_MOVE'

        if not(0 <= x_dac <= 65535) or not(0 <= y_dac <= 65535) or not(0 <= z_dac <= 65535):
            print('ERROR: DAC values must be between 0 and 65535')
            return None

        scan_type = ['DYN','CON','MAG']
        if scan_mode not in scan_type:
            print('ERROR: scan mode must be one of %s' % scan_type)
            return None

        proc['params'] = {
            'x_dac': x_dac,
            'y_dac': y_dac,
            'z_dac': z_dac,
            'scan_mode': scan_type.index(scan_mode),
            'fsynth_onoff': 'ON*' if fsynth else 'OFF*',
            'zout_onoff': 'ON*' if zout else 'OFF*' }

        self.generate(proc)

        return


    def instrument_setup(self, fscan_phase=False, last_z_lf=False, zero_lf=False, calc_retract=False,
        line_tx=True, ctrl_full=False, ctrl_retract=False, anti_creep=True, auto_exp=False,
        auto_thresh=32768, auto_trig=10, auto_seg=0, acreep=4, x_zoom=256, y_zoom=256, retr_m2=0, retr_m3=0,
        send_lines=True, mag_phase=False, hk1=0, hk2=0, hk3=0, hk4=0):
        """Calls SQ AMDF034B, which performs a variety of instrument setup tasks, including:

        - Set software flags
        - Setup automatic exposure parameters
        - Setup anti-creep parameters
        - Setup feature vector zoom parameters
        - Setup magnetic retraction parameters"""

        # The following parameters can be set by this routineL
        #
        #        VMDDE212 = <sw_flags> \         # SetSwFlagsPar
        #        VMDD2432 = <auto_exp_thresh> \  # SetAutoExpThresPar
        #        VMDD2442 = <auto_exp_trig> \    # SetAutoExpTriggerPar
        #        VMDD2452 = <auto_exp_seg> \     # SetAutoExpSegmentPar
        #        VMDD5082 = <acreep_factor> \    # SetAcreepFactorPar
        #        VMDD5102 = <xsteps_zoom> \      # SetXStepsZoomPar
        #        VMDD5112 = <ysteps_zoom> \      # SetYStepsZoomPar
        #        VMDDF122 = <mag_retr_2> \       # SetZRetractMag2Par
        #        VMDDF132 = <mag_retr_3> \       # SetZRetractMag3Par
        #        VMDDE112 = <hk1> \              # SetHkParam1Par
        #        VMDDE122 = <hk2> \              # SetHkParam2Par
        #        VMDDE132 = <hk3> \              # SetHkParam3Par
        #        VMDDE142 = <hk4> \              # SetHkParam4Par


        # Where sw_flags can be:
        #
        # Bit 0 (value = 0x0001) : SW_FSCAN_PHASE     1 = return phase signal during f-scan
        # Bit 1 (value = 0x0002) : SW_MOVEZ_LASTZ     1 = use last Z position on line feed
        # Bit 2 (value = 0x0004) : SW_MOVEZ_ZERO      1 = use zero position on line feed else use minimum Z position of last line
        # Bit 3 (value = 0x0008) : SW_CALC_RETRACT    1 = calculate Z retraction from X/Y step size
        #
        # Bit 4 (value = 0x0010) : SW_LSCAN_FULL      1 = transfer line scan data during full scan
        # Bit 5 (value = 0x0020) : SW_CDATA_FULL      1 = transfer control data during full scan
        # Bit 6 (value = 0x0040) : SW_CDATA_RETR      1 = transfer control data from retraction
        # Bit 7 (value = 0x0080) : SW_ANTI_CREEP      1 = perform anti-creep scan before full scan
        #
        # Bit 8 (value = 0x0100) : SW_AUTO_EXP        1 = enable auto-exposure mode
        # Bit 12                 : SW_ACREEP_FULL     1 = Tx anti-creep lines during fullscan
        # Bit 13                 : SW_MAGN_PHASE      1 = phase signal at magnetic positions
        proc = {}
        proc['template'] = 'SETUP'

        # Build the software flags word

        sw_flags = 0
        if fscan_phase: sw_flags  += 0x0001
        if last_z_lf: sw_flags    += 0x0002
        if zero_lf: sw_flags      += 0x0004
        if calc_retract: sw_flags += 0x0008
        if line_tx: sw_flags      += 0x0010
        if ctrl_full: sw_flags    += 0x0020
        if ctrl_retract: sw_flags += 0x0040
        if anti_creep: sw_flags   += 0x0080
        if auto_exp: sw_flags     += 0x0100
        if send_lines: sw_flags   += 0x1000
        if mag_phase: sw_flags    += 0x2000

        if not(0 <= auto_thresh <= 65535):
            print('ERROR: auto exposure dust count threshold must be between 0 and 65535 - setting default 32768')
            auto_thresh = 32768

        if not(0 <= auto_trig <= 65535):
            print('ERROR: auto exposure trigger count must be between 0 and 65535 - setting default 10')
            auto_trig = 10

        if not(0 <= auto_seg <= 1023):
            print('ERROR: auto exposure wheel segment must be between 0 and 1023 - setting default 0')
            auto_seg = 0

        if not(0 <= acreep <= 7):
            print('ERROR: anti-creep factor must be between 0 (2**0) and 7 (2**7) - setting default 4')
            auto_seg = 0

        x_zoom = int(x_zoom)
        y_zoom = int(y_zoom)

        if (x_zoom % 32 != 0) or (y_zoom % 32 != 0):
            print('ERROR: the number of pixels in X and Y for zooming must be a multiple of 32 - setting default 256')
            x_zoom = 256
            y_zoom = 256

        proc['params'] = {
            'sw_flags': sw_flags,
            'auto_exp_thresh': auto_thresh,
            'auto_exp_trig': auto_trig,
            'auto_exp_seg': auto_seg,
            'acreep_factor': acreep,
            'xsteps_zoom': x_zoom,
            'ysteps_zoom': y_zoom,
            'mag_retr_2': retr_m2,
            'mag_retr_3': retr_m3,
            'hk1': hk1,
            'hk2': hk2,
            'hk3': hk3,
            'hk4': hk4 }

        self.generate(proc)

        return


    def scan(self, cantilever, facet, channels=['ZS','PH','ST'], openloop=True, xpixels=256, ypixels=256, xstep=15, ystep=15, xorigin=False, yorigin=False, \
        xlh=True, ylh=True, mainscan_x=True, tip_offset=False, safety_factor=2.0, zstep=4, at_surface=False, pstp=False, fadj=85.0, op_amp=False, set_pt=False, \
        ac_gain=False, exc_lvl=False, auto=False, num_fcyc=8, fadj_numscans=2, set_start=True, z_settle=50, xy_settle=50, ctrl_data=False,
        contact=False, threshold=False, segment=None, dc_set=False, zapp_pos=0.):
        """Generic scan generator - minimum required is timing information, cantilever and facet - other parameters can
        be overridden if the defaults are not suitable. Generates an ITL fragment."""

        import scanning

        proc = {}

        if pstp: at_surface = True
        if contact: set_start = False

        # validate inputs
        if (cantilever<1) or (cantilever>16):
            print('ERROR: cantilever number must be between 1 and 16')
            return False

        self.cantilever = cantilever

        if (facet<0) or (facet>63):
            print('ERROR: facet number must be between 0 and 63')
            return False

        if (xpixels%32!=0) or (ypixels%32!=0):
            print('ERROR: number of X and Y pixels must be a multiple of 32')

        if (fadj<0.) or (fadj>100.):
            print('ERROR: frequency adjust must be between 0 and 100%')
            return False

        if (xpixels * ypixels * len(channels)) > common.memory_size:
            print('ERROR: memory capacity for a single image exceeded - reduce number of pixels of channels')
            return False

        if safety_factor < 1.0:
            print('WARNING: safety factor is < 1 - be sure you want to do this! ')

        #  This is an analogue value in the range +5V (minimum; Z piezo fully retracted) to -5V (maximum; Z piezo fully elongated)
        # defining the Z piezo start position to be set after an successful approach.
        # A value of 0.0 V (default) centers the Z piezo after an successful approach.
        # Because of measurement errors of the analogue signals, the maximum and minimum values (+5V, -5V) must not be used (approach does probably not converge)

        if not (-4.9 < zapp_pos < 4.9):
            print('ERROR: Z position after approach (zapp_pos) outside of the range -4.9 - +4.9 V')
            return False

        z_settle = int(z_settle)
        xy_settle = int(xy_settle)

        # Return parameters for selected cantilever
        # fscan = self.cantilever_select(cantilever)
        fscan_params = self.freq_scan(cantilever, num_scans=num_fcyc, params_only=True)

        # Centre the scan on the XY table unless specified
        if type(xorigin)==bool and type(yorigin)==bool:
            xorigin, yorigin = table_centre(xpixels, ypixels, xstep, ystep, openloop=openloop)

        xstep_nm = xstep * common.xycal['open']
        ystep_nm = ystep * common.xycal['open'] if openloop else ystep * common.xycal['closed']
        zretract_nm = max(xstep_nm,ystep_nm) * safety_factor
        zretract = int(np.around(zretract_nm / common.zcal))

        # Calculate scan duration and data rate
        dtype = common.get_channel(channels)
        ntypes = bin(dtype).count("1")
        if not dtype:
            print('ERROR: invalid datatype')
            return False

        # TODO check if this parameter is set!
        # In this template, the piezo creep avoidance is enabled by default
        # Need to add 1/8th of the pixels in the slow direction
        duration_xpix = xpixels + xpixels // 8 if not mainscan_x else xpixels
        duration_ypix = ypixels + ypixels // 8 if mainscan_x else ypixels

        duration_s = scanning.calc_duration(xpoints=duration_xpix, ypoints=duration_ypix, ntypes=ntypes, zretract=zretract,
            zsettle=z_settle, xysettle=xy_settle, zstep=zstep)
        duration_m = (duration_s // 60) + 1

        # Calculate data rate to insert into the Z record

        # data from line scans if we transmit lines as well
        # TODO implement this as a parameter
        # line scan 1072 bytes per line (32-512 points)
        num_lines = ypixels if mainscan_x else xpixels

        # data from image scan headers and packets
        image_data_bytes = ntypes * (80 + 2096 * xpixels * ypixels / 1024)
        line_data_bytes = (num_lines*1072)
        data_bytes = image_data_bytes + line_data_bytes

        # Add additional data rate if control data packets are enabled
        if ctrl_data:
            data_bytes += (32 * 32 * 2096)

        data_rate = (data_bytes*8/duration_s)

        # Set up the list of parameters for this template - these will be replaced in the template
        proc['params'] = {}

        if segment is None:
            segment = facet*16
        else:
            seg_facet = common.seg_to_facet(segment)
            if seg_facet != facet:
                print('ERROR: selected segment and facet do not agree!')
                return False

        # Linear and wheel move params
        linear_posn = self.tip_centre(cantilever) if not tip_offset else self.tip_position(cantilever,tip_offset)
        wheel_move_params = { \
                'linear_posn': linear_posn,
                # to always approach from the same direction, so first make a linear move 0.5 voltages "below"
                # the desired position
                'linear_posn_pre': linear_posn-0.5,
                'segment': segment }

        setup_params = { \

            # Parameters for AMDC101B  MIDAS scan setup
            'xorig': xorigin,
            'yorig': yorigin,
            'xpixels': xpixels,
            'ypixels': ypixels,
            'xstep': xstep,
            'ystep': ystep,
            'zstep': zstep,
            'z_ret': zretract,
            'fadj_numscans': fadj_numscans,
            'z_settle': z_settle,
            'xy_settle': xy_settle,
            'ctrl_data': 'ON*' if ctrl_data else 'OFF*',
            'scan_algo': 1 if threshold else 0 }

            # Contact mode parameters:
            # <dc_set_pt>, <contact_window>, <delta_dc_contact>
            # Typically set to:
            # dc_set_pt = 0.1
            # delta_dc_contact = 2*dc_set_pt
            # contact_window = 0.1 * dc_set_pt

        if contact:

            dc_set = 0.1 if type(dc_set)==bool else dc_set

            proc['params'].update({
                'dc_set_pt': dc_set,
                'delta_dc_contact': 2. * dc_set,
                'contact_window': 0.1 * dc_set })

        freq_params = { \

            # Parameters for AMDF026A MIDAS frequency scan
            'cant_num': fscan_params['cant_num'],
            'cant_block': fscan_params['cant_block'],
            'block1_pwr': fscan_params['block1_pwr'],
            'block2_pwr': fscan_params['block2_pwr'],
            'ac_gain': fscan_params['ac_gain'] if type(ac_gain)==bool else ac_gain,
            'exc_lvl': fscan_params['exc_lvl'] if type(exc_lvl)==bool else exc_lvl,
            'freq_hi': fscan_params['freq_hi'],
            'freq_lo': fscan_params['freq_lo'],
            'num_scans': fscan_params['num_scans'],
            'fstep_fine': fscan_params['fstep_fine'],
            'fstep_coarse': fscan_params['fstep_coarse'],
            'op_amp_per': -90 if type(op_amp)==bool else op_amp,
            'set_pt_amp': 80 if type(set_pt)==bool else set_pt }

        fullscan_params = { \

            # Parameters for AMDF028A MIDAS full scan
            'y_op_cl': 'OPEN' if openloop else 'CLOSED',
            'cssc_pwr': 'REL_ON', # 'REL_OFF' if openloop else 'REL_ON',
            'scan_dir': 'X' if mainscan_x else 'Y',
            'x_low_high': 'L_H' if xlh else 'H_L',
            'y_low_high': 'L_H' if ylh else 'H_L',
            'freq_adj': fadj,
            'channel': dtype,
            'zapp_pos': zapp_pos } # actually in the approach sequence

        zrec_params = { \
            'scan_data_rate': "%3.2f" % (data_rate) }


        if not auto:
            proc['params'].update(setup_params)

        if pstp:
            proc['template'] = 'PSTP_SETUP'
            proc['params'].update(setup_params)
            proc['params'].update( { \
                'x_low_high': 'L_H' if xlh else 'H_L',
                'y_low_high': 'L_H' if ylh else 'H_L' })
            self.generate(proc, timedelta(minutes=1))

            del proc
            proc = {}
            proc['params'] = {}
            proc['template'] = 'PSTP_SCAN'
            proc['params'].update(freq_params)
            proc['params'].update(fullscan_params)
            self.generate(proc, timedelta(minutes=duration_m))

        else:

            if auto:
                proc['template'] = 'SCAN_NOSETUP'
            elif at_surface:
                if contact:
                    proc['template'] = 'CONTACT_SCAN_AT_SURFACE'
                else:
                    proc['template'] = 'SCAN_AT_SURFACE'
            else:
                if contact:
                    proc['template'] = 'CONTACT_SCAN'
                else:
                    proc['template'] = 'SCAN'
                proc['params'].update(wheel_move_params)

            proc['params'].update(freq_params)
            proc['params'].update(fullscan_params)
            proc['params'].update(zrec_params)

            # allow frequency to be set to the start manually via a TechCmd
            if set_start:

                self.tech_cmd( [779, fscan_params['freq_hi_dig'], 780, fscan_params['freq_lo_dig'], 33536])
                self.wait(timedelta(minutes=1))

            self.generate(proc, timedelta(minutes=duration_m))

        return

    def xy_cal(self, cantilever, channels=['ZS', 'PH', 'ST'], openloop=True, xpixels=256, ypixels=256, xstep=15, ystep=15, \
        xlh=True, ylh=True, mainscan_x=True, at_surface=False, set_start=True, fadj_numscans=2,
        ac_gain=False, exc_lvl=False, set_pt=False, op_amp=False, fadj=85.0, z_settle=50, xy_settle=50,
        contact=False, threshold=False, dc_set=False, zapp_pos=0.0):
        """XY calibration - calls scan() with default cal parameters (which can be overriden)"""

        self.scan(cantilever=cantilever, facet=2, channels=channels, openloop=openloop, \
            xlh=xlh, ylh=ylh, mainscan_x=mainscan_x, at_surface=at_surface, safety_factor=4.0, \
            xpixels=xpixels, ypixels=ypixels, xstep=xstep, ystep=ystep,
            ac_gain=ac_gain, exc_lvl=exc_lvl, set_start=set_start, z_settle=z_settle, xy_settle=xy_settle,
            op_amp=op_amp, fadj=fadj,  set_pt=set_pt, fadj_numscans=fadj_numscans, contact=contact, threshold=threshold,
            dc_set=dc_set, zapp_pos=zapp_pos)

        return


    def z_cal(self, cantilever, channels=['ZS', 'PH', 'ST'], openloop=True, xpixels=256, ypixels=256, xstep=10, ystep=10, \
        xlh=True, ylh=True, mainscan_x=True, zstep=4, contact=False, threshold=False, dc_set=False, zapp_pos=0.0):
        """Z calibration - calls scan() with default cal parameters (which can be overriden)"""

        self.scan(cantilever=cantilever, facet=1, channels=channels, openloop=openloop, \
            xpixels=xpixels,  ypixels=ypixels, xstep=xstep, ystep=ystep, zstep=zstep, \
            xlh=xlh, ylh=ylh, mainscan_x=mainscan_x, safety_factor = 4.0, contact=contact, threshold=threshold,
            dc_set=dc_set, zapp_pos=zapp_pos)

        return

    def tip_cal(self, cantilever, channels=['ZS', 'PH', 'ST'], openloop=True, xpixels=256, ypixels=256, xstep=False, ystep=False,
        xlh=True, ylh=True, mainscan_x=True, zstep=4, xorigin=False, yorigin=False,
        fadj=85.0, op_amp=False, set_pt=False, ac_gain=False, exc_lvl=False, num_fcyc=8, set_start=True, fadj_numscans=2,
        z_settle=50, xy_settle=50, contact=False, threshold=False, dc_set=False, zapp_pos=0.0):
        """Tip calibration - calls scan() with default cal parameters (can be overridden)"""

        # set default steps according to open or closed loop mode
        if not xstep:
            xstep = 4
        if not  ystep:
            if openloop:
                ystep = 4
            else:
                ystep = 11

        self.scan(cantilever=cantilever, facet=3, channels=channels, openloop=openloop,
            xpixels=xpixels,  ypixels=ypixels, xstep=xstep, ystep=ystep, zstep=zstep,
            xlh=xlh, ylh=ylh, mainscan_x=mainscan_x, safety_factor = 4.0, xorigin=xorigin, yorigin=yorigin,
            op_amp=op_amp, fadj=fadj, set_pt=set_pt, ac_gain=ac_gain, exc_lvl=exc_lvl, num_fcyc=num_fcyc,
            set_start=set_start, fadj_numscans=fadj_numscans, z_settle=z_settle, xy_settle=xy_settle,
            contact=contact, threshold=threshold, dc_set=dc_set, zapp_pos=zapp_pos)

        return


    def post_scan(self):
        """Retracts approach stage to minimum and switches off un-used subsystems"""

        self.approach_min()

        self.power_subsys(cssc_on=False, app_on=False, lin_on=False, preamp_on=False,
            block1_on=True if self.cantilever <= 8 else False,
            block2_on=False if self.cantilever <= 8 else True,
            motor_on=False, wheel_on=False, hv_on=False, zstrain_on=False)

        return


    def approach_min(self):
        """Move the approach stage to minimum position"""

        proc = {}
        proc['template'] = 'APP_MIN'
        proc['params'] = { }

        self.generate(proc, timedelta(minutes=3))
        return


    def approach_abs(self, abs_pos=6.0):
        """Move the approach stage to an absolute position"""

        proc = {}
        proc['template'] = 'APP_ABS'
        proc['params'] = { 'abs_pos': abs_pos }

        self.generate(proc, timedelta(minutes=3))
        return


    def abort(self):
        """Clears the TC queue and aborts any active task"""

        proc = {}
        proc['template'] = 'ABORT'
        proc['params'] = { }

        self.generate(proc, timedelta(seconds=30))
        return



    def cant_survey(self, cant_num):
        """Performs a surface of usable parameters for a given cantilever using
        all available gain and excitation values"""

        proc = {}
        proc['template'] = 'CANT_SURVEY'

        # Fix many parameters - always use 4 scans with fstep of 2 Hz centred on resonance
        fscan = self.cantilever_select(cant_num)
        num_scans = 4
        fstep_coarse = 2.0
        fstep_fine = 0.1

        # calculate start frequency based on step size and number of scans
        res_freq = fscan['res_freq']
        freq_range = fstep_coarse * 256 * num_scans
        start_freq = res_freq - freq_range/2.

        freq_hi_dig = int(start_freq / common.freq_hi_cal)
        freq_hi = freq_hi_dig * common.freq_hi_cal

        freq_lo_dig = int((start_freq - freq_hi)/common.freq_lo_cal)
        freq_lo = freq_lo_dig * common.freq_lo_cal

        freq_hi = round(freq_hi,3)
        freq_lo = round(freq_lo,3)

        proc['params'] = { \

            # set these parameters according to the cantilever
            'cant_num': fscan['cant_num'],
            'cant_block': fscan['cant_block'],
            'block1_pwr': fscan['block1_pwr'],
            'block2_pwr': fscan['block2_pwr'],

            'freq_hi': freq_hi,
            'freq_lo': freq_lo,
            'fstep_coarse': fstep_coarse,
            'fstep_fine': fstep_fine,
            'num_scans': num_scans,
            'freq_hi_dig': freq_hi_dig }

        self.generate(proc, timedelta(minutes=10))
        return


    def freq_scan(self, cantilever, ac_gain=False, excitation=False, fstep_coarse=False, fstep_fine=False, num_scans=8, \
        op_amp=-90, set_pt=80, fadj=85, params_only=False, set_start=False):

        proc = {}
        proc['template'] = 'FSCAN'

        fscan = self.cantilever_select(cantilever)
        self.cantilever = cantilever

        # calculate start frequency based on step size and number of scans
        res_freq = fscan['res_freq']
        fstep_coarse = fscan['fstep_coarse'] if type(fstep_coarse) == bool else fstep_coarse

        freq_range = fstep_coarse * 256 * num_scans
        start_freq = res_freq - freq_range/2.

        freq_hi_dig = int(start_freq / common.freq_hi_cal)
        freq_hi = freq_hi_dig * common.freq_hi_cal

        freq_lo_dig = int((start_freq - freq_hi)/common.freq_lo_cal)
        freq_lo = freq_lo_dig * common.freq_lo_cal

        freq_hi = round(freq_hi,3)
        freq_lo = round(freq_lo,3)

        proc['params'] = { \

            # set these parameters according to the cantilever
            'cant_num': fscan['cant_num'],
            'cant_block': fscan['cant_block'],
            'block1_pwr': fscan['block1_pwr'],
            'block2_pwr': fscan['block2_pwr'],

            # use "standard" values for excitation and gain unless overridden
            'ac_gain': fscan['ac_gain'] if type(ac_gain) == bool else ac_gain,
            'exc_lvl': fscan['exc_lvl'] if type(excitation) == bool else excitation,

            # Start frequency is set to value calculated from resonance
            'freq_hi': freq_hi,
            'freq_hi_dig': freq_hi_dig,
            'freq_lo': freq_lo,
            'freq_lo_dig': freq_lo_dig,

            'fstep_coarse': fstep_coarse,
            'fstep_fine': fscan['fstep_fine'] if type(fstep_fine) == bool else fstep_fine,
            'num_scans': num_scans,
            'op_amp_per': op_amp,
            'set_pt_amp': set_pt,
            'freq_adj': fadj }

        if params_only:
            return proc['params']
        else:

            # allow frequency to be set to the start manually via a TechCmd
            if set_start:

                self.tech_cmd( [779, freq_hi_dig, 780, freq_lo_dig, 33536])
                self.wait(timedelta(minutes=1))

            duration = common.fscan_duration(num_scans)
            self.generate(proc, duration)

        return


    def linear_tile(self, num_tiles, overlap, cantilever, facet, channels=['ZS','PH', 'ST'], openloop=True,
        xpixels=256, ypixels=256, xstep=15, ystep=15, xlh=True, ylh=True, mainscan_x=True, fadj=85.,
        safety_factor=2.0, zstep=4):
        """Generate a series of identical scans at different linear positions, with a given
        overlap (specified as a percentage). Scans are centered on the wheel."""

        num_tiles = int(num_tiles)
        overlap = float(overlap)

        xlen_um = xpixels*xstep*common.xycal['open']
        xlen_um = xlen_um / 1000.
        total_len_um = xlen_um*float(num_tiles) - (num_tiles-1)*xlen_um*overlap
        start_pos = -(total_len_um-xlen_um)/2
        tip_offset = start_pos + np.arange(0.,num_tiles)*xlen_um*(1.-overlap)

        for scan_count in range(0,num_tiles):
            offset = tip_offset[scan_count]
            print('INFO: starting scan at linear offset %3.3f' % offset)

            self.scan(cantilever=cantilever, facet=facet, channels=channels, openloop=openloop, xpixels=xpixels,
                ypixels=ypixels, xstep=xstep, ystep=ystep, xlh=xlh, ylh=ylh, mainscan_x=mainscan_x,
                fadj=fadj, safety_factor=safety_factor, zstep=zstep, tip_offset=offset)

            # TODO - quick hack to also add fine prescans in
            # self.scan(cantilever=cantilever, facet=facet, openloop=openloop, xpixels=128, ypixels=128, xstep=14, ystep=39, at_surface=True)

        return

# cantilever, facet, channels=['ZS','PH','ST'], openloop=True, xpixels=256, ypixels=256, xstep=15, ystep=15, xorigin=False, yorigin=False, \
#    xlh=True, ylh=True, mainscan_x=True, tip_offset=False, safety_factor=2.0, zstep=4, at_surface=False, pstp=False, fadj=85.0, op_amp=False, set_pt=False, \
##    ac_gain=False, exc_lvl=False,

    def tile_scan(self, x_tiles, y_tiles, overlap, cantilever, facet, channels=['ZS','PH', 'ST'], openloop=True, xpixels=256, ypixels=256,
        xstep=15, ystep=15, xlh=True, ylh=True, mainscan_x=True, tip_offset=False, fadj=85.0, safety_factor=2.0, zstep=4, at_surface=False,
        xorigin=False, yorigin=False, exc_lvl=False, ac_gain=False, op_amp=False, set_start=False, xy_settle=50., z_settle=50, set_pt=False):
        """Generates a series of identical tiled scans of a single target following an approach.

        The number of x and y tiles is given, all other parameters are as per scan().
        overlap= specifies the fractional overlap between scans (0-1)"""

        num_scans = x_tiles*y_tiles

        # Calculate the total coverage of the tiled scans, taking into account the requested overlap
        x_overlap = int(xpixels*xstep*overlap)
        x_extent = x_tiles*xpixels*xstep - (x_tiles-1)*x_overlap
        y_overlap = int(ypixels*ystep*overlap)
        y_extent = y_tiles*ypixels*ystep - (y_tiles-1)*y_overlap

        # Calculate the X/Y origin for hte start of the tile
        x_centre = common.centre_open # X is always in open loop!
        y_centre = common.centre_open if openloop else common.centre_closed

        if not xorigin: xorigin = x_centre-(x_extent)/2
        if not yorigin: yorigin = y_centre-(y_extent)/2

        for y in range(y_tiles):
            for x in range(x_tiles):
                if (y==0) & (x==0) & (at_surface==False): # first scan in sequence
                    surface = False
                else:
                    surface = True
                    set_start = False # fscan already in correct range

                # Calculate the origin for this scan
                xorig = xorigin+x*(xpixels*xstep-x_overlap)
                yorig = yorigin+y*(ypixels*ystep-y_overlap)

                self.scan(cantilever=cantilever, facet=facet, channels=channels, openloop=openloop,
                xpixels=xpixels, ypixels=ypixels, xstep=xstep, ystep=ystep, xorigin=xorig, yorigin=yorig,
                xlh=xlh, ylh=ylh, mainscan_x=mainscan_x, tip_offset=tip_offset, fadj=fadj,
                safety_factor=safety_factor, zstep=zstep, at_surface=surface, exc_lvl=exc_lvl, set_pt=set_pt,
                ac_gain=ac_gain, set_start=set_start, xy_settle=xy_settle, z_settle=z_settle, op_amp=op_amp)

        return


    def feature(self, trend=True, median=True, count_pix=True, pix_gt=True, pix_lt=False, check_height=True,
            set_zoom=False, zoom_max=False, check_shape=False,
            height_thresh=50.0, x_marg=(0,0), y_marg=(0,0), num_points=20, avg_height=40, pix_area=50.0, zoom=-0.0031):

        proc = {}
        proc['template'] = 'FEATURE'

        # Bit 0 : FVECT_TREND_COR perform trend correction
        # Bit 1: FVECT_MLINE_COR perform median line subtraction
        #
        # Bit 4 : FVECT_NPIX_CRIT apply number of pixels criterion
        # Bit 5 : FVECT_MIN_PIXEL number of pixels must be >= given value
        # Bit 6 : FVECT_MAX_PIXEL number of pixels must be <= given value
        #
        # Bit 8 : FVECT_AVGZ_CRIT apply average height over Z criterion
        # Bit 9 : FVECT_ZOOM_NEW use new (commanded) dimensions for zooming
        # Bit 10 : FVECT_ZOOM_MAX maximize zoom area (different zoom factors in X and Y)
        #
        # Bits 12-13 : FVECT_XYS_CRIT apply X/Y shape criterion (0 = disabled)

        # SetFvectModePar (hex) -  <fvec_mode>
        mode = 0
        if trend: mode += 2**0
        if median: mode += 2**1
        if count_pix: mode += 2**4
        if pix_gt: mode += 2**5
        if pix_lt: mode += 2**6
        if check_height: mode += 2**8
        if set_zoom: mode += 2**9
        if zoom_max: mode += 2**10

        if type(check_shape)!=bool:
            # X/Y extent is checked and 3 ratios (R) allowed:
            # 0: disabled
            # 1: 1/2 >= R <=2
            # 2: 1/4 >= R <= 4
            # 3: 1/8 >= R <= 8
            if (check_shape >=0) & (check_shape <=3):
                mode += 0x1000 * check_shape
            else:
                print('WARNING: invalid shape criteria, disabling')

        # SetFvectLpercPar (%) - <height_thresh>
        if 0.0 >= height_thresh >= 100.0:
            print('WARNING: invalid height threshold, must be between 0 and 100%')
            height_thresh = 50.0

        # SetFvectRatioPar (%) - <pix_area_ratio>
        if 0.0 >= pix_area <= 100.0:
            print('WARNING: invalid pixel/area ratio, must be between 0 and 100%')
            pix_area = 50.0

        # SetFvectZfactorPar (%) - <zoom_factor>
        # if 0.0 >= zoom >= 200.0:
        #     print('WARNING: zoom factor must be between 0 and 200%')
        #     zoom = 50.0

        proc['params'] = {
            # Feature vector parameters
            'fvec_mode': mode,
            'height_thresh': height_thresh,
            'x_margin': x_marg[1]*256+x_marg[0],
            'y_margin': y_marg[1]*256+x_marg[0],
            'num_points': num_points,
            'avg_height': avg_height,
            'pix_area_ratio': pix_area,
            'zoom_factor': zoom }

        self.generate(proc, timedelta(minutes=5))

        return

    def auto_zoom(self, channels=['ZS','PH','ST'], xlh=True, ylh=True, mainscan_x=True,
            trend=True, median=True, count_pix=True, pix_gt=True, pix_lt=False, check_height=True,
            set_zoom=False, zoom_max=False, check_shape=False,
            height_thresh=50.0, x_marg=(0,0), y_marg=(0,0), num_points=20, avg_height=40, pix_area=50.0, zoom=-0.0031):

        self.feature(trend=True, count_pix=count_pix, check_height=check_height, check_shape=check_shape,
            height_thresh=height_thresh, x_marg=x_marg, y_marg=y_marg, num_points=num_points, avg_height=avg_height, pix_area=pix_area, zoom=zoom)

        # notice that facet=0 is a dummy argument here since when at_surface=True facet is not needed/used
        self.scan(cantilever=self.cantilever, facet=0, channels=channels, xlh=xlh, ylh=ylh, mainscan_x=mainscan_x, at_surface=True, auto=True)

        return

    def line_scan(self, cantilever, facet, channels=['ZS'], openloop=True, xpixels=128, ypixels=128, xstep=15, ystep=15, \
        xorigin=False, yorigin=False, xlh=True, ylh=True, mainscan_x=True, fadj=85.0, safety_factor=2.0, zstep=4,
        ac_gain=False, exc_lvl=False, op_amp=False, set_pt=False, num_fcyc=8, fadj_numscans=2, set_start=True,
        at_surface=False, ctrl_data=False, tip_offset=False, app_max=-6.0, z_settle=50, xy_settle=50, threshold=False, contact=False,
        segment=None, dc_set=False, zapp_pos=0.0):

        import scanning
        proc = {}

        if contact: set_start=False

        if at_surface:
            if contact:
                proc['template'] = 'LINE_CON_SURF'
            else:
                proc['template'] = 'LINE_SURF'
        else:
            if contact:
                proc['template'] = 'LINE_CONTACT'
            else:
                proc['template'] = 'LINE_SCAN'

        # validate inputs
        if (cantilever<1) or (cantilever>16):
            print('ERROR: cantilever number must be between 1 and 16')
            return False

        if (facet<0) or (facet>63):
            print('ERROR: facet number must be between 0 and 63')
            return False

        if (xpixels%32!=0) or (ypixels%32!=0):
            print('ERROR: number of X and Y pixels must be a multiple of 32')

        if (fadj<0.) or (fadj>100.):
            print('ERROR: frequency adjust must be between 0 and 100%')
            return False

        if not (-4.9 < zapp_pos < 4.9):
            print('ERROR: Z position after approach (zapp_pos) outside of the range -4.9 - +4.9 V')
            return False

        # Return parameters for selected cantilever
        # fscan = self.cantilever_select(cantilever)
        fscan_params = self.freq_scan(cantilever, num_scans=num_fcyc, params_only=True)

        # Centre the scan on the XY table unless specified
        if type(xorigin)==bool and type(yorigin)==bool:
            xorigin, yorigin = table_centre(xpixels, ypixels, xstep, ystep, openloop=openloop)

        xstep_nm = xstep * common.xycal['open']
        ystep_nm = ystep * common.xycal['open'] if openloop else ystep * common.xycal['closed']

        zretract_nm = max(xstep_nm,ystep_nm) * safety_factor
        zretract = int(np.around(zretract_nm / common.zcal))

        # Calculate scan duration and data rate
        dtype = common.get_channel(channels)
        ntypes = bin(dtype).count("1")
        if not dtype:
            print('ERROR: invalid datatype')
            return False

        if mainscan_x:
            ypixels = 1
        else:
            xpixels = 1

        duration_s = scanning.calc_duration(xpoints=xpixels, ypoints=ypixels, ntypes=ntypes, zretract=zretract, zstep=zstep,
            ctrl=ctrl_data, zsettle=z_settle, xysettle=xy_settle)

        # Control data packets: 1048 words (2096 bytes), one packet per line point, max 32
        # Line scan as well: 1072 bytes (single line)

        # Round up to the next minute
        duration_s = int(60 * round(duration_s/60))

        data_bytes = 1072 # line scan
        if ctrl_data: data_bytes += (32 * 2096)
        data_rate = data_bytes*8/duration_s

        linear_posn = self.tip_centre(cantilever) if not tip_offset else self.tip_position(cantilever,tip_offset)

        if segment is None:
            segment = facet*16
        else:
            seg_facet = common.seg_to_facet(segment)
            if seg_facet != facet:
                print('ERROR: selected segment and facet do not agree!')
                return False

        # Set up the list of parameters for this template - these will be replaced in the template
        proc['params'] = { \

            'linear_posn': linear_posn,
            'linear_posn_pre': linear_posn-0.5,
            'segment': segment,

            # Cantilever dependent fscan parameters
            'cant_num': fscan_params['cant_num'],
            'cant_block': fscan_params['cant_block'],
            'block1_pwr': fscan_params['block1_pwr'],
            'block2_pwr': fscan_params['block2_pwr'],
            'freq_hi': fscan_params['freq_hi'],
            'freq_lo': fscan_params['freq_lo'],
            'ac_gain': fscan_params['ac_gain'] if type(ac_gain)==bool else ac_gain,
            'exc_lvl': fscan_params['exc_lvl'] if type(exc_lvl)==bool else exc_lvl,
            'fstep_fine': fscan_params['fstep_fine'],
            'fstep_coarse': fscan_params['fstep_coarse'],
            'op_amp_per': -90 if type(op_amp)==bool else op_amp,
            'set_pt_amp': 80 if type(set_pt)==bool else set_pt,
            'num_scans': num_fcyc,

            # Scan related parameters
            'y_op_cl': 'OPEN' if openloop else 'CLOSED',
            'cssc_pwr': 'REL_ON', # 'REL_OFF' if openloop else 'REL_ON',
            'xpixels': xpixels,
            'ypixels': ypixels,
            'xstep': xstep,
            'ystep': ystep,
            'zstep': zstep,
            'xorig': xorigin,
            'yorig': yorigin,
            'scan_dir': 'X' if mainscan_x else 'Y',
            'x_low_high': 'L_H' if xlh else 'H_L',
            'y_low_high': 'L_H' if ylh else 'H_L',
            'z_ret': zretract,
            'zapp_pos': zapp_pos,
            'freq_adj': fadj,
            'fadj_numscans': fadj_numscans,
            'channel': dtype,
            'ctrl_data': 'ON*' if ctrl_data else 'OFF*',
            'app_max': app_max,
            'xy_settle': xy_settle,
            'z_settle': z_settle,
            'scan_algo': 1 if threshold else 0,

            'scan_data_rate': "%3.2f" % (data_rate) }

            # Contact mode parameters:
            # <dc_set_pt>, <contact_window>, <delta_dc_contact>
            # Typically set to:
            # dc_set_pt = 0.1
            # delta_dc_contact = 2*dc_set_pt
            # contact_window = 0.1 * dc_set_pt

        if contact:

            dc_set = 0.1 if type(dc_set)==bool else dc_set

            proc['params'].update({
                'dc_set_pt': dc_set,
                'delta_dc_contact': 2. * dc_set,
                'contact_window': 0.1 * dc_set })

        if set_start:
            self.tech_cmd( [779, fscan_params['freq_hi_dig'], 780, fscan_params['freq_lo_dig'], 33536])
            self.wait(timedelta(minutes=1))

        self.generate(proc, timedelta(seconds=duration_s))

        return


    def cantilever_select(self, cant_num):
        """Accepts an integer cantilever number (1-16) and returns appropriate
        parameters (frequency, block power, etc.)"""

        if (cant_num<1) or (cant_num>16):
            print('ERROR: cantilever number must be between 1 and 16')
            return False

        # resonance frequency, used to calculate alternative windows
        # values taken from the re-commissioning MD01 block
        # Update in ongoing STPs!

        if self.model=='FM':

            res_freq = [ 83670.2, 84252.0, 84177.9, 89675.2, 81685.1, 82996.1, 87750.0, 93768.9,
                        108411.1, 85449.6, 86508.1, 94853.1, 84628.8, 84056.2, 83569.9, 89290.5 ]

            # cant     1 2 3 4 5 6 7 8  9 0 1 2 3 4 5 6
            ac_gain = [3,3,1,6,4,4,5,3, 7,3,2,4,2,2,1,3]
            exc_lvl = [2,1,1,2,3,3,5,6, 4,1,2,6,2,2,2,1]


        elif self.model=='FS':

            res_freq = [100040.,  99090., 80000.0, 80000.0, 80000.0, 80000.0, 97300.0, 100240.0,
                        102970., 101340., 80000.0, 80000.0, 80000.0, 80000.0, 85000.0,  88070.0 ]

            # cant     1 2 3 4 5 6 7 8  9 0 1 2 3 4 5 6
            exc_lvl = [2,1,1,2,3,3,4,6, 4,1,2,6,2,2,2,1]
            ac_gain = [2,2,1,3,3,4,5,3, 7,2,2,4,2,2,1,2]

        else:
            print('ERROR: invalid MIDAS model specified!')
            return False


        # start values for a standard frequency sweep with 1 Hz step and 8 scans
        # freq_hi = [82745, 84160, 84080, 89550, 81650, 82210, 86950, 93100, \
        #        108000, 85400, 86450, 94850, 84550, 84000, 83500, 89200]

        freq_hi = [res - 1000. for res in res_freq]

        fstep_coarse = [0.2, 0.2, 0.2, 1.0,     0.5, 1.0, 2.0, 2.0, \
                        1.5, 0.3, 0.3, 1.5,     0.3, 0.3, 0.3, 0.3]

        fstep_fine = [fstep * 0.1 for fstep in fstep_coarse]

        # frequency step for the fine tuning - determine from FWHM TODO!!
        # fstep_fine = [0.03, 0.03, 0.03, 0.03, 0.1, 0.1, 0.5, 0.1, \
        #               0.15, 0.03, 0.03, 0.1, 0.03, 0.03, 0.03, 0.03]

        params = {
            'cant_num'  : (cant_num-1) % 8,
            'cant_block': 0 if cant_num <= 8 else 1,
            'block1_pwr': 'REL_ON', # if cant_num <= 8 else 'REL_OFF',
            'block2_pwr': 'REL_ON', # if cant_num <= 8 else 'REL_ON',
            'freq_hi'   : freq_hi[cant_num-1],
            'ac_gain'   : ac_gain[cant_num-1],
            'exc_lvl'   : exc_lvl[cant_num-1],
            'res_freq'  : res_freq[cant_num-1],
            'fstep_fine'  : fstep_fine[cant_num-1],
            'fstep_coarse' : fstep_coarse[cant_num-1] }

        return params


    def tip_centre(self, cant_num):
        """Returns the voltage of the linear position necessary to centre the wheel
        on the selected cantilever"""

        if (cant_num<1) or (cant_num>16):
            print('ERROR: cantilever number must be between 1 and 16')
            return False

        return common.lin_centre_pos_fm[cant_num-1] if self.model=='FM' else common.lin_centre_pos_fs[cant_num-1]


    def tip_position(self, cant_num, distance):
        """Returns the linear position voltage corresponding to a positive or negative
        offset from the centre of the wheel for the given cantilever."""

        centre = self.tip_centre(cant_num)
        offset = distance * common.linearcal
        if abs(offset) > common.lin_max_offset:
            print('WARNING: linear stage out of range for tip %i' % cant_num)
            return False
        offset_v = centre + offset
        v_per_bit = 20./65535.
        offset_v = np.rint(offset_v/v_per_bit)*v_per_bit



        return offset_v

#------ end of itl class definition



def table_centre(xpixels, ypixels, xstep, ystep, openloop=True):
    """Takes an image size (number of steps, step size) and returns the origin
    to position this image at the centre of the XY table (to ensure linearity).

    If openloop=False then the Y axis is assumed to be in closed loop mode"""

    x_centre = common.centre_open # X is always in open loop!
    y_centre = common.centre_open if openloop else common.centre_closed

    xorigin = x_centre-(xpixels*xstep)/2
    yorigin = y_centre-(ypixels*ystep)/2

    return xorigin,yorigin


def load_fecs(filename=fecs_evf_file):
    """Loads the SGS produced EVF format version of the FECS file from ROS_SGS"""

    # Format
    # 20-January-2014_14:35:00       COMMS_START       (COUNT =   1)
    # Events:
    # 'COMMS_START',     'COMMS_END',
    # 'DUMP_START',      'DUMP_END',
    # 'USO_CHECK_START', 'USO_CHECK_END',
    # 'DELTA_DOR_START', 'DELTA_DOR_END']

    cols = ['time','event','cnt']
    fecs = pd.read_table(filename,delim_whitespace=True,names=cols,usecols=[0,1,4],header=None,skiprows=3)

    # remove brackets and make new column an integer
    fecs.cnt=fecs.cnt.apply( lambda x: int(x.split(')')[0]) )

    # replace the underscore in the time witih a space and convert to datetime
    fecs.time = fecs.time.str.replace('_',' ')
    fecs.time = pd.to_datetime(fecs.time)

    return fecs


def get_dumps(start, end):
    """Lists the data dumps between start and end (both datetime)."""

    fecs = load_fecs()
    fecs = fecs[ (fecs.time>=start) & (fecs.time<=end)]

    dump_start = fecs[ (fecs.event=='DUMP_START') | (fecs.event=='DUMP_70_START') ]
    dump_end = fecs[ (fecs.event=='DUMP_END') | (fecs.event=='DUMP_70_END') ]

    dump_start = dump_start.rename(columns={'time': 'dump_start'})
    dump_end = dump_end.rename(columns={'time': 'dump_end'})

    dump_end = dump_end.drop('event', axis=1)
    dumps = pd.merge( dump_start, dump_end, on=['cnt']).reset_index()
    dumps = dumps.drop(['cnt'], axis=1)
    dumps['is_70m'] = dumps.event.apply( lambda ev: True if ev=='DUMP_70_START' else False )
    dumps = dumps.drop(['event'], axis=1)

    return dumps


def next_pass(time=False):
    """Returns the time of the next DUMP_START in the FECS file. If time=False
    the current time is used, otherwise time= is parsed for a date string.

    Times should be specified in UTC, and returned dump times are in CET"""

    import pytz

    # if time is not False, parse it as a date string
    time = parser.parse(time) if time else datetime.utcnow()

    # load the FECS EVF file
    fecs = load_fecs()

    # find first DUMP_START and DUMP_END events after time
    dump_start = fecs[ (fecs.time>time) & ((fecs.event=='DUMP_START') | (fecs.event=='DUMP_70_START') ) ].time.irow(0)
    dump_end = fecs[ (fecs.time>time) & ( (fecs.event=='DUMP_END') | (fecs.event=='DUMP_70_END') ) ].time.irow(0)

    # if we are already in a dump (end<start) find the start time (in the past)
    if dump_end < dump_start:
        dump_start = fecs[ (fecs.time<time) & (fecs.event=='DUMP_START') ].time.irow(-1)

    # Localise the times to UTC
    dump_start = pytz.utc.localize(dump_start)
    dump_end = pytz.utc.localize(dump_end)

    # specify timezone for output
    cet = pytz.timezone('CET')

    dump_start = dump_start.astimezone(cet).to_datetime()
    dump_end = dump_end.astimezone(cet).to_datetime()

    print('Dump start: %s, dump end: %s' % (dump_start, dump_end))

    # return the times localised to CET
    return dump_start, dump_end



def filename_counter(filename, to_stp=True, increment_by=0):
    """Accepts an EVF or ITL filename and increments the PI counter. If
    to_stp=True the ITLS PI MTP index is changed to the PI STP index."""

    # 0        10        20        30
    # 01234567890123456789012345678901234
    # EVF__MD_M006_S015_01_A_RSM0PIM0.evf - example format

    if filename[0:4] != 'ITLS' and filename[0:4] != 'EVF_':
        print('ERROR: filename prefix not ITLS or EVF_')
        return False

    filename = list(filename)
    counter = int(filename[30])
    filename[30] = '%s' % (counter+1+increment_by)

    if to_stp: filename[29] = 'S'

    return "".join(filename)





def orfa_submit(filelist):
    """Accepts a list of files and submits them to ORF-A"""

    import dds_utils

    if type(filelist) != list: filelist = [filelist]

    orfa=dds_utils.sftp()
    orfa.open(sftpURL='ssols01.esac.esa.int',sftpUser='midas',sftpPass='wr_7772_')
    orfa.sftp.chdir('requests_to_RSOC')

    for f in filelist:
        localname = os.path.basename(f)
        orfa.sftp.put(f,localname+'.tmp')
        orfa.sftp.rename(localname+'.tmp',localname)

    orfa.close()

    return


#---------------- UNUSED ---------------------

def schedule(observations):
    """Accepts a set of MTP level observation definitions (type, start, duration, power
    drate etc.) and tries to insert STP level observations into these slots. For EXPOSE
    this is mapped 1:1 unless duration or power constraints are broken. SCAN slots are
    scheduled based on a ranked list of proposed scans read from an external file."""

    procedures = []

    # open the facet status file
    facet_list = read_facetlist(facet_file)

    # update the tip status file
    tip_list = read_tiplist(tip_file)

    # update the scan list file to add cal scans etc. according to the rules
    # also sorts and returns a sorted list of prioritised scans
    scan_list = update_scanlist(scan_file, facet_list, tip_list)

    # run through each observation and insert a proposed operation (exposure
    # or scan)
    for counter in range(len(observations)):
        obs = observations.iloc[counter]

        procedure = {}

        if obs['obs_type'] == 'TARGET_EXPOSE':

            # set up the parameter list for exposure

            # choose a segment to expose - for now take the first pre-scanned
            # sol-gel target

            ready_facets = facet_list[(facet_list['status']=='PRESCANNED') & \
                (facet_list['type']=='SOLGEL')]

            if len(ready_facets) < 1:
                logging.error('no pre-scanned facets to expose!!')
                return None

            facet = ready_facets.index[0]

            # update the status of the segment
            update_facetlist(facet_file, facet, 'EXPOSED')

            # by default scan the central stripe of each facet
            segment = 512 - (facet * 16) # TODO: check numbers

            # check to see if power allows us to remain ON during exposure
            if obs['power'] < exposure_on_pwr:
                procedure['template'] = 'EXPOSE_OFF'
            else:
                procedure['template'] = 'EXPOSE_STBY'

            procedure['params'] = { \
                'target_expose_start': obs['start_id'],
                'target_expose_end' : obs['end_id'],
                'target_expose_count' : obs['obs_id'],
                'seg_num_exp' : segment }

            hours = obs['duration'] / np.timedelta64(1, 'h')
            logging.debug('facet %i scheduled for %3.2f exposure' % (facet, hours))

            # add the parameters to the observation definition
            procedures.append(procedure)

        elif obs['obs_type'] == 'TARGET_SCAN':

            duration = obs['duration'] / np.timedelta64(1,'s')
            power = obs['power']
            drate = obs['drate']

            # calculate the durations of all scans listed in the scan file
            import scan_duration
            for counter in range(len(scan_list)):
                scan = scan_list.iloc[counter]
                scandur = scan_duration.calc_duration(  \
                    scan['xpoints'],  scan['ypoints'],  \
                    scan['ntypes'],   scan['zsettle'],  \
                    scan['xysettle'], scan['zretract'], \
                    scan['zstep'], scan['avg'])

                logging.debug('calculated scan duration: %3.2f h' % (scandur/60./60.))

                if scandur <= duration: # schedule this scan
                    pass

            # Check instrument state between observations and add power ON/OFF
            # if necessary.

        else:

            logging.error('invalid observation type: %s' % obs['obs_type'])

        # return the list of observations, and a dictionary containing the
        # observation type (corresponding to the template to use) and the
        # parameters necessary

    return(observations, procedures)


def get_orfa_seq(seq_type):
    """Connects to the RSGS server and uses the ORFA Sequence Number Dispenser to
    return the sequence number for a given file type."""

    if seq_type not in orfa_seq_types:
        logging.error('invalid ORF-A sequence type %s' % seq_type)
        return None

    # use pycurl to login and query the ORF-A sequence dispenser
    import tempfile, pycurl, StringIO, urllib

    url='http://www.sciops.esa.int/index.php?project=ROSETTA&page=osnd&type=%s&source=PI5&destination=RSO' % (seq_type)
    post_data = dict({
        'user':'mbentley',
        'pass':'Dohpaz4*',
        'submit':'Login' })

    sgscookies = os.path.join(tempfile.gettempdir(),'sgscookies.tmp')

    buf = StringIO.StringIO()
    c = pycurl.Curl()
    c.setopt(pycurl.URL,url)
    c.setopt(pycurl.WRITEFUNCTION, buf.write)
    c.setopt(pycurl.COOKIEFILE, sgscookies)
    c.setopt(pycurl.COOKIEJAR, sgscookies)
    c.setopt(pycurl.POST, 1)
    c.setopt(pycurl.POSTFIELDS, urllib.urlencode(post_data))
    c.setopt(pycurl.VERBOSE, False) # set to true for debugging
    c.perform()

    data = buf.getvalue()

    seq_num = int(data[data.find('.ROS')-12:data.find('.ROS')-7])

    return seq_num



def upload_fs(localfile, directory='fs_commanding', serverfile=None):
    """Upload a file to the MIDAS account on the IWF FTP server. Typically used to transfer
    files to/from the FS EGSE computer"""

    import ftplib

    ftp = ftplib.FTP('ftp.iwf.oeaw.ac.at')
    ftp.login('midas', 'middoc')
    ftp.cwd(directory)
    if serverfile is None:
        filename = os.path.basename(localfile)
    else:
        filename = serverfile

    f = open(localfile, 'r')
    put_result = ftp.storlines('STOR ' + filename, f)
    f.close()

    if debug: print('DEBUG: FTP results (put): %s ' % (put_result))

    return


if __name__ == "__main__":

    # Called interactively - check for a filename as an input

    import os

    if len(sys.argv) != 3:
        sys.exit('ERROR: usage planning <evf file> <itl file>')

    elif len(sys.argv) == 3:
        evf_file = str(sys.argv[1])
        itl_file = str(sys.argv[2])
        if not os.path.isfile(evf_file) and not os.path.isfile(itl_file):
            sys.exit('EVF or ITL file does not exist')

    # set up logging to file
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        datefmt='%m/%d/%Y %I:%M:%S %p',
                        filename=log_file)

    # define a handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # check the EVF and ITLM files refer to the same period, and extract
    # the STP and other info
    #
    # filename template from ORF-A
    # ffff_sordes_txxxxxxxxxxxxxx_vvvvv.ROS
    #
    # where t is always 'D' for data, sor=RSO and des=PI5 for SGS and MIDAS
    # xxxxxxxxxxxxxx is the event period name = CCC_YYY_A_VMM
    # where CCC = LTP/MTP/STP, YYY = planning cycle counter, A = activity, MM = version number
    #
    # so the ITL and EVF files at MTP level have format:
    #
    # ITL__RSOPI5_D_STP_nnn_A_VMM__vvvvv.ROS
    # EVF__RSOPI5_D_STP_nnn_A_VMM__vvvvv.ROS
    #

    itlname = os.path.basename(itl_file)
    evfname = os.path.basename(evf_file)
    if itlname[5:23] != evfname[5:23]:
        logging.error('ITL and EVF file refer to different planning periods')
        sys.exit()

    cycle_type, cycle_number, activity = itlname[14:23].split('_')
    if cycle_type not in planning_cycle_types or activity not in activity_case_types:
        logging.error('invalid planning cycle or activity in filename')
        sys.exit()

    # Now have valid and matching files - read in both
    evf_start, evf_end, event_list = read_evf(evf_file)
    itl_start, itl_end, itl_ver, seq_list = read_itlm(itl_file)

    # combine these into a set of observations, and perform basic checks
    observations = validate_obs(evf_start, evf_end, event_list, itl_start, itl_end, seq_list)

    # try to schedule exposures and scans into these observation slots
    proposed_obs, procedures = schedule(observations)

    # TODO: probably we want a GUI or similar hear to tweak the details if necessary?
    # The final set of observations is passed to the code which fleshes out the templates

    # Generate an ITL file from these templates and parameters
    itl = generate_itl(procedures)

    # Assume sequence numbers of internal files take that of the container
    seq_num = get_orfa_seq('ITLS')

    # Write the final ITL file
    itls_filename = 'ITL__PI5RSO_D_%s_%s_%s_V01__%05d.ROS' % (cycle_type, cycle_number, activity, seq_num)

    itls_file = open(itls_filename, 'a')
    [itls_file.write(fragment) for fragment in itl]
    itls_file.close()
