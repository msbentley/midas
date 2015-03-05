#!/usr/bin/python
"""
eps_utils.py - a collection of utilities for dealing withe EPS inputs, for
example ITL timeline files, and outputs (e.g. power and data profiles)."""

debug = True
import numpy as np
import os
from midas import common, planning

dateformat = '%d-%B-%Y_%H:%M:%S'

def parse_itl(filename, header=True):

    # Uses PyParsing http://pyparsing.wikispaces.com
    import pyparsing as pp

    from datetime import timedelta

    pp.ParserElement.setDefaultWhitespaceChars(" \t\n")

    # Basic type definitions
    num = pp.Word(pp.nums+".")
    word = pp.Word(pp.alphas)
    alphanum = pp.Word(pp.alphanums)
    alphanumunder = pp.Word(pp.alphanums + '_')

    # Define comment and line continuation characters (to ignore)
    hashmark = pp.Literal('#')
    comment = hashmark + pp.restOfLine
    continuation = pp.Literal('\\')

    # Define the header (sequence of key:value pairs)
    headerKey = pp.Word(pp.alphanums + '_')
    headerVal = pp.Word(pp.alphanums + '-._:')
    header_entry = pp.Group(headerKey('key') + pp.Suppress(":") + headerVal('val'))

    header = pp.Group(pp.OneOrMore(header_entry)).setResultsName('header')

    # Define the timeline, a series of TC SQs with corresponding timing, mode etc.

    # Event label
    eventLabel = pp.Word(pp.alphanums + '_-')

    # Event options, commonly the event count - optional
    eventOption = alphanum('key') + pp.Suppress("=") + alphanum('val')
    eventOptions = pp.Suppress('(') + pp.OneOrMore(eventOption) + pp.Suppress(')')

    # Relative time (positive or negative)
    plusminus = pp.oneOf('+ -')
    relTime = pp.Combine(pp.Optional(plusminus) + pp.Optional( num('days') + ('_')) + pp.Combine( num + ":" + num + ":" + num )('hhmmss'))

    def validateRelTime(tokens):
        try:
            days = int(tokens[0].days) if 'days' in tokens[0] else 0
            hours, minutes, seconds = tokens[0].hhmmss.split(':')
            negative=True if tokens[0][0] == '-' else False
            hours, minutes, seconds = map(int, (hours, minutes, seconds))
            negator = -1 if negative else 1
            return timedelta( 0, negator * (days*24*60*60 + hours * 60*60 + minutes * 60 + seconds))
            # time.strptime(tokens[0], "%m/%d/%Y")

        except ValueError:
            raise pp.ParseException("Invalid date string (%s)" % tokens[0])

    relTime.setParseAction(validateRelTime)

    # Instrument name
    instrument = word

    # Instrument mode
    mode = (word ^ '*')

    # TC sequence and parameters

    paramName = alphanum
    paramValType = pp.Word(pp.alphanums + '.-_*')
    paramUnitType = pp.Combine(pp.Suppress("[") + pp.Word(pp.alphas+'/%') + pp.Suppress("]"))
    paramVal = (pp.quotedString.setParseAction(pp.removeQuotes) ^ paramValType)
    seqParam = pp.Group(paramName('name')  + pp.Suppress('=') + paramVal('value') + pp.Optional(paramUnitType('unit')))
    seqParams = pp.ZeroOrMore(seqParam)

    # Optional Z record data
    zrec_param = pp.CaselessLiteral('POWER_PROFILE') ^ pp.CaselessLiteral('DATA_RATE_PROFILE')
    zrec = pp.OneOrMore(pp.Group(zrec_param('name') + pp.Suppress('=') + relTime('time') + num('value') + pp.Optional(paramUnitType)('unit')))

    sequence = alphanumunder('name') + pp.Optional(pp.Suppress('(') + seqParams('params') + pp.Optional(zrec('zrec')) + pp.Suppress(')'))

    timeline_entry = pp.Group(eventLabel('label') + pp.Optional(eventOptions('options')) + relTime('time') + \
        instrument('instrument') + mode('mode') + sequence('sequence'))

    timeline = pp.Group(pp.OneOrMore(timeline_entry)).setResultsName('timeline')

    # M.S.Bentley 28/08/2014 - not all ITL files (especially those used by MAPPS) have a header, so making
    # it an option (set by header=True)
    if header:
        itlparse = header + timeline + pp.StringEnd()
    else:
        itlparse = timeline + pp.StringEnd()
    itlparse.ignore(comment)
    itlparse.ignore(continuation)

    itl = itlparse.parseString(open(filename,'U').read())

    print('INFO: Timeline file %s read successfully with %i entries' % (os.path.basename(filename), len(itl.timeline)))

    return itl


def run_eps(itl_file, evf_file, ng=False, ros_sgs=False, por=False, mtp=False, case=False, outputdir='.'):
    """Spawn an external process to run and EPS and validate a given EPS and ITL file"""

    import subprocess

    if ros_sgs:
        os.environ["EPS_DATA"] = os.path.join(common.ros_sgs_path, 'CONFIG')
        os.environ["EPS_CFG_DATA"] = os.path.join(common.ros_sgs_path, 'CONFIG')

    if ng:

        obspath = os.path.join(common.ros_sgs_path, 'PLANNING/OBS_LIB')

        # Config files now have format: ROS_SGS\CONFIG\ros_eps_MTP011P.cfg
        ng_exec = os.path.join(os.path.expanduser('~/Dropbox/bin'), 'epsng')
        # command_string = [ng_exec, 'exec', '-disable-plugins', '-t', '1', '-s', '3600', '-c', 'ros_eps_MTP%03i%c.cfg' % (mtp, case.upper()), '-i', itl_file, '-e', 'rosetta.edf', '-ei', evf_file, '-ed', outputdir]
        command_string = [ng_exec, 'exec', '-t', '1', '-s', '3600', '-c', 'ros_eps_MTP%03i%c.cfg' % (mtp, case.upper()), '-i', itl_file, '-e', 'rosetta.edf', '-ei', evf_file, '-ed', outputdir, '-tt', 'abs',
            '-obs', obspath, 'DEF_ROS_TOP___________V001.def' ]
    else:
        command_string = ['eps', 'exec', '-i', itl_file, '-e', 'midas_stand_alone.edf', '-ei', evf_file]

    if por:
        command_string.extend( ['-f', 'por', '-o', por])

    if debug: print('DEBUG: EPS running as: \n\n %s' % " ".join(command_string))

    try:
        epscmd = subprocess.check_output(command_string, shell=False, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError, e:
        print "ERROR: EPS run failed! Output: \n\n", e.output
        return False

    print('INFO: EPS run successful. Output: \n\n %s' % epscmd)

    return True



def run_mtp(mtp, case, outfolder=None):
    """Runs the EPS-NG on MTP level products in the ROS_SGS repository"""

    import glob

    mtp_folder = planning.locate_mtp(mtp, case)

    if outfolder is None:
        local_folder = os.path.expanduser('~/Dropbox/work/midas/operations/MTP%03i%c/eps/' % (mtp,case.upper()))
    else:
        local_folder = outfolder

    if not os.path.exists(local_folder):
        os.makedirs(local_folder)

    # Find the MTp level scenario file:
    # SCEN_MTP014P_01_01_________PTRM.ini
    scen=glob.glob(os.path.join(mtp_folder,'SCENARIOS','SCEN_MTP%03i%c_01_01_________PTRM.ini' % (mtp, case.upper())))
    if len(scen)>1:
        print('ERROR: more than one scenario file present!')
        return None

    # Open the scenario file and find the eventInputFile (EVF) file entry
    evf = False
    f=open(scen[0],'r')
    data=f.readlines()
    for item in data:
        if item.split('=')[0]=='MTP_VER_A\InputData\TimelineEvents\eventInputFile':
            evf = os.path.join(mtp_folder,item.split('=')[1])
            break
    f.close()

    print('DEBUG: EVF file: %s' % evf)
    if not evf:
        print('ERROR: eventInputFile entry not found in scenario file %s' % scen)

    # Find the latest TLIS*.evf and TLIS*.itl files
    # e.g. TLIS_PL_M009______01_A_OPS0001A.evf
    itl=glob.glob(os.path.join(mtp_folder,'TLIS_PL_M%03i______01_%c_OPS?????.itl' % (mtp, case.upper())))
    # evf=glob.glob(os.path.join(mtp_folder,'TLIS_PL_M%03i______01_%c_OPS?????.evf' % (mtp, case.upper())))

    if len(itl)!=1: # or (len(evf)>1 or len(evf)==0):
        print('ERROR: cannot find latest TLIS ITL file')
        return None

    os.chdir(mtp_folder)
    # files = os.listdir(mtp_folder)

    status = run_eps(itl[0], evf, ng=True, ros_sgs=True, mtp=mtp, case=case, outputdir=local_folder)

    if not status: return False

    return status



def read_output(directory='.', ng=False):
    """Reads the power and data EPS output files."""

    import pandas as pd
    from dateutil import parser
    from datetime import timedelta

    powerfile = os.path.join(directory,'power_avg.out')
    if not os.path.isfile(powerfile):
        print('ERROR: power_avg.out file not found in directory %s' % (directory))
        return False

    datafile = os.path.join(directory,'data_rate_avg.out')
    if not os.path.isfile(datafile):
        print('ERROR: data_rate_avg.out file not found in directory %s' % (directory))
        return False

    abs_date = True

    f = open(powerfile)
    for idx,line in enumerate(f):
        if (line[0] == '#') or (line[0]=='\n'): continue
        if line.split(':')[0] == 'Ref_date':
            ref_date = line.split(':')[1].strip()
            abs_date = False
            break
    f.close()

    if ng:
        cols = ['time', 'Total', 'ROSETTA', 'ANTENNA', 'ALICE', 'CONSERT', 'COSIMA', \
            'GIADA', 'LANDER', 'MIDAS', 'MIRO', 'NAVCAM', 'OSIRIS', 'ROSINA', 'RPC', 'SREM', \
            'RSI', 'VIRTIS', 'SSMM', 'SGS']

        power = pd.read_table(powerfile,header=None,skiprows=27,names=cols,usecols=[0,9],skipinitialspace=True, delimiter=' ')

    else:
        cols = ['time', 'Total', 'SSMM', 'HK_SSMM', 'HGA', 'MIDAS', 'ALICE', 'MIRO', \
            'OSIRIS', 'COSIMA', 'GIADA', 'VIRTIS', 'CONSERT', 'LANDER', 'ROSINA', 'SREM', 'RPC', 'NAVCAM' ]

        power = pd.read_table(powerfile,header=None,skiprows=27,names=cols,usecols=[0,5],skipinitialspace=True, delimiter=' ')

    if not abs_date:

        ref_date = parser.parse(ref_date)

        # convert the reltime into a python timedelta, add the reference time and set as the index
        power = power.set_index( power['time'].apply(lambda time : ref_date + timedelta( \
            days=int(time.split('_')[0]),
            hours=int(time.split('_')[1].split(':')[0]),
            minutes=int(time.split('_')[1].split(':')[1]),
            seconds=int(time.split('_')[1].split(':')[2]))) )

    else: # have date times in format 14-Jan-2015_00:25:00

        power = power.set_index( power['time'].apply(lambda time : parser.parse(" ".join(time.split('_')))) )

    # also load the data file output

    if ng:
        cols = ['time', 'upload', 'mem_accum', 'ssmm', 'ground']
        data = pd.read_table(datafile,header=None,skiprows=28,skipinitialspace=True, delimiter=' ',names=cols,usecols=[0,27,30,75,76])
    else:
        cols = ['time','memory','accum','upload','download']
        data = pd.read_table(datafile,header=None,skiprows=28,skipinitialspace=True, delimiter=' ',names=cols,usecols=[0,13,14,33,34])

    data.index = power.index

    return power, data


def plot_eps_output(power,data,observations=False):
    """Plot the EPS output (average power and data). If an MTP observation set is given, also
    plot the envelope resources"""

    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.set_title('Power and data for MIDAS in the EPS')
    power=power.replace(0.0,np.nan)
    power_line = ax.plot(power.index,power.MIDAS,'r',label='Average power (W)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Average power (W)')
    ax_right = ax.twinx()
    data_line = ax_right.plot(data.index,data.memory,'b',label='Memory (Mb)')
    ax_right.set_ylabel('Memory (Mb)')
    lines = power_line + data_line
    labs = [l.get_label() for l in lines]
    ax.legend(lines, labs, loc=0)
    ax.grid(True)
    fig.autofmt_xdate()
    ax.fmt_xdata = mdates.DateFormatter('%Y-%m-%d')

    if type(observations) != bool:

        observations = observations.sort('start_time')

        # power/data constant between the start and end times
        start = observations['start_time'].astype(object).values
        end = observations['end_time'].astype(object).values
        times = np.concatenate((start, end))

        pwr = observations['power'].values
        mtp_power = np.concatenate((pwr, pwr))

        dr = observations['drate'].values
        mtp_data = np.concatenate((dr, dr))
        # order both arrays by time so that the plot line is connected correctly
        order = times.argsort()
        times = times[order]
        mtp_power = mtp_power[order]
        mtp_data = mtp_data[order]

        # plot power and data rates on separate Y axes
        ax.plot(times,mtp_power,color='b',label='MTP envelope power')
        ax_right.plot(times,mtp_data,color='r',label='MTP envelope data')
