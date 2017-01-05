#!/usr/bin/python
"""
archive.py - archives MIDAS HK TM in HDF5 and allows querying.

Mark S. Bentley (mark@lunartech.org), 2013

"""

import os
import numpy as np
import pandas as pd
from bitstring import ConstBitStream, ReadError
from pandas import HDFStore
from midas import common, ros_tm

archive_path = common.tlm_path

import logging
log = logging.getLogger(__name__)

# Use https://code.google.com/p/python-bitstring/ to read the binary data (see also https://pythonhosted.org/bitstring/)

def get_packet_format(apid=1076, sid=1):
    """Loads the packet and parameters information from the RMIB and returns a BitString
    packet format. Note that global acquisition parameters that have corresponding detailed
    parameters are ignored and only the detailed parameters returned."""

    packet = ros_tm.pid[ (ros_tm.pid.apid==apid) & (ros_tm.pid.sid==sid)].squeeze()

    if len(packet)==0:
        log.error('could not find packet with APID %i and SID %i' % (apid, sid))
        return False, False

    pkt_name = ros_tm.pid[ (ros_tm.pid.apid==apid) & (ros_tm.pid.sid==sid) ].description.squeeze()

    params = ros_tm.plf[ros_tm.plf.spid==packet.spid].sort_values(by=['byte_offset','bit_offset'])
    params = params.merge(ros_tm.pcf)
    params.width = params.width.astype(np.int64)

    params = params[ (params.param_name.str.startswith('NMD')) ]

    # fix an apparently database error:
    #  NMDD3193  2067          456           3
    params.bit_offset.loc[ params[ params.param_name=="NMDD3193"].index ] = 0

    params['bit_position'] = params.byte_offset * 8 + params.bit_offset

    global_params = params[params.param_name.str.startswith('NMDA')]
    detailed_params = params[params.param_name.str.startswith('NMDD')]

    # remove detailed parameters corresponding to wax actuators since global_params
    # parameter was re-purposed. Also NMDD3193 which seems to be badly defined in the RMIB?
    detailed_params = detailed_params[~detailed_params.param_name.isin(["NMDD2142", "NMDD2141", "NMDD2140"])]

    # Remove any A parameters that have one or more D parameters inside
    global_remove = []
    for idx, param in global_params.iterrows():
        start_bit = param.bit_position
        end_bit = param.bit_position + param.width
        overlapping = detailed_params[ (detailed_params.bit_position >= start_bit) & (detailed_params.bit_position < end_bit) ]
        if len(overlapping)>0:
            log.debug('removing global parameter %s' % param.param_name)
            log.debug('overlaps with detailed parameters: %s' % " ".join(overlapping.param_name.tolist()))
            global_remove.append(param.param_name)

    params = params[ ~params.param_name.isin(global_remove) ]

    current_bit = 0 # bit pointer - end of current parameter
    fmt = ''

    for idx, param in params.iterrows():

        start_bit = param.bit_position

        if start_bit > current_bit: # add padding
            fmt += 'pad:%i, ' % (start_bit-current_bit)
            log.debug('adding pad of %d bits' % (start_bit-current_bit))
            current_bit += (start_bit-current_bit)

        current_bit += param.width

        log.debug('parameter %s at bit position %d with width %d (after: %d)' % (param.param_name, param.bit_position, param.width, param.bit_position+param.width))
        log.debug('current bit counter: %d' % current_bit)

        # Determine type and hence correct format code
        if (param.ptc == 1) or (param.width==1):
            fmt += 'bool, '
        elif (param.ptc == 2) or (param.ptc==3):
            fmt += 'uint:%i, ' % param.width
        elif param.ptc == 4:
            fmt += 'int:%i, ' % param.width
        elif param.ptc == 5:
            if (param.pfc == 1) or (param.pfc == 2):
                fmt += 'float:%i, ' % param.width
            elif param.pfc == 3:
                fmt += 'uint:%i, ' % param.width  # remember this needs further processing!
        else:
            log.error('format code not recognised')

    fmt = fmt[:-2] # strip the trailing ', '
    param_names = params.param_name.tolist()
    param_desc = params.description.tolist()

    # Find the maximum length of string parameters in the dbase to initialise
    # the HDFStore (pass a dictionary of column:length values)
    status_len = []
    status_params = [(param.param_name, int(param.cal_id)) for idx, param in params.iterrows() if param.cal_cat == 'S']
    for param in status_params:
        status_len.append(len(max(ros_tm.txp[ros_tm.txp.txp_id==param[1]].alt.tolist(), key=len)))
    status_params = [param[0] for param in status_params]
    status_len = dictionary = dict(zip(status_params, status_len))

    log.info('packet format for %s created with length %i bits' % (pkt_name, current_bit))

    return pkt_name, fmt, param_names, param_desc, status_len


def search_params(search=''):
    """Performs a case insensitive search of the MIDAS HK parameter descriptions; useful
    if you can't remember the parameter name you want to retrieve from the archive!"""

    params = ros_tm.search_params(search)[ ['param_name','description', 'unit'] ]
    print(params.to_string(index=False))

    return params




def read_data(files, apid, sid, calibrate=False, tsync=True, use_index=False, on_disk=False, rows=None):
    """Read in data for a given APID and SID and return a dataframe of calibrated
    data for all matching frames. This can then be written to an archive.

    If calibrate=False calibration is performed on query, otherwise calibration is performed
    when writing the file.

    If tsync=True packets with no time synch are ignored.

    If use_index=True the telemetry index file is used instead of indexing each TLM file."""

    # Look up the packet format in the database
    pkt_name, fmt, param_names, param_desc, status_len = get_packet_format(apid=apid, sid=sid)
    if not fmt: return False

    # Index TM files and filter by APID and SID
    if use_index:
        tm = ros_tm.tm()
        if rows is not None:
            tm.query_index(rows=rows)
        else:
            tm.query_index(what='hk')
    else:
        tm = ros_tm.tm(files)

    pkts = tm.pkts[ (tm.pkts.apid==apid) & (tm.pkts.sid==sid) & (tm.pkts.tsync) ]
    tm.pkts.sort_values(by='obt', inplace=True, axis=0)

    if tsync:
        pkts = pkts[ pkts.tsync ]

    log.info('packet index complete (%i %s packets)' % (len(pkts),pkt_name))

    hk_rows = []
    obt = []
    skipped = 0

    for f in pkts.filename.unique():

        log.debug('processing file %s' % f)

        # Open each file as a bitstream and filter the packet list to frames from this packet
        if on_disk:
            s = ConstBitStream(filename=f)
        else:
            s = ConstBitStream(bytes=open(f, 'rb').read(), length=os.path.getsize(f)*8)

        frames = pkts[pkts.filename==f]
        frames.reset_index(inplace=True, drop=True)

        # Create a list of the on-board-times for these packets
        obt.extend(frames.obt.tolist())

        # Loop through each packet in the index and unpack the data using the frame
        # description, appending to a list of dictionaries
        for idx, frame in frames.iterrows():

            hk_dict = {}

            s.pos = frame.offset*8
            try:
                frame_data = s.readlist(fmt)
            except ReadError as e:
                log.error('%s (skipping packet)' % (e.msg))
                del(obt[idx])
                skipped += 1
                continue
            hk_dict.update(zip(param_names, frame_data))
            hk_rows.append(hk_dict)

    hk_data = pd.DataFrame(hk_rows)
    hk_data['obt'] = obt
    hk_data.set_index('obt', inplace=True)

    log.info('data read complete (%i packets skipped)' % (skipped))

    if calibrate:

        log.debug('starting parameter calibration')
        # loop through parameters and calibrate each one
        for param in param_names:
            if param in hk_data.columns:
                hk_data['%s' % param] = ros_tm.calibrate('%s' % param,hk_data['%s' % param])

        log.info('calibration complete')

    hk_data._metadata = [pkt_name, dict(zip(param_names, param_desc)), status_len]

    return hk_data


def create(files='TLM__MD_M*.DAT', tlm_path=common.tlm_path, archive_path=common.tlm_path, archfile='tm_archive_raw.h5',
        calibrate=False, use_index=False, on_disk=False, chunksize=None, pack=True):
    """Writes a DataFrame of (optionally) calibrated TM data to an hdf5 archive.

    TM files (and optionally a TM index) are read from tlm_path, matching files
    Archive files are written to archive_path with the filename archfile
    if calibrate=True, data are calibrated first (slow/larger file, but cross platform)
    if use_index=True TLM files are not read individually but an existing packet index is loaded
    chunksize= specifies how many HK packets are processed in one chunk when use_index=True,
    if None then all data are read."""

    import glob

    apid = 1076

    store = HDFStore(os.path.join(archive_path,archfile), 'w', complevel=9, complib='blosc')

    if not use_index:
        files = os.path.join(tlm_path, files)
        files = sorted(glob.glob(files))

        for f in files:

            hk = read_data(files=f, apid=apid, sid=1, calibrate=calibrate, on_disk=on_disk)
            store.append('HK1', hk, format='table', data_columns=True, min_itemsize=hk._metadata[2], index=False)

            hk = read_data(files=f, apid=apid, sid=2, calibrate=calibrate, on_disk=on_disk)
            store.append('HK2', hk, format='table', data_columns=True, min_itemsize=hk._metadata[2], index=False)

    else:

        if chunksize is None:

            hk = read_data(files=None, apid=apid, sid=1, calibrate=calibrate, use_index=True, on_disk=on_disk)
            store.append('HK1', hk, format='table', data_columns=True, min_itemsize=hk._metadata[2], index=False)

            hk = read_data(files=None, apid=apid, sid=2, calibrate=calibrate, use_index=True, on_disk=on_disk)
            store.append('HK2', hk, format='table', data_columns=True, min_itemsize=hk._metadata[2], index=False)

        else:

            pkt_store = pd.HDFStore(os.path.join(common.tlm_path, 'tlm_packet_index.hd5'), 'r')
            table = 'pkts'
            apid = 1076
            col = pkt_store.select_column(table,'apid')
            nrows = pkt_store.get_storer('pkts').nrows
            pkt_store.close()
            selected = set(np.arange(nrows))
            selected = list(selected.intersection( col[ col==apid ].index ))
            # store.select(table, where=list(selected))
            num_hk = len(selected)

            for i in xrange(num_hk//chunksize + 1):

                start=i*chunksize
                stop=(i+1)*chunksize
                idx = selected[start:stop]

                log.info('processing chunk %i of %i' % (i+1, num_hk//chunksize))

                hk = read_data(files=None, apid=apid, sid=1, calibrate=calibrate, use_index=True, on_disk=on_disk, rows=idx)
                store.append('HK1', hk, format='table', data_columns=True, min_itemsize=hk._metadata[2], index=False)

                hk = read_data(files=None, apid=apid, sid=2, calibrate=calibrate, use_index=True, on_disk=on_disk, rows=idx)
                store.append('HK2', hk, format='table', data_columns=True, min_itemsize=hk._metadata[2], index=False)

    store.root._v_attrs.calibrated = calibrate

    log.debug('indexing HDF5 tables')

    store.create_table_index('HK1', columns=['index'], optlevel=9, kind='full')
    store.create_table_index('HK2', columns=['index'], optlevel=9, kind='full')

    store.close()

    if pack:
        ptrepack(in_file=archfile, archive_path=archive_path)

    return


def append(tlm_files='TLM__MD_M*.DAT', tlm_path=common.tlm_path, archive_path=common.tlm_path, archfile='tm_archive.h5', repack=True, on_disk=False):
    """Appends data in HK packets contained in tlm_path/tlm_files to archive_path/archfile. Only packets with
    OBTs after the final entry in the archive file will be added! Calibration status is maintained."""

    import shutil

    apid = 1076

    store = HDFStore(os.path.join(archive_path,archfile), 'a')
    calibrated = True if store.root._v_attrs.calibrated else False

    hk = read_data(files=os.path.join(tlm_path,tlm_files), apid=apid, sid=1, calibrate=calibrated, on_disk=on_disk)
    metadata = hk._metadata
    obt_max = store.select_column('HK1','index').max()
    hk = hk[hk.index>obt_max]
    log.info('adding %i new HK1 frames to the archive' % len(hk))
    store.append('HK1', hk, format='table', data_columns=True, min_itemsize=metadata[2], index=False, on_disk=on_disk)

    hk = read_data(files=os.path.join(tlm_path,tlm_files), apid=apid, sid=2, calibrate=calibrated)
    metadata = hk._metadata
    obt_max = store.select_column('HK2','index').max()
    hk = hk[hk.index>obt_max]
    log.info('adding %i new HK2 frames to the archive' % len(hk))
    store.append('HK2', hk, format='table', data_columns=True, min_itemsize=metadata[2], index=False)

    store.create_table_index('HK1',columns=['index'],optlevel=9,kind='full')
    store.create_table_index('HK2',columns=['index'],optlevel=9,kind='full')

    store.close()

    if repack:

        # Copy to a temporary file and run ptrepack
        archfile = os.path.join(archive_path, archfile)
        archfile_temp = os.path.join(archive_path, archfile.split('.')[-2]+'_raw.'+archfile.split('.')[-1])

        shutil.copyfile(archfile, archfile_temp)
        ptrepack(in_file=archfile_temp, out_file=archfile)
        os.remove(archfile_temp)

    return


def ptrepack(in_file='tm_archive_raw.h5', out_file='tm_archive.h5', archive_path=common.tlm_path, options='--chunkshape=auto --propindexes --complevel=9 --complib=blosc'):
    """Runs the ptrepack command to re-write an HDF5/PyTables file. The string given in
    options is passed directly to ptrepack with no checking!"""

    # ptrepack --chunkshape=auto --propindexes --complevel=9 --complib=blosc in.h5 out.h5


    import subprocess

    in_file = os.path.join(archive_path, in_file)
    out_file = os.path.join(archive_path, out_file)

    if os.path.isfile(out_file):
        os.remove(out_file)

    command_string = ['ptrepack']
    command_string.extend(options.split())
    command_string.extend([in_file, out_file])

    try:
        ptrepack_cmd = subprocess.check_output(command_string, shell=False, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError, e:
        print "ERROR: ptrepack failed! Output: \n\n", e.output
        return False

    # log.info('ptrepack output:\n\n %s' % ptrepack_cmd)

    return ptrepack_cmd


def query(params, start=None, end=None, archive_path=common.tlm_path, archfile='tm_archive.h5'):
    """Searches archfile for param= between times start= and end="""

    if type(params) != list:
        params = [params]

    store = pd.HDFStore(os.path.join(archive_path,archfile), 'r')

    # Extract the available parameter (column) names
    hk1_params = store.root.HK1.table.colpathnames
    hk1_params.remove('index')

    hk2_params = store.root.HK2.table.colpathnames
    hk2_params.remove('index')

    if not set(params).issubset(hk1_params + hk2_params):
        log.error('one or more parameters not found in the archive')
        return None

    # Work out which of the requested parameters is in HK1 and which in HK2
    # Extract indices (different for both tables!), then data
    hk1_list = [param for param in params if param in hk1_params]
    if len(hk1_list) > 0:
        hk1_index = store.select_column('HK1','index').values
        hk1_data = pd.DataFrame(columns=hk1_list, index=hk1_index)

    hk2_list = [param for param in params if param in hk2_params]
    if len(hk2_list) > 0:
        hk2_index = store.select_column('HK2','index').values
        hk2_data = pd.DataFrame(columns=hk2_list, index=hk2_index)

    # Extract data from the HK tables and calibrate if necessary
    for param in params:

        if param in hk1_params:
            hk1_data[param] = pd.Series(store.select_column('HK1',param).values, index=hk1_index)
            if not store.root._v_attrs.calibrated:
                hk1_data[param] = ros_tm.calibrate('%s' % param, hk1_data[param].values)

        if param in hk2_params:
            hk2_data[param] = pd.Series(store.select_column('HK2',param).values, index=hk2_index)
            if not store.root._v_attrs.calibrated:
                hk2_data[param] = ros_tm.calibrate('%s' % param, hk2_data[param].values)

    store.close()

    # If parameters are from both HK1 and HK2, merge and sort by timestamp
    # Also filter by start/end time here (quicker to load entire dataset and
    # reduce here, but if memory becomes a problem can do during HDF read)
    if len(hk1_list)==0:
        if start is None: start = hk2_data.index.min()
        if end is None: end = hk2_data.index.max()
        return hk2_data[ (hk2_data.index>=start) & (hk2_data.index<=end)]
    elif len(hk2_list)==0:
        if start is None: start = hk1_data.index.min()
        if end is None: end = hk1_data.index.max()
        return hk1_data[ (hk1_data.index>=start) & (hk1_data.index<=end)]
    else:
        if start is None:
            start = min(hk1_data.index.min(), hk2_data.index.min())
        if end is None:
            end = max(hk1_data.index.max(), hk2_data.index.max())

        data = pd.concat([hk1_data, hk2_data]).sort_index()
        return data[ (data.index>=start) & (data.index<=end) ]


def plot(param, start=None, end=None, max_pts=10000, **kwargs):

    import matplotlib.pyplot as plt
    import matplotlib.dates as md

    # Resample code cribbed from http://matplotlib.org/examples/event_handling/resample.html
    class DataDisplayDownsampler(object):
        def __init__(self, xdata, ydata):
            self.origYData = ydata
            self.origXData = xdata
            self.numpts = max_pts
            self.delta = xdata[-1] - xdata[0]

        def resample(self, xstart, xend):
            if type(xstart)==np.float64: xstart = md.num2date(xstart)
            if type(xend)==np.float64: xend = md.num2date(xend)

            mask = (self.origXData > xstart) & (self.origXData < xend)
            xdata = self.origXData[mask]
            ratio = int(xdata.size / self.numpts) + 1
            xdata = xdata[::ratio]

            ydata = self.origYData[mask]
            ydata = ydata[::ratio]

            return xdata, ydata

        def update(self, ax):
            lims = ax.viewLim
            if np.abs(  md.num2date(lims.x1)-md.num2date(lims.x0) - self.delta) > pd.Timedelta(seconds=1):
                self.delta = md.num2date(lims.x1)-md.num2date(lims.x0)
                xstart, xend = lims.intervalx
                self.line.set_data(*self.resample(xstart, xend))
                ax.figure.canvas.draw_idle()

    data = query(param, start=start, end=end)
    if data is None:
        return None

    xdata = data.index
    ydata = data

    d = DataDisplayDownsampler(xdata, ydata)

    fig, ax = plt.subplots()
    xdata, ydata = d.resample(xdata[0], xdata[-1])

    fig.autofmt_xdate()
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    ax.xaxis_date()

    ax.set_xlabel('On-board time')

    param_name = ros_tm.pcf[ros_tm.pcf.param_name==param].description.squeeze()
    ax.set_ylabel(param_name)

    d.line, = ax.plot(xdata, ydata, **kwargs)
    ax.set_autoscale_on(False)  # Otherwise, infinite loop
    ax.callbacks.connect('xlim_changed', d.update)

    fig.tight_layout()

    plt.show()
