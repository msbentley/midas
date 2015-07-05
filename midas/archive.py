#!/usr/bin/python
"""
archive.py - archives MIDAS HK TM in HDF5 and allows querying.

Mark S. Bentley (mark@lunartech.org), 2013

"""

debug = True

import os
import numpy as np
import pandas as pd
from bitstring import ConstBitStream, ReadError
from pandas import HDFStore
from midas import common, ros_tm

# Use https://code.google.com/p/python-bitstring/ to read the binary data (see also https://pythonhosted.org/bitstring/)

def get_packet_format(apid=1076, sid=1):
    """Loads the packet and parameters information from the RMIB and returns a BitString
    packet format. Note that global acquisition parameters that have corresponding detailed
    parameters are ignored and only the detailed parameters returned."""

    packet = ros_tm.pid[ (ros_tm.pid.apid==apid) & (ros_tm.pid.sid==sid)].squeeze()

    if len(packet)==0:
        print('ERROR: could not find packet with APID %i and SID %i' % (apid, sid))
        return False, False

    pkt_name = ros_tm.pid[ (ros_tm.pid.apid==apid) & (ros_tm.pid.sid==sid) ].description.squeeze()

    params = ros_tm.plf[ros_tm.plf.spid==packet.spid].sort(['byte_offset','bit_offset'])
    params = params.merge(ros_tm.pcf)
    params.width = params.width.astype(np.int64)

    params = params[ (params.param_name.str.startswith('NMD')) ]

    params['bit_position'] = params.byte_offset * 8 + params.bit_offset

    global_params = params[params.param_name.str.startswith('NMDA')]
    detailed_params = params[params.param_name.str.startswith('NMDD')]

    # Remove any A parameters that have one or more D parameters inside
    global_remove = []
    for idx, param in global_params.iterrows():
        start_bit = param.bit_position
        end_bit = param.bit_position + param.width
        overlapping = detailed_params[ (detailed_params.bit_position >= start_bit) & (detailed_params.bit_position <= end_bit) ]
        if len(overlapping)>0:
            global_remove.append(param.param_name)

    params = params[ ~params.param_name.isin(global_remove) ]

    current_bit = 0 # bit pointer - end of current parameter
    fmt = ''

    for idx, param in params.iterrows():

        start_bit = param.bit_position

        if start_bit > current_bit: # add padding
            fmt += 'pad:%i, ' % (start_bit-current_bit)
        current_bit = start_bit + param.width

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
            print('ERROR: format code not recognised')

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

    print('INFO: packet format for %s created with length %i bits' % (pkt_name, current_bit))

    return pkt_name, fmt, param_names, param_desc, status_len


def search_params(search=''):
    """Performs a case insensitive search of the MIDAS HK parameter descriptions; useful
    if you can't remember the parameter name you want to retrieve from the archive!"""

    params = ros_tm.search_params(search)[ ['param_name','description', 'unit'] ]
    print(params.to_string(index=False))

    return # params




def read_data(files, apid, sid, calibrate=False, tsync=True, use_index=False):
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
        tm.query_index(what='hk')
    else:
        tm = ros_tm.tm(files)

    pkts = tm.pkts[ (tm.pkts.apid==apid) & (tm.pkts.sid==sid) & (tm.pkts.tsync) ]
    tm.pkts.sort('obt', inplace=True, axis=0)

    if tsync:
        pkts = pkts[ pkts.tsync ]

    print('INFO: packet index complete (%i %s packets)' % (len(pkts),pkt_name))

    hk_rows = []
    obt = []
    skipped = 0

    for f in pkts.filename.unique():

        if debug: print('DEBUG: processing file %s' % f)

        # Open each file as a bitstream and filter the packet list to frames from this packet
        s = ConstBitStream(filename=f)
        frames = pkts[pkts.filename==f]
        frames.reset_index(inplace=True)

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
                print('ERROR: %s (skipping packet)' % (e.msg))
                del(obt[idx])
                skipped += 1
                continue
            hk_dict.update(zip(param_names, frame_data))
            hk_rows.append(hk_dict)

    hk_data = pd.DataFrame(hk_rows)
    hk_data['obt'] = obt
    hk_data.set_index('obt', inplace=True)

    print('INFO: data read complete (%i packets skipped)' % (skipped))

    if calibrate:

        # loop through parameters and calibrate each one
        for param in param_names:
            if param in hk_data.columns:
                hk_data['%s' % param] = ros_tm.calibrate('%s' % param,hk_data['%s' % param])

        print('INFO: calibration complete')

    hk_data._metadata = [pkt_name, dict(zip(param_names, param_desc)), status_len]

    return hk_data


def create(files='TLM__MD_M*.DAT', tlm_path=common.tlm_path, archive_path=common.tlm_path, archfile='tm_archive.h5', calibrate=False, use_index=False):
    """Writes a DataFrame of calibrated TM data to an hdf5 archive"""

    # data.to_hdf(hdf5file, pkt_name, mode='w', format='table', data_columns=True)

    import glob

    apid = 1076

    store = HDFStore(os.path.join(archive_path,archfile), 'w') # , complevel=9, complib='blosc')

    if not use_index:
        files = os.path.join(tlm_path, files)
        files = sorted(glob.glob(files))

        for f in files:

            hk = read_data(files=f, apid=apid, sid=1, calibrate=calibrate)
            store.append('HK1', hk, format='table', data_columns=True, min_itemsize=hk._metadata[2], index=False)

            hk = read_data(files=f, apid=apid, sid=2, calibrate=calibrate)
            store.append('HK2', hk, format='table', data_columns=True, min_itemsize=hk._metadata[2], index=False)

    else:

        hk = read_data(files=None, apid=apid, sid=1, calibrate=calibrate, use_index=True)
        store.append('HK1', hk, format='table', data_columns=True, min_itemsize=hk._metadata[2], index=False)

        hk = read_data(files=None, apid=apid, sid=2, calibrate=calibrate, use_index=True)
        store.append('HK2', hk, format='table', data_columns=True, min_itemsize=hk._metadata[2], index=False)

    store.root._v_attrs.calibrated = calibrate

    if debug: print('DEBUG: indexing HDF5 tables')

    store.create_table_index('HK1',columns=['index'],optlevel=9,kind='full')
    store.create_table_index('HK2',columns=['index'],optlevel=9,kind='full')

    store.close()

    return


def append(tlm_files='TLM__MD_M*.DAT', tlm_path=common.tlm_path, archive_path=common.tlm_path, archfile='tm_archive.h5'):
    """Appends data in HK packets contained in tlm_path/tlm_files to archive_path/archfile. Only packets with
    OBTs after the final entry in the archive file will be added! Calibration status is maintained."""

    apid = 1076

    store = HDFStore(os.path.join(archive_path,archfile), 'a')
    calibrated = True if store.root._v_attrs.calibrated else False

    # TODO - getting the max() index is not helpful, since unsync'd OBTs are large!

    hk = read_data(files=os.path.join(tlm_path,tlm_files), apid=apid, sid=1, calibrate=calibrated)
    metadata = hk._metadata
    obt_max = store.select_column('HK1','index').max()
    hk = hk[hk.index>obt_max]
    store.append('HK1', hk, format='table', data_columns=True, min_itemsize=meta[2], index=False)

    hk = read_data(files=os.path.join(tlm_path,tlm_files), apid=apid, sid=2, calibrate=calibrated)
    metadata = hk._metadata
    obt_max = store.select_column('HK2','index').max()
    hk = hk[hk.index>obt_max]
    store.append('HK2', hk, format='table', data_columns=True, min_itemsize=meta[2], index=False)

    store.close()

    return


def ptrepack(in_file, out_file, options):
    """Runs the ptrepack command to re-write an HDF5/PyTables file. The string given in
    options is passed directly to ptrepack with no checking!"""

    # ptrepack --chunkshape=auto --propindexes --complevel=9 --complib=blosc in.h5 out.h5


    return


def query(param, start=None, end=None, archive_path=common.tlm_path, archfile='tm_archive.h5'):
    """Searches archfile for param= between times start= and end="""

    store = pd.HDFStore(os.path.join(archive_path,archfile), 'r')

    hk1_params = store.root.HK1.table.colnames
    hk1_params.remove('index')

    hk2_params = store.root.HK2.table.colnames
    hk2_params.remove('index')

    if param in hk1_params:
        table='HK1'
    elif param in hk2_params:
        table='HK2'
    else:
        print('ERROR: parameter %s not found in the archive' % (param))
        return False

    if (start is None) and (end is None): # read the entire column (fast!)
        data = pd.Series(store.select_column(table,param).values, index=store.select_column(table,'index').values, name=param)
    else:

        obt = store.select_column(table,'index')

        if start is None:
            start = obt.min()

        if end is None:
            end = obt.max()

        where = obt[ (obt>start) & (obt<end) ].index

        # data = store.select(table, where="index>Timestamp(%r) and index<Timestamp(%r)" % (start,end), columns=[param])
        data = store.select(table, where=where, columns=[param])
        data = data['%s' % param]

    # If data is uncalibrated, run the calibration
    if not store.root._v_attrs.calibrated:
        data = pd.Series( ros_tm.calibrate('%s' % param,data.values), index=data.index, name=data.name)

    store.close()

    return data
