#!/usr/bin/python
"""
pds3_utils.py

Mark S. Bentley (mark@lunartech.org), 2017

A module containing a set of PDS3 utilities useful for MIDAS.

"""

import pvl
import glob
import os
import common
import logging
import pandas as pd
log = logging.getLogger(__name__)



def get_datasets(arc_path=common.arc_path, latest=True):
    """Scans PDS3 datasets in the directory given by arc_path= and
    returns the dataset ID and start/stop times for each.
    
    If latest=True then only the latest version of each dataset is
    returned, otherwise all versions are returned."""

    # Build a live list of archive datasets and their start/stop times

    # look at CATALOG/DATASET.CAT for start/stop times
    catalogues = common.locate('DATASET.CAT', arc_path)
    catalogues = [file for file in catalogues]

    dsets = {}

    for cat in catalogues:
        label = pvl.load(cat)
        start = label['DATA_SET']['DATA_SET_INFORMATION']['START_TIME']
        stop = label['DATA_SET']['DATA_SET_INFORMATION']['STOP_TIME']
        dset_id = label['DATA_SET']['DATA_SET_ID']
        vid = float(dset_id.split('-V')[-1])
        lid = dset_id.split('-V')[0] 
        dsets.update( {dset_id: (start, stop, lid, vid)})
        
    dsets = pd.DataFrame(dsets).T
    dsets.columns=['start_time', 'stop_time', 'lid', 'vid']
    dsets.sort_values(['lid', 'vid'], inplace=True)

    if latest:
        dsets.drop_duplicates('lid', keep='last', inplace=True)

    return dsets


def get_products(arc_path=common.arc_path, image_only=True, latest=True):
    """Scans PDS3 datsets in the directory given by arc_path=
    and returns the dataset, product IDs and paths to each product"""

    import pudb

    log.debug('indexing PDS products in root %s' % arc_path)

    cols = ['dataset', 'prod_id', 'start']
    products = pd.DataFrame([], columns=cols)

    dsets = get_datasets(arc_path, latest=latest)
    for idx, dset in dsets.iterrows():
        log.debug('processing dataset %s' % dset)
        dset_root = os.path.join(arc_path, dset.name)
        if image_only:
            labels = glob.glob(os.path.join(dset_root, 'DATA/IMG/*ZS.LBL'))
        else:
            labels = glob.glob(os.path.join(dset_root, 'DATA/*.LBL'))
        for lab in labels:
            label = pvl.load(lab)
            prod_id = label['PRODUCT_ID'].encode()
            start = pd.Timestamp(label['START_TIME'])
            products = products.append( pd.DataFrame([[dset.name, prod_id, start]], columns=cols), ignore_index=True )

    products.sort_values('start', inplace=True)
    log.info('located %d products in %d datasets' % (len(products), len(dsets)))

    return products


def get_tgt_history(arc_path=common.arc_path, latest=True):
    """Scans PDS3 datasets for the (cumulative) scan history. Using the latest dataset
    available the files (one per target) are read into a Pandas DataFrame"""

    dsets = get_datasets(arc_path, latest=latest)
    last_dset = dsets.sort_values('stop_time').iloc[-1].name
    log.debug('Latest dataset ID: %s' % last_dset)
    dset_root = os.path.join(arc_path, last_dset)
    tgh_files = glob.glob(os.path.join(dset_root, 'DATA/TGH*.TAB'))

    columns = ['start_sc', 'start_utc', 'stop_sc', 'stop_utc', 'event', 'tip']
    history_list = []

    for f in tgh_files:
        target = int(os.path.splitext(f)[0].split('_')[-1]) - 1
        history = pd.read_csv(f, skipinitialspace=True, names=columns)
        history['target'] = target
        history_list.append(history)

    history = pd.concat(history_list)
    history.start_utc = pd.to_datetime(history.start_utc)
    history.stop_utc = pd.to_datetime(history.stop_utc)
    history.sort_values(by='start_utc', inplace=True)

    return history

def get_tip_history(arc_path=common.arc_path, latest=True):
    """Scans PDS3 datasets for the (cumulative) cantilever history. Using the latest dataset
    available the files (one per target) are read into a Pandas DataFrame"""

    dsets = get_datasets(arc_path, latest=latest)
    last_dset = dsets.sort_values('stop_time').iloc[-1].name
    log.debug('Latest dataset ID: %s' % last_dset)
    dset_root = os.path.join(arc_path, last_dset)
    tgh_files = glob.glob(os.path.join(dset_root, 'DATA/CAH*.TAB'))

    columns = ['start_sc', 'start_utc', 'stop_sc', 'stop_utc', 'event',
        'ac_gain', 'dc_gain', 'exc_lvl', 'max_amp', 'max_freq', 'scan_type']
    history_list = []

    for f in tgh_files:
        tip = int(os.path.splitext(f)[0].split('_')[-1]) - 1
        history = pd.read_csv(f, skipinitialspace=True, names=columns)
        history['tip'] = tip
        history_list.append(history)

    history = pd.concat(history_list)
    history.start_utc = pd.to_datetime(history.start_utc)
    history.stop_utc = pd.to_datetime(history.stop_utc)
    history.sort_values(by='start_utc', inplace=True)

    return history


def get_events(arc_path=common.arc_path, latest=True):
    """Extracts event information from all datasets found in the root
    folder given by arc_path and returns as a pandas DataFrame"""

    dsets = get_datasets(arc_path, latest=latest)

    columns = ['start_sc', 'start_utc', 'event_cnt', 'event_id', 'event']
    event_list = []

    for idx, dset in dsets.iterrows():
        log.debug('processing dataset %s' % dset)
        dset_root = os.path.join(arc_path, dset.name)
        event_files = glob.glob(os.path.join(dset_root, 'DATA/EVN/EVN_*.TAB'))
        event_list.append(pd.concat((pd.read_csv(f, skipinitialspace=True, names=columns) for f in event_files)))

    events = pd.concat(event_list)
    events.start_utc = pd.to_datetime(events.start_utc)
    events.event.apply( lambda event: event.strip() )
    events.sort_values(by='start_utc', inplace=True)
    log.info('%d events read from %d datsets' % (len(events), len(dsets)))

    return events


def get_fscans(arc_path=common.arc_path):
    """Extracts frequency scan (FSC) data fro mall datasets found in
    the root folder given by arc_path and returns a pandas DataFrame"""

    dsets = get_datasets(arc_path)

    columns = ['start_time', 'start_freq', 'fstep', 'num_cycles', 'tip_num', 'exc_lvl', 'ac_gain']
    fscan_list = []

    for dset in dsets:
        log.debug('processing dataset %s' % dset)
        dset_root = os.path.join(arc_path, dset)
        fscan_files = glob.glob(os.path.join(dset_root, 'DATA/FSC/FSC_*.DAT'))


    # events = pd.concat(event_list)
    # events.start_utc = pd.to_datetime(events.start_utc)
    # events.event.apply( lambda event: event.strip() )
    # events.sort_values(by='start_utc', inplace=True)
    # log.info('%d events read from %d datsets' % (len(events), len(dsets)))

    return events



    return fscans



def scan2arc(scanfile, arc_path=common.arc_path):
    """Accepts a scan file string and attempts to locate the corresponding
    image in the archive. This is done simply by comparing the date to
    a list of dataset start/end times and then grepping the image labels
    to find one in the archive"""

    import ros_tm
    import pvl

    dset_matches = []

    image = ros_tm.load_images(data=False).query('scan_file==@scanfile')
    if len(image)==0:
        log.error('scan file %s not found in the image database' % scanfile)
        return None
    else:
        image = image.iloc[0]

    # img_start = image.start_time.round('100ms')
    log.debug('scan file start time: %s' % image.start_time)

    dsets = get_datasets(arc_path)
    for dset in dsets:
        start, stop = dsets[dset]
        if start < image.start_time < stop:
            dset_matches.append(dset)

    if len(dset_matches)==0:
        log.warning('no archive products found with start time %s' % img_start)
        return None
    elif len(dset_matches) > 1:
        log.warning('more than one datasets match contain the image time')
    else:
        dataset = dset_matches[0]

    image_files = glob.glob(os.path.join(arc_path, dataset, 'DATA/IMG/*ZS.LBL'))

    for f in image_files:
        label = pvl.load(f)
        found = False
        arc_time = pd.Timestamp(label['START_TIME'])
        if (arc_time - pd.Timedelta(seconds=0.5)) < image.start_time < (arc_time + pd.Timedelta(seconds=0.5)):
            found = True
            break

    if not found:
        log.error('no matching archive product found')
        return None, None

    arcfile = os.path.basename(f).split('.')[0]

    log.debug('scan file %s is located in the %s dataset as product %s' % (scanfile, dataset, arcfile))

    return dataset, arcfile


def arc2scan(arcfile, arc_path=common.arc_path):
    """Accepts am archive file product and attempts to locate the corresponding
    image in the MIDAS database. This is done simply by comparing the date to
    a list of dataset start/end times and then grepping the image labels
    to find one in the archive"""

    import ros_tm
    import pvl
    import glob

    arcfile += '.LBL'
    arcfiles = common.locate(arcfile, arc_path)
    arcfiles = [file for file in arcfiles]

    if len(arcfiles) > 1:
        log.warning('more than one archive product matching %s' % arcfile)

    arcfile = arcfiles[0]
    label = pvl.load(arcfile)
    start = pd.Timestamp(label['START_TIME']).round('100ms')
    log.debug('start time in archive product: %s' % start)

    images = ros_tm.load_images(data=False)
    # images['start_time_s'] = images.start_time.apply(lambda x: x.round('100ms'))
    delta = pd.Timedelta(seconds=0.5)
    image = images[ ((images.start_time-delta) < start) & ((images.start_time+delta) > start )]
    if len(image)==0:
        log.error('no matching image found with start time %s' % start)
        return None
    else:
        image = image.squeeze()

    scanfile = image.scan_file

    return scanfile
