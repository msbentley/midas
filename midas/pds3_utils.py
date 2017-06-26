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



def get_datasets(arc_path=common.arc_path):
    """Scans PDS3 datasets in the directory given by arc_path= and
    returns the dataset ID and start/stop times for each"""

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
        dsets.update( {dset_id: (start, stop)} )

    return dsets


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

    img_start = image.start_time.round('s')
    log.debug('scan file start time: %s' % img_start)

    dsets = get_datasets(arc_path)
    for dset in dsets:
        start, stop = dsets[dset]
        if start < img_start < stop:
            dset_matches.append(dset)

    if len(dset_matches) > 1:
        log.warning('more than one datasets match contain the image time')
    else:
        dataset = dset_matches[0]

    image_files = glob.glob(os.path.join(arc_path, dataset, 'DATA/IMG/*ZS.LBL'))

    for f in image_files:
        label = pvl.load(f)
        found = False
        arc_time = pd.Timestamp(label['START_TIME']).round('s')
        if arc_time == img_start:
            found = True
            break

    if not found:
        log.error('no matching archive product found')
        return None, None

    arcfile = os.path.basename(f).split('.')[0]

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
    start = pd.Timestamp(label['START_TIME']).round('s')
    log.debug('start time in archive product: %s' % start)

    images = ros_tm.load_images(data=False)
    images['start_time_s'] = images.start_time.apply(lambda x: x.round('s'))

    image = images.query('start_time_s==@start')
    if len(image)==0:
        log.error('no matching image found with start time %s' % start)
        return None
    else:
        image = image.squeeze()

    scanfile = image.scan_file

    return scanfile
