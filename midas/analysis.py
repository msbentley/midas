
#!/usr/bin/python
"""analysis.py

Mark S. Bentley (mark@lunartech.org), 2015

A module containing various routines related to the analysis of MIDAS data,
including investigating particle statistics, exposure geometries etc."""

import os
import pandas as pd
from midas import common, ros_tm
import matplotlib.pyplot as plt
import numpy as np

grain_cat_file = 'grain_cat.csv'
grain_cat_file = os.path.join(common.config_path, grain_cat_file)


def read_grain_cat(grain_cat_file=grain_cat_file):
    """Read the grain catalogue file"""

    col_names = ['scan_file', 'xpos', 'ypos']

    grain_cat = pd.read_table(grain_cat_file, sep=',', header=0,
                              skipinitialspace=True, na_filter=False, names=col_names)

    grain_cat['particle'] = grain_cat.index + 1

    return grain_cat


def find_overlap(image=None, calc_overlap=False, same_tip=True, query=None):
    """Loads all image metadata and loops through all images looking for overlaps.

    image can be a scan_file name, or None to search all images.

    If same_tip=True then matches are only returned for images taken with the same tip.

    If calc_overlap=True a modified dataframe is returned with the overlap in square microns.

    query= can be used to run addition queries on the image dataframe"""

    images = ros_tm.load_images(data=False)
    images = images[images.channel == 'ZS']
    images = images[(images.x_step > 0) & (images.y_step > 0)]

    if image is not None:
        if type(image) == str:
            image = [image]
        target = images[images.scan_file.isin(image)]
    else:
        target = images

    if query is not None:
        images = images.query(query)

    if len(images) == 0:
        print('ERROR: no images match these criteria')
        return None

    cols = ['left', 'right']
    over = pd.DataFrame(columns=cols)

    for idx, image in target.iterrows():

        matches = images[
            (images.scan_file != image.scan_file) & (images.wheel_pos == image.wheel_pos)]

        if same_tip:
            matches = matches[matches.tip_num == image.tip_num]

        h_overlaps = (matches.x_orig_um <= image.x_orig_um +
                      image.xlen_um) & (matches.x_orig_um + matches.xlen_um >= image.x_orig_um)
        v_overlaps = (matches.y_orig_um <= image.y_orig_um +
                      image.ylen_um) & (matches.y_orig_um + matches.ylen_um >= image.y_orig_um)

        matched = matches[h_overlaps & v_overlaps]

        latest = pd.DataFrame(
            {'right': matched.index, 'left': [image.name] * len(matched)})

        dedupe = pd.concat([over, latest.rename(
            columns={'left': 'right', 'right': 'left'})], ignore_index=True)
        idx = dedupe[dedupe.duplicated()].index
        over = pd.concat([over, latest], ignore_index=True).drop(idx)

    if calc_overlap:

        h_overlap = []
        v_overlap = []

        images['x_ext_um'] = images.x_orig_um + images.xlen_um
        images['y_ext_um'] = images.y_orig_um + images.ylen_um

        left_images = images.ix[over.left].reset_index()
        right_images = images.ix[over.right].reset_index()

        for idx in range(len(over)):

            h_overlap.append(abs(min(right_images.x_ext_um.iloc[idx], left_images.x_ext_um.iloc[
                             idx]) - max(right_images.x_orig_um.iloc[idx], left_images.x_orig_um.iloc[idx])))
            v_overlap.append(abs(min(right_images.y_ext_um.iloc[idx], left_images.y_ext_um.iloc[
                             idx]) - max(right_images.y_orig_um.iloc[idx], left_images.y_orig_um.iloc[idx])))

        over['area'] = np.array(h_overlap) * np.array(v_overlap)
        over['perc_left'] = over.area / \
            (left_images.xlen_um * left_images.ylen_um)
        over['perc_right'] = over.area / \
            (right_images.xlen_um * right_images.ylen_um)
        over['tip_left'] = left_images.tip_num
        over['tip_right'] = right_images.tip_num
        over.sort('area', inplace=True)

    over['left'] = over.left.apply(lambda l: images.scan_file.ix[l])
    over['right'] = over.right.apply(lambda r: images.scan_file.ix[r])

    return over


def find_followup(same_tip=True, image_index=None, sourcepath=os.path.expanduser('~/Copy/midas/data/tlm')):
    """Similar to find_exposures() - reads a list of scans containing grains and for
    each grain finds all later scans containing this region, within a window."""

    import glob

    cat = read_grain_cat()
    pcle_imgs = cat.groupby('scan_file')
    scan_files = pcle_imgs.groups.keys()
    num_pcles = [len(x) for x in pcle_imgs.groups.values()]

    tm = ros_tm.tm()

    if image_index:
        images = ros_tm.load_images(data=True)
    else:

        tm_files = sorted(
            glob.glob(os.path.join(common.tlm_path, 'TLM__MD*.DAT')))
        if len(tm_files) == 0:
            print('ERROR: no files matching pattern')
            return False
        for f in tm_files:
            tm.get_pkts(f, append=True)

        tm.pkts = tm.pkts[
            ((tm.pkts.apid == 1084) & ((tm.pkts.sid == 129) | (tm.pkts.sid == 130)))]
        images = tm.get_images(info_only=False)

    grain_images = pd.merge(
        left=cat, right=images[images.channel == 'ZS'], how='inner')

    cols = images.columns.tolist()
    cols.append('particle')
    followup = pd.DataFrame(columns=cols)

    for idx, img in grain_images.iterrows():

        pcle = idx + 1

        ycal = common.xycal['closed'] if img.y_closed else common.xycal['open']
        xcal = common.xycal['closed'] if img.x_closed else common.xycal['open']

        # Calculate the centre position (from the grain cat) in microns
        xc_microns = img.x_orig_um + (img.xpos - img.x_orig) * (xcal / 1000.)
        yc_microns = img.y_orig_um + (img.ypos - img.y_orig) * (ycal / 1000.)

        # Now find all images containing this point (for same facet and segment,
        # POSSIBLY the same tip)
        # i.e. see if xc_microns is between x_orig_um and x_orig_um + xlen_um
        img_seg = images[
            (images.wheel_pos == img.wheel_pos) & (images.channel == 'ZS')]

        if same_tip:
            img_seg = img_seg[img_seg.tip_num == img.tip_num]

        matches = img_seg[(img_seg.x_orig_um < xc_microns) & (xc_microns < img_seg.x_orig_um + img_seg.xlen_um) &
                          (img_seg.y_orig_um < yc_microns) & (yc_microns < img_seg.y_orig_um + img_seg.ylen_um)]

        matches = matches[matches.start_time > img.end_time]

        if len(matches) == 0:
            print(
                'WARNING: no subsequent images found for particle %i, skipping' % pcle)
        else:
            print('INFO: %i subsequent images found containing the position of particle %i' % (
                len(matches), pcle))
            matches['particle'] = pcle
            followup = followup.append(matches, ignore_index=True)

    followup.particle = followup.particle.astype(int)

    return followup


def find_exposures(same_tip=True, tlm_index=None, image_index=None, sourcepath=os.path.expanduser('~/Copy/midas/data/tlm')):
    """Reads a list of scans containing grains from the catalogue and
    finds exposures between this and the previous scan"""

    import glob

    cat = read_grain_cat()
    pcle_imgs = cat.groupby('scan_file')
    scan_files = pcle_imgs.groups.keys()
    num_pcles = [len(x) for x in pcle_imgs.groups.values()]

    tm = ros_tm.tm()

    if tlm_index is not None:
        tm.query_index(sourcepath=sourcepath)
    else:
        tm_files = sorted(
            glob.glob(os.path.join(common.tlm_path, 'TLM__MD*.DAT')))
        if len(tm_files) == 0:
            print('ERROR: no files matching pattern')
            return False
        for f in tm_files:
            tm.get_pkts(f, append=True)

    tm.pkts = tm.pkts[((tm.pkts.apid == 1084) & ((tm.pkts.sid == 129) | (tm.pkts.sid == 130))) |
                      (((tm.pkts.apid == 1079) & ((tm.pkts.sid == 42553) | (tm.pkts.sid == 42554))) |
                       ((tm.pkts.apid == 1076) & (tm.pkts.sid == 2)))]

    if image_index:
        images = ros_tm.load_images(data=False)
    else:
        images = tm.get_images(info_only=True)

    grain_images = pd.merge(
        left=cat, right=images[images.channel == 'ZS'], how='inner')
    # grain_images = images[ (images.channel=='ZS') & (images.scan_file.isin(scan_files)) ]

    all_exposures = tm.get_exposures()
    cols = all_exposures.columns.tolist()
    cols.append('particle')
    exposures = pd.DataFrame(columns=all_exposures.columns)

    # For each grain image, find the previous image containing the coordinates
    # This is assumed to be prior to collection (no dust seen). Exposures
    # between these two times are then collated and geometric information
    # returned.
    for idx, img in grain_images.iterrows():

        pcle = idx + 1

        ycal = common.xycal['closed'] if img.y_closed else common.xycal['open']
        xcal = common.xycal['closed'] if img.x_closed else common.xycal['open']

        # Calculate the centre position (from the grain cat) in microns
        xc_microns = img.x_orig_um + (img.xpos - img.x_orig) * (xcal / 1000.)
        yc_microns = img.y_orig_um + (img.ypos - img.y_orig) * (ycal / 1000.)

        # Now find all images containing this point (for same facet and segment,
        # POSSIBLY the same tip)
        # i.e. see if xc_microns is between x_orig_um and x_orig_um + xlen_um
        img_seg = images[
            (images.wheel_pos == img.wheel_pos) & (images.channel == 'ZS')]
        if same_tip:
            img_seg = img_seg[img_seg.tip_num == img.tip_num]

        matches = img_seg[(img_seg.x_orig_um < xc_microns) & (xc_microns < img_seg.x_orig_um + img_seg.xlen_um) &
                          (img_seg.y_orig_um < yc_microns) & (yc_microns < img_seg.y_orig_um + img_seg.ylen_um)]

        if len(matches) == 0:
            print(
                'INFO: no images found containing the position of particle %i' % pcle)
        else:
            print('INFO: %i images found containing the position of particle %i' % (
                len(matches), pcle))

        # The last scan of this position is assumed to contain no grain, and hence be our
        # last pre-scan (by the definition of the entry in the grain catalogue)
        pre_scan = matches[matches.end_time < img.start_time]
        if len(pre_scan) == 0:
            print('WARNING: no pre-scan found for particle %i, skipping' %
                  pcle)
            # exposures = None
            continue
        else:
            pre_scan = pre_scan.iloc[-1]

            # Particle must be collected between the end of the pre-scan and start of the discovery
            # scan, so filter exposures between these limits. Must of course
            # have the same target as the image!
            exp = (all_exposures[(all_exposures.target == img.target) &
                                 (all_exposures.start > pre_scan.end_time) & (all_exposures.end < img.start_time)])

            # print('INFO: particle %i found after %i exposures with total duration %s' % (pcle, len(exposures), duration))
            print('INFO: particle %i found after %i exposures' %
                  (pcle, len(exp)))

            # add particle number and pre-scan (last scan before collection)
            # image nanme
            exp['particle'] = pcle
            exp['prescan'] = pre_scan.scan_file

            exposures = exposures.append(exp)

    exposures.particle = exposures.particle.astype(int)

    return exposures


def read_lap_file(filename):
    """Read a LAP TAB file"""

    # Start and stop UT for each sweep (having two times is not really good, use the start, sweeps are short anyway).
    # Start and stop SCT for each sweep.
    # Some quality flag, which as yet is mostly some kind of decoration.
    # Plasma density estimate (cm-3), do not use or trust!!!
    # Electron temperature (eV), same caveat though perhaps with only two exclamation marks.
    # Photoelectron knee potential Vph (V), proxy for Vsc. In this we trust, though it should be off from Vsc by some factor 1.5 - 2.5 or so.
    # Which probe (1 or 2) the data originates from. Don't care.
    # Direction of sweep. Don't care.
    # Probe illumination, should be 1.00 for all Vph data or something is very strange.
    # Sweep group number, just a running index, forget about it.

    columns = ['start_utc', 'end_utc', 'start_obt', 'end_obt', 'quality', 'plasma_dens', 'e_temp',
               'sc_pot', 'probe', 'direction', 'illum', 'sweep_grp']

    lap = pd.read_table(filename, sep=',', skipinitialspace=True, header=False, names=columns,
                        parse_dates=['start_utc', 'end_utc'])

    return lap


def read_lap(directory, geom=False):
    """Read a directory of LAP data files and append to a dataframe, optionally adding SPICE geometry."""

    import glob
    import spice_utils

    lap_files = sorted(glob.glob(os.path.join(directory, 'RPCLAP*.TAB')))

    df = pd.DataFrame()

    for lap in lap_files:
        df = df.append(read_lap_file(lap))

    df.sort('start_utc', inplace=True)
    df.set_index('start_utc', inplace=True)

    if geom:
        geometry = spice_utils.get_geometry_at_times(df.index.to_datetime())
        return pd.merge(left=df, right=geometry, left_index=True, right_index=True)
    else:
        return df


def plot_lap(lap):
    """Plot LAP spacecraft potential data"""

    import matplotlib.dates as md

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(lap.index, lap.sc_pot, '.', label='potential')
    ax.grid(True)

    fig.autofmt_xdate()
    # ax.fmt_xdata = md.DateFormatter('%Y-%m-%d')
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)

    ax.set_xlabel('Sweep time')
    ax.set_ylabel('s/c potential (V)')

    plt.show()

    return


def extract_masks(gwy_file, channel=None):
    """Looks for mask channels within the Gwyddion file and returns these
    masks as a list of numpy arrays. If channel= is given a string, only
    channels with names containing this string are returned."""

    import gwy
    import re

    C = gwy.gwy_file_load(gwy_file, gwy.RUN_NONINTERACTIVE)
    keys = zip(C.keys(), C.keys_by_name())

    # Use regex to locate all channels
    masks = []
    for key in C.keys_by_name():
        m = re.match(r'^/(?P<i>\d+)/mask$', key)
        if not m:
            continue
        masks.append(int(m.group('i')))
    masks = sorted(masks)
    if len(masks) == 0:
        print('WARNING: No masks found in Gwyddion file %s' % gwy_file)
        return None

    # If requested, find only those channels contaning substring "channel"
    #
    if channel is not None:
        matching = []
        for mask in masks:
            name = C.get_value_by_name('/%d/data/title' % mask)
            if channel in name:
                matching.append(channel)

        if len(matching) == 0:
            print('WARNING: No masks in channels matching "%s" found' %
                  (channel))
            return None

    else:
        matching = masks

    # Now get the mask data for each (requested) channel
    mask_data = []
    for idx in range(len(matching)):

        datafield = C.get_value_by_name('/%d/mask' % masks[idx])
        data = np.array(datafield.get_data(), dtype=np.bool).reshape(
            datafield.get_yres(), datafield.get_xres())
        mask_data.append(data)

    print('INFO: %d masks extracted from Gwyddion file %s' %
          (len(mask_data), gwy_file))

    return mask_data



def get_gwy_data(gwy_file, chan_name=None):
    """Returns data from a Gwyddion file with channel matching channel=, or
    the first channel if channel=None."""

    import gwy, re

    C = gwy.gwy_file_load(gwy_file, gwy.RUN_NONINTERACTIVE)
    keys = zip(C.keys(), C.keys_by_name())

    channels = []
    for key in C.keys_by_name():
        m = re.match(r'^/(?P<i>\d+)/data$', key)
        if not m:
            continue
        channels.append(int(m.group('i')))
    channels = sorted(channels)
    if len(channels) == 0:
        print('WARNING: No data channels found in Gwyddion file %s' % gwy_file)
        return None

    if chan_name is None:
        print('INFO: No channel specified, using %s' % C.get_value_by_name('/%d/data/title' % channels[0]))
        selected = channels[0]
    else:
        selected = None
        for channel in channels:
            name = C.get_value_by_name('/%d/data/title' % channel)
            if name==chan_name:
                selected = channel
                break
        if selected is None:
            print('WARNING: channel not found!')
            return None

    datafield = C.get_value_by_name('/%d/data' % selected)
    data = np.array(datafield.get_data(), dtype=np.float32).reshape(
        datafield.get_yres(), datafield.get_xres())

    return data
