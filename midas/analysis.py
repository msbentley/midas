
#!/usr/bin/python
"""analysis.py

Mark S. Bentley (mark@lunartech.org), 2015

A module containing various routines related to the analysis of MIDAS data,
including investigating particle statistics, exposure geometries etc."""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from midas import common, ros_tm
import matplotlib.pyplot as plt

grain_cat_file = 'grain_cat.csv'
grain_cat_file = os.path.join(common.config_path, grain_cat_file)


def read_grain_cat(grain_cat_file=grain_cat_file):
    """Read the grain catalogue file"""

    col_names = ['scan_file', 'xpos', 'ypos']

    grain_cat = pd.read_table(grain_cat_file, sep=',', header=0,
        skipinitialspace=True, na_filter=False, names=col_names)

    return grain_cat



def find_overlap(images, scanfile, calc_overlap=False, same_tip=True):
    """Accepts an image dataframe (from ros_tm.get_images()) and an image name and
    returns a list of overlapping images.

    If same_tip=True then matches are only returned for images taken with the same tip.
    If calc_overlap=True a modified dataframe is returned with the overlap in square microns."""

    src_image = images[ (images.scan_file==scanfile) & (images.channel=='ZS') ]

    if len(src_image)==0:
        print('ERROR: could not find (topography) image %s' % scanfile)
        return None
    elif len(src_image)>1:
        print('ERROR: more than one match for image %s' % scanfile)
        return None

    src_image = src_image.squeeze()

    matches = images[ images.scan_file!=src_image.scan_file ]
    matches = matches[ matches.wheel_pos==src_image.wheel_pos ]

    if same_tip:
        matches = images[ images.tip_num==src_image.tip_num ]

    h_overlaps = (matches.x_orig_um <= src_image.x_orig_um+src_image.xlen_um) & (matches.x_orig_um+matches.xlen_um >= src_image.x_orig_um)
    v_overlaps = (matches.y_orig_um <= src_image.y_orig_um+src_image.ylen_um) & (matches.y_orig_um+matches.ylen_um >= src_image.y_orig_um)

    matches = matches[ h_overlaps & v_overlaps ]

    return matches




def find_exposures(same_tip=True, tlm_index=None, image_index=None, sourcepath=None):
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
        tm_files = sorted(glob.glob(os.path.join(common.tlm_path,'TLM__MD*.DAT')))
        if len(tm_files)==0:
            print('ERROR: no files matching pattern')
            return False
        for f in tm_files:
            tm.get_pkts(f, append=True)

    tm.pkts = tm.pkts[ ((tm.pkts.apid==1084) & ((tm.pkts.sid==129) | (tm.pkts.sid==130))) |
        (((tm.pkts.apid==1079) & ( (tm.pkts.sid==42553) | (tm.pkts.sid==42554) )) |
        ((tm.pkts.apid==1076) & (tm.pkts.sid==2))) ]

    if image_index:
        images = ros_tm.load_images(data=False)
    else:
        images = tm.get_images(info_only=True)

    grain_images = pd.merge(left=cat, right=images[images.channel=='ZS'], how='inner')
    # grain_images = images[ (images.channel=='ZS') & (images.scan_file.isin(scan_files)) ]

    all_exposures = tm.get_exposures()
    cols = all_exposures.columns.tolist()
    cols.append('particle')
    exposures = pd.DataFrame(columns=all_exposures.columns)

    # For each grain image, find the previous image containing the coordinates
    # This is assumed to be prior to collection (no dust seen). Exposures
    # between these two times are then collated and geometric information returned.
    for idx, img in grain_images.iterrows():

        pcle = idx+1

        ycal = common.xycal['closed'] if img.y_closed else common.xycal['open']
        xcal = common.xycal['closed'] if img.x_closed else common.xycal['open']

        # Calculate the centre position (from the grain cat) in microns
        xc_microns = img.x_orig_um + (img.xpos-img.x_orig)*(xcal/1000.)
        yc_microns = img.y_orig_um + (img.ypos-img.y_orig)*(ycal/1000.)

        # Now find all images containing this point (for same facet and segment,
        # POSSIBLY the same tip)
        # i.e. see if xc_microns is between x_orig_um and x_orig_um + xlen_um
        img_seg = images[ (images.wheel_pos == img.wheel_pos) & (images.channel=='ZS') ]
        if same_tip:
            img_seg = img_seg[ img_seg.tip_num == img.tip_num ]

        matches = img_seg[ (img_seg.x_orig_um < xc_microns) & (xc_microns < img_seg.x_orig_um + img_seg.xlen_um) &
            (img_seg.y_orig_um < yc_microns) & (yc_microns < img_seg.y_orig_um + img_seg.ylen_um) ]

        if len(matches)==0:
            print('INFO: no images found containing the position of particle %i' % pcle)
        else:
            print('INFO: %i images found containing the position of particle %i' % (len(matches), pcle))

        # The last scan of this position is assumed to contain no grain, and hence be our
        # last pre-scan (by the definition of the entry in the grain catalogue)
        pre_scan = matches[ matches.end_time < img.start_time ]
        if len(pre_scan)==0:
            print('WARNING: no pre-scan found for particle %i, skipping' % pcle)
            # exposures = None
            continue
        else:
            pre_scan = pre_scan.iloc[-1]

            # Particle must be collected between the end of the pre-scan and start of the discovery
            # scan, so filter exposures between these limits. Must of course have the same target as the image!
            exp = (all_exposures[ (all_exposures.target==img.target) &
                (all_exposures.start > pre_scan.end_time) & (all_exposures.end < img.start_time) ])
            # duration = timedelta(seconds=all_exposures.duration.sum().squeeze()/np.timedelta64(1, 's'))

            # print('INFO: particle %i found after %i exposures with total duration %s' % (pcle, len(exposures), duration))
            print('INFO: particle %i found after %i exposures' % (pcle, len(exp)))

            # add particle number
            exp['particle'] = pcle

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

    columns = ['start_utc','end_utc', 'start_obt', 'end_obt', 'quality', 'plasma_dens', 'e_temp',
        'sc_pot', 'probe', 'direction', 'illum', 'sweep_grp']

    lap = pd.read_table(filename, sep=',', skipinitialspace=True, header=False, names=columns,
        parse_dates=['start_utc','end_utc'])

    return lap


def read_lap(directory, geom=False):
    """Read a directory of LAP data files and append to a dataframe, optionally adding SPICE geometry."""

    import glob, spice_utils

    lap_files = sorted(glob.glob(os.path.join(directory,'RPCLAP*.TAB')))

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
    ax = fig.add_subplot(1,1,1)

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
