
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



def find_exposures(scan_file=None, xpos=None, ypos=None, same_tip=True, tlm_index=None, pcle=None):
    """Reads a list of scans containing grains from the catalogue and
    finds exposures between this and the previous scan"""

    import glob

    if scan_file is None:
        cat = read_grain_cat()
    else:
        cat = pd.DataFrame( [[scan_file, xpos, ypos]], columns=['scan_file', 'xpos', 'ypos'] )

    if pcle is not None:
        if pcle not in cat.index+1:
            print('ERROR: particle number %i not found in catalogue' % pcle)
            return None
        else:
            cat = cat[cat.index==pcle-1]

    tm = ros_tm.tm()

    if tlm_index is not None:
        tm.load_index(tlm_index)
    else:
        tm_files = sorted(glob.glob(os.path.join(ros_tm.tlm_path,'TLM__MD*.DAT')))
        if len(tm_files)==0:
            print('ERROR: no files matching pattern')
            return False
        for f in tm_files:
            tm.get_pkts(f, append=True)

    tm.pkts = tm.pkts[ ((tm.pkts.apid==1084) & ((tm.pkts.sid==129) | (tm.pkts.sid==130))) |
        (((tm.pkts.apid==1079) & ( (tm.pkts.sid==42553) | (tm.pkts.sid==42554) )) |
        ((tm.pkts.apid==1076) & (tm.pkts.sid==2))) ]

    images = tm.get_images(info_only=True)

    grain_images = pd.merge(left=cat, right=images[images.channel=='ZS'], how='inner')
    exposures = tm.get_exposures()

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
            exposure = None
            continue
        else:
            pre_scan = pre_scan.iloc[-1]

            # Particle must be collected between the end of the pre-scan and start of the discvoery
            # scan, so filter exposures between these limits. Must of course have the same target as the image!
            exposure = exposures[ (exposures.target==img.target) &
                (exposures.start > pre_scan.end_time) & (exposures.end < img.start_time)  ]
            duration = timedelta(seconds=exposure.duration.sum().squeeze()/np.timedelta64(1, 's'))
            print('INFO: particle %i found after %i exposures with total duration %s' % (pcle, len(exposure), duration))

    return matches, exposure
