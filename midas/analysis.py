#!/usr/bin/python
# -*- coding: utf-8 -*-
"""analysis.py

Mark S. Bentley (mark@lunartech.org), 2015

A module containing various routines related to the analysis of MIDAS data,
including investigating particle statistics, exposure geometries etc."""

import os, glob
import pandas as pd
from midas import common, ros_tm, gwy_utils
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import numpy.ma as ma
import numpy as np

grain_cat_file = 'grain_cat.csv'
grain_cat_file = os.path.join(common.config_path, grain_cat_file)

import logging
log = logging.getLogger(__name__)


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

    if query is not None:
        images = images.query(query)

    if image is not None:
        if type(image) == str:
            image = [image]
        target = images[images.scan_file.isin(image)]
    else:
        target = images

    if len(images) == 0:
        log.error('no images match these criteria')
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
        over['left'] = over['left'].astype(np.int64)
        over['right'] = over['right'].astype(np.int64)

    if calc_overlap:

        h_overlap = []
        v_overlap = []

        images['x_ext_um'] = images.x_orig_um + images.xlen_um
        images['y_ext_um'] = images.y_orig_um + images.ylen_um

        left_images = images.loc[over.left].reset_index()
        right_images = images.loc[over.right].reset_index()

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
        over.sort_values(by='area', inplace=True)

    over['left'] = over.left.apply(lambda l: images.scan_file.loc[l])
    over['right'] = over.right.apply(lambda r: images.scan_file.loc[r])

    return over


def find_followup(same_tip=True, image_index=None, sourcepath=common.tlm_path):
    """Similar to find_exposures() - reads a list of scans containing grains and for
    each grain finds all later scans containing this region, within a window."""

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
            log.error('no files matching pattern')
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
            log.warning('no subsequent images found for particle %i, skipping' % pcle)
        else:
            log.info('%i subsequent images found containing the position of particle %i' % (
                len(matches), pcle))
            matches['particle'] = pcle
            followup = followup.append(matches, ignore_index=True)

    followup.particle = followup.particle.astype(int)

    return followup


def find_exposures(same_tip=True, tlm_index=None, image_index=None, sourcepath=common.tlm_path):
    """Reads a list of scans containing grains from the catalogue and
    finds exposures between this and the previous scan"""

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
            log.error('no files matching pattern')
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
            log.info('no images found containing the position of particle %i' % pcle)
        else:
            log.info('%i images found containing the position of particle %i' % (
                len(matches), pcle))

        # The last scan of this position is assumed to contain no grain, and hence be our
        # last pre-scan (by the definition of the entry in the grain catalogue)
        pre_scan = matches[matches.end_time < img.start_time]
        if len(pre_scan) == 0:
            log.warning('no pre-scan found for particle %i, skipping' % pcle)
            continue
        else:
            pre_scan = pre_scan.iloc[-1]

            # Particle must be collected between the end of the pre-scan and start of the discovery
            # scan, so filter exposures between these limits. Must of course
            # have the same target as the image!
            exp = (all_exposures[(all_exposures.target == img.target) &
                                 (all_exposures.start > pre_scan.end_time) & (all_exposures.end < img.start_time)])

            log.info('particle %i found after %i exposures' % (pcle, len(exp)))

            # add particle number and pre-scan (last scan before collection)
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

    lap = pd.read_table(filename, sep=',', skipinitialspace=True, header=None, names=columns,
                        parse_dates=['start_utc', 'end_utc'])

    return lap


def read_lap(directory, geom=False):
    """Read a directory of LAP data files and append to a dataframe, optionally adding SPICE geometry."""

    import spice_utils

    lap_files = sorted(glob.glob(os.path.join(directory, 'RPCLAP*.TAB')))

    df = pd.DataFrame()

    for lap in lap_files:
        df = df.append(read_lap_file(lap))

    df.sort_values(by='start_utc', inplace=True)
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

def read_grain_stats(basename, path='.'):

    files = sorted(glob.glob(os.path.join(path, basename+'*')))


    if len(files)==0:
        log.error('no files matching pattern: %s' % basename)
        return None

    with open(files[0], 'r') as f:
        header = f.readline()
        cols = header.split()[1:]

    grain_stats = pd.DataFrame([], columns=cols)

    for grainfile in files:
        grain_stats = grain_stats.append(pd.read_table(grainfile, header=None, skiprows=1, names=cols))

    return grain_stats


def get_subpcles(gwyfiles, directory='.', recursive=False, chan='sub_particle_', gwy_meta=True):
    """Reads channel data corresponding to chan= (or all, if None) and
    containing a mask.

    If gwy_meta=True meta-data are retrieved from the Gwyddion file, otherwise it is read from
    the images metadata file."""

    from skimage import measure

    if recursive:
        selectfiles = common.locate(gwyfiles, directory)
        filelist = [file for file in selectfiles]
    elif type(gwyfiles)==list:
        filelist = gwyfiles
    else:
        import glob
        filelist = glob.glob(os.path.join(directory, gwyfiles))

        pcle_data = []

    if len(filelist)==0:
        log.warning('No matching files found')
        return None
    else:
        log.info('Processing sub-particle data from %d files' % len(filelist))

    for gwyfile in filelist:

        # Get all masked channels matching the sub-particle filter
        channels = gwy_utils.list_chans(gwyfile, chan, masked=True, info=True)
        if channels is None:
            log.error('no channels found matching: %s' % chan)
            return None

        # Get basic meta-data from the first channel of the GWY file
        meta = channels.iloc[0]

        filemeta = gwy_utils.get_meta(gwyfile, channel='Topography (Z piezo position set value)')

        if not gwy_meta:
            scan_file = filemeta.scan_file
            meta = ros_tm.load_images(data=False).query('scan_file==@scan_file').squeeze()

        # Calculate the pixel area (simply x_step*y_step)
        pix_area = float(meta.x_step_nm)*1.e-9 * float(meta.y_step_nm)*1.e-9 #m2
        pix_len = np.sqrt(pix_area) #  in the case of non-uniform steps!

        for idx, channel in channels.iterrows():
            # chan_id, chan_name
            log.debug('processing channel %s' % channel['name'])
            mask = gwy_utils.extract_masks(gwyfile, channel['name'])
            xlen, ylen, data = gwy_utils.get_data(gwyfile, channel['name'])

            locs = np.where(mask)
            left, right = locs[1].min(), locs[1].max()
            up, down = locs[0].min(), locs[0].max()
            pcle = ma.masked_array(data, mask=~mask)
            pcle = pcle[up:down+1,left:right+1]

            labmask = measure.label(mask)
            regions = measure.regionprops(labmask)
            region = regions[0]

            if pcle.count() != int(region.area):
                log.warning('region count does not match mask count (>1 grain per mask) in channel %s' % channel['name'])

            subpcle = {

                'pcle': pcle,
                'filename': os.path.abspath(gwyfile),
                'scan_file': filemeta.scan_file if filemeta is not None else None,
                'id': int(channel.id),
                'name': channel['name'],
                'left': left,
                'right': right,
                'up': up,
                'down': down,
                'centre_x': region.centroid[0],
                'centre_y': region.centroid[1],
                'tot_min_z': data.min()*1.e6,
                'tot_max_z': data.max()*1.e6,
                'x_step_nm': meta.x_step_nm,
                'y_step_nm': meta.y_step_nm,
                'a_pix': pcle.count(),
                'a_pcle': pcle.count() * pix_area,
                'r_eq': np.sqrt(pcle.count() * pix_area / np.pi),
                'z_min': pcle.min(),
                'z_max': pcle.max(),
                'z_mean': pcle.mean(),
                'major': region.major_axis_length * pix_len,
                'minor': region.minor_axis_length * pix_len,
                'eccen': region.eccentricity,
                'compact': region.perimeter**2. / (4.*np.pi*pcle.count()),
                'sphericity': min( (region.minor_axis_length/region.major_axis_length),
                    (pcle.max()/(region.major_axis_length*pix_len)),
                    ((region.minor_axis_length*pix_len)/pcle.max()) ),
                'orient': np.degrees(region.orientation),
                'pdata': pcle
                }
            pcle_data.append(subpcle)

    pcle_data = pd.DataFrame.from_records(pcle_data)

    if len(pcle_data)==0:
        log.warning('No sub-particle data found')
        return None

    # pcle_data.sort_values(by='id', inplace=True)
    pcle_data['tot_z_diff'] = pcle_data.tot_max_z - pcle_data.tot_min_z
    pcle_data['z_diff'] = pcle_data.z_max - pcle_data.z_min
    pcle_data['d_eq'] = pcle_data['r_eq'] * 2.

    pcle_data.drop(['pcle'], axis=1, inplace=True)

    return pcle_data


def calc_errors(pcle_data, error_pix=0.15, half_angle=22.5, min_feature=0, diameter=True):
    """Calculate errors on the sub-particle data (from get_subpcles()) according to the
    given input. The diameter switch is used to determine if radius or diameter are
    being used in subsequent calculations, to ensure that err_lower is set correctly
    for plotting with e.g. plot_cumulative()"""

    pcle_data['err_mark'] = ((pcle_data.a_pix * np.pi)**(-0.5) * np.ceil(pcle_data.a_pix * error_pix)) * pcle_data.x_step_nm*1.e-9
    pcle_data['err_conv'] = 2. * np.tan(np.deg2rad(half_angle)) * (pcle_data.z_mean - pcle_data.z_min)
    pcle_data.loc[ pcle_data['err_conv'] < min_feature, 'err_conv']  = min_feature
    pcle_data['err_tot'] = pcle_data.err_conv + pcle_data.err_mark

    factor = 2. if diameter else 1.

    pcle_data['err_lower'] = pcle_data.apply( lambda grain: grain.err_tot if (grain.r_eq*factor-grain.err_tot)>0. else (grain.r_eq*factor), axis=1 )

    return pcle_data


def show_pcle_from_gwy(gwyfile, channel='Topography (Z piezo position set value)', savefig=False, save_prefix='', fontsize=10, cmap=None):

    xlen, ylen, data = gwy_utils.get_data(gwyfile, chan_name=channel)

    fig, ax = plt.subplots()

    cmap = plt.cm.afmhot if cmap is None else cmap

    image = ax.imshow(data * 1.e9, cmap=cmap, origin='upper', interpolation='nearest',
        extent=[0,xlen*1.e6,ylen*1.e6,0])

    ax.set_xlabel('X (microns)', fontsize=fontsize)
    ax.set_ylabel('Y (microns)', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)

    plt.setp(ax.get_xticklabels(), rotation=45)

    bar = fig.colorbar(image)
    bar.ax.tick_params(labelsize=fontsize)

    if savefig:
        figpath, figfile = os.path.split(gwyfile)
        figfile = ''.join(figfile.split('.')[:-1])
        figfile = os.path.join(figpath, save_prefix + figfile)
        fig.savefig('%s.eps' % figfile, format='eps', bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    return


def show_masked_pcle(pcles, channel='sub_particle_', outline=True, title='', query=None, savefig=False, save_prefix='',
        fontsize=10, cmap=None, norm_col='cyan', query_col='red'):

    from skimage import segmentation
    from matplotlib import colors, cm

    for scanfile in sorted(pcles.scan_file.unique().tolist()):

        pcle = pcles[pcles.scan_file==scanfile]
        gwyfile = pcle.filename.iloc[0]

        if outline:
            xlen, ylen, data = gwy_utils.get_data(gwyfile, chan_name='sub_particle_001')
        else:
            xlen, ylen, data = gwy_utils.get_data(gwyfile, chan_name=channel)
        data = data - data.min()

        fig, ax = plt.subplots()
        cmap = plt.cm.afmhot if cmap is None else cmap

        if outline:

            perim_norm = []
            perim_query = []

            masks = gwy_utils.extract_masks(gwyfile, channel=channel, return_channels=True)

            perim_norm_full = np.zeros_like(masks[0][1])
            perim_query_full = np.zeros_like(perim_norm_full)

            masks = pd.DataFrame(masks, columns=['name', 'mask'])
            pcle = pd.merge(masks, pcle, on='name')

            if query is None:

                for idx, mask in pcle['mask'].iteritems():
                    perim_norm.append(segmentation.find_boundaries(mask, connectivity=1, mode='inner', background=0))
                    perim_norm_full = np.logical_or(perim_norm_full, perim_norm[-1])

            else:

                pcles_query = pcle.query(query)
                rest_query = pcle[~pcle.index.isin(pcle.query(query).index)]

                for idx, mask in pcles_query['mask'].iteritems():
                    perim_query.append(segmentation.find_boundaries(mask, connectivity=1, mode='inner', background=0))
                    perim_query_full = np.logical_or(perim_query_full, perim_query[-1])

                for idx, mask in rest_query['mask'].iteritems():
                    perim_norm.append(segmentation.find_boundaries(mask, connectivity=1, mode='inner', background=0))
                    perim_norm_full = np.logical_or(perim_norm_full, perim_norm[-1])

                data[perim_query_full] = -1.

                cmap.set_under(query_col)

            cmap.set_bad(norm_col)

            perim_data = np.ma.array(data, mask=perim_norm_full)

        plotdata = perim_data if outline else data

        image = ax.imshow(plotdata * 1.e9, cmap=cmap, origin='upper', interpolation='nearest',
            extent=[0,xlen*1.e6,ylen*1.e6,0], norm=colors.Normalize(vmin=0., vmax=data.max()*1.e9))

        ax.set_xlabel('X (microns)', fontsize=fontsize)
        ax.set_ylabel('Y (microns)', fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.set_title(title, fontsize=fontsize)

        plt.setp(ax.get_xticklabels(), rotation=45)

        bar = fig.colorbar(image, ax=ax, extend='max')
        bar.ax.tick_params(labelsize=fontsize)
        # bar.set_label('height (nm)')

        if savefig:
            figpath, figfile = os.path.split(gwyfile)
            figfile = ''.join(figfile.split('.')[:-1])
            figfile += '_outline' if outline else ''
            figfile = os.path.join(figpath, save_prefix + figfile)
            fig.savefig('%s.eps' % figfile, format='eps', bbox_inches='tight', DPI=300)
            plt.close(fig)

    if not savefig:
        plt.show()

    return fig


def plot_cumulative(pcle_data, query=None, savefig=False, title=True, title_field=None, title_prefix=None, save_prefix='', show_mean=False):
    """Plots a cumulative size distribution from sub-particle data. Data from
    multiple scans can be included, and one plot will be opened for each. If
    savefig=True an EPS file will be output.

    If title=True, a plot title will be used - specified by title_field.

    If title_field=None the scan name is used. If title_field matches a
    column name which is unique, this is used, otherwise the passed string.

    If query is set to a pandas dataframe query statement, this will be applied
    and the selected points plotted in an alternative colour."""

    for scanfile in sorted(pcle_data.scan_file.unique().tolist()):

        pcle = pcle_data[pcle_data.scan_file==scanfile].sort_values(by='r_eq')
        pcle['idx'] = np.arange(1,len(pcle)+1)

        fig, ax_left = plt.subplots(figsize=(8,4))
        ax_left.set_xlabel('effective diameter (nm)')
        ax_left.set_ylabel('counts')

        if query is None:
            ax_left.errorbar(pcle.d_eq*1.e9, pcle.idx, xerr=[pcle.err_lower*1.e9, pcle.err_mark*1.e9], fmt='s',
                ecolor='Silver', markersize=3, color='black', capsize=2)
        else:
            highlight = pcle.query(query)
            ax_left.errorbar(highlight.d_eq*1.e9, highlight.idx, xerr=[highlight.err_lower*1.e9, highlight.err_mark*1.e9], fmt='s',
                markersize=3, color='red', capsize=2)
            rest = pcle[~pcle.index.isin(pcle.query(query).index)]
            ax_left.errorbar(rest.d_eq*1.e9, rest.idx, xerr=[rest.err_lower*1.e9, rest.err_mark*1.e9], fmt='s',
                ecolor='Silver', markersize=3, color='black', capsize=2)


        ax_left.set_ylim(bottom=0.)
        ax_left.set_xlim(left=0.)
        ax_left.grid(True)
        ymin, ymax = ax_left.get_ylim()
        ax_right = ax_left.twinx()
        ax_right.set_ylabel('fraction')
        ax_right.set_ylim(0., ymax/float(len(pcle)))

        if title:
            if title_field is None:
                title = scanfile
            elif (title_field in pcle.columns) and (len(pcle['%s' % title_field].unique()==1)):
                title = pcle['%s' % title_field].unique()[0]
            else:
                title = title_field

            if title_prefix is not None:
                title = title_prefix + title

            ax_left.set_title(title)

        if show_mean:

            legends = []

            mean = pcle.d_eq.mean()*1.e9
            mean_err_min = pcle.err_lower.mean()*1.e9
            mean_err_max = pcle.err_mark.mean()*1.e9
            legends.append(ax_left.axvline(mean, label='mean', color='black'))
            ax_left.axvspan(mean-mean_err_min, mean+mean_err_max, facecolor='yellow', alpha=0.5)

            median = pcle.d_eq.median()*1.e9
            legends.append(ax_left.axvline(median, linestyle='--', color='blue', label='median'))

            leg = ax_left.legend(legends, ['mean', 'median'], loc='lower right')

        if savefig:
            filename = '%s%s_cum.eps' % (save_prefix, scanfile)
            fig.savefig(filename, format='eps', bbox_inches='tight')
            log.info('saving file %s' % filename)
            plt.close(fig)
        else:
            plt.show()
            return fig

    return


def plot_assembly(pcle_data, anim=False, figure=None, axis=None, extent=None, centre=None):
    """Plot an assembly of sub-particles"""

    if figure is None:
        fig = plt.figure()
    else:
        fig = figure

    if axis is None:
        ax = fig.add_subplot(111)
    else:
        ax = axis

    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

    if centre is None:
        # cent = np.array( [ (pcle_data.down.max()+1.)/2., (pcle_data.right.max()+1.)/2. ] )
        if extent is not None:
            cent = np.array(extent)/2
    else:
        cent = np.array(centre)

    if extent is None:
        new_array = np.zeros( (pcle_data.down.max()+1, pcle_data.right.max()+1), dtype=np.float64)
    else:
        new_array = np.zeros( extent, dtype=np.float64)
        x_off = int(round(extent[0]/2 - cent[0]))
        y_off = int(round(extent[1]/2 - cent[1]))

    # new_array.mask = True

    for idx, pcle in pcle_data.iterrows():
        if extent is None:
            new_array[pcle.up:pcle.down+1, pcle.left:pcle.right+1][~pcle['pdata'].mask] = pcle['pdata'][ ~pcle['pdata'].mask ]
        else:
            new_array[pcle.up+y_off:pcle.down+y_off+1, pcle.left+x_off:pcle.right+x_off+1][~pcle['pdata'].mask] = pcle['pdata'][ ~pcle['pdata'].mask ]

    cmap = cm.afmhot
    cmap.set_bad('c', alpha=1.0)

    im = ax.imshow(new_array, cmap=cmap, interpolation='nearest', vmin=0, vmax=new_array.max(), animated=anim)

    plt.show()

    return im


def animate_assembly(orig_data, num_frames=50, pix_per_frame=1.0, centre=None, ani_file='subpcle_anim.mp4'):
    """Plot an animation of sub particle assembly and/or disassembly.

    If centre= is set to a tuple corresponding to a pixel position, this is used as
    the centre position, otherwise the centre of the bounding box is taken.

    num_frames= gives the number of frames to animate over

    ani_file= is the output file (with full path if necessary)"""

    import matplotlib.animation as animation

    # If no expansion centre is given, use the centre of the main particle
    # bounding box
    if centre is None:
        cent = np.array( [ (orig_data.down.max()+1.)/2., (orig_data.right.max()+1.)/2. ] )
    else:
        cent = np.array(centre)

    pcle_data = orig_data.copy()

    # add a new column to the dataframe with the (normalised) vector from the particle centre
    # to each sub-particle centre
    pcle_data['pos_x'] = (pcle_data.right-pcle_data.left)/2. + pcle_data.left - cent[1]
    pcle_data['pos_y'] = (pcle_data.up-pcle_data.down)/2. + pcle_data.down - cent[0]
    pcle_data['mag']   = np.sqrt( pcle_data.pos_x**2. + pcle_data.pos_y**2. )
    pcle_data['pos_x'] = pcle_data.pos_x / pcle_data.mag
    pcle_data['pos_y'] = pcle_data.pos_y / pcle_data.mag

    fig = plt.figure()
    frames = []

    # Calculate the bounding box of the final frame (need to know how large to
    # make the array before plotting the first frame)
    new_data = orig_data.copy()
    for idx, pcle in new_data.iterrows():
        delta_x = num_frames * pix_per_frame * pcle_data.pos_x.loc[idx]
        delta_y = num_frames * pix_per_frame * pcle_data.pos_y.loc[idx]
        pcle.up += delta_y
        pcle.down += delta_y
        pcle.left += delta_x
        pcle.right += delta_x

        new_data.loc[idx] = pcle

    max_x = int(round(max(abs(new_data.right.max()-cent[0]),abs(new_data.left.min()-cent[0]))))
    max_y = int(round(max(abs(new_data.up.max()-cent[1]),abs(new_data.down.min()-cent[1]))))

    for frame_cnt in range(num_frames):

        new_data = orig_data.copy()

        for idx, pcle in new_data.iterrows():

            # calculate the offset according to the unit vector (direction)
            # and speed (given in pixels per frame)
            delta_x = frame_cnt * pix_per_frame * pcle_data.pos_x.loc[idx]
            delta_y = frame_cnt * pix_per_frame * pcle_data.pos_y.loc[idx]

            # apply the offset to the bounding box (used for plotting the sub-pcle)
            pcle.up += delta_y
            pcle.down += delta_y
            pcle.left += delta_x
            pcle.right += delta_x

            new_data.loc[idx] = pcle

        im = plot_assembly(new_data, anim=True, figure=fig, extent=(max_y*2,max_x*2), centre=cent)
        frames.append([im])

    ani = animation.ArtistAnimation(fig, frames, interval=50, blit=True, repeat_delay=1000)

    ani.save(ani_file)
    plt.close()

    return




def plot_subpcles(pcle_data, num_cols=3, scale=False, title=None, savefile=None, cbar=False):
    """Simple overview plot of all sub-particle data. The number of columns can
    be set via num_cols= and the number of rows will be calculated accordingly.

    By default all sub-particles are drawn to fill the sub-plot, and so the pixel
    size of each is different. To force all sub-particles to be drawn to the
    scale of the largest, set scale=True.

    The final output can be written to a file by specifiying a filename (and path
    if required) in savefile=."""

    from matplotlib import gridspec

    # find min/max of all data for plotting
    vmin = pcle_data.z_min.min()
    vmax = pcle_data.z_max.max()

    if scale:
        # find the size of the bounding box of the particle with
        # the largest linear size (in either direction)
        max_size = max((pcle_data.right-pcle_data.left).max(),(pcle_data.down-pcle_data.up).max()) + 1

    # Display all of the sub-particles on a grid of subplots
    if len(pcle_data) < num_cols:
        num_cols = len(pcle_data)
        num_rows = 1
    else:
        num_rows = (len(pcle_data) / num_cols)
        if len(pcle_data) % num_cols != 0:
            num_rows += 1

    if cbar:
        num_rows += 1

    gs = gridspec.GridSpec(num_rows, num_cols, height_ratios=[1]*num_rows, width_ratios=[1]*num_cols)
    fig = plt.figure()

    grid = 0

    for idx, pcle in pcle_data.iterrows():
        ax = plt.subplot(gs[grid])
        ax.set_aspect('equal')
        ax.set_axis_off()
        ax.get_yaxis().set_visible(False)
        ax.get_xaxis().set_visible(False)

        if title is not None:
            if title not in pcle_data.columns:
                log.warning('label %s not valid for the title' % title)
                title_txt = pcle['name']
            else:
                title_txt = '%s: %s' % (title, pcle['%s' % title])

            ax.set_title(title_txt, fontsize=12)

        data = pcle['pdata']

        if scale:

            new_size = (max_size, max_size)
            new_arr = ma.zeros(new_size, dtype=data.dtype)
            new_arr.mask = True

            # insert the sub-particle data into this array
            new_arr[0:data.shape[0],0:data.shape[1]] = data
            # TODO - would it be better to centre in the array?

        else:
            new_arr = data

        im = ax.imshow(new_arr*1.e6, interpolation='nearest', cmap=cm.afmhot, vmin=vmin*1.e6, vmax=vmax*1.e6)
        grid += 1

    if cbar:
        cax = fig.add_axes([0.2, 0.1, 0.6, 0.05])
        # cax = plt.subplot(gs[-1,:])
        cbar = fig.colorbar(im, cax=cax, label='height (microns)', orientation='horizontal')
        cbar.ax.tick_params(labelsize=10)

    if savefile is not None:
        fig.savefig(savefile)

    plt.show()

    return


def plot_pcles(pcles, figure=None, axis=None, show_stripes=True, zoom_out=False, label=None, labelfont=8,
                show_scan=False, show_seg=False):
    """Plot particles in the passed data frame on their respective facets."""

    import matplotlib.collections as mcoll
    from matplotlib.patches import Rectangle

    for tgt in pcles.target.unique():

        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        # plot each individual pixel as a rectangular patch in a patch collection
        patches = []
        for idx, pcle in pcles.iterrows():

            ys, xs = np.nonzero(~(pcle.pdata.mask))

            for x,y in zip(xs,ys):
                x_um = pcle.pcle_xorig_um + x * pcle.x_step_nm /1.e3
                y_um = pcle.pcle_yorig_um - y * pcle.y_step_nm /1.e3
                rect = Rectangle((x_um, y_um), pcle.x_step_nm/1.e3, pcle.y_step_nm/1.e3, fill=True, linewidth=0, facecolor='k')
                patches.append(rect)

            if label is not None:
                if (label not in pcles.columns) and (label!='index'):
                    log.warning('specified label %s not found!' % label)
                    label = None
                else:
                    # plot at centre of bounding box - in microns!
                    bb_cent_x_um = pcle.pcle_xorig_um + pcle.bb_width/2.
                    bb_cent_y_um = pcle.pcle_yorig_um - pcle.bb_height/2.
                    if label=='index':
                        text = pcle.name
                    else:
                        text = pcle['%s' % label]
                    ax.text(bb_cent_x_um, bb_cent_y_um, text,
                        horizontalalignment='center', fontsize=labelfont, color='red', clip_on=True)

        collection = mcoll.PatchCollection(patches, match_original=True)
        ax.add_collection(collection)

    if show_scan:
        images = pcles.scan_file.unique().tolist()
        ros_tm.show_loc(images, facet=None, segment=None, tip=None, show_stripes=False, zoom_out=zoom_out,
            figure=fig, axis=ax, labels=False, title=False, font=None, interactive=False)

    if zoom_out:
        ax.set_xlim(-700.,700.)
        ax.set_ylim(-1400., 1400.)
    else: # scale plot to show all pcles
        xmin = pcles.pcle_xorig_um.min()
        xmax = (pcles.pcle_xorig_um + pcles.bb_width).max()
        ymin = (pcles.pcle_yorig_um - pcles.bb_height).min()
        ymax = pcles.pcle_yorig_um.max()
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)

    if show_stripes:
        for seg_off in range(-7,8):
            offset = common.seg_off_to_pos(seg_off)
            ax.axhspan(offset-50., offset+50., facecolor='g', alpha=0.2)

    if show_seg:
        tform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        centre_seg = pcles.target.unique()[0] * 16
        for seg_off in range(-7,8):
            offset = common.seg_off_to_pos(seg_off)
            print centre_seg, seg_off
            ax.text(1.0, offset, str(centre_seg + seg_off), fontsize='medium', color='b',
                transform=tform, ha='right', va='center', clip_on=True)

    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.set_xlabel('X position (microns)')
    ax.set_ylabel('Y position (microns)')

    plt.show()

    return ax


def exposure_summary(start='2014-08-06', print_stats=False, fontsize=14):
    """Produces a summary plot of all exposures and their targets, as well as printing
    some optional statistics."""

    from midas import spice_utils
    import matplotlib.patches as mpatch
    import matplotlib.dates as md

    start = pd.Timestamp(start)
    exposures = ros_tm.load_exposures().query('start>=@start')
    targets = sorted(exposures.target.unique())
    start = exposures.start.min()
    end = exposures.end.max()
    geom = spice_utils.get_geometry(start, end, no_ck=True)

    fig, ax = plt.subplots()
    ax.plot(geom.index, geom.sc_dist, 'k', linewidth=2)
    fig.autofmt_xdate()
    ax.set_ylabel('Altitude (km)', fontsize=fontsize)
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.grid(True)
    ax.tick_params(labelsize=fontsize)

    num_tgts = len(targets)
    # colours = cm.get_cmap('gist_rainbow')
    colours = cm.get_cmap('jet')
    colour_list = [colours(1.*i/num_tgts) for i in range(num_tgts)]

    index = 0
    for target in targets:
        for idx, exp in exposures[exposures.target==target].iterrows():
            ax.axvspan(exp.start, exp.end, alpha=0.5, color=colour_list[index])
        index += 1

    artists = [mpatch.Circle((0,0),fc=colour_list[idx], ec=colour_list[idx]) for idx in range(num_tgts)]
    fig.legend(artists, targets,title=u'Targets',loc="upper center", ncol=num_tgts, fontsize=14);

    plt.show()

    if print_stats:
        log.info('Fraction of time exposing: %3.2f%%' % (100.*(exposures.duration.sum()/(end-start)))  )

    return fig, ax

def usage_stats(start='2014-05-12 14:30:00', end=None):
    """Calculates the duration of all image scans and all exposures, and
    plots a pie chart of this and the remaining 'other' time"""

    images = ros_tm.load_images(data=False, topo_only=True, exclude_bad=False)
    exposures = ros_tm.load_exposures()
    start = pd.Timestamp(start)
    if end is None:
        end = pd.to_datetime('now')
    else:
        end = pd.Timestamp(end)

    total = end - start
    exp = exposures.query('start>@start & end<@end').duration.sum()
    scan = images.query('start_time>@start & end_time<@end').duration.sum()
    other = total - (exp + scan)

    exp_per = (exp/total)*100.
    scan_per = (scan/total)*100.
    other_per = 100. - (exp_per + scan_per)
    values = [exp, scan, other]

    fig, ax = plt.subplots()

    labels = 'scan', 'expose', 'other'
    sizes = [scan_per, exp_per, other_per]
    colors = ['yellowgreen', 'gold', 'lightskyblue'] # , 'lightcoral']
    explode = (0, 0.1, 0)

    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)#
    ax.set_aspect('equal')

    plt.show()

    return


def get_pcles(gwyfile, chan='particle'):
    """Reads channel data corresponding to chan= (or all, if None) and
    containing a mask."""

    from skimage import measure

    # Get all masked channels matching the particle filter
    channels = gwy_utils.list_chans(gwyfile, chan, masked=True, info=True, matchstart=True)
    if channels is None:
        log.error('no channels found matching: %s' % chan)
        return None

    # Get meta-data from topography channel
    meta = gwy_utils.get_meta(gwyfile, channel='Topography (Z piezo position set value)')
    if meta is None:
        log.error('no meta-data available')
        return None

    # Calculate the pixel area (simply x_step*y_step)
    pix_area = float(meta.x_step_nm)*1.e-9 * float(meta.y_step_nm)*1.e-9 #m2
    pix_len = np.sqrt(pix_area) #  in the case of non-uniform steps!

    pcle_data = []

    for idx, channel in channels.iterrows():

        # for each channel extract the mask and the actual data
        mask = gwy_utils.extract_masks(gwyfile, channel['name'])
        xlen, ylen, data = gwy_utils.get_data(gwyfile, channel['name'])

        labmask = measure.label(mask)
        regions = measure.regionprops(labmask)

        for region in regions:

            up, left, down, right = region.bbox
            parray = ma.masked_array(data[up:down, left:right], mask=~region.image)

            pcle = {
                'scan_file': meta.scan_file,
                'chan_name': channel['name'],
                'left': left, # bb
                'right': right,
                'up': up,
                'down': down,
                'centre_x': region.centroid[0],
                'centre_y': region.centroid[1],
                'a_pix': region.area,
                'a_pcle': region.area * pix_area,
                'x_offset_um': float(meta.x_step_nm)*1.e-3 * float(left),
                'y_offset_um': float(meta.y_step_nm)*1.e-3 * float(up), # offset from TOP of bb
                'r_eq': np.sqrt(region.area * pix_area / np.pi),
                'z_min': parray.min(),
                'z_max': parray.max(),
                'z_mean': parray.mean(),
                'major': region.major_axis_length * pix_len,
                'minor': region.minor_axis_length * pix_len,
                'eccen': region.eccentricity,
                'orient': np.degrees(region.orientation),
                'pdata': parray
                }
            pcle_data.append(pcle)

    pcle_data = pd.DataFrame.from_records(pcle_data)

    log.info('Gwyddion file %s processed with %d particles' % (gwyfile, len(pcle_data)))

    return pcle_data



def dbase_build(dbase='particles.msg', gwy_path='.', gwy_pattern='*.gwy', chan='particle'):
    """Accepts a path and file pattern matching a set of Gwyddion files
    containing particles marked with masks. This can either be one particle
    per channel, or a single masked channel showing all particles (when well
    separated).particles

    Required supplementary material can be added through the dbase_suppl()
    routine"""

    import cPickle as pkl

    gwy_files = sorted(glob.glob(os.path.join(gwy_path, gwy_pattern)))
    if len(gwy_files) == 0:
        log.error('no files matching pattern %s found in folder %s' % (gwy_pattern, gwy_path))

    for idx, gwyfile in enumerate(gwy_files):

        pcles = get_pcles(gwyfile, chan=chan)
        if pcles is None:
            log.warning('Gwyddion file %s contains no particles!' % gwyfile)
            continue

        if 'database' not in locals() and 'database' not in globals():
            database = pcles
        else:
            database = database.append(pcles)

    if os.path.exists(dbase):
        os.remove(dbase)

    pkl_f = open(dbase, 'wb')
    pkl.dump(database, file=pkl_f, protocol=pkl.HIGHEST_PROTOCOL)

    log.info('particle database created with %d particles' % len(database))

    return



def dbase_load(dbase='particles.msg', pcle_only=False):
    """Loads the particle dataframe and looks up anciliary information from other
    tables to return the fully searchable database"""

    import cPickle as pkl

    f = open(dbase, 'rb')

    objs = []
    while 1:
        try:
            objs.append(pkl.load(f))
        except EOFError:
            break

    if len(objs)==0:
        log.error('file %s appears to be empty' % dbase)
        return None

    pcles = pd.concat(iter(objs), axis=0)

    if pcle_only:
        return pcles

    # Load the image meta-data frame and merge key meta data for each particle
    images = ros_tm.load_images(data=False)
    cols = ['scan_file', 'tip_num', 'target', 'wheel_pos', 'scan_type', 'aborted',
        'x_orig_um', 'y_orig_um', 'x_step_nm', 'y_step_nm', 'xlen_um', 'ylen_um']
    images = images[ cols ]
    pcles = pcles.merge(images, how='left', on='scan_file')

    # calculate the offset in microns of the corner of each bounding box
    pcles['pcle_xorig_um'] = pcles.x_orig_um + pcles.x_offset_um
    pcles['pcle_yorig_um'] = pcles.y_orig_um + pcles.ylen_um - pcles.y_offset_um

    pcles['bb_width'] = (pcles.right-pcles.left)*pcles.x_step_nm/1.e3
    pcles['bb_height'] = (pcles.down-pcles.up)*pcles.y_step_nm/1.e3

    log.info('particle database restored with %d particles' % len(pcles))

    return pcles


def check_masks(gwyfile, show=True, interactive=False, pcle_types=['particle', 'sub_particle']):
    """This routine sanity-checks a Gwyddion file with applied masks. In particular
    it checks:
      -- how many particles and/or sub-particles are defined
      -- are the numbers of (sub)particles sequential?
      -- for individual channels, are their >1 masks?
      -- do any masks overlap?

    If all checks are passed, True is returned, otherwise False"""

    from skimage import measure

    if interactive:

        show = True

        # make a backup copy of gwyfile before any changes
        import shutil
        gwyfile_bak = gwyfile + '.bak'
        shutil.copyfile(gwyfile, gwyfile_bak)

    if show:

        from matplotlib.colors import LinearSegmentedColormap
        import matplotlib.patches as mpatch

        cmap_red = [ (1, 1, 1), (1, 0, 0) ]
        cmap_blue = [ (1, 1, 1), (0, 0, 1) ]
        cmap_green = [ (1, 1, 1), (0, 1, 0) ]
        ncols = 2
        cmap_red_name = 'binary_red'
        cmap_blue_name = 'binary_blue'
        cmap_green_name = 'binary_green'

        cm_red = LinearSegmentedColormap.from_list(cmap_red_name, cmap_red, N=ncols)
        cm_blue = LinearSegmentedColormap.from_list(cmap_blue_name, cmap_blue, N=ncols)
        cm_green = LinearSegmentedColormap.from_list(cmap_green_name, cmap_green, N=ncols)

    for pcle_type in pcle_types:

        return_val = True

        channels = gwy_utils.list_chans(gwyfile, filter=pcle_type, masked=True, matchstart=True)
        if channels is None:
            log.warn('no matching channels found for particle type: %s' % pcle_type)
            continue
        else:
            channels = channels.values()

        num_pcles = len(channels)

        log.info('%d %ss found' % (num_pcles, pcle_type))

        if num_pcles > 0:
            pcle_id = sorted([int(d.split('_')[-1]) for d in channels])
            if min(pcle_id) != 1:
                log.error('particle numbering must start with 1, not %d' % min(pcle_id))
                return False
            diffs = [j - i for i, j in enumerate(pcle_id)]
            if set(diffs) != {1}:
                log.error('particle number must be sequential, not: %d, %d' % (pcle_id[diffs.index(2)-1], pcle_id[diffs.index(2)]) )
                return False
            if max(pcle_id) != num_pcles:
                log.error('particle numbering must end with %d, not %d' % (num_pcles, max(pcle_id)))
                return False

            log.info('channel numbering of %ss successfully validated' % pcle_type)

            masks = gwy_utils.extract_masks(gwyfile, channel=pcle_type, match_start=True)

            # check for more than one mask per chanmnel
            for idx, mask in enumerate(masks):
                labmask = measure.label(mask)
                regions = measure.regionprops(labmask)
                if len(regions) > 1:
                    log.error('more than one distinct mask found in chamnnl %s' % channels[idx] )

            # check for overlaps between particles
            for idx1 in range(len(masks)):
                for idx2 in range(idx1+1, len(masks)):
                    log.debug('comparing channels %s and %s (indices %d/%d)' % (channels[idx1], channels[idx2], idx1, idx2))
                    pix_overlap = np.logical_and(masks[idx1], masks[idx2])
                    if  pix_overlap.sum() > 0:
                        return_val = False
                        log.warning('%d overlapping pixels detected in channels %s and %s' % (
                            pix_overlap.sum(), channels[idx1],channels[idx2]))

                        if show:

                            fig, ax = plt.subplots()

                            # create three plots, one for each channel and one for the overlap
                            ax.imshow(masks[idx1], interpolation='nearest', cmap=cm_red, vmin=0, vmax=1)
                            ax.imshow(ma.masked_equal(masks[idx2], False), interpolation='nearest', cmap=cm_blue, vmin=0, vmax=1)
                            ax.imshow(ma.masked_equal(pix_overlap, False), interpolation='nearest', cmap=cm_green, vmin=0, vmax=1)
                            ax.grid(True)

                            # draw patches to create a legend
                            redpatch = mpatch.Circle((0,0),fc='red', ec='red')
                            bluepatch = mpatch.Circle((0,0),fc='blue', ec='blue')
                            legend = ax.legend([redpatch,bluepatch], ['1: %s' % channels[idx1], '2: %s' % channels[idx2]],
                                loc='upper right', bbox_to_anchor=(1.0, 1.0))
                            fig.suptitle('%s\n%d pixels overlap' % (gwyfile, pix_overlap.sum()))

                            if interactive:

                                def pressed(event):
                                    if event.key=='1':
                                        selected = idx2
                                    elif event.key=='2':
                                        selected = idx1
                                    elif event.key=='q':
                                        # skip this pair of channels
                                        plt.close(fig)
                                        log.warning('channel skipped - no action taken')
                                        return
                                    else:
                                        return

                                    log.info('overlapping pixels will be removed from channel %s' % channels[selected])

                                    # call function to update Gwyddion file!
                                    plt.close(fig)
                                    new_mask = masks[selected] - pix_overlap
                                    gwy_utils.add_mask(gwyfile, new_mask, channels[selected], overwrite=True)

                                fig.canvas.mpl_connect('key_press_event', pressed)
                                plt.show(block=True)


    if show and not interactive:
        plt.show()

    if return_val:
        log.info('file successfully validated!')

    return return_val


def plot_geom(start_time, end_time, exposures=None, fontsize=12, savefig=None, title=''):

    import matplotlib.dates as md
    from midas import spice_utils
    import matplotlib.ticker as ticker

    geom = spice_utils.get_geometry(start=start_time, end=end_time)

    # Plot various geometric parameters over time and shade the exposures
    xfmt = md.DateFormatter('%Y-%m-%d') # %H:%M:%S')
    geom_plot = plt.figure()

    ax3 = geom_plot.add_subplot(3, 1, 3)
    ax2 = geom_plot.add_subplot(3, 1, 2, sharex=ax3)
    ax1 = geom_plot.add_subplot(3, 1, 1, sharex=ax2)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=True)

    # Distance and off-pointing
    dist_line = ax1.plot(geom.index, geom.sc_dist, label='comet-s/c distance')
    ax1.set_ylabel('Spacecraft distance (km)', fontsize=fontsize)
    ax1r = ax1.twinx()
    angle_line = ax1r.plot(geom.index, geom.offnadir, 'r', label='off-nadir')
    ax1r.set_ylabel('Off-nadir angle (deg)', fontsize=fontsize)
    ax1r.set_ylim(0,16)

    lines = dist_line + angle_line
    labs = [l.get_label() for l in lines]
    leg1 = ax1r.legend(lines, labs, loc='upper left', fancybox=True, fontsize=fontsize, framealpha=1.0)
    leg1.set_zorder(20)

    # Latitude and longitude
    lat_line = ax2.plot(geom.index, geom.latitude, label='latitude')
    ax2.set_ylabel('Sub-spacecraft latitude', fontsize=fontsize)
    ax2r = ax2.twinx()
    lon_line = ax2r.plot(geom.index, geom.longitude, 'r', label='longitude')
    ax2r.set_ylabel('Sub-spacecraft longitude', fontsize=fontsize)
    ax2r.set_ylim(-180.,180.)
    ax2r.yaxis.set_major_locator(ticker.MultipleLocator(60))


    lines = lat_line + lon_line
    labs = [l.get_label() for l in lines]
    leg2 = ax2r.legend(lines, labs, loc='upper left', fancybox=True, framealpha=1.0, fontsize=fontsize)

    # Comet distance
    ax3.xaxis.set_major_formatter(xfmt)
    ax3.yaxis.get_major_formatter().set_useOffset(False)

    comet_sun_dist = ax3.plot(geom.index, geom.cometdist, label='comet-sun distance')
    ax3.set_ylabel('Comet-sun distance (au)', fontsize=fontsize)

    geom_plot.subplots_adjust(wspace=0)
    geom_plot.autofmt_xdate()

    if exposures is not None:

        for idx,exposure in exposures.iterrows():
            ax1.axvspan(exposure.start,exposure.end, facecolor='Silver', alpha=0.2)
            ax2.axvspan(exposure.start,exposure.end, facecolor='Silver', alpha=0.2)
            ax3.axvspan(exposure.start,exposure.end, facecolor='Silver', alpha=0.2)

    ax1.set_title(title, fontsize=fontsize)
    ax1.tick_params(labelsize=fontsize)
    ax1r.tick_params(labelsize=fontsize)
    ax2.tick_params(labelsize=fontsize)
    ax2r.tick_params(labelsize=fontsize)
    ax3.tick_params(labelsize=fontsize)

    ax3.set_xlim(start_time, end_time)

    if savefig:
        geom_plot.savefig(savefig, format='eps', bbox_inches='tight')
        log.info('saving file %s' % savefig)
        plt.close(geom_plot)
    else:
        plt.show()
        return geom_plot

    return



#------ test routines

# def plot_pcles2(pcles, figure=None, axis=None, show_stripes=True, zoom_out=False):
#     """Plot particles in the passed data frame on their respective facets."""
#
#     for tgt in pcles.target.unique():
#
#         fig, ax = plt.subplots()
#         ax.get_yaxis().set_visible(False)
#         ax.get_xaxis().set_visible(False)
#
#         # find the smallest pixel size and create an array with this as unit size
#         # then scale other sizes accordingly - note that this may result in cases
#         # where the sizes are not precisely proportional!
#
#         smallest = min(pcles.x_step_nm.min(), pcles.y_step_nm.min())/1.e3
#         xmin = pcles.pcle_xorig_um.min()
#         xmax = (pcles.pcle_xorig_um + pcles.bb_width).max()
#         ymin = pcles.pcle_yorig_um.min()
#         ymax = (pcles.pcle_yorig_um + pcles.bb_height).max()
#
#         new_array = np.zeros( (pcles.down.max()+1, pcles.right.max()+1), dtype=np.float64)
#
#         for idx, pcle in pcles.iterrows():
#
#             new_array[pcle.up:pcle.down+1, pcle.left:pcle.right+1][~pcle['pdata'].mask] = pcle['pdata'][ ~pcle['pdata'].mask ]
#
#         cmap = cm.afmhot
#         cmap.set_bad('c', alpha=1.0)
#
#     im = ax.imshow(new_array, cmap=cmap, interpolation='nearest', vmin=0, vmax=new_array.max())
#
#     plt.show()
#
#     return im
#
# def plot_pcles3(pcles, figure=None, axis=None, show_stripes=True, zoom_out=False):
#     """Plot particles in the passed data frame on their respective facets."""
#
#     import matplotlib.collections as mcoll
#     from matplotlib.patches import Rectangle
#
#     for tgt in pcles.target.unique():
#
#         fig, ax = plt.subplots()
#         ax.set_aspect('equal')
#
#         # find the vertices of a polygon describing each particle mask
#         # then add as an mpl polygon
#
#         patches = []
#         for idx, pcle in pcles.iterrows():
#
#
#
#
#
#             xs, ys = np.nonzero(~pcle.pdata.mask)
#             for x,y in zip(xs,ys):
#                 x_um = pcle.pcle_xorig_um + x * pcle.x_step_nm /1.e3
#                 y_um = pcle.pcle_yorig_um + y * pcle.y_step_nm /1.e3
#                 rect = Rectangle((x_um, y_um), pcle.x_step_nm/1.e3, pcle.y_step_nm/1.e3, fill=True, linewidth=0, facecolor='k')
#                 patches.append(rect)
#
#         collection = mcoll.PatchCollection(patches, match_original=True)
#         ax.add_collection(collection)
#
#         # scale plot to show all pcles
#         xmin = pcles.pcle_xorig_um.min()
#         xmax = (pcles.pcle_xorig_um + pcles.bb_width).max()
#         ymin = pcles.pcle_yorig_um.min()
#         ymax = (pcles.pcle_yorig_um + pcles.bb_height).max()
#         ax.set_xlim(xmin,xmax)
#         ax.set_ylim(ymin,ymax)
#
#     plt.show()
#
#     return
