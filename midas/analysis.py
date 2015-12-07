
#!/usr/bin/python
"""analysis.py

Mark S. Bentley (mark@lunartech.org), 2015

A module containing various routines related to the analysis of MIDAS data,
including investigating particle statistics, exposure geometries etc."""

import os, math
import pandas as pd
from midas import common, ros_tm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.ma as ma
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

    if query is not None:
        images = images.query(query)

    if image is not None:
        if type(image) == str:
            image = [image]
        target = images[images.scan_file.isin(image)]
    else:
        target = images

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
        over['left'] = over['left'].astype(np.int64)
        over['right'] = over['right'].astype(np.int64)

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
        over.sort_values(by='area', inplace=True)

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

    import glob
    files = sorted(glob.glob(os.path.join(path, basename+'*')))


    if len(files)==0:
        print('ERROR: no files matching pattern: %s' % basename)
        return None

    with open(files[0], 'r') as f:
        header = f.readline()
        cols = header.split()[1:]

    grain_stats = pd.DataFrame([], columns=cols)

    for grainfile in files:
        grain_stats = grain_stats.append(pd.read_table(grainfile, header=None, skiprows=1, names=cols))

    return grain_stats


def get_subpcles(gwyfile, chan='sub_particle_'):
    """Reads channel data corresponding to chan= (or all, if None) and
    containing a mask."""

    from midas import gwy_utils
    from skimage import measure

    # Get all masked channels matching the sub-particle filter
    channels = gwy_utils.list_chans(gwyfile, chan, masked=True)
    if len(channels)==0:
        print('ERROR: no channels found matching: %s' % chan)
        return None

    # Get meta-data from the first channel of the GWY file
    meta = gwy_utils.get_meta(gwyfile)

    # Calculate the pixel area (simply x_step*y_step)
    pix_area = float(meta.x_step_nm)*1.e-9 * float(meta.y_step_nm)*1.e-9 #m2
    pix_len = np.sqrt(pix_area) #  in the case of non-uniform steps!

    pcle_data = []

    for chan_id, chan_name in channels.items():

        mask = gwy_utils.extract_masks(gwyfile, chan_name)
        xlen, ylen, data = gwy_utils.get_data(gwyfile, chan_name)

        locs = np.where(mask)
        left, right = locs[1].min(), locs[1].max()
        up, down = locs[0].min(), locs[0].max()
        pcle = ma.masked_array(data, mask=~mask)
        pcle = pcle[up:down+1,left:right+1]

        labmask = measure.label(mask)
        regions = measure.regionprops(labmask)
        region = regions[0]

        if pcle.count() != int(region.area):
            print('WARNING: region count does not match mask count')

        subpcle = {

            'pcle': pcle,
            'id': chan_id,
            'name': chan_name,
            'left': left,
            'right': right,
            'up': up,
            'down': down,
            'tot_min_z': data.min()*1.e6,
            'tot_max_z': data.max()*1.e6,
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
            'sphericity': min( (region.major_axis_length/region.minor_axis_length),
                (pcle.max()/(region.major_axis_length*pix_len)),
                (pcle.max()/(region.minor_axis_length*pix_len)) ),
            'orient': np.degrees(region.orientation),
            'pdata': pcle
            }
        pcle_data.append(subpcle)

    pcle_data = pd.DataFrame.from_records(pcle_data)
    pcle_data.sort_values(by='id', inplace=True)
    pcle_data['tot_z_diff'] = pcle_data.tot_max_z - pcle_data.tot_min_z
    pcle_data['z_diff'] = pcle_data.z_max - pcle_data.z_min

    pcle_data = pcle_data[ ['id', 'name', 'left', 'right', 'up', 'down', 'tot_min_z', 'tot_max_z', 'tot_z_diff', 'a_pix', 'a_pcle', 'r_eq',
        'z_min', 'z_max', 'z_mean', 'z_diff', 'major', 'minor', 'eccen', 'compact', 'sphericity', 'orient', 'pdata'] ]

    return pcle_data


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
        delta_x = num_frames * pix_per_frame * pcle_data.pos_x.ix[idx]
        delta_y = num_frames * pix_per_frame * pcle_data.pos_y.ix[idx]
        pcle.up += delta_y
        pcle.down += delta_y
        pcle.left += delta_x
        pcle.right += delta_x

        new_data.ix[idx] = pcle

    max_x = int(round(max(abs(new_data.right.max()-cent[0]),abs(new_data.left.min()-cent[0]))))
    max_y = int(round(max(abs(new_data.up.max()-cent[1]),abs(new_data.down.min()-cent[1]))))

    for frame_cnt in range(num_frames):

        new_data = orig_data.copy()

        for idx, pcle in new_data.iterrows():

            # calculate the offset according to the unit vector (direction)
            # and speed (given in pixels per frame)
            delta_x = frame_cnt * pix_per_frame * pcle_data.pos_x.ix[idx]
            delta_y = frame_cnt * pix_per_frame * pcle_data.pos_y.ix[idx]

            # apply the offset to the bounding box (used for plotting the sub-pcle)
            pcle.up += delta_y
            pcle.down += delta_y
            pcle.left += delta_x
            pcle.right += delta_x

            new_data.ix[idx] = pcle

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
                print('WARNING: label %s not valid for the title' % title)
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
