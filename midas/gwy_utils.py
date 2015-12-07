
#!/usr/bin/python
"""gwyutils.py

Mark S. Bentley (mark@lunartech.org), 2015

A module containing routines to manipulate Gwyddion (.gwy) files"""

import gwy, gwyutils, re
import pandas as pd
import numpy as np

def extract_masks(gwy_file, channel=None, return_df=False):
    """Looks for mask channels within the Gwyddion file and returns these
    masks as a list of numpy arrays. If channel= is given a string, only
    channels with names containing this string are returned."""

    C = gwy.gwy_file_load(gwy_file, gwy.RUN_NONINTERACTIVE)
    keys = zip(C.keys(), C.keys_by_name())

    # Use regex to locate all channels containing masks
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
                matching.append(mask)

        if len(matching) == 0:
            print('WARNING: No masks in channels matching "%s" found' %
                  (channel))
            return None

    else:
        matching = masks

    # Now get the mask data for each (requested) channel
    mask_data = []
    for chan in matching:

        datafield = C.get_value_by_name('/%d/mask' % chan)
        if return_df:
            mask_data.append(datafield)
        else:
            data = np.array(datafield.get_data(), dtype=np.bool).reshape(
                datafield.get_yres(), datafield.get_xres())
            mask_data.append(data)

    if len(mask_data)>1:
        print('INFO: %d masks extracted from Gwyddion file %s' %
          (len(mask_data), gwy_file))

    if len(mask_data)==1:
        mask_data = mask_data[0]

    return mask_data


def list_chans(gwy_file, filter=None, masked=False, info=False):
    """Lists the channels in a Gwyddion file.

    If filter is given, only channels whose names contain this substring will be returned.

    If masked=True only channels contaning a mask will be returned."""

    C = gwy.gwy_file_load(gwy_file, gwy.RUN_NONINTERACTIVE)
    keys = zip(C.keys(), C.keys_by_name())

    channels = {}
    for id, key in keys:
        m = re.match(r'^/(?P<i>\d+)/data$', key)
        if not m:
            continue
        channel = int(m.group('i'))
        channels.update({channel: C.get_value_by_name('/%d/data/title' % channel) })

    # If requested, find only those channels contaning substring "channel"
    #
    if filter is None:
        matching = channels
    else:
        for chan_num, chan_name in channels.items():
            if filter not in chan_name:
                del(channels[chan_num])

    # Also check if they contain masks
    if masked:
        for chan_num in channels.keys():
            if not C.contains_by_name('/%d/mask' % chan_num):
                del(channels[chan_num])

    if info:
        # If additional information is requested, use the selected channels
        # and retrieve the datafield and associated parameters
        channel_data = pd.DataFrame(columns=['id', 'name', 'xlen_um', 'ylen_um', 'xpix', 'ypix'])
        for idx, chan_num in enumerate(channels.keys()):
            datafield = C.get_value_by_name('/%d/data' % chan_num)
            channel_data.loc[idx] = pd.Series({
                'id': chan_num,
                'name': channels[chan_num],
                'xlen_um': 1.e6 * datafield.get_xreal(),
                'ylen_um': 1.e6 * datafield.get_yreal(),
                'xpix': datafield.get_xres(),
                'ypix': datafield.get_yres() })
        channel_data['x_step_nm'] = 1.e3 * channel_data.xlen_um / channel_data.xpix
        channel_data['y_step_nm'] = 1.e3 * channel_data.ylen_um / channel_data.ypix
        channels = channel_data

    return channels


def add_mask(gwy_file, mask, chan_name):
    """Creates a duplicate of channel chan_name, removes any mask present
    and creates a new mask with the passed boolean array if the
    dimensions match"""

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

    selected = None

    for channel in channels:
        name = C.get_value_by_name('/%d/data/title' % channel)
        if name==chan_name:
            selected = channel
            break
    if selected is None:
        print('WARNING: channel %s not found!' % chan_name)
        return None

    datafield = C.get_value_by_name('/%d/data' % selected)

    xpix = datafield.get_xres()
    ypix = datafield.get_yres()

    if mask.shape != (xpix, ypix):
        print('ERROR: mask shape (%d,%d) does not match image shape (%d,%d)' % (mask.shape[0], mask.shape[1], xpix, ypix))

    new_idx = max(channels) + 1

    mask_chan = datafield.duplicate()
    m = gwyutils.data_field_data_as_array(mask_chan)
    m[:] = mask.copy()
    C.set_object_by_name('/%i/data' % (new_idx), datafield)
    C.set_object_by_name('/%i/mask' % (new_idx), mask_chan)
    C.set_string_by_name('/%i/data/title' % (new_idx),'masked_channel')

    gwy.gwy_file_save(C, os.path.splitext(gwy_file)[0]+'_new.gwy', gwy.RUN_NONINTERACTIVE)

    return


def get_data(gwy_file, chan_name=None):
    """Returns data from a Gwyddion file with channel matching channel=, or
    the first channel if channel=None."""

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
        return None, None, None

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
            return None, None, None

    datafield = C.get_value_by_name('/%d/data' % selected)

    unit = datafield.get_si_unit_xy()
    xlen = datafield.get_xreal()
    ylen = datafield.get_yreal()

    data = np.array(datafield.get_data(), dtype=np.float32).reshape(
        datafield.get_yres(), datafield.get_xres())

    return xlen, ylen, data



def get_meta(gwyfile):
    """Returns all meta-data from a Gwyddion file. Note that all returned data
    will be in strings, and the user must ensure correct type conversion!"""

    C = gwy.gwy_file_load(gwyfile, gwy.RUN_NONINTERACTIVE)

    metadata = {}

    meta = C.get_value_by_name('/0/meta')
    keys = meta.keys_by_name()

    for key in keys:
        metadata.update({ key: meta.get_value_by_name(key) })

    return pd.Series(metadata) #.convert_objects(convert_numeric=True, convert_dates=True)


def gwy_to_bcr(gwyfile, channel, bcrfile=None):
    """Converts a given channel in a Gywddion .gwy file to a BCR. If
    bcrfile=None, the Gwyddion filename will be used and the extension
    changed."""

    import bcrutils, common

    # Read a given channel from GWY file
    xlen, ylen, data = get_data(gwyfile, chan_name=channel)

    if data is None:
        print('ERROR: no valid data found for channel %s in GWY file %s' % (
            channel, gwyfile) )
        return None

    bcrdata = {}
    bcrdata['filename'] = bcrfile if bcrfile is not None else gwyfile.replace('.gwy', '.bcr')

    bcrdata['xpixels'] = data.shape[1]
    bcrdata['ypixels'] = data.shape[0]

    bcrdata['xlength'] = xlen
    bcrdata['ylength'] = ylen

    bcrdata['bit2nm'] = common.zcal

    bcrdata['data'] = np.array(data*1.e9/common.zcal, dtype=np.int32).ravel()

    bcrutils.write(bcrdata)

    return bcrdata


    # Set minimal data to default values if not present
    if not 'xlength' in bcrdata: bcrdata['xlength'] = bcrdata['xpixels']
    if not 'ylength' in bcrdata: bcrdata['ylength'] = bcrdata['ypixels']


def get_grain_data(gwy_file, chan_name=None, datatype=None):
    """Extract grain data from the specificed channels in a Gwyddion file.
    Channels can be filtered via chan_name=, otherwise all channels containin
    a mask will be used. All grain properties are return unless datatype= is
    given as a list of data types."""

    channels = list_chans(gwy_file, filter=chan_name, masked=True)
    if len(channels)==0:
        print('ERROR: no channels matching %s in file %s' % (chan_name, gwy_file))
        return None

    masks = extract_masks(gwy_file, channel=chan_name, return_df=True)

    if datatype is not None:
        if type(datatype) != list:
            datatype = [datatype]

    # Obtain a list of all grain data types available in Gwyddion and tidy up
    grain_types = [item for item in dir(gwy) if item.startswith('GRAIN_VALUE')]
    grain_types = [item.split('GRAIN_VALUE_')[1] for item in grain_types]

    # If a datatype list is given, filter to requested types, if valid
    if datatype is not None:
        selected = [item for item in grain_types if item in datatype]
        if len(selected)==0 and datatype is not None:
            print('WARNING: cannot find specified grain data type, defaulting to all')
            datatype = None
            selected = grain_types
    else:
        selected = grain_types

    graindata = pd.DataFrame([], columns=selected)

    # For each channel, loop through the masks and extract named properties into a df
    for idx, (name, mask) in enumerate(zip(channels.values(), masks)):
        num_grains = mask.number_grains()
        row_data = {}
        row_data['channel'] = name
        for sel in selected:
            property = getattr(gwy, 'GRAIN_VALUE_'+sel)
            row_data[sel] = mask.grains_get_distribution(mask, num_grains, property, -1).get_real()
        graindata = graindata.append(row_data, ignore_index=True)

    # rename column names to be lower case
    graindata.columns = map(str.lower, graindata.columns)

    return graindata
