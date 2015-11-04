
#!/usr/bin/python
"""gwyutils.py

Mark S. Bentley (mark@lunartech.org), 2015

A module containing routines to manipulate Gwyddion (.gwy) files"""

import gwy, gwyutils, re
import pandas as pd
import numpy as np


def extract_masks(gwy_file, channel=None):
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
        data = np.array(datafield.get_data(), dtype=np.bool).reshape(
            datafield.get_yres(), datafield.get_xres())
        mask_data.append(data)

    if len(mask_data)>1:
        print('INFO: %d masks extracted from Gwyddion file %s' %
          (len(mask_data), gwy_file))

    return mask_data


def list_chans(gwy_file, filter=None, masked=False):
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

    unit = datafield.get_si_unit_xy()
    xlen = datafield.get_xreal()
    ylen = datafield.get_yreal()

    data = np.array(datafield.get_data(), dtype=np.float32).reshape(
        datafield.get_yres(), datafield.get_xres())

    return xlen, ylen, data



def get_meta(gwyfile):
    """Returns all meta-data from a Gwyddion file"""

    C = gwy.gwy_file_load(gwyfile, gwy.RUN_NONINTERACTIVE)

    metadata = {}

    meta = C.get_value_by_name('/0/meta')
    keys = meta.keys_by_name()

    for key in keys:
        metadata.update({ key: meta.get_value_by_name(key) })

    return pd.Series(metadata) #.convert_objects(convert_numeric=True, convert_dates=True)
