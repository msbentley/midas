
#!/usr/bin/python
"""gwyutils.py

Mark S. Bentley (mark@lunartech.org), 2015

A module containing routines to manipulate Gwyddion (.gwy) files"""

import gwy, gwyutils, re
import pandas as pd
import numpy as np
import os

import logging
log = logging.getLogger(__name__)

def extract_masks(gwy_file, channel=None, return_datafield=False):
    """Looks for mask channels within the Gwyddion file and returns these
    masks as a list of numpy arrays. If channel= is given a string, only
    channels with names containing this string are returned."""

    if not os.path.isfile(gwy_file):
        log.error('Gwyddion file %s does not exist!' % gwy_file)
        return None

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
        log.warning('No masks found in Gwyddion file %s' % gwy_file)
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
            log.warning('No masks in channels matching "%s" found' %
                  (channel))
            return None

    else:
        matching = masks

    # Now get the mask data for each (requested) channel
    mask_data = []
    for chan in matching:

        datafield = C.get_value_by_name('/%d/mask' % chan)
        if return_datafield:
            mask_data.append(datafield)
        else:
            data = np.array(datafield.get_data(), dtype=np.bool).reshape(
                datafield.get_yres(), datafield.get_xres())
            mask_data.append(data)

    if len(mask_data)>1:
        log.info('%d masks extracted from Gwyddion file %s' %
          (len(mask_data), gwy_file))

    if len(mask_data)==1:
        mask_data = mask_data[0]

    return mask_data


def list_chans(gwy_file, filter=None, getlog=False, masked=False, info=False, matchstart=False):
    """Lists the data channels in a Gwyddion file.

    If filter is given, only channels whose names contain this substring will be returned.

    If getlog is True, only data channels containing log information will be returned.

    If masked=True, channels containing a mask will be returned instead."""

    if getlog:
        masked = False

    if not os.path.isfile(gwy_file):
        log.error('Gwyddion file %s does not exist!' % gwy_file)
        return None

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
            if matchstart:
                if not chan_name.startswith(filter):
                    del(channels[chan_num])
            else:
                if filter not in chan_name:
                    del(channels[chan_num])

    # Also check if they contain log data
    if getlog:
        for chan_num in channels.keys():
            if not C.contains_by_name('/%d/data/log' % chan_num):
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


def add_mask(gwy_file, out_file, mask, chan_name):
    """Creates a duplicate of channel chan_name, and adds a new mask with the
    passed boolean array if the dimensions match.

    If out_file is None, a copy will be produced with the name _new
    appended."""

    if not os.path.isfile(gwy_file):
        log.error('Gwyddion file %s does not exist!' % gwy_file)
        return None

    C = gwy.gwy_file_load(gwy_file, gwy.RUN_NONINTERACTIVE)
    keys = zip(C.keys(), C.keys_by_name())

    channels = []
    for key in C.keys_by_name():
        m = re.match(r'^/(?P<i>\d+)/data$', key)
        if not m:
            continue
        channels.append(int(m.group('i')))
    channels = sorted(channels)

    if channels is None:
        log.warning('No data channels found in Gwyddion file %s' % gwy_file)
        return None

    selected = None

    for channel in channels:
        name = C.get_value_by_name('/%d/data/title' % channel)
        if name==chan_name:
            selected = channel
            break

    if selected is None:
        log.warning('channel %s not found!' % chan_name)
        return None

    datafield = C.get_value_by_name('/%d/data' % selected)

    xpix = datafield.get_xres()
    ypix = datafield.get_yres()

    if mask.shape != (xpix, ypix):
        log.error('mask shape (%d,%d) does not match image shape (%d,%d)' % (mask.shape[0], mask.shape[1], xpix, ypix))

    new_idx = max(channels) + 1

    mask_chan = datafield.duplicate()
    m = gwyutils.data_field_data_as_array(mask_chan)
    m[:] = mask.copy()
    C.set_object_by_name('/%i/data' % (new_idx), datafield)
    C.set_object_by_name('/%i/mask' % (new_idx), mask_chan)
    C.set_string_by_name('/%i/data/title' % (new_idx), chan_name)

    if out_file is None:
        output_name = os.path.splitext(gwy_file)[0]+'_new.gwy'
    else:
        output_name = os.path.join(os.path.basename(gwy_file), out_file)

    gwy.gwy_file_save(C, os.path.splitext(gwy_file)[0]+'_new.gwy', gwy.RUN_NONINTERACTIVE)

    log.info('Gwyddion file %s written with mask in channel %s' % (output_name, chan_name))

    return


def get_data(gwy_file, chan_name=None):
    """Returns data from a Gwyddion file with channel matching channel=, or
    the first channel if channel=None."""

    if not os.path.isfile(gwy_file):
        log.error('Gwyddion file %s does not exist!' % gwy_file)
        return None

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
        log.warning('No data channels found in Gwyddion file %s' % gwy_file)
        return None, None, None

    if chan_name is None:
        log.info('No channel specified, using %s' % C.get_value_by_name('/%d/data/title' % channels[0]))
        selected = channels[0]
    else:
        selected = None
        for channel in channels:
            name = C.get_value_by_name('/%d/data/title' % channel)
            if name==chan_name:
                selected = channel
                break
        if selected is None:
            log.warning('channel not found!')
            return None, None, None

    datafield = C.get_value_by_name('/%d/data' % selected)

    unit = datafield.get_si_unit_xy()
    xlen = datafield.get_xreal()
    ylen = datafield.get_yreal()

    data = np.array(datafield.get_data(), dtype=np.float32).reshape(
        datafield.get_yres(), datafield.get_xres())

    return xlen, ylen, data


def get_meta(gwyfile, channel=None):
    """Returns all meta-data from a Gwyddion file. Note that all returned data
    will be in strings, and the user must ensure correct type conversion!

    If channel=None, channel IDs will be checked (in numerical order) and
    meta data for the first channel will be returned."""

    if not os.path.isfile(gwyfile):
        log.error('Gwyddion file %s does not exist!' % gwyfile)
        return None

    C = gwy.gwy_file_load(gwyfile, gwy.RUN_NONINTERACTIVE)

    channels = list_chans(gwyfile, filter=channel)

    selected = None

    if channels is None:
        log.error('no channels matching %s in file %s' % (channel, gwyfile))
        return None
    elif len(channels)>1 and channel is not None:
        log.error('more than one channel matching %s in file %s' % (channel, gwyfile))
        return None
    elif channel is None: # find first channel
        for key in sorted(channels.keys()):
            if not C.contains_by_name('/%d/meta' % key):
                continue
            else:
                selected = key
                break
    else:
        selected = channels.keys()[0]

    metadata = {}

    if not C.contains_by_name('/%d/meta' % selected):
        log.error('channel contains no meta data')
        return None

    meta = C.get_value_by_name('/%d/meta' % selected)
    keys = meta.keys_by_name()

    for key in keys:
        metadata.update({ key: meta.get_value_by_name(key) })

    return pd.Series(metadata) #.convert_objects(convert_numeric=True, convert_dates=True)


def write_meta(gwyfile, metadata, channel=None):
    """Writes a pandas Series of meta-data to a Gwyddion file. If channel=None
    meta data are written to the first channel, otherwise to the
    named channel."""

    if not os.path.isfile(gwyfile):
        log.error('Gwyddion file %s does not exist!' % gwyfile)
        return None

    C = gwy.gwy_file_load(gwyfile, gwy.RUN_NONINTERACTIVE)

    channels = list_chans(gwyfile, filter=channel)
    selected = None

    if channels is None and channel is not None:
        log.error('no channels matching %s in file %s' % (channel, gwyfile))
        return None
    elif len(channels)>1 and channel is not None:
        log.error('more than one channel matching %s in file %s' % (channel, gwyfile))
        return None
    elif channel is None: # find first channel
        selected = min(channels.keys())
    else:
        selected = channels.keys()[0]

    # Meta channel returns a container, which itself has each meta
    # point as a key/value set

    meta = gwy.Container()
    for key in metadata.keys():
        meta.set_string_by_name(key, str(metadata[key]))

    C.set_object_by_name('/%d/meta' % selected, meta)

    gwy.gwy_file_save(C, gwyfile, gwy.RUN_NONINTERACTIVE)

    return


def rename_channels(gwyfile, search, replace):
    """Accepts a search string and looks for all channels
    in gwyfile containing this string. Replaces this substring
    with the value given in replace"""

    if not os.path.isfile(gwyfile):
        log.error('Gwyddion file %s does not exist!' % gwyfile)
        return None

    channels = list_chans(gwyfile, filter=search)

    if channels is None:
        log.error('no channels matching %s in file %s' % (search, gwyfile))
        return None

    C = gwy.gwy_file_load(gwyfile, gwy.RUN_NONINTERACTIVE)
    for key in channels:
        C.set_string_by_name('/%i/data/title' % key, channels[key].replace(search, replace))
    gwy.gwy_file_save(C, gwyfile, gwy.RUN_NONINTERACTIVE)

    return



def gwy_to_bcr(gwy_file, channel, bcrfile=None):
    """Converts a given channel in a Gywddion .gwy file to a BCR. If
    bcrfile=None, the Gwyddion filename will be used and the extension
    changed."""

    import bcrutils, common

    # Read a given channel from GWY file
    xlen, ylen, data = get_data(gwy_file, chan_name=channel)

    if data is None:
        log.error('no valid data found for channel %s in GWY file %s' % (
            channel, gwy_file) )
        return None

    bcrdata = {}
    bcrdata['filename'] = bcrfile if bcrfile is not None else gwy_file.replace('.gwy', '.bcr')

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

    if channels is None:
        return None

    if len(channels)==0:
        log.error('no channels matching %s in file %s' % (chan_name, gwy_file))
        return None

    masks = extract_masks(gwy_file, channel=chan_name, return_datafield=True)

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
            log.warning('cannot find specified grain data type, defaulting to all')
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


def read_log(gwy_file, channel=None):
    """Reads a Gwyddion log file"""

    channels = list_chans(gwy_file, filter=channel, log=True, masked=False)

    if channels is None:
        return None

    if channels is None:
        log.error('no channels matching %s in file %s' % (chan_name, gwy_file))
        return None

    C = gwy.gwy_file_load(gwy_file,gwy.RUN_NONINTERACTIVE)

    def logparser(entry):
    # type::function(param=value,...)@time
    # fn_type, fn, params, time =

        fn_type = entry.split('::')[0]
        fn = entry.split('::')[1].split('(')[0]
        param_list = entry.split('(')[1].split(')')[0].split(',')
        params = []
        for param in param_list:
            if len(param)==0:
                break
            par, val = param.split('=')
            params.append({'param': par.strip(), 'value':val.strip()})
        time = pd.Timestamp(entry.split('@')[1])

        return fn_type, fn, params, time

    # Extract log for all channels and put it in a dataframe
    log_list = []

    for channel in channels.keys():
        log = C.get_value_by_name('/%d/data/log' % channel)
        if log is None:
            log.warning('no log found for channel ID %d' % channel)
            continue
        logs = []
        for step in range(log.get_length()):
            log_entry = {}
            fn_type, fn, params, time = logparser(log.get(step))
            log_entry['channel'] = channel
            log_entry['function_type'] = fn_type
            log_entry['function'] = fn
            log_entry['params'] = None if len(params)==0 else params
            log_entry['time'] = time
            log_list.append(log_entry)

    log = pd.DataFrame(log_list)
    # log.sort('time', inplace=True)
    log = log.drop_duplicates('time')

    return log


def gwy_to_vtk(gwy_file, vtk_file, channel=None):
    """Convert a Gwyddion file to VTK. By default the first channel is used, otherwise
    the channel can be specified with the channel= parameter"""

    if not os.path.isfile(gwy_file):
        log.error('Gwyddion file %s not found' % gwy_file)
        return None

    if channel is None:
        channels = list_chans(gwy_file)
        idx = min(channels.keys())
    else:
        channels = list_chans(gwy_file, filter=channel)
        if len(channels)==0:
            channels = list_chans(gwy_file)
            idx = min(channels.keys())
            log.warning('selected channel not found, using %s' % channels[idx])
        elif len(channels)>1:
            idx = min(channels.keys())
            log.warning('more than one channel matching filter, selecting %s' % channels[idx])

        else:
            idx = channels.keys()[0]

    C = gwy.gwy_file_load(gwy_file, gwy.RUN_NONINTERACTIVE)
    gwy.gwy_app_data_browser_add(C)
    gwy.gwy_app_data_browser_select_data_field(C, idx)
    gwy.gwy_file_save(C, vtk_file, gwy.RUN_NONINTERACTIVE)
    gwy.gwy_app_data_browser_remove(C)

    return


def polynomial_distort(gwy_file, coeff_x, coeff_y, channel=None, new_chan=None):
    """Apply a polynomial distortion to the selected file/channel.

    If new_chan=None the original channel is overwritten, otherwise a new
    channel is created with the given name. Coefficients are also added
    as meta-data to this channel."""

    if len(coeff_x)!=10 or len(coeff_y)!=10:
        log.error('X and Y coefficients must both be 10 items in length')
        return None

    if not os.path.isfile(gwy_file):
        log.error('Gwyddion file %s not found' % gwy_file)
        return None

    if channel is None:
        channels = list_chans(gwy_file)
        idx = min(channels.keys())
    else:
        channels = list_chans(gwy_file, filter=channel)
        if len(channels)==0:
            channels = list_chans(gwy_file)
            idx = min(channels.keys())
            log.warning('selected channel not found, using %s' % channels[idx])
        elif len(channels)>1:
            idx = min(channels.keys())
            log.warning('more than one channel matching filter, selecting %s' % channels[idx])

        else:
            idx = channels.keys()[0]

    C = gwy.gwy_file_load(gwy_file, gwy.RUN_NONINTERACTIVE)

    datafield = C.get_value_by_name('/%d/data' % idx)
    newdata = datafield.new_alike(nullme=True)

    # distort(dest, invtrans, user_data, interp, exterior, fill_value)
    # datafield.distort(newdata, invtrans=?, interp=gwy.INTERPOLATION_ROUND, exterior=gwy.EXTERIOR_BORDER_EXTEND)

    return
