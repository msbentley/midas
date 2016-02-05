#!/usr/bin/python
"""
common.py

Mark S. Bentley (mark@lunartech.org), 2011

A module containing only constants relating to the MIDAS AFM, e.g. typical X/Y
calibration factors etc. This way I can keep the BCR writing and other files
separate from MIDAS specific data.

And more importantly I can change these in one point and have my other code reflect
the changes!
"""

import numpy as np
from midas.dust import fulle_data
import os, math

# Default paths
ros_sgs_path = os.path.expanduser('~/ROS_SGS/') if os.getenv("ROS_SGS_PATH") is None else os.getenv("ROS_SGS_PATH")
ops_path = os.path.expanduser('~/Dropbox/work/midas/operations/') if os.getenv("MIDAS_OPS_PATH") is None else os.getenv("MIDAS_OPS_PATH")
config_path = os.path.expanduser('~/Dropbox/work/midas/operations/config/') if os.getenv("MIDAS_CFG_PATH") is None else os.getenv("MIDAS_CFG_PATH")
tlm_path = os.path.expanduser('~/btsync/midas/tlm') if os.getenv("TLM_PATH") is None else os.getenv("TLM_PATH")
gwy_path = os.path.expanduser('~/btsync/midas/images/gwy') if os.getenv("GWY_PATH") is None else os.getenv("GWY_PATH")
kernel_path = os.path.expanduser('~/btsync/midas/spice') if os.getenv("SPICE_PATH") is None else os.getenv("SPICE_PATH")
s2k_path = os.path.join(ros_sgs_path,'PLANNING/RMOC/FCT/RMIB/ORIGINAL') if os.getenv("S2K_PATH") is None else os.getenv("S2K_PATH")


# Instrument acronyms
instruments = {
    'ALICE':   'AL',
    'CONSERT': 'CN',
    'COSIMA':  'CS',
    'GIADA':   'GD',
    'MIDAS':   'MD',
    'MIRO':    'MR',
    'ROSINA':  'RN',
    'RPC':     'RP',
    'SREM':    'SE',
    'OSIRIS':  'SR',
    'VIRTIS':  'VR' }

# Maximum memory (number of pixels * channels)
memory_size = 32 * 32 * 512

# Calibration factors
zcal = 0.164 # nm/bit
xycal = {'open': 3.81384, 'closed': 1.388889} # nm/bit
linearcal = 1.e-3/1.75 # V/um
xyorigin = 106792

# freq_cal =  3.e6/2.**32
# M.S.Bentley 13/12/2014 - frequency start values were not quite being correctly calculated
# Now using actual values from the RMIB
freq_hi_cal = 2999559.389 / 65535.
freq_lo_cal = 45.769644 / 65535.

# Table centre coordinates
centre_open = 44500
centre_closed = 32768 # TODO FIXME

# FM linear positions (hard coded)
# Tip Centre [V]       [mm]     Min [V]           [mm]          Max [V]    [mm]
#
# 1   21660   6,610  12,000     20577  6,280   11,400    22743  6,941    12,600
# 2   18772   5,729  10,400     17689  5,398    9,800    19855  6,059    11,000
# 3   15884   4,848   8,800     14801  4,517    8,200    16967  5,178    9,400
# 4   12996   3,966   7,200     11913  3,636    6,600    14079  4,297    7,800
# 5   10108   3,085   5,600     9025   2,754    5,000    11191  3,415    6,200
# 6    7220   2,203   4,000     6137   1,873    3,400    8303   2,534    4,600
# 7    4332   1,322   2,400     3249   0,992    1,800    5415   1,653    3,000
# 8    1444   0,441   0,800      361   0,110    0,200    2527   0,771    1,400
# 9   -1444  -0,441  -0,800    -2527  -0,771   -1,400    -361  -0,110   -0,200
# 10  -4332  -1,322  -2,400    -5415  -1,653   -3,000   -3249  -0,992   -1,800
# 11  -7220  -2,203  -4,000    -8303  -2,534   -4,600   -6137  -1,873   -3,400
# 12 -10108  -3,085  -5,600   -11191  -3,415   -6,200   -9025  -2,754   -5,000
# 13 -12996  -3,966  -7,200   -14079  -4,297   -7,800  -11913  -3,636   -6,600
# 14 -15884  -4,848  -8,800   -16967  -5,178   -9,400  -14801  -4,517   -8,200
# 15 -18772  -5,729  -10,400  -19855  -6,059  -11,000  -17689  -5,398   -9,800
# 16 -21660  -6,610  -12,000  -22743  -6,941  -12,600  -20577  -6,280  -11,400

lin_max_offset = 0.3305 # V

lin_centre_pos_fm = [6.610, 5.729, 4.847, 3.966, 3.085, 2.203, 1.322, 0.441, \
    -0.441, -1.322, -2.203, -3.085, -3.966, -4.847, -5.729, -6.610]

lin_centre_pos_fs = [6.655, 5.775, 4.895, 4.015, 3.135, 2.255, 1.375, 0.495, \
    -0.385, -1.265, -2.145, -3.025, -3.905, -4.785, -5.665, -6.545]


# Cantilever offsets in X (linear stage) - derived from line and image scans
# Cantilever 9 is left "untouched" and others are shifted to this value

#              1   2   3    4   5   6   7   8  9   10  11 12 13  14  15 16
tip_offset = [65, 80, 90, 105, 80, 85, 85, 60, 0, -15, -5, 0, 0, 10, 45, 0]

# Geometric data
facet_area = 2.4e-3*1.2e-3 # m (1.2 x 2.4 mm facet area)
funnel_angle = 30.0 # degrees, full cone angle
wheel_radius = 30.0 # mm

# Cal curves for raw data -> engineering values
#
#
ac_v = (20.0/65535.)        # -10.0 - +10.0 V
dc_v = (20.0/65535.)        # -10.0 - +10.0 V
phase_deg = (360./65535.)   # -180.0 - +180.0 deg

def fscan_duration(num_scans):

    from datetime import timedelta

    sweep_duration = 18.0 # s
    adj_time = 20.0 # s

    margin = num_scans # 1s per scan
    duration = (num_scans * sweep_duration + adj_time) + margin

    return timedelta(seconds=duration)


# a few simple functions that are generally useful

def open_step_closed_step(step):
    """Gives the closest closed loop step size to a given open loop step"""

    closed=np.round(xycal['open']*step/xycal['closed'])
    return int(closed)


def open_to_closed(x,y):
	"""Converts a tuple of (x,y) coordinates in open loop mode to a corresponding
	closed loop coordinate"""

	# check HV values for a position of (0,0) in closed loop = 32767

	x = int(round((x - 32768) * xycal['open']/xycal['closed']))
	y = int(round((y - 32768) * xycal['open']/xycal['closed']))

	return (x,y)

def closed_to_open(x,y):
	"""Converts a tuple of (x,y) coordinates in closed loop mode to a corresponding
	open loop coordinate"""

	# check HV values for a position of (0,0) in closed loop = 32767

	x = int(round((x * xycal['closed']/xycal['open']) + 32768))
	y = int(round((y * xycal['closed']/xycal['open']) + 32768))

	return (x,y)

scan_type = ['DYN','CON','MAG']

# Adding virtual channels for unpacked status
channels = ['ZS', 'AC', 'S2', 'PH', 'DC', 'XH', 'YH', 'ZH', 'M1', 'YP', 'ZP', 'M2', 'YE', 'ZE', 'M3', 'ST']
status_channels = ['NC', 'RP', 'LA', 'MC', 'PA', 'PC']
s2_channels = ['RD', 'CF']
retr_channel = ['RT']
data_channels = channels + status_channels + s2_channels + retr_channel

cal_factors = [ zcal, 20./65535., 1, 360./65535., 220./65535., 280./65535., 280./65535., 80./65535., 20./65535., 20./65535., 20./65535., 20./65535., 20./65535., 20./65535., 20./65535., 20./65535., 1,1,1,1,1,1,1,1,1, zcal]
offsets = [0., 0., 0., 0., 0., 100., 100., 100., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,0.]
units = ['nm', 'V', 'none', 'deg', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'V', 'none', 'none','none','none','none','none','none','none','none','nm' ]
channel_names = [
    'Topography (Z piezo position set value)', # ZS
    'Cantilever AC signal', # AC
    'Extended status (contact flag and retract dist)', # S2 (formerly AM)
    'Phase', # PH
    'Cantilever DC signal', # DC
    'X high voltage monitor', # XH
    'Y high voltage monitor', # YH
    'Z high voltage monitor', # ZH
    'RMS signal at position 1 (magnetic mode)', # M1
    'Y piezo position (capacitive sensor)', # YP
    'Z piezo position (strain gauge)', # ZP
    'RMS signal at position 1 (magnetic mode)', # M2
    'Y piezo offset error (capacitive sensor)', # YE
    'Z piezo offset error (strain gauge)', # ZE
    'RMS signal at position 1 (magnetic mode)', # M3
    'Status', # ST
    'Number of cycles', # NC
    'Retraction after point advance', # RP
    'Line aborted', # LA
    'Max. number of cycles reached', # MC
    'Point aborted', # PA
    'Point converged', # PC
    'Retraction distance', # RD
     'Contact flag',
     'Retraction exceeded'] # RT

ctrl_channels = ['ac','dc','phase','zpos']
ctrl_names = ['Cantilever AC', 'Cantilever DC', 'Phase', 'Z position']
ctrl_units = ['V','V','deg','none']

# 0 = window detection
# 1 = threshold detection
# 2 = pi-controller - UNUSED
# other value = window

scan_algo = ['WINDOW', 'THRESH', 'P-CTRL']

def get_channel(channels):
    """Accepts a list of data channel codes and returns the corresponding setting
    needed to record these channels"""

    if type(channels) is not list: channels = [channels]

    if not set(channels) < set(data_channels):
        print('ERROR: unrecognised data channel mnemonic')
        return False

    return sum([2**data_channels.index(chan) for chan in channels])


def channel_list(value):
    """Accepts a channel parameter and returns a list of the corresponding
    channel codes"""

    # this is a stupid solution!
    bitstring=bin(value)[2:]

    return bitstring


def seg_to_facet(seg):
    """Accepts a wheel segment (0-1023) and returns the corresponding facet (0-63)"""

    if seg<0 or seg>1023:
        print('ERROR: segment must be between 0 and 1023 - defaulting to zero')
        return 0

    target = (seg+8) // 16
    if target==64: target = 0

    return target


def opposite_facet(facet):
    """Returns the facet opposite on the wheel (input 0-63)"""

    if facet<0 or facet>63:
        print('ERROR: facet must be between 0 and 63')
        return False

    return (32+facet)%64

def facet_to_seg(facet):

    if facet<0 or facet>63:
        print('ERROR: facet must be between 0 and 63')
        return False

    return facet*16


def seg_off_to_pos(offset):
    """Accepts a segment offset from the central position and calculates
    the shift in target Y position that this corresponds to (in microns)"""

    import math

    if (offset<-7) or (offset>7):
        print('ERROR: segment offset must be in the range -7 to +7!')
        return None

    angle_per_seg = 360./1024.
    angle = offset * angle_per_seg
    distance = wheel_radius * math.tan(math.radians(angle)) * 1000.

    return distance


def target_type(target):
    """Accepts a target (0-63) and returns its type"""

    if (target==0) or (target>=8 and target<=63):
        return 'SOLGEL'
    elif (target>=4 and target<=7):
        return 'SILICON'
    else:
        return 'CAL'


def mass_to_diam(mass_kg, fluffy=True, density=None):
    """Accepts a mass in kg and returns the corresponding spherical grain
    diameter using either a compact or fluffy density"""

    if density is None:
        density = fulle_data.density_fluffy if fluffy else fulle_data.density_compact

    return ((3.*mass_kg)/(4*math.pi*density))**(1./3.)*2.e6


def diam_to_mass(diam_um, fluffy=True, density=None):

    if density is None:
        density = fulle_data.density_fluffy if fluffy else fulle_data.density_compact

    radius = diam_um/2.e6
    vol = (4./3.)*math.pi*radius**3.
    mass = vol * density

    return mass


def printtable(df, float_fmt=None):
    """Accepts a pd.DataFrame() prints a pretty-printed table, rendered with PrettyTable"""

    from prettytable import PrettyTable
    table = PrettyTable(list(df.columns))

    if float_fmt is not None:
        table.float_format = float_fmt

    for row in df.itertuples():
            table.add_row(row[1:])
    print(table)
    return
