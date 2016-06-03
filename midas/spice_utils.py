#!/usr/bin/python
"""
spice_utils.py - a collection of routines to facilitate using SPICE under
python - mainly using PySPICE and calling cspice executables
"""

from midas import common
import os, dateutil
import numpy as np
import spiceypy as spice

debug = True
km_to_au = (1/149598000.)
kernel_path = common.kernel_path


def spk_interval(spk_list):
    """Calls brief and extracts the start and end of interval data from list of
    SPK files, returning the largest overlapping range"""

    import subprocess
    from dateutil import parser

    if type(spk_list) != list:
        spk_list = [spk_list]

    # TODO: SPKs can have many intervals, here I'm assuming only one - this is valid for
    # Rosetta SPICE kernels, but not generally.

    spk_start = parser.parse('01 January 1970')
    spk_end = parser.parse('31 December 2020')

    for spk_file in spk_list:
        cmd = subprocess.Popen(['brief', spk_file], shell=False, stdout=subprocess.PIPE)
        output = cmd.stdout.readlines()
        interval = output[-2].strip().split('            ')
        start = parser.parse(interval[0])
        end = parser.parse(interval[1])

        spk_start = max(spk_start,start)
        spk_end = min(spk_end,end)

    print('INFO: SPK kernels have interval %s - %s' % (spk_start, spk_end))

    return start, end


def ck_interval(ck_list, tls_file=os.path.join(kernel_path, 'lsk/NAIF0011.TLS'), tsc_file=os.path.join(kernel_path,'sclk/ros_triv.tsc') ):
    """Calls ckbrief and extracts the start and end of interval data from the file"""

    import subprocess
    from dateutil import parser

    # For the CATT file (a derived, not planning, product) I need the up-to-date clock kernel, not
    # the "fake" trivial kernel. Find the latest version!
    # TODO

    ck_start = parser.parse('01 January 1970')
    ck_end = parser.parse('31 December 2020')

    for ck_file in ck_list:
        cmd = subprocess.Popen(['ckbrief', ck_file, tls_file, tsc_file], shell=False, stdout=subprocess.PIPE)
        output = cmd.stdout.readlines()
        interval = output[-2].strip().split()
        start = parser.parse(interval[0]+' '+interval[1])
        end = parser.parse(interval[2]+' '+interval[3])

        ck_start = max(ck_start,start)
        ck_end = min(ck_end,end)

    print('INFO: CK kernels have interval %s - %s' % (ck_start, ck_end))

    return start, end


def kernel_interval():
    """Checks the intervals of loaded SPK and CK kernels and
    returns the limits"""

    numspk = spice.ktotal('SPK')
    spks = [spice.kdata(idx, 'SPK', 30, 4, 30)[0] for idx in range(numspk)]

    numck = spice.ktotal('CK')
    cks = [spice.kdata(idx, 'CK', 30, 4, 30)[0] for idx in range(numck)]

    spk_start, spk_end = spk_interval(spks)
    ck_start, ck_end = ck_interval(cks)

    return max(spk_start, ck_start), min(spk_end, ck_end)


def load_default_kernels(kernel_path=kernel_path):

    kernel_list = [
                'fk/EARTHFIXEDIAU.TF',
                'fk/NEW_NORCIA_TOPO.TF',
                'fk/ROS_V24.TF',
                'ik/ROS_MIDAS_V10.TI',
                'lsk/NAIF0011.TLS',
                'pck/PCK00010.TPC',
                # 'sclk/ROS_160302_STEP.TSC',
                'fk/ROS_CHURYUMOV_V01.TF',
                'fk/ROS_AUX.TF',
                'fk/ROS_CGS_AUX_V01.TF',
                'spk/DE405.BSP',
                'spk/NEW_NORCIA.BSP'
                ]

    result = [spice.furnsh(os.path.join(kernel_path,kernel)) for kernel in kernel_list]


def load_kernels(kernels, kernel_path=kernel_path, load_defaults=True):

    if type(kernels) == dict:
        kernels = kernels.values()
    elif type(kernels) == str:
        kernels = [kernels]

    result = [spice.furnsh(os.path.join(kernel_path,kernel)) for kernel in kernels]
    if load_defaults: load_default_kernels(kernel_path)

    return

def unload_kernels():
    """Looks up all currently loaded kernels and unloads them"""

    kernlist = list_kernels()
    [spice.unload(kern[0]) for kern in kernlist]


def list_kernels():
    """Looks up all currently loaded kernels and returns a list"""

    nkern = spice.ktotal('all')
    kernlist = []
    [kernlist.append(spice.kdata(idx, 'all', 80, 4, 80)) for idx in range(nkern)]

    return kernlist


def mtp_kernels(mtp, case='P'):
    """Looks for the latest kernels for a given MTP - these will be the CORL
    and RORL corresponding to the chosen LTP and the MTP level RATM"""

    # Example filenames
    # 012345678901234567890123456789
    # RORL_DL_001_02____A__00002.BSP
    # CORL_DL_002_02____A__00040.BSP
    # RATM_DM_008_01____A__00057.BC

    import planning

    ltp = planning.ltp_from_mtp(mtp)
    cases = ['A','B','C','H','P']

    import glob

    spk_path = os.path.join(kernel_path,'spk')
    ck_path = os.path.join(kernel_path,'ck')

    # Find all files of the correct type and case
    rorl = glob.glob(os.path.join(spk_path, 'RORL_DL_%03i_??____%c__00???.BSP') % (ltp, case.upper()))
    corl = glob.glob(os.path.join(spk_path, 'CORL_DL_%03i_??____%c__00???.BSP') % (ltp,case.upper()))
    ratm = glob.glob(os.path.join(ck_path,  'RATM_DM_%03i_??____%c__00???.BC') % (mtp,case.upper()))
    catt = glob.glob(os.path.join(ck_path,  'CATT_DV_???_??_______00???.BC'))

    if len(rorl)==0 or len(corl)==0 or len(ratm)==0 or len(catt)==0:
        print('WARNING: incomplete set of RORL/CORL/RATM/CATT files found in folder')

    # Find the latest file
    rorl = sorted(rorl, key=lambda x: ( int(os.path.basename(x)[12:14]), int(os.path.basename(x)[21:26])) )[-1] if len(rorl)>0 else None
    corl = sorted(corl, key=lambda x: ( int(os.path.basename(x)[12:14]), int(os.path.basename(x)[21:26])) )[-1] if len(corl)>0 else None
    ratm = sorted(ratm, key=lambda x: ( int(os.path.basename(x)[12:14]), int(os.path.basename(x)[21:26])) )[-1] if len(ratm)>0 else None
    catt = sorted(catt, key=lambda x: ( int(os.path.basename(x)[12:14]), int(os.path.basename(x)[21:26])) )[-1] if len(catt)>0 else None

    return {'rorl': rorl, 'corl': corl, 'ratm': ratm, 'catt': catt}


def operational_kernels(no_ck=False):
    """Gets the latest operational kernel"""

    # Example filenames
    # 012345678901234567890123456789
    # RATT_DV_086_01_01____00146.BC
    # CATT_DV_086_01_______00146.BC
    # CORB_DV_086_01_______00146.BSP
    # RORB_DV_086_01_______00146.BSP
    # ROS_150227_STEP.TSC

    import glob

    spk_path = os.path.join(kernel_path,'spk')
    ck_path = os.path.join(kernel_path,'ck')
    sclk_path = os.path.join(kernel_path,'sclk')

    # Find all files of the correct type and case
    # RORB_DV_145_01___t19_00216.BSP
    rorb = glob.glob(os.path.join(spk_path, 'RORB_DV_???_??___t19_00???.BSP'))
    corb = glob.glob(os.path.join(spk_path, 'CORB_DV_???_??_??____00???.BSP'))
    ratt = glob.glob(os.path.join(ck_path,  'RATT_DV_???_??_??____00???.BC'))
    catt = glob.glob(os.path.join(ck_path,  'CATT_DV_???_??_??____00???.BC'))

    sclk = glob.glob(os.path.join(sclk_path,  'ROS_??????_STEP.TSC'))

    if len(rorb)==0 or len(corb)==0 or len(ratt)==0 or len(catt)==0:
        print('ERROR: no matching SPICE files found in folder')
        return False

    # Find the latest file
    rorb = sorted(rorb, key=lambda x: ( int(os.path.basename(x)[8:11]), int(os.path.basename(x)[23:26])) )[-1]
    corb = sorted(corb, key=lambda x: ( int(os.path.basename(x)[8:11]), int(os.path.basename(x)[23:26])) )[-1]
    ratt = sorted(ratt, key=lambda x: ( int(os.path.basename(x)[8:11]), int(os.path.basename(x)[23:26])) )[-1]
    catt = sorted(catt, key=lambda x: ( int(os.path.basename(x)[8:11]), int(os.path.basename(x)[23:26])) )[-1]
    sclk = sorted(sclk, key=lambda x: ( int(os.path.basename(x)[4:10])))[-1]

    if no_ck:
        return {
            'rorb_old': os.path.join(spk_path, 'RORB_DV_145_01___t19_00216.BSP'),
            'corb_old': os.path.join(spk_path, 'CORB_DV_145_01_______00216.BSP'),
            'rorb': rorb, 'corb': corb, 'sclk': sclk }
    else:
        return {
            'rorb_old': os.path.join(spk_path, 'RORB_DV_145_01___t19_00216.BSP'),
            'corb_old': os.path.join(spk_path, 'CORB_DV_145_01_______00216.BSP'),
            'ratt_old': os.path.join(ck_path, 'RATT_DV_145_01_01____00216.BC'),
            'catt_old': os.path.join(ck_path, 'CATT_DV_145_01_______00216.BC'),
            'rorb': rorb, 'corb': corb, 'ratt': ratt, 'catt': catt, 'sclk': sclk }

def get_geometry(start, end, timestep=3660., kernels=None, no_ck=False):
    """Accepts a start and end date/time (either a string or a datetime object)
    and an optional timestep. Loads operational kernels, calculates all
    geometric data and returns a time-indexed dataframe"""

    import pandas as pd

    if kernels is None:
        kernels = operational_kernels(no_ck=no_ck)

        if pd.Timestamp(start) < pd.Timestamp("2015-03-01T00:00"):
            # If old (pre 2015) kernels are present in the list, load them first
            old_list = ['ratt_old', 'catt_old', 'rorb_old', 'corb_old']
            for old_kern in old_list:
                if old_kern in kernels.keys():
                    load_kernels(kernels[old_kern], load_defaults=False)
                    kernels.pop(old_kern)
        load_kernels(kernels, load_defaults=True)
    else:
        load_kernels(kernels, load_defaults=True)

    ets, times = get_timesteps(start, end, timestep=timestep)

    cometdist = comet_sun_au(ets)
    distance, speed = trajectory(ets, speed=True)

    if not no_ck:

        phase = phase_angle(ets) #  runs quickly
        sun_angle = nadir_sun(ets)
        offpointing = off_nadir(ets)
        lat, lon = latlon(ets) # runs quickly

        geom = pd.DataFrame(np.column_stack( [cometdist, distance, speed, lat, lon, offpointing, phase, sun_angle]), index=times,
            columns=['cometdist','sc_dist','speed','latitude','longitude','offnadir','phase','sun_angle'])

    else:

        geom = pd.DataFrame(np.column_stack( [cometdist, distance, speed]), index=times,
            columns=['cometdist','sc_dist','speed' ])

    unload_kernels()

    return geom


def get_geometry_at_times(times, kernels=None):
    """Accepts a list of times. Loads operational kernels, calculates all
    geometric data and returns a time-indexed dataframe"""

    import pandas as pd

    if kernels is None:
        kernels = operational_kernels(no_ck=no_ck)
        # If old (pre 2015) kernels are present in the list, load them first
        old_list = ['ratt_old', 'catt_old', 'rorb_old', 'corb_old']
        for old_kern in old_list:
            if old_kern in kernels.keys():
                load_kernels(kernels[old_kern])
                kernels.pop(old_kern)
        load_kernels(operational_kernels(no_ck=no_ck).values(), load_defaults=True)
    else:
        load_kernels(kernels, load_defaults=True)

    # all(isinstance(x,int) for x in times)

    ets = [spice.str2et(tm.isoformat()) for tm in times]

    cometdist = comet_sun_au(ets)
    lat, lon = latlon(ets)
    phase = phase_angle(ets)
    sun_angle = nadir_sun(ets)
    distance, speed = trajectory(ets, speed=True)
    offpointing = off_nadir(ets)

    unload_kernels()

    geom = pd.DataFrame(np.column_stack( [cometdist, distance, speed, lat, lon, offpointing, phase, sun_angle]), index=times,
        columns=['cometdist','sc_dist','speed','latitude','longitude','offnadir','phase','sun_angle'])

    return geom


def get_timesteps(start, end, timestep=60.):
    """Accepts a start and end time (datetime) and a timestep and returns
    two arrays - an array of SPICE ETs and an array of datetimes for plotting"""

    from dateutil import parser

    if type(start)==str:
        start = parser.parse(start)

    if type(end)==str:
        end = parser.parse(end)

    start_time_et = spice.str2et(start.isoformat())
    end_time_et = spice.str2et(end.isoformat())

    timesteps = int(  (end_time_et-start_time_et)/timestep )-1
    times = np.arange(timesteps)*timestep + start_time_et

    # also back-calculate an ISO format time for each step, for plotting and filtering
    times_real = np.array([parser.parse(spice.et2utc(time,'ISOC',0)) for time in times])

    return times, times_real



def comet_sun_au(times):
    """Calculate the comet-Sun distance in AU and return"""

    # Define SPICE call constants
    frame = 'ECLIPJ2000'
    target = 'CHURYUMOV-GERASIMENKO'
    abcorr = 'none'
    observer = 'SUN'

    spkpos = [spice.spkpos(target, time, frame, abcorr, observer) for time in times]
    sunpos = np.array([spkpos[index][0] for index in range(len(times))])
    sunpos_x = sunpos[:,0]; sunpos_y = sunpos[:,1]; sunpos_z = sunpos[:,2]
    cometdist_au = np.sqrt( sunpos_x**2. + sunpos_y**2. + sunpos_z**2. ) * km_to_au

    return cometdist_au


def latlon(times):
    """Retrieves the sub-spacecraft latitude and longitude"""

    target = 'ROSETTA'
    frame = '67P/C-G_CK'
    observer = 'CHURYUMOV-GERASIMENKO'
    abcorr = 'none'

    # returns state (position, velocity) and light-time
    spkezr = [spice.spkezr(target, time, frame, abcorr, observer) for time in times]
    latlon = np.array([spice.reclat(spkezr[idx][0][0:3]) for idx in range(len(times))])
    longitude = latlon[:,1] * spice.dpr()
    latitude = latlon[:,2] * spice.dpr()

    return latitude, longitude


def phase_angle(times):
    """Retrieves the Sun-comet-Rosetta phase angle"""

    frame = 'ECLIPJ2000'

    spkpos = [spice.spkpos('SUN', time, frame, 'none', 'CHURYUMOV-GERASIMENKO') for time in times]
    sunpos = np.array([spkpos[index][0] for index in range(len(times))])

    # Constants for SPICE calls
    observer = 'CHURYUMOV-GERASIMENKO'
    target = 'ROSETTA'

    spkezr = [spice.spkezr(target, time, frame, 'none', observer) for time in times]
    posvel = np.array([spkezr[index][0] for index in range(len(times))])
    phase = np.array([spice.vsep(posvel[i,0:3],sunpos[i]) for i in range(len(times))])

    phase *= spice.dpr()

    return phase


def nadir_sun(times):

    observer = 'ROSETTA'
    target = 'SUN'
    abcorr = 'none'

    # vector from the s/c to the Sun in the s/c frame of reference
    sc_sun = [spice.spkpos(target, time, 'ROS_SPACECRAFT', abcorr, observer)[0] for time in times]
    print 'here'

    # angle between this vector and the s/c Z axis
    nadir_sun = np.rad2deg( [spice.vsep(sc_sun[count],spice.vpack(0.,0.,1.)) for count in range(len(times))])

    return nadir_sun


def trajectory(times, speed=False):
    """Retrieves the Rosetta trajectoyr wrt 67P and returns an array of
    distance values (in km)"""

    # Calculate Rosetta/comet position and velocity
    observer = 'ROSETTA'
    frame = 'ECLIPJ2000'
    target = 'CHURYUMOV-GERASIMENKO'
    abcorr = 'none'

    spkezr = [spice.spkezr(target, time, frame, abcorr, observer) for time in times]
    posvel = [spkezr[index][0] for index in range(len(times))]
    posvel = np.array(posvel)
    x = posvel[:,0]; y = posvel[:,1]; z = posvel[:,2]
    distance = np.sqrt(x*x + y*y + z*z) # km

    if speed:
        vx = posvel[:,3]; vy = posvel[:,4]; vz = posvel[:,5]
        speed = np.sqrt(vx*vx + vy*vy + vz*vz) # km/s
        return distance, speed
    else:
        return distance


def off_nadir(times):
    """Returns the off-nadir angles in degrees"""

    observer = 'ROSETTA'
    target = 'CHURYUMOV-GERASIMENKO'
    abcorr = 'none'

    # vector from the s/c to the comet in the s/c frame of reference
    sc_comet = [spice.spkpos(target, time, 'ROS_SPACECRAFT', abcorr, observer)[0] for time in times]

    # angle between this vector and the s/c Z axis
    off_nadir = np.rad2deg( [spice.vsep(sc_comet[count],spice.vpack(0.,0.,1.)) for count in range(len(times))])

    return off_nadir


def anisotropic(times):

    import math

    spkpos = [spice.spkpos('SUN', time, 'J2000', 'none', 'CHURYUMOV-GERASIMENKO') for time in times]
    sunpos = np.array([spkpos[index][0] for index in range(len(times))])

    observer = 'ROSETTA'
    frame = 'ECLIPJ2000'
    target = 'CHURYUMOV-GERASIMENKO'
    abcorr = 'none'

    spkezr = [spice.spkezr(target, time, frame, abcorr, observer) for time in times]
    posvel = np.array([spkezr[index][0] for index in range(len(times))])
    plus_x = np.array([spice.vsep(posvel[i,0:3],sunpos[i]) for i in range(len(times))]) # phase angle
    plus_z = np.array([spice.vsep(posvel[i,0:3],spice.vpack(0.,0.,1.)) for i in range(len(times))])
    plus_y = np.array([spice.vsep(posvel[i,0:3],spice.vcrss(spice.vpack(0.,0.,1.),sunpos[i])) for i in range(len(times))])
    minus_x = math.pi - plus_x
    minus_y = math.pi - plus_y
    minus_z = math.pi - plus_z

    angles = np.vstack( (plus_x, minus_x, plus_y, minus_y, plus_z, minus_z) )
    sector = np.argmin(angles, axis=0)

    return sector
