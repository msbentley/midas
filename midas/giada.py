
#!/usr/bin/python
"""giada.py

Mark S. Bentley (mark@lunartech.org), 2017

A module containing various routines related to the analysis of GIADA data"""

import common
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as md

km_to_au = (1/149598000.)
perih_date = pd.Timestamp('13 August 2015')



def read_phys(filenames, stream=False, directory='.', recursive=True):
    """Read a GIADA level 3 (physical parameters) file(s) retrieved from the PSA.

    filenames can either be a single filename, a list of names, or a wildcard.

    If recursive=True sub-directories will be traversed looking for matches.

    If stream=True, filenames should represent a file-like object,not a filename"""

    if not stream:

        if recursive:
            selectfiles = common.locate(filenames, directory)
            filelist = [file for file in selectfiles]
        elif type(files)==list:
            filelist = filenames
        else:
            import glob
            filelist = glob.glob(os.path.join(directory,files))

    cols = ['time', 'subsystem', 'gds_is_speed', 'gds_speed', 'momentum', 'momentum_err', 'mass',
        'mass_err', 'comet_sun', 'sc_comet', 'phase_angle']

    na_values = ['', '', '-9.99', '-9.99', '', '', '-1E32', '-1E32', '', '', '']

    gdata = pd.DataFrame(columns=cols)

    if stream:
        gdata = gdata.append(pd.read_table(filenames, sep=',', names=cols, na_values=na_values), ignore_index=True, verify_integrity=True)
    else:
        for f in filelist:
            gdata = gdata.append(pd.read_table(f, sep=',', names=cols, na_values=na_values), ignore_index=True, verify_integrity=True)

    gdata.time = pd.to_datetime(gdata.time)
    gdata.set_index('time', inplace=True)
    gdata.sort_index(inplace=True)

    return gdata


def plot_speed_au(gdata, colour_peri=True, sep_pre_post=False):

    fig, ax = plt.subplots()

    predata = gdata[gdata.index<perih_date]
    postdata = gdata[gdata.index>=perih_date]

    if sep_pre_post:
        predata.comet_sun = -predata.comet_sun

    if colour_peri:
        ax.plot(predata.comet_sun*km_to_au, predata.gds_is_speed, 'b.', label='pre-perihelion')
        ax.plot(postdata.comet_sun*km_to_au, postdata.gds_is_speed, 'r.', label='post-perihelion')
        legend = ax.legend(loc=0)
    else:
        ax.plot(gdata.comet_sun*km_to_au, gdata.gds_is_speed, '.')

    ax.set_xlabel('Comet-Sun distance (AU)')
    ax.set_ylabel('GDS+IS speed (m/s)')
    ax.grid(True)

    return



def read_mbs(filenames, directory='.', recursive=True):
    """Read a GIADA level 2 MBS file.

    filenames can either be a single filename, a list of names, or a wildcard.

    If recursive=True sub-directories will be traversed looking for matches."""

    if recursive:
        selectfiles = common.locate(filenames, directory)
        filelist = [file for file in selectfiles]
    elif type(files)==list:
        filelist = filenames
    else:
        import glob
        filelist = glob.glob(os.path.join(directory,files))

    cols = ['time_utc', 'time_scet', 'pkt_seq', 'mb_id', 'evt_utc', 'evt_scet',
        'freq_ovfl', 'freq_counts', 'freq_hz', 'temp_counts', 'temp_deg']

    mbs = pd.DataFrame(columns=cols)

    for f in filelist:
        mbs = mbs.append(pd.read_table(f, sep=',', names=cols), ignore_index=True, verify_integrity=True)

    mbs.time_utc = pd.to_datetime(mbs.time_utc)
    mbs.evt_utc = pd.to_datetime(mbs.evt_utc)
    mbs.set_index('time_utc', inplace=True)
    mbs.sort_index(inplace=True)

    return mbs


def plot_mbs_freq(mbs, mbs_ids=[1,2,3,4,5], start_time='2014-07-01', end_time='2016-09-30', savefile=None):

    if type(mbs_ids) != list:
        mbs_ids = [mbs_ids]

    fig, ax = plt.subplots()

    filtered = mbs[ (mbs.index>start_time) & (mbs.index<=end_time) ]

    avail_ids = sorted(filtered.mb_id.unique())

    for mbs_id in avail_ids:
        if mbs_id in mbs_ids:
            ax.plot(filtered[filtered.mb_id==mbs_id].index, filtered[filtered.mb_id==mbs_id].freq_hz, '.',
                label='MBS %d' % mbs_id, ms=2)

    ax.set_xlabel('Time')
    ax.set_ylabel('MBS frequency (Hz)')
    ax.legend(loc=0)
    ax.grid(True)

    fig.autofmt_xdate()
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    fig.tight_layout()

    if savefile is not None:
        fig.savefig(savefile)
    else:
        plt.show()

    return
