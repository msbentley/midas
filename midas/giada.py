
#!/usr/bin/python
"""giada.py

Mark S. Bentley (mark@lunartech.org), 2017

A module containing various routines related to the analysis of GIADA data"""

import common
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

km_to_au = (1/149598000.)
perih_date = pd.Timestamp('13 August 2015')



def read_phys(filenames, directory='.', recursive=True):
    """Read a GIADA level 3 (physical parameters) file(s) retrieved from the PSA.

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

    cols = ['time', 'subsystem', 'gds_is_speed', 'gds_speed', 'momentum', 'momentum_err', 'mass',
        'mass_err', 'comet_sun', 'sc_comet', 'phase_angle']

    na_values = ['', '', '-9.99', '-9.99', '', '', '-1E32', '-1E32', '', '', '']

    gdata = pd.DataFrame(columns=cols)

    for f in filelist:
        gdata = gdata.append(pd.read_table(f, sep=',', names=cols, na_values=na_values), ignore_index=True, verify_integrity=True)

    gdata.time = pd.to_datetime(gdata.time)
    gdata.set_index('time', inplace=True)
    gdata.sort_index(inplace=True)

    return gdata


def plot_speed_au(gdata, colour_peri=True):

    fig, ax = plt.subplots()

    if colour_peri:
        ax.plot(gdata[gdata.index<perih_date].comet_sun*km_to_au, gdata[gdata.index<perih_date].gds_is_speed, 'b.', label='pre-perihelion')
        ax.plot(gdata[gdata.index>=perih_date].comet_sun*km_to_au, gdata[gdata.index>=perih_date].gds_is_speed, 'r.', label='post-perihelion')
        legend = ax.legend(loc=0)
    else:
        ax.plot(gdata.comet_sun*km_to_au, gdata.gds_is_speed, '.')

    ax.set_xlabel('Comet-Sun distance (AU)')
    ax.set_ylabel('GDS+IS speed (m/s)')
    ax.grid(True)
