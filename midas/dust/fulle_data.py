#!/usr/bin/python
"""
fulle_data.py - simple module containing the data from Fulle (2010).
"""

import numpy as np
import math

# Remember that all of these data are *pre* perihelion, so fine the primary
# Rosetta mission (inbound phase), but perhaps not beyond...

AU =  [1.3, 1.9, 2.6, 3.0, 3.2, 3.4]
num_au = len(AU)
num_mass_bins = 16

massbin_fluffy_min = 10.0**np.arange(-13,3)
massbin_fluffy_max = massbin_fluffy_min*10.0
massbin_compact_min = 10.0**np.arange(-15,1)
massbin_compact_max = massbin_compact_min*10.0

bin_edges_compact = np.append(massbin_compact_min,massbin_compact_max[-1])
bin_edges_fluffy = np.append(massbin_fluffy_min,massbin_fluffy_max[-1])

density_compact = 1000.0 # kg/m3
density_fluffy = 100.0 # kg/m3

# Mass loss rate (kg/s) at the nucleus as a function of mass bin and distance (lower limit)

mass_flux_lower = [ \
    [1.1, 1.5, 2.0, 3.0, 4.0, 7.0, 11.0, 16.0, 22.0, 33.0, 40.0, 40.0, 40.0, 40.0, 40.0, 40.0], \
    [0.015, 0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0], \
    [0.0075, 0.015, 0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0], \
    [0.0018, 0.0037, 0.0075, 0.015, 0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 1.5, 1.5, 1.5, 1.5], \
    [0.00075, 0.0015, 0.003, 0.006, 0.012, 0.025, 0.05, 0.1, 0.2, 0.4, 0.5, 0.5, 0.5], \
    [0.0003, 0.00061, 0.0012, 0.0025, 0.005, 0.01, 0.02, 0.04] ]

# Mass loss rate (kg/s) at the nucleus as a function of mass bin and distance (upper limit)

mass_flux_upper = [ \
    [3.3, 4.5, 6.0, 9.0, 12.0, 21.0, 33.0, 48.0, 65.0, 100.0, 120.0, 120.0, 120.0, 120.0, 120.0, 120.0], \
    [0.08, 0.16, 0.32, 0.65, 1.3, 2.7, 5.5, 11.0, 22.0, 45.0, 60.0, 60.0, 60.0, 60.0, 60.0, 60.0], \
    [0.055, 0.11, 0.22, 0.45, 0.9, 1.8, 3.7, 7.5, 15.0, 30.0, 45.0, 45.0, 45.0, 45.0, 45.0], \
    [0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 20.0, 20.0, 20.0, 20.0], \
    [0.0077, 0.015, 0.03, 0.06, 0.12, 0.25, 0.5, 1.0, 2.0, 4.0, 5.0, 5.0, 5.0], \
    [0.0009, 0.0018, 0.0037, 0.0075, 0.015, 0.03, 0.06, 0.12] ]

# Terminal velocity (m/s) for the given grain size as a function of mass bin and distance (lower limit)

v_terminal_lower = [ \
    [210.0, 150.0, 100.0, 66.0, 47.0, 32.0, 21.0, 15.0, 10.0, 6.6, 4.7, 3.2, 2.1, 1.5, 1.0, 0.7], \
    [160.0, 110.0, 76.0, 52.0, 35.0, 24.0, 16.0, 11.0, 7.6, 5.1, 3.5, 2.4, 1.6, 1.1, 0.8, 0.5], \
    [150.0, 100.0, 66.0, 47.0, 32.0, 21.0, 15.0, 10.0, 6.6, 4.7, 3.2, 2.1, 1.5, 1.0, 0.7], \
    [100.0, 66.0, 47.0, 32.0, 21.0, 15.0, 10.0, 6.6, 4.7, 3.2, 2.1, 1.5, 1.0, 0.7], \
    [66.0, 47.0, 32.0, 21.0, 15.0, 10.0, 6.6, 4.7, 3.2, 2.1, 1.5, 1.0, 0.7], \
    [10.0, 6.6, 4.7, 3.2, 2.1, 1.5, 1.0, 0.7] ]

# Terminal velocity (m/s) for the given grain size as a function of mass bin and distance (upper limit)

v_terminal_upper = [ \
    [210.0, 150.0, 100.0, 66.0, 47.0, 32.0, 21.0, 15.0, 10.0, 6.6, 4.7, 3.2, 2.1, 1.5, 1.0, 0.7], \
    [160.0, 110.0, 76.0, 52.0, 35.0, 24.0, 16.0, 11.0, 7.6, 5.1, 3.5, 2.4, 1.6, 1.1, 0.8, 0.5], \
    [150.0, 100.0, 66.0, 47.0, 32.0, 21.0, 15.0, 10.0, 6.6, 4.7, 3.2, 2.1, 1.5, 1.0, 0.7], \
    [100.0, 66.0, 47.0, 32.0, 21.0, 15.0, 10.0, 6.6, 4.7, 3.2, 2.1, 1.5, 1.0, 0.7], \
    [66.0, 47.0, 32.0, 21.0, 15.0, 10.0, 6.6, 4.7, 3.2, 2.1, 1.5, 1.0, 0.7], \
    [10.0, 6.6, 4.7, 3.2, 2.1, 1.5, 1.0, 0.7] ]


massbin_compact_avg = [np.sqrt(massbin_compact_max[ind]*massbin_compact_min[ind]) for ind in range(len(massbin_compact_min))]				
massbin_fluffy_avg = [np.sqrt(massbin_fluffy_max[ind]*massbin_fluffy_min[ind]) for ind in range(len(massbin_fluffy_min))]

# For now just using data in the lowest 8 mass bins (where we have complete coverage in the tables)

useful_bins = 8

flux_lower = np.array([mass_flux_lower[au_bin][0:useful_bins] for au_bin in range(num_au)])
flux_upper = np.array([mass_flux_upper[au_bin][0:useful_bins] for au_bin in range(num_au)])

vterm_lower = np.array([v_terminal_lower[au_bin][0:useful_bins] for au_bin in range(num_au)])
vterm_upper = np.array([v_terminal_upper[au_bin][0:useful_bins] for au_bin in range(num_au)])

mass_compact_avg = np.array(massbin_compact_avg[0:useful_bins])
mass_fluffy_avg = np.array(massbin_fluffy_avg[0:useful_bins])

edges_compact = np.array(bin_edges_compact[0:useful_bins+1])
edges_fluffy = np.array(bin_edges_fluffy[0:useful_bins+1])

radius_compact = ((3.*mass_compact_avg)/(4*math.pi*density_compact))**(1./3.)
radius_fluffy = ((3.*mass_fluffy_avg)/(4*math.pi*density_fluffy))**(1./3.)

diam_compact = 2.*radius_compact
diam_fluffy = 2.*radius_fluffy

radius_compact_um = radius_compact * 1e6
diam_compact_um = diam_compact * 1e6

# M.S.Bentley 13/12/2012 - adding GIPSI anisotropy weightings

ani_dist = [3.0, 2.3, 1.9] # AU

# Note - ani_weight from the tables is given in terms of +X, -X, +Y, -Y, +Z, -Z
#
ani_weight = [ \
    [0.00, 0.05, 0.50, 0.10, 0.30, 0.05], \
    [0.15, 0.00, 0.15, 0.15, 0.50, 0.05], \
    [0.05, 0.00, 0.15, 0.15, 0.60, 0.05] ]
    
# Test plot of these data
#[plt.plot(data.ani_dist,np.array(data.ani_weight)[:,sector],label='Sector %i' % (sector),'-x') for sector in range(0,6)]
#plt.grid(True)
#plt.legend(loc=0)
#plt.xlabel('Distance (AU)')
#plt.ylabel('Weighting')
#plt.xlim(1.8,3.1)
#plt.ylim(-0.05,0.65)
#plt.title('Anisotropic flux weightings')




