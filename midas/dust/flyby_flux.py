# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
flyby_flux.py - test routine to calculate fluxes and coverage using SPICE and Fulle (2010) input!
"""

from midas import common, bcrutils
import numpy.random as random
import math, os, sys
import numpy as np
import fulle_data as data
import spiceypy as spice
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from flyby_flux import *

def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index."""

    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero()

    # We need to start things after the change in "condition". Therefore,
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size - 1]

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

def path_exists(path):
    import errno
    import os

    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise



def run_flyby(meta_kernel, start_time, end_time, timestep, description, filename, pointing=False):
    """Using SPICE kernels specified in flyby_integration-furnsh.txt, calculate distance,
    speed etc. over the entire trajectory file in the number of specified timesteps.

    Returns: timestep, days, distance, speed, cometdist_au, angle"""

    # Constants
    km_to_au = (1/149598000.)

    # Use the SPICE furnsh command to load several kernels (here the sample kernels and one leapsecond)
    if not os.path.isfile(meta_kernel):
        print('Meta kernel not found - specify the full path')
        return 0

    spice.furnsh(meta_kernel) # furnish all kernels

    # Start and end times are found by inspecting the .bsp kernels
    #
    # TODO: find a way to extract these data from the kernel at runtime

    # Convert these from dates to ET
    if type(start_time) == str:
        start_time_et = spice.str2et(start_time)
    else:
        start_time_et = start_time
    if type(end_time) == str:
        end_time_et = spice.str2et(end_time)
    else:
        end_time_et = end_time

    timesteps = int(round( (end_time_et-start_time_et)/timestep ))

    # Create an array of times for which to call the SPICE kernel reader
    times = np.arange(timesteps)*timestep + start_time_et

    ###### COMET POSITION WRT THE SUN ######

    # For each time, find the comet-Sun distance
    spkpos = [spice.spkpos('SUN', time, 'J2000', 'none', 'CHURYUMOV-GERASIMENKO') for time in times]
    sunpos = np.array([spkpos[index][0] for index in range(timesteps)])
    sunpos_x = sunpos[:,0]; sunpos_y = sunpos[:,1]; sunpos_z = sunpos[:,2]
    cometdist_au = np.sqrt( sunpos_x**2. + sunpos_y**2. + sunpos_z**2. )
    cometdist_au = cometdist_au * km_to_au

    ###### SPACECRAFT POSN & VEL WRT THE COMET ######

    # Constants for SPICE calls
    observer = 'CHURYUMOV-GERASIMENKO'
    target = 'ROSETTA'
    # frame = 'J2000'
    frame = 'ECLIPJ2000' # needed for anisotropic case to match GIPSI geometry
    # frame = 'CG_CENTRIC'
    abcorr = 'none'

    # The spkezr routine returns the position (X,Y,Z), velocity and lighttime from observer to target
    # for a given time
    spkezr = [spice.spkezr(target, time, frame, abcorr, observer) for time in times]

    # spkezr returns an array of tuplets, which I will manually unpack into numpy arrays
    # [perhaps there's a better, more pythonic, way to do this...?]

    # lighttime = np.array([spkezr[index][1] for index in range(timesteps)])

    posvel = [spkezr[index][0] for index in range(timesteps)]
    posvel = np.array(posvel)
    x = posvel[:,0]; y = posvel[:,1]; z = posvel[:,2]
    vx = posvel[:,3]; vy = posvel[:,4]; vz = posvel[:,5]

    # Calculated distance and speed scalars for plotting
    distance = np.sqrt(x*x + y*y + z*z) # km
    speed = np.sqrt(vx*vx + vy*vy + vz*vz) # km/s

    ###### SPACECRAFT POINTING ######

    # Also need the off-nadir angle! Here we calculate the MIDAS-comet CoM vector and then
    # find the angle between this and the MIDAS frame Z axis:
    #
    if pointing:
        midas_pos = [spice.spkpos(observer, time, 'ROS_MIDAS', 'none', observer) for time in times]
        angle = np.rad2deg([spice.vsep(midas_pos[count][0],spice.vpack(0.,0.,1.)) for count in range(timesteps)])
    else:
        angle = np.zeros_like(time)

    # M.S.Bentley 08/11/2012 - if requested, calculate anisotroptic distribution
    # using Fulle's SWT presentation and results from the GIPSI team

    plus_x = np.array([spice.vsep(posvel[i,0:3],sunpos[i]) for i in range(timesteps)]) # phase angle
    plus_z = np.array([spice.vsep(posvel[i,0:3],spice.vpack(0.,0.,1.)) for i in range(timesteps)])
    plus_y = np.array([spice.vsep(posvel[i,0:3],spice.vcrss(spice.vpack(0.,0.,1.),sunpos[i])) for i in range(timesteps)])
    minus_x = math.pi - plus_x
    minus_y = math.pi - plus_y
    minus_z = math.pi - plus_z

    angles = np.vstack( (plus_x, minus_x, plus_y, minus_y, plus_z, minus_z) )
    sector = np.argmin(angles, axis=0)

    # Plot the time spent in each "sector" used in the anisotropic case
    bins=range(0,7)
    plt.hist(sector,bins=bins,normed=True, align='left')
    plt.xlim(-0.5,5.5)
    plt.xlabel('Sector')
    plt.ylabel('fraction of time in sector')
    plt.suptitle(description)
    plt.savefig(filename+'_sectors.png')
    plt.clf()
    plt.close()

    # unload the kernels in use
    spice.unload(meta_kernel)

    return timesteps, times, distance, speed, cometdist_au, angle, sector, plus_x


def calculate_dustflux(timestep, days, cometdist_au, distance, off_nadir, sector, scan_width, scan_height, fluffy, upper_lower, divine, anisotropic, description, plotflux=False, filename=''):
    """Calculate the dustflux based on data from Fulle et al. (2010), interpolating to small
    sizes either with a simple power law extrapolation, or using the Divine transition to
    a shallower gradient. Inputs are:

    timestep - the interval between distance steps (s)
    cometdist_au - an array of heliocentric distance at each timestep (typically from SPICE trajectory)
    distance - an array of spacecraft-comet distance (km) (typically from SPICE trajectory)

    scan_width = 30. # scan width in microns
    scan_height = 30. # scan height in microns
    fluffy = False # set to True for fluffy particles, False for compact
    upper_lower = 'upper' # set to 'lower' for the lower limit, 'upper' for the upper
    divine = False # True = transition to smaller mass index for smaller particles, False = keep one index throughout
    description = label for output plots
    """

    import fulle_flux, midas

    # Assuming isotropic emission but this time calculate a number density and
    # include the effects of dust velocity dispersion and spacecraft velocity.

    # Assuming a radially symmetric coma, the number density at a given distance from the comet is given as:
    #
    # n_d(m) = Q(m) / 4pi V_esc r^2
    #
    # where Q(m) is the number of particles of mass class m that are released from the nucleus per unit time.

    # Need value of spacecraft velocity at each timestep
    #
    # TODO

    # And terminal velocity of dust particles
    #
    # TODO
    # Calculate number density of dust particles accounting for spacecraft distance and velocity wrt the nucleus

    scan_area = (scan_width*1.e-6)*(scan_height*1.e-6)

    timesteps = len(cometdist_au)

    min_mass = -20
    max_mass = -6
    num_edges = abs(max_mass-min_mass+1) # decadal bins

    bin_edges_mass = np.logspace(min_mass,max_mass,num=num_edges)
    num_bins = num_edges - 1

    # surface_number_flux returns: F, a, b, c, m_t
    cdf = np.array([fulle_flux.surface_number_flux(dist, fluffy, upper_lower, divine, False) for dist in cometdist_au])

    # now use these CDF coeffs to calculate binned flux and corresponding weighted average mass per bin
    binned = np.array([fulle_flux.binned_flux(cdf[step,0], cdf[step,1], cdf[step,2], cdf[step,3], cdf[step,4], min_mass, max_mass, num_edges) for step in range(timesteps)])

    mean_mass = binned[0,1,:] # this is the same for all timesteps
    pcles_s = binned[:,0,:]

    if fluffy:
        mean_diam = 2.*((3.*mean_mass)/(4.*math.pi*data.density_fluffy))**(1./3.)
    else:
        mean_diam = 2.*((3.*mean_mass)/(4.*math.pi*data.density_compact))**(1./3.)

    distance_m = distance*1.e3 # km -> m
    shell_area = 4.*math.pi*distance_m**2. # area of a spherical shell at each cometocentric distance

    pcles_m2_s = np.array([pcles_s[:,mass]/shell_area for mass in range(num_bins)]) # particles /m2/s
    pcles_m2_timestep = pcles_m2_s * timestep # particles /ms/timestep
    pcles_scan_timestep = pcles_m2_timestep * scan_area # particles collected in the given scan area in one timestep

    # M.S.Bentley 13/12/2012 - if anisotropy is requested, weight according to the GIPSI tables
    # ani_dist = distance (AU) over which data apply
    # ani_weight = fraction of particles emitted over this sector

    if anisotropic:

        # interpolate the weighting piecewise as a function of heliocentric distance
        weighting = [ np.interp( cometdist_au[time], np.array(data.ani_dist), np.array(data.ani_weight)[:,sector[time]] ) for time in range(timesteps)]

        # FIXME plot a comparison
        # plt.figure()
        # plt.semilogy(days,pcles_scan_timestep[5,:], label='isotropic')
        # plt.semilogy(days,pcles_scan_timestep[5,:] * weighting * 6., label='anisotropic')
        # plt.grid(True)
        # plt.legend(loc=0)
        # plt.show()

        # multiply the pcles_scan_timestep by the appropriate (normalised) weighting factor
        pcles_scan_timestep = pcles_scan_timestep * weighting * 6.

    # [semilogy(days,pcles_scan_timestep[massbin,:]) for massbin in range(num_bins)] # plot the number of particles over time for each mass bin

    # The flyby routine also returns off-nadir pointing (in degree), so we can use this to modify the effective target area
    # sinusoidally with off-pointing to some critical point beyond which we collect no dust.
    no_dust = off_nadir > common.funnel_angle/2.
    mask = np.array([no_dust for count in range(num_bins)])
    cos_offnadir = np.array([np.cos(np.deg2rad(off_nadir)) for count in range(num_bins)])
    pcles_scan_timestep[mask] = 0.
    pcles_scan_timestep[np.invert(mask)] = pcles_scan_timestep[np.invert(mask)] * cos_offnadir[np.invert(mask)]

    # Now sum over the trajectory, so that we integrate the total number of particles per mass bin
    cumulative = np.zeros([timesteps,(num_bins)],dtype=float)

    for step in range(1,timesteps):
        cumulative[step,:] = [cumulative[step-1,mass] + pcles_scan_timestep[mass,step] for mass in range(num_bins)]

    if plotflux:
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        [ax1.semilogy(days,pcles_scan_timestep[massbin,:],label="%3.2e" % (mean_mass[massbin]*1.e6)+ ' kg') for massbin in range(num_bins) if pcles_scan_timestep[massbin,:].max() > 0.]
        import matplotlib.font_manager as fm
        prop = fm.FontProperties(size=11)
        ax1.legend(loc=0,prop=prop,title='Mean mass')
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('# particles per timestep (%3.2f s)' % timestep)
        ax1.grid(True)
        fig.suptitle(description[0] + '\n' + description[1])

        if filename != "":
            plotfile = filename+'_flux.png'
            plt.savefig(plotfile)
            plt.clf()
            plt.close()
    if plotflux and filename != "":
        return bin_edges_mass, list(mean_mass), list(mean_diam), pcles_scan_timestep, cumulative, plotfile
    else:
        return bin_edges_mass, list(mean_mass), list(mean_diam), pcles_scan_timestep, cumulative


def plot_flux(times, mean_diam, cumulative, scan_times=[], description='', filename=''):
    """Plot the cumulative fluences for each particle size bin (exclude mass bins with <1 particle)"""

    plotfile = ''

    if max(cumulative[-1,:]) > 1.0:

        fig1 = plt.figure()
        ax1 = fig1.add_subplot(1,1,1)
        [ax1.semilogy(times,cumulative[:,mass],label="%3.2f" % (mean_diam[mass]*1.e6)+ u' µm') for mass in range(cumulative.shape[1]) if(cumulative[-1,mass] > 1.)]
        ax1.grid(True)

        import matplotlib.font_manager as fm
        prop = fm.FontProperties(size=11)
        ax1.legend(loc=0,prop=prop,title='Mean diameter',fancybox=True)

        ax1.set_ylim(ymin=1.)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('# accumulated particles')
        # fig1.suptitle(description[0] + '\n' + description[1])
        fig1.autofmt_xdate()


        # 09/07/12: adding capability to show scheduled scans with a translucent rectangle
        #
        # scan_times contains an array of start/stop indices
        if len(scan_times) > 0:
            [ax1.axvspan(days[startstop[0]],days[startstop[1]], alpha=0.5) for startstop in scan_times]

        if filename != "":
            plotfile = filename+'_accumulated.png'
            plt.savefig(plotfile)
            plt.clf()
            plt.close()

    return plotfile




def plot_histogram(bin_edges_mass, mean_diam, cumulative, fluffy, description, filename=''):
    """Plot a histogram showing integer number of particles collected at the end of the flyby"""

    plotfile = ''
    mean_diam = np.array(mean_diam)

    # Produce an integer count of particles collected per bin
    if len(cumulative.shape) == 1:
        final_count = np.array(np.floor(cumulative),dtype=int)
    else:
        final_count = np.array(np.floor(cumulative[-1,:]),dtype=int)

    num_sizes = (final_count != 0).sum() # number of bins with non-zero counts

    if (final_count.sum() > 0):

        # If we have many points, switch to a logarithmic scale on the count axis
        if final_count.sum() > 1000:
            log=True
            ymin=1.
        else:
            log=False
            ymin=0.

        fig2 = plt.figure()
        ax2 = fig2.add_subplot(1,1,1)

        # rescale the plot to leave room for both title and upper axis label
        box2 = ax2.get_position()
        ax2.set_position([box2.x0, box2.y0, box2.width, box2.height*0.9])

	# set bottom=1. to avoid bug when setting log=True
        ax2.bar(left=np.log10(bin_edges_mass[:-1]),height=final_count,width=1.,log=True,bottom=1.)

        ax2.set_xlabel('log(mass [kg])')
        ax2.set_ylabel('# accumulated particles')
        ax2.set_ylim(ymin=ymin)

        # create a second X axis and plot diameter ticks corresponding to the mean mass per bin
        xlimits = ax2.get_xlim() # current limits of the lower axis

        if fluffy:
            newxmin = ((3.*(10**xlimits[0]))/(4*math.pi*data.density_fluffy))**(1./3.) * 2.e6 # diam in microns
            newxmax = ((3.*(10**xlimits[1]))/(4*math.pi*data.density_fluffy))**(1./3.) * 2.e6 # diam in microns
        else:
            newxmin = ((3.*(10**xlimits[0]))/(4*math.pi*data.density_compact))**(1./3.) * 2.e6 # diam in microns
            newxmax = ((3.*(10**xlimits[1]))/(4*math.pi*data.density_compact))**(1./3.) * 2.e6 # diam in microns

        diam_ticks = np.log10(mean_diam*1.e6)
        diam_label = ["%3.2f" % (mean_diam[mass]*1.e6) for mass in range(len(mean_diam))]

        ax2_upper = ax2.twiny() # new axis instance
	ax2_upper.set_position([box2.x0, box2.y0, box2.width, box2.height*0.9])

        ax2_upper.set_xlim(np.log10(newxmin),np.log10(newxmax))
        ax2_upper.set_xticks(diam_ticks[0:num_sizes])
        ax2_upper.set_xticklabels(diam_label[0:num_sizes])

        ax2_upper.set_xlabel(u'mean diameter [µm]')
        ax2_upper.grid(True)

        fig2.suptitle(description[0] + '\n' + description[1])

        if filename != '':
            plotfile = filename+'_histogram.png'
            plt.savefig(plotfile)
            plt.clf()
            plt.close()


    return plotfile



def plot_massflux_histogram(bin_edges_mass, mean_mass, mean_diam, pcles_s, fluffy, description):
    """Plot a histogram showing integer number of particles collected at the end of the flyby"""

    mean_diam = np.array(mean_diam)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.bar(left=np.log10(bin_edges_mass[:-1]),height=(pcles_s*mean_mass),width=1.,log=True)

    # rescale the plot to leave room for both title and upper axis label
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0, box2.width, box2.height*0.9])

    ax2.set_xlabel('log(mass [kg])')
    ax2.set_ylabel('Mass flux (kg/s)')

    # create a second X axis and plot diameter ticks corresponding to the mean mass per bin
    xlimits = ax2.get_xlim() # current limits of the lower axis

    if fluffy:
        newxmin = ((3.*(10**xlimits[0]))/(4*math.pi*data.density_fluffy))**(1./3.) * 2.e6 # diam in microns
        newxmax = ((3.*(10**xlimits[1]))/(4*math.pi*data.density_fluffy))**(1./3.) * 2.e6 # diam in microns
    else:
        newxmin = ((3.*(10**xlimits[0]))/(4*math.pi*data.density_compact))**(1./3.) * 2.e6 # diam in microns
        newxmax = ((3.*(10**xlimits[1]))/(4*math.pi*data.density_compact))**(1./3.) * 2.e6 # diam in microns

    diam_ticks = mean_diam*1.e6
    diam_label = ["%3.2f" % tick for tick in diam_ticks]

    ax2_upper = ax2.twiny() # new axis instance

    ax2_upper.set_xlim(np.log10(newxmin),np.log10(newxmax))
    ax2_upper.set_xticks(np.log10(diam_ticks))
    ax2_upper.set_xticklabels(diam_label)

    ax2_upper.set_xlabel(u'mean diameter [microns]')
    ax2_upper.grid(True)

    ax2.yaxis.grid(True)

    fig2.suptitle(description)

    return




def write_counts(bin_edges_mass,diameters,counts,width,height,filename):
    """Write particle counts to a file for later analysis

    Inputs: mass bin edge (kg), diameter (microns), counts (int), width, height (microns) and a filename"""

    if (width <= 0.) or (height <= 0.):
        print('Width and height must be positive!')
        return 0

    if len(diameters) != len(counts):
        print('Error: number of diameters and counts must be the same!')
        return 0

    count_file = open(filename, 'w')

    count_file.write('width: ' + str(width) + '\n')
    count_file.write('height: ' + str(height) + '\n')

    # write a header describing the fields
    count_file.write('min mass bin edge (kg), max mass bin edge (kg), diameter (microns), particle count\n')

    [count_file.write(str(bin_edges_mass[bin]) + ', ' + str(bin_edges_mass[bin+1]) + ', ' + str(diameters[bin]) + ', ' + str(counts[bin]) + '\n') for bin in range(len(diameters))]

    count_file.close()

    return


def read_counts(filename=''):
    """Reads previously generated particle counts to a file.

    Takes as inputs the filename, and returns scan parameters, mass bin edges, mean diameter and counts"""

    import os

    if filename == '': # blank, so prompt

        # Import gtk libraries for the file dialogue
        import pygtk
        pygtk.require('2.0')
        import gtk

        dialog = gtk.FileChooserDialog("Open MIDAS particle counts file",
            None,
            gtk.FILE_CHOOSER_ACTION_OPEN,
            (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
            gtk.STOCK_OPEN, gtk.RESPONSE_OK))

        dialog.set_default_response(gtk.RESPONSE_OK)

        filter = gtk.FileFilter()
        filter.set_name("data files")
        filter.add_pattern("*.dat")
        dialog.add_filter(filter)

        # Open the dialogue and deal with the response
        response = dialog.run()

        if response == gtk.RESPONSE_OK:
            filename = dialog.get_filename()
            print filename, 'selected'
        elif response == gtk.RESPONSE_CANCEL:
            sys.exit('No file selected, aborting!')

        # Close the dialogue
        #
        dialog.destroy()

    else: # need to check if the file exists
        if not os.path.isfile(filename):
            return 0

    diameter = []
    count = []
    edge = []

    # Open file and read linewise
    for line in open(filename):
        if len(line.split(':')) == 2:
            param, val = line.split(':')
            if param.lower() == 'width':
                width = float(val.strip())
            elif param.lower() == 'height':
                height = float(val.strip())
            else:
                print 'Unknown parameter!'
                return 0
        elif len(line.split(',')) == 3:
            edge_tmp, diam_tmp, ct_tmp = line.split(',')
            edge.append(float(edge_tmp))
            diameter.append(float(diam_tmp))
            count.append(long(ct_tmp))
        else:
            print ('Invalid data in file!')
            return 0

    return edge, diameter,count,width,height


def generate_facet(scan_width, scan_height, diameter, count):
    """Takes the scan size (width, height) and a calculated histogram of collected particles,
    and returns an array of particles positions (x,y,d) arranged over the given facet area."""

    if (len(diameter) != len(count) or len(diameter) <= 0):
        # print('ERROR: number of diameter bins and counts must be equal and greater than zero!')
        return 0

    # Generate the random particle distribution and add to a list of tuples (x,y,d)
    particles = []
    for bin in range(len(diameter)):
        [particles.append( (rand()*scan_width,rand()*scan_height,diameter[bin]) ) for num in range(count[bin])]

    return particles


def view_facet(scan_width, scan_height, diameter, count, description, show_counts, filename='', particles=False, xpixels=0, ypixels=0):
    """Display the facet coverage, generated either from a histogram, or previously defined
    particle positions..."""

    import matplotlib.cm as cm
    from pylab import rcParamsDefault

    plotfile = ''

    # FIXME: particles are allowed to overlap, so any coverage calculations for high particle counts my not be accurate!
    # FIXME: also, particle are generated that can overlap the edge of the exposure area

    if (len(diameter) != len(count) or len(diameter) <= 0):
        print('ERROR: number of diameter bins and counts must be equal and greater than zero!')
        return 0, plotfile

    num_sizes = len(diameter)

    # if not particles:
    #
    #     # Generate the random particle distribution and add to a list of tuples (x,y,d)
    #     particles = []
    #     for bin in range(len(diameter)):
    #         pcles = [ (rand()*scan_width,rand()*scan_height,diameter[bin]) for num in range(count[bin])]
    #         particles.append( pcles )
    # else: # particle positions defined by previous call
    #     if len(particles) != len(diameter):
    #         print('Mismatch between histogram and particle coordinate data!')
    #         return False

    # Size the plot window to the aspect ratio of the figure
    # keep the existing longest side and reduce the other
    # get defaults from previous plot

    # Get default image size
    fig_size = rcParamsDefault['figure.figsize']

    aspect = scan_width/scan_height
    if aspect != 1.0:
        if scan_width > scan_height: # landscape
            new_size=(fig_size[0],fig_size[1]*(scan_height/scan_width)*1.5)
        else: # portrait
            new_size=(fig_size[0]*(scan_width/scan_height)*1.5,fig_size[0])
    else:
        new_size = fig_size

    fig3 = plt.figure(figsize=new_size)
    ax3 = fig3.add_subplot(111)
    ax3.set_aspect('equal', adjustable='box-forced')
    ax3.set_xlabel('X (microns)')
    ax3.set_ylabel('Y (microns)')
    ax3.set_xlim(0,scan_width)
    ax3.set_ylim(0,scan_height)

    # rescale the plot to leave room for the legend to the right
    box3 = ax3.get_position()
    ax3.set_position([0., box3.y0, box3.width, box3.height])

    # If xpixels and ypixels are set, show gridlines to represent where the image will be sampled
    if (xpixels > 0) and (ypixels > 0):

        xstep_cal = common.xycal['open']/1.e3 # using the open loop calibration factor
        ystep_cal = common.xycal['open']/1.e3 # since that's all we have for the FM

        # If x and y steps not specified, calculate to fit the exposure area
        xstep = np.floor(float(scan_width) / float(xpixels) / xstep_cal)
        ystep = np.floor(float(scan_height) / float(ypixels) / ystep_cal)
        print('X and Y step sizes: ', str(xstep*xstep_cal), str(ystep*ystep_cal))

        xsteps = np.arange(xpixels) * xstep * xstep_cal
        ysteps = np.arange(ypixels) * ystep * ystep_cal

        ax3_upper = ax3.twiny() # new axis x instance
        ax3_right = ax3.twinx() # new axis y instance
        ax3_upper.set_aspect('equal', adjustable='box-forced')
        ax3_right.set_aspect('equal', adjustable='box-forced')

        ax3_upper.set_xticklabels([]) # remove labels
        ax3_right.set_yticklabels([])

        ax3_upper.set_xticks(xsteps)
        ax3_right.set_yticks(ysteps)

        ax3_upper.set_xlim(0, scan_width)
        ax3_right.set_ylim(0, scan_height)

        ax3_upper.grid(True)
        ax3_right.grid(True)

    else:
        ax3.grid(True)

    # make a colour index with the correct number of colors, spanning the colourmap
    colours = cm.get_cmap('gist_rainbow')
    colour_list = [colours(1.*i/num_sizes) for i in range(num_sizes)]

    # label each class accordingly
    diam_label = ["%3.2f" % (diameter[diam]) for diam in range(len(diameter))]

    max_circles = 10000 # maximum circles drawn of each type
    too_many = False
    import matplotlib.collections as mcoll

    particles = []
    for bin in range(len(diameter)): # use range(len(diameter))[::-1] to draw large particles first (reverse order)

        if count[bin] > max_circles:
            counter = max_circles
            too_many = True
        else:
            counter = count[bin]

        x = random.random(counter)*scan_width
        y = random.random(counter)*scan_height

        y0 = ax3.transData.transform([(0,0),(0,1)])[0][1]
        y1 = ax3.transData.transform([(0,0),(0,1)])[1][1]
        scale = abs(y1-y0)
        radius = diameter[bin]/2. * scale

        sizes = np.ones(counter) * np.pi*radius**2.
        circles = mcoll.CircleCollection(sizes, offsets=zip(x,y), transOffset=ax3.transData, edgecolor=colour_list[bin], facecolor=colour_list[bin])
        ax3.add_collection(circles)

        # return the particle coordinates as a set of tuples (x,y,d) for BCR generation etc.
        particles.extend(zip(x,y,np.ones_like(x)*diameter[bin]))

	ax3.autoscale_view()
    ax3.legend()

    # Create a legend for each colour/size bin - optionally include counts
    # using proxy artist rather than collection
    artists = [mpatch.Circle((0,0),fc=colour_list[bin], ec=colour_list[bin]) for bin in range(num_sizes)] # one representative patch from each bin
    if show_counts:
        diam_label_counts = [diam_label[bin] + ' [' + str(count[bin]) + ']' for bin in range(num_sizes)]
        fig3.legend(artists,diam_label_counts,title=u'Diam (µm) [Counts]',loc='right')
    else:
        fig3.legend(artists,diam_label,title=u'Diam (µm)',loc='right')

    if too_many:

        message = 'Warning: only %i particles per class are displayed!' % max_circles
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.6)
        ax3.text(scan_width*0.5,scan_height*0.95, message, horizontalalignment='center', fontsize=12, verticalalignment='center', bbox=props)

    fig3.suptitle(description[0] + '\n' + description[1])

    if filename !='':
        plotfile = filename+'_facet.png'
        plt.savefig(plotfile)
        plt.clf()
        plt.close()

    return particles, plotfile


def generate_scan(particles, width, height, xpixels, ypixels, xstep='', ystep='', filename=''):
    """Takes a set of particles (x,y,d) and samples given MIDAS scan parameters as input,
    assuming a delta-like AFM tip to output a BCR file.

    Usage: generate_scan(particles,width,height,xpixels,ypixels, xstep, ystep)

    where diameter is a list of mean diameters, and count the corresponding particle counts
    width and height describe the exposure size (microns)
    xpixels, ypixels = number of pixels to use over the scan area
    xstep, ystep (optional) = MIDAS step sizes = if not given, the closest integer to fill the exposure is used
    filename (optional): name of the BCR file to write

    A BCR data structure is returned (ready to be written or plotted using bcrutils).
    """

    # TODO: how do we deal with particles on top of particles?
    #       - could do true 3D and allow particles to stick
    #       - could find nearest adjacent free space
    #
    # TODO: need to find height of delta function at any given point
    #       - could generate in Gwyddion to get true AFM image
    #       - ideally convolute with an approximate tip shape to give real AFM image

    xstep_cal = common.xycal['open']/1.e3 # using the open loop calibration factor
    ystep_cal = common.xycal['open']/1.e3 # since that's all we have for the FM

    # If x and y steps not specified, calculate to fit the exposure area
    if (xstep == '' or ystep == ''):
        xstep = np.floor(float(width) / float(xpixels) / xstep_cal)
        ystep = np.floor(float(height) / float(ypixels) / ystep_cal)
        print('X and Y step sizes: ', str(xstep*xstep_cal), str(ystep*ystep_cal))
    else:
        # or check that the requested number of pixels and x/y step fits within the scan area
        scan_width = xstep_cal * xstep * xpixels # cal in nm/bit
        scan_height = ystep_cal * ystep * ypixels # cal in nm/bit

        if (scan_width > width) or (scan_height > height):
            print('Requested MIDAS scan size is larger than the calculated exposure size!')
            return 0

    # Initialise AFM height array
    afm_height = np.zeros((xpixels,ypixels),dtype=float)

    # Ensure that the list is sorted by diameter
    # particles.sort(key=lambda diam: diam[2])

    if len(particles) > 0:

        max_diam = max(particles[:,2]) # largest particle size
        max_rad = max_diam/2.

        # Loop over pixels and search for overlapping particles, calculating height of each
        for ycount in range(ypixels):
            y = ycount * ystep * ystep_cal
            for xcount in range(xpixels):
                close_pcles = []
                x = xcount * xstep * xstep_cal

                # Now for each pixel, find out which circular particles overlap this location, i.e.
                # only particles that have (pixel posn - pcle centre) < pcle radius contribute.
                # Only search a window around the current pixel bounded by max diameter/2. for speed

                xmask = np.logical_and(particles[:,0] > (x - max_rad), particles[:,0] < (x + max_rad))
                ymask = np.logical_and(particles[:,1] > (y - max_rad), particles[:,1] < (y + max_rad))
                [close_pcles.append(pcle) for pcle in particles[xmask&ymask] if ( (x-pcle[0])**2. + (y-pcle[1])**2. ) <= (pcle[2]/2.)**2.]
                close_pcles = np.array(close_pcles)

                # Note: this is still slow for thousands of particles and a 512x512 scan - consider a
                # better search algorithm, e.g. a quadtree?

                # Now have a list of particles that overlap with the current scan point - next calculate
                # the distance from the AFM point to the centre of the sphere...
                #
                if len(close_pcles) > 0:

                    distance = np.array( [np.sqrt( (x-pcle[0])**2. + (y-pcle[1])**2.) for pcle in close_pcles] )

                    # and now the height of the sphere with given radius at this distance
                    # equation of a circle at (0,0): x^2 + y^2 = r^2
		            # y = sqrt( r^2 - x^2 ), positive solution only
                    height = [ close_pcles[num,2]/2. + np.sqrt( (close_pcles[num,2]/2.)**2. - distance[num]**2. ) for num in range(len(close_pcles))]

                    # and the returned point is the highest of these
                    afm_height[xcount,ycount] = max(height)

    afm_height = afm_height.transpose()[::-1] # transpose rows and columns and mirror to match facet view

    # Now we have a height field, we can generate a BCR structure and return it
    nm2bit = (1./common.zcal)
    bcrdata = {}

    # Insert the minimum data necessary for a valid BCR file
    bcrdata['fileformat'] = 'bcrstm' # typically used by MIDAS
    bcrdata['xlength'] = (xpixels-1) * xstep * common.xycal['open']
    bcrdata['ylength'] = (ypixels-1) * ystep * common.xycal['open']
    bcrdata['xpixels'] = xpixels
    bcrdata['ypixels'] = ypixels
    bcrdata['xoffset'] = 0.0
    bcrdata['yoffset'] = 0.0
    bcrdata['scanspeed'] = 0.0
    bcrdata['bit2nm'] = common.zcal
    bcrdata['zunit'] = 'nm'
    if filename == '':
        bcrdata['filename'] = 'tempfile.bcr'
    else:
        bcrdata['filename'] = filename

    # Check if bit or little endian
    import sys
    if sys.byteorder=='little':
         bcrdata['intelmode'] = 1
    else:
         bcrdata['intelmode'] = 0

    # The data should be in integer form, using the MIDAS closed loop zcal factor
    bcrdata['data'] = np.array(np.floor(afm_height * 1.e3 * nm2bit),dtype=long).flatten()

    # M.S.Bentley 16/07/2012: the MIDAS cal factors are scaled for the piezo length
    # but we can generate particles here much larger than this, meaning that the
    # data no longer fits into a 16-bit integer.
    #
    # Need to be in the range −32768 to 32767
    #
    # For now, I'll cut off all data below the 65535 limit and set these pixels to the
    # lowest value (mimicking MIDAS behaviour)

    if bcrdata['data'].max() > 32768:

        # shift so that the highest height is +32766
        bcrdata['data'] = bcrdata['data'] - (bcrdata['data'].max() - 32766)

        # mask values below -32768 and set to +32766
        bcrdata['data'][bcrdata['data'] < -32768] = -32768

    return bcrdata


def plot_summary(input_file, output_path, isotropic=False, compact=True, fulle=True):

    import pandas as pd

    iso = 'isotropic' if isotropic else 'anisotropic'
    comp = 'compact' if compact else 'fluffy'
    extrapolation = 'linear' if fulle else 'divine'

    # Read in the results CSV file and plot the numbers of collected 1 micron particles
    # for each mission segment and for several scenarios over a 100 x 100 micron area
    df = pd.read_csv(input_file)
    upper_count = df[ (df.anisotropic == iso) & (df.upper_lower == 'upper') & \
            (df.fluffy_compact == comp)  & (df.divine_fulle == extrapolation) & \
            (df.scan_width == 100.) & (df.scan_height == 100.) ]

    lower_count = df[ (df.anisotropic == iso) & (df.upper_lower == 'lower') & \
            (df.fluffy_compact == comp)  & (df.divine_fulle == extrapolation) & \
            (df.scan_width == 100.) & (df.scan_height == 100.) ]

    from pylab import rcParamsDefault
    fig_size = rcParamsDefault['figure.figsize']
    new_size=(fig_size[0]*2.5,fig_size[1])
    sum_fig = plt.figure(figsize=new_size)
    sum_ax = sum_fig.add_subplot(1,1,1)

    weeknum=[float(label.split('_')[-1])-0.5 for label in upper_count.run_name]
    sum_ax.bar(weeknum,upper_count.pcle_count,width=1.,label='upper',color='blue')
    sum_ax.bar(weeknum,lower_count.pcle_count,width=1.,label='lower',color='red')
    sum_ax.set_xlabel('Week number')
    sum_ax.set_ylabel('# particles collected')
    sum_ax.set_xlim(weeknum[0],weeknum[-1]+1.)
    sum_ax.grid(True)
    sum_ax.legend(loc=0)
    sum_ax.set_title('~1 micron diameter particles collected in a 100x100 micron area\n%s, %s, %s' % (comp, extrapolation, iso))
    summary_plot_file = 'bone_summary_%s_%s_%s.png' % (comp, extrapolation, iso)
    summary_plot = os.path.join(output_path,summary_plot_file)
    plt.savefig(summary_plot)
    plt.clf()
    plt.close()

    # Also plot this as a cumulative distribution

    cum_fig = plt.figure(figsize=new_size)
    cum_ax = cum_fig.add_subplot(1,1,1)

    cum_ax.bar(weeknum,upper_count.pcle_count.cumsum(),width=1.,label='upper',color='blue')
    cum_ax.bar(weeknum,lower_count.pcle_count.cumsum(),width=1.,label='lower',color='red')
    cum_ax.set_xlabel('Week number')
    cum_ax.set_ylabel('Cumulative # particles collected')
    cum_ax.set_xlim(weeknum[0],weeknum[-1]+1.)
    cum_ax.grid(True)
    cum_ax.legend(loc=0)
    cum_ax.set_title('~1 micron diameter particles collected in a 100x100 micron area\n%s, %s, %s' % (comp, extrapolation, iso))
    cumulative_plot_file = 'bone_summary_cumulative_%s_%s_%s.png' % (comp, extrapolation, iso)
    cumulative_plot = os.path.join(output_path,cumulative_plot_file)
    plt.savefig(cumulative_plot)
    plt.clf()
    plt.close()

    return summary_plot_file, cumulative_plot_file


#------------------------

def flyby():

    current_dir = os.getcwd()
#    kernel_path = '/home/mark/work/kernels' # location of SPICE kernels
#    output_path = '/home/mark/work/skeleton_planning' # location for output (html, png, bcr, dat files)
    kernel_path = '/media/uhura/Software/spice/kernels' # location of SPICE kernels
    output_path = '/media/uhura/DOC/Operatio/skeleton_planning/M1-M14_HAC' # location for output (html, png, bcr, dat files)

    #--------------- bone selection -----------

    # Create a dictionary of cases from which we can choose, or iterate through...


    # Generate bone cases by splitting kernel into weekly segments

#    bone_meta_case = \
#        { 'start_time' : '2014 NOV 19 00:01:06.182',
#        'end_time' : '2015 DEC 02 00:00:00.00',
#        'meta_kernel' : 'roviz_t124_p000_c000_s000_g000_r003_a0_v000.TM',
#        'run_prefix' : 'M1-M14_HAC',
#        'comment' : 'HAC' }


# LAC
# BSP - 2014 NOV 19 01:01:06.182            2015 APR 08 01:01:06.182
# CK  - 2014-NOV-19 01:01:45.259            2015-APR-08 01:01:46.065

    bone_meta_case = \
       { 'start_time' : '2014-NOV-19 01:01:45.259',
       'end_time' : '2015 APR 08 01:01:06.182',
       'meta_kernel' : 'roviz_lac_m1-m5_2013-0712.tm',
       'run_prefix' : 'LAC',
       'comment' : 'LAC' }

# HAC
# TRAL_DL_002_01_c1_H.bsp 2014 NOV 19 01:01:07.182 - 2015 MAR 11 01:01:07.182
# CK                      2014-NOV-19 01:01:46.259 - 2015-MAR-11 01:01:46.904

    bone_meta_case = \
      { 'start_time' : '2014-NOV-19 01:01:46.259',
      'end_time' : '2015 MAR 11 01:01:07.182',
      'meta_kernel' : 'TRAL_DL_002_01_c1_H.TM',
      'run_prefix' : 'HAC',
      'comment' : 'HAC' }


    os.chdir(kernel_path) # SPICE looks for the kernels in the current directory
    spice.furnsh(bone_meta_case['meta_kernel'])
    start_time_et = spice.str2et(bone_meta_case['start_time'])
    end_time_et = spice.str2et(bone_meta_case['end_time'])

    duration = end_time_et - start_time_et
    # print 'DEBUG: duration = %f seconds' % (duration)
    num_weeks = np.round(duration / (7. * 24. * 60. * 60.))
    print 'Number of weeks in meta case: %i' % (num_weeks)
    bone_cases = []

    time_fmt = 'HR:MN:SC.### Mon DD, YYYY ::RND'

    for week in np.arange(0,num_weeks):
        month_num = int((week)/4.)+1
        week_num = week + 1
        if (week == (num_weeks-1)):
            # print 'DEBUG: final week, setting end time to end of meta case'
            bone = { 'start_time' : spice.timout(start_time_et + week * 7.*24.*60.*60.,time_fmt),
                'end_time' : bone_meta_case['end_time'],
                'meta_kernel' : bone_meta_case['meta_kernel'],
                'run_prefix' : bone_meta_case['run_prefix'] + '_week_%i' % (week_num),
                'comment' : 'M%i' % (month_num) + ' ' + bone_meta_case['comment'] + ', week %i' % (week_num) }
        else:
            bone = { 'start_time' : spice.timout(start_time_et + week * 7.*24.*60.*60.,time_fmt),
                'end_time' : spice.timout(start_time_et + (week+1) * 7.*24.*60.*60.,time_fmt),
                'meta_kernel' : bone_meta_case['meta_kernel'],
                'run_prefix' : bone_meta_case['run_prefix'] + '_week_%i' % (week_num),
                'comment' : 'M%i' % (month_num) + ' ' + bone_meta_case['comment'] + ', week %i' % (week_num) }
        # print 'DEBUG: Start time: %s, stop time: %s' % (bone['start_time'],bone['end_time'])
        bone_cases.append(bone)

    spice.unload(bone_meta_case['meta_kernel'])
    os.chdir(current_dir)

    single_cases = [

    { 'start_time' : '2014-09-01T00:00:00',
    'end_time' : '2014-12-17T04:00:00',
    'meta_kernel' : 'MET-prelanding_2013-02-22_ORHR_130206_CompleteTrajectory_orbiter_u002_1.TM',
    'run_prefix' : 'pre-landing-fdyn-nadir',
    'comment' : 'Pre-landing trajectory from FDyn, nadir pointing' } ]

    # [bone_cases.append(bone) for bone in single_cases]

    test = [

    ###### Phase A example trajectories (per bone)

#        { 'start_time' : '2014 NOV 19 00:01:06.182',
#        'end_time' : '2015 JAN 14 00:01:06.182',
        { 'start_time' : '2014-11-19T00:00:40',
        'end_time' : '2015-01-13T23:00:40',
        'meta_kernel' : 'roviz_t115_p000_c000_s000_g000_r003_a0_v000.TM',
        'run_prefix' : 'M1_M2_example',
        'comment' : 'M1 + M2 icosahedral orbits + 1 flyby' },
        { 'start_time' : '2014 DEC 25 18:01:06.182',
        'end_time' : '2014 DEC 29 06:01:06.182',
        'meta_kernel' : 'roviz_t115_p000_c000_s000_g000_r003_a0_v000.TM',
        'run_prefix' : 'M2_icosahedral_3_5d',
        'comment' : 'M2 icosahedral segment, 3.5 days' },

        { 'start_time' : '2015 JAN 03 00:01:06.182',
        'end_time' : '2015 JAN 04 00:01:06.182',
        'meta_kernel' : 'roviz_t115_p000_c000_s000_g000_r003_a0_v000.TM',
        'run_prefix' : 'M2_flyby_1_0d',
        'comment' : 'M2 flyby, 1 day' },

        { 'start_time' : '2015 JAN 01 18:01:06.182',
        'end_time' : '2015 JAN 05 06:01:06.182',
        'meta_kernel' : 'roviz_t115_p000_c000_s000_g000_r003_a0_v000.TM',
        'run_prefix' : 'M2_flyby_3_5d',
        'comment' : 'M2 flyby, 3.5 days' },

        { 'start_time' : '2014 DEC 24 00:01:06.182',
        'end_time' : '2014 DEC 31 00:01:06.182',
        'meta_kernel' : 'roviz_t115_p000_c000_s000_g000_r003_a0_v000.TM',
        'run_prefix' : 'M2_icosahedral_7d',
        'comment' : 'M2 icosahedral segment, 7 days' },

        { 'start_time' : '2014 DEC 31 00:01:06.182',
        'end_time' : '2015 JAN 07 00:01:06.182',
        'meta_kernel' : 'roviz_t115_p000_c000_s000_g000_r003_a0_v000.TM',
        'run_prefix' : 'M2_flyby_7d',
        'comment' : 'M2 flyby, 7 days' },

        { 'start_time' : '2014 NOV 19 00:01:06.182',
        'end_time' : '2014 NOV 26 00:01:06.182',
        'meta_kernel' : 'roviz_t115_p000_c000_s000_g000_r003_a0_v000.TM',
        'run_prefix' : 'M1_M2-week-1',
        'comment' : 'M1-M2 example trajectory, week 1' },

        { 'start_time' : '2014 NOV 26 00:01:06.182',
        'end_time' : '2014 DEC 03 00:01:06.182',
        'meta_kernel' : 'roviz_t115_p000_c000_s000_g000_r003_a0_v000.TM',
        'run_prefix' : 'M1_M2-week-2',
        'comment' : 'M1-M2 example trajectory, week 2' },

        { 'start_time' : '2014 DEC 03 00:01:06.182',
        'end_time' : '2014 DEC 10 00:01:06.182',
        'meta_kernel' : 'roviz_t115_p000_c000_s000_g000_r003_a0_v000.TM',
        'run_prefix' : 'M1_M2-week-3',
        'comment' : 'M1-M2 example trajectory, week 3' },

        { 'start_time' : '2014 DEC 10 00:01:06.182',
        'end_time' : '2014 DEC 17 00:01:06.182',
        'meta_kernel' : 'roviz_t115_p000_c000_s000_g000_r003_a0_v000.TM',
        'run_prefix' : 'M1_M2-week-4',
        'comment' : 'M1-M2 example trajectory, week 4' },

        { 'start_time' : '2014 DEC 17 00:01:06.182',
        'end_time' : '2014 DEC 24 00:01:06.182',
        'meta_kernel' : 'roviz_t115_p000_c000_s000_g000_r003_a0_v000.TM',
        'run_prefix' : 'M1_M2-week-5',
        'comment' : 'M1-M2 example trajectory, week 5' },

        { 'start_time' : '2014 DEC 24 00:01:06.182',
        'end_time' : '2014 DEC 31 00:01:06.182',
        'meta_kernel' : 'roviz_t115_p000_c000_s000_g000_r003_a0_v000.TM',
        'run_prefix' : 'M1_M2-week-6',
        'comment' : 'M1-M2 example trajectory, week 6' },

        { 'start_time' : '2014 DEC 31 00:01:06.182',
        'end_time' : '2015 JAN 07 00:01:06.182',
        'meta_kernel' : 'roviz_t115_p000_c000_s000_g000_r003_a0_v000.TM',
        'run_prefix' : 'M1_M2-week-7',
        'comment' : 'M1-M2 example trajectory, week 7' },

        { 'start_time' : '2015 JAN 07 00:01:06.182',
        'end_time' : '2015 JAN 14 00:01:06.182',
        'meta_kernel' : 'roviz_t115_p000_c000_s000_g000_r003_a0_v000.TM',
        'run_prefix' : 'M1_M2-week-8',
        'comment' : 'M1-M2 example trajectory, week 8' } ]

        ###### Pre-landing trajectory, incomplete, from FDyn

    test = [
         ## Early mission icosahedron segments, no flyby

        { 'start_time' : '2014 DEC 24 00:01:06.183',
        'end_time' : '2015 FEB 25 00:01:06.183',
        'meta_kernel' : 'icosahedral_orbit_2013-01-22.TM',
        'run_prefix' : 'icosahedral_first_9_wks',
        'comment' : 'Icosahedral orbit test case (first 9 weeks), no flyby' },

#        { 'start_time' : '2014-09-11T00:00',
#        'end_time' : '2014-10-31T00:00',
#        { 'start_time' : '2014-09-01T00:00:00',
#        'end_time' : '2014-12-17T04:00:00',
#        'meta_kernel' : 'ORHR_121217_CompleteTrajectory_orbiter.TM',
#        'run_prefix' : 'pre-landing-fdyn-nadir',
#        'comment' : 'Pre-landing trajectory from FDyn, nadir pointing' },

        ## Phase A example (entire phase, example trajectories)

        { 'start_time' : '2014-NOV-15 00:01:45.235',
        'end_time' : '2014 DEC 15 00:01:06.182',
        'meta_kernel' : 'MET-phase_a_30km_dawn_nadir.TM',
        'run_prefix' : 'phase_a_example_v003_bone_a1',
        'comment' : 'Bone A1, Phase A example V003 30 km orbit' },

        { 'start_time' : '2014-DEC-15 00:01:45.409',
        'end_time' : '2015 JAN 15 00:01:06.183',
        'meta_kernel' : 'MET-phase_a_30km_dawn_nadir.TM',
        'run_prefix' : 'phase_a_example_v003_bone_a2',
        'comment' : 'Bone A2, Phase A example V003 30 km orbit' },

        { 'start_time' : '2015-01-15T00:00',
        'end_time' : '2015-02-15T00:00',
        'meta_kernel' : 'MET-A3-A-00-V01.TM',
        'meta_kernel' : 'MET-phase_a_30km_dawn_nadir.TM',
        'run_prefix' : 'phase_a_example_v003_bone_a3',
        'comment' : 'Bone A3, Phase A example V003 30 km orbit' },

        { 'start_time' : '2015-02-15T00:00',
        'end_time' : '2015 MAR 15 00:01:06.182',
        'meta_kernel' : 'MET-phase_a_30km_dawn_nadir.TM',
        'run_prefix' : 'phase_a_example_v003_bone_a4',
        'comment' : 'Bone A4, Phase A example V003 30 km orbit' },


        ## Phase A example (entire phase, example trajectories)

        { 'start_time' : '2014-NOV-15 00:01:45.235',
        'end_time' : '2015 MAR 15 00:01:06.182',
        'meta_kernel' : 'MET-roviz_t104_p000_c000_s000_g000_r003_a0_v000_u002_V001.TM',
        'run_prefix' : 'phase_a_example_v001',
        'comment' : 'Phase A example trajectory V001, 2x flybys, 20 km' },

        { 'start_time' : '2014-NOV-15 00:01:45.235',
        'end_time' : '2015 MAR 15 00:01:06.182',
        'meta_kernel' : 'MET-roviz_t104_p000_c000_s000_g000_r003_a0_v000_u002_V002.TM',
        'run_prefix' : 'phase_a_example_20km_v002',
        'comment' : 'Phase A example trajectory V002, no flybys, 20 km' },

        { 'start_time' : '2014-NOV-15 00:01:45.235',
        'end_time' : '2015 MAR 15 00:01:06.182',
        'meta_kernel' : 'MET-phase_a_30km_dawn_nadir.TM',
        'run_prefix' : 'phase_a_example_30km_v003',
        'comment' : 'Phase A example trajectory V003, no flybys, 30 km' },

        ###### Wake-up to Lander Delivery

        { 'start_time' : '2014-09-11T00:00',
        'end_time' : '2014-10-31T00:00',
        'meta_kernel' : 'MET-ORHR_orbiter_11nov_u002_COP.TM',
        'run_prefix' : 'close_observation',
        'comment' : 'ORHR_orbiter_11nov_u002 COP (Close Observation Phase) (nadir)' },

        ###### Skeleton Plan Bones

        ### Phase A

        # Bone A1-A

        { 'start_time' : '2014-NOV-15 00:01:45.235',
        'end_time' : '2014 DEC 15 00:01:06.182',
        'meta_kernel' : 'MET-A1-A-01-V03.TM',
        'run_prefix' : 'bone_a1-a',
        'comment' : 'Bone A1-A, with pointing profile' },

        { 'start_time' : '2014-NOV-15 00:01:45.235',
        'end_time' : '2014 DEC 15 00:01:06.182',
        'meta_kernel' : 'MET-A1-A-01-V03.TM',
        'run_prefix' : 'bone_a1-a-nadir',
        'comment' : 'Bone A1-A, nadir pointing' },

        # Bone A2-A

        { 'start_time' : '2014-DEC-15 00:01:45.409',
        'end_time' : '2015 JAN 15 00:01:06.183',
        'meta_kernel' : 'MET-A2-A-00-V01.TM',
        'run_prefix' : 'bone_a2-a-nadir',
        'comment' : 'Bone A2-A, nadir pointing' },

        { 'start_time' : '2014-DEC-15 00:01:45.409',
        'end_time' : '2015 JAN 15 00:01:06.183',
        'meta_kernel' : 'MET-A2-A-01-V01.TM',
        'run_prefix' : 'bone_a2-a',
        'comment' : 'Bone A2-A, with pointing profile' },

        # Bone A3-A

        { 'start_time' : '2015-01-15T00:00',
        'end_time' : '2015-02-15T00:00',
        'meta_kernel' : 'MET-A3-A-00-V01.TM',
        'run_prefix' : 'bone_a3-a',
        'comment' : 'Bone A3-A-00-V01, 10 km orbit with 20 deg tilt for corotational flyby, nadir' },

        # Bone A3-B

        { 'start_time' : '2015-01-15T00:00',
        'end_time' : '2015-02-15T00:00',
        'meta_kernel' : 'MET-A3-B-00-V02.TM',
        'run_prefix' : 'bone_a3-b',
        'comment' : 'Bone Bone A3-B-00-V02, 10 km orbit with 20 deg tilt, corotational flyby, return to orbit, nadir' },

        ### Phase B

        # Bone B1-A

        { 'start_time' : '2015 MAR 27 23:01:46.004',
        'end_time' : '2015-APR-05 16:01:46.054',
        'meta_kernel' : 'MET-B1-A-00-V01.TM',
        'run_prefix' : 'bone_b1-a00',
        'comment' : 'Bone B1-A-00 (roviz_2011-1028-1315_u002) (nadir)' },

        { 'start_time' : '2015 MAR 27 23:01:46.004',
        'end_time' : '2015-APR-05 16:01:46.054',
        'meta_kernel' : 'MET-B1-A-01-V01.TM',
        'run_prefix' : 'bone_b1-a01',
        'comment' : 'Bone B1-A-01 (roviz_2011-1028-1315_u000_V0002)' },

        ###### Icosahedral Segments

        { 'start_time' : '2015-JUL-13 19:11:46.624',
        'end_time' : '2015-SEP-12 04:00:0.0',
        'meta_kernel' : 'MET-icosahedral_orbit_u002.TM',
        'run_prefix' : 'icosahedral',
        'comment' : 'Icosahedral segments' } ]


    ###### end of bone case definition



    #--------------- exposure and scan parameters -----------

    # Should we assume nadir pointing, or take data from the SPICE kernel?
    use_pointing = True

    # GIADA target area (1 dm2) for Service 19 comparison
    giada_width = 1.e5 # 10 cm
    giada_height = 1.e5 # 10 cm
    giada_timestep = 60. # 1 min

    # COSIMA target size 1cm2 square: 1e4 x 1e4 µm
    cosima = False

    if cosima:
        scan_width_list = [10000.] # microns
        scan_height_list = [10000.] # microns
    else:
#        scan_width_list = [1000., 1400., 1400., 100., 40., 10.] # microns
#        scan_height_list = [1000., 2400., 94., 100., 40., 10.] # microns
        scan_width_list = [100.] # microns
        scan_height_list = [100.] # microns

    exposure_fraction = 0.5 # fraction of time spent exposing (centred on CA)
    show_counts = True # display counts on the graphs
    write_bcr = False # output simulated scan BCRs?
    facet_view = True # generate facets view images?

    # Open a data file to write the results to
    summary_path = os.path.join(output_path,'bone_summary.csv')
    print 'Writing summary to file: %s' % (summary_path)
    summary_file = open(summary_path, 'w')

    # Write the header
    summary_file.write('run_name,scan_width,scan_height,fluffy_compact,divine_fulle,upper_lower,anisotropic,pcle_count\n')

    # Also write a root html file linking to all of the cases - start by copying a template and opening it for append
    template = os.path.join(output_path,'template.html')
    index = os.path.join(output_path,'index.html')
    import shutil
    shutil.copyfile(template, index)
    summary_html = open(index,'a')


    #------ Loop over all bones in the list (for now)

    # TODO: refactor to break out the content below to separate calls

    for bone in bone_cases:
        start_time = bone['start_time']
        end_time = bone['end_time']
        meta_kernel = bone['meta_kernel']
        run_prefix = bone['run_prefix']
        comment = bone['comment']

        # Create a directory for this bone / scenario
        bone_path = os.path.join(output_path,run_prefix)
        path_exists(bone_path)

        # Create in index.html file in this directory for storing results
        bone_html = open(os.path.join(bone_path,'index.html'),'w')

        # Write some basic bone / run information
        bone_html.write('''
    <h1 align=center>%s</h1>
    <p>Start time: %s<br>
    <p>End time: %s<br>
    <p>Meta kernel: %s<br>''' % (comment, start_time, end_time, meta_kernel))

        # Link to the orbit plots
        bone_html.write('''
    <p>&nbsp;</p>
    <p>&nbsp;</p>
    <table width=85%% border=0 align=center>
        <tr>
            <th colspan=3 align=center>Orbital summary plots (click to enlarge)</th>
        </tr>
        <tr>
            <td width=33%% align=center><a href=%s><img src=%s width=300></a></td>
            <td width=33%% align=center><a href=%s><img src=%s width=300></a></td>
            <td width=33%% align=center><a href=%s><img src=%s width=300></a></td>
        </tr>
    </table>''' % (run_prefix+'_comet_dist.png',run_prefix+'_comet_dist.png',run_prefix+'_sc_dist.png',run_prefix+'_sc_dist.png',run_prefix+'_sectors.png',run_prefix+'_sectors.png'))

        # Start the summary table

        bone_html.write('''
    <table width=85% border=1 cellspacing=1 cellpadding=1 align=center>
    <tr>
        <th>Width (um)</th> <th>Height (um)</th> <th>Density</th> <th>Extrapolation</th> <th>Limit</th> <th>Anisotropy</th> <th>1 &micro;m count</th> <th>Flux</th> <th>Accumulated</th> <th>Histogram</th> <th>Facet</th> <th>GIADA count</th>
    </tr>''')


        # Run for each of the possible scenarios of particle density, observational limit and
        # mass index for small particles...

        print 'Running bone: %s, start time: %s, stop time: %s' % (comment, start_time, end_time)

        # Use the SPICE kernels to iterate through the specified trajectory and return distance, speed etc.
        os.chdir(kernel_path) # SPICE looks for the kernels in the current directory
        temp_path = os.path.join(bone_path,run_prefix)

        # Make a sane choice of timestep (60 s for durations < 2 days, otherwise 1 hour)
        # need to load a leapsecond kernel
        if type(start_time) == str: # assume a time string
            spice.furnsh('naif0009.tls')
            start_time_et = spice.str2et(start_time)
            end_time_et = spice.str2et(end_time)
            spice.unload('naif0009.tls')
        else:
            end_time_et = end_time
            start_time_et = start_time

        duration = end_time_et - start_time_et
        # if duration < (2.*60.*60.*24.):
        #     timestep = 60. # s
        # else:
        #     timestep = 60. * 60. # s

        timestep = 60. # s - fixing for now to match GIADA service 19

        timesteps, times, distance, speed, cometdist_au, off_nadir, sector, phase_angle = run_flyby(meta_kernel, start_time, end_time, timestep, comment, temp_path, pointing=use_pointing)
        os.chdir(current_dir)

        # Convert the times into managable units
        days = times/60.0/60.0/24.0
        days = days - days.min()
        hours = days*24.0

        # Expose for exposure_fraction of the time, centred on the CA
        # (this assumes each leg has an ico or flyby CA)

        # Find closest approach time/position
        ca_index=np.where(min(distance)==distance)[0][0]
        ca_time=times[ca_index]
        ca_day=days[ca_index]
        ca_hour=hours[ca_index]

        exposure_time = exposure_fraction * duration
        exposure_timesteps = int(round(exposure_time / timestep))
        exposure_start = ca_index - int(round(exposure_timesteps/2.))
        exposure_stop = ca_index + int(round(exposure_timesteps/2.))

        if (exposure_start < 0):
            exposure_start = 0
            exposure_stop = exposure_timesteps-1
        elif (exposure_stop >= len(distance)):
            exposure_start = len(distance)-exposure_timesteps
            exposure_stop = len(distance)-1

        # Plot the distance/time curve for this trajectory segment
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(days,distance)
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Spacecraft distance (km)')
        ax1.grid(True)
        ax1.set_title(comment)
        # plot a shaded region to show where we are exposing
        ax1.axvspan(days[exposure_start],days[exposure_stop], alpha=0.5)
        plt.savefig(os.path.join(bone_path,run_prefix+'_'+'sc_dist.png'))
        plt.clf()
        plt.close()

        # and the comet-Sun distance
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(days,cometdist_au)
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Comet distance (AU)')
        ax1.grid(True)
        ax1.set_title(comment)
        plt.savefig(os.path.join(bone_path,run_prefix+'_'+'comet_dist.png'))
        plt.clf()
        plt.close()

        # and phase angle
        fig = plt.figure()
        ax1 = fig.add_subplot(1,1,1)
        ax1.plot(days,np.degrees(phase_angle))
        ax1.set_xlabel('Time (days)')
        ax1.set_ylabel('Phase angle')
        ax1.grid(True)
        ax1.set_title(comment)
        plt.savefig(os.path.join(bone_path,run_prefix+'_'+'phase_angle.png'))
        plt.clf()
        plt.close()

        #### Now run the 8 parameter combinations for each trajectory

        bool_list = [True, False]
        upper_lower_list = ['upper','lower']

        for dimension in range(len(scan_width_list)):

            scan_width = scan_width_list[dimension]
            scan_height = scan_height_list[dimension]

            for fluffy in bool_list:
                for divine in bool_list:
                    for upper_lower in upper_lower_list:
                        for anisotropic in bool_list:

                            # Generate filenames and descriptions for the current scenario
                            if fluffy:
                                fluffytext = 'fluffy'
                            else:
                                fluffytext = 'compact'

                            if divine:
                                divinetext = 'divine'
                            else:
                                divinetext = 'linear'

                            if anisotropic:
                                anitext = 'anisotropic'
                            else:
                                anitext = 'isotropic'

                            descript1 = fluffytext + ' particles, ' + upper_lower + ' limit ' + 'with ' + divinetext + ' extrapolation, ' + anitext
                            descript2 = 'scan area ' + str(scan_width) + u' µm x ' + str(scan_height) + u' µm'
                            description = (descript1,descript2)
                            run_suffix = fluffytext + '_' + upper_lower + '_' + divinetext + '_' + anitext + '_' + str(int(scan_width)) + 'x' + str(int(scan_height))
                            filename = os.path.join(bone_path,run_prefix + '_' + run_suffix)


                            #--------------- calculate dustflux -----------

                            # Now calculate the flux of particles collected in this trajectory, given the above inputs
                            bin_edges_mass, mean_mass, mean_diam, pcles_scan_timestep, cumulative, fluxfile = \
                                calculate_dustflux(timestep, days[exposure_start:exposure_stop], cometdist_au[exposure_start:exposure_stop], \
                                distance[exposure_start:exposure_stop], off_nadir[exposure_start:exposure_stop], \
                                sector[exposure_start:exposure_stop], \
                                scan_width, scan_height, fluffy, upper_lower, divine, anisotropic, description, True, filename)

                            # Scale up the appropriate bin for the GIADA GDS+IS size
                            giada_size_factor = (giada_width*giada_height) / (scan_width*scan_height)
                            giada_time_factor = giada_timestep / timestep

                            if fluffy:
                                giada_count = cumulative[-1,8] * giada_size_factor * giada_time_factor
                                giada_diam = mean_diam[8]
                            else:
                                giada_count = cumulative[-1,9] * giada_size_factor * giada_time_factor
                                giada_diam = mean_diam[9]
                            # print 'GIADA count = %i for a diameter of %3.1f microns' % (giada_count, giada_diam*1.e6)

                            # Plot the GIADA service (19,12) counts
                            fig = plt.figure()
                            ax1 = fig.add_subplot(1,1,1)
                            if fluffy:
                                ax1.plot(days[exposure_start:exposure_stop],np.round(pcles_scan_timestep[8,:]* giada_size_factor * giada_time_factor))
                            else:
                                ax1.plot(days[exposure_start:exposure_stop],np.round(pcles_scan_timestep[9,:]* giada_size_factor * giada_time_factor))
                            ax1.set_xlabel('Time (days)')
                            ax1.set_ylabel('Particles per timestep (%3.2f)' % (timestep))
                            ax1.grid(True)
                            ax1.set_title(descript1 + '\nGIADA GDS+IS count in dm2/min (service 19,12)')
                            giadafile = os.path.join(bone_path,filename+'_giada.png')
                            plt.savefig(giadafile)
                            plt.clf()
                            plt.close()

                            # if num_scans > 0:
                            #     scan_times = scan_times[valid_scans[0]]
                            # else:
                            #     scan_times = []
                            scan_times = []

                            #--------------- plot representative details -----------

                            # Plot the flux in each mass bin over time
                            accumfile = plot_flux(days[exposure_start:exposure_stop], mean_diam, cumulative, scan_times, description, filename)

                            # Plot the histogram of cumulative numbers of collected particles
                            histfile = plot_histogram(bin_edges_mass, mean_diam, cumulative, fluffy, description, filename)

                            # Produce an integer count of particles collected per bin
                            final_count = np.array(np.floor(cumulative[-1,:]),dtype=long)
                            num_sizes = (final_count != 0).sum() # number of bins with non-zero counts
                            diameter = np.array(mean_diam[0:num_sizes],dtype=float) * 1.e6 # microns
                            count = final_count[0:num_sizes]

                            # View this scan area (view_facet_histogram and view_facet_particles take histo and defined positions,
                            # respectively, and then call the same display routine)

                            if facet_view:
                                # particles, facetfile = view_facet(scan_width, scan_height, diameter, count, description, show_counts, filename, xpixels = 128, ypixels=128)
                                particles, facetfile = view_facet(scan_width, scan_height, diameter, count, description, show_counts, filename)
                            else:
                                particles = False
                                facetfile = ''

                            # Write the counts for ~1 µm particles to a CSV file to track particle collection per bone
                            #
                            # Compact
                            # 1e-16 - 1e-15 kg =  0.822 µm mean diameter (0.872 µm with Divine)
                            # 1e-15 - 1e-14 kg =  1.771 µm mean diameter (1.867 µm with Divine)
                            #
                            # bin_edges_mass[4] = 1e-16
                            # bin_edges_mass[5] = 1e-15
                            #
                            # Fluffy - need bins [3] and [4] instead

                            if fluffy:

                                if cosima:
                                    count_1um = final_count[6] # ~ 10 µm particle size for COSIMA
                                else:
                                    count_1um = final_count[3] # +final_count[4]
                            else:

           	                if cosima:
                                    count_1um = final_count[7] # ~ 10 µm particle size for COSIMA
                                else:
                                    count_1um = final_count[4] # +final_count[5]

                            output_txt = ('%s,%.2f,%.2f,%s,%s,%s,%s,%d' % (run_prefix, scan_width, scan_height, fluffytext, divinetext, upper_lower, anitext, count_1um))
                            print output_txt
                            summary_file.write(output_txt+'\n')

                            # For a "standard" case, write the 1 micron count to the summary table

                            if scan_width == 100. and scan_height == 100. and \
                                anisotropic and (not fluffy) and (not divine) and upper_lower == 'upper':
                                upper_count = count_1um

                            if scan_width == 100. and scan_height == 100. and \
                                anisotropic and (not fluffy) and (not divine) and upper_lower == 'lower':
                                lower_count = count_1um

                            # want relative pathnames here
                            fluxfile = os.path.basename(fluxfile)
                            accumfile = os.path.basename(accumfile)
                            histfile = os.path.basename(histfile)
                            facetfile = os.path.basename(facetfile)
                            giadafile = os.path.basename(giadafile)

                            ##### write the key data to an html table

                            bone_html.write('''
            <tr>
                <td>%.2f</td> <td>%.2f</td> <td>%s</td> <td>%s</td> <td>%s</td> <td>%s</td> <td>%d</td> <td><a href=%s>link</a></td> <td><a href=%s>link</a></td> <td><a href=%s>link</a></td> <td><a href=%s>link</a></td><td><a href=%s>link</a></td>
            </tr>''' % (scan_width, scan_height, fluffytext, divinetext, upper_lower, anitext, count_1um, fluxfile, accumfile, histfile, facetfile,giadafile) )

                            #--------------- simulate a scan and write a BCR file -----------

                            # flatten the particle list (currently a list of lists) as input to the scan tool
                            if particles:

                                particle_array = np.array([item for sublist in particles for item in sublist])
                                # particle_array = np.array(particles)

                                # Generate .bcr files equivalent to the "first scan" sequence - so incremental zooms of
                                # the same generated facet: this means 40 µm, 10 µm and 1 µm each at 128x128 pixels
                                #
                                # particle_array is a numpy array containing an array of (x,y,d) data
                                #
                                # We need to generate an array which contains data for xlo<x<xhi and ylo<y<yhi with the
                                # same form, then generate BCRs for that.
                                #
                                if write_bcr:
                                    scan_sizes = [100.0, 10.0, 1.0]
                                    scan_npix = 128
                                    for size in scan_sizes:
                                        if (scan_width < size) or (scan_height < size):
                                            continue
                                        else:
                                            # zoom on the centre of the larger scan
                                            xlo = (scan_width/2.0) - (size/2.0)
                                            xhi = (scan_width/2.0) + (size/2.0)
                                            ylo = (scan_height/2.0) - (size/2.0)
                                            yhi = (scan_height/2.0) + (size/2.0)

                                            # extract the particles matching these xlo/xhi, ylo/yhi criteria
                                            zoom = particle_array[(particle_array[:,0] > xlo) & (particle_array[:,0] < xhi) & (particle_array[:,1] > ylo) & (particle_array[:,1] < yhi)]
                                            # shift to zero for BCR generation
                                            zoom[:,0] -= xlo
                                            zoom[:,1] -= ylo

                                            scan_filename = filename +'_'+str(scan_npix)+'_pix'
                                            bcr = generate_scan(zoom, size, size, scan_npix, scan_npix, xstep='', ystep='', filename=scan_filename+'.bcr')
                                            bcrutils.plot2d(bcr,description[0]+'\n',writefile=True)
                                            # plt.savefig(scan_filename+'.png')
                                            bcrutils.write(bcr)

        # Add an entry to the index file
        summary_html.write('''
            <tr>
                <td><a href=%s>%s</a></td><td align=center>%i</td><td align=center>%i</td><td>%s</td><td>%s</td>
            </tr>''' % (os.path.join(os.path.relpath(bone_path,output_path),'index.html'), comment, lower_count, upper_count, start_time, end_time))


        bone_html.write('''
        </table>
        <p>
        <p align=center><a href="../index.html">return to the case list</a></p>
    </body>
</html>''')

    summary_file.close()

    import time
    summary_html.write('''
        </table>

        <h2>Summary</h2>''')

    # Read in the results CSV file and plot the numbers of collected 1 micron particles
    # for each mission segment and for several scenarios over a 100 x 100 micron area
    summary, summary_cum = plot_summary(summary_path, output_path)

    summary_html.write('''
        <p>Summary plots for the anisotropic case, using Fulle upper/lower limits with ~1 micron compact particles, linear extrapolation:</p>
        <center><a href=./%s><img align=center width=800 src=./%s></a></center>
        <center><a href=./%s><img align=center width=800 src=./%s></a></center>

        <p>%d test cases generated at %s</p>
        <p></p>
        <p>A summary of results from all cases can be downloaded as a CSV file: <a href=./bone_summary.csv>bone_summary.csv</a></p>

    </body>
</html>''' % (summary, summary, summary_cum, summary_cum, len(bone_cases),time.strftime("%a, %d %b %Y %H:%M:%S")) )
    summary_html.close()


if __name__ == '__main__':

    # if running on the server (no X display), set the backend to Agg
    # if not os.environ.has_key("DISPLAY"):
    #     import matplotlib as mpl
    #     mpl.use('Agg')

    flyby()
