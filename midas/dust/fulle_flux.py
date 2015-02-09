# -*- coding: utf-8 -*-
#!/usr/bin/python
"""
fulle_flux.py - provides interpolated values of dust flux as a function of
cometocentric and helicentric distance. All values are interpolated from
Fulle (2010).
"""

import fulle_data as data
from dustutils import *
import matplotlib.pyplot as plt
import numpy as np
import math


def summarise_cdf(fluffy=False, upper_lower='upper', divine=True):
    """Summarise the cumulative distribution function for various cometocentric distances"""

    # Set up some values at which to evaluate the CDF for binning
    min_mass = -22
    max_mass = -3
    num_edges = abs(max_mass-min_mass+1) # decadal bins for now, although any number of bins is allowed
    massbin_edges = np.logspace(min_mass,max_mass,num=num_edges)

    cdf = np.array([surface_number_flux(dist, fluffy, upper_lower, divine, False) for dist in data.AU])
    x = [ (massbin_edges/cdf[AU,4])**(1./cdf[AU,3]) for AU in data.AU]
    flux = [ cdf[AU,0] * ( (1.+x[AU])**(cdf[AU,2]-1.) / x[AU]**cdf[AU,2] )**(cdf[AU,1]*cdf[AU,3]) for AU in range(len(data.AU)) ]

    gamma = cdf[:,1] # gamma values
    alpha =  -3.*gamma -1.

    fig = plt.figure()
    [ plt.loglog(massbin_edges, flux[AU], marker='x', label='%2.1f AU, gamma = %3.2f, alpha = %3.2f' % (data.AU[AU], gamma[AU], alpha[AU])) for AU in range(len(data.AU)) ]

    plt.grid(True)
    import matplotlib.font_manager as fm
    prop = fm.FontProperties(size=11)
    legend = plt.legend(loc=0,prop=prop,fancybox=True)
    # legend.get_frame().set_alpha(0.5)

    plt.xlabel('Mass (kg)')
    plt.ylabel('Cumulative flux (#/s)')

    # Some text strings describing the current scenario
    if fluffy:
        fluffyness = 'Fluffy'
    else:
        fluffyness = 'Compact'

    if divine:
        divine_text = 'with Divine extrapolation'
    else:
        divine_text = 'without Divine extrapolation'

    plt.title(fluffyness + ' particles, ' + upper_lower + ' limit ' + divine_text)
    return


def summarise_massflux(fluffy=False, upper_lower='upper', divine=True):
    """Summarise the massflux for various cometocentric distances"""

    import matplotlib.cm as cm

    cdf = np.array([surface_number_flux(dist, fluffy, upper_lower, divine, False) for dist in data.AU])

    # Set up some values at which to evaluate the CDF for binning
    min_mass = -22
    max_mass = -3
    num_edges = abs(max_mass-min_mass+1) # decadal bins for now, although any number of bins is allowed
    massbin_edges = np.logspace(min_mass,max_mass,num=num_edges)

    # Calculate the flux and mean mass from the CDF and above bin edges
    binned = np.array([binned_flux(cdf[step,0], cdf[step,1], cdf[step,2], cdf[step,3], cdf[step,4], min_mass, max_mass, num_edges) for step in range(len(data.AU))])

    mean_mass = binned[:,1,:] # [AU, mass]
    pcles_s = binned[:,0,:] # [AU, mass]
    mass_flux = mean_mass * pcles_s # [AU, mass]

    # Prepare a bar plot, overlaying for different distances (smaller bars on top)
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    colours = cm.get_cmap('gist_rainbow')
    colour_list = [colours(1.*i/len(data.AU)) for i in range(len(data.AU))]
    [ ax1.bar(left=np.log10(massbin_edges[:-1]),height=mass_flux[AU,:],width=1.,log=True, label='%2.1f AU' % data.AU[AU],color=colour_list[AU]) for AU in range(len(data.AU)) ]
    ax1.set_xlabel('log(mass [kg])')
    ax1.set_ylabel('Mass flux (kg/s)')
    ax1.grid(True)
    ax1.legend(loc=0)
    ax1.set_xlim(min_mass,max_mass)

    # Some text strings describing the current scenario
    if fluffy:
        fluffyness = 'Fluffy'
    else:
        fluffyness = 'Compact'
    if divine:
        divine_text = 'with Divine extrapolation'
    else:
        divine_text = 'without Divine extrapolation'
    fig1.suptitle(fluffyness + ' particles, ' + upper_lower + ' limit ' + divine_text + '\nCalculated values')

    # For comparison, also make a bar plot of the tabulated values!
    if fluffy:
        edges = data.edges_fluffy
    else:
        edges = data.edges_compact

    if upper_lower.lower() == 'lower':
        mass_flux_orig = data.flux_lower
    else:
        mass_flux_orig = data.flux_upper

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    # Set limits as for the calculated plot for easier comparison (although it will be somewhat blank!)
    xlimits = ax1.get_xlim()
    ylimits = ax1.get_ylim()
    ax2.set_xlim(xlimits)
    ax2.set_ylim(ylimits)
    colours = cm.get_cmap('gist_rainbow')
    [ ax2.bar(left=np.log10(edges[:-1]),height=mass_flux_orig[AU,:],width=1.,log=True, label='%2.1f AU' % data.AU[AU],color=colour_list[AU]) for AU in range(len(data.AU)) ]
    ax2.set_xlabel('log(mass [kg])')
    ax2.set_ylabel('Mass flux (kg/s)')
    ax2.grid(True)
    ax2.legend(loc=0)
    ax2.set_xlim(min_mass,max_mass)
    fig2.suptitle(fluffyness + ' particles, ' + upper_lower + ' limit ' + divine_text + '\nTabulated values')


    return



def compare_massflux(distance=0.0, fluffy=False, upper_lower='upper', divine=False):
    """Compare the calculated and tabluated massfluxes for various cometocentric distances"""

    # If the optional distance parameter has one of the distances tabulated in Fulle, use
    # just this AU, otherwise show plots for all distances.
    if distance not in data.AU:
        distance = data.AU
        dist_index = range(len(data.AU))
    else:
        dist_index = data.AU.index(distance)
        distance = (distance,) # a single element iterable (tuple)

    # Evaluate the cumulative distribution (with or without divine interpolation) at each distance
    cdf = np.array([surface_number_flux(dist, fluffy, upper_lower, divine, False) for dist in distance])

    # Set up some values at which to evaluate the CDF for binning
    min_mass = -22
    max_mass = -3
    num_edges = abs(max_mass-min_mass+1) # decadal bins for now, although any number of bins is allowed
    massbin_edges = np.logspace(min_mass,max_mass,num=num_edges)

    # Calculate the flux and mean mass from the CDF and above bin edges
    binned = np.array([binned_flux(cdf[step,0], cdf[step,1], cdf[step,2], cdf[step,3], cdf[step,4], min_mass, max_mass, num_edges) for step in range(len(distance))])

    mean_mass = binned[:,1,:] # [AU, mass]
    pcles_s = binned[:,0,:] # [AU, mass]
    mass_flux = mean_mass * pcles_s # [AU, mass]

    # For comparison, also make a bar plot of the tabulated values!
    if fluffy:
        edges = data.edges_fluffy
        fluffyness = 'Fluffy'
    else:
        edges = data.edges_compact
        fluffyness = 'Compact'

    if upper_lower.lower() == 'lower':
        mass_flux_orig = np.array(data.flux_lower[dist_index])
        print("Lower flux used")
    else:
        mass_flux_orig = np.array(data.flux_upper[dist_index])

    if len(distance) == 1: mass_flux_orig = mass_flux_orig.reshape( len(distance), mass_flux_orig.size)

    # Some text strings describing the current scenario
    if divine:
        divine_text = 'with Divine extrapolation'
    else:
        divine_text = 'without Divine extrapolation'

    # Prepare a series of bar plots (one per distance) comparing calculated and tabulated massflux per mass bin
    for AU in range(len(distance)):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)

        ax.bar(left=np.log10(edges[:-1])+0.5,height=mass_flux_orig[AU,:],width=0.5,log=True,color='blue',label='Tabulated')
        ax.bar(left=np.log10(massbin_edges[:-1]),height=mass_flux[AU,:],width=0.5,log=True,color='red', label='Calculated')
        ax.set_xlabel('log(mass [kg])')
        ax.set_ylabel('Mass flux (kg/s)')
        ax.grid(True)
        ax.legend(loc=0)
        ax.set_xlim(min_mass,max_mass)
        fig.suptitle(fluffyness + ' particles, ' + upper_lower + ' limit ' + divine_text + '\n %2.1f AU' % distance[AU])

    return



def summary_plot():
    """Plot summary graphs of the input data from Fulle (2010)"""

    # Mass flux versus distance, lower limit, compact particles, average radius per mass bin
    plt.figure()
    [plt.semilogy(data.AU,data.flux_lower[:,bin],ls='dashed',marker='+',label="%3.2f" % data.diam_compact_um[bin]+' um') for bin in range(data.useful_bins)]
    plt.legend(loc=0)
    plt.grid(1)
    plt.title('Mass flux versus heliocentric distance from Fulle (2010)\nLower limit, average radius, compact particles')
    plt.xlabel('Heliocentric distance (AU)')
    plt.ylabel('Mass flux (kg/s)')

    # Mass flux versus mean particle diameter, compact particles

    plt.figure()
    [plt.loglog(data.diam_compact,data.flux_lower[bin,:],ls='dashed',marker='+',label="%3.2f" % data.AU[bin]+' AU') for bin in range(data.num_au)]
    plt.legend(loc=0)
    plt.grid(1)
    plt.title('Mass flux versus mean diameter from Fulle (2010)\nLower limit, compact particles')
    plt.xlabel('Mean diameter (m)')
    plt.ylabel('Mass flux (kg/s)')

    # Number flux per mass bin as a function of average diameter per mass bin (one curve for each distance)

    plt.figure()
    [plt.loglog(data.diam_compact,data.flux_upper[bin,:]/data.mass_compact_avg,ls='dashed',marker='+',label="%3.2f" % data.AU[bin]+' AU') for bin in range(data.num_au)]
    plt.legend(loc=0)
    plt.grid(1)
    plt.ylabel('Dust flux (#/s)')
    plt.xlabel('Mean diameter (m)')
    plt.title('Number flux versus mean diameter from Fulle (2010)\nUpper limit, compact particles')

    plt.show()


def mass_flux(helio_dist, diameter,fluffy,upper_lower):
    """Uses the Fulle et al. (2010) mass flux data to interpolate and return values
    of mass flux for a given heliocentric and particle diameter. Usage:

    mass_flux(helio_dist (AU), diameter (m),fluffy (bool),upper_lower ('upper'/'lower')"""

    input_radius = diameter/2.

    # Interpolating flux versus distance (piecewise)
    if str.lower(upper_lower) == 'lower':
        flux_scaled_au = [10.**np.interp(helio_dist,data.AU,np.log10(data.flux_lower[:,bin])) for bin in range(data.useful_bins)]
    else:
        flux_scaled_au = [10.**np.interp(helio_dist,data.AU,np.log10(data.flux_upper[:,bin])) for bin in range(data.useful_bins)]

    # Now interpolate this in log-log space to scale for the particle size by a linear fit
    if fluffy:
        (m,c) = np.polyfit(np.log10(data.radius_fluffy),np.log10(flux_scaled_au),1)
    else:
        (m,c) = np.polyfit(np.log10(data.radius_compact),np.log10(flux_scaled_au),1)

    # And calculate fitted value
    flux_scaled_size = 10.**np.polyval([m,c],np.log10(input_radius))

    return flux_scaled_size


def surface_number_flux(helio_dist, fluffy, upper_lower, divine, plot=0):
    """Uses the Fulle et al. (2010) flux data for large masses, but the Divine semi-empirical
    formula to transition to a different cumulative mass flux index for smaller particles.

    Returns coefficients of the cumulative mass distribution at the surface for the given
    heliocentric distance. From this the velocity-corrected distribution at a given cometocentric
    distance can be found, and thereafter the number of particles in a given bin."""

    # Interpolating mass flux versus distance (piecewise)
    if str.lower(upper_lower) == 'lower':
        flux_scaled_au = [10.**np.interp(helio_dist,data.AU,np.log10(data.flux_lower[:,bin])) for bin in range(data.useful_bins)]
    else:
        flux_scaled_au = [10.**np.interp(helio_dist,data.AU,np.log10(data.flux_upper[:,bin])) for bin in range(data.useful_bins)]

    # Set mass and radius bin edges according to particle density ("fluffy" or "compact")
    if fluffy:
        bin_edges_mass = data.edges_fluffy
        bin_edges_radius = ((3.*bin_edges_mass)/(4.*np.pi*data.density_fluffy))**(1./3.)
    else:
        bin_edges_mass = data.edges_compact
        bin_edges_radius = ((3.*bin_edges_mass)/(4.*np.pi*data.density_compact))**(1./3.)

    # Calculate the cumulative mass index
    (gamma,f) = np.polyfit(np.log10(bin_edges_mass[:-1]),np.log10(flux_scaled_au),1)
    gamma = 1. - gamma

    ##### Previous calculation method - gives similar, but not identical results
    ##### TODO: check why!

    # Normalise by the bin widths
    # bin_widths_mass = np.array([bin_edges_mass[n+1]-bin_edges_mass[n] for n in range(len(bin_edges_mass)-1)])
    # num_flux = flux_scaled_au/bin_widths_mass

    # Sum to produce the cumulative distribution
    # cdf = np.cumsum(num_flux[::-1])[::-1]

    # Now we can directly calculate the value of the cumulative mass distribution and get index gamma
    # (gamma, f) = np.polyfit(np.log10(bin_edges_mass[:-1]),np.log10(cdf),1)
    # gamma = gamma*-1. # the definition of gamma is such that it doesn't include the negative

    # Check the fit
    # fit = 10.**np.polyval([-1.*gamma,f],np.log10(bin_edges_mass))
    # plt.loglog(bin_edges_mass[:-1],cdf,marker='x',lw=0)
    # plt.loglog(bin_edges_mass,fit)



    # The Divine fit approximates the cumulative mass distribution F(m) with:
    #
    # F(m) = [ (1+x)^(b-1) / x^b ]^(a*c)  where x = (m/m_t)^(1/c)
    #
    # "This function is specified by the positive parameters  a, b, c, and m_t.
    # The exponent -gamma of the cumulative mass distribution tends towards -a for heavy particles
    # (m >> m_t) and towards -ab for light ones. m_t is the mass where the transition between
    # the two exponents takes place, and c determines the sharpness of the transition."
    #
    # Here we fix the transition mass as 1e-13 kg (the Fulle data go down to 1e-15 kg) and the
    # sharpness as 2.0. Then for large particles a = gamma, and ab = 0.26

    low_gamma = 0.26
    a = gamma

    # If the Divine fit is not requested, set b = 1 to collapse back to the single mass exponent.
    #
    if divine:
        b = low_gamma/a
    else:
        b = 1

    c = 2.0
    m_t = 1e-13 # kg

    # mass = bin_edges_mass[-2]

    # Now we need to normalise the cumulative distribution at some point - i.e. at a high enough
    # mass far from the transition mass m_t... The last point in the CDF makes sense!
    #
    # We calculate the mean mass for this bin (using the known gradient and exponential interpolation)
    # and use this to scale the calculated mass flux from the formula against the tabulated value...

    x=(bin_edges_mass[-2:]/m_t)**(1./c)      # use the last two mass bin edges
    F_m = ( (1.+x)**(b-1.) / x**b )**(a*c)   # calculate the Divine function for the edges of the last mass bin
    diff_flux = F_m[0]-F_m[1]                # calculate the number flux according to this function

    mean_mass = dmeanmass(bin_edges_mass[-2],bin_edges_mass[-1],F_m[0],F_m[1])
    F =  flux_scaled_au[-1] / (diff_flux * mean_mass)

    # If requested, plot the function over a sensible range
    if plot:

        # span range 1e-20 to 1e-4 kg
        min_mass = -20
        max_mass = -4
        massbin_edges = 10.0**np.arange(min_mass,max_mass)

        if divine:
            label='Divine extrapolation, ' + upper_lower + ' limit'
        else:
            label='Linear extrapolation, ' + upper_lower + ' limit'

        # Generate the CDF, scaled to max unity to give the number fraction
        x=(massbin_edges/m_t)**(1./c)
        F_m = F * ( (1.+x)**(b-1.) / x**b )**(a*c) # cumulative distribution
        # F_m = F_m/F_m[0] # normalise

        if not(plt.fignum_exists(plot)): # figure does not exist, create it!

            fig = plt.figure(plot)
            ax = fig.add_subplot(1,1,1)

            # rescale the plot to leave room for both title and upper axis label
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height*0.9])

            # calculate the diameter values corresponding to the min and max mass
            if fluffy:
                newxmin = ((3.*(10.**min_mass))/(4.*math.pi*data.density_fluffy))**(1./3.) * 2.e6 # diam in microns
                newxmax = ((3.*(10.**max_mass))/(4.*math.pi*data.density_fluffy))**(1./3.) * 2.e6 # diam in microns
                diam_edges = ((3.*massbin_edges[::2])/(4.*math.pi*data.density_fluffy))**(1./3.) * 2.e6 # diam in microns
                fluffyness = 'fluffy'
            else:
                newxmin = ((3.*(10.**min_mass))/(4.*math.pi*data.density_compact))**(1./3.) * 2.e6 # diam in microns
                newxmax = ((3.*(10.**max_mass))/(4.*math.pi*data.density_compact))**(1./3.) * 2.e6 # diam in microns
                diam_edges = ((3.*massbin_edges[::2])/(4.*math.pi*data.density_compact))**(1./3.) * 2.e6 # diam in microns
                fluffyness = 'compact'

            ax_upper = ax.twiny() # new axis instance
            ax_upper.set_xscale('log') # make it logarithmic
            ax_upper.set_xlim(newxmin,newxmax) # set the diameter limits to match the min/max mass for compact/fluffy particles

#   Old code to label the existing mass ticks with corresponding diameters
#            diam_label = ["%3.1e" % (diam_edges[mass]) for mass in range(len(diam_edges))]
#            ax_upper.set_xlim(np.log10(newxmin),np.log10(newxmax))
#            ax_upper.set_xticks(np.log10(diam_edges))
#            ax_upper.set_xticklabels(diam_label)

            ax_upper.set_xlabel(u'diameter [µm]')

            ax.set_xlabel('Mass (kg)')
            ax.set_ylabel('Cumulative flux (#/s)')
            ax.grid(True)

            title_text='CDF based on Fulle et al. (2010) data \n' + str(helio_dist) + ' AU, ' + fluffyness + ' particles'

            fig.suptitle(title_text)

        else:

            fig = plt.figure(plot)       # retrieve the existing figure instance
            ax = fig.add_subplot(1,1,1)  # returns the subplot instance if it already exists

        ax.loglog(massbin_edges,F_m,marker='x',label=label)

    # Return the coefficients of the fit: F, a, b, c, m_t
    return F, a, b, c, m_t


def binned_flux(F, a, b, c, m_t, min_mass, max_mass, num_edges):
    """Takes calculated CDF parameters (describing the cumulative distribution including the
    decrease in gradient for smaller particles used by Divine) and calculates a binned differential
    flux. The end result is an array containing number flux per bin, and the mean mass in that that bin.

    A minimum and maximum mass and number of bins must be given - bins will be evenly spaced on a log scale"""

    # TODO: Do some checking that the min and max masses are coherent, especially compared to the transition mass

    # Create a set of bins spanning min_mass and max_mass with logarithmic spacing
    bin_edges = np.logspace(min_mass,max_mass,num=num_edges)

    # Generate the CDF
    x=(bin_edges/m_t)**(1./c)
    F_m = F * (((1.+x)**(b-1.) / x**b )**(a*c)) # cumulative distribution

    # Calculate mean mass in each interval based on the cumulative mass flux at bin edges
    mean_mass = np.array([dmeanmass(bin_edges[bin],bin_edges[bin+1],F_m[bin],F_m[bin+1]) for bin in range(len(F_m)-1)])

    # and now get the incremental flux (i.e. the number in this bin)
    number_flux = np.array([F_m[bin]-F_m[bin+1] for bin in range(len(F_m)-1)])

    return number_flux, mean_mass


def number_density(num_flux):

    return
# f(m) = V_s/c * n_d(m)
#
# where V_s/c is the spacecraft velocity relative to the comet and n_d is the number density of a given size range. Assuming a radially symmetric coma, the number density at a given distance from the comet is given as:
#
# n_d(m) = Q(m) / 4pi V_esc r^2
#
# where Q(m) is the number of particles of mass class m that are released from the nucleus per unit time.
#
# So I can try to do something similar to calculate the flux at Rosetta based on the Fulle data and the appropriate SPICE kernel etc. The best way to do this is to produce a cumulative distribution function based on the index and scaling in the Fulle data, and then to use this to produce number fluxes based on the average mass per bin etc.



if __name__ == "__main__":

    summary_plot()
