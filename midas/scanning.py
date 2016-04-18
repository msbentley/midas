#!/usr/bin/python
"""
scanning.py - module to calculate various MIDAS parameters (scan duration,
step sizes etc.) for various planning scenarios.
"""

from midas import common
import numpy as np
import matplotlib.pyplot as plt

debug = True

def calc_duration(xpoints, ypoints, ntypes, zretract, zsettle=50, xysettle=50, zstep=4, avg=1, ctrl=False):

    ifact_1 = 0.0003
    ofact_1 = 0.00035
    afact_1 = 0.00075
    subcycles = 100

    xytotal = (xpoints+2)*xysettle/1000.*ypoints
    ztotal = (xpoints+3)*zsettle/1000.*ypoints
    innerloop = xpoints*zretract/zstep*(ifact_1+avg*afact_1)*ypoints # seconds
    outerloop = zretract/(subcycles*zstep)*ofact_1*xpoints*ypoints # seconds
    duration = xytotal + ztotal + innerloop + outerloop + ypoints*1.0625 + xpoints*ypoints*ntypes*0.001 # seconds

    # Extend duration if control data is enabled
    if ctrl:
        duration += ((xpoints*ypoints)*zretract/zstep)*(0.00334)

    # duration = duration * 0.75 # approx. speedup factor for latest OBSW version

    return duration


def scan_params(xlen_um, ylen_um, xpoints, ypoints, safety_factor = 2., ntypes=2, xysettle=50, zsettle=50,
                openloop=True, duration=True, zstep=4):
    """Returns scan duration for a given X and Y length and number of pixels.
    Other scan parameters are set to the following defaults:

    avg=1, zstep=4

    Open loop is assumed unless xclosed or yclosed are set to True.

    If return_params is set to True, the retraction and step sizes are returned"""

    from datetime import timedelta

    xcal = common.xycal['open']
    ycal = common.xycal['open'] if openloop else common.xycal['closed']

    xstep = int(np.around(xlen_um*1.e3/xcal/xpoints))
    ystep = int(np.around(ylen_um*1.e3/ycal/ypoints))

    xstep_nm = xstep * xcal
    ystep_nm = ystep * ycal

    zretract_nm = max(xstep_nm,ystep_nm) * safety_factor
    zretract = int(np.around(zretract_nm / common.zcal))

    realx = xpoints * xstep_nm/1.e3
    realy = ypoints * ystep_nm/1.e3

    if duration:
        avg = 1
        duration = calc_duration(xpoints, ypoints, ntypes, zretract, zsettle, xysettle, zstep, avg)

        return realx, realy, xstep, ystep, zretract, timedelta(seconds=duration)

    return realx,realy, xstep, ystep, zretract


def scan_realunits(xlen_um, ylen_um, xpoints, ypoints, openloop=True, dur=True, safety=2.,
        zstep=4, xysettle=50, zsettle=50):
    """Accepts an x and y length (in microns) and a number of x and y
    pixels and returns the duration (in seconds)."""

    # set some default parameters
    avg = 1
    ntypes = 2

    # def scan_params(xlen_um, ylen_um, xpoints, ypoints, safety_factor = 2., openloop=True, duration=True):
    realx, realy, xstep, ystep, zretract, duration = scan_params(xlen_um=xlen_um, ylen_um=ylen_um,
        xpoints=xpoints, ypoints=ypoints, openloop=openloop, duration=dur, safety_factor=safety,
        zstep=zstep, xysettle=xysettle, zsettle=zsettle)
    # duration = calc_duration(xpoints, ypoints, ntypes, zretract, zsettle, xysettle, zstep, avg)

    return duration


def plot_duration_size(sizes_micron, safety=2.0, zstep=4):
    """Takes as input a list of size (um) and plots the duration of a square scan as
    a function of number of points for every possible value (32..512). If only one
    value is given, the upper scale shows resolution."""

    if type(sizes_micron) != list:
        sizes_micron = [sizes_micron]

    xypoints = np.arange(32,512+32,32)
    num_sizes = len(sizes_micron)
    duration = np.zeros( [num_sizes,len(xypoints)] )

    duration_plot = plt.figure()
    duration_axes = duration_plot.add_subplot(1,1,1)

    for cnt in range(num_sizes):
        duration[cnt,:] = np.array([scan_realunits(sizes_micron[cnt], sizes_micron[cnt], pts, pts, safety=safety, zstep=zstep).total_seconds() for pts in xypoints])
        duration_axes.plot(xypoints,duration[cnt,:]/60./60.,label='%i microns' % (sizes_micron[cnt]))

    duration_axes.set_xlabel('Number of points')
    duration_axes.set_ylabel('Duration (hours)')
    duration_axes.set_xlim(xypoints[0],xypoints[-1])
    duration_axes.set_xticks(xypoints)
    duration_axes.legend(loc=0)
    duration_axes.grid(True)

    if num_sizes == 1: # show resolution on the upper axis

        axis_upper = duration_axes.twiny()
        resolutions = sizes_micron[0]*1.e3/xypoints
        res_label = ["%3.0f" % rez for rez in resolutions]
        axis_upper.set_xlim(duration_axes.get_xlim())
        axis_upper.set_xticks(duration_axes.get_xticks())
        axis_upper.set_xticklabels(res_label)
        axis_upper.set_xlabel('Resolution (nm)')
        axis_upper.grid(True)

    plt.show()

    return xypoints, duration


def plot_duration_zstep(size_microns, safety=2.0, zsteps=4):
    """Takes as input a list of size (um) and plots the duration of a square scan as
    a function of number of points for every possible value (32..512). If only one
    value is given, the upper scale shows resolution."""

    if type(zsteps) != list:
        zsteps = [zsteps]

    xypoints = np.arange(32,512+32,32)
    num_sizes = len(zsteps)
    duration = np.zeros( [num_sizes,len(xypoints)] )

    duration_plot = plt.figure()
    duration_axes = duration_plot.add_subplot(1,1,1)

    for cnt in range(num_sizes):
        duration[cnt,:] = np.array([scan_realunits(size_microns, size_microns, pts, pts, safety=safety, zstep=zsteps[cnt]).total_seconds() for pts in xypoints])
        duration_axes.plot(xypoints,duration[cnt,:]/60./60.,label='zstep = %i' % (zsteps[cnt]))

    duration_axes.set_xlabel('Number of points')
    duration_axes.set_ylabel('Duration (hours)')
    duration_axes.set_xlim(xypoints[0],xypoints[-1])
    duration_axes.set_xticks(xypoints)
    duration_axes.legend(loc=0, fontsize='small')
    duration_axes.grid(True)

    axis_upper = duration_axes.twiny()
    resolutions = size_microns*1.e3/xypoints
    res_label = ["%3.0f" % rez for rez in resolutions]
    axis_upper.set_xlim(duration_axes.get_xlim())
    axis_upper.set_xticks(duration_axes.get_xticks())
    axis_upper.set_xticklabels(res_label)
    axis_upper.set_xlabel('Resolution (nm)')
    axis_upper.grid(True)

    plt.suptitle('Scan duration for %3.2f micron square area' % (size_microns), y=1.00)

    plt.setp(axis_upper.get_xticklabels(), rotation=45)
    plt.setp(duration_axes.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.show()

    return xypoints, duration





def plot_duration_resolution(resolution_nm, closed_loop=False):
    """Takes as input a list of resolutions (nm) and plots the duration of a square scan as
    a function of number of points for every possible value (32..512). If only one
    value is given, the upper scale shows image size."""

    if type(resolution_nm) != list:
        resolution_nm = [resolution_nm]

    xypoints = np.arange(32,512+32,32)
    num_resolns = len(resolution_nm)
    duration = np.zeros( [num_resolns,len(xypoints)] )

    duration_plot = plt.figure()
    duration_axes = duration_plot.add_subplot(1,1,1)

    for cnt in range(num_resolns):
        size_nm = xypoints * resolution_nm[cnt]
        duration[cnt,:] = np.array([scan_realunits(size_nm[cnt], size_nm[cnt], pts, pts) for pts in xypoints])

        duration_axes.plot(xypoints,duration[cnt,:]/60./60.,label='%i nm' % (resolution_nm[cnt]))

    duration_axes.set_xlabel('Number of points')
    duration_axes.set_ylabel('Duration (hours)')
    duration_axes.set_xlim(xypoints[0],xypoints[-1])
    duration_axes.set_xticks(xypoints)
    duration_axes.legend(loc=0)
    duration_axes.grid(True)

    if num_resolns == 1: # show image size

        axis_upper = duration_axes.twiny()
        image_sizes = resolution_nm * xypoints
        size_label = ["%3.0f" % size for size in (image_sizes/1.e3)]
        axis_upper.set_xlim(duration_axes.get_xlim())
        axis_upper.set_xticks(duration_axes.get_xticks())
        axis_upper.set_xticklabels(size_label)
        axis_upper.set_xlabel('Image size (microns)')
        axis_upper.grid(True)

    plt.draw()
    plt.show()

    return xypoints, duration



def read_fileinfo(filename):
    """Reads the text file produced by FileInfo for Show / Image Scans and extracts the scan duration
    and parameters necessary to calculate the duration, in order to compare and calibrate"""

    skiprows = 7
    index = 0
    num_scans = 0

    # set up some lists to hold outout
    xpoints = []
    ypoints = []
    num_types = []
    z_settle = []
    xy_settle = []
    retract = []
    zstep = []
    avg = []
    duration = []
    sourcefile = []
    x_loop = []
    y_loop = []
    scan_algo = []

    with open(filename, 'r') as f:

        index += 1

        # skip header rows
        for line in range(skiprows):
            f.readline() # throw away

        # loop through file extracting valid scans (ignore aborted)

        while 1:
            line=f.readline()
            if not line: break

            # look for '#### SCAN STARTED'
            # i.e. #### in column 4 with split()

            # if len(line.split()) <= 3: line=f.readline() # ignore blank lines
            if (len(line.split()) <= 3) or (len(line.split()) == 11): continue # ignore blank lines

            if line.split()[3] == '####': # scan start
                dataline = f.readline()
                if len(dataline.split()) <= 3: dataline = f.readline()
                endline = f.readline()
                if endline.split()[3] == '####' and endline.split()[5] == 'ABORTED':
                    pass # invalid scan
                elif endline.split()[3] == '####' and endline.split()[5] == 'FINISHED':

                    # read the next line (first channel) to get x and y points
                    channel1 = f.readline()
                    xpoints.append(int(channel1.split()[-2]))
                    ypoints.append(int(channel1.split()[-1]))

                    dataline=dataline.split()
                    sourcefile.append(dataline[0])
                    datatype = int(dataline[5].split('=')[-1])
                    num_types.append(bin(datatype).count("1"))
                    xy_settle.append(int(dataline[7].split('=')[1].split('/')[0]))
                    z_settle.append(int(dataline[7].split('=')[1].split('/')[1]))
                    x_loop.append(int(dataline[9].split('=')[1].split('/')[0]))
                    y_loop.append(int(dataline[9].split('=')[1].split('/')[1]))
                    scan_algo.append(int(dataline[13].split('=')[1]))

                    zstep.append(int(dataline[10].split('=')[1]))
                    avg.append(int(dataline[11].split('=')[1]))
                    retract.append(int(dataline[14].split('=')[1]))
                    duration.append(int(endline.split()[6][1:-1]))

                    num_scans += 1

                    # if debug: print '%ix%i scan, %i types, XY settle=%i, Z settle=%i, retract=%i, Zstep=%i, avg=%i' % (xpoints, ypoints, num_types, z_settle, xy_settle, retract, zstep, avg)
                    # if debug: print 'Actual time: %i s, predicted time %i s\n' % (scan_duration, predict)

    f.close()


    print '%i valid scans read' % (num_scans)
    return num_scans, sourcefile, xpoints, ypoints, num_types, z_settle, xy_settle, x_loop, y_loop, retract, scan_algo, zstep, avg, duration

def compare_predicted_fileinfo(filename, csvfile):
    """Accepts scan durations and parameters from a FileInfo file and compares with the predicted values
    Both inputs and outputs are written to a CSV file"""

    # return num_scans, sourcefile, xpoints, ypoints, num_types, z_settle, xy_settle, x_loop, y_loop, retract, scan_algo, zstep, avg, duration

    num_scans, sourcefile, xpoints, ypoints, num_types, z_settle, xy_settle, x_loop, y_loop, retract, scan_algo, zstep, avg, duration = read_fileinfo(filename)

    predicted_duration = [calc_duration(xpoints[scan], ypoints[scan], num_types[scan], z_settle[scan], xy_settle[scan], \
        retract[scan], zstep[scan], avg[scan]) for scan in range(num_scans)]

    predicted_duration = np.array(np.rint(predicted_duration),dtype=int)

    outputdata = np.array((sourcefile, xpoints, ypoints, num_types, z_settle, xy_settle, x_loop, y_loop, retract, scan_algo, zstep, avg, duration, predicted_duration))
    header='sourcefile, xpoints, ypoints, num_types, z_settle, xy_settle, x_loop, y_loop, retract, scan_algo, zstep, avg, real_duration, predicted_duration'

    # save to a csv file
    np.savetxt(csvfile,outputdata.T,header=header,delimiter=',',fmt="%s",comments='')


def compare_predicted(query=None):

    import ros_tm
    import pandas as pd
    from datetime import timedelta

    images = ros_tm.load_images(data=False, topo_only=False)

    if query is not None:
        images = images.query(query)

    if len(images)==0:
        print('ERROR: no images to compare!')
        return None

    predicted = []
    idx = []

    for scan in images.scan_file.unique():

        num_chans = len(images[ images.scan_file == scan ])
        image = images[ (images.scan_file == scan ) & (images.channel=='ZS') ].squeeze()

        if image.duration is pd.NaT:
            continue

        # calc_duration(xpoints, ypoints, ntypes, zretract, zsettle=50, xysettle=50, zstep=4, avg=1, ctrl=False):

        predicted.append(
            calc_duration(image.xsteps, image.ysteps, num_chans, image.z_ret, image.z_settle, image.xy_settle, image.z_step, 1, image.ctrl_image ) )
        idx.append(image.name)

    images = images.ix[idx]

    images['predicted'] = np.array(np.rint(predicted),dtype=int)
    images.predicted = images.predicted.apply( lambda t: timedelta(seconds=t) )

    return images



if __name__ == "__main__":

    sizes=1.
    plot_duration_size(sizes)
