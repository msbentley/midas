#!/usr/bin/python
"""
read_bcr.py

Mark S. Bentley (mark@lunartech.org), 2011

This module loads in an atomic force microscopy .BCR file to allow subsequent
data manipulation

It can be called from the command line with a filename, e.g.

bcrutils.py testfile.bcr

and will read and display a 2D image, or it can be called with no parameters
and will prompt for a filename.

If called from a script, bcrutils.read can be called with a dictionary name that
will return the scan parameters and data, e.g.xvals = np.linspace(0, bcrdata['xlength'], num=bcrdata['xpixels'])

data = bcrutils.read('testfile.bcr')

"""

debug = False

# Numpy and Matplotlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys, os, array, struct
from midas import common

# Module level definitions

# Create a dictionary to hold scan data parameters
scanparams = {}

# Valid BCR header entries, taken from here:
# http://www.imagemet.com/WebHelp/spip.htm#bcr_stm_file_format.htm
#
# Comments can be written by starting the line with % or #
#
# Will define a dictionary with the names of valid parameters and some
# default values, also used to do type conversion of the rea "anglia radio"d in params.
#
valid_headers = {
  'fileformat': 'bcrstm',
  'headersize': 2048, # if missing, header size is 2048 characters (=4096 bytes in Unicode)
  'xpixels': 32,
  'ypixels': 32,
  'xlength': 32, # in units of xunit
  'ylength': 32, # in units of yunit
  'xunit': 'nm', # nm if not set
  'yunit': 'nm', # nm if not set
  'zunit': 'nm', # nm if not set
  'xlabel': 'X',
  'ylabel': 'Y',
  'zlabel': 'Z',
  'current': 1.000, # nA
  'bias': 1.000, # V
  'starttime': 'time', # MM DD YY hh:mm:ss:hh
  'scanspeed': 1.000,  # nm/sec
  'intelmode': 1, # 1 =  little endian, 0 = big endian
  'forcecurve': 0, # 1 = data contain force curves with the approach followed by retraction curves
  'bit2nm': 1.000,
  'xoffset': 0.000, # nm
  'yoffset': 0.000, # nm
  'voidpixels': 0 } # no. of void pixels, if not present = 0.

# NB. for the 16 bit integer bcrstm format void pixels should be set equal to 32767.
# For the 32 bit floating point bcrf format void pixels are set to 1.7014118219281863150e+38.

string_headers = ['fileformat', 'xunit', 'yunit', 'zunit', 'xlabel', 'ylabel', 'zlabel', 'starttime']
int_headers = ['headersize', 'xpixels', 'ypixels', 'intelmode', 'forcecurve', 'voidpixels']
float_headers = ['current', 'bias', 'scanspeed', 'bit2nm', 'xoffset', 'yoffset', 'xlength', 'ylength']
valid_formats = ['bcrstm', 'bcrf', 'bcrstm_unicode', 'bcrf_unicode']

# This module should have two functions - one that reads in the data and returns in, and
# another that plots the data. If called from the command line, the script should run both!

def read(filename):
    """bcrutils.read(filename) reads a BCR file and returns a structure containing the header
    information and the data, taking into account system endidness etc."""

    # First check if filename is valid
    if not os.path.isfile(filename):
        print('ERROR: file %s not found' % (filename))
        return False

    for line in open(filename):    # Now make an array of x and y values for each pixel

        if '\x00' in line: break # binary data, end header read
        if line[0] == '%' or line[0] == '#': continue # skip comments
        # print line
        param, value = line.split('=')    # Now have 3 np arrays containing X, Y and Z coords - fit a plane
        param = param.strip()
        value = value.strip() # remove whitespace, including the trailing carriage return

        # now have a param, value tuple, validate the param and do type conversion of the val
        #
        if not valid_headers.has_key(param):
            sys.exit('Invalid BCR file!')
        else:
            if param in string_headers:
                scanparams[param] = value
                if scanparams[param] == 'fileformat' and value not in valid_formats:
                    sys.exit('Format must be one of: bcrstm, bcrf, bcrstm_unicode, bcrf_unicode')
            elif param in int_headers:
                scanparams[param] = int(value)
            elif param in float_headers:
                scanparams[param] = float(value)
            else:
                print 'Type of parameter unknown'

    # scanparams now has valid entries
    # for param in scanparams.keys(): print param + str(scanparams[param])

    scanparams['filename'] = filename
    bcrfile = open(filename, 'rb') # open again for binary read

    if 'headersize' not in scanparams:
      scanparams['headersize'] = valid_headers['headersize'] # bytes

    bcrfile.seek(scanparams['headersize']) # skip the ASCII header

    # The bcrstm and bcrstm_unicode formats identifies the data following the header to be in 16 bit integer format
    # while the bcrf and bcr_ unicode formats identifies the data as being in 32 bit floating point.
    #
    # h = signed short (16 bit integer), f = float (32 bit floating point)
    #
    if scanparams['fileformat'] == 'bcrf' or scanparams['fileformat'] == 'bcrf_unicode':
      rawvalues = array.array('f')
    else:
      rawvalues = array.array('h')

    # Read the binary data!
    #
    rawvalues.fromfile(bcrfile, (scanparams['xpixels'] * scanparams['ypixels']))

    # Now take care of the byte order of the imported data according to the system type and the value of the
    # parameter scanparams['intelmode'], which is 1 for little endian, 0 for big endian
    #
    if (sys.byteorder == 'little' and  scanparams['intelmode'] == 0) or (sys.byteorder == 'big' and  scanparams['intelmode'] == 1):
        print 'Warning: byte order swapped'
        rawvalues.byteswap()

    # Now convert to a numpy long array
    data = np.array(rawvalues[0:scanparams['xpixels']*scanparams['ypixels']],dtype=long)

    # add this data to the structure and return
    #
    scanparams['data'] = data

    return scanparams


def write(bcrdata):
    """Write a BCR file given the appropriate data structure"""

    # Open the file for output
    bcrfile = open(bcrdata['filename'], 'w')

    # WARNING - no error checking or handling here - CAUTION!

    # Set minimal data to default values if not present
    if not 'fileformat' in bcrdata: bcrdata['fileformat'] = 'bcrstm' # typically used by MIDAS
    if not 'intelmode' in bcrdata: bcrdata['intelmode'] = 1 # little Endian
    if not 'xoffset' in bcrdata: bcrdata['xoffset'] = 0.0
    if not 'yoffset' in bcrdata: bcrdata['yoffset'] = 0.0
    if not 'scanspeed' in bcrdata: bcrdata['scanspeed'] = 0.0
    if not 'bit2nm' in bcrdata: bcrdata['bit2nm'] = common.zcal
    if not 'xlength' in bcrdata: bcrdata['xlength'] = bcrdata['xpixels']
    if not 'ylength' in bcrdata: bcrdata['ylength'] = bcrdata['ypixels']
    if not 'zunit' in bcrdata: bcrdata['zunit'] = 'nm'


    # Write the ASCII header
    bcrfile.write('fileformat = ' + bcrdata['fileformat'] + '\n')
    bcrfile.write('intelmode = ' + str(bcrdata['intelmode']) + '\n')
    bcrfile.write('xpixels = ' + str(bcrdata['xpixels']) + '\n')
    bcrfile.write('ypixels = ' + str(bcrdata['ypixels']) + '\n')
    bcrfile.write('xlength = ' + str(bcrdata['xlength']) + '\n')
    bcrfile.write('xoffset = ' + str(bcrdata['xoffset']) + '\n')
    bcrfile.write('xunit = nm\n')
    bcrfile.write('ylength = ' + str(bcrdata['ylength']) + '\n')
    bcrfile.write('yoffset = ' + str(bcrdata['yoffset']) + '\n')
    bcrfile.write("yunit = nm\n")
    bcrfile.write('scanspeed = ' + str(bcrdata['scanspeed']) + '\n')
    bcrfile.write('bit2nm = ' + str(bcrdata['bit2nm']) + '\n')
    bcrfile.write('zunit = ' + str(bcrdata['zunit']) + '\n')

    bcrfile.close() # finished the ASCII header
    filesize = os.path.getsize(bcrdata['filename']) # returns filesize in bytes

    # Open the file again to append binary data
    bcrfile = open(bcrdata['filename'], 'ab')

    # Need to pad the header to 2048 characters = 2048 bytes for standard mode, 4096 bytes for unicode
    paddingsize = 2048 - filesize

    pad = struct.Struct('x') # padding byte

    for i in range(paddingsize):
        bcrfile.write(pad.pack())

    byteswritten = 0

    # We need to write binary 16-bit signed integer values
    # In python structures, < = little-endian, > = big-endian
    # intelmode: 1 =  little endian, 0 = big endian
    if bcrdata['intelmode'] == 1:
        s = struct.Struct('<H') if bcrdata['zunit']=='none' else struct.Struct('<h')
    else:
        s = struct.Struct('>H') if bcrdata['zunit']=='none' else struct.Struct('>h')


    # Now write the data array - there is probably a more pythonic way to do this in one go...
    for element in range(len(bcrdata['data'])):
        bcrfile.write(s.pack(bcrdata['data'][element]))
        byteswritten += 1

    if debug: print 'Header size = ' + str(filesize) + ', writing ' + str(paddingsize) + ' padding bytes and ' + str(byteswritten) + ' data bytes'

    # Close the file
    bcrfile.close()


def plotrowcol(bcrdata):
    """Plot a 2D image of contained BCR data using matplotlib, with pixel numbers and raw Z values"""

    xpixels = bcrdata['xpixels']
    ypixels = bcrdata['ypixels']

    data = np.array(bcrdata['data'], dtype=long) # make a copy of the data in float format
    redata = np.reshape(data, (xpixels, ypixels)) # re-shape into a 2D array

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    plot1 = ax1.imshow(redata, origin='upper', interpolation='nearest')
    ax1.set_title(bcrdata['filename'])
    fig1.colorbar(plot1) # Now plot using a colourbar
    plt.draw()

#    plt.show()

    return

def plotline(bcrdata, line, direction):
    """Plot a line from an image scan in either X or Y."""

    # validate the inputs against the image data
    direction = str(direction).lower()

    if direction != 'x' and direction != 'y':
        print 'Error: direction must be X or Y'
        return

    xpixels = bcrdata['xpixels']
    ypixels = bcrdata['ypixels']

    numlines = ypixels if direction == 'x' else xpixels

    line = int(line)

    if line >= numlines:
        print 'Error: there are only %i lines in direction %s' % (numlines, direction)
        return

    redata = np.reshape(bcrdata['data'], (bcrdata['xpixels'], bcrdata['ypixels']))

    if direction == 'x':
        linedata = redata[line,:]
    else:
        linedata = redata[:,line]

    description = 'Line %i/%i in direction %s' % (line,numlines,direction)

    plt.plot(linedata)
    plt.xlabel('Data point')
    plt.ylabel('Raw value')
    plt.xlim(0,len(linedata))
    plt.grid(True)
    plt.title(bcrdata['filename'] + '\n' + description)
    plt.draw()


def plotraw(bcrdata):
    """Plot a 2D image of contained BCR data using matplotlib, with raw X, Y and Z values"""

    xpixels = bcrdata['xpixels']
    ypixels = bcrdata['ypixels']

    data = np.array(bcrdata['data'], dtype=long) # make a copy of the data in float format
    redata = np.reshape(data, (xpixels, ypixels)) # re-shape into a 2D array

    # To gives the correct X and Y (DAC) values, we need the origin and step size

    xstep, ystep = get_step(bcrdata)
    xorigin, yorigin = get_origin(bcrdata)

    # Now need a matrix of ticks
    # x = (arange(xorigin,xorigin+(xpixels*xstep),xstep))
    # y = (arange(yorigin,yorigin+(ypixels*ystep),ystep))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    plot1 = ax1.imshow(redata, origin='upper', interpolation='nearest', extent=[xorigin,xorigin+xpixels*xstep,yorigin+ypixels*ystep,yorigin])
    ax1.set_xlabel('X (raw)')
    ax1.set_ylabel('Y (raw)')
    ax1.set_title(bcrdata['filename'])
    fig1.colorbar(plot1) # Now plot using a colourbar
    plt.draw()

    return



def plot2d(bcrdata, description='', writefile=False, realunits=True):
    """Plot a 2D image of contained BCR data using matplotlib. Accepts a BCR data dictionary and optional description"""

    import os

    data = np.array(bcrdata['data'], dtype=float) # make a copy of the data in float format
    data = data - data.min() # shift it to 0 minimum
    redata = np.reshape(data, (bcrdata['ypixels'], bcrdata['xpixels'])) # re-shape into a 2D array
    redata = redata*bcrdata['bit2nm'] # calibrate into nm

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)

    if realunits:
        plot1 = ax1.imshow(redata, origin='upper', interpolation='nearest',extent=[0,bcrdata['xlength']/1.e3,0,bcrdata['ylength']/1.e3],
            cmap=cm.afmhot)
        ax1.set_xlabel('X (microns)')
        ax1.set_ylabel('Y (microns)')

    else:
        plot1 = ax1.imshow(redata, origin='upper', interpolation='nearest',cmap=cm.afmhot)
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')

    ax1.grid(True)
    cbar = fig1.colorbar(plot1) # Now plot using a colourbar
    cbar.set_label(bcrdata['zunit'], rotation=90)

    if description == '':
        desc = "_".join(os.path.basename(bcrdata['filename']).split('_')[:-1])
        ax1.set_title(desc)
    else:
        ax1.set_title(description)

    if writefile:
        plotfile = os.path.splitext(bcrdata['filename'])[0] + '.png'
        plt.savefig(plotfile)
        plt.clf()
        plt.close()

    return fig1, ax1



def plot3d(bcrdata, description=''):
    """Plot a 3D image of contained BCR data using matplotlib"""

    from mpl_toolkits.mplot3d import Axes3D

    data = bcrdata['data'] - min(bcrdata['data']) # shift it to 0 minimum
    redata = np.reshape(data, (bcrdata['xpixels'], bcrdata['ypixels'])) # re-shape into a 2D array
    redata = redata*bcrdata['bit2nm'] # calibrate into nm

    # set up x and y axes (in microns)
    xvals = np.linspace(0, bcrdata['xlength']/1.e3, num=bcrdata['xpixels'])
    yvals = np.linspace(0, bcrdata['ylength']/1.e3, num=bcrdata['ypixels'])
    xs, ys = np.meshgrid(xvals, yvals)

    # Now setting up for a surface plot

    fig3 = plt.figure()
    ax3= Axes3D(fig3)
    ax3.set_ylabel('Y (microns)')
    ax3.set_xlabel('X (microns)')
    ax3.set_zlabel('height (nm)')

    if description == '':
        ax3.set_title(bcrdata['filename'])
    else:
        ax3.set_title(description)

    plot3 = ax3.plot_surface(xs,ys,redata,rstride=1, cstride=1, cmap=cm.jet,linewidth=0, antialiased=False)

    # fig3.colorbar(plot3, shrink=0.5, aspect=5)
    # or for a wireframe
    # ax3.plot_wireframe(xs,ys,redata,rstride=24, cstride=1)

    return

def plot3dbar(bcrdata):
    """Plot a 3D barchart of contained BCR data using MayaVI2"""

    from mayavi.mlab import barchart

    # set up x and y axes
    xvals = np.linspace(0, bcrdata['xlength'], num=bcrdata['xpixels'])
    yvals = np.linspace(0, bcrdata['ylength'], num=bcrdata['ypixels'])
    xs, ys = np.meshgrid(xvals, yvals)

    data = bcrdata['data'] - min(bcrdata['data']) # shift it to 0 minimum
    data = data*bcrdata['bit2nm'] # calibrate into nm

    barchart(xs.ravel(),ys.ravel(), data, scale_factor=10.0, lateral_scale=10.0)

#    plt.show()
    return


def plotsurf(bcrdata):
    """Plot a 3D surface using MayaVI"""

    from mayavi.mlab import surf, axes

    data = np.array(bcrdata['data'], dtype=float) # make a copy of the data in float format
    data = data - min(data) # shift it to 0 minimum
    redata = np.reshape(data, (bcrdata['xpixels'], bcrdata['ypixels'])) # re-shape into a 2D array
    redata = redata*bcrdata['bit2nm'] # calibrate into nm

    xstep=bcrdata['xlength']/bcrdata['xpixels']
    ystep=bcrdata['ylength']/bcrdata['ypixels']
    x, y = np.mgrid[0.:bcrdata['xlength']:xstep, 0.:bcrdata['ylength']:ystep]
    surf(x,y,redata)
    axes

    return


def animate(bcrdata):
    """Animate a mayavi2 surface plot by rotating around the azimuth angle"""

    from mayavi import mlab

    # Set up off-screen rendering
    mlab.options.offscreen = True

    # Create a figure with white background (colours are RGB tuples with range 0..1)
    mlab.figure(bgcolor=(1,1,1), size=(800, 600))

    # Use the BCR filename as a base for the output images
    filename = bcrdata['filename']

    # Call the previous plot routine to draw the BCR file as a surface
    plotsurf(bcrdata)

    # store default view
    view = mlab.view()

    # loop over azimuth angle, redraw and output the figure to a PNG file
    num_steps = 360*4.
    azimuth_step = 360./num_steps
    step = 0
    for az in np.arange(0.,360.,azimuth_step):
        mlab.view(azimuth=az) # set the azimuth angle
        mlab.savefig(filename+'_'+str(step)+'.png')
        step+=1

    # restore on-screen rendering
    mlab.options.offscreen = False

    # one can encode these with things like@
    #
    # mencoder "mf://bcrfilename.bcr_%d.png" -mf fps=25 -o output1.avi -ovc lavc -lavcopts vcodec=mpeg4
    # for i in `seq 1 3`; do mencoder "mf://bcrfilename.bcr_%d.png" -sws 10 -vf scale=800:592 -ovc x264 -x264encopts qp=40:subq=7:pass=$i -o output.avi; done
    # ffmpeg -f image2 -i bcrfilename.bcr_%d.png output.mpg

def planesub(bcrdata):
    """Return a BCR structure with a best-fit plane subtracted"""

    # Make an array of x and y values for each pixel
    xvals = np.arange(bcrdata['xpixels'])
    yvals = np.arange(bcrdata['ypixels'])
    xs, ys = np.meshgrid(xvals, yvals)
    x = xs.ravel()
    y = ys.ravel()

    # Now have 3 np arrays containing X, Y and Z coords - fit a plane

    A = np.column_stack([x, y, np.ones_like(x)])
    abc, residuals, rank, s = np.linalg.lstsq(A, bcrdata['data'])

    # print 'Coefficients: ' + str(abc)

    # The "real" regression coefficients can be calculated, but we need the
    # MIDAS origin and step size

    xstep, ystep = get_step(bcrdata)
    xorigin, yorigin = get_origin(bcrdata)
    a1 = abc[0] / xstep
    b1 = abc[1] / ystep
    c1 = abc[2] - (a1 * xorigin) - (b1 * yorigin)

    # print 'Real coefficients: ' + str(a1) + ', ' + str(b1) + ', ' + str(c1)

    # Create a grid containing this fit
    zgrid = np.array( [x[ind]*abc[0] + y[ind]*abc[1] + abc[2] for ind in np.arange(len(x))] )
    zgrid = zgrid.round().astype(long)

    # Subtract fit and shift Z values
    imagefit = bcrdata['data'] - zgrid + abc[2].round().astype(long)

    # Return a copy of the original data structure, but with the plane subtracted image data

    newbcr = bcrdata
    newbcr['data'] = imagefit
    return newbcr


# The origin and step sizes are calculated into "real" units in the BCR file
# here we want them back in XY table pixels

def get_origin(bcrdata):
    xorigin = int(round((bcrdata['xoffset']*1000 + common.xyorigin)/common.xycal['open'])-1)
    yorigin = int(round((bcrdata['yoffset']*1000 + common.xyorigin)/common.xycal['open'])-1)
    return xorigin,yorigin

def set_origin(xorigin, yorigin):
    xoffset = xorigin * common.xycal['open'] + common.xyorigin/1000.
    yoffset = yorigin * common.xycal['open'] + common.xyorigin/1000.
    return xoffset,yoffset

def get_step(bcrdata):
    xstep = int(round( bcrdata['xlength'] / (bcrdata['xpixels']-1) / common.xycal['open'] ))
    ystep = int(round( bcrdata['ylength'] / (bcrdata['ypixels']-1) / common.xycal['open'] ))
    return xstep,ystep

if __name__ == "__main__":

    # Called interactively - check for a filename as an input, if none, prompt for if

    if len(sys.argv) > 2:
        sys.exit('Too many parameters - either run without anything for interactive, or give the image filename')

    elif len(sys.argv) == 2: # we have one input param, that should be parsed as a filename
        filename = str(sys.argv[1])
        if not os.path.isfile(filename):
            sys.exit('File ' + filename + ' does not exist!')

    else: # no parameters, prompt for a filename

        # Import gtk libraries for the file dialogue

        import pygtk
        pygtk.require('2.0')
        import gtk

        dialog = gtk.FileChooserDialog("Open a control data file",
            None,
            gtk.FILE_CHOOSER_ACTION_OPEN,
            (gtk.STOCK_CANCEL, gtk.RESPONSE_CANCEL,
            gtk.STOCK_OPEN, gtk.RESPONSE_OK))

        dialog.set_default_response(gtk.RESPONSE_OK)

        filter = gtk.FileFilter()
        filter.set_name("BCR files")
        filter.add_pattern("*.bcr")
        filter.set_name("BCRF files")
        filter.add_pattern("*.bcrf")
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

    # If called interactively, we now have a filename, so load the data using the read function, then plot
    # the raw data using the 2D plot function, and also a plane subtracted version.

    bcrdata = read(filename)
    plot2d(bcrdata)
    plotraw(bcrdata)
    plotrowcol(bcrdata)
    # levelled = planesub(bcrdata)
    # plot2d(levelled)
    # plot3d(levelled)
    plt.draw()
    plt.show()
