#!/usr/bin/python

# Module to manipulate control data extracted from MIDAS TM using FileInfo.
# The following routines are available:
#
# read(filename) - reads a control data .txt file and returns a data structure


# TODO: deal correctly with negative 

valid_headers = { # valid headers, and the short forms we use here
  'Generation Date': 'gen_date', 
  'X (linear) Position': 'linear_pos',
  'Y (wheel) Position': 'wheel_pos',
  'X Scan Origin': 'x_origin',
  'Y Scan Origin': 'y_origin',
  'Step size': 'zstep',
  'Number of Steps': 'num_steps',
  'Cantilever Number': 'cantilever',
  'Scan Mode': 'scan_mode',
  'Main Scan Direction': 'mainscandir',
  'Line Scan Direction': 'linescandir' }
  
string_headers = ['gen_date','scan_mode','mainscandir','linescandir']

#### classes

class pickpoint:

    def __init__(self,xs,ys):

        import numpy as np
        import matplotlib.pyplot as plt

        self.xs = np.array(xs)
        self.ys = np.array(ys)

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1,1,1)
        self.ax.set_xlabel('Z displacement (nm)')
        self.ax.set_ylabel('Peak cantilever amplitude (V)')

        self.line, = self.ax.plot(self.xs,self.ys,'ro ', picker=1)

        self.text = self.ax.text(0.05, 0.05, 'Datapoint index selected: none',
                            transform=self.ax.transAxes, va='top')

        self.lastind = 0

        self.ax.set_title('Click point and hit any key')

        self.selected,  = self.ax.plot([self.xs[0]],
                                       [self.ys[0]], 'o', ms=12, alpha=0.4,
                                       color='yellow', visible=False)


        self.cid = self.fig.canvas.mpl_connect('pick_event', self.onpick)
               

    def onpick(self, event):

        import numpy as np

        if event.artist!=self.line: return True

        N = len(event.ind)
        if not N: return True

        # the click locations
        x = event.mouseevent.xdata
        y = event.mouseevent.ydata

        dx = np.array(x-self.xs[event.ind],dtype=float)
        dy = np.array(y-self.ys[event.ind],dtype=float)

        distances = np.hypot(dx,dy)
        indmin = distances.argmin()
        dataind = event.ind[indmin]

        self.lastind = dataind
        self.update()


    def update(self):

        if self.lastind is None: 
            self.fig.canvas.mpl_disconnect(self.cid)
            return

        dataind = self.lastind

        self.selected.set_visible(True)
        self.selected.set_data(self.xs[dataind], self.ys[dataind])

        self.text.set_text('Datapoint index selected: %d'%dataind)

        self.fig.canvas.draw()


def read(filename):
    """Opens a control data file and returns a list of control data
    dictionaries, one per point approach"""

    import numpy as np
    
    header = True
    data = False
    controldata = {}
    metadata = {}
    controldata['points'] = []
    controlpoints = 0
    
    with open(filename) as f:
        for line in f:
            # Lines containing a ":" are header, blank lines are throwaway
            # Control data starts with the line "Main Scan Counter"
            # followed by "Number measurements : 256", a header line, and data
            
            if line[0] == '\n': continue # skip blank lines
            
            if line.strip() == 'Main Scan Counter   : 1': # end of metadata header
                header = False
                controldata['meta'] = metadata

            if line.strip() == 'Control data (Z position, AC, DC, phase)': # prepare for new data block
                data = True
                z_pos = []
                ac = []
                dc = []
                phase = []
                points_read = 0
                controlpoints = controlpoints + 1
                continue
            
            if not data:
            
                param, value = line.split(':',1)
                param = param.strip()        
                value = value.strip() # remove whitespace, including the trailing carriage return

                if header:
                    if not valid_headers.has_key(param):
                        print('Unrecognised header %s in file %s' % (param, filename))
                        return(1)
                    else:
                        param = valid_headers[param]
                        if param in string_headers:
                            metadata[param] = value
                        else: # all others are ints
                            metadata[param] = int(value)
               
                else:
                    if param == 'Main Scan Counter': 
                        counter = int(value)
                    if param == 'Number measurements': num_pts = int(value)


            else: # have a data block

                a,b,c,d = [int(element) for element in line.split()]
                z_pos.append(a)
                ac.append(b)
                dc.append(c)
                phase.append(d)
                points_read = points_read + 1
                
                if points_read == num_pts: # end of table
                    data = False
               
                    controldict = {}
                    # build dictionary for this data block
                    controldict['counter'] = counter
                    controldict['numpoints'] = num_pts
                    controldict['data'] = np.array([z_pos,ac,dc,phase])
                    
                    # append this to a list (one entry per point)
                    controldata['points'].append(controldict)
                    
        if len(controldata['points']) < controlpoints:
            controldict = {}
            # build dictionary for this data block
            controldict['counter'] = counter
            controldict['numpoints'] = num_pts
            controldict['data'] = np.array([z_pos,ac,dc,phase])
            
            # append this to a list (one entry per point)
            controldata['points'].append(controldict)
    
        print '%i control data points read from file %s' % (len(controldata['points']),filename)
    
    return controldata


def viewraw(controldata, select=False):
    """Plots raw data (AC, DC, phase, strain gauge)"""

    import matplotlib.pyplot as plt
    
    numpoints = len(controldata['points'])
    select = 0 if not select else select
    print select
    if select < numpoints:
        zposn = controldata['points'][select]['data'][0] 
        ac = controldata['points'][select]['data'][1] 
        dc = controldata['points'][select]['data'][2] 
        phase = controldata['points'][select]['data'][3]
    else:
        print 'ERROR: invalid control point selected for plot'
        return 1
        
    # extra some meta data for the titles etc.
    cantilever = controldata['meta']['cantilever']
    segment = controldata['meta']['wheel_pos']
    facet =  63 if segment >1015 else (segment+8)/16
    num_steps = controldata['meta']['num_steps']
    zstep = controldata['meta']['zstep']
  
    # and for this particular point
    line_count = controldata['points'][select]['counter']
    numpoints = controldata['points'][select]['numpoints']
    
    xdata = range(0,numpoints)
    
    ctrl_fig = plt.figure(figsize=(8,8))
    ax4 = ctrl_fig.add_subplot(4, 1, 4)
    ax3 = ctrl_fig.add_subplot(4, 1, 3, sharex=ax4)
    ax2 = ctrl_fig.add_subplot(4, 1, 2, sharex=ax4)
    ax1 = ctrl_fig.add_subplot(4, 1, 1, sharex=ax4)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)
    
    line_ac, = ax1.plot(xdata, ac,'x')
    line_dc, = ax2.plot(xdata, dc,'x')
    line_phase, = ax3.plot(xdata,phase,'x')
    line_zpos, = ax4.plot(xdata,zposn,'x')
    
    ax1.set_yticks(ax1.get_yticks()[1:-1])    
    ax2.set_yticks(ax2.get_yticks()[1:-1])    
    ax3.set_yticks(ax3.get_yticks()[1:-1])           
    ax4.set_yticks(ax4.get_yticks()[1:-1])
           
    ax1.set_ylabel('AC (RAW)')
    ax2.set_ylabel('DC (RAW)')
    ax3.set_ylabel('Phase (RAW)')
    ax4.set_ylabel('Z position (RAW)')
    
    ax1.yaxis.set_label_coords(-0.12, 0.5)
    ax2.yaxis.set_label_coords(-0.12, 0.5)    
    ax3.yaxis.set_label_coords(-0.12, 0.5)
    ax4.yaxis.set_label_coords(-0.12, 0.5)
        
    ax4.set_xlabel('Data point')
        
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)
    ax4.grid(True)
    
    ax4.set_xlim(xdata[0],xdata[-1])

    plt.subplots_adjust(hspace=0.0)
    
    title = plt.suptitle('Cantilever %i, facet %i, %i points, zstep = %i\nControl point %i' % (cantilever, facet, num_steps, zstep, line_count))
    
    plt.show()

    
def view(controldata, select=False, interactive=False):    
    """Accepts a controldata dictionary and plots cantilever AC, DC and phase
    versus the Z position"""
    
    import matplotlib.pyplot as plt
    import common    
    
    def update_data(select):
        zposn = controldata['points'][select]['data'][0] 
        zposn = zposn - zposn.min()
        zposn = zposn * common.zcal * 2.0
        ac = controldata['points'][select]['data'][1] * common.ac_v
        dc = controldata['points'][select]['data'][2] * common.dc_v
        phase = controldata['points'][select]['data'][3] * common.phase_deg
        
        return zposn, ac, dc, phase
    
    numpoints = len(controldata['points'])
    select = 0 if not select else select
    if select < numpoints:
        zposn, ac, dc, phase = update_data(select)
    else:
        print 'ERROR: invalid control point selected for plot'
        return 1
    
    # plot all control data for this point on a common x axis 
    # 
    
    # extra some meta data for the titles etc.
    cantilever = controldata['meta']['cantilever']
    segment = controldata['meta']['wheel_pos']
    facet =  63 if segment >1015 else (segment+8)/16
    num_steps = controldata['meta']['num_steps']
    zstep = controldata['meta']['zstep']
    
    # and for this particular point
    line_count = controldata['points'][select]['counter']
    
    ctrl_fig = plt.figure(figsize=(8,8))
    ax3 = ctrl_fig.add_subplot(3, 1, 3)
    ax1 = ctrl_fig.add_subplot(3, 1, 1, sharex=ax3)
    ax2 = ctrl_fig.add_subplot(3, 1, 2, sharex=ax3)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    line_ac, = ax1.plot(zposn,ac,'x')
    line_dc, = ax2.plot(zposn,dc,'x')
    line_phase, = ax3.plot(zposn,phase,'x')
       
    ax1.set_yticks(ax1.get_yticks()[1:-1])    
    ax2.set_yticks(ax2.get_yticks()[1:-1])    
    ax3.set_yticks(ax3.get_yticks()[1:-1])           
       
    ax1.set_ylabel('AC (V RMS)')
    ax2.set_ylabel('DC (V)')
    ax3.set_ylabel('Phase (deg)')
    
    ax1.yaxis.set_label_coords(-0.12, 0.5)
    ax2.yaxis.set_label_coords(-0.12, 0.5)    
    ax3.yaxis.set_label_coords(-0.12, 0.5)
        
    ax3.set_xlabel('Z position (nm)')
        
    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)    

    plt.subplots_adjust(hspace=0.0)
    
    title = plt.suptitle('Cantilever %i, facet %i, %i points, zstep = %i\nControl point %i' % (cantilever, facet, num_steps, zstep, line_count)) 
    
    if interactive:
        from matplotlib.widgets import Button

        class Index:
            selected = select
            def next(self, event):
                self.selected += 1
                if self.selected > (numpoints-1): self.selected = (numpoints-1)
                line_count = controldata['points'][self.selected]['counter']
                zposn, ac, dc, phase = update_data(self.selected)
                line_ac.set_xdata(zposn)
                line_ac.set_ydata(ac)   
                line_dc.set_xdata(zposn)
                line_dc.set_ydata(dc)
                line_phase.set_xdata(zposn)
                line_phase.set_ydata(phase)
                ax1.relim()
                ax1.autoscale_view(scalex=True,scaley=True)
                ax2.relim()
                ax2.autoscale_view(scalex=True,scaley=True)
                ax3.relim()
                ax3.autoscale_view(scalex=True,scaley=True)
                title.set_text('Cantilever %i, facet %i, %i points, zstep = %i\nControl point %i' % (cantilever, facet, num_steps, zstep, line_count)) 
                ctrl_fig.canvas.draw()

            def prev(self, event):
                self.selected -= 1
                if self.selected < 0: self.selected = 0
                line_count = controldata['points'][self.selected]['counter']
                zposn, ac, dc, phase = update_data(self.selected)
                line_ac.set_xdata(zposn)
                line_ac.set_ydata(ac)   
                line_dc.set_xdata(zposn)
                line_dc.set_ydata(dc)
                line_phase.set_xdata(zposn)
                line_phase.set_ydata(phase)
                ax1.relim()
                ax1.autoscale_view(scalex=True,scaley=True)
                ax2.relim()
                ax2.autoscale_view(scalex=True,scaley=True)
                ax3.relim()
                ax3.autoscale_view(scalex=True,scaley=True)
                title.set_text('Cantilever %i, facet %i, %i points, zstep = %i\nControl point %i' % (cantilever, facet, num_steps, zstep, line_count)) 
                ctrl_fig.canvas.draw()
                
            def runcal(self, event):
                calibrate(controldata['points'][self.selected])

        callback = Index()
        
        # buttons are linked to a parent axis, and scale to fit
        button_width  = 0.05
        button_height = 0.05
        start_height = 0.95
        start_width = 0.8
        axprev = plt.axes([start_width, start_height, button_width, button_height])
        axnext = plt.axes([start_width+button_width, start_height, button_width, button_height])
        axcal = plt.axes([start_width,start_height-button_height, button_width*2, button_height])

        bnext = Button(axnext, '>')
        bnext.on_clicked(callback.next)

        bprev = Button(axprev, '<')
        bprev.on_clicked(callback.prev)
        
        bcal = Button(axcal, 'Calibrate')
        bcal.on_clicked(callback.runcal)
    
    plt.show()


def calibrate(controlpoint):
    """Takes a control data structure and performs amplitude calibration, 
    prompting the user for the end of the tapping section"""

    import matplotlib.pyplot as plt    
    from scipy import stats
    import numpy as np
    import common
        
    # Interpolate piezo Z piezo values
    # First calibrate into nm (factor 0.328)

    z_nm = controlpoint['data'][0] * common.zcal * 2.0
    z_nm =z_nm - max(z_nm)
    xdata = range(len(z_nm))

    amp_v = (20.0/65535)*controlpoint['data'][1]
    peak_amp_v = amp_v * np.sqrt(2)

    # Fit Z piezo data with a straight line
    (m,c) = np.polyfit(xdata,z_nm,1)

    # And calculate fitted values
    z_fit = np.polyval([m,c],xdata)

    # Check confidence of fit and warn if too large/multi-valued etc.
    rms_err = np.sqrt(np.sum((z_fit-z_nm)**2)/256)
    # numpy.sum((yi - ybar)**2)

    # Prompt user to pick the end of the linear approach
    #
    plt.ion()
    p = pickpoint(z_fit,peak_amp_v)
    plt.draw()

    # Need to pause the script here to wait for the user to respond
    while not plt.waitforbuttonpress():
        pass

    plt.ioff()

    # Read out the index of the selected data point
    start = p.lastind

    print 'Start of linear approaching fitting at point: ', start

    # Fit a straight line from this point until the end of the data set
    #
    (m2,c2) = np.polyfit(z_fit[start:],peak_amp_v[start:],1)
    slopefit = np.polyval([m2,c2],z_fit[start:])

    # Also fit using the linregress function
    gradient, intercept, r_value, p_value, std_err = stats.linregress(z_fit[start:],peak_amp_v[start:])
    print gradient, intercept, r_value, p_value, std_err

    # With assumptions about gradient, calibrate the Y axis into nm
    # This gradient should be ~1.0 for a hard surface
    #
    print 'Gradient of actual slope (V/nm): ', m2
    cal_amp = peak_amp_v / m2

    print 'Calibrate amplitude of first point: ', cal_amp[0]

    # Plot calibrated approach curve!
    #
    # plt.close()
    fig = plt.figure()
    ax_approach = fig.add_subplot(1,1,1) # 1x1 plot grid, 1st plot # creates an object of class "Axes"
    ax_approach.grid(True)
    ax_approach.set_xlabel('Z displacement (nm)')
    ax_approach.set_ylabel('Cantilever amplitude (nm)')

    # ax_approach.set_title(os.path.basename(filename))

    # Tuple unpacking used to get first item in list returned from plot (one per line)
    amp_line, = ax_approach.plot(z_fit,cal_amp,'x')
    fit_line, = ax_approach.plot(z_fit[start:],slopefit/m2)
    fit_line, = ax_approach.plot(z_fit[start:],(slopefit+std_err)/m2)
    fit_line, = ax_approach.plot(z_fit[start:],(slopefit-std_err)/m2)

    plt.show()

    return (z_nm,cal_amp)


if __name__ == "__main__":

    import sys, os

    # Called interactively - check for a filename as an input, if none, prompt for if

    if len(sys.argv) > 2:
        sys.exit('Too many parameters - either run without anything for interactive, or give the control data filename')
        
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
        filter.set_name("Control data files")
        filter.add_pattern("*.txt")
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

    # now we have a filename - read in the data file and pass to the interactive
    # viewer (the user can choose a curve and click calibrate)

    data = read(filename)
    view(data, select=False, interactive=True)
    



