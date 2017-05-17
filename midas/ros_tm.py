#!/usr/bin/python
"""ros_tm.py

Mark S. Bentley (mark@lunartech.org), 2014

A module to read Rosetta/MIDAS telemetry and manipulate packets.

The bulk of the work is done with the tm class, which is initialised
with a filename, list of files or wildcard. A packet index is create
as tm.pkts and the class provides a set of methods to manipulate them.

e.g. tm.events() to retrieve event data, tm.get_images() to find MIDAS
scans, tm.get_params() to extract named TM parameters and tm.plot_params()
to plot the same.
"""

import struct, collections, pytz, os, math
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from midas import common
import matplotlib.pyplot as plt
import matplotlib as mpl

import logging
log = logging.getLogger(__name__)

# datefmt='%m/%d/%Y %I:%M:%S %p'
isofmt = '%Y-%m-%dT%H%M%SZ'

obt_epoch = datetime(year=2003, month=1, day=1, hour=0, minute=0, second=0) # , tzinfo=pytz.UTC)
sun_mjt_epoch = datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0)
dds_obt_epoch = datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0, tzinfo=pytz.UTC)

# Configuration file paths
gwy_settings_file = os.path.join(common.config_path, 'gwy-settings.gwy')

# Define struct types for various packet formats, to allow easy extraction of values from the
# binary data into named tuples

pkt_header_fmt = '>3HIH4B'
pkt_header_size = struct.calcsize(pkt_header_fmt)
pkt_header_names = collections.namedtuple("pkt_header_names","pkt_id pkt_seq pkt_len obt_sec obt_frac \
    checksum pkt_type pkt_subtype pad")

dds_header_fmt = '>3I2H2B'
dds_header_len = struct.calcsize(dds_header_fmt)
dds_header_names = collections.namedtuple("dds_header_names", "scet1 scet2 pkt_length gnd_id vc_id sle time_qual")

tc_pkt_header_fmt = '>3H4B'
tc_pkt_header_size = struct.calcsize(tc_pkt_header_fmt)
tc_pkt_header_names = collections.namedtuple("tc_pkt_header_names","pkt_id pkt_seq pkt_len ver_check pkt_type pkt_subtype pad")

dds_tc_header_fmt = '>4I5HI8s8s'
dds_tc_header_len = struct.calcsize(dds_tc_header_fmt)
dds_tc_header_names = collections.namedtuple("dds_tc_header_names", "uplink_t_s uplink_t_us exec_t_s exec_t_us cuv cav_vc0 cev_vc0 cav_vc1 cev_vc1 pkt_len tc sq")



# List of MIDAS specific APIDs
midas_apids = [1073, 1076, 1079, 1081, 1083, 1084]

#------------- Helper functions

def obt_to_datetime(obt):
        # isofmt = "%Y-%m-%dT%H%M%SZ"
    time = obt_epoch + timedelta(seconds=int(obt))
    return time

def obt_to_iso(obt):
    time = obt_epoch + timedelta(seconds=int(obt))
    return datetime.strftime(time,isofmt)

def lorentzian(x, offset, amp, cen, wid):
    "lorentzian function: wid = half-width at half-max"
    return (offset + amp / (1 + ((x-cen)/wid)**2))

def polycal(value, curve):
    return curve.a0 + curve.a1*value + curve.a2*value**2. + curve.a3*value**3. + curve.a4*value**4.

def list_midas_pkts(apid=False):
    """Simply filters pid.dat by the relevant MIDAS APIDs"""

    if type(apid)!=bool:
        return pid[pid.apid==apid]
    else:
        return pid[pid.apid.isin(midas_apids)]

def mil1750a_to_float(mil1750):
    """Accepts a mil1750a 32 value and returns a float"""

    from bitstring import BitArray

    raw = BitArray(format(mil1750,'#034b'))
    exponent = raw[24:32]
    mantissa = raw[0:24]

    return mantissa.int*2**(exponent.int-23)



def calibrate(param_name, values):
    """Calibrates the value(s) given according to the calibration details specified
    for parameter param_name, returns the real value and units"""

    # Extract parameter information
    param = pcf[pcf.param_name==param_name].squeeze()

    # Check if this value needs calibration
    if pd.isnull(param.cal_id):
        return values

    # See if a numerical or status parameter cal is needed
    if param.cal_cat == 'N':

        # Read cal info from the CAF
        calinfo = caf[caf.cal_id==int(param.cal_id)].squeeze()

        if len(cap[cap.cal_id==param.cal_id]): # linear cal curve
            raw_vals = cap[cap.cal_id==param.cal_id].raw_val.values
            eng_vals = cap[cap.cal_id==param.cal_id].eng_val.values
            values = np.interp(values,raw_vals,eng_vals)

        elif len(mcf[mcf.cal_id==param.cal_id]): # polynomial cal curve

            calcurve = mcf[mcf.cal_id==param.cal_id].squeeze()
            values = np.apply_along_axis(polycal, 0, values, calcurve)

        else:
            log.error('no numerical calcurve found with ID' % (param.cal_id))
            return False

    elif param.cal_cat == 'S': # status parameter

        calinfo = txf[txf.txf_id==int(param.cal_id)]
        fmt_type = calinfo.raw_fmt.squeeze()
        # Format type of the raw values in txp.dat.
        # I = signed integer, U = unsigned integer, R = real
        # BUT all data in the RMIB is U, which makes it easier
        caldata = txp[txp.txp_id==int(param.cal_id)]
        # gives from and to values and a textual status per range
        statii = np.empty_like(values, dtype=np.object)
        for idx,cal in caldata.iterrows():
            statii[np.where( (values>=cal['from']) & (values<=cal['to']))]=cal.alt
        values = statii

    return values


def midas_events():
    """Simply returns the packet ID of all event packets in the RMIB"""

    return pid[ (pid.type==5) & (pid.apid==1079) ]


def plot_line_scans(lines, units='real', label=None, align=False, title=None):
    """Plot one or more line scans. units= can be real or DAC.
    label= can be set to any attribute of lines.
    If align=True, the tip offset will be used to align plots in real space. Setting this also sets units=real"""

    if type(lines) == pd.Series:
        lines = pd.DataFrame(columns=lines.to_dict().keys()).append(lines)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    if align:
        units='real'

    if label is not None:
        if label not in lines.columns:
            log.warning('requested label (%s) not found - disabling legend' % label)
            label = None
            lab = ''
    else:
        lab = ''

    for idx, line in lines.iterrows():

        if label is not None:
            lab = line['%s'%label]
            if type(lab) == float:
                lab = "%3.2f" % lab

        if units=='real':

            ax.set_xlabel(' %c distance (microns)' % line.fast_dir.upper())
            ax.set_ylabel('Height (nm)')

            height = (line['data'] - line['data'].min()) * common.cal_factors[0]
            distance = (line.step_size*common.xycal['open']/1000.)*np.arange(line.num_steps)

            if align:
                distance += line.tip_offset

        elif units=='dac':

            ax.set_xlabel(' %c distance (DAC)' % line.fast_dir.upper())
            ax.set_ylabel('Height (DAC)')

            height = line['data']

            # TODO get open/closed loop status (not in packet header, get from HK)
            distance = (line.step_size*common.xycal['open']/1000.)*np.arange(line.num_steps)

        ax.grid(True)
        ax.plot(distance, height, label=lab)

    if label is not None:
        leg = ax.legend(loc=0, prop={'size':10}, fancybox=True, title=label)
        leg.get_frame().set_alpha(0.7)

    if title is not None:
        ax.set_title(title)

    plt.show()

    return


def plot_fscan(fscans, showfit=False, legend=True, cantilever=None, xmin=False, xmax=False, ymin=False, ymax=False,
        figure=None, axis=None, title=True, marker=None):
    """Plots one or more frequency scan (read previously with get_freq_scans()). Optionally
    plot a Lorentzian fit"""

    if type(fscans) == pd.Series:
        fscans = pd.DataFrame(columns=fscans.to_dict().keys()).append(fscans)

    if cantilever is not None:
        fscans = fscans[ fscans.tip_num==cantilever ]

    if len(fscans)==0:
        log.error('no frequency scans available (for selected cantilever)')
        return None

    if figure is None:
        fig = plt.figure()
    else:
        fig = figure

    if axis is None:
        ax = fig.add_subplot(1,1,1)
    else:
        ax = axis

    ax.set_xlabel('Frequency (Hz)')
    ax.grid(True)

    for idx, scan in fscans.iterrows():

        if scan.is_phase:
            ax.set_ylabel('Phase (deg)')
        else:
            ax.set_ylabel('Amplitude (V)')

        ax.plot(scan['frequency'],scan['amplitude'], label='%s' % scan.start_time, marker=marker)

        if showfit and ~scan.is_phase:
            if not 'offset' in scan.index:
                log.warning('no fit data in current frequency scan')
            else:
                ax.plot(scan.frequency, lorentzian(scan.frequency, scan.offset, scan.fit_max-scan.offset, \
                        scan.res_freq, scan.half_width),label='Lorentzian fit')

    if len(fscans)==1 and title:
        #  Freq start/step: %3.2f/%3.2f % scan.freq_start, scan.freq_step
        ax.set_title('Tip %i, Ex/Gn: %i/%i, peak %3.2f V @ %3.2f Hz' % \
            (scan.tip_num, scan.exc_lvl, scan.ac_gain, scan.max_amp, scan.max_freq), fontsize=10 )

        if set(['res_amp','work_pt', 'set_pt', 'fadj', 'op_pt_hi', 'op_pt_lo']).issubset(set(scan.keys())) and not scan.is_phase:
            # Also drawn lines showing the working point and set point
            ax.axhline(scan.res_amp,color='k')
            ax.axhline(scan.work_pt,color='r')
            ax.axhline(scan.set_pt,color='g')
            ax.axhline(scan.fadj,color='g', ls='--')
            ax.axhline(scan.op_pt_hi,color='g', ls='-.')
            ax.axhline(scan.op_pt_lo,color='g', ls='-.')

    if legend:
        leg = ax.legend(loc=0, prop={'size':10}, fancybox=True)
        leg.get_frame().set_alpha(0.7)

    if not xmin: xmin = ax.get_xlim()[0]
    if not xmax: xmax = ax.get_xlim()[1]
    if not ymin: ymin = ax.get_ylim()[0]
    if not ymax: ymax = ax.get_ylim()[1]

    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)

    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.xaxis.get_major_formatter().set_useOffset(False)

    if figure is None:
        plt.show()

    return fig


def plot_fscan_grid(fscans, cols=4, title=False):
    """Simply calls plot_fscan over a grid, typically for showing
    the results of a frequency scan survey"""

    from matplotlib import gridspec

    num_fscans = len(fscans)
    rows = num_fscans / cols
    if num_fscans % cols != 0:
        rows += 1
    if num_fscans<cols:
        cols=num_fscans

    def onresize(event):
        plt.tight_layout()

    gs = gridspec.GridSpec(rows, cols)
    fig = plt.figure()
    cid = fig.canvas.mpl_connect('resize_event', onresize)
    grid = 0

    for idx, fscan in fscans.iterrows():

        ax = plt.subplot(gs[grid])

        if len(fscans.query('~is_phase'))==len(fscans): # all amplitude
            plot_fscan(fscan, legend=False, ymax=10.0, figure=fig, axis=ax, title=title)
        else:
            plot_fscan(fscan, legend=False, figure=fig, axis=ax, title=title)

        plt.setp(ax.get_xticklabels(), rotation=45)
        grid += 1

    plt.tight_layout()
    plt.show()

    return

def plot_ctrl_data(ctrldata, interactive=True, fix_scale=False, channels=['AC', 'DC', 'PH', 'ZS'], zero_z=True):
    """Accepts a set of control data generated by get_ctrl_data() and plots all four channels.

    If interactive=True the user can navigate through all data present.

    If fix_scale=True the scale will be fixed to encompase the limits of the entire data set."""

    from matplotlib.widgets import Button as mplButton

    if type(ctrldata) == pd.Series:
        ctrldata = pd.DataFrame(columns=ctrldata.to_dict().keys()).append(ctrldata)

    # check requested channels against valid options
    valid_chans = ['AC', 'DC', 'PH', 'ZS']
    chan_labels = {
        'AC': 'Cant AC (V)',
        'DC': 'Cant DC (V)',
        'PH': 'Phase (deg)',
        'ZS': 'Z position (nm)' }

    if type(channels)!=list:
        channels=[channels]
    channels = list(set(channels) & set(valid_chans))

    num_chans = len(channels)
    if num_chans==0:
        log.error('no valid channel provided')
        return None

    numpoints = len(ctrldata)
    if numpoints==0:
        log.warning('no control data passed')
        return None

    ctrl_fig = plt.figure()
    axes = {}
    chan_idx = 0
    for chan in channels[::-1]:
        if chan_idx==0:
            first_ax = ctrl_fig.add_subplot(num_chans, 1, num_chans-chan_idx)
            axes.update({chan: ctrl_fig.add_subplot(num_chans, 1, num_chans-chan_idx)})
        else:
            axes.update({chan: ctrl_fig.add_subplot(num_chans, 1, num_chans-chan_idx, sharex=axes[channels[-1]])})
        chan_idx += 1

    class Index:

        def __init__(self):
            self.selected = 0
            self.ctrl = ctrldata.iloc[self.selected]
            self.update(firsttime=True)
            self.update()

        def update(self, firsttime=False):

            self.ctrl = ctrldata.iloc[self.selected]
            xdata = range(0,int(self.ctrl.num_meas/4))

            if firsttime:

                if 'AC' in channels:
                    self.line_ac, =  axes['AC'].plot(xdata, self.ctrl.ac,'x-')
                    # draw the set point and set point up/dn limits if possible
                    if set(['set_pt', 'op_pt_lo', 'op_pt_hi']).issubset( self.ctrl.keys() ):
                        axes['AC'].axhline(self.ctrl.set_pt, color='g', ls='-')
                        axes['AC'].axhline(self.ctrl.op_pt_lo, color='r', ls='--')
                        axes['AC'].axhline(self.ctrl.op_pt_hi, color='r', ls='--')

                if 'DC' in channels:
                    self.line_dc, = axes['DC'].plot(xdata, self.ctrl.dc,'x-')
                if 'PH' in channels:
                    self.line_phase, = axes['PH'].plot(xdata, self.ctrl.phase,'x-')
                if 'ZS' in channels:
                    if zero_z:
                        self.line_zpos, = axes['ZS'].plot(xdata, self.ctrl.zpos-self.ctrl.zpos.min(),'x-')
                    else:
                        self.line_zpos, = axes['ZS'].plot(xdata, self.ctrl.zpos,'x-')
                [plt.setp(axes[chan].get_xticklabels(), visible=False) for chan in channels[:-1]]
                [axes[chan].set_yticks(axes.values()[0].get_yticks()[1:-1]) for chan in channels[:-1]]
                [axes[chan].set_ylabel(chan_labels[chan]) for chan in channels]
                [axes[chan].yaxis.set_label_coords(-0.12, 0.5) for chan in channels]
                [axes[channels[-1]].set_xlabel('Data point')]
                [ax.grid(True) for ax in axes.values()]

                plt.subplots_adjust(hspace=0.0)

                if fix_scale:
                    if 'AC' in channels:
                        self.ac_max = max([ctrldata.ac.iloc[idx].max() for idx in range(len(ctrldata.ac))])
                        self.ac_min = min([ctrldata.ac.iloc[idx].min() for idx in range(len(ctrldata.ac))])
                    if 'DC' in channels:
                        self.dc_max = max([ctrldata.dc.iloc[idx].max() for idx in range(len(ctrldata.dc))])
                        self.dc_min = min([ctrldata.dc.iloc[idx].min() for idx in range(len(ctrldata.dc))])
                    if 'PH' in channels:
                        self.ph_max = max([ctrldata.phase.iloc[idx].max() for idx in range(len(ctrldata.phase))])
                        self.ph_min = min([ctrldata.phase.iloc[idx].min() for idx in range(len(ctrldata.phase))])
                    if 'ZS' in channels:
                        self.zpos_max = max([ctrldata.zpos.iloc[idx].max() for idx in range(len(ctrldata.zpos))])
                        self.zpos_min = max([ctrldata.zpos.iloc[idx].min() for idx in range(len(ctrldata.zpos))])

                [ax.yaxis.get_major_formatter().set_useOffset(False) for ax in axes.values()]

                from matplotlib.ticker import MaxNLocator
                [axes[chan].yaxis.set_major_locator(MaxNLocator(nbins=5, prune='both')) for chan in channels]

                self.title = plt.suptitle('Wheel seg %i, tip #%i, origin (%i,%i), step %i/%i' %
                    ( self.ctrl.wheel_pos, self.ctrl.tip_num, self.ctrl.x_orig, self.ctrl.y_orig, self.ctrl.main_cnt, self.ctrl.num_steps))

            else:

                if 'AC' in channels:
                    self.line_ac.set_xdata(xdata)
                    self.line_ac.set_ydata(self.ctrl.ac)
                if 'DC' in channels:
                    self.line_dc.set_xdata(xdata)
                    self.line_dc.set_ydata(self.ctrl.dc)
                if 'PH' in channels:
                    self.line_phase.set_xdata(xdata)
                    self.line_phase.set_ydata(self.ctrl.phase)
                if 'ZS' in channels:
                    self.line_zpos.set_xdata(xdata)
                    if zero_z:
                        self.line_zpos.set_ydata(self.ctrl.zpos-self.ctrl.zpos.min())
                    else:
                        self.line_zpos.set_ydata(self.ctrl.zpos)

                self.title.set_text('Wheel seg %i, tip #%i, origin (%i,%i), step %i/%i' %
                    ( self.ctrl.wheel_pos, self.ctrl.tip_num, self.ctrl.x_orig, self.ctrl.y_orig, self.ctrl.main_cnt, self.ctrl.num_steps))

                if fix_scale:
                    if 'AC' in channels:
                        axes['AC'].set_ylim(self.ac_min, self.ac_max)
                    if 'DC' in channels:
                        axes['DC'].set_ylim(self.dc_min, self.dc_max)
                    if 'PH' in channels:
                        axes['PH'].set_ylim(self.ph_min, self.ph_max)
                    if 'ZS' in channels:
                        axes['ZS'].set_ylim(self.zpos_min, self.zpos_max)

                else:
                    for chan in channels:
                        axes[chan].relim()
                        axes[chan].autoscale_view(scaley=True)

            axes[channels[-1]].set_xlim(xdata[0],xdata[-1])
            ctrl_fig.canvas.draw()


        def next(self, event):
            self.selected += 1
            if self.selected > (numpoints-1):
                self.selected = (numpoints-1)
            self.update()

        def prev(self, event):
            self.selected -= 1
            if self.selected < 0: self.selected = 0
            self.update()

        def runcal(self, event):
            calibrate_amplitude(ctrldata.iloc[self.selected])

    callback = Index()

    if interactive:

        def pressed(event):
            if event.key=='left':
                callback.prev(event)
            if event.key=='right':
                callback.next(event)
            if event.key=='home':
                callback.selected = 0
                callback.update()
            if event.key=='end':
                callback.selected = numpoints-1
                callback.update()
            if event.key=='c':
                callback.runcal(event)

        # buttons are linked to a parent axis, and scale to fit
        button_width  = 0.05
        button_height = 0.05
        start_height = 0.95
        start_width = 0.8
        axprev = plt.axes([start_width, start_height, button_width, button_height])
        axnext = plt.axes([start_width+button_width, start_height, button_width, button_height])
        axcal = plt.axes([start_width,start_height-button_height, button_width*2, button_height])

        bnext = mplButton(axnext, '>')
        bnext.on_clicked(callback.next)

        bprev = mplButton(axprev, '<')
        bprev.on_clicked(callback.prev)

        bcal = mplButton(axcal, 'Calibrate')
        bcal.on_clicked(callback.runcal)

        ctrl_fig.canvas.mpl_connect('key_press_event', pressed)

    plt.show()

def stp_from_name(name):
    """Simply returns the integer STP number from a scan or TLM filename"""

    if name[13].upper()!='S':
        log.error('this filename does not seem to contain an STP number...')
        return None

    return int(name[14:17])


def calibrate_amplitude(ctrl, return_data=False):
    """Interactively calibrate the cantilever amplitude via control data. Accepts a
    single point approach and the user must select the end of the linear approach
    section. A straight line is fit and the amplitude returned."""

    class pickpoint:

        def __init__(self,xs,ys):

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

    ### Start of function

    from scipy import stats

    # Interpolate piezo Z piezo values
    # First calibrate into nm (factor 0.328)

    z_nm = ctrl['zpos'] * common.zcal * 2.0
    z_nm =z_nm - max(z_nm)
    xdata = range(len(z_nm))

    amp_v = (20.0/65535)*ctrl['ac']
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

    # print 'Start of linear approaching fitting at point: ', start

    # Fit a straight line from this point until the end of the data set
    #
    (m2,c2) = np.polyfit(z_fit[start:],peak_amp_v[start:],1)
    slopefit = np.polyval([m2,c2],z_fit[start:])

    # Also fit using the linregress function
    gradient, intercept, r_value, p_value, std_err = stats.linregress(z_fit[start:],peak_amp_v[start:])
    # print gradient, intercept, r_value, p_value, std_err

    # With assumptions about gradient, calibrate the Y axis into nm
    # This gradient should be ~1.0 for a hard surface
    #
    # print 'Gradient of actual slope (V/nm): ', m2
    cal_amp = peak_amp_v / m2

    # print 'Calibrate amplitude of first point: ', cal_amp[0]

    # Plot calibrated approach curve!
    p.fig.clf()
    ax_approach = p.fig.add_subplot(1,1,1) # 1x1 plot grid, 1st plot # creates an object of class "Axes"
    ax_approach.grid(True)
    ax_approach.set_xlabel('Z displacement (nm)')
    ax_approach.set_ylabel('Cantilever amplitude (nm)')

    # ax_approach.set_title(os.path.basename(filename))

    # Tuple unpacking used to get first item in list returned from plot (one per line)
    amp_line, = ax_approach.plot(z_fit,cal_amp,'x')
    fit_line, = ax_approach.plot(z_fit[start:],slopefit/m2)
    fit_line, = ax_approach.plot(z_fit[start:],(slopefit+std_err)/m2)
    fit_line, = ax_approach.plot(z_fit[start:],(slopefit-std_err)/m2)

    plt.draw()

    if return_data:
        return (z_nm,cal_amp)


def image_from_lines(lines, slow_dir='L_H', get_hk=False):

    if len(lines[lines.sw_ver<661])>0:
        log.warning('OBSW < 661 does not flag if lines come from images')
    else:
        lines = lines.query('in_image')
        if len(lines)==0:
            log.error('no image-forming line scans')
            return None

    if lines.line_cnt.max()!=len(lines):
        log.warning('%d lines available, but maximum line count is %d' % (len(lines),lines.line_cnt.max()))

    # Do a few sanity checks on the provided lines
    cols = lines.columns.tolist()
    dropcols = ['obt', 'line_cnt', 'aborted', 'data', 'start_time'] # these are necessarily difference, even in a single image scan
    cols = [col for col in cols if col not in dropcols]
    if lines.duplicated(subset=cols).sum()+1 != len(lines):
        log.error('some metadata is different, make sure lines are from the same image scan!')
        return None

    # take meta-data from first line (should all be identical)
    line = lines.iloc[0].squeeze()

    if not get_hk:
        slow_dir = slow_dir.upper()
        if slow_dir not in ['L_H','H_L']:
            log.error('slow_dir must be either L_H or H_L')
            return None
    else:
        tlm = tm(lines.filename.unique()[0])
        meta = tlm.meta_from_hk(lines.obt.min(), lines.obt.max())
        # extracting linear position from line data is better
        meta.lin_pos = line.lin_pos
        meta.tip_offset = (meta.lin_pos - common.lin_centre_pos_fm[meta.tip_num-1]) / common.linearcal
        meta.tip_offset = meta.tip_offset - common.tip_offset[meta.tip_num-1]
        slow_dir = meta.x_dir if meta.fast_dir=='Y' else meta.y_dir
        meta.duration = meta.end_time - meta.start_time

    # extract line scan data into an array
    data = []
    lines = lines.sort_values(by='line_cnt')
    [data.append(line['data']) for idx,line in lines.iterrows()]
    image = np.array(data)

    # re-shape array according to main scan direction
    ysteps = line.num_steps if line.fast_dir=='Y' else len(lines)
    xsteps = line.num_steps if line.fast_dir=='X' else len(lines)

    # flip if necessary to account for piezo direction
    if line.fast_dir=='Y':
        image = np.transpose(image.reshape( (xsteps,ysteps) ))
    else:
        image = image.reshape( (ysteps,xsteps) )

    if line.fast_dir=='Y':
        if line.dir=='H_L':
            image = np.flipud(image)
        if slow_dir=='H_L':
            image = np.fliplr(image)

    if line.fast_dir=='X':
        if line.dir=='H_L':
            image = np.flipud(image)
        if slow_dir=='H_L':
            image = np.fliplr(image)

    if get_hk:
        meta['data'] = image
        return meta
    else:
        return image


def combine_linescans(linescans, bcr=False):
    """Accepts a DataFrame containing linescans and simply returns a numpy array of the lines
    stacked into an array, ordered as given.

    If bcr= is set to a filename, a minimal BCR file is produced - note that the
    line data are assumed to be in the X direction, and the Y step is assumed
    to be the same as the X!"""

    data = []
    [data.append(line['data']) for idx,line in linescans.iterrows()]
    image = np.array(data)

    if bcr:
        import bcrutils, midas
        bcrdata = {}
        bcrdata['filename'] = bcr
        bcrdata['xpixels'] = image.shape[1]
        bcrdata['ypixels'] = image.shape[0]
        # TODO include closed loop?
        bcrdata['xlength'] = linescans.iloc[0].step_size*common.xycal['open']*bcrdata['xpixels']
        bcrdata['ylength'] = linescans.iloc[0].step_size*common.xycal['open']*bcrdata['ypixels']
        bcrdata['data'] = image.flatten()
        # bcrdata['data'] = 32767 - bcrdata['data']
        bcrutils.write(bcrdata)

    return bcrdata if bcr else image


def to_bcr(images, outputdir='.'):
    """Accepts one or more sets of image data from get_images() and returns individual
    BCR files using the bcrutils module."""

    import bcrutils, midas
    import  math

    bcrs = []
    scan_count = 0
    obt = 0

    # Rather hacky way of taking a series (if passed) and converting back
    # to a dataframe (to deal with a single image)
    if type(images) == pd.Series:
        images = pd.DataFrame(columns=images.to_dict().keys()).append(images)

    if 'data' not in images.columns:
        log.error('image data not found - be sure to run tm.get_images with info_only=False')
        return False

    images = images.sort_values( by=['filename', 'start_time'] )
    last_filename = images.filename.iloc[0]

    for idx in range(len(images)):

        image = images.iloc[idx]

        if image.filename != last_filename:
            scan_count = 0
            last_filename = image.filename

        if image.start_time != obt:
            obt = image.start_time
            scan_count = scan_count + 1

        # deal with open and closed loop setting!
        ycal = common.xycal['closed'] if image.y_closed else common.xycal['open']
        xcal = common.xycal['closed'] if image.x_closed else common.xycal['open']

        bcrdata = {}

        xoffset, yoffset = bcrutils.set_origin(image.x_orig,image.y_orig, xopen=not image.x_closed, yopen=not image.y_closed)

        bcrdata['xpixels'] = image.xsteps
        bcrdata['ypixels'] = image.ysteps
        bcrdata['xlength'] = image.x_step * xcal * bcrdata['xpixels']
        bcrdata['xoffset'] = xoffset
        bcrdata['ylength'] = image.y_step * ycal * bcrdata['ypixels']
        bcrdata['yoffset'] = yoffset

        bcrdata['data'] = images.data.iloc[idx].ravel() - 32768

        chan_idx = common.data_channels.index(image.channel)

        suffix = common.data_channels[chan_idx]

        # Calibrate according to the channel
        bcrdata['bit2nm'] = common.cal_factors[chan_idx]

        # offset = int(round(common.offsets[chan_idx] / common.cal_factors[chan_idx]))
        # bcrdata['data'] += offset

        bcrdata['zunit'] = common.units[chan_idx]

        target = common.seg_to_facet(image.wheel_pos)


        # Filename convention, e.g.
        # SCAN_MD_Mmmm_Ssss_Ccc_Iiii__TGTtt_ZS.bcr
        #    mmm & sss = MTP and STP cycle
        #    cc = observation count (assuming we always use the TARGET_SCAN observation)
        #    iii = integer count of image (unique OBT) in this observation
        #    tt = facet number (0-63)
        src_filename = os.path.basename(image['filename'])
        # check if the source data filename has the MTP/STP format
        # TLM__MD_Mnnn_Smmm_zzzzzzzzzzzzzzzzzzzz_COUNT_cc.DAT
        if src_filename[0:9]=='TLM__MD_M' and src_filename[13]=='S':
            filename = src_file_to_img_file(src_filename, image.start_time, target) + ('_%s.bcr' % suffix)
            # mtp = src_filename[9:12]
            # stp = src_filename[14:17]
            # text = src_filename[18:38]
            # cnt = src_filename[45:47]
            # scan_cnt = '%03d'%(scan_count)
            # filename = 'SCAN_MD_M%s_S%s_C%s_I%s_TGT%02d_%s.bcr' % (mtp,stp,cnt,scan_cnt,target,suffix)
            # filename = 'SCAN_MD_M%s_S%s_%s_TGT%02d_%s.bcr' % (mtp,stp,start_time,target,suffix)
            # filename = 'SCAN_MD_M%s_S%s_%s_TGT%02d_%s.bcr' % (mtp,stp,start_time,target,suffix)

        else:
            start_time = datetime.strftime(image.start_time,isofmt)
            filename = 'MD_IMG_%s_TGT%02d_%s.bcr' % (start_time,target,suffix)

        bcrdata['filename'] = os.path.join(outputdir,filename)
        bcrs.append(bcrdata)

    return bcrs if len(bcrs) > 1 else bcrs[0]


def save_gwy(images, outputdir='.', save_png=False, pngdir='.', telem=None):
    """Accepts one or more sets of image data from get_images() and returns individual
    GWY files, with multiple channels combined in single files and all meta-data.

    If save_png=True and the image contains a topographic channel, a bitmap will be
    produced after a polynominal plane subtraction and saved to pngdir."""

    import common, gwy, gwyutils, math

    if images is None: return None

    # tidy up the input data
    if type(images) == pd.Series: images = pd.DataFrame(columns=images.to_dict().keys()).append(images)
    if type(images)==bool:
        if not images: return

    if 'data' not in images.columns:
        log.error('image data not found - be sure to run tm.get_images with info_only=False')
        return None

    scan_count = 0
    first_time = True
    filenames = []

    last_filename = images.filename.iloc(0)
    xy_unit, xy_power = gwy.gwy_si_unit_new_parse('nm')

    for start_time in images.start_time.unique():

        scan = images[images.start_time==start_time]
        scan_count += 1
        num_channels = len(scan)
        target = common.seg_to_facet(scan.wheel_pos.iloc[0])
        src_filename = os.path.basename(scan['filename'].iloc[0])

        if src_filename[0:9]=='TLM__MD_M' and src_filename[13]=='S':
            filename = src_file_to_img_file(src_filename, scan.iloc[0].start_time, target)+'.gwy'
        else:
            start = datetime.strftime(scan.iloc[0].start_time,isofmt)
            filename = 'MD_IMG_%s_TGT%02d.gwy' % (start,target)

        # Create a new gwy image container
        c = gwy.Container()

        # Loop through the channels in each scan
        for idx in range(len(scan)):

            channel = scan.iloc[idx]
            chan_idx = common.data_channels.index(channel.channel)

            if chan_idx==0: meta_channel = channel

            # deal with open and closed loop setting!
            xcal = common.xycal['closed'] if channel.x_closed else common.xycal['open']
            ycal = common.xycal['closed'] if channel.y_closed else common.xycal['open']

            xlen = channel.xsteps*channel.x_step*xcal*10**xy_power
            ylen = channel.ysteps*channel.y_step*ycal*10**xy_power

            # Default calibration needs to be changed if we requested phase data in M1, M2 or M3
            # M1 = 256, M2 = 2048, M3 = 16384
            mag_chans = ['M1','M2','M3']
            if (channel.channel in mag_chans) and channel.mag_phase:
                phase_idx = common.data_channels.index('PH')
                cal_factor = common.cal_factors[phase_idx]
                offset = int(round(common.offsets[phase_idx] / common.cal_factors[phase_idx]))
                z_unit, z_power = gwy.gwy_si_unit_new_parse(common.units[phase_idx])
                phase_title = common.channel_names[chan_idx]
                phase_title = phase_title.replace('RMS signal', 'Phase')
                c.set_string_by_name('/%i/data/title' % (chan_idx), phase_title)
            else:
                cal_factor = common.cal_factors[chan_idx]
                offset = int(round(common.offsets[chan_idx] / common.cal_factors[chan_idx]))
                z_unit, z_power = gwy.gwy_si_unit_new_parse(common.units[chan_idx])
                c.set_string_by_name('/%i/data/title' % (chan_idx),common.channel_names[chan_idx])

            datafield = gwy.DataField(int(channel.xsteps), int(channel.ysteps), xlen, ylen, True)
            datafield.set_si_unit_xy(xy_unit)
            datafield.set_si_unit_z(z_unit)

            a = gwyutils.data_field_data_as_array(datafield)
            a[:] = scan.data.iloc[idx].T

            # Create a mask of "bad" pixels for topgraphy channels
            if chan_idx==0:
                # duplicate the image channel to get dimensions etc. right
                mask = datafield.duplicate()
                m = gwyutils.data_field_data_as_array(mask)
                # Set all points to unmasked, then flag pixels with min and max extension
                m[:] = 0.
                m[a==0] = 1.
                m[a==65535] = 1.
                # if there are bad pixels, add the max - otherwise don't
                if len(m[m==1.])>0:
                    c.set_object_by_name('/%i/mask' % (chan_idx), mask)

                if channel.ctrl_image:
                    # check for control data packets sent between image start and image end (+/- 5 minutes)
                    if telem is None:
                        telem = tm(scan['filename'].iloc[0])
                    ctrl = telem.get_ctrl_data()
                    if ctrl is not None:

                        ctrl = ctrl[ (ctrl.wheel_pos==channel.wheel_pos) &
                            (ctrl.tip_num==channel.tip_num) & (ctrl.in_image) &
                            (ctrl.obt > (channel.start_time-pd.Timedelta(minutes=5))) &
                            (ctrl.obt < (channel.end_time+pd.Timedelta(minutes=5)))]

                        if len(ctrl)>0:

                            for ctrl_chan in range(len(common.ctrl_channels)): # GWY stores point per channel, so loop over this first

                                spec = gwy.Spectra()
                                spec.set_title(common.ctrl_names[ctrl_chan])
                                spec.set_si_unit_xy(xy_unit)
                                x_unit, x_power = gwy.gwy_si_unit_new_parse('index')
                                y_unit, y_power = gwy.gwy_si_unit_new_parse(common.ctrl_units[ctrl_chan])

                                i = 0

                                for idx, ctrl_pt in ctrl.iterrows(): # loop over control data points

                                    num_pix = len(ctrl_pt.zpos)
                                    spec_data = gwy.DataLine(num_pix, num_pix, False)
                                    spec_data.set_si_unit_x(x_unit)
                                    spec_data.set_si_unit_y(y_unit)

                                    ctrl_data = ctrl_pt['%s' % common.ctrl_channels[ctrl_chan]]
                                    for pt in range(num_pix):
                                        spec_data.set_val(pt, ctrl_data[pt])

                                    if channel.fast_dir=='X':
                                        xpos = ctrl_pt.main_cnt * ctrl_pt.step_size * xcal * 10**xy_power
                                        # ypos = range(1,int(channel.ysteps),(int(channel.ysteps)/32))[(i // 32)] * channel.y_step * ycal * 10**xy_power
                                        # ypos = ((i // 32)+1) * channel.y_step * ycal * 10**xy_power
                                        ypos = ctrl_pt.line_cnt * channel.y_step * ycal * 10**xy_power

                                    else:
                                        ypos = ctrl_pt.main_cnt * ctrl_pt.step_size * ycal * 10**xy_power
                                        # xpos = range(1,int(channel.xsteps),(int(channel.xsteps)/32))[(i // 32)] * channel.x_step * xcal * 10**xy_power
                                        # xpos = ((i // 32)+1) * channel.x_step * xcal * 10**xy_power
                                        xpos = ctrl_pt.line_cnt * channel.x_step * xcal * 10**xy_power

                                    spec.add_spectrum(spec_data, xpos, ypos)
                                    i += 1

                                c.set_object_by_name('/sps/%i' % ctrl_chan, spec)

            # Calibrate channel according to cal-factor
            a *= cal_factor*10**z_power

            c.set_object_by_name('/%i/data' % (chan_idx), datafield)

            # Set topography channel to visible, hide others by default
            visibility = True if chan_idx==0 else False

            c.set_boolean_by_name('/%i/data/visible' % (chan_idx), visibility)
            # Set data display to physically square by default
            c.set_boolean_by_name('/%i/data/realsquare' % (chan_idx), True)

            # Add relevant metadata

            # Tidy up some data
            if not pd.isnull(channel.duration): channel.duration = "%s" % timedelta( seconds =  channel.duration / np.timedelta64(1, 's'))
            channel.filename = os.path.basename(channel.filename)

            meta = gwy.Container()
            for key,value in channel.iteritems():
                if key == 'data': continue
                if pd.isnull(value): continue
                meta.set_string_by_name(key,str(value))

            c.set_object_by_name('/%i/meta' % (chan_idx), meta)

        # end of channel loop - now container c has all channels

        filename = os.path.join(outputdir,filename)
        filenames.append(filename)
        gwy.gwy_file_save(c, filename, gwy.RUN_NONINTERACTIVE)

        # Produce a PNG here if required
        if save_png:

            if first_time:
                import socket
                if socket.gethostname()=='midas':
                    os.environ["DISPLAY"]=':1'
                gwy.gwy_app_settings_load(gwy_settings_file)
                first_time = False

            if '/0/data' in c.keys_by_name(): # i.e. the topography channel

                datafield = c.get_value_by_name('/0/data')
                gwy.gwy_app_data_browser_add(c) # pops up a window :-(

                # Perform basic operations - 3rd order polynominal levelling, fix zero.
                datafield.subtract_polynom(3,3,datafield.fit_polynom(3,3))
                datafield.add(-1.0 * datafield.get_min())
                datafield.data_changed() # possibly not necessary?

                pngfile = os.path.join(pngdir, os.path.splitext(os.path.basename(filename))[0]+'.png' )
                gwyutils.save_dfield_to_png(c, '/0/data', pngfile, gwy.RUN_NONINTERACTIVE)

                # Produce a metadata text file, read by the image gallery software
                meta_channel.drop(['data','filename'], inplace=True)
                meta_file = pngfile+'.txt'

                meta_channel.to_frame(name=os.path.basename(filename)).to_csv(meta_file, index=True, sep=':', index_label=False, line_terminator='\r\n')
                gwy.gwy_app_data_browser_remove(c)

            else:
                log.warning('image contains no topography channel, no PNG produced')

    log.info('written %i Gwyddion files to directory %s' % (scan_count, os.path.abspath(outputdir)))

    return filenames


def open_gwy(images, path=common.gwy_path):
    """Accepts one or more images and loads the corresponding files into Gwyddion. If
    path= is not set, the default image path is used. If images is a string, this is
    assumed to be a scan file name."""

    import subprocess

    if type(images) == pd.Series:
        # images = pd.DataFrame(columns=images.to_dict().keys()).append(images)
        gwyfiles = [images.scan_file]
    elif type(images) == pd.DataFrame:
        gwyfiles = images.scan_file.unique().tolist()
    elif type(images) == str:
        gwyfiles = [images]
    elif type(images) == list:
        gwyfiles = images
    else:
        log.error('unrecognised image type')
        return None

    gwyfiles = [os.path.join(path,gwy+'.gwy') for gwy in gwyfiles]
    command_string = ['gwyddion', '--remote-new'] + gwyfiles

    subprocess.Popen(command_string)

    return



def cp_gwy(images, output='.', path=common.gwy_path):
    """Accepts one or more images or scan file names and copies the corresponding
    Gwyddion files to the path given in output= (default is current directory)"""

    import shutil

    if type(images) == pd.Series:
        images = pd.DataFrame(columns=images.to_dict().keys()).append(images)
    elif type(images) == pd.DataFrame:
        gwyfiles = images.scan_file.unique().tolist()

    if type(images) == str:
        gwyfiles = [images]

    if type(images) == list:
        gwyfiles = images

    gwyfiles = [os.path.join(path,gwy+'.gwy') for gwy in gwyfiles]

    [shutil.copy(f,output) for f in gwyfiles]

    return


def save_bcr(images, outputdir='.', write_meta=False):
    """Save a set of images as BCRs. If write_meta=True then a matching
    txt file is written with all image meta-data"""

    import bcrutils

    if images is None: return None

    if 'data' not in images.columns:
        log.error('image data not found - be sure to run tm.get_images with info_only=False')
        return None

    bcrs = to_bcr(images,outputdir=outputdir)

    if not bcrs: return False

    if type(bcrs) is not list: bcrs = [bcrs]
    [bcrutils.write(bcr) for bcr in bcrs]

    if write_meta:
        # strip the data column if it exists (drop throws an exception)
        images = images[ [col for col in images.columns if col not in ['data'] ] ]
        bcr_idx = 0
        for idx in range(len(images)):
            image = images.iloc[idx]
            meta_file = os.path.splitext(bcrs[bcr_idx]['filename'])[0]+'.txt'
            if not pd.isnull(image.duration): image.duration = "%s" % timedelta( seconds =  image.duration / np.timedelta64(1, 's'))
            image.filename = os.path.basename(image.filename)
            image.to_csv(meta_file, index=True, sep=':')
            bcr_idx += 1

    log.info('written %i BCR files to directory %s' % (len(images), os.path.abspath(outputdir)))

    return


def save_png(images, outputdir='.'):
    """Save a set of png images produced from images"""

    import bcrutils
    bcrs=to_bcr(images, outputdir)
    if not bcrs: return False
    if type(bcrs) is not list: bcrs = [bcrs]
    [bcrutils.plot2d(bcr,writefile=True) for bcr in bcrs]


def img_crop(image, (xmin, xmax), (ymin, ymax)):
    """Accepts an image and performs a crop specified in pixels as a
    tuple with form (xmin, xmax), (ymin, ymax)"""

    newimage = image.copy()

    if (xmin<0) or (xmin>xmax) or (xmin>image.xsteps):
        log.error('xmin value out of range')
        return None

    if (xmax<xmin) or (xmax>image.xsteps):
        log.error('xmax value out of range')
        return None

    if (ymin<0) or (ymin>ymax) or (ymin>image.ysteps):
        log.error('ymin value out of range')
        return None

    if (ymax<ymin) or (ymax>image.ysteps):
        log.error('ymax value out of range')
        return None

    newimage['data'] = newimage['data'][ymin:ymax,xmin:xmax]

    newimage['xsteps'] = newimage['data'].shape[1]
    newimage['ysteps'] = newimage['data'].shape[0]

    xcal = common.xycal['closed'] if image.x_closed else common.xycal['open']
    ycal = common.xycal['closed'] if image.y_closed else common.xycal['open']

    newimage['xlen_um'] = newimage['xsteps'] * newimage['x_step'] * xcal / 1.e3
    newimage['ylen_um'] = newimage['ysteps'] * newimage['y_step'] * ycal / 1.e3

    return newimage


def img_scale(image, scale_factor):
    """Scales an image by a given scale factor"""

    from scipy.ndimage.interpolation import zoom

    newimage = image.copy()
    data = np.array(image['data'], dtype=np.float64)

    log.info('original image size %dx%d' % (data.shape[1], data.shape[0]))

    # Resample via ndimage.interpolation - order = 0 = nearest neighbour
    newdata = zoom(data, zoom=scale_factor, mode='nearest', order=0)
    newimage['data'] = np.array(newdata, dtype=np.int64)

    newimage['xsteps'] = newimage['data'].shape[1]
    newimage['ysteps'] = newimage['data'].shape[0]

    newimage['x_step'] = image['x_step'] * 1./ scale_factor
    newimage['y_step'] = image['y_step'] * 1./ scale_factor

    newimage['x_step_nm'] = image['x_step_nm'] * 1./ scale_factor
    newimage['y_step_nm'] = image['y_step_nm'] * 1./ scale_factor


    log.info('resampled (zoom factor %3.2f) image size %dx%d' % (scale_factor, newdata.shape[1], newdata.shape[0]))

    return newimage




def img_planesub(image):
    """Accepts an image and performs a best-fit plane subtraction"""

    data = image['data']

    # Make an array of x and y values for each pixel
    xvals = np.arange(data.shape[0])
    yvals = np.arange(data.shape[1])

    xs, ys = np.meshgrid(xvals, yvals)
    x = xs.ravel()
    y = ys.ravel()
    z = data.ravel()

    A = np.column_stack([x, y, np.ones_like(x)])
    abc, residuals, rank, s = np.linalg.lstsq(A, z)

    a1 = abc[0] / image.x_step
    b1 = abc[1] / image.y_step
    c1 = abc[2] - (a1 * image.x_orig) - (b1 * image.y_orig)

    # Create a grid containing this fit
    zgrid = np.array( [x[ind]*abc[0] + y[ind]*abc[1] + abc[2] for ind in np.arange(len(x))] )
    zgrid = zgrid.round().astype(long)

    # Subtract fit and shift Z values
    imagefit = z - zgrid + abc[2].round().astype(long)

    newdata = imagefit.reshape((data.shape[0],data.shape[1]))
    image['data'] = newdata

    return image


def img_polysub(image, order=3):
    """Accepts an image and performs a polynomial plane subtraction of order n"""

    import itertools

    def polyfit2d(x, y, z, order=order):
        ncols = (order + 1)**2
        G = np.zeros((x.size, ncols))
        ij = itertools.product(range(order+1), range(order+1))
        for k, (i,j) in enumerate(ij):
            G[:,k] = x**i * y**j
        m, _, _, _ = np.linalg.lstsq(G, z)
        return m

    def polyval2d(x, y, m):
        order = int(np.sqrt(len(m))) - 1
        ij = itertools.product(range(order+1), range(order+1))
        z = np.zeros_like(x)
        for a, (i,j) in zip(m, ij):
            z += a * x**i * y**j
        return z

    data = image['data']

    xvals = np.arange(data.shape[0], dtype=np.float64)
    yvals = np.arange(data.shape[1], dtype=np.float64)
    xs, ys = np.meshgrid(xvals, yvals)

    x = xs.ravel()
    y = ys.ravel()
    z = data.ravel()

    m = polyfit2d(x,y,z,order=order)
    z = polyval2d(x, y, m)
    newdata = data - z.reshape((data.shape[0],data.shape[1]))
    image['data'] = (newdata - newdata.min())
    return image


def show_grid(images, cols=2, figsize=None, **kwargs):
    """Plots multiple images in a grid with a given column width"""

    import matplotlib.gridspec as gridspec

    if type(images) == pd.Series:
        images = pd.DataFrame(columns=images.to_dict().keys()).append(images)
    elif (type(images) == str) or (type(images) == list):
        images = load_images(data=True).query('scan_file==@images')
    elif type(images) != pd.DataFrame:
        log.error('input data type must be a scan name, list of names or images dataframe')
        return None

    num_images = len(images)
    rows, rem = divmod(num_images, cols)
    if rem != 0: rows += 1
    gs = gridspec.GridSpec(rows, cols)

    if figsize is not None:
        fig = plt.figure(figsize=figsize)
    else:
        fig = plt.figure()

    grid = 0
    axes = []

    for idx, image in images.iterrows():

        axes.append(plt.subplot(gs[grid]))
        show(image, fig=fig, ax=axes[grid], **kwargs)
        grid += 1

    # fig.tight_layout()

    plt.show()

    return


def show_facets(facet_select=None, query=None, savefig=None, cols=3,  show_stripes=True, zoom_out=False,
        title=True, exp_only=False, fontsize=8):
    """Show the coverage to date of all facets if facet_select=None
    or show a single, or list of facets"""

    import matplotlib.gridspec as gridspec

    images = load_images(data=True)
    if query is not None:
        images = images.query(query)

    if len(images)==0:
        log.error('no images matching query')
        return None

    facets = sorted(images.target.unique())

    if facet_select is not None:
        if type(facet_select)!=list:
            facet_select = [facet_select]
        facets = list(set(facets) & set(facet_select))

    if exp_only:
        start = pd.Timestamp('2014-01-01')
        exposures = load_exposures()
        exp_facets = sorted(exposures.query('start>@start').target.unique())
        facets = sorted(set(facets) & set(exp_facets))

    num_facets = len(facets)
    if num_facets==0:
        log.warning('no images found matching facets')
        return None

    images = images[ images.target.isin(facets) ]

    fig = plt.figure(figsize=(15,15))

    rows = num_facets / cols
    if num_facets % cols != 0:
        rows += 1

    gs = gridspec.GridSpec(rows, cols)

    for idx, facet in enumerate(facets):
        ax = plt.subplot(gs[idx])
        plt.setp(ax.get_xticklabels(), rotation=45)

        # def show_loc(images, facet=None, segment=None, tip=None, show_stripes=True, zoom_out=False):
        show_loc( images[ images.target==facet ], figure=fig, axis=ax, labels=False, title=title, font=fontsize,
            show_stripes=show_stripes, zoom_out=zoom_out)

    fig.tight_layout(h_pad=5.0)

    def onresize(event):
        plt.tight_layout()

    cid = fig.canvas.mpl_connect('resize_event', onresize)

    if savefig is None:
        plt.show()
    else:
        plt.savefig(savefig)
        plt.close()

    return


def show_tips(savefig=None, info=False):
    """Shows the latest tip image for all 16 tips"""

    import matplotlib.gridspec as gridspec

    gs = gridspec.GridSpec(4,4)
    fig = plt.figure(figsize=(15,15))

    images = load_images(data=True)

    if images is None:
        return None

    tip_images = images[ (images.target==3) & (images.channel=='ZS') & (~images.aborted) ].copy()
    tip_images.sort_values(by='start_time', inplace=True)

    for tip_num in range(1,17):
        ax = plt.subplot(gs[tip_num-1])
        ax.set_aspect('equal')
        tip_image = tip_images[ tip_images.tip_num==tip_num ]
        if len(tip_image)==0:
            ax.get_yaxis().set_visible(False)
            ax.get_xaxis().set_visible(False)
            ax.set_title('Tip %d' % tip_num, fontsize=12)
            continue
        else:

            tip_image = tip_image.iloc[-1].squeeze()

            if info:
                # get number of images since this tip image
                num_images = len(images[ (images.tip_num==tip_num) & (images.channel=='ZS') &
                    (images.start_time > tip_image.end_time) ])
                title = 'Tip %d (%s)\n%d images following' % (tip_num, tip_image.start_time, num_images)
            else:
                title='Tip %d' % tip_num

            show(tip_image, title=title, fig=fig, ax=ax, cbar=False, planesub='plane')

    fig.tight_layout(h_pad=2.5)

    if savefig is not None:
        fig.savefig(savefig)
        plt.close()
    else:
        plt.show()

    return


def show(images, units='real', planesub='poly', title=True, cbar=True, fig=None, ax=None,
            shade=False, show_fscans=False, show=True, rect=None, fontsize=10, figsize=None):
    """Accepts one or more images from get_images() and plots them in 2D.

    units= can be 'real', 'dac' or 'pix'
    planesub= can be 'plane', 'poly'  or 'median'
    title= can be True (defaults to scan name), None (no title), or a string
    cbar= display colour bar (True/False)
    fig, ax = existing figure and axis instances can be passed
    shade=True will add surface illumination and shading
    show_fscans=True marks lines where frequency re-tuning has occured (note - slow!)"""

    import matplotlib.cm as cm
    from matplotlib.colors import LightSource

    if type(images) == pd.Series:
        images = pd.DataFrame(columns=images.to_dict().keys()).append(images)
    elif (type(images) == str) or (type(images) == list):
        images = load_images(data=True).query('scan_file==@images')
    elif type(images) != pd.DataFrame:
        log.error('input data type must be a scan name, list of names or images dataframe')
        return None

    if 'data' not in images.columns:
        log.error('image data not found - be sure to run tm.get_images with info_only=False')
        return False

    unit_types = ['real', 'dac', 'pix']
    units = units.lower()
    if units not in unit_types:
        log.error('unit type %s invalid' % units.lower() + " must be one of " + ", ".join(unit_types))
        return None

    if rect is not None:
        units = 'dac'
        if len(units)==4:
            log.error('if rect= is given, 4 values must be passed (X,Y) origin and (width,height) as a tuple')
            return None

    if planesub is not None:
        planetypes = ['plane', 'poly']
        planesub = planesub.lower()
        if planesub not in planetypes:
            log.error('planesub type %s invalid' % planesub.lower() + " must be one of " + ", ".join(planetypes))
            return None

    if show_fscans: # for now just use pix, then identifying the line of the fscan is straightforward
        units = 'pix'

    ls = LightSource(azdeg=0,altdeg=65)
    cmap = cm.afmhot

    for idx, image in images.iterrows():

        if planesub=='plane':
            image = img_planesub(image)
        elif planesub=='poly':
            image = img_polysub(image)

        chan_idx = common.data_channels.index(image.channel)
        unit = common.units[chan_idx]
        target = common.seg_to_facet(image.wheel_pos)

        if fig is None:
            if figsize is None:
                figure = plt.figure()
            else:
                figure = plt.figure(figsize=figsize)
        else:
            figure = fig

        if ax is None:
            axis = figure.add_subplot(1,1,1)
        else:
            axis = ax

        data = image['data']

        if units == 'real':
            data = (data - data.min()) * common.cal_factors[chan_idx]
            plot1 = axis.imshow(data, origin='upper', interpolation='nearest', extent=[0,image.xlen_um,image.ylen_um,0], cmap=cmap)
            axis.set_xlabel('X (microns)', fontsize=fontsize)
            axis.set_ylabel('Y (microns)', fontsize=fontsize)

        elif units == 'dac':
            xstart = image.x_orig
            xstop = image.x_orig + image.xsteps * image.x_step
            ystart = image.y_orig
            ystop = image.y_orig + image.ysteps * image.y_step
            plot1 = axis.imshow(data, origin='upper', interpolation='nearest', extent=[xstart,xstop,ystop,ystart], cmap=cmap)

            # Check for hybrid mode and set aspect ratio correctly if using DAC units
            if image.y_closed and ~image.x_closed:
                axis.set_adjustable('box')
                axis.set_aspect(common.xycal['closed']/common.xycal['open'])
            plt.setp(axis.get_xticklabels(), rotation=45)

            if rect is not None:
                from matplotlib.patches import Rectangle
                box = Rectangle((rect[0], rect[1]), rect[2], rect[3], fill=False, linewidth=2, edgecolor='w')
                axis.add_patch(box)

        elif units == 'pix':
            plot1 = axis.imshow(data, origin='upper', interpolation='nearest', cmap=cmap)
            axis.set_xlabel('X (pixels)', fontsize=fontsize)
            axis.set_ylabel('Y (pixels)', fontsize=fontsize)
            data = image['data']

        plt.setp(axis.get_xticklabels(), rotation=45, fontsize=fontsize)
        plt.setp(axis.get_yticklabels(),fontsize=fontsize)

        axis.grid(True)

        if shade:
            data = ls.shade(data,cmap)

        if (not shade) and (cbar):
            cbar = figure.colorbar(plot1, ax=ax) # Now plot using a colourbar
            if units=='real':
                cbar.set_label(unit, rotation=90)

        if title is not None:
            if title==True:
                axis.set_title(image.scan_file, fontsize=fontsize)
            else:
                axis.set_title(title, fontsize=fontsize)

        # If show_fscans is True, index the appropriate TLM file, get the OBTs of frequency scans
        # between the start and end times, find the line number of these times in HK and then
        # plot as arrows on the axis...
        if show_fscans:

            telem = tm(image.filename)
            fscans = telem.get_freq_scans(info_only=True)
            fscans = fscans[ (fscans.start_time > image.start_time) & (fscans.start_time < image.end_time) ]

            hk2 = telem.pkts[ (telem.pkts.type==3) & (telem.pkts.subtype==25) & (telem.pkts.apid==1076) & (telem.pkts.sid==2) ]

            if len(fscans) > 0:
                log.info('%i frequency re-tunes occured during the image at OBT %s' % (len(fscans), image.start_time))
                obts = fscans.start_time.values
                for obt in obts:

                    frame = hk2[hk2.obt>obt].index
                    if len(frame)==0:
                        log.warning('no HK2 frame found after frequency scan at %s' % image.start_time)
                        continue
                    else:
                        frame = frame[0]

                    line = telem.get_param('NMDA0165', frame=frame)[1]

                    # arrow sig: arrow(x, y, dx, dy, **kwargs)

                    if image.fast_dir=='X':
                        xstart, xstop = axis.get_xlim()
                        arrow_delta = (xstop-xstart)*0.025
                        axis.arrow(xstop+arrow_delta*2, line, -arrow_delta, 0, head_width=4, head_length=abs(arrow_delta), fc='k', ec='k', clip_on=False)
                    else:
                        ystart, ystop = axis.get_ylim()
                        arrow_delta = (ystart-ystop)*0.025
                        axis.arrow(line, ystop-arrow_delta*2, 0, arrow_delta, head_width=4, head_length=abs(arrow_delta) , fc='k', ec='k', clip_on=False)
            else:
                log.info('no frequency re-tunes occured during the image at OBT %s' % image.start_time)

    if fig is None and show:
        plt.show()

    return figure, axis


def locate_scans(images):
    """Accepts a list of scans returned by get_images() and calculates the origin of each scan,
    relative to the wheel/target centre. These are added to the images DataFrame and returned."""

    x_orig_um = []; y_orig_um = []

    if type(images)==pd.core.series.Series:
        images = images.to_frame().T

    for idx, scan in images.iterrows():

        # Calculate offset from centre of table in X and Y
        x_centre =  common.centre_closed if scan.x_closed else common.centre_open
        x_cal = common.xycal['closed'] if scan.x_closed else common.xycal['open']
        x_cal /= 1000.
        x_offset = (scan.x_orig - x_centre) * x_cal

        y_centre = common.centre_closed if scan.y_closed else common.centre_open
        y_cal = common.xycal['closed'] if scan.y_closed else common.xycal['open']
        y_cal /= 1000.
        y_offset = (y_centre - scan.y_orig) * y_cal

        # Take the cantilever and linear stage position into account for X position
        left = ( (scan.lin_pos-common.lin_centre_pos_fm[int(scan.tip_num)-1]) / common.linearcal ) + x_offset
        left -= common.tip_offset[scan.tip_num-1]
        x_orig_um.append(left)

        # Y position in this stripe is simple related to the offset from the Y origin
        centre_seg = common.facet_to_seg(scan.target)

        if (scan.target==0) and (scan.wheel_pos>1016):
            seg_offset = -1*(seg_offset+1024)
        else:
            seg_offset = scan.wheel_pos - centre_seg

        y_offset += common.seg_off_to_pos(seg_offset)
        y_orig_um.append(y_offset)

    images['x_orig_um'] = x_orig_um
    images['y_orig_um'] = y_orig_um

    return images


def show_loc(images, facet=None, segment=None, tip=None, show_stripes=True, zoom_out=False,
    figure=None, axis=None, labels=True, title=True, font=None, interactive=False, show_seg=False):
    """Plot the location of a series of images"""

    import subprocess

    if font is None:
        font = mpl.rcParams['font.size']

    if (type(images)==str) or (type(images)==list):
        images = load_images(data=False).query('scan_file==@images')
    elif type(images)==pd.core.series.Series:
        images = images.to_frame().T

    # filter out dummy scans
    images = images[ images.dummy==False ]

    if tip is not None:
        images = images[ images.tip_num == tip ]

    if segment is not None:
        images = images[ images.wheel_pos==segment ]

    if facet is not None:
        images = images[ images.target==facet ]

    if len(images.target.unique())>1:
        log.error('more than one target specified - filter images or use keyword segment=')
        return None

    if len(images)==0:
        log.error('no matching images for the given facet and segment')
        return None

    from matplotlib.patches import Rectangle

    def onpick(event):
        ind = event.ind.tolist()

        if event.mouseevent.dblclick: # open the Gwyddion file
            gwyfiles = [scan_names[idx] for idx in ind]
            gwyfiles = [os.path.join(common.gwy_path, scan+'.gwy') for scan in gwyfiles]
            gwycall = ['gwyddion', '--remote-new']
            subprocess.Popen(gwycall + gwyfiles)
        else:
            print('Select scans: %s' % "\n".join([scan_names[idx] for idx in ind]))

    if figure is None:
        fig = plt.figure()
    else:
        fig = figure

    if axis is None:
        ax = fig.add_subplot(1,1,1)
    else:
        ax = axis

    ax.invert_yaxis()

    # Plot open and closed loop scans in different colours
    opencolor='red'
    closedcolor='blue'

    if title:
        title = ''
        if len(images.tip_num.unique())==1:
            title = title.join('Tip %i, ' % images.tip_num.unique()[0] )

        title += ('Target %i (%s)' % (images.target.unique()[0], images.target_type.unique()[0] ))
        ax.set_title(title, fontsize=font)

    if labels:
        ax.set_xlabel('Offset from wheel centre (microns)', fontsize=font)
        ax.set_ylabel('Offset from segment centre (microns)', fontsize=font)

    # Use now a patch collection
    import matplotlib.collections as mcoll
    patches = []
    scan_names = []

    for idx, scan in images.iterrows():
        edgecolor=closedcolor if scan.y_closed else opencolor
        rect = Rectangle((scan.x_orig_um, scan.y_orig_um), scan.xlen_um, -scan.ylen_um, fill=False, linewidth=1, edgecolor=edgecolor, picker=True)
        patches.append(rect)
        scan_names.append(scan.scan_file)

    collection = mcoll.PatchCollection(patches,  match_original=True, picker=True)
    ax.add_collection(collection)

    if show_stripes:
        for seg_off in range(-7,8):
            offset = common.seg_off_to_pos(seg_off)
            ax.axhspan(offset-50., offset+50., facecolor='g', alpha=0.2)

    if show_seg:
        tform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        centre_seg = images.target.unique()[0] * 16
        for seg_off in range(-7,8):
            offset = common.seg_off_to_pos(seg_off)
            ax.text(1.0, offset, str(centre_seg + seg_off), fontsize='medium', color='k',
                transform=tform, ha='right', va='center', clip_on=True)

    # Make sure we plpot fix a fixed aspect ratio!
    ax.autoscale(enable=True)
    ax.set_aspect('equal')

    ax.tick_params(axis='x', labelsize=font)
    ax.tick_params(axis='y', labelsize=font)

    if zoom_out:
        ax.set_xlim(-700.,700.)
        ax.set_ylim(-1400., 1400.)
    else:
        ax.set_xlim(images.x_orig_um.min()-50.,images.x_orig_um.max()+images[images.x_orig_um==images.x_orig_um.max()].xlen_um.max()+50.)
        ax.set_ylim(images.y_orig_um.min()-50.,images.y_orig_um.max()+images[images.y_orig_um==images.y_orig_um.max()].ylen_um.max()+50.)

    if interactive:
        cid = fig.canvas.mpl_connect('pick_event', onpick)

    if figure is None:
        plt.show()

    return fig


class tc:

    def __init__(self, files=None, directory='.', recursive=False, apid=1084, simple=True):

            self.pkts = None

            if files is not None:
                self.get_pkts(files=files, directory=directory, recursive=recursive, simple=simple, apid=apid)


    def simple_locate_pkts(self, filename):
        """Scans a TC file and returns OBT, packet type and sub-type, APID and offset.
        Minimal checking is performed; use get_pkts() for a more robust method of TC extraction."""

        f = open(filename, 'rb')
        tc = bytearray(os.path.getsize(filename))
        log.debug('file %s has length %d bytes' % (filename, (os.path.getsize(filename))))

        f.readinto(tc)

        num_pkts = 0
        offsets = []
        offset = 0

        while(offset < len(tc)):

            offsets.append(offset+dds_tc_header_len)

            # "pkt_id pkt_seq pkt_len ver_check pkt_type pkt_subtype pad
            dds_tc_header = dds_tc_header_names(*struct.unpack_from(dds_tc_header_fmt, tc, offset))
            tc_pkt_header = tc_pkt_header_names(*struct.unpack_from(tc_pkt_header_fmt, tc, offset + dds_tc_header_len))
            log.debug('packet seq %d, type %d, subtype %d' %
                (tc_pkt_header.pkt_seq, tc_pkt_header.pkt_type, tc_pkt_header.pkt_subtype))

            offset = offset + (dds_tc_header_len + dds_tc_header.pkt_len)
            num_pkts += 1

        print 'INFO: %i packets read from file %s' % (num_pkts, filename)

        return np.array(offsets)


    def locate_pkts(self, filename, apid=None):
        """Reads a file or TC packets into a numpy array and searches for the unique bit-pattern
        defining the packet header. Returns an index of offsets (in bytes) to the start of each
        packet. If APID is set, this APID is located - otherwise the MIDAS APID is assumed."""

        f = open(filename, 'rb')

        # We read in an array of words, but file can be non-integer number of words long
        # so check and ignore last byte if so (doesn't matter for packet finding)
        data = f.read()
        if len(data) % 2 != 0: data=data[0:-1]
        pkts = np.fromstring(data, dtype=('>H'), count=-1)

        num_words = len(pkts)

        if apid is None:

            pkt_offsets = np.where( \

                # Packet ID (word 0) version number (bits 0-2) = 0b000
                # Packet ID (word 0) type (bit 3) = 1 (telecommand)
                # Packet ID (word 0) data field header (bit 4) = 1
                (pkts[0:num_words-4] & 0xF800 == 0x1800) & \

                # Packet sequence control (word 1) bits 0-2 segmentation flag = 0b11
                (pkts[1:num_words-3] & 0xC000 == 0xC000) & \

                # Data field header, word 3 / byte 0 - PUS version, checksum, ack (3,1,4 bits)
                # PUS - no pattern
                # checksum (1 bit) - always 1b
                # ack = bits 5/6 always 0
                (pkts[3:num_words-1] & 0x1600 == 0x1000 ) & \

                # Data field header, word 4
                # byte 1 = pad (0)
                (pkts[4:num_words] & 0xFF == 0 ))[0]

        else:

            pkt_offsets = np.where( \

                # Packet ID (word 0) version number (bits 0-2) = 0b000
                # Packet ID (word 0) type (bit 3) = 1 (telecommand)
                # Packet ID (word 0) data field header (bit 4) = 1
                (pkts[0:num_words-4] & 0xF800 == 0x1800) & \

                # Packet ID (word 0)
                # bits 5-15 (APID)
                #     bits 5-11: PID = 67 for MIDAS
                #     bits 12-15: PCAT = always 12 (dec)
                (pkts[0:num_words-4] & 0x7FF == apid ) & \

                # Packet sequence control (word 1) bits 0-2 segmentation flag = 0b11
                (pkts[1:num_words-3] & 0xC000 == 0xC000) & \

                # Data field header, word 3 / byte 0 - PUS version, checksum, ack (3,1,4 bits)
                # PUS - no pattern
                # checksum (1 bit) - always 1b
                # ack = bits 5/6 always 0
                (pkts[3:num_words-1] & 0x1600 == 0x1000 ) & \

                # Data field header, word 4
                # byte 1 = pad (0)
                (pkts[4:num_words] & 0xFF == 0 ))[0]


        log.info('file %s contains %i matching TC packets') % (filename, len(pkt_offsets))

        return pkt_offsets*2



    def get_pkts(self, files, directory='.', recursive=False, append=False, apid=None, simple=True):
        """Read telecommand packets from the DDS history"""

        if recursive:
            selectfiles = common.locate(files, directory)
            filelist = [file for file in selectfiles]
        elif type(files)==list:
            filelist = files
        else:
            import glob
            filelist = glob.glob(os.path.join(directory,files))

        filelist.sort()

        pkt_list = []

        ver_flags = {
            0: 'N/A',
            1: 'IDLE',
            2: 'PENDING',
            4: 'PASSED',
            8: 'FAILED',
            16: 'UNVERIFIED',
            32: 'UNKNOWN',
            64: 'TIMED-OUT',
            128: 'SSC ONLY',
            256: 'ASSUMED',
            512: 'BIT 10?',
            1024: 'BIT 11?',
            2048: 'BIT 12?',
            4096: 'BIT 13?',
            8192: 'BIT 14?' }

        for fl in filelist:

            log.info('indexing TC history file %s' % fl)

            if simple:
                offsets = self.simple_locate_pkts(fl)
            else:
                offsets = self.locate_pkts(fl, apid=apid)

            if len(offsets) == 0:
                log.error('no packets found in file %s' % fl)
                continue

            f = open(fl, 'rb')
            tc = f.read()
            f.close()

            for idx, offset in enumerate(offsets):

                pkt = {}

                dds_tc_header = dds_tc_header_names(*struct.unpack_from(dds_tc_header_fmt, tc, offset-(dds_tc_header_len)))
                pkt_header = tc_pkt_header_names(*struct.unpack_from(tc_pkt_header_fmt, tc, offset))

                if pkt_header.pkt_len == 0:
                    continue

                pkt['offset'] = offset
                pkt['filename'] = os.path.abspath(fl)

                # DDS: uplink_t_s uplink_t_us exec_t_s exec_t_us cuv cav_vc0 cev_vc0 cav_vc1 cev_vc1 pkt_len
                pkt['t_uplink'] = sun_mjt_epoch + timedelta(seconds=dds_tc_header.uplink_t_s) + timedelta(microseconds=dds_tc_header.uplink_t_us)
                pkt['t_exec'] =   sun_mjt_epoch + timedelta(seconds=dds_tc_header.exec_t_s) + timedelta(microseconds=dds_tc_header.exec_t_us)
                pkt['cuv'] = ver_flags[dds_tc_header.cuv]
                pkt['cav_vc0'] = ver_flags[dds_tc_header.cav_vc0]
                pkt['cev_vc0'] = ver_flags[dds_tc_header.cev_vc0]
                pkt['cav_vc1'] = ver_flags[dds_tc_header.cav_vc1]
                pkt['cev_vc1'] = ver_flags[dds_tc_header.cev_vc1]
                pkt['dds_pkt_len'] = dds_tc_header.pkt_len
                pkt['name'] = dds_tc_header.tc
                pkt['sequence'] = dds_tc_header.sq

                # TC: pkt_id pkt_seq pkt_len ver_check pkt_type pkt_subtype pad
                pkt['apid'] = pkt_header.pkt_id & 0x7FF
                pkt['pkt_seq'] = pkt_header.pkt_seq
                pkt['length'] = pkt_header.pkt_len
                pkt['type'] = pkt_header.pkt_type
                pkt['subtype'] = pkt_header.pkt_subtype

                if pkt_header.pkt_len==0:
                    continue

                pkt_list.append(pkt)

        if len(pkt_list)==0:
            log.error('no valid frames found in TC file(s)')
            return False

        # turn the packet list into a DataFrame, interpret the obt columns as a datetime
        tcpkts = pd.DataFrame(pkt_list)

        tcpkts.name = tcpkts.name.apply( lambda name : name.replace("\x00", " ").strip() )
        tcpkts.sequence = tcpkts.sequence.apply( lambda sq : sq.replace("\x00", " ").strip() )
        pkts = pd.merge(tcpkts, ccf,  how='inner', on=['type', 'subtype', 'apid', 'name'])

        if apid is not None:
            pkts = pkts[pkts.apid==apid]

        pkts.cuv = pkts.cuv.astype("category", categories=ver_flags.values(), ordered=False)
        pkts.cav_vc0 = pkts.cav_vc0.astype("category", categories=ver_flags.values(), ordered=False)
        pkts.cev_vc0 = pkts.cev_vc0.astype("category", categories=ver_flags.values(), ordered=False)
        pkts.cav_vc1 = pkts.cav_vc1.astype("category", categories=ver_flags.values(), ordered=False)
        pkts.cev_vc1 = pkts.cev_vc1.astype("category", categories=ver_flags.values(), ordered=False)


        self.pkts = pkts.sort_values(by='t_exec')

        return



class tm:

    def __init__(self, files=None, directory='.', recursive=False, pkts=False, apid=False, dedupe=False, simple=True, dds_header=True, sftp=False, model='FM'):

        self.sftp = False
        self.pkts = None

        self.model = model.upper()
        if self.model=='FM':
            self.lin_centre_pos = common.lin_centre_pos_fm
            self.tcorr = read_timecorr(os.path.join(common.tlm_path,'TLM__MD_TIMECORR.DAT'))
            if files is not None:
                self.get_pkts(files=files, directory=directory, recursive=recursive, apid=apid, dedupe=dedupe, simple=simple, dds_header=dds_header, sftp=sftp)
        elif self.model=='FS':
            self.lin_centre_pos = common.lin_centre_pos_fs
            if files is not None:
                self.get_pkts(files=files, directory=directory, recursive=recursive, apid=apid, dedupe=True, simple=False, dds_header=False, sftp=sftp)

    def save_index(self, filename):
        """Saves the packet index to a file, to save time re-indexing TLM files. This can be reloaded with
        load_index()"""

        self.pkts.to_pickle(filename)


    def load_index(self, filename, append=False):
        """Reads a packet index saved with save_index()"""

        if append and self.pkts is not None:
            self.pkts = self.pkts.append(pd.read_pickle(filename))

        else:
            self.pkts = pd.read_pickle(filename)

        self.pkts.sort_values(by='obt', inplace=True)
        log.info('packet index restored with %i packets' % (len(self.pkts)))


    def query_index(self, filename=os.path.join(common.tlm_path, 'tlm_packet_index.hd5'),
        start=None, end=None, stp=None, what='all', sourcepath=common.tlm_path, rows=None, tsync=True):
        """Restores a TLM packet index from filename. The entire file is read if no other options are given, otherwise
        filters can be applied:

        start=, end= accept any sane string date/time format
        stp= accepts an integer STP number
        what= can be 'all', 'hk', 'events' or 'science' and filters packets by APID
        tsync=True will only return values from packets which are time synchronised.
"""

        what_types = {
            'hk': 1076,
            'events': 1079,
            'science': 1084 }

        what = what.lower()

        if (what != 'all') and (what not in what_types.keys()):
            log.warning('what= must be set to all, hk, events or science. Defaulting to all.')
            what = 'all'

        if type(start)==str:
            start = pd.Timestamp(start)

        if type(end)==str:
            end = pd.Timestamp(end)

        if stp is not None and type(stp)!=list:
            stp = [stp]

        table = 'pkts'
        store = pd.HDFStore(filename, 'r')

        if rows is not None:
            self.pkts = store.select(table, where=list(rows))
            if sourcepath is not None:
                self.pkts.filename = self.pkts.filename.apply( lambda f: os.path.join(sourcepath, os.path.basename(f)) )
            store.close()
            return

        if start is None and end is None and stp is None and (what=='all'):
            self.pkts = store.get(table)
        else:

            # In principle one can simply query the HDF5 file with AND'd statements, however the memory
            # usage kills the server. So we will find the matching rows per query, AND them and then
            # select the matching rows at the end...

            # selected = set(store.select_column(table,'index'))
            selected = set(np.arange(store.get_storer('pkts').nrows))

            if what!='all':
                col = store.select_column(table,'apid')
                selected = selected.intersection( col[ col==what_types[what] ].index )

            if stp is not None:
                col = store.select_column(table,'filename')
                col = col.apply( lambda f: int(os.path.basename(f)[14:17]) )
                selected = selected.intersection(col[ col.isin(stp) ].index)

            if (start is not None) or (end is not None):

                col = store.select_column(table,'obt')

                # get the OBT column and use this to find the indices to slice the entire table
                if start is None:
                    start = col.min()

                if end is None:
                    end = col.max()

                selected = selected.intersection(col[ (col>start) & (col<end) ].index)

            if len(selected)==0:
                log.warning('no packets match the criteria!')
                return

            self.pkts = store.select(table, where=list(selected))

        if tsync:
            self.pkts = self.pkts[ self.pkts.tsync ]

        self.pkts.sort_values(by='obt', inplace=True)

        if sourcepath is not None:
            self.pkts.filename = self.pkts.filename.apply( lambda f: os.path.join(sourcepath, os.path.basename(f)) )

        # reset the index to a monotonic increasing function, needed for selecting frames later
        self.pkts.index=range(len(self.pkts))
        log.info('packet index restored with %i packets' % (len(self.pkts)))

        store.close()

        return


    def get_dv(self, apid=None, output=False):
        """Calculates and returns the datavolume (in bytes) for all APIDs or a single, specified, APID"""

        if apid is not None:
            if apid not in midas_apids:
                log.error('APID is not valid for MIDAS - defaulting to all APIDs')
                apids = midas_apids
            else:
                if type(apid)!=list:
                    apids = [apid]
        else:
            apids = midas_apids

        dv = {}

        for apid in apids:
            volume = (self.pkts[self.pkts.apid==apid].length+6).sum()
            dv.update( { apid: volume } )
            if output:
                log.info('APID %d generated %3.2f MB' % (apid, volume/1024./1024.))

        return dv


    def get_pkts(self, files, directory='.', recursive=False, append=False, apid=False, simple=True, dedupe=False, dds_header=True, sftp=False):
        """Locates packets by searching for unique bit patterns in the headers, then unpacks
        the packet primary header to extract packet type, subtype, APID, and length. Validates
        length against expected offsets and checks for DDS header.

        A single filename, a list of filenames, or a wildcard expression can be provided. Recursive searches
        can be enabled with recursive=True.

        By default all MIDAS APIDs are located, but by setting the keyword apid= any
        APID, including s/c, can be located.

        If the append= keyword is True, packets will be appended to the current list.

        If sftp=True a connection will be made to the MIMAS sftp server and files opened
        there instead of the local filesystem."""

        if recursive:
            selectfiles = common.locate(files, directory)
            filelist = [file for file in selectfiles]
        elif type(files)==list:
            filelist = files
        else:
            import glob
            filelist = glob.glob(os.path.join(directory,files))

        filelist.sort()

        pkt_list = []

        if sftp:
            import dds_utils
            sftp = dds_utils.sftp()
            sftp.open()

        for fl in filelist:

            log.info('indexing TLM file %s' % fl)

            # Scan the file for unique bit pattern to identify the start of the TM packets
            if simple:
                offsets = self.simple_locate_pkts(fl, dds_header=dds_header)
            else:
                offsets = self.locate_pkts(fl, apid=apid, sftp=sftp)

                if len(offsets) == 0: # no packets found
                    log.warning('no packets found for given APID, trying fallback method...')
                    offsets = self.simple_locate_pkts(fl)

            if len(offsets) == 0:
                log.error('no packets found in file %s' % fl)
                continue

            # Re-open the file and read into a bytearray
            if sftp:
                f = sftp.sftp.open(fl, 'rb')
            else:
                f = open(fl, 'rb')
            tm = f.read()
            f.close()

            # using numpy instead of pandas here for speed
            picfile = os.path.join(common.s2k_path, 'pic.dat')
            pic = np.loadtxt(picfile,dtype=np.int32)

            # Using the byte offsets, unpack each packet header
            for idx,offset in enumerate(offsets):

                pkt = {}
                pkt_header = pkt_header_names(*struct.unpack_from(pkt_header_fmt,tm,offset))

                if pkt_header.pkt_len == 0:
                    continue

                # Check for out-of-sync packets - MIDAS telemetry packets are not time synchronised when the MSB
                # of the 32 bit coarse time (= seconds since reference date) is set to "1".
                pkt['tsync'] = not bool(pkt_header.obt_sec >> 31)

                obt_s = pkt_header.obt_sec + pkt_header.obt_frac / 2.**16
                if self.model=='FM' and pkt['tsync']:
                    obt = self.correlate_time(obt_s)
                else:
                    obt = obt_epoch + timedelta(seconds=long(obt_s))

                pkt['offset'] = offset
                pkt['type'] = pkt_header.pkt_type
                pkt['length'] = pkt_header.pkt_len
                pkt['subtype'] = pkt_header.pkt_subtype
                pkt['apid'] = pkt_header.pkt_id & 0x7FF
                pkt['seq'] = pkt_header.pkt_seq & 0x3FFF
                pkt['obt'] = obt
                pkt['filename'] = os.path.abspath(fl)

                if pkt_header.pkt_len==0:
                    continue

                # 07/07/2014: now using the pic.dat table to look up the location of p1val
                # 29/08/2014: FIXME - just profiled the code and this causes a MASSIVE slowdown
                #
                pkt_pic = pic[ np.logical_and(pic[:,0]==pkt_header.pkt_type,pic[:,1]==pkt_header.pkt_subtype) ]

                if len(pkt_pic)==0:
                    log.warning('packet type (%i,%i) not found in the PIC' % (pkt_header.pkt_type,pkt_header.pkt_subtype))
                    log.debug('OBT: %s, APID: %i, SEQ: %i' % (pkt['obt'], pkt['apid'], pkt['seq']))
                    continue
                if pkt_pic[0][2] ==-1:
                    if pkt['apid'] in midas_apids:
                        pkt['midsid'], = struct.unpack_from('>H',tm,offset+pkt_header_size)
                        pkt['sid'] = 0
                    else:
                        pkt['sid'] = 0
                else:
                    if pkt_pic[0][3]==8:
                        pkt['sid'], = struct.unpack_from('>B',tm,offset+pkt_pic[0][2])
                    elif pkt_pic[0][3]==16:
                        pkt['sid'], = struct.unpack_from('>H',tm,offset+pkt_pic[0][2])
                    elif  pkt_pic[0][3]==32:
                        pkt['sid'], = struct.unpack_from('>I',tm,offset+pkt_pic[0][2])
                    else:
                        log.error('p1val location cannot be identified for this packet')
                        return False

                pkt_list.append(pkt)

        # turn the packet list into a DataFrame, interpret the obt columns as a datetime
        tlm = pd.DataFrame(pkt_list)

        if len(tlm)==0:
            log.error('no valid frames found in TM file(s)')
            return False

        tlm.obt = pd.to_datetime(tlm.obt)

        # Due to a database change, the event with SID 42777 has changed from 1 to 2
        # Easiest fix is to change the substype here
        tlm.subtype.loc[ tlm[ (tlm.sid==42777) & (tlm.subtype==1) ].index ] = 2
        # Also EV_APP_ERROR with SID 42765
        tlm.subtype.loc[ tlm[ (tlm.sid==42765) & (tlm.subtype==1) ].index ] = 2
        # And EV_APP_CONTACT with SID 42764
        tlm.subtype.loc[ tlm[ (tlm.sid==42764) & (tlm.subtype==1) ].index ] = 2

        # Merge with the packet list, adding spid and description, then sort by OBT
        tlm = pd.merge(tlm,pid,how='left').sort_values(by='obt')
        tlm.spid = tlm.spid.astype(np.int64)

        # Deal with the fact that MIDAS uses private SIDs that are not in the RMIB
        if 'midsid' in tlm.columns:
            idx = tlm[tlm.midsid.notnull()].index
            tlm.sid.loc[idx]=tlm.midsid[idx]
            tlm.drop('midsid', axis=1, inplace=True)

        tlm.sid = tlm.sid.astype(np.int64)

        num_pkts = len(tlm)

        if dedupe:
            # Remove duplicates (packets with the same OBT, APID and SID)
            tlm = tlm.drop_duplicates(subset=('obt','apid','sid'), keep='last')
            log.info('%i duplicate packets removed' % ( (num_pkts)-len(tlm) ) )

        log.info('%i packets read' % (len(tlm)) )

        # Ignore invalid frames with zero lengths
        tlm = tlm[tlm.length>0]

        # Check if the DDS header is present
        tlm = self.dds_check(tlm)

        # Add the DDS time to the packet list
        tlm = self.dds_time(tlm)

        # append to an existing packet list if already existing
        if append and type(self.pkts) == pd.core.frame.DataFrame:
            tlm = tlm.append(self.pkts)
            tlm.sort_values(by='obt', inplace=True, axis=0)

        # Reset the index
        tlm = tlm.reset_index(drop=True)

        if sftp: self.sftp = sftp

        # tlm['description'] = tlm['description'].astype('category')

        self.pkts = tlm

        return


    def simple_locate_pkts(self, filename, dds_header=True):
        """Scans a TM file and returns OBT, packet type and sub-type, APID and offset.
        Minimal checking is performed, and the dds_header flag must be set correctly.

        use get_pkts() for a more robust method of TM extraction."""

        f = open(filename, 'rb')
        tm = bytearray(os.path.getsize(filename))
        f.readinto(tm)

        num_pkts = 0
        offsets = []
        offset = dds_header_len if dds_header else 0

        while(offset < len(tm)):

            pkt = {}
            pkt_header = pkt_header_names(*struct.unpack_from(pkt_header_fmt,tm,offset))
            offsets.append(offset)
            offset = offset + (pkt_header.pkt_len + 7 + dds_header_len) if dds_header else offset + (pkt_header.pkt_len + 7)
            num_pkts += 1

        # print '%i packets read from file %s' % (num_pkts, filename)

        return np.array(offsets)


    def locate_pkts(self, filename, apid=False, sftp=False):
        """Reads a file or TM packets into a numpy array and searches for the unique bit-pattern
        defining the packet header. Returns an index of offsets (in bytes) to the start of each
        packet. If APID is set, this APID is located - otherwise all MIDAS APIDs are allowed."""

        if sftp:
            f = sftp.sftp.open(filename, 'rb')
        else:
            f = open(filename, 'rb')

        # We read in an array of words, but file can be non-integer number of words long
        # so check and ignore last byte if so (doesn't matter for packet finding)
        data = f.read()
        if len(data) % 2 != 0: data=data[0:-1]
        pkts = np.fromstring(data, dtype=('>H'), count=-1)

        num_words = len(pkts)

        if apid:

            # if type(apid) != list: apid = [apid]

            pkt_offsets = np.where( \

                # Packet ID (word 0) version number (bits 0-2) = 0b100
                # Packet ID (word 0) type (bit 3) = 0 (telememetry)
                # Packet ID (word 0) secondary header flag (bit 4) = 0 - not sure why, though?
                (pkts[0:num_words-6] & 0xF800 == 0x0800) & \

                # Packet ID (word 0) bits 11-15 = APID
                (pkts[0:num_words-6] & 0x07FF == apid ) & \
                # (pkts[0:num_words-6] & 0x07FF in apid) & \

                # Packet sequence control (word 1) bits 0-2 segmentation flag = 0b11
                (pkts[1:num_words-5] & 0xC000 == 0xC000) & \

                # Data field header, word 6 - PUS version, checksum, spare (3,1,4 bits)
                # PUS - either 0 or 2 (0b000 or 0b010) - so bit pattern 0b0x0 is fixed
                # checksum (1 bit) - not used, set to zero
                # spare (4 bits) - not used, set to zero
                (pkts[6:num_words] & 0xBF00 == 0000 ) )[0]

        else:

            pkt_offsets = np.where( \

                # Packet ID (word 0) version number (bits 0-2) = 0b100
                # Packet ID (word 0) type (bit 3) = 0 (telememetry)
                # Packet ID (word 0) secondary header flag (bit 4) = 0 - not sure why, though?
                (pkts[0:num_words-6] & 0xF800 == 0x0800) & \

                # Packet ID (word 0) bits 11-15 = APID
                (  (pkts[0:num_words-6] & 0x07FF == 84L) | \
                   (pkts[0:num_words-6] & 0x07FF == 196L) | \
                   (pkts[0:num_words-6] & 0x07FF == 1073L) | \
                   (pkts[0:num_words-6] & 0x07FF == 1076L) | \
                   (pkts[0:num_words-6] & 0x07FF == 1079L) | \
                   (pkts[0:num_words-6] & 0x07FF == 1081L) | \
                   (pkts[0:num_words-6] & 0x07FF == 1083L) | \
                   (pkts[0:num_words-6] & 0x07FF == 1084L) ) & \

                # Packet sequence control (word 1) bits 0-2 segmentation flag = 0b11
                (pkts[1:num_words-5] & 0xC000 == 0xC000) & \

                (pkts[2:num_words-4] < 4106L ) & \

                # Data field header, word 6 - PUS version, checksum, spare (3,1,4 bits)
                # PUS - either 0 or 2 (0b000 or 0b010) - so bit pattern 0b0x0 is fixed
                # checksum (1 bit) - not used, set to zero
                # spare (4 bits) - not used, set to zero
                (pkts[6:num_words] & 0xBF00 == 0000 ) )[0]

        # log.info('File %s contains %i matching TM packets') % (filename, len(pkt_offsets))

        return pkt_offsets*2


    def read_pkts(self, pkts, pkt_type=False, subtype=False, apid=False, sid=False, sftp=False):
        """Accepts a packet list and reads the binary packet source data into a packet array.
        Note that for efficiency only one packet type (length) is allowed. Such a packet
        list can be passed, or keywords can be set to filter by type, subtype, APID and SID.

        If byte_offset, bit_offset and width are given, only this particular parameter will
        be returned (still raw data)"""

        import math

        pkt_data = []

        if pkts is None: pkts = self.pkts

        # Filter according to keywords
        pkts = pkts[pkts.type==pkt_type] if pkt_type else pkts
        pkts = pkts[pkts.subtype==subtype] if subtype else pkts
        pkts = pkts[pkts.apid==apid] if apid else pkts
        pkts = pkts[pkts.sid==sid] if sid else pkts

        num_pkts = len(pkts)
        if num_pkts == 0:
            log.warning('no packets to read matching requested filters')
            return pkt_data

        if len(pkts.apid.unique()) + len(pkts.sid.unique()) != 2:
            log.warning('multiple packets types cannot be read together, filter first')
            return pkt_data

        filenames = pkts.filename.unique()

        for filename in filenames:
            if sftp:
                f = sftp.sftp.open(filename, 'rb')
            else:
                f = open(filename, 'rb')
            log.debug('opening binary file %s' % (filename))
            for index,pkt in pkts[pkts.filename==filename].sort_values(by='offset').iterrows():
                f.seek(pkt.offset+pkt_header_size)
                bindata = f.read(pkt.length+1)
                pkt_data.append(bindata)
            f.close()

        pkts['data'] = pkt_data

        return pkts


    def write_pkts(self, outputfile, outputpath='.', sftp=False, strip_dds=True):
        """Reads all TM source packets (skipping the DDS header
        if strip_dds=True) and writes all packets to a single file"""

        pkt_data = []

        if self.pkts is None:
            log.warning('tm object not initialised, no packet list present')
            return None

        num_pkts = len(self.pkts)
        if num_pkts == 0:
            log.warning('no packets found')
            return None

        pkts = self.pkts.sort_values(by='obt')

        for idx,packet in pkts.iterrows():

            if sftp:
                f = sftp.sftp.open(packet.filename, 'rb')
            else:
                f = open(packet.filename, 'rb')

            if ~strip_dds and packet.dds_header:
                f.seek(packet.offset-dds_header_len)
            else:
                f.seek(packet.offset)

            # Note the 10 here - the packet header is 48-bits (6 octets) and then the
            # data field header is 10 octets - but this is already part of the packet
            # data field, and counted in the packet length.
            # So the total length is: 6 + length + 1.
            if ~strip_dds and packet.dds_header:
                pkt_data.append(f.read(dds_header_len+6+packet.length+1))
            else:
                pkt_data.append(f.read(6+packet.length+1))
            f.close()

        pkt_data = "".join(pkt_data)

        f = open(os.path.join(outputpath,outputfile), 'wb')
        f.write(pkt_data)

        return


    def get_exposures3(self, html=False):
        """Retrieves shutter open/close events and calculates exposure duration. The exposed
        target is also tabulated. If the html= keyword is set, an HTML table is produced."""

        # Grab the shutter open and close events from the packet index
        shutter_open = self.pkts.obt[ (self.pkts.apid==1079) & (self.pkts.type==5) & (self.pkts.sid==42553) ]
        shutter_close = self.pkts.obt[ (self.pkts.apid==1079) & (self.pkts.type==5) & (self.pkts.sid==42554) ]

        shutter_open.sort()
        shutter_close.sort()

        # Check for cases where the shutter is open at the start or end of the period
        if shutter_close.iloc[0] < shutter_open.iloc[0]:
            # shutter open at start - ignore this exposure
            shutter_close = shutter_close.iloc[1:]

        if shutter_open.iloc[-1] > shutter_close.iloc[-1]:
            # shutter open at end - ignore this exposure
            shutter_open = shutter_open.iloc[0:-1]

        if len(shutter_open) == 0 or len(shutter_close)==0:
            log.warning('not enough shutter events to calculate exposure')
            return False

        if len(shutter_open) < len(shutter_close):
            log.warning('shutter open/close count mismatch')
            return False

        # Use the event OBT to calculate start/stop of exposure
        exposures = pd.DataFrame( zip(shutter_open, shutter_close), columns=['start', 'end'])
        exposures['duration'] = exposures.end - exposures.start

        # To find the target number we need to go to HK
        hk2 = self.pkts.query('type==3 & subtype==25 & apid==1076 & sid==2')
        hk2.sort_values(by='obt', inplace=True, axis=0)

        # Loop through the OBTs and find the next HK2 packet, then extract parameters
        target = []

        for idx, exp in exposures.iterrows():
            frame = hk2[hk2.obt>exp.start].index
            if len(frame)==0:
                log.warning('no HK2 frame found after shutter open at %s' % exp.start)
                continue
            else:
                frame = frame[0]

            target.append(common.opposite_facet(common.seg_to_facet(self.get_param('NMDA0196', frame=frame)[1]))) # WheSegmentNum

        exposures['target'] = target

        if html:

            timeformatter = lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S')
            deltaformatter = lambda x: strfdelta(timedelta(seconds=x/np.timedelta64(1, 's')), "{days} days, {hours:02d}:{minutes:02d}:{seconds:02d}")
            exposure_html = exposures.to_html(classes='alt_table', na_rep='',index=False, formatters={ 'start': timeformatter, 'end': timeformatter, 'duration': deltaformatter } )
            css_write(html=exposure_html, filename=html)

        return exposures


    def get_exposures2(self, html=False):
        """Retrieves shutter open/close events and calculates exposure duration. The exposed
        target is also tabulated. If the html= keyword is set, an HTML table is produced."""

        # Grab the shutter open and close events from the packet index
        shutter_open = self.pkts.obt[ (self.pkts.apid==1079) & (self.pkts.type==5) & (self.pkts.sid==42553) ]
        shutter_close = self.pkts.obt[ (self.pkts.apid==1079) & (self.pkts.type==5) & (self.pkts.sid==42554) ]

        # Now we have a list of "something happened" times for the shutter, check if open/closed or no change
        shutter_ev = shutter_open.append(shutter_close)
        shutter_ev.sort(inplace=True)
        shutter_ev = pd.DataFrame(shutter_ev, columns=['obt'])

        # To find the target number we need to go to HK
        hk1 = self.pkts[ (self.pkts.type==3) & (self.pkts.subtype==25) & (self.pkts.apid==1076) & (self.pkts.sid==1) ]
        hk1.sort('obt', inplace=True, axis=0)
        hk2 = self.pkts[ (self.pkts.type==3) & (self.pkts.subtype==25) & (self.pkts.apid==1076) & (self.pkts.sid==2) ]
        hk2.sort('obt', inplace=True, axis=0)

        event = []
        target = []

        # Take the HK frame before and after the shutter event (+/- one min)
        for idx, ev in shutter_ev.iterrows():
            frame_before = hk1[hk1.obt<(ev.obt-timedelta(minutes=5))].index
            frame_after = hk1[hk1.obt>(ev.obt+timedelta(minutes=5))].index
            if len(frame_before)==0 or len(frame_after)==0:
                log.warning('no HK2 frame found after shutter open at %s' % ev.obt)
                continue
            else:
                before = frame_before[-2]
                after = frame_after[1]

                open_before = self.get_param('NMDD013B', frame=before)[1]
                closed_before = self.get_param('NMDD013C', frame=before)[1]
                open_after = self.get_param('NMDD013B', frame=after)[1]
                closed_after = self.get_param('NMDD013C', frame=after)[1]

                log.debug('before (o/c) %s/%s, after (o/c) %s/%s' % (open_before, closed_before, open_after, closed_after))

                if (open_before=='OFF') & (closed_before=='ON') & (open_after=='ON') & (closed_after=='OFF'): # open
                    event.append('OPEN')
                elif (open_before=='ON') & (closed_before=='OFF') & (open_after=='OFF') & (closed_after=='ON'): # close
                    event.append('CLOSE')
                else:
                    log.warning('unknown shutter event transition at %s, skipping...' % ev.obt)
                    event.append('UNKNOWN')

            # Get segment exposed from HK2
            hk2_frame = hk2[hk2.obt>ev.obt].index[0]
            target.append(common.opposite_facet(common.seg_to_facet(self.get_param('NMDA0196', frame=hk2_frame)[1]))) # WheSegmentNum

        # Add events to the table and remove unknowns
        shutter_ev['event'] = event
        shutter_ev['target'] = target
        shutter_ev = shutter_ev.drop(shutter_ev[shutter_ev.event=='UNKNOWN'].index)

        shutter_open = shutter_ev[shutter_ev.event=='OPEN']
        shutter_close = shutter_ev[shutter_ev.event=='CLOSE']

        # Check for cases where the shutter is open at the start or end of the period
        if shutter_close.obt.iloc[0] < shutter_open.obt.iloc[0]:
            # shutter open at start - ignore this exposure
            shutter_close = shutter_close.iloc[1:]

        if shutter_open.obt.iloc[-1] > shutter_close.obt.iloc[-1]:
            # shutter open at end - ignore this exposure
            shutter_open = shutter_open.iloc[0:-1]

        if len(shutter_open) == 0 or len(shutter_close)==0:
            log.warning('not enough shutter events to calculate exposure')
            return False

        if len(shutter_open) < len(shutter_close):
            log.warning('shutter open/close count mismatch')
            return False

        # Use the event OBT to calculate start/stop of exposure
        exposures = pd.DataFrame( zip(shutter_open.obt.tolist(), shutter_close.obt.tolist(), shutter_close.target.tolist()), columns=['start','end','target'])
        exposures['duration'] = exposures.end - exposures.start

        if html:

            timeformatter = lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S')
            deltaformatter = lambda x: strfdelta(timedelta(seconds=x/np.timedelta64(1, 's')), "{days} days, {hours:02d}:{minutes:02d}:{seconds:02d}")
            exposure_html = exposures.to_html(classes='alt_table', na_rep='',index=False, formatters={ 'start': timeformatter, 'end': timeformatter, 'duration': deltaformatter } )
            css_write(html=exposure_html, filename=html)

        return exposures


    def get_exposures(self, html=False):
        """Retrieves shutter open/close events and calculates exposure duration. The exposed
        target is also tabulated. If the html= keyword is set, an HTML table is produced."""

        # Grab the shutter open and close events from the packet index
        open_times = self.pkts.obt[ (self.pkts.apid==1079) & (self.pkts.type==5) & (self.pkts.sid==42553) ].tolist()
        close_times = self.pkts.obt[ (self.pkts.apid==1079) & (self.pkts.type==5) & (self.pkts.sid==42554) ].tolist()

        # Create two series indexed by these times for open/closed
        open_state = [True] * len(open_times)
        close_state = [False] * len(close_times)

        shut_open = pd.Series(open_state, index=open_times)
        shut_close = pd.Series(close_state, index=close_times)

        # Merge the series
        state = shut_open.append(shut_close)

        # Initialise the shutter at launch time to be closed
        # state = state.append(pd.Series(False, index=[pd.Timestamp('2 March 2004, 07:17 UTC')]))

        # Order by time, and remove and successive identical events
        state = state.sort_index()
        state = state.loc[state.shift(1) != state]

        if state.iloc[-1]: # shutter open at end
            state = state.iloc[:-1]

        start_times = pd.Series(state[state].index.asobject, name='start')
        end_times = pd.Series(state[~state].index.asobject, name='end')

        exposures = pd.concat([start_times, end_times], axis=1)
        exposures['duration'] = exposures.end - exposures.start

        # To find the target number we need to go to HK
        hk2 = self.pkts[ (self.pkts.type==3) & (self.pkts.subtype==25) & (self.pkts.apid==1076) & (self.pkts.sid==2) ]
        hk2.sort_values(by='obt', inplace=True, axis=0)

        # Loop through the OBTs and find the next HK2 packet, then extract parameters
        target = []

        for idx, exp in exposures.iterrows():
            frame = hk2[hk2.obt>exp.start].index
            if len(frame)==0:
                log.warning('no HK2 frame found after shutter open at %s' % exp.start)
                continue
            else:
                frame = frame[0]

            target.append(common.opposite_facet(common.seg_to_facet(self.get_param('NMDA0196', frame=frame)[1]))) # WheSegmentNum

        exposures['target'] = target

        if html:

            timeformatter = lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S')
            deltaformatter = lambda x: strfdelta(timedelta(seconds=x/np.timedelta64(1, 's')), "{days} days, {hours:02d}:{minutes:02d}:{seconds:02d}")
            exposure_html = exposures.to_html(classes='alt_table', na_rep='',index=False, formatters={ 'start': timeformatter, 'end': timeformatter, 'duration': deltaformatter } )
            css_write(html=exposure_html, filename=html)

        return exposures


    def get_events(self, ignore_giada=True, info=False, html=False, verbose=True):
        """List MIDAS events. If the keyword info= is set to True,
        additional information will be decoded from events that support it.

        If the html= keyword is set to a filename, an html version of the
        event list is produced."""

        # We don't even need to read the packets here, just filter the list for APID 1079 and
        # show the packet descriptions!

        # We can add a bit more info to some events, e.g. TC acceptance and failure
        # SID 42501 = TC accept, 42701 = TC reject

        pkts = self.pkts[ (self.pkts.apid==1079) & (self.pkts.type==5) ].copy()

        event_severity = ['PROGRESS','ANOMALY - NO ACTION','ANOMALY - GROUND ACTION','ANOMALY - ONBOARD ACTION']

        if len(pkts) == 0:
            log.warning('no events found')
            return None

        pkts['doy'] = pkts.obt.apply( lambda x: x.dayofyear )
        pkts.rename(columns={'description':'event'}, inplace=True)

        if info:

            # TC accept/reject events return the listed telecommand
            tcs = self.get_tc_status()

            if tcs is None:
                info = False
            else:

                pkts = pd.merge(pkts,tcs,left_index=True, right_index=True, how='outer')
                pkts.rename(columns={'telecommand':'information'}, inplace=True)

                # The following events return elapsed time
                # TODO: not sure if these events are actually used?!
                # SID 42511: EvScanStarted
                # SID 42700: EvTimer
                time_evt = pkts[ (pkts.apid==1079) & ( (pkts.sid==42511) | (pkts.sid==42700) ) ]
                if len(time_evt) > 0:
                    time_evt = self.read_pkts(time_evt) # add binary packet data
                    indices = []; times = []
                    for idx,packet in time_evt.iterrows():
                        times.append(struct.unpack(">2I", packet['data'][4:12]))
                        indices.append(idx)

                if ignore_giada: pkts = pkts[pkts.information!='GiadaDustFluxDist']

        pkts['severity'] = [event_severity[pkt.subtype-1] for idx,pkt in pkts.iterrows()]

        timeformatter = lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S')

        if info:
            events = pkts[['obt', 'doy', 'sid', 'event','information', 'severity']]
        else:
            events = pkts[['obt', 'doy', 'sid', 'event', 'severity']]

        if html:
            event_html = events.to_html(classes='alt_table', na_rep='',index=False, \
                formatters={ 'obt': timeformatter } )
            css_write(event_html, html)


        uhoh = events[events.severity!='PROGRESS']
        if (len(uhoh)>0) and verbose:
            log.warning('non-nominal events detected:\n')
            common.printtable(uhoh[['obt','event']])

        return events



    def dds_check(self, pkts):
        """Simply adds a column to the pkts dataframe which lists if a
        DDS header is present (per packet)"""

        filenames = pkts.filename.unique()
        pkts['dds_header'] = pd.Series()
        for filename in filenames:
            pkts_per_file = pkts[pkts.filename==filename].sort_values(by='offset')
            header_gap = (pkts_per_file.offset.shift(-1)-(pkts_per_file.offset+pkts_per_file.length+7)).shift(1)
            header_gap.iloc[0] = pkts_per_file.offset.iloc[0]
            pkts.dds_header.loc[header_gap.index] = header_gap.apply( lambda x: True if x==18 else False )

        return pkts


    def dds_time(self, pkts):
        """Unpacks the DDS header and returns the UTC time (including time correlation, but
        does not account for leap seconds!) and adds this to the pkts dataframe"""

        import struct
        from datetime import timedelta

        pkts['dds_time'] = pd.Series()
        dds_pkts = pkts[pkts.dds_header] # filter by frames container a DDS header
        filenames = dds_pkts.filename.unique()

        for filename in filenames:
            with open(filename,'rb') as f:
                tm = f.read()

            offsets = dds_pkts[dds_pkts.filename==filename].offset

            dds_time = []

            for idx,offset in offsets.iteritems():
                dds_header = dds_header_names(*struct.unpack_from(dds_header_fmt,tm,offset-dds_header_len))
                # 19/06/14 - datetime does NOT take leap seconds etc. into account
                delta_t = timedelta(seconds=dds_header.scet1, microseconds=dds_header.scet2)
                dds_time.append(dds_obt_epoch + delta_t)
                if dds_header.time_qual != 0:
                    log.debug('bad DDS time stamp at %s' % (dds_obt_epoch + delta_t))

            pkts.dds_time = pkts.dds_time.astype(pd.Timestamp)
            pkts.dds_time.loc[offsets.index] = dds_time

        pkts.dds_time = pd.to_datetime(pkts.dds_time, errors='raise')

        return pkts



    def get_param(self, param, frame=False, cal=True, start=False, end=False, value_after=None, tsync=True):
        """Accepts a parameter ID. Uses the SCOS-2K tables to search packets for this
        parameter, reads the relevant binary data and applies necessary calibration.

        Returns a pd.Series object of calibrated parameter values indexed by OBT.

        start= and false= can be used to constrain the time period of returned values.

        tsync=True will only return values from packets which are time synchronised.

        If frame= is provided, a specific HK frame is used to return a single value."""

        import struct
        from bitstring import Bits

        pkts = self.pkts.loc[frame] if frame else self.pkts

        if type(pkts)==pd.core.series.Series:
            pkts = pkts.to_frame().T

        # Extract parameter information
        param = pcf[pcf.param_name==param].squeeze()

        pkt_info = pid.merge(plf[plf.param_name==param.param_name])
        log.debug('parameter %s found in the following %i packet(s): ' % \
            (param.param_name, len(pkt_info)) + ", ".join(pkt_info.description.values))

        if value_after is not None:
            start = False
            end = False
            if type(value_after)==str:
                value_after = pd.Timestamp(value_after)

        if type(start)==str:
            start = pd.Timestamp(start)

        if type(end)==str:
            end = pd.Timestamp(end)

        if start and end:
            if start > end:
                log.error('start time must be before end!')
                return None

        # filter packet list to those containing the parameter in question
        pkts = pkts[ (pkts.type.isin(pkt_info.type.tolist())) & (pkts.subtype.isin(pkt_info.subtype.tolist())) \
            & (pkts.apid.isin(pkt_info.apid.tolist())) & (pkts.sid.isin(pkt_info.sid.tolist())) ]

        if start:
            pkts = pkts[pkts.obt>start]

        if end:
            pkts = pkts[pkts.obt<end]

        if value_after:
            pkts = pkts[pkts.obt>value_after].iloc[0]
            pkts = pd.DataFrame(columns=pkts.to_dict().keys()).append(pkts)

        # If requested, filter out packets that are not time-sync'd
        if tsync and self.model!='FS':
            pkts = pkts[ pkts.tsync ]

        num_params = len(pkts)

        if num_params == 0:
            log.error('no matching TM packets found')
            return None

        # group packets by their SPID
        spids = pkts[pkts.spid.notnull()].spid.unique().astype(int)
        log.debug('processing packets with SPIDs: %s' % spids)

        obt = []
        pkt_data = []

        # A given parmeter can occur in several packets, with different SPIDs, and at
        # different positions within those packets
        for spid in spids:

            pkts_per_spid = pkts[pkts.spid==spid]
            byte_offset = pkt_info[pkt_info.spid==spid].byte_offset.squeeze()
            bit_offset = pkt_info[pkt_info.spid==spid].bit_offset.squeeze()

            # packets this this SPID can be in one or more source files
            filenames = pkts_per_spid.filename.unique()

            for filename in filenames:
                f = Bits(filename=filename)
                # loop through each packet and grab the correct number of bits
                for index,pkt in pkts[pkts.filename==filename].sort_values(by='offset').iterrows():
                    pkt_data.append( f[((int(pkt.offset)+byte_offset)*8+bit_offset):((int(pkt.offset)+byte_offset)*8+bit_offset)+int(param.width)] )
                    obt.append( pkt.obt )

        # Now we simply need to convert the binary representation to the appropriate value
        # This is mostly done by calling the relevant bitstring function, but sometimes extra
        # conversions are needed.

        # PTC = 1: 1-bit boolean (extract byte, bit offset is 0-7)
        if param.ptc == 1:
            if param.pfc == 0: # boolean
                values = np.array(pkt_data) # nothing to do, 1-bit bitstrings are already bools
            else:
                log.error('PTC 1 parameters must have PFC 0')

        # PTC = 2: enumerated parameter, max 32-bits
        # PFC 1-32, corresponds to width
        # PTC = 3: unsigned integer, 4-bits (PFC = 0) to 16 (12), 24 (13) and 32 (14)
        elif (param.ptc == 2) or (param.ptc==3):
            # treat enumerated as unsigned integer
            values = np.array([ val.uint for val in pkt_data ])

        # PTC = 4: signed integer (same bit lengths as PTC 3)
        # Signed integers: b = 1 byte, h = 2 bytes, l = 4 bytes, q = 8 bytes
        elif param.ptc == 4:
            values = np.array([ val.int for val in pkt_data ])

        # PTC = 5: real number, single/double precision IEEE (PFC 1/2, 32/64 bit) or MIL std (PFC 3/4)
        elif param.ptc == 5:
            if (param.pfc == 1) or (param.pfc == 2):
                values = np.array([ val.float for val in pkt_data ])

            elif param.pfc == 3:
                values = np.array([ val.uint for val in pkt_data ])
                mil2float = np.vectorize(mil1750a_to_float)
                values = mil2float(values)

        # PTC = 8: ASCII string of PFC octets length (PFC > 0)

        else:
            log.error('PTC/PFC type not supported')

        # Now calibrate the engineering value to a real unit
        if cal:
            real_val = calibrate(param.param_name, values)
        else:
            real_val = values

        if frame:
            return obt[0], real_val[0]
        else:
            return pd.Series(real_val, index=obt)


    def hk_as_image(self, param, image):
        """Accepts an image and uses the corresponding start and end times to sample a
        given HK value and display matching points as a 2D image"""

        import archive

        start = image.start_time
        end = image.end_time

        values = self.get_param(param, start=start, end=end).values

        # NMDA0165 = ScnLineCnt
        # NMDA0170 = ScnMainCnt

        line = self.get_param('NMDA0165', start=start, end=end).values
        main = self.get_param('NMDA0170', start=start, end=end).values

        hk_img = np.zeros( (image.xsteps, image.ysteps) )

        log.debug('image (%ix%i), main scan min/max: %i/%i, line scan min/max: %i/%i' % \
            (image.xsteps, image.ysteps, main.min(), main.max(), line.min(), line.max()))

        if (len(line) != len(main)) or (len(line) != len(values)):
            log.error('mismatch of values, lines and main scan values - note that HK1 parameters not supported')
            return

        for (idx, coord) in enumerate(zip(line,main)):
            hk_img[coord[0],coord[1]] = values[idx]

        plt.imshow(hk_img,  interpolation='nearest')

        return hk_img




    def plot_params(self, param_names, start=False, end=False, label_events=None, symbol=True, line='', axis=None):
        """Plot a TM parameter vs OBT. Requires a packet list and parameter
        name. start= and end= can be set to a string or DateTime to limit the returned
        data.

        label_events= can be one of scan, fscan, lines or all and will label events."""

        import matplotlib.dates as md
        import matplotlib.transforms as transforms
        from dateutil import parser
        import math
        from cycler import cycler

        if type(param_names) != list: param_names = [param_names]
        units = pcf[pcf.param_name.isin(param_names)].unit.unique()

        if type(start)==str:
            start = parser.parse(start)

        if type(end)==str:
            end = parser.parse(end)

        if start and end:
            if start > end:
                log.error('start time must be before end time!')
                return None

        if len(units) > 2:
            log.error('cannot plot parameters with more than 2 different units on the same figure')
            return False

        if 'S' in pcf[pcf.param_name.isin(param_names)].cal_cat.tolist():
            log.error('one or more parameters is non-numeric, cannot plot!')
            return False

        if symbol:
            marker_left = 'x'
            marker_right = '+'
        else:
            marker_left = None
            marker_right = None

        if axis is None:
            plot_fig, ax_left = plt.subplots()
        else:
            ax_left = axis
            plot_fig = ax_left.figure

        ax_left.set_prop_cycle(cycler('color', ['c', 'm', 'y', 'k']))

        if len(units)==2:
            ax_right = ax_left.twinx()
            ax_right.set_prop_cycle(cycler('color', ['r', 'g', 'b', 'y']))

        lines = []

        for param_name in param_names:

            data = self.get_param(param_name, start=start, end=end, tsync=True)

            if data is None:
                log.warning('could not get data for parameter %s, skipping' % (param_name))
                continue

            param = pcf[pcf.param_name==param_name].squeeze()

            if param.unit==units[0] or (pd.isnull(param.unit) & (pd.isnull(units[0]))): # plot on left axis
                lines.append(ax_left.plot( data.index, data, label=param.description, linestyle=line, marker=marker_left )[0])
                ax_left.set_ylabel( "%s" % (param.unit))
            else:
                # ax_left._get_lines.color_cycle.next()
                ax_left._get_lines.prop_cycler.next()['color']
                lines.append(ax_right.plot( data.index, data, label=param.description, linestyle='-.', marker=marker_right )[0])
                ax_right.set_ylabel( "%s" % (param.unit))

        if len(lines)==0:
            log.error('no data to plot')
            return False

        if not start: start = data.index[0]
        if not end: end = data.index[-1]

        labs = [l.get_label() for l in lines]
        leg = ax_left.legend(lines, labs, loc=0, fancybox=True)
        leg.get_frame().set_alpha(0.7)
        ax_left.grid(True)

        if len(param_names)==1: ax_left.set_title( "%s (%s)" % (param.description, param_name)  )
        ax_left.set_xlabel('On-board time')

        plot_fig.autofmt_xdate()
        xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
        ax_left.xaxis.set_major_formatter(xfmt)

        ax_left.yaxis.get_major_formatter().set_useOffset(False)

        if label_events is not None:

            events = self.get_events(ignore_giada=True, verbose=False)

            if events is not None:

                if label_events=='scan':
                    # 42656 - EvFullScanStarted
                    # 42756 - EvFullScanAborted
                    # 42513 - EvScanFinished
                    # 42713 - EvScanAborted
                    scan_events = [42656, 42756, 42513, 42713]
                    events = events[events.sid.isin(scan_events)]

                elif label_events=='wheel':
                    # 42904 - EvSegSearchTimeout
                    # 42592 - EvSegmentFound
                    # 42591 - EvSearchForRefPulse
                    scan_events = [42904, 42592, 42591]
                    events = events[events.sid.isin(scan_events)]

                elif label_events=='fscan':
                    # 42641 - EvFScanStarted
                    # 42645 - EvAutoFScanFinshed
                    scan_events = [42641, 42645]
                    events = events[events.sid.isin(scan_events)]

                elif label_events=='lines':
                    # 42655 - EvLineScanStarted
                    # 42611 - EvLineScanFinished
                    scan_events = [42655, 42611]
                    events = events[events.sid.isin(scan_events)]

                elif label_events=='approach':
                    # 42662 - EvApproachStarted
                    # 42664 - EvApproachFinished
                    # 42762 - EvApproachAborted
                    # 42764 - EvAppContact
                    # 42765 - EvAppError
                    # 42766 - EvApproachStuck
                    # 42906 - EvApproachTimeout
                    # 42916 - EvMoveAbortedApp
                    # 42917 - EvAppAbortedLin
                    # 42623	- EvSurfaceFound
                    # 42665	- EvZpiezoFineAdj
                    app_events = [42662, 42664, 42762, 42764, 42765, 42766, 42906, 42916, 42917, 42623, 42665 ]
                    events = events[events.sid.isin(app_events)]
                elif label_events=='all':
                    # Ignore events that don't bring much diagonstic info or flood the plot
                    ignore_sids = [ \
                        42501, # TC accept
                        42701, # TC reject
                        42611, # EvLineScanFinished
                        42512, # EvScanProgress
                        42642, # EvFScanCycleStarted
                        42643, # EvFScanCycleFinished
                        42699] # KernelHello
                    events = events[-events.sid.isin(ignore_sids)]
                else:
                    log.warning('invalid event label, ignoring')
                    events = pd.DataFrame()

                # Fixing mark/MIDAS#2 - filter events to given time range first
                events = events[ (events.obt>start) & (events.obt<end) ]

                for idx, event in events.iterrows():
                    ax_left.axvline(event.obt,color='r')
                    trans = transforms.blended_transform_factory(ax_left.transData, ax_left.transAxes)
                    ax_left.text(event.obt,0.9,event.event,rotation=90, transform=trans, clip_on=True)
            else:
                log.warning('labelling of events requested, but TM files contains no event data')

        plot_fig.tight_layout()

        if axis is None:
            plt.show()
            return plot_fig
        else:
            return


    def plot_temps(self, start=False, end=False, cssc=False, label='scan', **kwargs):
        """Plot the temperatures from all MIDAS sensors for the given TM packages.
        If start= and end= are set to date/time strings the data will be limited
        to those times, otherwise all data are plotted."""

        # temp_params = self.search_params('temp').param_name.tolist()
        temp_params = [
            'NMDA0003', # baseplate temp
            'NMDA0005', # converter temp
            'NMDA0008', # dust inlet temp
            'NMDA0004', ] # preamlifier temp

        if cssc:
            temp_params.extend(['NMDA0006','NMDA0007']) # CSSC X ref temp, CSSC Y ref temp

        fig = self.plot_params(temp_params, label_events=label, start=start, end=end, **kwargs)

        return fig


    def plot_hv(self, start=False, end=False, label='scan', **kwargs):
        """Plot piezo high voltages (HV)s"""

        hv_params = ['NMDA0110', 'NMDA0111', 'NMDA0115']

        fig = self.plot_params(hv_params, label_events=label, start=start, end=end, **kwargs)

        return fig


    def plot_app(self, start=False, end=False, label='approach', **kwargs):
        """Plot approach related parameters"""

        # NMDA0105  app LVDT signal
        # NMDA0115  Z piezo HV mon
        # NMDA0114  Z piezo position

        app_params = ['NMDA0105', 'NMDA0115', 'NMDA0114']

        fig = self.plot_params(app_params, start=start, end=end, label_events=label, **kwargs)

        return fig


    def plot_volts(self, cssc=False, start=False, end=False, label='scan', **kwargs):
        """Plot the voltages from all MIDAS lines for the given TM packages.
        If start= and end= are set to date/time strings the data will be limited
        to those times, otherwise all data are plotted."""

        volt_params = self.search_params('voltage mon').param_name.tolist()

        # if not cssc:
        #     volt_params.remove( ['NMDA0206', 'NMDA0207', 'NMDA0208', 'NMDA0209'] )

        if len(volt_params)==0:
            print('No MIDAS voltage data found in these packets')
        else:
            self.plot_params(volt_params, label_events=label, start=start, end=end, **kwargs)


    def plot_cantilever(self, start=False, end=False, label='scan', **kwargs):
        """Plot the cantilever AC and DC values"""

        cant_params = ['NMDA0102', 'NMDA0103', 'NMDA0104']

        fig = self.plot_params(cant_params, label_events=label, start=start, end=end, **kwargs)

        return fig


    def list_params(self, spid=None):
        """Lists available parameters in the listed packets. If SPID= is set to an
        SPID packets are filtered, otherwise parameters in all packets are returned."""

        params = pd.DataFrame()
        spids = self.pkts[self.pkts.spid.notnull()].spid.unique().astype(int)

        if spid:
            spids = [spid]
        else:
            get_spids = self.pkts[self.pkts.spid.notnull()].spid.unique().astype(int)

        params = plf[plf.spid.isin(spids)].merge(pcf).drop_duplicates('param_name')\
            [['param_name','description']].sort_values(by='description')

        return params



    def search_params(self, search):
        """Creates a list of available parameters and searches for the search term
        within their description"""

        param_list = self.list_params()

        result = param_list[param_list.description.str.contains(search, case=False)]

        if len(result)==0:
            log.warning('no parameters found containing search term "%s"' % (search))
            return False

        return result


    def get_tc_status(self):
        """Retrieves the MIDAS TC exec/failure events and lists details"""

        import struct

        # Filter events by SIDs for s/w TC acceptance and rejection
        ack = self.pkts[ (self.pkts.apid==1079) & ( (self.pkts.sid==42501) | (self.pkts.sid==42701) ) ]
        if len(ack)==0:
            log.warning('no TC ACK packets found')
            return None

        pkts = pd.DataFrame()

        # add binary packet data
        for sid in ack.sid.unique():
            pkts = pkts.append(self.read_pkts(ack[ack.sid==sid]))

        # load the command characteristics file and filter for MIDAS TCs
        commands = ccf[ccf.apid==1084]

        indices = []; tcs = []

        # loop through the packets and decode the binary data
        for idx,packet in pkts.iterrows():
            tcdata1, tcdata2 = struct.unpack(">2H", packet['data'][10:14])
            srv_type = tcdata1 & 255
            srv_subtype = tcdata2 / 256
            tc = commands.description[ (commands.type==srv_type) & (commands.subtype==srv_subtype) ].tolist()
            if len(tc)==0:
                tcs.append('UNKNOWN TC')
                log.warning('unknown TC retrieving ACK status (type %i, subtype %i) at OBT %s' % (srv_type, srv_subtype, packet.obt))
            elif (len(tc) > 1) & (len(np.unique(tc))==1):
                tcs.append(tc[0])
            elif (len(tc) > 1) & (len(np.unique(tc))>1):
                log.warning('more than one command matching type/subtype %i/%i' % ( srv_type, srv_subtype ))
                tcs.append(tc[0])
            else:
                tcs.append(tc[0])
            indices.append(idx)

        return pd.DataFrame( zip(indices,tcs), columns=['index','telecommand'] ).set_index('index')


    def check_seq(self, apids=False, html=False, verbose=False):
        """Checks the packet sequence counter for gaps. If apids= is set then only
        these APIDs are checked, otherwise all MIDAS APIDs are used.

        If verbose=True a description of lost packets and their OBTs is given."""

        apids = midas_apids if not apids else apids
        if type(apids) != list: apids = [apids]

        missing = {}

        obt_start = []; obt_end = []
        dds_start = []; dds_end = []
        seq_start = []; seq_end = []
        cur_apid = []; count = []

        for apid in apids:

            pkts_apid = self.pkts[self.pkts.apid==apid]
            # df.diff() gives the difference between successive packets
            missing_pkts = pkts_apid.seq.diff()[pkts_apid.seq.diff()>1]-1
            if not missing_pkts.any(): continue
            gap_end = pkts_apid.seq.loc[missing_pkts.index]
            start_rows = [pkts_apid.index.tolist().index(index)-1 for index in missing_pkts.index]
            gap_start = pkts_apid.seq.iloc[start_rows]
            missing[apid] = zip(gap_start.index, gap_end.index)

            log.info('%i missing packets in %i gaps for APID %i' % (missing_pkts.sum(), len(missing_pkts), apid))

            for gap in missing[apid]:
                start = self.pkts.loc[gap[0]]
                end = self.pkts.loc[gap[1]]

                obt_start.append(start.obt)
                obt_end.append(end.obt)
                dds_start.append(start.dds_time if start.dds_header else np.nan)
                dds_end.append(end.dds_time if end.dds_header else np.nan)
                seq_start.append(start.seq)
                seq_end.append(end.seq)
                count.append(end.seq-start.seq-1)
                cur_apid.append(apid)

        if len(count)==0:
            log.info('no missing packets found')
            return False
        else:

            missing_pkts = pd.DataFrame(zip(cur_apid, count, seq_start, seq_end, obt_start, obt_end, dds_start,dds_end),\
                columns=('apid','pkt_count', 'seq_start','seq_end','obt_start','obt_end','dds_start','dds_end'))

            timeformatter = lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S')

            if html:
                missing_html = missing_pkts.to_html(classes='alt_table', na_rep='',index=False, formatters={ 'dds_start': timeformatter, 'dds_end': timeformatter, \
                    'obt_start': timeformatter, 'obt_end': timeformatter } )
                css_write(html=missing_html, filename=html)

            if verbose:

                from prettytable import from_html
                table = from_html(missing_pkts.to_html(na_rep='',index=False, formatters={ 'dds_start': timeformatter, 'dds_end': timeformatter, \
                    'obt_start': timeformatter, 'obt_end': timeformatter } ))
                print(table[0])

            return missing_pkts


    def get_piezo_centre(self, time):
        """Accepts a time (string, datetime or timestamp) typically corresponding to a line
        or image start. Locates the time of the last successful approach and returns the
        corresponding strain gauge value at the end of the approach and fine tuning."""

        hk2 = self.pkts[ (self.pkts.type==3) & (self.pkts.subtype==25) & (self.pkts.apid==1076) & (self.pkts.sid==2) ]
        frame = hk2[hk2.obt>time].index[0]
        return self.get_param('NMDA0348', frame=frame)



    def get_line_scans(self, info_only=False, expand_params=False, ignore_image=True, use_archive=False):
        """Extracts line scans from TM packets.

        info_only=True will return meta-data, but not line scan data.
        expand_params=True will look up additional data in HK but only for non-image lines!
        ignore_image=True will ignore lines forming part of an image scan.
        use_archive=True will use the HK archive to retrieve params if expand_params=True"""

        line_scan_fmt = ">H2Bh4Hh7H"
        line_scan_size = struct.calcsize(line_scan_fmt)
        line_scan_names = collections.namedtuple("line_scan_names", "sid sw_minor sw_major lin_pos \
            wheel_pos tip_num x_orig y_orig step_size num_steps scan_mode_dir line_cnt sw_flags \
            mode_params start_msw start_lsw")

        line_scan_pkts = self.read_pkts(self.pkts, pkt_type=20, subtype=3, apid=1084, sid=132)

        if len(line_scan_pkts)==0:
            log.warning('no line image packets found')
            return False

        linescans = []
        pkt_idx = []
        filename = []

        if expand_params:
            if use_archive:
                import archive
                params = ['NMDA0118','NMDA0142','NMDA0147','NMDA0181','NMDA0188','NMDA0231','NMDA0244','NMDA0245',
                    'NMDA0270','NMDA0271','NMDA0287','NMDA0298','NMDA0306','NMDA0347']
                hk2 = archive.query(params=params, archive_path=common.tlm_path)
            else:
                hk2 = self.pkts[ (self.pkts.type==3) & (self.pkts.subtype==25) & (self.pkts.apid==1076) & (self.pkts.sid==2) ]

        for idx,pkt in line_scan_pkts.iterrows():

            line_type = {}
            line_type['info'] = line_scan_names(*struct.unpack(line_scan_fmt,pkt['data'][0:line_scan_size]))
            num_steps = line_type['info'].num_steps
            in_image = bool(line_type['info'].sw_flags & 1)

            if ignore_image and in_image:
                continue

            if not info_only:
                line_type['data'] = np.array(struct.unpack(">%iH" % (num_steps),pkt['data'][line_scan_size:line_scan_size+num_steps*2]))

            linescans.append(line_type)
            pkt_idx.append(idx)
            filename.append(pkt.filename)

        # pack data into a pandas dataframe
        cols = line_scan_names._fields
        lines = pd.DataFrame([line['info'] for line in linescans],columns=cols,index=pkt_idx)
        lines['filename'] = filename
        if not info_only:
            lines['data'] = [line['data'] for line in linescans]

        # Calibrate raw data
        lines['in_image'] = lines.sw_flags.apply( lambda flag: bool(flag & 1))
        lines['aborted'] = lines.sw_flags.apply( lambda flag: bool(flag >> 3 & 1))
        lines['anti_creep'] = lines.sw_flags.apply( lambda flag: bool(flag >> 1 & 0b11))
        lines['obt'] = line_scan_pkts.obt
        lines['tip_num'] += 1
        lines['sw_ver'] = lines.sw_major.apply( lambda major: '%i%i' % (major >> 4, major & 0x0F))
        lines['sw_ver'] = lines['sw_ver'].str.cat(lines['sw_minor'].values.astype(str)).astype(int)
        lines['lin_pos'] = lines.lin_pos.apply( lambda pos: pos*20./65535.)

        lines['fast_dir'] = lines.apply( lambda row: 'X' if (
            ((row.scan_mode_dir & 2**12)==0 & row.sw_ver<666) or
            ( row.scan_mode_dir & 2**3)==0 & row.sw_ver>=666) else 'Y', axis=1)

        lines['dir'] = lines.apply( lambda row: 'L_H' if (
            ( (row.scan_mode_dir & 2**8)==0 & row.sw_ver<666 ) or
            ( (row.scan_mode_dir & 2**2)==0 & row.sw_ver>=666 )) else 'H_L', axis=1)

        lines['scan_type'] = lines.scan_mode_dir.apply( lambda mode: common.scan_type[ mode & 0b11 ] )

        lines['tip_offset'] = lines.apply( lambda row: (row.lin_pos-self.lin_centre_pos[row.tip_num-1]) / common.linearcal, axis=1 )
        lines['target'] = lines.wheel_pos.apply( lambda seg: common.seg_to_facet(seg) )

        lines['x_closed'] = lines.mode_params.apply( lambda mode: bool(mode >> 4 & 1) )
        lines['y_closed'] = lines.mode_params.apply( lambda mode: bool(mode >> 5 & 1) )
        lines['z_closed'] = lines.mode_params.apply( lambda mode: bool(mode >> 6 & 1) )
        lines['ac_gain'] = lines.mode_params.apply( lambda mode: (mode >>  7) & 0b111 )
        lines['dc_gain'] = lines.mode_params.apply( lambda mode: (mode >> 10) & 0b111 )
        lines['exc_lvl'] = lines.mode_params.apply( lambda mode: (mode >> 13) & 0b111 )
        lines.step_size = abs(lines.step_size)

        lines['z_step'] = np.NaN
        lines['z_step'][lines.sw_ver<665] = np.NaN
        lines['z_step'][lines.sw_ver==665] = lines.start_msw
        lines['z_step'][lines.sw_ver>=666] = lines.scan_mode_dir.apply(  lambda x: x >> 4 )

        # Add start time from event history if possible
        events = self.get_events(verbose=False)
        start_times = events.query('sid==42655').obt

        lines['start_time'] = lines.apply( lambda row: start_times[start_times<row.obt].iloc[-1] if
                len(start_times[start_times<row.obt])>0 else pd.NaT, axis=1)

        lines['start_time'][lines.sw_ver>=666] = lines.apply( lambda row:
            self.correlate_time(row.start_msw*65536+row.start_lsw), axis=1 )

        # correct scans in early OBSW versions prior to re-centering
        # lines['tip_offset'][lines.obsw_ver<645] += (0.243/common.linearcal)

        lines['scan_algo'] = lines.mode_params.apply(lambda mode: common.scan_algo[mode & 0b1111] )

        # Apply corrections due to relative tip offsets
        lines['tip_offset'] = lines.apply( lambda row: (row.tip_offset - common.tip_offset[row.tip_num-1]), axis=1 )


        if ignore_image:
            lines = lines[ (~lines.in_image) & (~lines.anti_creep) ]
            if len(lines)==0:
                log.warning('ignore_image is set, but there are no lines not forming an image')
                return None

        if expand_params:
            cols = ('work_pt_per', 'work_pt', 'set_pt_per', 'set_pt', 'res_amp', 'fadj', 'xy_settle', 'z_settle', 'z_ret')
            # create new columns
            for col in cols:
                lines[col] = np.nan

            for idx, line in lines.iterrows():

                log.debug('Processing expanded parameters for line %i' % idx)

                # line_start = line.obt if line.start_time is pd.NaT else line.start_time
                if line.start_time is pd.NaT:
                    line_start = line.obt
                    log.debug('using OBT: %s' % line_start)
                else:
                    line_start = line.start_time
                    log.debug('using line start: %s' % line_start)
                if use_archive:
                    frame = hk2[hk2.index>line_start].index
                else:
                    frame = hk2[hk2.obt>line_start].index
                if len(frame)==0:
                    log.warning('no HK2 frame found after line scan at %s' % pkt.obt)
                    in_image = True # hack to put NaNs in the table if no HK available
                else:
                    frame = frame[0]

                if line.scan_type=='CON':

                    if use_archive:
                        lines['work_pt'].loc[idx] = np.nan if line.in_image else hk2['NMDA0287'].loc[frame]
                        lines['set_pt'].loc[idx] = np.nan if line.in_image else hk2['NMDA0298'].loc[frame]
                    else:

                        lines['work_pt'].loc[idx] = np.nan if line.in_image else self.get_param('NMDA0287', frame=frame)[1]
                        lines['set_pt'].loc[idx] = np.nan if line.in_image else self.get_param('NMDA0298', frame=frame)[1]

                    # set expanded parameters that are not relevant in CON mode to NaN
                    lines['res_amp'].loc[idx] = np.nan
                    lines['work_pt_per'].loc[idx] = np.nan
                    lines['set_pt_per'].loc[idx] = np.nan
                    lines['fadj'].loc[idx] = np.nan
                    lines['res_amp'].loc[idx] = np.nan

                else: # MAG or DYN modes

                    if use_archive:
                        lines['res_amp'].loc[idx] = np.nan if line.in_image else hk2['NMDA0306'].loc[frame]
                        lines['work_pt_per'].loc[idx] = np.nan if line.in_image else hk2['NMDA0181'].loc[frame]
                        lines['work_pt'].loc[idx] = np.nan if line.in_image else lines['res_amp'].loc[idx] * abs(lines['work_pt_per'].loc[idx]) / 100.
                        lines['set_pt'].loc[idx] = np.nan if line.in_image else hk2['NMDA0245'].loc[frame]
                        lines['set_pt_per'].loc[idx] = np.nan if line.in_image else hk2['NMDA0244'].loc[frame]
                        lines['fadj'].loc[idx] = np.nan if line.in_image else hk2['NMDA0347'].loc[frame]
                    else:
                        lines['res_amp'].loc[idx] = np.nan if line.in_image else self.get_param('NMDA0306', frame=frame)[1]
                        lines['work_pt_per'].loc[idx] = np.nan if line.in_image else self.get_param('NMDA0181', frame=frame)[1]
                        lines['work_pt'].loc[idx] = np.nan if line.in_image else lines['res_amp'].loc[idx] * abs(lines['work_pt_per'].loc[idx]) / 100.
                        lines['set_pt'].loc[idx] = np.nan if line.in_image else self.get_param('NMDA0245', frame=frame)[1]
                        lines['set_pt_per'].loc[idx] = np.nan if line.in_image else self.get_param('NMDA0244', frame=frame)[1]
                        lines['fadj'].loc[idx] = np.nan if line.in_image else self.get_param('NMDA0347', frame=frame)[1]

                if line.sw_ver < 665: # have to get these from HK
                    if use_archive:
                        lines['z_step'].loc[idx] = np.nan if line.in_image else hk2['NMDA0231'].loc[frame]
                        lines['exc_lvl'].loc[idx] = np.nan if line.in_image else hk2['NMDA0147'].loc[frame]
                        lines['ac_gain'].loc[idx] = np.nan if line.in_image else hk2['NMDA0118'].loc[frame]
                        lines['dc_gain'].loc[idx] = np.nan if line.in_image else hk2['NMDA0142'].loc[frame]
                    else:
                        lines['z_step'].loc[idx] = np.nan if line.in_image else self.get_param('NMDA0231', frame=frame)[1]
                        lines['exc_lvl'].loc[idx] = np.nan if line.in_image else self.get_param('NMDA0147', frame=frame)[1]
                        lines['ac_gain'].loc[idx] = np.nan if line.in_image else self.get_param('NMDA0118', frame=frame)[1]
                        lines['dc_gain'].loc[idx] = np.nan if line.in_image else self.get_param('NMDA0142', frame=frame)[1]

                # get these regardless of mode
                if use_archive:

                    lines['xy_settle'].loc[idx] = np.nan if line.in_image else hk2['NMDA0271'].loc[frame]
                    lines['z_settle'].loc[idx] = np.nan if line.in_image else hk2['NMDA0270'].loc[frame]
                    lines['z_ret'].loc[idx] = np.nan if line.in_image else hk2['NMDA0188'].loc[frame]
                else:
                    lines['xy_settle'].loc[idx] = np.nan if line.in_image else self.get_param('NMDA0271', frame=frame)[1]
                    lines['z_settle'].loc[idx] = np.nan if line.in_image else self.get_param('NMDA0270', frame=frame)[1]
                    lines['z_ret'].loc[idx] = np.nan if line.in_image else self.get_param('NMDA0188', frame=frame)[1]

            lines['z_ret_nm'] = lines.z_ret * common.zcal

        lines.drop( ['sw_major', 'sw_minor', 'sid', 'scan_mode_dir', 'sw_flags', 'mode_params', 'start_msw', 'start_lsw'], inplace=True, axis=1)

        log.info('%i line scans extracted' % (len(lines)))

        return lines


    def get_ctrl_data(self, rawdata=False, info_only=False, expand_params=False):
        """Extracts control data from TM packets"""

        ctrl_data_fmt = ">H2Bh6H3H3H"
        ctrl_data_size = struct.calcsize(ctrl_data_fmt)
        ctrl_data_names = collections.namedtuple("line_scan_names", "sid sw_minor sw_major lin_pos \
            wheel_pos tip_num x_orig y_orig step_size num_steps scan_mode main_cnt \
            num_meas block_addr sw_flags line_cnt")

        ctrl_data_pkts = self.read_pkts(self.pkts, pkt_type=20, subtype=3, apid=1084, sid=133)

        if len(ctrl_data_pkts)==0:
            log.warning('no control data packets found')
            return None

        # M.S.Bentley 23/09/2014 - using df to collect ctrl data meta info

        ctrl_data = []

        for idx,pkt in ctrl_data_pkts.iterrows():
            ctrl_data.append(ctrl_data_names(*struct.unpack(ctrl_data_fmt,pkt['data'][0:ctrl_data_size])))

        ctrl_data = pd.DataFrame(ctrl_data,columns=ctrl_data_names._fields,index=ctrl_data_pkts.index)

        # Convert data as necessary in the df
        ctrl_data['obt'] = ctrl_data_pkts.obt
        ctrl_data['z_retract'] = ctrl_data.block_addr.apply( lambda block: bool(block & 1) )
        ctrl_data['block_addr'] = ctrl_data.block_addr.apply( lambda block: block >> 1 )
        ctrl_data['sw_ver'] = ctrl_data.sw_major.apply( lambda major: '%i%i' % (major >> 4, major & 0x0F))
        ctrl_data['sw_ver'] = ctrl_data['sw_ver'].str.cat(ctrl_data['sw_minor'].values.astype(str)).astype(int)
        ctrl_data['lin_pos'] = ctrl_data.lin_pos.apply( lambda pos: pos*20./65535.)
        ctrl_data['tip_num'] += 1
        # Scan mode and direction:
        # Bits 15-13: Spare
        # Bit     12: Main scan direction (0 = X, 1 = Y)
        # Bit   11-9: Spare
        # Bit      8: Line scan direction (0 = low2high, 1 = high2low)
        # Bit    7-2: Spare
        # Bit    0-1: Mode (0 = dynamic, 1 = contact, 2 = magnetic)
        ctrl_data['fast_dir'] = ctrl_data.scan_mode.apply( lambda fast: 'X' if (fast >> 12 & 1)==0 else 'Y')
        ctrl_data['scan_dir'] = ctrl_data.scan_mode.apply( lambda xdir: 'H_L' if (xdir >> 8 & 1) else 'L_H')
        ctrl_data['scan_type'] = ctrl_data.scan_mode.apply( lambda mode: common.scan_type[ mode & 0b11 ] )
        ctrl_data['in_image'] = ctrl_data.apply( lambda row: bool(row.sw_flags & 1) if row.sw_ver>=661 else False, axis=1 )
        ctrl_data['hires'] = ctrl_data.apply( lambda row: bool(row.sw_flags >> 1 & 1) if row.sw_ver>=661 else False, axis=1 )

        ctrl_data['line_abort'] = ctrl_data.apply( lambda row: bool(row.sw_flags >> 2 & 1) if row.sw_ver>=666 else False, axis=1 )
        ctrl_data['max_cycles'] = ctrl_data.apply( lambda row: bool(row.sw_flags >> 3 & 1) if row.sw_ver>=666 else False, axis=1 )

        ctrl_data['x_closed'] = ctrl_data.apply( lambda row: bool(row.sw_flags >> 4 & 1) if row.sw_ver>=665 else False, axis=1 )
        ctrl_data['y_closed'] = ctrl_data.apply( lambda row: bool(row.sw_flags >> 5 & 1) if row.sw_ver>=665 else False, axis=1 )
        ctrl_data['z_closed'] = ctrl_data.apply( lambda row: bool(row.sw_flags >> 6 & 1) if row.sw_ver>=665 else False, axis=1 )

        ctrl_data['target'] = ctrl_data.wheel_pos.apply( lambda pos: common.seg_to_facet(pos) )

        ctrl_data.drop(['sid', 'sw_major', 'sw_minor', 'scan_mode'], inplace=True, axis=1)

        if info_only:
            return ctrl_data

        if expand_params:
            hk2 = self.pkts[ (self.pkts.type==3) & (self.pkts.subtype==25) & (self.pkts.apid==1076) & (self.pkts.sid==2) ]

        control = []

        # Now extract actual data
        for idx, pkt in ctrl_data_pkts.iterrows():

            num_steps = ctrl_data.num_meas.loc[idx]
            point_data = {}

            # 07/01/15 - M.S.Bentley - in fact all control data are signed integers!
            data = struct.unpack(">%ih" % (num_steps),pkt['data'][ctrl_data_size:ctrl_data_size+num_steps*2])

            point_data['ac']    = np.array(data[0::4])
            point_data['dc']    = np.array(data[1::4])
            point_data['phase'] = np.array(data[2::4])
            point_data['zpos']  = np.array(data[3::4])

            if not rawdata:

                point_data['ac'] = point_data['ac'] * (20./65535.)
                point_data['dc'] = point_data['dc'] * (20./65535.)
                point_data['phase'] = point_data['phase'] * (360./65535.)
                point_data['zpos'] = point_data['zpos'] # - point_data['zpos'].min()
                point_data['zpos'] = point_data['zpos'] * common.zcal * 2.0

            # Get exc_lvl, ac_gain, op_amp and set_pt from HK TM
            if expand_params:

                sw_ver = ctrl_data.loc[idx].sw_ver
                frame = hk2[hk2.obt>pkt.obt].index[0]

                if sw_ver < 664:
                    point_data['work_pt_per'] = self.get_param('NMDA0181', frame=frame)[1]
                    point_data['set_pt'] = self.get_param('NMDA0244', frame=frame)[1]
                    point_data['set_pt_per'] = self.get_param('NMDA0244', frame=frame)[1]
                    point_data['exc_lvl'] = self.get_param('NMDA0147', frame=frame)[1]
                    point_data['ac_gain'] = self.get_param('NMDA0118', frame=frame)[1]

                elif sw_ver == 664:
                    # Software flags:
                    # Bits  11-8: Excitation level (since version 6.6.4)
                    # Bits   7-4: AC gain level (since version 6.6.4)
                    point_data['work_pt_per'] = self.get_param('NMDA0181', frame=frame)[1]
                    point_data['set_pt'] = self.get_param('NMDA0245', frame=frame)[1]
                    point_data['set_pt_per'] = self.get_param('NMDA0244', frame=frame)[1]
                    point_data['exc_lvl'] = ctrl_data.loc[idx].sw_flags >> 8 & 0b111
                    point_data['ac_gain'] = ctrl_data.loc[idx].sw_flags >> 4 & 0b111

                elif sw_ver > 664:
                    # Bits 15-13: Excitation level (since v6.6.5)
                    # Bits 12-10: DC gain level (since v6.6.5)
                    # Bits   9-7: AC gain level (since v6.6.5)
                    point_data['work_pt_per'] = self.get_param('NMDA0181', frame=frame)[1]
                    point_data['set_pt'] = self.get_param('NMDA0245', frame=frame)[1]
                    point_data['set_pt_per'] = self.get_param('NMDA0244', frame=frame)[1]
                    point_data['exc_lvl'] = ctrl_data.loc[idx].sw_flags >> 13 & 0b111
                    point_data['ac_gain'] = ctrl_data.loc[idx].sw_flags >>  7 & 0b111

                point_data['res_amp'] = self.get_param('NMDA0306', frame=frame)[1]
                point_data['work_pt'] = point_data['res_amp'] * abs(point_data['work_pt_per']) / 100.
                point_data['op_pt_lo'] = self.get_param('NMDA0247', frame=frame)[1]
                point_data['op_pt_hi'] = self.get_param('NMDA0246', frame=frame)[1]


            control.append(point_data)

        ctrl_data['ac'] = [data['ac'] for data in control]
        ctrl_data['dc'] = [data['dc'] for data in control]
        ctrl_data['phase'] = [data['phase'] for data in control]
        ctrl_data['zpos'] = [data['zpos'] for data in control]

        if expand_params:
            ctrl_data['res_amp'] = [data['res_amp'] for data in control]
            ctrl_data['work_pt_per'] = [data['work_pt_per'] for data in control]
            ctrl_data['work_pt'] = [data['work_pt'] for data in control]
            ctrl_data['set_pt'] = [data['set_pt'] for data in control]
            ctrl_data['set_pt_per'] = [data['set_pt_per'] for data in control]
            ctrl_data['exc_lvl'] = [data['exc_lvl'] for data in control]
            ctrl_data['ac_gain'] = [data['ac_gain'] for data in control]
            ctrl_data['op_pt_lo'] = [data['op_pt_lo'] for data in control]
            ctrl_data['op_pt_hi'] = [data['op_pt_hi'] for data in control]

        log.info('%i control data scans extracted' % (len(ctrl_data)))

        return ctrl_data



    def get_feature_vectors(self, header=False):
        """Checks for feature vector packets and returns the relevant data"""

        import struct

        fvec_header_fmt = ">HBB9H2i"
        fvec_header_size = struct.calcsize(fvec_header_fmt)
        fvec_header_names = collections.namedtuple("fvec_header_names", \
                "sid sw_minor sw_major data_id num_feat first_feat num_fvec \
                min_z max_z feat_sel feat_weight regress_offset regress_x regress_y")

        fvec_fmt = ">H4BH6I"
        fvec_size = struct.calcsize(fvec_fmt)
        fvec_names = collections.namedtuple("fvec_names", "num_pts x_max x_min y_max y_min \
            max_z_over x_sum y_sum z_sum xx_sum yy_sum zz_sum")

        feature = []

        fvec_pkts = self.read_pkts(self.pkts, pkt_type=20, subtype=3, apid=1084, sid=134)

        if len(fvec_pkts)==0:
            log.warning('no feature vector packets found')
            return False

        feature_out = pd.DataFrame([], columns=fvec_names._fields)

        header = []

        for idx,pkt in fvec_pkts.iterrows():

            fvec_header = fvec_header_names(*struct.unpack(fvec_header_fmt,pkt['data'][0:fvec_header_size]))
            fvec_header = fvec_header._asdict()
            fvec_header['num_feat'] -= 1
            fvec_header['first_feat'] /= 2
            fvec_header['feat_weight'] *= (4./65535.)
            fvec_header['regress_x'] /= 65536.
            fvec_header['regress_y'] /= 65536.
            fvec_header['obt'] = pkt.obt

            header.append(fvec_header)

            for feat_num in range(fvec_header['num_fvec']):
                feature_start = fvec_header_size+feat_num*fvec_size
                feature_end = feature_start + fvec_size
                feature.append(fvec_names(*struct.unpack(fvec_fmt,pkt['data'][feature_start:feature_end])))

            current_feature = pd.DataFrame(feature, columns=fvec_names._fields)
            current_feature['obt'] = pkt.obt
            feature_out = feature_out.append(current_feature)

        log.info('%i feature vectors extracted' % len(feature))

        if header:
            header = pd.DataFrame(header)
            header.drop(['sid', 'sw_major', 'sw_minor'], inplace=True, axis=1)
            return header, feature_out
        else:
            return feature_out


    def get_images(self, info_only=False, rawheader=False, rawdata=False, sw_flags=False,
        expand_params=False, unpack_status=False, add_retr=False, skip_badsize=True):
        """Extracts images from telemetry packets. Setting info_only=True returns a
        dataframe containing the scan metadata, but no actual images.

        info_only=True returns only meta data
        sw_flags=True will unpack the software flags word into component values
        rawheader=True returns uncalibrated meta data (debugging only)
        rawdata=True returns uncalibrated image data (debugging only)
        expand_params=True will query HK for additional parameters (slow!)
        unpack_status=True will unpack the ST and S2 channels into new channels
        add_retr=True adds an extra channel flagging where the retraction height is exceeded."""

        # structure definition for the image header packet
        image_header_fmt = ">H2B2IHh17H4h3H"
        image_header_size = struct.calcsize(image_header_fmt)
        image_header_names = collections.namedtuple("img_header_names", "sid sw_minor sw_major start_time end_time \
            channel lin_pos wheel_pos tip_num x_orig y_orig x_step y_step xsteps_dir ysteps_dir scan_type \
            z_step scan_mode status sw_flags line_err_cnt scn_err_cnt z_ret mag_ret res_amp set_pt set_win fadj \
            dblock_start num_pkts checksum")

        image_header = []; filename = []

        # filter image header packets
        img_header_pkts = self.read_pkts(pkts=None, pkt_type=20, subtype=3, apid=1084, sid=129)
        log.debug('%i image header packets found' % (len(img_header_pkts)))

        if len(img_header_pkts) == 0:
            log.info('no images found')
            return None

        # Find the index of HK2 packets, used to extract anciliary info
        if expand_params:
            hk2 = self.pkts[ (self.pkts.type==3) & (self.pkts.subtype==25) & (self.pkts.apid==1076) & (self.pkts.sid==2) ]

        # Loop over header packets, unpack the header values and append them to a list
        for idx,pkt in img_header_pkts.iterrows():
            image_header.append(image_header_names(*struct.unpack(image_header_fmt,pkt['data'][0:image_header_size])))
            log.debug('processing file %s' % pkt.filename)
            filename.append(pkt.filename) # src filename needed to correctly name scans

        # structure definition for the individual image packets
        image_fmt = ">H2BI4H7H"
        image_size = struct.calcsize(image_fmt)
        image_names = collections.namedtuple("img_names", "sid sw_minor sw_major start_time channel pkt_num \
            first_line last_line spare1 spare2 spare3 spare4 spare5 spare6 spare7")

        if not info_only:

            # filter image packets
            log.debug('starting image data packet unpack')
            img_pkts = self.read_pkts(self.pkts, pkt_type=20, subtype=3, apid=1084, sid=130)
            log.debug('%i image packets found' % (len(img_pkts)))

            image_data = [] # holds image packet headers and data

            log.debug('unpacking image packet headers and image data')

            # loop through image packets and unpack the image packet header and image data
            # THIS EATS LOTS OF RAM
            for count, (idx, pkt) in enumerate(img_pkts.iterrows()):
                log.debug('image packet %i of %i' % (count+1, len(img_pkts)))
                image = {}
                image['header'] = image_names(*struct.unpack(image_fmt,pkt['data'][0:image_size]))
                if image['header'].channel in [1, 4, 32768]: # ZS, S2, ST
                    # unsigned
                    image['data'] = struct.unpack(">1024H",pkt['data'][image_size:image_size+2048])
                else:
                    # signed
                    image['data'] = struct.unpack(">1024h",pkt['data'][image_size:image_size+2048])
                image_data.append(image)

            log.debug('all image packet data read, processing..')

            # convert the image packet headers to a pandas dataframe for easy filtering
            img_data=pd.DataFrame([img['header'] for img in image_data],columns=image_names._fields)

            if rawdata: return image_header, img_data, image_data

            # image_header - list of raw unpacked image header packets
            # img_data - DataFrame of image packets without raw data
            # image_data - list of image packet data content

            image_temp = []

            # for each image header packet
            for idx, image in enumerate(image_header):

                log.debug('processing image %i of %i' % (idx+1, len(image_header)))

                imagedict = {}
                imagedict['info'] = image # image header into -> image['info']

                # Now match the packets to the corresponding header by OBT and channel
                data = img_data[ (img_data.start_time==image.start_time) & (img_data.channel==image.channel) ]
                data = data.sort_values(by='pkt_num')
                log.debug('%i image data packets expected, %i packets found' % (image.num_pkts, len(data)) )

                if len(data) < image.num_pkts:
                    log.warning('image at %s: missing image packets' % (self.correlate_time(image.start_time)))
                    # continue
                elif len(data) > image.num_pkts:
                    log.error('image at %s: too many image packets (%i instead of %i) - image may be corrupt!' % (self.correlate_time(image.start_time),len(data),image.num_pkts))
                    data = data.drop_duplicates(subset=('start_time','pkt_num'), keep='last')

                # Use the xsteps and ysteps values from the header to mash together and reform the data
                xsteps = image.xsteps_dir & 0x3ff
                ysteps = image.ysteps_dir & 0x3ff

                if skip_badsize:

                    if (xsteps==0) or (ysteps==0):
                        log.info('image at %s: invalid image dimensions (%i,%i), skipping...' % (self.correlate_time(image.start_time),xsteps,ysteps))
                        continue

                    if (image.x_step==0) or (image.y_step==0):
                        log.info('image at %s: invalid image step size (%i,%i), skipping....' % (self.correlate_time(image.start_time), image.x_step, image.y_step))
                        continue

                main_y = bool(image.scan_type & 2**15) # False = X, True = Y

                # merge the ordered image data into a linear array
                # 01/12/2014 - instead of skipping image, add dummy data
                image_array = []
                pkt_idx = 1
                # while pkt_idx <= image.num_pkts:
                for pkt_num in range(1,image.num_pkts+1):
                    if pkt_num not in data.pkt_num.tolist():
                        log.debug('inserting one missing packet at packet number %i' % pkt_num)
                        image_array.extend([65535]*1024)
                    else:
                        image_array.extend(image_data[ data.index[pkt_idx-1] ]['data'])
                        pkt_idx += 1

                image_array=np.array(image_array)

                if not rawdata and not info_only:

                    if len(image_array) != xsteps*ysteps:
                        log.error('image at %s: number of data points (%i) doesn\'t match image dimensions (%ix%i=%i)'
                            % (self.correlate_time(image.start_time),len(image_array),xsteps,ysteps,xsteps*ysteps))
                        continue

                    # Need to invert for Z piezo/topography since 0=piezo retracted (image max height).
                    # if image.channel==1: image_array = 32767-image_array
                    if image.channel==1: image_array = 65535 - image_array

                    # re-shape the array according to the image size and scan direction
                    if main_y:
                        image_array = np.transpose(image_array.reshape( (xsteps,ysteps) ))
                    else:
                        image_array = image_array.reshape( (ysteps,xsteps) )

                    # X L-H or H-L
                    if (image.xsteps_dir & 2**15) != 0:
                        image_array = np.fliplr(image_array)

                    # Y L-H or H-L
                    if (image.ysteps_dir & 2**15) != 0:
                        image_array = np.flipud(image_array)

                imagedict['data'] = image_array.copy()
                imagedict['filename'] = filename[idx]
                image_temp.append(imagedict)

        # Process the header data and combine into a dataframe
        if info_only:
            info = pd.DataFrame([image for image in image_header],columns=image_header[0]._fields)
        else:
            info = pd.DataFrame([image['info'] for image in image_temp],columns=image_temp[0]['info']._fields)

        if rawheader: return info

        # Calibrate and calculate anything which doesn't directly come from the header words
        info.start_time = pd.to_datetime(info.start_time.apply( lambda obt: np.nan if obt==0 else self.correlate_time(obt) ))
        info.end_time = pd.to_datetime(info.end_time.apply( lambda obt: np.nan if obt==0 else self.correlate_time(obt) ))
        info['duration'] = info.end_time-info.start_time
        info['filename'] = pd.Series([name for name in filename])
        info['lin_pos'] = info.lin_pos.apply( lambda pos: pos*20./65535.)

        # Bits 0-9:X num steps = number of data points (32, 64, ..., 512)
        # Bit 15 (MSB): X scan direction (0 = standard, 1 = opposite)
        info['xsteps'] = info.xsteps_dir.apply( lambda xsteps: xsteps & 0x3ff )
        info['ysteps'] = info.ysteps_dir.apply( lambda ysteps: ysteps & 0x3ff )
        info['x_dir'] = info.xsteps_dir.apply( lambda xdir: 'L_H' if (xdir & 2**15)==0 else 'H_L')
        info['y_dir'] = info.ysteps_dir.apply( lambda ydir: 'L_H' if (ydir & 2**15)==0 else 'H_L' )

        # scan mode: 1 = dynamic, 2 = contact, 3 = magnetic, others TBD
        # Bit 15 (MSB): main scan direction (0 = X, 1 = Y)
        info['fast_dir'] = info.scan_type.apply( lambda fast: 'X' if (fast & 2**15)==0 else 'Y')
        info['aborted'] = info.scan_type.apply( lambda stype: bool(stype >> 14 & 1) )
        info['dummy'] = info.scan_type.apply( lambda mode: bool(mode >> 2 & 1) )
        info['scan_type'] = info.scan_type.apply( lambda mode: common.scan_type[ mode & 0b11 ] )
        info['exc_lvl'] = info.scan_mode.apply( lambda mode: mode >> 13 )
        info['dc_gain'] = info.scan_mode.apply( lambda mode: mode >> 10 & 0b111 )
        info['ac_gain'] = info.scan_mode.apply( lambda mode: mode >> 7  & 0b111 )
        info['z_closed'] = info.scan_mode.apply( lambda mode: bool(mode >> 6 & 1) )
        info['y_closed'] = info.scan_mode.apply( lambda mode: bool(mode >> 5 & 1) )
        info['x_closed'] = info.scan_mode.apply( lambda mode: bool(mode >> 4 & 1) )
        info['scan_algo'] = info.scan_mode.apply(lambda mode: common.scan_algo[mode & 0b1111] )
        info['target'] = info.wheel_pos.apply( lambda seg: common.seg_to_facet(seg) )
        info['target_type'] = info.target.apply( lambda tgt: common.target_type(tgt) )
        info['tip_num'] = info.tip_num+1
        info['sw_ver'] = info.sw_major.apply( lambda major: '%i%i' % (major >> 4, major & 0x0F))
        info['sw_ver'] = info['sw_ver'].str.cat(info['sw_minor'].values.astype(str)).astype(int)
        info['channel'] = info.channel.apply( lambda channel: common.data_channels[int(math.log10(channel)/math.log10(2))] )

        # Remove data from z_step in older OBSW versions (was formerly data set ID)
        info['z_step'][info['sw_ver']<665] = np.nan

        info['res_amp'] = info.res_amp * 20./65535.
        info['set_pt'] = info.set_pt * 20./65535.
        info['fadj'] = info.fadj * 20./65535.

        # Instrument status
        info['mtl_disabled'] = info.status.apply( lambda status: bool(status >> 15 & 1) )
        info['auto_expose_mode'] = info.status.apply( lambda status: bool(status >> 4 & 0b111 ) )
        info['tech_mode'] = info.status.apply( lambda status: bool(status & 0b1 ) )

        # Software flags
        info['mag_phase'] = info.sw_flags.apply( lambda flag: bool( flag >> 13 & 0b1 ) )
        info['line_tx'] = info.sw_flags.apply( lambda flag: bool( flag >> 12 & 0b1 ) )
        info['auto_expose'] = info.sw_flags.apply( lambda flag: bool( flag >> 8 & 0b1 ) )
        info['anti_creep'] = info.sw_flags.apply( lambda flag: bool( flag >> 7 & 0b1 ) )
        info['ctrl_retract'] = info.sw_flags.apply( lambda flag: bool( flag >> 6 & 0b1 ) )
        info['ctrl_image'] = info.sw_flags.apply( lambda flag: bool( flag >> 5 & 0b1 ) )
        info['ctrl_image'][info['sw_ver']<660] = False
        info['line_in_img'] = info.sw_flags.apply( lambda flag: bool( flag >> 4 & 0b1 ) )
        info['calc_zret'] = info.sw_flags.apply( lambda flag: bool( flag >> 3 & 0b1 ) )
        info['linefeed_zero'] = info.sw_flags.apply( lambda flag: bool( flag >> 2 & 0b1 ) )
        info['linefeed_last_min'] = info.sw_flags.apply( lambda flag: bool( flag >> 1 & 0b1 ) )
        info['fscan_phase'] = info.sw_flags.apply( lambda flag: bool( flag & 0b1 ) )

        sw_flags_names = ['line_tx', 'auto_expose', 'anti_creep', 'ctrl_retract', 'line_in_img',
            'linefeed_zero', 'linefeed_last_min', 'fscan_phase']

        # Calculate the real step size
        info['x_step_nm'] = info.x_step.apply( lambda step: step * common.xycal['open'])
        info['y_step_nm'] = info.apply( lambda row: (row.y_step * common.xycal['closed']) if row.y_closed else (row.y_step * common.xycal['open']), axis=1)
        info['xlen_um'] = info.xsteps * info.x_step_nm /1000.
        info['ylen_um'] = info.ysteps * info.y_step_nm /1000.
        info['z_ret_nm'] = info.z_ret * common.zcal

        # Check for OBSW version < 663 in which tip number was incorrectly calculated
        if len(info.query('sw_ver<663'))>0:
            hk2 = self.pkts[ (self.pkts.type==3) & (self.pkts.subtype==25) & (self.pkts.apid==1076) & (self.pkts.sid==2) ]
            old_images = info.query('sw_ver<663')
            for idx, img in old_images.iterrows():
                frame = hk2[hk2.obt>img.start_time].index
                if len(frame)==0:
                    log.warning('no HK2 frame found after scan start at %s' % img.start_time)
                    continue
                else:
                    frame = frame[0]
                    # NMDA0127 (CanSelect) 0-7
                    cant_num = self.get_param('NMDA0127', frame=frame)[1]
                    # NMDA0128 (CanBlockSelect)
                        #  <660: 1 - 2
                        # >=660: 0 - 1
                    block_num = self.get_param('NMDA0128', frame=frame)[1]
                    if img.sw_ver < 660:
                        block_num -= 1
                    info['tip_num'].loc[idx] = block_num*8 + cant_num+1

        # Calculate the tip offset from the wheel centre
        info['tip_offset'] = info.apply( lambda row: (row.lin_pos-self.lin_centre_pos[row.tip_num-1]) / common.linearcal, axis=1 )

        # Apply corrections due to relative tip offsets
        info['tip_offset'] = info.apply( lambda row: (row.tip_offset - common.tip_offset[row.tip_num-1]), axis=1 )

        # Add the filename
        info['scan_file'] = info.apply( lambda row: src_file_to_img_file(os.path.basename(row.filename), row.start_time, row.target), axis=1 )

        if expand_params:
            # If requested, extract additional data from HK - note that this will make the routine SLOOOOOW!
            #
            # NMDA0306 (ResonanceAmpl),
            # NMDA0245 (OpPointAmpl), NMDA0244 (OpPointPerc)
            # NMDA0347 (FadjustAmpl),
            # NMDA0181 (OpPntPercentAmpl)
            # NMDA0271 (SettleTimeXY), # NMDA0270 (SettleTimeZ)

            expanded_names = ['work_pt', 'set_pt_per', 'work_pt_per', 'xy_settle', 'z_settle']

            # Create new columns
            info = pd.concat( [info, pd.DataFrame(columns=expanded_names)], axis=1 )

            times = info.start_time.unique()

            for time in times:

                frame = hk2[hk2.obt>time].index
                if len(frame)==0:
                    log.warning('no HK2 frame found after scan start at %s' % time)
                    continue
                else:
                    frame = frame[0]

                indices = info[info.start_time==time].index

                # numeric version of the OBSW version
                sw_ver = info[info.start_time==time].sw_ver.iloc[0]

                if sw_ver < 664:

                    info['res_amp'].loc[indices] = self.get_param('NMDA0306', frame=frame)[1]
                    info['set_pt'].loc[indices] = self.get_param('NMDA0245', frame=frame)[1]
                    info['fadj'].loc[indices] = self.get_param('NMDA0347', frame=frame)[1]

                if sw_ver < 665:
                    info['z_step'].loc[indices] = self.get_param('NMDA0231', frame=frame)[1]

                info['set_pt_per'].loc[indices] = self.get_param('NMDA0244', frame=frame)[1]
                info['work_pt_per'].loc[indices] = self.get_param('NMDA0181', frame=frame)[1]
                info['work_pt'].loc[indices] = info['res_amp'] * abs(info['work_pt_per'].loc[indices].iloc[0]) / 100.
                info['xy_settle'].loc[indices] = self.get_param('NMDA0271', frame=frame)[1]
                info['z_settle'].loc[indices] = self.get_param('NMDA0270', frame=frame)[1]

        return_data = ['filename', 'scan_file', 'sw_ver', 'start_time','end_time', 'duration', 'channel', 'tip_num', 'lin_pos', 'tip_offset', 'wheel_pos', 'target', 'target_type', \
            'x_orig','y_orig','xsteps', 'x_step','x_step_nm','xlen_um','ysteps','y_step','y_step_nm','ylen_um', 'z_step', 'z_ret', 'z_ret_nm', 'x_dir','y_dir','fast_dir','scan_type',\
            'exc_lvl', 'ac_gain', 'x_closed', 'y_closed', 'aborted', 'dummy', 'res_amp', 'set_pt', 'fadj', 'ctrl_image', 'mag_phase']

        if sw_flags: return_data.extend(sw_flags_names)
        if expand_params: return_data.extend(expanded_names)

        images = info[return_data]

        # De-dupe (should be no dupes, but in certain cases the instrument generates them)
        num_images = len(images)
        images = images.drop_duplicates(subset=['start_time','channel'])
        if len(images)<num_images:
            log.warning('duplicate images or image headers detected and removed!')

        # Remove dummy scans
        if len(images)>0:
            dummy = images.query('x_orig==0 & y_orig==0 & exc_lvl==0 & ac_gain==0')
            images.drop(dummy.index, inplace=True)
            dummy = images.query('exc_lvl==0 & ac_gain==0 & lin_pos>-0.0005 & lin_pos<0.0005 & tip_num==[1,16]')
            images.drop(dummy.index, inplace=True)
            images = images[ ~images.dummy ]

        if len(images)==0:
            log.warning('only dummy images found!')
            return None

        # Add the origin of the scan, in microns, from the wheel centre
        images = locate_scans(images)

        if not info_only:
            images['data'] = pd.Series([image['data'] for image in image_temp])

            # Unpack ST channel into multiple virtual channels if requested. Format:
            # NC: Bits 15-5 = number of cycles (limited to 0x7FF)
            # RP: Bit 4 = retraction after point advance flag
            # LA: Bit 3 = line aborted flag
            # MC: Bit 2 = max. number of cycles reached flag
            # PA: Bit 1 = point aborted flag
            # PC: Bit 0 = point converged flag

            if unpack_status:

                bit_start = [5, 4, 3, 2, 1, 0]
                bit_len =  [11, 1, 1, 1, 1, 1]
                status_images = images[images.channel=='ST']

                # duplicate the status channel the correct number of times
                # then correctly mask the data and edit the channel designation
                for idx, status in status_images.iterrows():

                    images = images.append([status]*(len(common.status_channels)-1)) # duplicate
                    images.loc[ images.index==idx, 'channel' ] = common.status_channels # re-label channel names

                images.reset_index(inplace=True, drop=True) # to avoid duplicate indices

                # loop through each status channel and apply bit shifts and masks
                for idx, channel in enumerate(common.status_channels):
                    images.loc[ images.channel==channel, 'data' ] = images.loc[ images.channel==channel, 'data' ].apply( lambda row: row >> bit_start[idx] )
                    images.loc[ images.channel==channel, 'data' ] = images.loc[ images.channel==channel, 'data' ].apply( lambda row: row & 2**bit_len[idx]-1 )

                    # Set the 1-bit types to booleans
                    if channel != 'NC':
                        images.loc[ images.channel==channel, 'data' ] = images.loc[ images.channel==channel, 'data' ].apply(lambda item: item.astype('bool'))


                # Unpack S2 if present
                # 0x0004 = Scan status parameters 2 (S2):
                #   Bits 15-1 = retraction distance / 2
                #   Bit        0 = contact flag

                bit_start = [1, 0]
                bit_len =  [15, 1]
                status_images = images[images.channel=='S2']

                for idx, status in status_images.iterrows():

                    images = images.append([status]*(len(common.s2_channels)-1)) # duplicate
                    images.loc[ images.index==idx, 'channel' ] = common.s2_channels # re-label channel names

                images.reset_index(inplace=True, drop=True) # to avoid duplicate indices

                # loop through each status channel and apply bit shifts and masks
                for idx, channel in enumerate(common.s2_channels):
                    images.loc[ images.channel==channel, 'data' ] = images.loc[ images.channel==channel, 'data' ].apply( lambda row: row >> bit_start[idx] )
                    images.loc[ images.channel==channel, 'data' ] = images.loc[ images.channel==channel, 'data' ].apply( lambda row: row & 2**bit_len[idx]-1 )

                    # Set the 1-bit types to booleans
                    if channel == 'CF':
                        images.loc[ images.channel==channel, 'data' ] = images.loc[ images.channel==channel, 'data' ].apply(lambda item: item.astype('bool'))


            # Add a channel showing the any pixels where the retraction height was exceeded
            if add_retr:
                images = check_retract(images, boolmask=False)

        log.info('%i images found' % (len(info.start_time.unique())))

        # Use categories instead of pure strings for columns where the input range is limited
        catcols = {
            'channel': common.data_channels,
            'target_type': ['CAL', 'SOLGEL', 'SILICON'],
            'x_dir': ['L_H', 'H_L'],
            'y_dir': ['L_H', 'H_L'],
            'fast_dir': ['X', 'Y'],
            'scan_type': common.scan_type}
        for cat in catcols.keys():
            images['%s'%cat] = images['%s'%cat].astype('category')
            images['%s'%cat].cat.set_categories(catcols[cat], inplace=True)

        return images.sort_values(by=['start_time','channel'])


    def get_freq_scans(self, printdata=False, fit=False, info_only=False, get_thresh=False):
        """Extracts frequency scans from TM packets.

        fit=True fits a Lorentzian to all curves and returns the fit parameters.
        printdata=True displays the parameters
        info_only=True returns the meta-data without the raw data
        get_thresh=True returns extra parameters from HK"""

        # Define frequency packet format
        # M.S.Bentley 11/09/2014 - updating for latest OBSW
        freq_scan_fmt = ">H2B2I2HI6H7H"
        freq_scan_size = struct.calcsize(freq_scan_fmt)
        freq_scan_names = collections.namedtuple("freq_scan_names", "sid sw_minor sw_major start_time freq_start \
            freq_step max_amp freq_at_max num_scans fscan_cycle tip_num cant_block exc_lvl ac_gain fscan_type \
            set_pt res_amp fadj work_pt set_pt_delta spare")

        freq_scan_pkts = self.read_pkts(self.pkts, pkt_type=20, subtype=3, apid=1084, sid=131)

        if len(freq_scan_pkts)==0:
            log.warning('no frequency scan packets found')
            return False

        # convert the fscan packet header (everything up to the scan data) into a df
        fscans = []

        for idx,pkt in freq_scan_pkts.iterrows():
            fscans.append(freq_scan_names(*struct.unpack(freq_scan_fmt,pkt['data'][0:freq_scan_size])))
        fscans = pd.DataFrame(fscans,columns=freq_scan_names._fields,index=freq_scan_pkts.index)

        # find the number of unique scans (unique start OBTs)
        start_times = fscans.start_time.unique()

        # Find the index of HK2 packets, used to extract anciliary info
        if get_thresh:
            hk2 = self.pkts.query('type==3 & subtype==25 & apid==1076 & sid==2')

        freqscans = []
        amp_scans = []

        for idx, start in enumerate(start_times):

            single_scan = fscans[fscans.start_time==start]
            single_scan  = single_scan.sort_values(by='fscan_cycle',ascending=True)
            if not 1 in single_scan.fscan_cycle.tolist():
                log.error('missing frequency scan packets, skipping scan')
                continue
            else:
                first_pkt = single_scan[single_scan.fscan_cycle==1].iloc[0]
                last_pkt = single_scan.iloc[-1]

            num_scans = first_pkt.num_scans

            if len(single_scan) != num_scans:
                log.error('missing frequency scan packets, skipping scan')
                continue

            # take the scan info from the first packet where generic to entire scan
            scan = {}
            scan['info'] = {}

            # apply time correlation here also!

            # start_time = obt_epoch + timedelta(seconds=first_pkt.start_time)
            start_time = self.correlate_time(first_pkt.start_time)

            # Look at HK2 data to retrieve the resonance amplitude, working point, frequency
            # adjust point and set-point derived from this scan...

            # frequency scans can last up to ~3 minutes, so taking the first HK data
            # after the start is often wrong! Using nun_scans to calculate the predicted
            # duration and finding first HK packet after that

            if get_thresh:
                duration = common.fscan_duration(num_scans)
                frame = hk2[hk2.obt>(start_time+duration)].index
                if len(frame)==0:
                    log.warning('no HK2 frame found after frequency scan at %s' % start)
                    continue
                else:
                    frame = frame[0]

            sw_ver = int("%i%i%i" % (first_pkt.sw_major >> 4, first_pkt.sw_major & 0x0F, first_pkt.sw_minor))
            # set_pt res_amp fadj work_pt set_pt_delta spare
            scan['info'] = {
                'sw_ver': sw_ver,
                'start_time': start_time,
                'freq_start': first_pkt.freq_start * 0.0006984, # 00021065027,
                'freq_step': first_pkt.freq_step * 0.0006984, # 00021065027,
                'max_amp': single_scan.max_amp.max()*(20./65535.),
                'max_freq': single_scan.freq_at_max.max() * 0.0006984, # 00021065027,
                'num_scans': first_pkt.num_scans,
                'tip_num': (first_pkt.tip_num+1 + first_pkt.cant_block*8) if sw_ver>=660 else
                    (first_pkt.tip_num + (first_pkt.cant_block-1)*8),
                'cant_block': first_pkt.cant_block if sw_ver>=660 else first_pkt.cant_block-1,
                'exc_lvl': first_pkt.exc_lvl,
                'ac_gain': first_pkt.ac_gain,
                'res_amp': last_pkt.res_amp * (20./65535.) if sw_ver>=664 else np.nan,
                'fadj': last_pkt.fadj * (20./65535.) if sw_ver>=664 else np.nan,
                'work_pt': last_pkt.work_pt * (20./65535.) if sw_ver>=665 else np.nan,
                'set_pt': last_pkt.set_pt * (20./65535.) if sw_ver>=664 else np.nan,
                'set_pt_delta': last_pkt.set_pt_delta * (20./65535.) if sw_ver>=665 else np.nan,
                'is_phase': True if (first_pkt.fscan_type & 1) else False,
                'above_thresh': False if (last_pkt.fscan_type >> 1 & 1) else True } # threshold scan flag (0=ok, 1=not found) }

            if scan['info']['is_phase']:
                do_fit = False
            else:
                do_fit = fit
                amp_scans.append(idx)

            if get_thresh:

                if sw_ver < 665:
                    scan['info']['op_pt_delta'] = self.get_param('NMDA0242', frame=frame)[1]
                    scan['info']['work_pt'] = (scan['info']['res_amp'] * abs(self.get_param('NMDA0181', frame=frame)[1]) / 100.)
                    scan['info']['set_pt_delta'] = self.get_param('NMDA0242', frame=frame)[1]

                if sw_ver < 664:
                    scan['info']['res_amp'] = self.get_param('NMDA0306', frame=frame)[1]
                    scan['info']['set_pt'] = self.get_param('NMDA0245', frame=frame)[1]
                    scan['info']['fadj'] = self.get_param('NMDA0347', frame=frame)[1]

                scan['info']['set_pt_per'] = self.get_param('NMDA0244', frame=frame)[1]
                scan['info']['work_pt_per'] = self.get_param('NMDA0181', frame=frame)[1]

                # NMDA0247 (OpLo) and NMDA0246 (OpUp)
                scan['info']['op_pt_lo'] = self.get_param('NMDA0247', frame=frame)[1]
                scan['info']['op_pt_hi'] = self.get_param('NMDA0246', frame=frame)[1]

            if printdata:
                log.info('cantilever %i/%i with gain/exc %i/%i has peak amplitude %3.2f V at frequency %3.2f Hz' % \
                    (scan['info']['cant_block'], scan['info']['tip_num'], \
                    scan['info']['ac_gain'], scan['info']['exc_lvl'], \
                    scan['info']['max_amp'], scan['info']['max_freq']) )

            # loop through the packets and add the fscan data
            if not info_only:
                scan['amplitude'] = []
                for idy in single_scan.index:
                    if scan['info']['is_phase']:
                        scan['amplitude'].extend(struct.unpack(">256h",freq_scan_pkts.data.loc[idy]\
                            [freq_scan_size:freq_scan_size+512]))
                    else:
                        scan['amplitude'].extend(struct.unpack(">256H",freq_scan_pkts.data.loc[idy]\
                            [freq_scan_size:freq_scan_size+512]))

                if first_pkt.fscan_type==1:
                    scan['amplitude'] = np.array(scan['amplitude']) * (360./65535.)#-180.
                else:
                    scan['amplitude'] = np.array(scan['amplitude']) * (20./65535.)

                scan['frequency'] = np.linspace(start=scan['info']['freq_start'], \
                    stop=scan['info']['freq_start']+scan['info']['num_scans']*256.*scan['info']['freq_step'], \
                    num=scan['info']['num_scans']*256,endpoint=False)

            freqscans.append(scan)

            if do_fit and not info_only: # perform a Lorentzian fit to the frequency curve
                from scipy.optimize import curve_fit

                # initial estimate
                offset = 0.0
                amp = scan['info']['max_amp']
                cen = scan['info']['max_freq']
                wid = cen - scan['frequency'][np.where(scan['amplitude'] > max(scan['amplitude'])/2.)[0][0]]

                try:
                    popt, pcov = curve_fit(lorentzian,scan['frequency'],scan['amplitude'],p0=[offset, amp, cen, wid])
                except RuntimeError:
                    log.warning('problem with curvefit, skipping this scan...')
                    amp_scans.remove(idx)
                    continue

                fit_params={}
                fit_params['offset'] = popt[0]
                fit_params['fit_max'] = popt[1]+popt[0]
                fit_params['res_freq'] = popt[2]
                fit_params['half_width'] = popt[3]
                fit_params['q'] = fit_params['res_freq'] / (2. * fit_params['half_width'])
                fit_params['cov'] = pcov

                fit_keys = fit_params.keys()

                scan['fit_params'] = fit_params

        log.info('%i frequency scans found and extracted' % (len(freqscans)))

        # Convert to a DataFrame for return
        scans = pd.DataFrame( [scan['info'] for scan in freqscans], columns=freqscans[0]['info'].keys())
        scans = scans[ scans.columns.sort_values() ]

        # Include fit parameters if calculated
        if fit:
            fits = pd.DataFrame(
                [scan['fit_params'] for scan in freqscans if scan.has_key('fit_params')],
                columns=fit_keys, index=amp_scans )
            scans = pd.concat( [scans,fits], axis=1, join='inner')

        # Add the actual data
        if not info_only:
            scans['frequency'] = pd.Series([scan['frequency'] for scan in freqscans])
            scans['amplitude'] = pd.Series([scan['amplitude'] for scan in freqscans])

        return scans


    def app_at_surface(self):
        """Scans pkts for ApproachFinished events, finds the next HK2 packet, and extracts
        the approach position, cantilever and segment number."""

        # Filter packets to find the ApproachFinished event
        srf_found = self.pkts[ (self.pkts.apid==1079) & (self.pkts.sid==42664) ]
        log.info('%i ApproachFinished events found' % len(srf_found))

        # Filter HK2 packets
        hk2 = self.pkts[ (self.pkts.type==3) & (self.pkts.subtype==25) & (self.pkts.apid==1076) & (self.pkts.sid==2) ]

        approaches = []

        # Loop through the OBTs and find the next HK2 packet, then extract parameters
        for idx,event in srf_found.iterrows():
            approach = {}
            frame = hk2[hk2.obt>event.obt].index
            if len(frame)==0:
                log.warning('no HK2 frame found after approach at %s' % event.obt)
                continue
            else:
                frame = frame[0]

            approach['obt'] = self.pkts.obt.loc[frame]
            approach['position'] = self.get_param('NMDA0123', frame=frame)[1] # AppPosition
            approach['segment'] = self.get_param('NMDA0196', frame=frame)[1] # WheSegmentNum
            # approach['block'] = self.get_param('NMDA0128', frame=frame)[1] # CanBlockSelect
            approach['cantilever'] = self.get_param('NMDA0203', frame=frame)[1]+1  # ScnTipNumber
            approach['z_hv'] = self.get_param('NMDA0115', frame=frame)[1] # Z piezo HV mon
            approach['z_pos'] = self.get_param('NMDA0114', frame=frame)[1] # Z piezo position


            approaches.append(approach)

        # Note that a statistical summary of the approach value per segment can be found with:
        # approaches.groupby('segment').describe().position

        return pd.DataFrame(approaches)


    def cantilever_usage(self, cantilever=None, html=None):
        """Summarises cantilever usage (number of image and line scans etc.) for all
        cantilevers, or for those specified by cantilever="""

        images = self.get_images(info_only=True)
        lines = self.get_line_scans(info_only=True)

        events = self.get_events(info=False, verbose=False)
        events = events[ events.sid.isin([42611, 42655])]

        lines = lines[ ~lines.in_image ]
        lines = lines[ ~lines.anti_creep ]
        lines = lines[ lines.line_cnt==1 ]

        for idx, image in images.iterrows():
            lines = lines[ (lines.obt<image.start_time) | (lines.obt>image.end_time) ]

        cant_usage = pd.DataFrame(columns=['tip_num', 'num_images', 'num_lines', 'num_points'])

        if cantilever is None:
            cantilever=range(1,17)
        elif type(cantilever) not in [list, int]:
            log.warning('cantilever= must be an integer or list of integers')
        elif type(cantilever)==int:
            cantilever=[cantilever]

        for cant in cantilever:

            num_points = 0

            start_times = images.query('tip_num==%i' % cant).start_time.unique()

            for time in start_times:
                img = images[ (images.tip_num==cant) & (images.start_time==time) & (images.channel=='ZS') ].squeeze()
                num_points += img.xsteps * img.ysteps

            num_points += lines.query('tip_num==%i' % cant).num_steps.sum()

            cant_usage = cant_usage.append(
                {'tip_num': cant,
                'num_images': len(images.query('channel=="ZS" & tip_num==%i' % cant)),
                'num_lines':  len(lines.query('tip_num==%i' % cant)),
                'num_points': num_points }, ignore_index=True)

        cant_usage.sort_values(by='num_points', ascending=False, inplace=True)

        if html is not None:

            usage_html = cant_usage.to_html(classes='alt_table', na_rep='', index=False)
            css_write(html=usage_html, filename=html)


        return cant_usage

    def target_history(self, target=1, images=None, exposures=None, html=None):
        """Produces a summary of target usage (exposures, image scans)"""

        if (target<1) or (target>64):
            log.error('target= must be between 1 and 64')
            return None

        if images is None:
            images = self.get_images(info_only=True)
        if exposures is None:
            exposures = self.get_exposures()

        images = images[ (images.target==target) & (images.channel=='ZS') ]
        if len(images)>0:
            images.rename(columns={'start_time':'start'}, inplace=True)
            images['description'] = images.apply( lambda row: '%d x %d (%3.2f x %3.2f) with tip %d' % (row.xsteps, row.ysteps, row.xlen_um, row.ylen_um, row.tip_num), axis=1)
            images = images[ ['start', 'description', 'scan_file' ] ]
            images['activity'] = 'IMAGE SCAN'

        if len(exposures)>0:
            exposures = exposures[exposures.target==target]
            exposures['activity'] = 'TARGET EXPOSE'
            exposures['description'] = exposures.duration.apply( lambda dur: '%s' % timedelta( seconds = dur / np.timedelta64(1, 's')))
            exposures.drop(['end','target', 'duration'], inplace=True, axis=1)

        if len(exposures)==0 and len(images)==0:
            log.warning('target %d has no exposures or images' % target)
            return None

        history = pd.concat([exposures, images]).sort_values(by='start')

        history = history[ ['start','activity','scan_file', 'description'] ]

        if html is not None:

            timeformatter = lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S')
            history_html = history.to_html(classes='alt_table', na_rep='', index=False, formatters={ 'start': timeformatter} )
            css_write(html=history_html, filename=html)

        return history


    def commanding_events(self, itl_file, evf_file, html=False):
        """Accepts an ITL and EVF file corresponding to the TM object and lists the commands
        along with the matching event history"""

        from midas import planning

        events=self.get_events(info=True, verbose=False)
        events.rename(columns={'obt': 'event_time'}, inplace=True)
        events.drop(labels=['doy', 'sid'], inplace=True, axis=1)
        events.set_index(keys='event_time', drop=True, inplace=True)

        timeline = planning.resolve_time(itl_file=itl_file, evf_file=evf_file)

        if timeline is None:
            return None

        timeline.rename(columns={'abs_time': 'tc_time'}, inplace=True)
        timeline.drop(labels=['doy'], inplace=True, axis=1)
        timeline.set_index(keys='tc_time', drop=True, inplace=True)

        merged = timeline.join(events, how='outer')
        merged['time'] = merged.index
        merged['time'] = merged.time.astype(pd.Timestamp)

        merged = merged[ ['time', 'sequence', 'description', 'event', 'information', 'severity' ] ]

        if html:

            timeformatter = lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S')
            merged_html = merged.to_html(classes='alt_table', na_rep='',index=False, formatters={ 'time': timeformatter } )
            css_write(html=merged_html, filename=html)

        return merged


    def correlate_time(self, obt_s):
        """Apply time correlation to obt, given as a datetime."""
        # Use the appropriate line in the time correlation packet to correct
        # UTC = gradient * OBT + offset
        obt = obt_epoch + timedelta(seconds=obt_s)
        if obt<self.tcorr.index[0]:
            tcp = self.tcorr.iloc[0]
        else:
            tcp = self.tcorr[self.tcorr.index<obt].iloc[-1]
        utc_s = tcp.gradient * obt_s + tcp.offset
        obt_corr = sun_mjt_epoch + timedelta(seconds=utc_s)

        return obt_corr


    def meta_from_hk(self, start_obt, end_obt, get_times=True, time_win=120):
        """Retrieves image meta-data (as is usually given in the image packet header)
        from housekeeping. Data are taken from the first packet after start_obt. The event
        history is queried within time_win before start_obt and after end_obt to look for a
        scan start/end event.

        If get_times=True the event history is queried to fine start and end times.
        Otherwise the passed start and end OBTs are used."""

        hk1 = self.pkts.query('type==3 & subtype==25 & apid==1076 & sid==1')
        hk2 = self.pkts.query('type==3 & subtype==25 & apid==1076 & sid==2')
        time_win = pd.Timedelta(minutes=time_win)

        if type(start_obt) != pd.Timestamp:
            start_obt = pd.Timestamp(start_obt)

        if type(end_obt) != pd.Timestamp:
            end_obt = pd.Timestamp(end_obt)

        hk1_frame = hk1[hk1.obt > start_obt].index
        hk2_frame = hk2[hk2.obt > start_obt].index

        if len(hk1_frame)==0:
            log.warning('no HK1 frame found after OBT %s' % obt)
            return None
        else:
            hk1_frame = hk1_frame[0]

        if len(hk2_frame)==0:
            log.warning('no HK2 frame found after OBT %s' % obt)
            return None
        else:
            hk2_frame = hk2_frame[0]

        events = self.get_events(ignore_giada=True, verbose=False)

        meta = {
            'channel': 'ZS',
            'dummy': False }

        # NMDD0024       SwKernelVersion
        # NMDD0020        SwMajorVersion
        # NMDD0028        SwMinorVersion
        kernel = self.get_param('NMDD0024', frame=hk2_frame)[1]
        major = self.get_param('NMDD0020', frame=hk2_frame)[1]
        minor = self.get_param('NMDD0028', frame=hk2_frame)[1]
        meta.update({'sw_ver': int('%d%d%d' % (kernel, major, minor))})

        # NMDA0128        CanBlockSelect (0-1)
        # NMDA0127             CanSelect (0-7)
        block = self.get_param('NMDA0128', frame=hk2_frame)[1]
        cant = self.get_param('NMDA0127', frame=hk2_frame)[1]
        meta.update({'tip_num': block*8 + cant+1})

        if get_times:

            # get from event history
            start_ev = events[ (events.sid==42656) & (events.obt>(start_obt-time_win)) & (events.obt<start_obt) ]  # EvFullScanStarted
            if len(start_ev)==0:
                log.error('no start time found within %d seconds, using OBT %s' % (time_win.seconds, start_obt))
                meta.update({'start_time': start_obt})
            elif len(start_ev)>1:
                log.warning('more than one start event in window - selecting last')
                meta.update({'start_time': start_ev.iloc[-1].obt})
            else:
                meta.update({'start_time': start_ev.iloc[0].obt})

            end_ev = events[  (events.sid==42513) & (events.obt>(end_obt)) & (events.obt<(end_obt+time_win)) ]  # EvScanFinished
            if len(end_ev)==0:
                log.error('no end time found within %d seconds' % time_win.seconds)
                meta.update({'end_time': end_obt})
            elif len(end_ev)>1:
                log.warning('more than one end event in window - selecting first')
                meta.update({'end_time': end_ev.iloc[0].obt})
            else:
                meta.update({'end_time': end_ev.iloc[0].obt})

        else:

            meta.update( {
                'start_time': start_obt,
                'end_time': end_obt })

        # deal with linear position and tip offset
        meta.update({'lin_pos': self.get_param('NMDA0139', frame=hk2_frame)[1] }) # NMDA0139

        meta.update({'wheel_pos': self.get_param('NMDA0196', frame=hk2_frame)[1]})
        meta.update({'target': common.seg_to_facet(meta['wheel_pos'])})
        meta.update({'target_type': common.target_type(meta['target'])})

        meta.update({
            'x_orig': self.get_param('NMDA0218', frame=hk2_frame)[1],
            'y_orig': self.get_param('NMDA0228', frame=hk2_frame)[1],
            'xsteps': self.get_param('NMDA0219', frame=hk2_frame)[1],
            'ysteps': self.get_param('NMDA0230', frame=hk2_frame)[1],
            'x_step': self.get_param('NMDA0221', frame=hk2_frame)[1],
            'y_step': self.get_param('NMDA0227', frame=hk2_frame)[1],
            'z_step': self.get_param('NMDA0231', frame=hk2_frame)[1],
            'z_ret': self.get_param('NMDA0188', frame=hk2_frame)[1],
            'fast_dir': self.get_param('NMDA0171', frame=hk2_frame)[1],
            'x_dir': self.get_param('NMDA0220', frame=hk2_frame)[1],
            'y_dir': self.get_param('NMDA0229', frame=hk2_frame)[1],
            'scan_type': common.scan_type[ self.get_param('NMDA0189', frame=hk2_frame)[1] ],
            'x_closed': True if self.get_param('NMDD0015', frame=hk1_frame)[1]=='ON' else False,
            'y_closed': True if self.get_param('NMDD0014', frame=hk1_frame)[1]=='ON' else False,
            'xy_settle': self.get_param('NMDA0271', frame=hk2_frame)[1],
            'z_settle': self.get_param('NMDA0270', frame=hk2_frame)[1],
            'exc_lvl': self.get_param('NMDA0147', frame=hk2_frame)[1],
            'ac_gain': self.get_param('NMDA0118', frame=hk2_frame)[1],
            'aborted': True if self.get_param('NMDA0116', frame=hk2_frame)[1]==1 else False,
            'res_amp': self.get_param('NMDA0306', frame=hk2_frame)[1],
            'set_pt': self.get_param('NMDA0245', frame=hk2_frame)[1],
            'fadj': self.get_param('NMDA0347', frame=hk2_frame)[1],
            'work_pt_per': self.get_param('NMDA0181', frame=hk2_frame)[1],
            'set_pt_per': self.get_param('NMDA0244', frame=hk2_frame)[1] })

        sw_flags = self.get_param('NMDA0349', frame=hk2_frame)[1]
        meta.update({
            'mag_phase': bool(sw_flags >> 13 & 0b1),
            'ctrl_image': bool(sw_flags >> 5 & 0b1) })

        meta = pd.Series(meta)

        src_file = os.path.basename(hk2.loc[hk2_frame].filename)
        meta['filename'] = src_file
        meta['scan_file'] = src_file_to_img_file(src_file, meta.start_time, meta.target)

        # calculated properties
        meta['x_step_nm'] = (meta['x_step'] * common.xycal['closed']) if meta['x_closed'] else (meta['x_step'] * common.xycal['open'])
        meta['y_step_nm'] = (meta['y_step'] * common.xycal['closed']) if meta['y_closed'] else (meta['y_step'] * common.xycal['open'])
        meta['xlen_um'] = meta.x_step_nm * meta.xsteps / 1.e3
        meta['ylen_um'] = meta.y_step_nm * meta.ysteps / 1.e3
        meta['z_ret_nm'] = meta.z_ret * common.zcal
        meta['work_pt'] = meta.res_amp * abs(meta.work_pt_per) / 100.
        meta['tip_offset'] = (meta.lin_pos - self.lin_centre_pos[meta.tip_num-1]) / common.linearcal
        meta['tip_offset'] = meta.tip_offset - common.tip_offset[meta.tip_num-1]
        meta['duration'] = meta.end_time - meta.start_time

        # 'x_orig_um' and 'y_orig_um' added by locate_scans()
        meta = locate_scans(meta).squeeze()


        return meta

#----- end of TM class


def load_exposures(exp_file=os.path.join(common.tlm_path, 'exposures.msg')):
    """Loads a msgpack file containing all exposures. If exp_file is not given, the
    default file and path are used"""

    return pd.read_msgpack(exp_file)


def build_pkt_index(files='TLM__MD_M*.DAT', tlm_dir=common.tlm_path, tm_index_file=os.path.join(common.tlm_path,'tlm_packet_index.hd5')):
    """Builds an HDF5 (pandas/PyTables) table of the packet index (tm.pkts). This can be used for
    on-disk queries to retrieve a selection of packets as input to ros_tm.tm()."""

    import glob
    from pandas import HDFStore

    try:
        store = HDFStore(tm_index_file, 'w', complevel=9, complib='blosc')
    except IOError as (errno, strerror):
        print "ERROR: I/O error({0}): {1}".format(errno, strerror)
        return None

    table = 'pkts'

    tm_files = sorted(glob.glob(os.path.join(tlm_dir,files)))

    longest_filename = len(max(tm_files, key=len))

    if len(tm_files)==0:
        log.error('no files matching pattern')
        return False

    for f in tm_files:
        telem = tm(f)
        # data_columns determines which columns can be queried - here use OBT for time slicing, APID for choosing data
        # source and filename to allow selection by MTP or STP.

        try:
            nrows = store.get_storer(table).nrows
        except:
            nrows = 0

        telem.pkts.index = pd.Series(telem.pkts.index) + nrows
        try:
            store.append(table, telem.pkts, format='table', min_itemsize={'filename': longest_filename}, data_columns=['obt','apid','filename'])
        except Exception as e:
            log.error('error appending to file: %s' % e)
            store.close()
            return None

    store.close()



def sample_slope(images, add_to_df=True):
    """Accepts one or more images returned by tm.get_images(), performs a least
    squares fit to the image data and returns the angle of this fit in X and Y."""

    indices = []
    x_deg = []
    y_deg = []

    if type(images) == pd.Series:
        images = pd.DataFrame(columns=images.to_dict().keys()).append(images)

    topo = images.query('channel=="ZS"')

    for idx, image in topo.iterrows():

        data = image['data']

        xvals = np.arange(data.shape[0]) * image.x_step_nm
        yvals = np.arange(data.shape[1]) * image.y_step_nm
        z = data.ravel() * common.zcal

        xs, ys = np.meshgrid(xvals, yvals)
        x = xs.ravel()
        y = ys.ravel()

        A = np.column_stack([x, y, np.ones_like(x)])
        abc, residuals, rank, s = np.linalg.lstsq(A, z)

        indices.append(idx)
        x_deg.append(np.rad2deg(np.arctan(abc[0])))
        y_deg.append(np.rad2deg(np.arctan(abc[1])))


    if add_to_df:
        images['x_deg'] = pd.Series(x_deg, index=indices)
        images['y_deg'] = pd.Series(y_deg, index=indices)
        return images
    else:
        return (indices, x_deg, y_deg)


def height_diff(images, mode='diff'):
    """Accepts an image produced by get_images() and compares height difference
    in the scanning direction.

    mode='diff': the height difference is returned
    mode='above': the height of pizels exceeding the retraction height is returned
    mode='bool': a boolean mask of pixels above the retraction is returned."""

    if type(images) == pd.Series:
        images = pd.DataFrame(columns=images.to_dict().keys()).append(images)

    if mode not in ['diff', 'above', 'bool']:
        log.error('mode must be one of: diff, above or bool!')
        return None

    topo_images = images[ images.channel=='ZS']

    if len(topo_images)==0:
        log.error('no topography channels present in images')
        return None

    for idx, image in topo_images.iterrows():

        data = image['data']

        # Transpose and flip according to main scan direction and H_L parameters
        if image.x_dir == 'H_L': data = np.fliplr(data)
        if image.y_dir == 'H_L': data = np.flipud(data)
        if image.fast_dir=='Y': data = np.transpose(data)

        # Loop through rows, checking for pixels with height diffs > z_ret
        num_rows = int(image.ysteps) if image.fast_dir=='X' else int(image.xsteps)

        if mode=='bool':
            mask = np.zeros_like(data, dtype=bool) # mask is a boolean array of bad pixels
        else:
            mask = np.zeros_like(data)

        for row in range(num_rows):
            line = data[row,:]
            if mode=='bool':
                mask[row,1:] = (line[1:]-line[:-1]) > image.z_ret
            elif mode=='diff':
                mask[row,1:] = (line[1:]-line[:-1])
            else:
                mask[row,1:] = (line[1:]-line[:-1]) - image.z_ret
                mask[ mask<0 ] = 0

        # Transpose and flip according to main scan direction and H_L parameters
        if image.fast_dir=='Y': mask = np.transpose(mask)
        if image.y_dir == 'H_L': mask = np.flipud(mask)
        if image.x_dir == 'H_L': mask = np.fliplr(mask)

        image['data'] = mask
        image.channel = 'RT'

        images.loc[images.index.max() + 1] = image

    return images.sort_values(by='start_time').reset_index(drop=True)


def css_write(html, filename, cssfile='midas.css'):
    """Accepts an html string (usually generated by pd.DataFrame.to_html() ) and
    adds a reference to a given CSS file"""

    cssfile = os.path.join(common.config_path,cssfile)
    css = open(cssfile, 'r').read()

    outfile = open(filename, 'w')
    outfile.write(css + '\n\n' + html)
    outfile.close()

    return


def find_param(query):
    """Searches the RMIB for a parameter description containing the query string."""

    results = pcf[pcf.description.str.contains(query, case=False).fillna(False)]
    results = pid.merge(plf.merge(results),on='spid')

    results.rename(columns={'description_x':'packet'}, inplace=True)
    results.rename(columns={'description_y':'parameter'}, inplace=True)

    return results[ ['apid', 'spid', 'packet', 'param_name', 'parameter']]



def simple_get_pkts(filename, dds_header=True, verbose=False):
    """Scans a TM file and returns OBT, packet type and sub-type, APID and offset.
    Minimal checking is performed, and the dds_header flag must be set correctly.

    use get_pkts() for a more robust method of TM extraction."""

    f = open(filename, 'rb')
    tm = bytearray(os.path.getsize(filename))
    f.readinto(tm)

    num_pkts = 0
    pkt_list = []
    offset = dds_header_len if dds_header else 0

    while(offset < len(tm)):

        pkt = {}
        pkt_header = pkt_header_names(*struct.unpack_from(pkt_header_fmt,tm,offset))

        # Calculate OBT - the format here uses 6 octets to represent OBT, 4 octects for the number
        # of seconds, and two octects for fractional seconds. The reference epoch is 01/01/2013 00:00:00.0

        # Check for out-of-sync packets - MIDAS telemetry packets are not time synchronised when the MSB
        # of the 32 bit coarse time (= seconds since reference date) is set to "1".
        pkt['tsync'] = not bool(pkt_header.obt_sec >> 31)

        delta_t = timedelta(seconds=pkt_header.obt_sec, milliseconds=(pkt_header.obt_frac * 2.**-16 * 1000.))

        # version = pkt_header.pkt_id  & 0b1110000000000000
        # pkt_type = pkt_header.pkt_id & 0b0001000000000000

        pkt['offset'] = offset
        pkt['type'] = pkt_header.pkt_type
        pkt['subtype'] = pkt_header.pkt_subtype
        pkt['apid'] = pkt_header.pkt_id     & 0b0000011111111111
        pkt['obt'] = obt_epoch + delta_t
        pkt['sid'], = struct.unpack_from('>H',tm,offset+pkt_header_size)
        pkt['length'] = pkt_header.pkt_len

        pkt_list.append(pkt)

        offset = offset + (pkt_header.pkt_len + 7 + dds_header_len) if dds_header else offset + (pkt_header.pkt_len + 7)
        num_pkts += 1

        if verbose: print 'Pkt %i length %i, type/subtype %i/%i and APID %i generated at OBT %s' % \
            (num_pkts, pkt_header.pkt_len, pkt_header.pkt_type, pkt_header.pkt_subtype, pkt['apid'], pkt['obt'].strftime('%m/%d/%Y %I:%M:%S %p'))

    print '%i packets read from file %s' % (num_pkts, filename)

    pkt_list= pd.DataFrame(pkt_list)

    apids = pkt_list.apid.unique()
    for apid in apids:
        if apid not in midas_apids:
            log.warning('APID %i not a MIDAS APID' % (apid))

    return pkt_list


def src_file_to_img_file(src_filename, start_time, target):
    mtp = src_filename[9:12]
    stp = src_filename[14:17]
    start = datetime.strftime(start_time,isofmt)
    filename = 'SCAN_MD_M%s_S%s_%s_TGT%02d' % (mtp,stp,start,target)
    return filename


def read_ccf(filename=False):
    """Reads the SCOS-2000 ccf.dat file"""

    if not filename:
        filename = os.path.join(common.s2k_path, 'ccf.dat')

    cols = ('name', 'description', 'type', 'subtype', 'apid')
    ccf = pd.read_table(filename,header=None,names=cols,usecols=[0,1,6,7,8])
    ccf = ccf.dropna()
    ccf.type = ccf.type.astype(np.int16)
    ccf.subtype = ccf.subtype.astype(np.int16)
    ccf.apid = ccf.apid.astype(np.int16)

    return ccf

def read_csf(filename=False):
    """Load the command sequence file (CSF)"""

    # Get the description of each sequence from the command sequence file (csf.dat)
    csf_file = os.path.join(common.s2k_path, 'csf.dat')
    cols = ('sequence', 'description', 'csf_plan')
    csf = pd.read_table(csf_file,header=None,names=cols,usecols=[0,1,7],na_filter=False)

    return csf


def read_csp(midas=False):
    """Load the command sequence parameters (CSP) file"""

    csp_file = os.path.join(common.s2k_path, 'csp.dat')
    cols = ('sequence', 'param', 'param_num','param_descrip','ptc','pfc','format', \
        'radix', 'param_type', 'val_type', 'default', 'cal_type', 'range_set', \
        'numeric_cal', 'text_cal', 'eng_unit')
    csp = pd.read_table(csp_file,header=None,names=cols,usecols=range(0,16),na_filter=True,index_col=False)
    if midas: csp = csp[csp.sequence.str.startswith('AMD')] # filter by MIDAS sequences
    csp = csp.sort_values( by=['sequence','param_num'] )

    return csp


def read_css(midas=False):
    """Load the command sequence definition (CSS) file"""

    css_file = os.path.join(common.s2k_path, 'css.dat')
    cols = ('sequence', 'description', 'entry', 'type', 'name', 'num_params', 'reltype','reltime','extime')

    css = pd.read_table(css_file,header=None,names=cols,usecols=[0,1,2,3,4,5,7,8,9],na_filter=True,index_col=False)
    if midas: css = css[css.sequence.str.startswith('AMD')] # filter by MIDAS sequences
    css = css.sort_values( by=['sequence','entry'] )

    return css


def read_sdf(midas=False):
    """Load the command sequence element parameters (SDF) file"""

    sdf_file = os.path.join(common.s2k_path, 'sdf.dat')
    cols = ('sequence', 'entry', 'el_name', 'param', 'val_type', 'value')

    sdf = pd.read_table(sdf_file,header=None,names=cols,usecols=[0,1,2,4,6,7],na_filter=True,index_col=False)
    if midas: sdf = sdf[sdf.sequence.str.startswith('AMD')] # filter by MIDAS sequences
    sdf = sdf.sort_values( by=['sequence','entry'] )

    return sdf





def read_pid(filename=False):
    """Reads the SCOS-2000 pid.dat file and parses it for TM packet data. This
    is used to validate and identify valid telemetry frames. If no filename
    is given the global S2K path is searched for the PID file."""

    if not filename:
        filename = os.path.join(common.s2k_path, 'pid.dat')

    # cols = ('type','subtype','apid','sid','p2val','spid','description','unit','tpsd','dfh_size','time','inter','valid')
    cols = ('type','subtype','apid','sid','spid','description')
    pid=pd.read_table(filename,header=None,names=cols,usecols=[0,1,2,3,5,6])

    # Database upgrade corresponding to 6.6.4 changed the packet type of even with SID 42777. In order that the old
    # packets are not rejected, I will duplicate the row and correct here.
    # pid = pid.append(pid[ (pid.sid==42777) & (pid.apid==1079) ], ignore_index=True)
    # pid.subtype.iloc[-1] = 1

    return pid

def read_pcf(filename=False):

    if not filename:
        filename = os.path.join(common.s2k_path, 'pcf.dat')

    cols = ('param_name', 'description', 'unit', 'ptc', 'pfc', 'width', 'cal_cat', 'cal_id','group')
    pcf = pd.read_table(filename,header=None,names=cols,usecols=[0,1,3,4,5,6,9,11,16], na_filter=True )
    # pcf = pd.to_numeric(pcf)

    return pcf


def read_plf(filename=False):

    if not filename:
        filename = os.path.join(common.s2k_path, 'plf.dat')

    cols = ('param_name','spid','byte_offset','bit_offset')
    plf = pd.read_table(filename,header=None,names=cols,usecols=[0,1,2,3])
    # plf = pd.to_numeric(plf)

    return plf


def read_caf(filename=False):
    """Read the CAF (numerical calibration curve) file"""

    if not filename:
        filename = os.path.join(common.s2k_path, 'caf.dat')

    cols = ('cal_id','cal_descrip','eng_fmt','raw_fmt','radix')
    caf = pd.read_table(filename,header=None,names=cols,usecols=[0,1,2,3,4])
    # caf = pd.to_numeric(caf)
    caf.cal_id = caf.cal_id.astype(int)

    return caf


def read_pic(filename=False):
    """Read the PIC (packet identification criteria) table"""

    if not filename:
        filename = os.path.join(common.s2k_path, 'pic.dat')

    cols = ('pkt_type','pkt_subtype','sid_offset','sid_width')
    pic = pd.read_table(filename,header=None,names=cols,usecols=[0,1,2,3])
    # pic = pd.to_numeric(pic)
    # caf.cal_id = caf.cal_id.astype(int)

    return pic



def read_cap(filename=False):
    """Read the CAP (numerical calibration curve definition) file"""

    if not filename:
        filename = os.path.join(common.s2k_path, 'cap.dat')

    cols = ('cal_id','raw_val','eng_val')
    cap = pd.read_table(filename,header=None,names=cols,usecols=[0,1,2])
    # cap = pd.to_numeric(cap)

    return cap

def read_mcf(filename=False):
    """Read the MCF (polynomical calibration curve defintion) file"""

    if not filename:
        filename = os.path.join(common.s2k_path, 'mcf.dat')

    cols = ('cal_id','cal_descrip','a0','a1','a2','a3','a4')
    mcf = pd.read_table(filename,header=None,names=cols)
    # mcf = pd.to_numeric(mcf)

    return mcf


def read_txf(filename=False):
    """Read the TXF (textual calibration) file"""

    if not filename:
        filename = os.path.join(common.s2k_path, 'txf.dat')

    cols = ('txf_id','txf_descr','raw_fmt')
    txf = pd.read_table(filename,header=None,names=cols,usecols=[0,1,2])
    # txf = pd.to_numeric(txf)

    return txf

def read_txp(filename=False):
    """Read the TXP (textual calibration definition) file"""

    import os

    if not filename:
        filename = os.path.join(common.s2k_path, 'txp.dat')

    cols = ('txp_id','from','to','alt')
    txp = pd.read_table(filename,header=None,names=cols)
    # txp = txp.convert_objects(convert_numeric=True)

    return txp


def read_scos():
    """Reads the relevant SCOS-2000 ASCII export tables and returns pd DataFrames
    with the useful fields"""

    plf = read_plf()
    pid = read_pid()
    pcf = read_pcf()
    caf = read_caf()
    cap = read_cap()
    txf = read_txf()
    txp = read_txp()
    mcf = read_mcf()
    ccf = read_ccf()
    csf = read_csf()
    csp = read_csp()
    css = read_css()
    pic = read_pic()
    sdf = read_sdf()

    return plf, pid, pic, pcf, caf, cap, txf, txp, mcf, ccf, csf, csp, css, sdf


# Load in the SCOS tables at module level so that they are always available
plf, pid, pic, pcf, caf, cap, txf, txp, mcf, ccf, csf, csp, css, sdf = read_scos()

# Set up some default lists
def search_tcs(search, midas=True):

    if midas:
        tcs = ccf[ccf.apid==1084]
    else:
        tcs = ccf

    return tcs[tcs.description.str.contains(search, case=False)]


def search_sqs(search, midas=True):

    if midas:
        sqs = csf[csf.sequence.str.startswith('AMD')]
    else:
        sqs = csf

    return sqs[sqs.description.str.contains(search, case=False)]


def search_params(search, midas=True):

    if midas:
        params = pcf[pcf.param_name.str.startswith('NMD')]
    else:
        params = pcf

    return params[params.description.str.contains(search, case=False, na=False)]


def search_param_name(search, midas=True):

    if midas:
        params = pcf[pcf.param_name.str.startswith('NMD')]
    else:
        params = pcf

    return params[params.param_name.str.contains(search, case=False, na=False)]


def strfdelta(tdelta, fmt):
    d = {"days": tdelta.days}
    d["hours"], rem = divmod(tdelta.seconds, 3600)
    d["minutes"], d["seconds"] = divmod(rem, 60)
    return fmt.format(**d)


def load_images(filename=None, data=False, sourcepath=common.tlm_path, topo_only=True, exclude_bad=True, manual=True):
    """Load a messagepack file containing all image meta-data.

    data=True/False - is only meta-data returned, or are binary images included?
    topo_only=True - only ZS channels will be returned
    exclude_bad=True - bad images described in bad_scans.txt will be removed
    manual=True - manual images will be appended, and de-duplication performed (on scan_file) """

    if data:

        if filename is None:
            filename = os.path.join(common.tlm_path, 'all_images_data.msg')
        f = open(filename, 'rb')
        import cPickle as pkl

        # objs = []
        # while 1:
        #     try:
        #         objs.append(pkl.load(f))
        #     except EOFError:
        #         break
        #
        # if len(objs)==0:
        #     log.error('file %s appears to be empty' % filename)
        #     return None
        #
        # images = pd.concat(iter(objs), axis=0)

        images = pd.read_msgpack(filename)

    else:
        if filename is None:
            filename = os.path.join(common.tlm_path, 'all_images.pkl')
        images = pd.read_pickle(filename)

    if exclude_bad:
        badlist = pd.read_table(os.path.join(common.config_path,'bad_scans.txt'), header=None)[0].tolist()
        images = images.query('scan_file!=@badlist')

    if manual:
        man_imgs = load_manual_scans()
        images = images.append(man_imgs, ignore_index=True)
        images.drop_duplicates(subset=['scan_file','channel'], keep='last', inplace=True)

    if topo_only:
        images = images[ images.channel=='ZS' ]

    images.sort_values(by='start_time', inplace=True)
    images.reset_index(inplace=True, drop=True)

    if sourcepath is not None:
        images.filename = images.filename.apply( lambda f: os.path.join(sourcepath, os.path.basename(f)) )

    log.info('%d images restored' % len(images))

    return images


def load_lines(filename=None, ignore_image=True):
    """Loads a line scan dataframe from disk. The options ignore_image and
    exppand_params are used as per get_line_scans() and simply define which
    of four data files are loaded. If filename is given, then this file is
    simply loaded instead."""


    import cPickle as pkl

    if filename is not None:
        fname = filename
    else:
        fname = 'lines.pkl'
        fname = os.path.join(common.tlm_path, fname)
    f = open(fname, 'rb')

    objs = []
    while 1:
        try:
            objs.append(pkl.load(f))
        except EOFError:
            break

    if len(objs)==0:
        log.error('file %s appears to be empty' % fname)
        return None

    lines = pd.concat(iter(objs), axis=0)

    if ignore_image:
        lines = lines[ ((~lines.in_image) & (lines.sw_ver<661)) |
            ( (lines.sw_ver>=661) & ~(lines.anti_creep) & ~(lines.in_image) ) ]

    lines.sort_values(by='obt', inplace=True)
    lines.reset_index(inplace=True, drop=True)

    log.info('%d lines restored' % len(lines))

    return lines



def load_manual_scans(filename=os.path.join(common.config_path, 'manual_images.pkl')):
    """Loads a dataframe containing manually (re-)packed image files, i.e. scans that
    required manual attention. These can be appended to or overwrite existing data"""

    # import cPickle as pkl
    #
    # f = open(filename, 'rb')
    #
    # objs = []
    # while 1:
    #     try:
    #         objs.append(pkl.load(f))
    #     except EOFError:
    #         break
    #
    # if len(objs)==0:
    #     log.error('file %s appears to be empty' % filename)
    #     return None
    #
    # images = pd.concat(iter(objs), axis=0)

    return pd.read_pickle(filename)



def read_timecorr(tcorr_file='TLM__MD_TIMECORR.DAT'):
    """Reads a Rosetta time correlation file and returns a dataframe of all coefficients"""

    # These files have only a DDS header and packet data field, no standard packet header!

    with open(tcorr_file,'rb') as f:
        tcorr = f.read()

    pkt_dfield_len = 30
    pkt_len = dds_header_len + pkt_dfield_len # = 48 bytes
    num_pkts = len(tcorr) / pkt_len

    timecorr = pd.DataFrame(columns=['validity', 'validity_s', 'gradient', 'offset', 'std_dev', 'gen_time'])

    for pkt in range(num_pkts):

        dds_header = dds_header_names(*struct.unpack_from(dds_header_fmt,tcorr,pkt_len*pkt))
        delta_t_s = dds_header.scet1 + dds_header.scet2/1.e6
        # delta_t = timedelta(seconds=dds_header.scet1, microseconds=dds_header.scet2)
        validity = dds_obt_epoch + timedelta(seconds=delta_t_s)
        validity = validity.replace(tzinfo=None)

        grad, offs, std, gent1, gent2 = struct.unpack('>3dIH',tcorr[pkt_len*pkt+dds_header_len:pkt_len*pkt+dds_header_len+pkt_dfield_len])
        gentime = obt_epoch + timedelta(seconds = gent1 + (gent2/2.**16))

        timecorr.loc[pkt] = pd.Series({'validity':validity, 'validity_s': delta_t_s, 'gradient':grad, 'offset':offs, 'std_dev':std, 'gen_time': gentime})

    timecorr = timecorr.set_index('validity')

    return timecorr


def wheel_aborts():
    """Finds wheel aborts and timeouts and reports the times, starting
    and ending segment to diagnose problems"""

    tlm = tm()

    # Find aborts and segment move timeouts

    tlm.query_index(what='events')
    wheel_sids = [
        42752,  # EvWheelMoveAborted
        42904 ] # EvSegSearchTimeout
    wheel_evts = tlm.pkts[ tlm.pkts.sid.isin(wheel_sids) ]
    wheel_evts = wheel_evts[ ['obt','sid', 'description']]

    # Find the previous wheel move start
    # 42591 = EvSearchForRefPulse
    start_obts = []
    start_sid = 42591
    for idx, evt in wheel_evts.iterrows():
        start_obts.append(tlm.pkts[ (tlm.pkts.sid==start_sid) & (tlm.pkts.obt < evt.obt) ].obt.iloc[-1])
    wheel_evts['start_obt'] = start_obts

    # Find the segment number before the move, and the value after seg search start
    # NMDA0196 = WheSegmentNum
    seg_from = []
    seg_to = []
    seg_param = 'NMDA0196'

    for idx, evt in wheel_evts.iterrows():
        tlm.query_index(what='hk', start=evt.obt-pd.Timedelta(hours=1), end=evt.obt+pd.Timedelta(hours=1))
        hk2 = tlm.pkts[ tlm.pkts.sid==2 ]

        frame = hk2[hk2.obt < evt.start_obt].index
        if len(frame)==0:
            log.warning('no HK2 frame found before wheel start at %s' % evt.start_obt)
            seg_from.append(None)
        else:
            frame = frame[-1]
            seg_from.append( tlm.get_param(seg_param, frame=frame)[1]) # WheSegmentNum

        frame = hk2[hk2.obt < evt.obt].index
        if len(frame)==0:
            log.warning('no HK2 frame found after wheel start at %s' % evt.start_obt)
            seg_to.append(None)
        else:
            frame = frame[-1]
            seg_to.append( tlm.get_param(seg_param, frame=frame)[1]) # WheSegmentNum

    wheel_evts['seg_from'] = seg_from
    wheel_evts['seg_to'] = seg_to

    return wheel_evts

def compare_modes(mtp, stp, case='P', axis=None, start=None, end=None, cmap=None):

    import eps_utils
    import matplotlib.cm as cm
    import matplotlib.dates as md

    mtp_folder = os.path.expanduser('~/Dropbox/work/midas/operations/MTP%03i%c/' % (mtp,case.upper()))
    eps_folder = os.path.join(mtp_folder, 'eps')

    # Use the EPS modes.out file to produce a list of mode transitions that
    # can be plot on the chart if requested.

    modes = eps_utils.read_modes(eps_folder, expand_modes=True)

    times = []
    mid_modes = []
    curr_mode = None

    for idx, (time, mode)  in enumerate(modes.iteritems()):
        if mode==curr_mode:
            continue
        curr_mode = mode
        mid_modes.append(mode)
        times.append(time)

    modes = pd.Series(index=times, data=mid_modes, name='mode')

    num_modes = len(modes.unique())

    # make a colour index with the correct number of colors, spanning the colourmap
    if cmap is None:
        colours = cm.get_cmap('gist_rainbow')
    else:
        colours = cm.get_cmap(cmap)

    colour_list = [colours(1.*i/num_modes) for i in range(num_modes+2)]

    if axis is None:
        fig, ax = plt.subplots()
    else:
        ax = axis
        fig = ax.figure

    blocks = []
    used = []

    # fig.autofmt_xdate()
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    ax.xaxis_date()


    for idx, (time, mode) in enumerate(modes.iloc[:-1].iteritems()):
        colour = colour_list[modes.unique().tolist().index(mode)]
        spam = ax.axvspan(modes.index[idx], modes.index[idx+1], facecolor=colour, alpha=0.7)
        if mode not in used:
            used.append(mode)
            blocks.append(spam)

    ax.grid(True)

    if start is None:
        start = modes.index.min()
    if end is None:
        end = modes.index.max()

    ax.set_xlim(start, end)
    plt.setp(ax.get_xticklabels(), rotation=45)
    ax.get_yaxis().set_ticks([])

    pt_legend = ax.legend(labels=used, handles=blocks, loc=0, fancybox=True)
    pt_legend.get_frame().set_alpha(0.7)

    if axis is None:
        plt.show()

    return modes



def compare_power(mtp, stp, case='P', ax=None, show_events=None, start=None, end=None):
    """Accepts an STP, loads the corresponding MTP EPS run (if it doesn't exist, the EPS
    is executed first) and the s/c TLM file to compare predicted and actual power usage.

    If show_events=True then *actual* events are overplotted."""

    import eps_utils, glob
    import matplotlib.transforms as transforms
    import matplotlib.dates as md


    mtp_folder = os.path.expanduser('~/Dropbox/work/midas/operations/MTP%03i%c/' % (mtp,case.upper()))
    eps_folder = os.path.join(mtp_folder, 'eps')
    power, data = eps_utils.read_output(eps_folder, ng=True)

    sc_files = glob.glob(os.path.join(common.tlm_path, 'TLM__SC_*%03d*.DAT' % stp))
    sc_tm = tm(sc_files)

    if show_events is not None:

        md_files = glob.glob(os.path.join(common.tlm_path, 'TLM__MD_*%03d*.DAT' % stp))
        md_tm = tm(md_files)

        events = md_tm.get_events(ignore_giada=True, verbose=False)

    lcl_curr = sc_tm.get_param('NPWDA548')
    real_pwr = 28.*lcl_curr

    if start is None:
        start = max( power.index.min(), real_pwr.index.min() )
    if end is None:
        end = min( power.index.max(), real_pwr.index.max() )

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    fig.autofmt_xdate()
    xfmt = md.DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    # ax.xaxis_date()

    real_line = ax.plot( real_pwr.index, real_pwr, 'b-', label='real power' )
    predict_line = ax.plot( power.index, power.MIDAS, 'r-', label='predicted power' )

    ax.set_ylabel('Power consumption (W)')
    ax.grid(True)
    ax.set_xlim(start, end)

    plt.setp(ax.get_xticklabels(), rotation=45)
    leg = ax.legend(loc=0, prop={'size':10}, fancybox=True)

    if show_events is not None:

        events = md_tm.get_events(ignore_giada=True, verbose=False)

        if show_events=='scan':
            # 42656 - EvFullScanStarted
            # 42756 - EvFullScanAborted
            # 42513 - EvScanFinished
            # 42713 - EvScanAborted
            scan_events = [42656, 42756, 42513, 42713]
            events = events[events.sid.isin(scan_events)]

        elif show_events=='wheel':
            # 42904 - EvSegSearchTimeout
            # 42592 - EvSegmentFound
            # 42591 - EvSearchForRefPulse
            scan_events = [42904, 42592, 42591]
            events = events[events.sid.isin(scan_events)]

        elif show_events=='fscan':
            # 42641 - EvFScanStarted
            # 42645 - EvAutoFScanFinshed
            scan_events = [42641, 42645]
            events = events[events.sid.isin(scan_events)]

        elif show_events=='lines':
            # 42655 - EvLineScanStarted
            # 42611 - EvLineScanFinished
            scan_events = [42655, 42611]
            events = events[events.sid.isin(scan_events)]

        elif show_events=='approach':
            # 42662 - EvApproachStarted
            # 42664 - EvApproachFinished
            # 42762 - EvApproachAborted
            # 42764 - EvAppContact
            # 42765 - EvAppError
            # 42766 - EvApproachStuck
            # 42906 - EvApproachTimeout
            # 42916 - EvMoveAbortedApp
            # 42917 - EvAppAbortedLin
            # 42623	- EvSurfaceFound
            # 42665	- EvZpiezoFineAdj
            app_events = [42662, 42664, 42762, 42764, 42765, 42766, 42906, 42916, 42917, 42623, 42665 ]
            events = events[events.sid.isin(app_events)]
        elif show_events=='all':
            # Ignore events that don't bring much diagonstic info or flood the plot
            ignore_sids = [ \
                42501, # TC accept
                42701, # TC reject
                42611, # EvLineScanFinished
                42512, # EvScanProgress
                42642, # EvFScanCycleStarted
                42643, # EvFScanCycleFinished
                42699] # KernelHello
            events = events[-events.sid.isin(ignore_sids)]
        else:
            log.warning('invalid event label, ignoring')
            events = pd.DataFrame()

        # Fixing mark/MIDAS#2 - filter events to given time range first
        events = events[ (events.obt>start) & (events.obt<end) ]

        for idx, event in events.iterrows():
            ax.axvline(event.obt,color='r')
            trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(event.obt,0.9, event.event,rotation=90, transform=trans, clip_on=True)

    fig.tight_layout()

    plt.show()

    return real_pwr, power.MIDAS



if __name__ == "__main__":

    log.warning('this module cannot be called interactively')
