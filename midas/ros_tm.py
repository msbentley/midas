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


# datefmt='%m/%d/%Y %I:%M:%S %p'
isofmt = '%Y-%m-%dT%H%M%SZ'

obt_epoch = datetime(year=2003, month=1, day=1, hour=0, minute=0, second=0) # , tzinfo=pytz.UTC)
dds_obt_epoch = datetime(year=1970, month=1, day=1, hour=0, minute=0, second=0, tzinfo=pytz.UTC)

# File path config
s2k_path = os.path.join(common.ros_sgs_path,'PLANNING/RMOC/FCT/RMIB/ORIGINAL')
gwy_settings_file = os.path.expanduser('~/Dropbox/work/midas/operations/gwy-settings.gwy')
gwy_path = os.path.expanduser('~/Copy/midas/data/images/gwy')
css_path = os.path.expanduser('~/Dropbox/work/midas/software/python')

debug = False

# Define struct types for various packet formats, to allow easy extraction of values from the
# binary data into named tuples

pkt_header_fmt = '>3HIH4B'
pkt_header_size = struct.calcsize(pkt_header_fmt)
pkt_header_names = collections.namedtuple("pkt_header_names","pkt_id pkt_seq pkt_len obt_sec obt_frac \
    checksum pkt_type pkt_subtype pad")

dds_header_fmt = '>3I2H2B'
dds_header_len = struct.calcsize(dds_header_fmt)
dds_header_names = collections.namedtuple("dds_header_names", "scet1 scet2 pkt_length gnd_id vc_id sle time_qual")

# List of MIDAS specific APIDs
midas_apids = [1073, 1076, 1079, 1081, 1083, 1084]

#------------- Helper functions

def printtable(df):
    """Accepts a pd.DataFrame() prints a pretty-printed table, rendered with PrettyTable"""

    from prettytable import PrettyTable
    table = PrettyTable(list(df.columns))
    for row in df.itertuples():
            table.add_row(row[1:])
    print(table)
    return

def obt_to_datetime(obt):
        # isofmt = "%Y-%m-%dT%H%M%SZ"
    time = obt_epoch + timedelta(seconds=obt)
    return time

def obt_to_iso(obt):
    time = obt_epoch + timedelta(seconds=obt)
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
            print('ERROR: no numerical calcurve found with ID' % (param.cal_id))
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


def plot_line_scans(lines, units='real', label=None):
    """Plot one or more line scans"""

    if type(lines) == pd.Series:
        lines = pd.DataFrame(columns=lines.to_dict().keys()).append(lines)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    for idx, line in lines.iterrows():

        if label in lines.columns:
            lab = line['%s'%label]

        if units=='real':

            ax.set_xlabel(' %c distance (microns)' % line.fast_dir.upper())
            ax.set_ylabel('Height (nm)')

            height = (line['data'] - line['data'].min()) * common.cal_factors[0]
            distance = (line.step_size*common.xycal['open']/1000.)*np.arange(line.num_steps)

        elif units=='dac':

            ax.set_xlabel(' %c distance (DAC)' % line.fast_dir.upper())
            ax.set_ylabel('Height (DAC)')

            height = line['data']

            # TODO get open/closed loop status (not in packet header, get from HK)
            distance = (line.step_size*common.xycal['open']/1000.)*np.arange(line.num_steps)

        ax.grid(True)
        ax.plot(distance, height, label=lab)

    leg = ax.legend(loc=0, prop={'size':10}, fancybox=True, title=label)
    leg.get_frame().set_alpha(0.7)

    return


def plot_fscan(fscans, showfit=False, legend=True, cantilever=None, xmin=False, xmax=False, ymin=False, ymax=False):
    """Plots one or more frequency scan (read previously with get_freq_scans()). Optionally
    plot a Lorentzian fit"""

    if type(fscans) == pd.Series:
        fscans = pd.DataFrame(columns=fscans.to_dict().keys()).append(fscans)

    if cantilever is not None:
        fscans = fscans[ fscans.cant_num==cantilever ]

    if len(fscans)==0:
        print('ERROR: no frequency scans available (for selected cantilever)')
        return None

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    ax.set_xlabel('Frequency (Hz)')
    ax.grid(True)

    for idx, scan in fscans.iterrows():

        if scan.is_phase:
            ax.set_ylabel('Phase (deg)')
        else:
            ax.set_ylabel('Amplitude (V)')

        ax.plot(scan['frequency'],scan['amplitude'], label='%s' % scan.start_time)

        if showfit and ~scan.is_phase:
            if not 'offset' in scan.index:
                print('WARNING: no fit data in current frequency scan')
            else:
                ax.plot(scan.frequency, lorentzian(scan.frequency, scan.offset, scan.fit_max-scan.offset, \
                        scan.res_freq, scan.half_width),label='Lorentzian fit')

    if len(fscans)==1:
        ax.set_title('Ex/Gn: %i/%i, Freq start/step: %3.2f/%3.2f, Peak amp %3.2f V @ %3.2f Hz' % \
            (scan.excitation, scan.gain, scan.freq_start, scan.freq_step, scan.max_amp, scan.max_freq),
            fontsize=12 )

        if set(['res_amp','work_pt', 'set_pt', 'fadj']).issubset(set(scan.keys())) and not scan.is_phase:
            # Also drawn lines showing the working point and set point
            ax.axhline(scan.res_amp,color='b')
            ax.axhline(scan.work_pt,color='r')
            ax.axhline(scan.set_pt,color='g')
            ax.axhline(scan.fadj,color='g', ls='--')

    # ax.set_ylim(0)
    if legend:
        leg = ax.legend(loc=0, prop={'size':10}, fancybox=True)
        leg.get_frame().set_alpha(0.7)

    if not xmin: xmin = ax.get_xlim()[0]
    if not xmax: xmax = ax.get_xlim()[1]
    if not ymin: ymin = ax.get_ylim()[0]
    if not ymax: ymax = ax.get_ylim()[1]

    ax.set_xlim(xmin,xmax)
    ax.set_ylim(ymin,ymax)

    return

def plot_ctrl_data(ctrl):
    """Accepts a single control data entry and plots all four channels"""

    xdata = range(0,ctrl.num_meas/4)

    ctrl_fig = plt.figure()
    ax4 = ctrl_fig.add_subplot(4, 1, 4)
    ax3 = ctrl_fig.add_subplot(4, 1, 3, sharex=ax4)
    ax2 = ctrl_fig.add_subplot(4, 1, 2, sharex=ax4)
    ax1 = ctrl_fig.add_subplot(4, 1, 1, sharex=ax4)

    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), visible=False)

    line_ac, = ax1.plot(xdata, ctrl['ac'],'x-')
    line_dc, = ax2.plot(xdata, ctrl['dc'],'x-')
    line_phase, = ax3.plot(xdata,ctrl['phase'],'x-')
    line_zpos, = ax4.plot(xdata,ctrl['zpos'],'x-')

    ax1.set_yticks(ax1.get_yticks()[1:-1])
    ax2.set_yticks(ax2.get_yticks()[1:-1])
    ax3.set_yticks(ax3.get_yticks()[1:-1])
    ax4.set_yticks(ax4.get_yticks()[1:-1])

    ax1.set_ylabel('Cant AC (V)')
    ax2.set_ylabel('Cant DC (V)')
    ax3.set_ylabel('Phase (deg)')
    ax4.set_ylabel('Z position (nm)')

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
    plt.suptitle('Wheel seg %i, tip #%i, origin (%i,%i), step %i/%i' % ( \
        ctrl.wheel_posn, ctrl.tip_num, ctrl.x_orig, ctrl.y_orig, ctrl.main_cnt, ctrl.num_steps))

    plt.show()


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



def combine_linescans(linescans, bcr=False):
    """Accepts a DataFrame containing linescans and simply returns a numpy array of the lines
    stacked into an array, ordered as given.

    If bcr= is set to a filename, a minimal BCR file is produced - note that the
    line data are assumed to be in the X direction, and the Y step is assumed
    to be the same as the X!

    Note also that open loop operation is assumed!"""

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
        bcrdata['xlength'] = linescans.irow(0).step_size*common.xycal['open']*bcrdata['xpixels']
        bcrdata['ylength'] = linescans.irow(0).step_size*common.xycal['open']*bcrdata['ypixels']
        bcrdata['data'] = image.flatten()
        bcrdata['data'] = 32767 - bcrdata['data']
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
        print('ERROR: image data not found - be sure to run tm.get_images with info_only=False')
        return False

    images = images.sort( ['filename', 'start_time'] )
    last_filename = images.filename.irow(0)

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

        xoffset, yoffset = bcrutils.set_origin(image.x_orig,image.y_orig)

        bcrdata['xpixels'] = image.xsteps
        bcrdata['ypixels'] = image.ysteps
        bcrdata['xlength'] = image.x_step * xcal * bcrdata['xpixels']
        bcrdata['xoffset'] = xoffset
        bcrdata['ylength'] = image.y_step * ycal * bcrdata['ypixels']
        bcrdata['yoffset'] = yoffset
        bcrdata['data'] = images.data.iloc[idx].ravel()

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


def save_gwy(images, outputdir='.', save_png=False, pngdir='.'):
    """Accepts one or more sets of image data from get_images() and returns individual
    GWY files, with multiple channels combined in single files and all meta-data.

    If save_png=True and the image contains a topographic channel, a bitmap will be
    produced after a polynominal plane subtraction and saved to pngdir."""

    import common, gwy, gwyutils, math

    if images is None: return None

    if 'data' not in images.columns:
        print('ERROR: image data not found - be sure to run tm.get_images with info_only=False')
        return None

    scan_count = 0
    first_time = True
    filenames = []

    # tidy up the input data
    if type(images) == pd.Series: images = pd.DataFrame(columns=images.to_dict().keys()).append(images)
    if type(images)==bool:
        if not images: return

    last_filename = images.filename.irow(0)
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

            cal_factor = common.cal_factors[chan_idx]
            offset = int(round(common.offsets[chan_idx] / common.cal_factors[chan_idx]))
            z_unit, z_power = gwy.gwy_si_unit_new_parse(common.units[chan_idx])

            c.set_string_by_name('/%i/data/title' % (chan_idx),common.channel_names[chan_idx])
            datafield = gwy.DataField(channel.xsteps, channel.ysteps, xlen, ylen, True)
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
                m[a==32767] = 1.
                m[a==-32768] = 1.
                # if there are bad pixels, add the max - otherwise don't
                if len(m[m==1.])>0:
                    c.set_object_by_name('/%i/mask' % (chan_idx), mask)

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
                meta_channel.to_frame(name=os.path.basename(filename)).to_csv(meta_file, index=True, sep=':', line_terminator='<br>\n', index_label=False)
                gwy.gwy_app_data_browser_remove(c)

            else:
                print('WARNING: image contains no topography channel, no PNG produced')

    print('INFO: written %i Gwyddion files to directory %s' % (scan_count, os.path.abspath(outputdir)))

    return filenames


def open_gwy(images, path=gwy_path):
    """Accepts one or more images and loads the corresponding files into Gwyddion. If
    path= is not set, the default image path is used."""

    import subprocess

    if type(images) == pd.Series:
        images = pd.DataFrame(columns=images.to_dict().keys()).append(images)

    gwyfiles = images.scan_file.unique().tolist()
    gwyfiles = [os.path.join(path,gwy+'.gwy') for gwy in gwyfiles]
    command_string = ['gwyddion', '--remote-new'] + gwyfiles

    subprocess.Popen(command_string)


def save_bcr(images, outputdir='.', write_meta=False):
    """Save a set of images as BCRs. If write_meta=True then a matching
    txt file is written with all image meta-data"""

    import bcrutils

    if images is None: return None

    if 'data' not in images.columns:
        print('ERROR: image data not found - be sure to run tm.get_images with info_only=False')
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

    print('INFO: written %i BCR files to directory %s' % (len(images), os.path.abspath(outputdir)))

    return


def save_png(images, outputdir='.'):
    """Save a set of png images produced from images"""

    import bcrutils
    bcrs=to_bcr(images, outputdir)
    if not bcrs: return False
    if type(bcrs) is not list: bcrs = [bcrs]
    [bcrutils.plot2d(bcr,writefile=True) for bcr in bcrs]


# def show_images(images, planesub=False, realunits=True):
#     """Accepts a list of images returned by get_images(), converts to BCR
#     on the fly, performs and plane subtraction and displays with bcrutils"""
#
#     import bcrutils
#     bcrs=to_bcr(images)
#     if not bcrs: return False
#     if type(bcrs) != list: bcrs=[bcrs]
#     if planesub:
#         bcrs = [bcrutils.planesub(bcr) for bcr in bcrs]
#     [bcrutils.plot2d(bcr, realunits=realunits) for bcr in bcrs]



def do_planesub(image):
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


def do_polysub(image, order=3):
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

    xvals = np.arange(data.shape[0])
    yvals = np.arange(data.shape[1])
    xs, ys = np.meshgrid(xvals, yvals)

    x = xs.ravel()
    y = ys.ravel()
    z = data.ravel()

    m = polyfit2d(x,y,z,order=order)
    z = polyval2d(x, y, m)
    newdata = data - z.reshape((data.shape[0],data.shape[1]))
    image['data'] = (newdata - newdata.min())
    return image


def show(images, units='real', planesub='poly', title=True, fig=None, ax=None, shade=False, show_fscans=False):
    """Accepts one or more images from get_images() and plots them in 2D.

    units= can be 'real', 'dac' or 'pix'
    planesub= can be 'plane', 'poly'  or 'median'
    placesub=True will peform a least-square plane subtraction
    shade=True will add surface illumination and shading
    show_fscans=True marks lines where frequency re-tuning has occured (note - slow!)"""

    import matplotlib.cm as cm
    from matplotlib.colors import LightSource

    if type(images) == pd.Series:
        images = pd.DataFrame(columns=images.to_dict().keys()).append(images)

    if 'data' not in images.columns:
        print('ERROR: image data not found - be sure to run tm.get_images with info_only=False')
        return False

    unit_types = ['real', 'dac', 'pix']
    units = units.lower()
    if units not in unit_types:
        print('ERROR: unit type %s invalid' % units.lower() + " must be one of " + ", ".join(unit_types))
        return None

    if planesub is not None:
        planetypes = ['plane', 'poly']
        planesub = planesub.lower()
        if planesub not in planetypes:
            print('ERROR: planesub type %s invalid' % planesub.lower() + " must be one of " + ", ".join(planetypes))
            return None

    if show_fscans: # for now just use pix, then identifying the line of the fscan is straightforward
        units = 'pix'

    ls = LightSource(azdeg=0,altdeg=65)
    cmap = cm.afmhot

    for idx, image in images.iterrows():

        if planesub=='plane':
            image = do_planesub(image)
        elif planesub=='poly':
            image = do_polysub(image)

        chan_idx = common.data_channels.index(image.channel)
        unit = common.units[chan_idx]
        target = common.seg_to_facet(image.wheel_pos)

        if fig is None or idx>0: fig = plt.figure()
        if ax is None or idx>0: ax = fig.add_subplot(1,1,1)

        data = image['data']

        if units == 'real':
            data = (data - data.min()) * common.cal_factors[chan_idx]
            plot1 = ax.imshow(data, origin='upper', interpolation='nearest', extent=[0,image.xlen_um,0,image.ylen_um], cmap=cmap)
            ax.set_xlabel('X (microns)')
            ax.set_ylabel('Y (microns)')

        elif units == 'dac':
            xstart = image.x_orig
            xstop = image.x_orig + image.xsteps * image.x_step
            ystart = image.y_orig
            ystop = image.y_orig + image.ysteps * image.y_step
            plot1 = ax.imshow(data, origin='upper', interpolation='nearest', extent=[xstart,xstop,ystop,ystart], cmap=cmap)

        elif units == 'pix':
            plot1 = ax.imshow(data, origin='upper', interpolation='nearest', cmap=cmap)
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            data = image['data']

        ax.grid(True)

        if shade:
            data = ls.shade(data,cmap)

        if not shade:
            cbar = fig.colorbar(plot1, ax=ax) # Now plot using a colourbar
            if units=='real':
                cbar.set_label(unit, rotation=90)

        if title:
            ax.set_title(image.scan_file, fontsize=12)

        # If show_fscans is True, index the appropriate TLM file, get the OBTs of frequency scans
        # between the start and end times, find the line number of these times in HK and then
        # plot as arrows on the axis...
        if show_fscans:

            telem = tm(image.filename)
            fscans = telem.get_freq_scans(info_only=True)
            fscans = fscans[ (fscans.start_time > image.start_time) & (fscans.start_time < image.end_time) ]

            hk2 = telem.pkts[ (telem.pkts.type==3) & (telem.pkts.subtype==25) & (telem.pkts.apid==1076) & (telem.pkts.sid==2) ]

            if len(fscans) > 0:
                print('INFO: %i frequency re-tunes occured during the image at OBT %s' % (len(fscans), image.start_time))
                obts = fscans.start_time.values
                for obt in obts:

                    frame = hk2[hk2.obt>obt].index
                    if len(frame)==0:
                        print('WARNING: no HK2 frame found after frequency scan at %s' % start)
                        continue
                    else:
                        frame = frame[0]

                    line = telem.get_param('NMDA0165', frame=frame)[1]

                    xstart, xstop = ax.get_xlim()
                    arrow_delta = (xstop-xstart)*0.025
                    ax.arrow(xstop+arrow_delta*2, line, -arrow_delta, 0, head_width=4, head_length=arrow_delta, fc='k', ec='k', clip_on=False)

    return fig, ax



def locate_scans(images, facet=None, segment=None, tip=None, show=False):
    """Accepts a list of scans returned by get_images() and plots the positions of the scans
    as annotated rectangles. If images only contains one facet/segment, that is used.

    If facet= is set and there is only one segment available, this
    is used - otherwise segment= must be set also. tip= can be used to
    filter scans by a given tip.

    The origin of each scan, relative to the wheel/target centre, are also
    added to the images DataFrame and returned."""

    if tip is not None:
        images = images[ images.tip_num == tip ]

    if segment is not None:
        images = images[ images.wheel_pos==segment ]

    if facet is not None:
        images = images[ images.target==facet ]

    if len(images)==0:
        print('ERROR: no matching images for the given facet and segment')
        return None

    title = ''
    if len(images.tip_num.unique())==1:
        title = title.join('Tip %i, ' % images.tip_num.unique()[0] )

    x_orig_um = []; y_orig_um = []

    for idx, scan in images.iterrows():

        # Calculate offset from centre of table in X and Y
        x_centre =  common.centre_closed if scan.x_closed else common.centre_open
        x_cal = common.xycal['closed'] if scan.x_closed else common.xycal['open']
        x_cal /= 1000.
        x_offset = (scan.x_orig - x_centre) * x_cal

        y_centre = common.centre_closed if scan.y_closed else common.centre_open
        y_cal = common.xycal['closed'] if scan.y_closed else common.xycal['open']
        y_cal /= 1000.
        y_offset = (scan.y_orig - y_centre) * y_cal

        # Take the cantilever and linear stage position into account for X position
        left = ( (scan.lin_pos-common.lin_centre_pos[scan.tip_num-1]) / common.linearcal ) + x_offset
        x_orig_um.append(left)

        # Y position in this stripe is simple related to the offset from the Y origin
        bottom = y_offset
        y_orig_um.append(bottom)

    images['x_orig_um'] = x_orig_um
    images['y_orig_um'] = y_orig_um

    if not show:
        return images
    else:

        if len(images.wheel_pos.unique())>1:
            print('ERROR: more than one segment specified - filter images or use keyword segment=')
            return None

        from matplotlib.patches import Rectangle

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.invert_yaxis()

        # Plot open and closed loop scans in different colours
        opencolor='red'
        closedcolor='blue'

        title += ('Target %i (%s), segment %i' % (images.target.unique()[0], images.target_type.unique()[0], images.wheel_pos.unique()[0]))
        ax.set_title(title)
        ax.set_xlabel('Offset from wheel centre (microns)')
        ax.set_ylabel('Offset from segment centre (microns)')

        for idx, scan in images.iterrows():
            edgecolor=closedcolor if scan.y_closed else opencolor
            ax.add_patch(Rectangle((scan.x_orig_um, scan.y_orig_um), scan.xlen_um, scan.ylen_um, fill=False, linewidth=1, edgecolor=edgecolor))

        # Make sure we plpot fix a fixed aspect ratio!
        ax.autoscale(enable=True)
        ax.set_aspect('equal')
        plt.show()

    return


def locate(pattern, root_path):
    """Returns a generator using os.walk and fnmatch to recursively
    match files with pattern under root_path"""

    import fnmatch

    for path, dirs, files in os.walk(os.path.abspath(root_path)):
        for filename in fnmatch.filter(files, pattern):
            yield os.path.join(path, filename)


def select_files(wildcard, directory='.', recursive=False):
    """Create a file list from a directory and wildcard - recusively if
    recursive=True"""

    # recursive search
    # result = [os.path.join(dp, f) for dp, dn, filenames in os.walk('.') for f in filenames if os.path.splitext(f)[1] == '.DAT']

    if recursive:
        selectfiles = locate(wildcard, directory)
        filelist = [file for file in selectfiles]
    else:
        import glob
        filelist = glob.glob(os.path.join(directory,wildcard))

    filelist.sort()

    return filelist


class tm:

    def __init__(self, files=None, directory='.', recursive=False, pkts=False, apid=False, dedupe=False, simple=True, dds_header=True, sftp=False):

        self.sftp = False
        self.pkts = None
        if files: self.get_pkts(files=files, directory=directory, recursive=recursive, apid=apid, dedupe=dedupe, simple=simple, dds_header=dds_header, sftp=sftp)


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

        self.pkts.sort('obt', inplace=True)
        print('INFO: packet index restored with %i packets' % (len(self.pkts)))


    def query_index(self, filename=os.path.join(common.tlm_path, 'tlm_packet_index.hd5'),
        start=None, end=None, stp=None, what='all', sourcepath=os.path.expanduser('~/Copy/midas/data/tlm')):
        """Restores a TLM packet index from filename. The entire file is read if no other options are given, otherwise
        filters can be applied:

        start=, end= accept any sane string date/time format
        stp= accepts an integer STP number
        what= can be 'all', 'hk', 'events' or 'science' and filters packets by APID"""

        what_types = {
            'hk': 1076,
            'events': 1079,
            'science': 1084 }

        what = what.lower()

        if (what != 'all') and (what not in what_types.keys()):
            print('WARNING: what= must be set to all, hk, events or science. Defaulting to all.')
            what = 'all'

        if type(start)==str:
            start = pd.Timestamp(start)

        if type(end)==str:
            end = pd.Timestamp(end)

        if stp is not None and type(stp)!=list:
            stp = [stp]

        table = 'pkts'
        store = pd.HDFStore(filename, 'r')

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
                print('WARNING: no packets match the criteria!')
                return

            self.pkts = store.select(table, where=list(selected))

        self.pkts.sort('obt', inplace=True)

        if sourcepath is not None:
            self.pkts.filename = self.pkts.filename.apply( lambda f: os.path.join(sourcepath, os.path.basename(f)) )

        # reset the index to a monotonic increasing function, needed for selecting frames later
        self.pkts.index=range(len(self.pkts))
        print('INFO: packet index restored with %i packets' % (len(self.pkts)))

        store.close()

        return


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
            selectfiles = locate(files, directory)
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

            print('INFO: indexing TLM file %s' % fl)

            # Scan the file for unique bit pattern to identify the start of the TM packets
            if simple:
                offsets = self.simple_locate_pkts(fl, dds_header=dds_header)
            else:
                offsets = self.locate_pkts(fl, apid=apid, sftp=sftp)

                if len(offsets) == 0: # no packets found
                    print('WARNING: no packets found for given APID, trying fallback method...')
                    offsets = self.simple_locate_pkts(fl)

            if len(offsets) == 0:
                print('ERROR: no packets found in file %s' % fl)
                continue

            # Re-open the file and read into a bytearray
            if sftp:
                f = sftp.sftp.open(fl, 'rb')
            else:
                f = open(fl, 'rb')
            tm = f.read()
            f.close()

            # using numpy instead of pandas here for speed
            picfile = os.path.join(s2k_path, 'pic.dat')
            pic = np.loadtxt(picfile,dtype=np.int32)

            # Using the byte offsets, unpack each packet header
            for idx,offset in enumerate(offsets):

                pkt = {}
                pkt_header = pkt_header_names(*struct.unpack_from(pkt_header_fmt,tm,offset))
                delta_t = timedelta(seconds=pkt_header.obt_sec, milliseconds=(pkt_header.obt_frac * 2.**-16 * 1000.))

                # Check for out-of-sync packets - MIDAS telemetry packets are not time synchronised when the MSB
                # of the 32 bit coarse time (= seconds since reference date) is set to "1".
                pkt['tsync'] = not bool(pkt_header.obt_sec >> 31)

                pkt['offset'] = offset
                pkt['type'] = pkt_header.pkt_type
                pkt['length'] = pkt_header.pkt_len
                pkt['subtype'] = pkt_header.pkt_subtype
                pkt['apid'] = pkt_header.pkt_id & 0x7FF
                pkt['seq'] = pkt_header.pkt_seq & 0x3FFF
                pkt['obt'] = obt_epoch + delta_t
                pkt['filename'] = os.path.abspath(fl)

                # print('DEBUG: pkt %i/%i has declared length %i, real len %i' % (pkt_header.pkt_type,pkt_header.pkt_subtype,pkt_header.pkt_len,offsets[idx+1]-offsets[idx]))

                if pkt_header.pkt_len==0:
                    # print('WARNING: packet with type (%i,%i) has zero length, but real length %i at byte offset %i' % (pkt_header.pkt_type,pkt_header.pkt_subtype,offsets[idx+1]-offsets[idx], offset))
                    # print('WARNING: APID = %i' % (pkt_header.pkt_id & 0x7FF))
                    continue

                # 07/07/2014: now using the pic.dat table to look up the location of p1val
                # 29/08/2014: FIXME - just profiled the code and this causes a MASSIVE slowdown
                #
                # pkt_pic = pic[ (pic.pkt_type==pkt_header.pkt_type) & (pic.pkt_subtype==pkt_header.pkt_subtype)].squeeze()
                pkt_pic = pic[ np.logical_and(pic[:,0]==pkt_header.pkt_type,pic[:,1]==pkt_header.pkt_subtype) ]

                if len(pkt_pic)==0:
                    print('ERROR: packet type (%i,%i) not found in the PIC' % (pkt_header.pkt_type,pkt_header.pkt_subtype))
                    if debug: print('DEBUG: OBT: %s, APID: %i, SEQ: %i' % (pkt.obt, pkt.apid, pkt.seq))
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
                        print('ERROR: p1val location cannot be identified for this packet')
                        return False

                pkt_list.append(pkt)

        # turn the packet list into a DataFrame, interpret the obt columns as a datetime
        tlm = pd.DataFrame(pkt_list)

        if len(tlm)==0:
            print('ERROR: no valid frames found in TM file(s)')
            return False

        tlm.obt = pd.to_datetime(tlm.obt)

        # Merge with the packet list, adding spid and description, then sort by OBT
        tlm = pd.merge(tlm,pid,how='left').sort('obt')

        # Deal with the fact that MIDAS uses private SIDs that are not in the RMIB
        if 'midsid' in tlm.columns:
            idx = tlm[tlm.midsid.notnull()].index
            tlm.sid.ix[idx]=tlm.midsid[idx]
            tlm.drop('midsid', axis=1, inplace=True)

        num_pkts = len(tlm)

        if dedupe:
            # Remove duplicates (packets with the same OBT, APID and SID)
            tlm = tlm.drop_duplicates(subset=('obt','apid','sid'), take_last=False)
            print('INFO: %i duplicate packets removed' % ( (num_pkts)-len(tlm) ) )

        print('INFO: %i packets read' % (len(tlm)) )

        # Ignore invalid frames with zero lengths
        tlm = tlm[tlm.length>0]

        # Check if the DDS header is present
        tlm = self.dds_check(tlm)

        # Add the DDS time to the packet list
        tlm = self.dds_time(tlm)

        # append to an existing packet list if already existing
        if append and type(self.pkts) == pd.core.frame.DataFrame:
            tlm = tlm.append(self.pkts)
            tlm.sort('obt', inplace=True, axis=0)

        # Reset the index
        tlm = tlm.reset_index(drop=True)

        if sftp: self.sftp = sftp
        self.pkts = tlm

        return


    def simple_locate_pkts(self, filename, dds_header=True, verbose=False):
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

        # print('INFO: File %s contains %i matching TM packets') % (filename, len(pkt_offsets))

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
            print('WARNING: no packets to read matching requested filters')
            return pkt_data

        if len(pkts.apid.unique()) + len(pkts.sid.unique()) != 2:
            print('WARNING: multiple packets types cannot be read together, filter first')
            return pkt_data

        filenames = pkts.filename.unique()

        for filename in filenames:
            if sftp:
                f = sftp.sftp.open(filename, 'rb')
            else:
                f = open(filename, 'rb')
            if debug: print('DEBUG: opening binary file %s' % (filename))
            for index,pkt in pkts[pkts.filename==filename].sort('offset').iterrows():
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

        num_pkts = len(self.pkts)
        if num_pkts == 0:
            print('WARNING: no packets found')
            return pkt_data

        pkts = self.pkts.sort('obt')

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
            print('WARNING: not enough shutter events to calculate exposure')
            return False

        if len(shutter_open) < len(shutter_close):
            print('WARNING: shutter open/close count mismatch')
            return False

        # Use the event OBT to calculate start/stop of exposure
        exposures = pd.DataFrame( zip(shutter_open, shutter_close), columns=['start', 'end'])
        exposures['duration'] = exposures.end - exposures.start

        # To find the target number we need to go to HK
        hk2 = self.pkts[ (self.pkts.type==3) & (self.pkts.subtype==25) & (self.pkts.apid==1076) & (self.pkts.sid==2) ]
        hk2.sort('obt', inplace=True, axis=0)

        # Loop through the OBTs and find the next HK2 packet, then extract parameters
        target = []

        for idx, exp in exposures.iterrows():
            frame = hk2[hk2.obt>exp.start].index
            if len(frame)==0:
                print('WARNING: no HK2 frame found after shutter open at %s' % exp.start)
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
                print('WARNING: no HK2 frame found after shutter open at %s' % ev.obt)
                continue
            else:
                before = frame_before[-2]
                after = frame_after[1]

                open_before = self.get_param('NMDD013B', frame=before)[1]
                closed_before = self.get_param('NMDD013C', frame=before)[1]
                open_after = self.get_param('NMDD013B', frame=after)[1]
                closed_after = self.get_param('NMDD013C', frame=after)[1]

                if debug: print('DEBUG: before (o/c) %s/%s, after (o/c) %s/%s' % (open_before, closed_before, open_after, closed_after))

                if (open_before=='OFF') & (closed_before=='ON') & (open_after=='ON') & (closed_after=='OFF'): # open
                    event.append('OPEN')
                elif (open_before=='ON') & (closed_before=='OFF') & (open_after=='OFF') & (closed_after=='ON'): # close
                    event.append('CLOSE')
                else:
                    print('WARNING: unknown shutter event transition at %s, skipping...' % ev.obt)
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
            print('WARNING: not enough shutter events to calculate exposure')
            return False

        if len(shutter_open) < len(shutter_close):
            print('WARNING: shutter open/close count mismatch')
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
        hk2.sort('obt', inplace=True, axis=0)

        # Loop through the OBTs and find the next HK2 packet, then extract parameters
        target = []

        for idx, exp in exposures.iterrows():
            frame = hk2[hk2.obt>exp.start].index
            if len(frame)==0:
                print('WARNING: no HK2 frame found after shutter open at %s' % exp.start)
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

        pkts = self.pkts[ (self.pkts.apid==1079) & (self.pkts.type==5) ]

        event_severity = ['PROGRESS','ANOMALY - NO ACTION','ANOMALY - GROUND ACTION','ANOMALY - ONBOARD ACTION']

        if len(pkts) == 0:
            print('WARNING: no events found')
            return False

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


        if info:
            events = pkts[['obt', 'doy', 'sid', 'event','information', 'severity']]
        else:
            events = pkts[['obt', 'doy', 'sid', 'event', 'severity']]

        if html:
            event_html = events.to_html(classes='alt_table', na_rep='',index=False, \
                formatters={ 'obt': lambda x: pd.to_datetime(x).strftime('%Y-%m-%d %H:%M:%S') } )
            css_write(event_html, html)


        uhoh = events[events.severity!='PROGRESS']
        if (len(uhoh)>0) & verbose:
            print('WARNING: non-nominal events detected:\n')
            printtable(uhoh[['obt','event']])

        return events



    def dds_check(self, pkts):
        """Simply adds a column to the pkts dataframe which lists if a
        DDS header is present (per packet)"""

        filenames = pkts.filename.unique()
        pkts['dds_header'] = pd.Series()
        for filename in filenames:
            pkts_per_file = pkts[pkts.filename==filename].sort('offset')
            header_gap = (pkts_per_file.offset.shift(-1)-(pkts_per_file.offset+pkts_per_file.length+7)).shift(1)
            header_gap.iloc[0] = pkts_per_file.offset.iloc[0]
            pkts.dds_header.ix[header_gap.index] = header_gap.apply( lambda x: True if x==18 else False )

        return pkts


    def dds_time(self, pkts):
        """Unpacks the DDS header and returns the UTC time (including time correlation)
        and adds this to the pkts dataframe"""

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
                delta_t = timedelta(seconds=dds_header.scet1, milliseconds=dds_header.scet2)
                dds_time.append(dds_obt_epoch + delta_t)

                # if debug: print('DEBUG: DDS time: %s' % (dds_obt_epoch + delta_t))

            pkts.dds_time.loc[offsets.index] = dds_time

        pkts.dds_time = pd.to_datetime(pkts.dds_time)

        return pkts



    def get_param(self, param, frame=False, start=False, end=False, tsync=True):
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
        if debug: print('DEBUG: parameter %s found in the following %i packet(s): ' % \
            (param.param_name, len(pkt_info)) + ", ".join(pkt_info.description.values))

        if type(start)==str:
            start = pd.Timestamp(start)

        if type(end)==str:
            end = pd.Timestamp(end)

        if start > end:
            print('ERROR: start time must be before end!')
            return None

        if start:
            pkts = pkts[pkts.obt>start]

        if end:
            pkts = pkts[pkts.obt<end]

        # filter packet list to those containing the parameter in question
        pkts = pkts[ (pkts.type.isin(pkt_info.type)) & (pkts.subtype.isin(pkt_info.subtype)) \
            & (pkts.apid.isin(pkt_info.apid)) & (pkts.sid.isin(pkt_info.sid)) ]

        # If requested, filter out packets that are not time-sync'd
        if tsync:
            pkts = pkts[ pkts.tsync ]

        num_params = len(pkts)

        if num_params == 0:
            print('ERROR: no matching TM packets found')
            return None

        # group packets by their SPID
        spids = pkts[pkts.spid.notnull()].spid.unique().astype(int)

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
                for index,pkt in pkts[pkts.filename==filename].sort('offset').iterrows():
                    pkt_data.append( f[((pkt.offset+byte_offset)*8+bit_offset):((pkt.offset+byte_offset)*8+bit_offset)+int(param.width)] )
                    obt.append( pkt.obt )

        # Now we simply need to convert the binary representation to the appropriate value
        # This is mostly done by calling the relevant bitstring function, but sometimes extra
        # conversions are needed.

        # PTC = 1: 1-bit boolean (extract byte, bit offset is 0-7)
        if param.ptc == 1:
            if param.pfc == 0: # boolean
                values = np.array(pkt_data) # nothing to do, 1-bit bitstrings are already bools
            else:
                print('ERROR: PTC 1 parameters must have PFC 0')

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
            print('ERROR: PTC/PFC type not supported')

        # Now calibrate the engineering value to a real unit
        real_val = calibrate(param.param_name, values)

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

        hk_img = np.zeros( (image.xsteps,image.ysteps) )

        if debug:
            print('DEBUG: image (%ix%i), main scan min/max: %i/%i, line scan min/max: %i/%i' % \
                (image.xsteps, image.ysteps, main.min(), main.max(), line.min(), line.max()))

        if (len(line) != len(main)) & (len(line) != len(values)):
            print('ERROR: mismatch of values, lines and main scan values')
            return

        for (idx, coord) in enumerate(zip(line,main)):
            hk_img[coord[0],coord[1]] = values[idx]

        plt.imshow(hk_img,  interpolation='nearest')

        return hk_img




    def plot_params(self, param_names, start=False, end=False, label_events=False):
        """Plot a TM parameter vs OBT. Requires a packet list and parameter
        name. label_events= can optionally be set to label key events."""

        import matplotlib.dates as md
        from dateutil import parser
        import math

        if type(param_names) != list: param_names = [param_names]
        units = pcf[pcf.param_name.isin(param_names)].unit.unique()

        if type(start)==str:
            start = parser.parse(start)

        if type(end)==str:
            end = parser.parse(end)

        if start > end:
            print('ERROR: start time must be before end time!')
            return None

        if len(units) > 2:
            print('ERROR: cannot plot parameters with more than 2 different units on the same figure')
            return False

        if 'S' in pcf[pcf.param_name.isin(param_names)].cal_cat.tolist():
            print('ERROR: one or more parameters is non-numeric, cannot plot!')
            return False

        plot_fig = plt.figure()
        ax_left = plot_fig.add_subplot(1,1,1)
        if len(units)==2:
            ax_right = ax_left.twinx()

        lines = []

        for param_name in param_names:

            data = self.get_param(param_name, start=start, end=end, tsync=True)

            if data is None:
                print('WARNING: could not get data for parameter %s, skipping' % (param_name))
                continue

            param = pcf[pcf.param_name==param_name].squeeze()

            if param.unit==units[0] or (pd.isnuotll(param.unit) & (pd.isnull(units[0]))): # plot on left axis
                lines.append(ax_left.plot( data.index, data, label=param.description, linestyle='-' )[0])
                ax_left.set_ylabel( "%s" % (param.unit))
            else:
                ax_left._get_lines.color_cycle.next()
                lines.append(ax_right.plot( data.index, data, label=param.description, linestyle='-.' )[0])
                ax_right.set_ylabel( "%s" % (param.unit))

        if len(lines)==0:
            print('ERROR: no data to plot')
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

        if label_events:

            events = self.get_events(ignore_giada=True, verbose=False)

            if label_events=='scan':
                # 42656 - EvFullScanStarted
                # 42756 - EvFullScanAborted
                # 42513 - EvScanFinished
                # 42713 - EvScanAborted
                scan_events = [42656, 42756, 42513, 42713]
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
                print('WARNING: invalid event label, ignoring')
                events = pd.DataFrame()

            # Fixing mark/MIDAS#2 - filter events to given time range first
            events = events[ (events.obt>start) & (events.obt<end) ]

            for idx, event in events.iterrows():
                ax_left.axvline(event.obt,color='r')
                ax_left.text(event.obt,max(data)*0.9,event.event,rotation=90)

        plt.draw()

        return


    def plot_temps(self, cssc=False, start=False, end=False):
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

        self.plot_params(temp_params, label_events='scan', start=start, end=end)
        return


    def plot_hv(self, start=False, end=False, events='scan'):
        """Plot piezo high voltages (HV)s"""

        hv_params = ['NMDA0110', 'NMDA0111', 'NMDA0115']
        self.plot_params(hv_params, label_events=events, start=start, end=end)



    def plot_volts(self, cssc=False, start=False, end=False):
        """Plot the voltages from all MIDAS lines for the given TM packages.
        If start= and end= are set to date/time strings the data will be limited
        to those times, otherwise all data are plotted."""

        volt_params = self.search_params('voltage mon').param_name.tolist()

        # if not cssc:
        #     volt_params.remove( ['NMDA0206', 'NMDA0207', 'NMDA0208', 'NMDA0209'] )

        if len(volt_params)==0:
            print('No MIDAS voltage data found in these packets')
        else:
            self.plot_params(volt_params, label_events='scan', start=start, end=end)



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
            [['param_name','description']].sort('description')

        return params



    def search_params(self, search):
        """Creates a list of available parameters and searches for the search term
        within their description"""

        param_list = self.list_params()

        result = param_list[param_list.description.str.contains(search, case=False)]

        if len(result)==0:
            print('WARNING: no parameters found containing search term "%s"' % (search))
            return False

        return result


    def get_tc_status(self):
        """Retrieves the MIDAS TC exec/failure events and lists details"""

        import struct

        # Filter events by SIDs for s/w TC acceptance and rejection
        ack = self.pkts[ (self.pkts.apid==1079) & ( (self.pkts.sid==42501) | (self.pkts.sid==42701) ) ]
        if len(ack)==0:
            print('WARNING: no TC ACK packets found')
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
                print('WARNING: unknown TC retrieving ACK status (type %i, subtype %i) at OBT %s' % (srv_type, srv_subtype, packet.obt))
            elif (len(tc) > 1) & (len(np.unique(tc))==1):
                tcs.append(tc[0])
            elif (len(tc) > 1) & (len(np.unique(tc))>1):
                print('WARNING: more than one command matching type/subtype %i/%i' % ( srv_type, srv_subtype ))
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

            print('INFO: %i missing packets in %i gaps for APID %i' % (missing_pkts.sum(), len(missing_pkts), apid))

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
            print('INFO: no missing packets found')
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



    def get_line_scans(self, info_only=False):
        """Extracts line scans from TM packets"""

        line_scan_fmt = ">H2Bh6H6H"
        line_scan_size = struct.calcsize(line_scan_fmt)
        line_scan_names = collections.namedtuple("line_scan_names", "sid sw_minor sw_major lin_pos \
            wheel_pos tip_num x_orig y_orig step_size num_steps scan_mode_dir line_cnt sw_flags \
            spare1 spare2 spare3")

        line_scan_pkts = self.read_pkts(self.pkts, pkt_type=20, subtype=3, apid=1084, sid=132)

        scan_type = ['DYN','CON','MAG']

        if len(line_scan_pkts)==0:
            print('WARNING: no line image packets found')
            return False

        linescans = []

        for idx,pkt in line_scan_pkts.iterrows():
            line_type = {}
            line_type['info'] = line_scan_names(*struct.unpack(line_scan_fmt,pkt['data'][0:line_scan_size]))
            num_steps = line_type['info'].num_steps
            if not info_only:
                line_type['data'] = np.array(struct.unpack(">%iH" % (num_steps),pkt['data'][line_scan_size:line_scan_size+num_steps*2]))
                # line_type['data'] = 32767 - line_type['data'] # invert to get height
            linescans.append(line_type)

        lines = pd.DataFrame([line['info'] for line in linescans],columns=line_scan_names._fields,index=line_scan_pkts.index)
        # Adding new flags
        # Software flags:
        # Bits   15-3: Spare
        # Bit     2-1: Anti-creep status (0=idle, 1=init, 2=active)
        # Bit       0: Image scan active flag
        lines['in_image'] = lines.sw_flags.apply( lambda flag: bool(flag & 1))
        lines['anti_creep'] = lines.sw_flags.apply( lambda flag: bool(flag >> 1 & 0b11))

        lines['obt'] = line_scan_pkts.obt
        lines['tip_num'] += 1
        lines['sw_ver'] = lines.sw_major.apply( lambda major: '%i.%i' % (major >> 4, major & 0x0F) )
        lines['sw_ver'] = lines['sw_ver'].str.cat(lines['sw_minor'].values.astype(str),sep='.')
        lines['lin_pos'] = lines.lin_pos.apply( lambda pos: pos*20./65535.)

        lines['fast_dir'] = lines.scan_mode_dir.apply( lambda fast: 'X' if (fast & 2**12)==0 else 'Y')
        lines['dir'] = lines.scan_mode_dir.apply( lambda xdir: 'L_H' if (xdir & 2**8)==0 else 'H_L')
        lines['scan_type'] = lines.scan_mode_dir.apply( lambda mode: scan_type[ mode & 0b11 ] )
        lines['tip_offset'] = lines.apply( lambda row: (row.lin_pos-common.lin_centre_pos[row.tip_num-1]) / common.linearcal, axis=1 )

        lines.drop( ['sw_major', 'sw_minor', 'sid', 'scan_mode_dir', 'sw_flags', 'spare1', 'spare2', 'spare3'], inplace=True, axis=1)

        if not info_only:
            lines['data'] = [line['data'] for line in linescans]

        print('INFO: %i line scans extracted' % (len(linescans)))

        return lines


    def get_ctrl_data(self, show=False, rawdata=False, info_only=False):
        """Extracts control data from TM packets"""

        import common

        scan_type = ['DYN','CON','MAG']

        ctrl_data_fmt = ">H2Bh6H2B2H3H"
        ctrl_data_size = struct.calcsize(ctrl_data_fmt)
        ctrl_data_names = collections.namedtuple("line_scan_names", "sid sw_minor sw_major lin_pos \
            wheel_posn tip_num x_orig y_orig step_size num_steps scan_mode scan_dir main_cnt \
            num_meas block_addr sw_flags spare")

        ctrl_data_pkts = self.read_pkts(self.pkts, pkt_type=20, subtype=3, apid=1084, sid=133)

        if len(ctrl_data_pkts)==0:
            print('WARNING: no control data packets found')
            return False

        # M.S.Bentley 23/09/2014 - using df to collect ctrl data meta info

        ctrl_data = []

        for idx,pkt in ctrl_data_pkts.iterrows():
            ctrl_data.append(ctrl_data_names(*struct.unpack(ctrl_data_fmt,pkt['data'][0:ctrl_data_size])))
        ctrl_data = pd.DataFrame(ctrl_data,columns=ctrl_data_names._fields,index=ctrl_data_pkts.index)
        ctrl_data['obt'] = ctrl_data_pkts.obt

        # Convert data as necessary in the df
        ctrl_data.block_addr = ctrl_data.block_addr.apply( lambda block: block >> 1 )
        ctrl_data['z_retract'] = ctrl_data.block_addr.apply( lambda block: bool(block & 1) )
        ctrl_data['sw_ver'] = ctrl_data.sw_major.apply( lambda major: '%i.%i' % (major >> 4, major & 0x0F) )
        ctrl_data['sw_ver'] = ctrl_data['sw_ver'].str.cat(ctrl_data['sw_minor'].values.astype(str),sep='.')
        ctrl_data['lin_pos'] = ctrl_data.lin_pos.apply( lambda pos: pos*20./65535.)
        ctrl_data.tip_num += 1
        ctrl_data['scan_dir'] = ctrl_data.scan_mode.apply( lambda fast: 'X' if (fast >> 12 & 1)==0 else 'Y')
        ctrl_data['x_dir'] = ctrl_data.scan_mode.apply( lambda xdir: 'H_L' if (xdir >> 8 & 1) else 'L_H')
        ctrl_data['scan_type'] = ctrl_data.scan_mode.apply( lambda mode: scan_type[ mode & 0b11 ] )

        ctrl_data['hires'] = ctrl_data.sw_flags.apply( lambda flags: bool(flags >> 1 & 1))
        ctrl_data['in_image'] = ctrl_data.sw_flags.apply( lambda flag: bool(flag & 1))

        ctrl_data.drop(['sid', 'sw_major', 'sw_minor', 'scan_mode', 'sw_flags', 'spare'], inplace=True, axis=1)

        if info_only:
            return ctrl_data

        control = []

        # Now extract actual data
        for idx,pkt in ctrl_data_pkts.iterrows():

            num_steps = ctrl_data.num_meas.ix[idx]
            point_data = {}

            # 07/01/15 - M.S.Bentley - in fact all control data are signed integers!
            data = struct.unpack(">%ih" % (num_steps),pkt['data'][ctrl_data_size:ctrl_data_size+num_steps*2])

            point_data['ac']    = np.array(data[0::4])
            point_data['dc']    = np.array(data[1::4])
            point_data['phase'] = np.array(data[2::4])
            point_data['zpos']  = np.array(data[3::4])

            if not rawdata:

                point_data['ac'] = point_data['ac'] * (20./65535.)

                # point_data['dc'][point_data['dc'] > 32878] = point_data['dc'][point_data['dc'] > 32878] - 65535
                point_data['dc'] = point_data['dc'] * (20./65535.)

                point_data['phase'] = point_data['phase'] * (360./65535.)

                point_data['zpos'] = point_data['zpos'] - point_data['zpos'].min()
                point_data['zpos'] = point_data['zpos'] * common.zcal * 2.0

            control.append(point_data)

        ctrl_data['ac'] = [data['ac'] for data in control]
        ctrl_data['dc'] = [data['dc'] for data in control]
        ctrl_data['phase'] = [data['phase'] for data in control]
        ctrl_data['zpos'] = [data['zpos'] for data in control]

        print('INFO: %i control data scans extracted' % (len(ctrl_data)))

        return ctrl_data



    def get_feature_vectors(self):
        """Checks for feature vector packets and returns the relevant data"""

        # TODO: fix so it handles multiple fvec sets

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
            print('WARNING: no feature vector packets found')
            return False

        for idx,pkt in fvec_pkts.iterrows():

            fvec_header = fvec_header_names(*struct.unpack(fvec_header_fmt,pkt['data'][0:fvec_header_size]))
            fvec_header = fvec_header._asdict()
            fvec_header['num_feat'] -= 1
            fvec_header['first_feat'] /= 2
            fvec_header['feat_weight'] *= (4./65535.)
            fvec_header['regress_x'] /= 65536.
            fvec_header['regress_y'] /= 65536.

            for feat_num in range(fvec_header['num_fvec']):
                feature_start = fvec_header_size+feat_num*fvec_size
                feature_end = feature_start + fvec_size
                feature.append(fvec_names(*struct.unpack(fvec_fmt,pkt['data'][feature_start:feature_end])))

            feature = pd.DataFrame(feature, columns=fvec_names._fields)

        print('INFO: %i feature vectors extracted' % len(feature))

        return fvec_header, feature


    def get_images(self, info_only=False, rawheader=False, rawdata=False, sw_flags=False):
        """Extracts images from telemetry packets. Setting info_only=True returns a
        dataframe containing the scan metadata, but no actual images"""

        import common

        scan_type = ['DYN','CON','MAG']

        # structure definition for the image header packet
        image_header_fmt = ">H2B2IHh11H11H2H"
        image_header_size = struct.calcsize(image_header_fmt)
        image_header_names = collections.namedtuple("img_header_names", "sid sw_minor sw_major start_time end_time \
            channel lin_pos wheel_pos tip_num x_orig y_orig x_step y_step xsteps_dir ysteps_dir scan_type \
            dset_id scan_mode status sw_flags line_err_cnt scn_err_cnt z_ret mag_ret dc_mag_1 dc_mag_2 dc_mag_3 spare1 \
            dblock_start num_pkts checksum")

        image_header = []; filename = []

        # filter image header packets
        img_header_pkts = self.read_pkts(pkts=None, pkt_type=20, subtype=3, apid=1084, sid=129)
        if debug:
            print('DEBUG: %i image header packets found' % (len(img_header_pkts)))

        if len(img_header_pkts) == 0:
            print('INFO: no images found')
            return None

        # Loop over header packets, unpack the header values and append them to a list
        for idx,pkt in img_header_pkts.iterrows():
            image_header.append(image_header_names(*struct.unpack(image_header_fmt,pkt['data'][0:image_header_size])))
            if debug: print('DEBUG: processing file %s' % pkt.filename)
            filename.append(pkt.filename) # src filename needed to correctly name scans

        # structure definition for the individual image packets
        image_fmt = ">H2BI4H7H"
        image_size = struct.calcsize(image_fmt)
        image_names = collections.namedtuple("img_names", "sid sw_minor sw_major start_time channel pkt_num \
            first_line last_line spare1 spare2 spare3 spare4 spare5 spare6 spare7")

        if not info_only:

            # filter image packets
            if debug: print('DEBUG: starting image data packet unpack')
            img_pkts = self.read_pkts(self.pkts, pkt_type=20, subtype=3, apid=1084, sid=130)
            if debug: print('DEBUG: %i image packets found' % (len(img_pkts)))

            image_data = [] # holds image packet headers and data

            if debug: print('DEBUG: unpacking image packet headers and image data')

            # loop through image packets and unpack the image packet header and image data
            # THIS EATS LOTS OF RAM
            for idx,pkt in img_pkts.iterrows():
                if debug: print('DEBUG: image packet %i of %i' % (idx, len(img_pkts)))
                image = {}
                image['header'] = image_names(*struct.unpack(image_fmt,pkt['data'][0:image_size]))
                if (image['header'].channel == 1) or (image['header'].channel == 32768):
                    image['data'] = struct.unpack(">1024H",pkt['data'][image_size:image_size+2048])
                else:
                    image['data'] = struct.unpack(">1024h",pkt['data'][image_size:image_size+2048])
                image_data.append(image)

            if debug: print('DEBUG: all image packet data read, processing..')

            # convert the image packet headers to a pandas dataframe for easy filtering
            img_data=pd.DataFrame([img['header'] for img in image_data],columns=image_names._fields)

            if rawdata: return image_header, img_data, image_data

            # image_header - list of raw unpacked image header packets
            # img_data - DataFrame of image packets without raw data
            # image_data - list of image packet data content

            image_temp = []

            # for each image header packet
            for idx, image in enumerate(image_header):

                if debug: print('DEBUG: processing image %i of %i' % (idx, len(image_header)))

                imagedict = {}
                imagedict['info'] = image # image header into -> image['info']

                # Now match the packets to the corresponding header by OBT and channel
                data = img_data[ (img_data.start_time==image.start_time) & (img_data.channel==image.channel) ]
                data = data.sort('pkt_num')
                if debug: print('DEBUG: %i image data packets expected, %i packets found' % (image.num_pkts, len(data)) )

                if len(data) < image.num_pkts:
                    print('WARNING: image at %s: missing image packets' % (obt_to_iso(image.start_time)))
                    # continue
                elif len(data) > image.num_pkts:
                    print('ERROR: image at %s: too many image packets (%i instead of %i) - image may be corrupt!' % (obt_to_iso(image.start_time),len(data),image.num_pkts))
                    data = data.drop_duplicates(subset=('start_time','pkt_num'), take_last=True)

                # Use the xsteps and ysteps values from the header to mash together and reform the data
                xsteps = image.xsteps_dir & 0x3ff
                ysteps = image.ysteps_dir & 0x3ff

                if (xsteps==0) or (ysteps==0):
                    print('INFO: image at %s: invalid image dimensions (%i,%i), skipping...' % (obt_to_iso(image.start_time),xsteps,ysteps))
                    continue

                main_y = bool(image.scan_type & 2**15) # False = X, True = Y

                # merge the ordered image data into a linear array
                # 01/12/2014 - instead of skipping image, add dummy data
                image_array = []
                pkt_idx = 1
                # while pkt_idx <= image.num_pkts:
                for pkt_num in range(1,image.num_pkts+1):
                    if pkt_num not in data.pkt_num.tolist():
                        if debug: print('DEBUG: inserting one missing packet at packet number %i' % pkt_num)
                        image_array.extend([65535]*1024)
                    else:
                        image_array.extend(image_data[ data.index[pkt_idx-1] ]['data'])
                        pkt_idx += 1

                image_array=np.array(image_array)

                if not rawdata and not info_only:

                    if len(image_array) != xsteps*ysteps:
                        print('ERROR: image at %s: number of data points (%i) doesn\'t match image dimensions (%ix%i=%i)'
                            % (obt_to_iso(image.start_time),len(image_array),xsteps,ysteps,xsteps*ysteps))
                        continue

                    # Need to invert for Z piezo/topography since 0=piezo retracted (image max height).
                    if image.channel==1: image_array = 32767-image_array

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
        info.start_time = pd.to_datetime(info.start_time.apply( lambda obt: np.nan if obt==0 else obt_to_datetime(obt) ))
        info.end_time = pd.to_datetime(info.end_time.apply( lambda obt: np.nan if obt==0 else obt_to_datetime(obt) ))
        info['duration'] = info.end_time-info.start_time
        #info['filename'] = pd.Series([image['filename'] for image in image_temp])
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
        info['scan_type'] = info.scan_type.apply( lambda mode: scan_type[ mode & 0b11 ] )
        info['exc_lvl'] = info.scan_mode.apply( lambda mode: mode >> 13 )
        info['dc_gain'] = info.scan_mode.apply( lambda mode: mode >> 10 & 0b111 )
        info['ac_gain'] = info.scan_mode.apply( lambda mode: mode >> 7  & 0b111 )
        info['z_closed'] = info.scan_mode.apply( lambda mode: bool(mode >> 6 & 1) )
        info['y_closed'] = info.scan_mode.apply( lambda mode: bool(mode >> 5 & 1) )
        info['x_closed'] = info.scan_mode.apply( lambda mode: bool(mode >> 4 & 1) )
        info['scan_algo'] = info.scan_mode.apply( lambda mode: mode & 0b1111 )
        info['target'] = info.wheel_pos.apply( lambda seg: common.seg_to_facet(seg) )
        info['target_type'] = info.target.apply( lambda tgt: common.target_type(tgt) )
        info['tip_num'] = info.tip_num+1
        info['sw_ver'] = info.sw_major.apply( lambda major: '%i.%i' % (major >> 4, major & 0x0F) )
        info['sw_ver'] = info['sw_ver'].str.cat(info['sw_minor'].values.astype(str),sep='.')
        info['channel'] = info.channel.apply( lambda channel: common.data_channels[int(math.log10(channel)/math.log10(2))] )

        # Instrument status
        info['mtl_disabled'] = info.status.apply( lambda status: bool(status >> 15 & 1) )
        info['auto_expose_mode'] = info.status.apply( lambda status: bool(status >> 4 & 0b111 ) )
        info['tech_mode'] = info.status.apply( lambda status: bool(status & 0b1 ) )

        # Software flags
        info['auto_expose'] = info.sw_flags.apply( lambda flag: bool( flag >> 8 & 0b1 ) )
        info['anti_creep'] = info.sw_flags.apply( lambda flag: bool( flag >> 7 & 0b1 ) )
        info['ctrl_retract'] = info.sw_flags.apply( lambda flag: bool( flag >> 6 & 0b1 ) )
        info['ctrl_image'] = info.sw_flags.apply( lambda flag: bool( flag >> 5 & 0b1 ) )
        info['line_in_img'] = info.sw_flags.apply( lambda flag: bool( flag >> 4 & 0b1 ) )
        info['linefeed_zero'] = info.sw_flags.apply( lambda flag: bool( flag >> 2 & 0b1 ) )
        info['linefeed_last_min'] = info.sw_flags.apply( lambda flag: bool( flag >> 1 & 0b1 ) )
        info['fscan_phase'] = info.sw_flags.apply( lambda flag: bool( flag & 0b1 ) )

        sw_flags_names = ['auto_expose', 'anti_creep', 'ctrl_retract', 'ctrl_image', 'line_in_img',
            'linefeed_zero', 'linefeed_last_min', 'fscan_phase']

        # Calculate the real step size
        info['x_step_nm'] = info.x_step.apply( lambda step: step * common.xycal['open'])
        info['y_step_nm'] = info.apply( lambda row: (row.y_step * common.xycal['closed']) if row.y_closed else (row.y_step * common.xycal['open']), axis=1)
        info['xlen_um'] = info.xsteps * info.x_step_nm /1000.
        info['ylen_um'] = info.ysteps * info.y_step_nm /1000.
        info['z_ret_nm'] = info.z_ret * common.zcal

        # Calculate the tip offset from the wheel centre
        info['tip_offset'] = info.apply( lambda row: (row.lin_pos-common.lin_centre_pos[row.tip_num-1]) / common.linearcal, axis=1 )

        # Add the filename
        info['scan_file'] = info.apply( lambda row: src_file_to_img_file(os.path.basename(row.filename), row.start_time, row.target), axis=1 )

        return_data = ['filename', 'scan_file', 'sw_ver', 'start_time','end_time', 'duration', 'channel', 'tip_num', 'lin_pos', 'tip_offset', 'wheel_pos', 'target', 'target_type', \
            'x_orig','y_orig','xsteps', 'x_step','x_step_nm','xlen_um','ysteps','y_step','y_step_nm','ylen_um','z_ret', 'z_ret_nm', 'x_dir','y_dir','fast_dir','scan_type',\
            'exc_lvl', 'ac_gain', 'x_closed', 'y_closed', 'aborted', 'dummy']

        if sw_flags: return_data.extend(sw_flags_names)

        images = info[return_data]

        # De-dupe (should be no dupes, but in certain cases the instrument generates them)
        num_images = len(images)
        images = images.drop_duplicates(subset=['start_time','channel'])
        if len(images)<num_images:
            print('WARNING: duplicate images or image headers detected and removed!')

        # Add the origin of the scan, in microns, from the wheel centre
        images = locate_scans(images)

        if not info_only:
            images['data'] = pd.Series([image['data'] for image in image_temp])

        print('INFO: %i images found' % (len(info.start_time.unique())))

        return images.sort('start_time')


    def get_freq_scans(self, printdata=False, cantilever=False, fit=False, info_only=False, get_thresh=False):
        """Extracts frequency scans from TM packets. If cantilever= is set to a
        cantilever number from 1-16, only fscans for that cantilever are extracted.

        fit=True fits a Lorentzian to all curves and returns the fit parameters.
        printdata=True displays the parameters
        info_only=True returns the meta-data without the raw data"""

        # Define frequency packet format
        # M.S.Bentley 11/09/2014 - updating for latest OBSW
        freq_scan_fmt = ">H2B2I2HI6H7H"
        freq_scan_size = struct.calcsize(freq_scan_fmt)
        freq_scan_names = collections.namedtuple("freq_scan_names", "sid sw_minor sw_major start_time freq_start \
            freq_step max_amp freq_at_max num_scans fscan_cycle cant_num cant_block exc_lvl ac_gain fscan_type \
            spare2 spare3 spare4 spare5 spare6 spare7")

        freq_scan_pkts = self.read_pkts(self.pkts, pkt_type=20, subtype=3, apid=1084, sid=131)

        if len(freq_scan_pkts)==0:
            print('WARNING: no frequency scan packets found')
            return False

        # convert the fscan packet header (everything up to the scan data) into a df
        fscans = []

        for idx,pkt in freq_scan_pkts.iterrows():
            fscans.append(freq_scan_names(*struct.unpack(freq_scan_fmt,pkt['data'][0:freq_scan_size])))
        fscans = pd.DataFrame(fscans,columns=freq_scan_names._fields,index=freq_scan_pkts.index)

        # if the cantilever keyword is set, filter the fscans
        if cantilever:
            if (cantilever >=1) and (cantilever <= 16):
                block = 0 if cantilever <= 8 else 1
                cant = (cantilever-1) % 8
                fscans = fscans[ (fscans.cant_num==cant) & (fscans.cant_block==block)]
                if len(fscans)==0:
                    print('WARNING: no frequency scans found for cantilever %i' % (cantilever))
                    return False
            else:
                print('ERROR: cantilever number must be between 1 and 16')

        # find the number of unique scans (unique start OBTs)
        start_times = fscans.start_time.unique()

        # Find the index of HK2 packets, used to extract anciliary info
        if get_thresh:
            hk2 = self.pkts[ (self.pkts.type==3) & (self.pkts.subtype==25) & (self.pkts.apid==1076) & (self.pkts.sid==2) ]

        freqscans = []
        amp_scans = []

        for idx, start in enumerate(start_times):

            single_scan = fscans[fscans.start_time==start]
            single_scan  = single_scan.sort('fscan_cycle',ascending=True)
            if not 1 in single_scan.fscan_cycle.tolist():
                print('ERROR: missing frequency scan packets, skipping scan')
                continue
            else:
                first_pkt = single_scan[single_scan.fscan_cycle==1].iloc[0]

            num_scans = first_pkt.num_scans

            if len(single_scan) != num_scans:
                print('ERROR: missing frequency scan packets, skipping scan')
                continue

            # take the scan info from the first packet where generic to entire scan
            scan = {}
            scan['info'] = {}
            start_time = obt_epoch + timedelta(seconds=first_pkt.start_time)

            # Look at HK2 data to retrieve the resonance amplitude, working point, frequency
            # adjust point and set-point derived from this scan...
            if get_thresh:
                frame = hk2[hk2.obt>start_time].index
                if len(frame)==0:
                    print('WARNING: no HK2 frame found after frequency scan at %s' % start)
                    continue
                else:
                    frame = frame[0]

            scan['info'] = {
                'sw_ver': '%i.%i.%i' % (first_pkt.sw_major >> 4, first_pkt.sw_major & 0x0F, first_pkt.sw_minor),
                'start_time': start_time,
                'freq_start': first_pkt.freq_start * 0.0006984, # 00021065027,
                'freq_step': first_pkt.freq_step * 0.0006984, # 00021065027,
                'max_amp': single_scan.max_amp.max()*(20./65535.),
                'max_freq': single_scan.freq_at_max.max() * 0.0006984, # 00021065027,
                'num_scans': first_pkt.num_scans,
                'cant_num': first_pkt.cant_num+1 + first_pkt.cant_block*8,
                'cant_block': first_pkt.cant_block,
                'excitation': first_pkt.exc_lvl,
                'gain': first_pkt.ac_gain,
                'is_phase': True if first_pkt.fscan_type else False }

            if scan['info']['is_phase']:
                do_fit = False
            else:
                do_fit = fit
                amp_scans.append(idx)

            if get_thresh:

                scan['info']['res_amp'] = self.get_param('NMDA0306', frame=frame)[1]
                scan['info']['set_pt'] = self.get_param('NMDA0245', frame=frame)[1]
                scan['info']['fadj'] = self.get_param('NMDA0347', frame=frame)[1]
                scan['info']['work_pt'] = scan['info']['res_amp'] * abs(self.get_param('NMDA0181', frame=frame)[1]) / 100.

            if printdata:
                print('INFO: cantilever %i/%i with gain/exc %i/%i has peak amplitude %3.2f V at frequency %3.2f Hz' % \
                    (scan['info']['cant_block'], scan['info']['cant_num'], \
                    scan['info']['gain'], scan['info']['excitation'], \
                    scan['info']['max_amp'], scan['info']['max_freq']) )

            # loop through the packets and add the fscan data
            if not info_only:
                scan['amplitude'] = []
                for idx in single_scan.index:
                    scan['amplitude'].extend(struct.unpack(">256H",freq_scan_pkts.data.loc[idx]\
                    [freq_scan_size:freq_scan_size+512]))

                if first_pkt.fscan_type==1:
                    scan['amplitude'] = np.array(scan['amplitude']) * (180./65535.)-180.
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
                    popt,pcov = curve_fit(lorentzian,scan['frequency'],scan['amplitude'],p0=[offset, amp, cen, wid])
                except RuntimeError:
                    print('WARNING: problem with curvefit, skipping this scan...')
                    continue

                fit_params={}
                fit_params['offset'] = popt[0]
                fit_params['fit_max'] = popt[1]+popt[0]
                fit_params['res_freq'] = popt[2]
                fit_params['half_width'] = popt[3]
                fit_params['q'] = fit_params['res_freq'] / (2. * fit_params['half_width'])

                fit_keys = fit_params.keys()

                scan['fit_params'] = fit_params

        print('INFO: %i frequency scans found and extracted' % (len(freqscans)))

        # Convert to a DataFrame for return
        scans = pd.DataFrame( [scan['info'] for scan in freqscans], columns=freqscans[0]['info'].keys())

        # Include fit parameters if calculated
        if fit:
            fits = pd.DataFrame(
                [scan['fit_params'] for scan in freqscans if scan.has_key('fit_params')],
                columns=fit_keys, index=amp_scans )
            scans = pd.concat([scans,fits],axis=1)

        # Add the actual data
        # TODO check if this slows down display in iPython as it seems to for images
        if not info_only:
            scans['frequency'] = pd.Series([scan['frequency'] for scan in freqscans])
            scans['amplitude'] = pd.Series([scan['amplitude'] for scan in freqscans])

        return scans


    def app_at_surface(self):
        """Scans pkts for ApproachFinished events, finds the next HK2 packet, and extracts
        the approach position, cantilever and segment number."""

        # Filter packets to find the ApproachFinished event
        srf_found = self.pkts[ (self.pkts.apid==1079) & (self.pkts.sid==42664) ]
        print('INFO: %i ApproachFinished events found' % len(srf_found))

        # Filter HK2 packets
        hk2 = self.pkts[ (self.pkts.type==3) & (self.pkts.subtype==25) & (self.pkts.apid==1076) & (self.pkts.sid==2) ]

        approaches = []

        # Loop through the OBTs and find the next HK2 packet, then extract parameters
        for idx,event in srf_found.iterrows():
            approach = {}
            frame = hk2[hk2.obt>event.obt].index
            if len(frame)==0:
                print('WARNING: no HK2 frame found after approach at %s' % event.obt)
                continue
            else:
                frame = frame[0]

            approach['obt'] = self.pkts.obt.ix[frame]
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


    def sample_slope(self):
        """Extracts all images in pkts and fits a plane to the image, returning the
        plane fit and maximum height difference in the image, as well as the target
        segment"""

        import common

        #TODO

        images = self.get_images(self.pkts)
        if images is None:
            print('WARNING: no images in this TM')
            return False

        for image in images:
            height_diff = (image['data'].max()-image['data'].min())*common.zcal
            # TODO - get new bits/bytes in image header to extract segment

        return segment, fit, height_diff

#----- end of TM class

def check_retract(image, boolmask=True):
    """Accepts an image produced by get_images() and checks whether any points have
    pixel-to-pixel height differences greater than the retraction.

    If boolmask=True a boolean array is returned masking points.
    If boolmask=False a numerical array of the height above retraction is returned."""

    # if type(images) == pd.Series:
    #     images = pd.DataFrame(columns=images.to_dict().keys()).append(images)

    if image.channel!='ZS':
        print('ERROR: retraction checking can only be performed on a topography channel')
        return False

    data = image['data']

    # Transpose and flip according to main scan direction and H_L parameters
    if image.x_dir == 'H_L': data = np.fliplr(data)
    if image.y_dir == 'H_L': data = np.flipud(data)
    if image.fast_dir=='Y': data = np.transpose(data)

    # Loop through rows, checking for pixels with height diffs > z_ret
    num_rows = image.ysteps if image.fast_dir=='X' else image.xsteps

    if boolmask:
        mask = np.zeros_like(data, dtype=bool) # mask is a boolean array of bad pixels
    else:
        mask = np.zeros_like(data)

    for row in range(num_rows):
        line=data[row,:]
        if boolmask:
            mask[row,1:] = (line[1:]-line[:-1]) > image.z_ret
        else:
            mask[row,1:] = (line[1:]-line[:-1]) - image.z_ret

    if not boolmask: mask[ mask<0 ] = 0

    # Transpose and flip according to main scan direction and H_L parameters
    if image.fast_dir=='Y': mask = np.transpose(mask)
    if image.y_dir == 'H_L': mask = np.flipud(mask)
    if image.x_dir == 'H_L': mask = np.fliplr(mask)

    return mask





def css_write(html, filename, cssfile='midas.css'):
    """Accepts an html string (usually generated by pd.DataFrame.to_html() ) and
    adds a reference to a given CSS file"""

    cssfile = os.path.join(css_path,cssfile)
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
            print('WARNING: APID %i not a MIDAS APID' % (apid))

    return pkt_list


def src_file_to_img_file(src_filename, start_time, target):
    mtp = src_filename[9:12]
    stp = src_filename[14:17]
    start = datetime.strftime(start_time,isofmt)
    filename = 'SCAN_MD_M%s_S%s_%s_TGT%02d' % (mtp,stp,start,target)
    return filename


# TODO implement time correlation

def read_time_correlation(filename):

    return []


def read_ccf(filename=False):
    """Reads the SCOS-2000 ccf.dat file"""

    if not filename:
        filename = os.path.join(s2k_path, 'ccf.dat')

    cols = ('name', 'description', 'type', 'subtype', 'apid')
    ccf=pd.read_table(filename,header=None,names=cols,usecols=[0,1,6,7,8])

    return ccf

def read_csf(filename=False):
    """Load the command sequence file (CSF)"""

    # Get the description of each sequence from the command sequence file (csf.dat)
    csf_file = os.path.join(s2k_path, 'csf.dat')
    cols = ('sequence', 'description', 'csf_plan')
    csf = pd.read_table(csf_file,header=None,names=cols,usecols=[0,1,7],na_filter=False)

    return csf


def read_csp(midas=False):
    """Load the command sequence parameters (CSP) file"""

    csp_file = os.path.join(s2k_path, 'csp.dat')
    cols = ('sequence', 'param', 'param_num','param_descrip','ptc','pfc','format', \
        'radix', 'param_type', 'val_type', 'default', 'cal_type', 'range_set', \
        'numeric_cal', 'text_cal', 'eng_unit')
    csp = pd.read_table(csp_file,header=None,names=cols,usecols=range(0,16),na_filter=True,index_col=False)
    if midas: csp = csp[csp.sequence.str.startswith('AMD')] # filter by MIDAS sequences
    csp = csp.sort( ['sequence','param_num'] )

    return csp


def read_css(midas=False):
    """Load the command sequence definition (CSS) file"""

    css_file = os.path.join(s2k_path, 'css.dat')
    cols = ('sequence', 'description', 'entry', 'type', 'name', 'num_params', 'reltype','reltime','extime')

    css = pd.read_table(css_file,header=None,names=cols,usecols=[0,1,2,3,4,5,7,8,9],na_filter=True,index_col=False)
    if midas: css = css[css.sequence.str.startswith('AMD')] # filter by MIDAS sequences
    css = css.sort( ['sequence','entry'] )

    return css


def read_sdf(midas=False):
    """Load the command sequence element parameters (SDF) file"""

    sdf_file = os.path.join(s2k_path, 'sdf.dat')
    cols = ('sequence', 'entry', 'el_name', 'param', 'val_type', 'value')

    sdf = pd.read_table(sdf_file,header=None,names=cols,usecols=[0,1,2,4,6,7],na_filter=True,index_col=False)
    if midas: sdf = sdf[sdf.sequence.str.startswith('AMD')] # filter by MIDAS sequences
    sdf = sdf.sort( ['sequence','entry'] )

    return sdf





def read_pid(filename=False):
    """Reads the SCOS-2000 pid.dat file and parses it for TM packet data. This
    is used to validate and identify valid telemetry frames. If no filename
    is given the global S2K path is searched for the PID file."""

    if not filename:
        filename = os.path.join(s2k_path, 'pid.dat')

    # cols = ('type','subtype','apid','sid','p2val','spid','description','unit','tpsd','dfh_size','time','inter','valid')
    cols = ('type','subtype','apid','sid','spid','description')
    pid=pd.read_table(filename,header=None,names=cols,usecols=[0,1,2,3,5,6])

    return pid

def read_pcf(filename=False):

    if not filename:
        filename = os.path.join(s2k_path, 'pcf.dat')

    cols = ('param_name', 'description', 'unit', 'ptc', 'pfc', 'width', 'cal_cat', 'cal_id','group')
    pcf = pd.read_table(filename,header=None,names=cols,usecols=[0,1,3,4,5,6,9,11,16], na_filter=True )
    pcf = pcf.convert_objects(convert_numeric=True)
    # pcf.cal_id = pcf.cal_id.astype(int)

    return pcf


def read_plf(filename=False):

    if not filename:
        filename = os.path.join(s2k_path, 'plf.dat')

    cols = ('param_name','spid','byte_offset','bit_offset')
    plf = pd.read_table(filename,header=None,names=cols,usecols=[0,1,2,3])
    plf = plf.convert_objects(convert_numeric=True)

    return plf


def read_caf(filename=False):
    """Read the CAF (numerical calibration curve) file"""

    if not filename:
        filename = os.path.join(s2k_path, 'caf.dat')

    cols = ('cal_id','cal_descrip','eng_fmt','raw_fmt','radix')
    caf = pd.read_table(filename,header=None,names=cols,usecols=[0,1,2,3,4])
    caf = caf.convert_objects(convert_numeric=True)
    caf.cal_id = caf.cal_id.astype(int)

    return caf


def read_pic(filename=False):
    """Read the PIC (packet identification criteria) table"""

    if not filename:
        filename = os.path.join(s2k_path, 'pic.dat')

    cols = ('pkt_type','pkt_subtype','sid_offset','sid_width')
    pic = pd.read_table(filename,header=None,names=cols,usecols=[0,1,2,3])
    pic = pic.convert_objects(convert_numeric=True)
    # caf.cal_id = caf.cal_id.astype(int)

    return pic



def read_cap(filename=False):
    """Read the CAP (numerical calibration curve definition) file"""

    if not filename:
        filename = os.path.join(s2k_path, 'cap.dat')

    cols = ('cal_id','raw_val','eng_val')
    cap = pd.read_table(filename,header=None,names=cols,usecols=[0,1,2])
    cap = cap.convert_objects(convert_numeric=True)

    return cap

def read_mcf(filename=False):
    """Read the MCF (polynomical calibration curve defintion) file"""

    if not filename:
        filename = os.path.join(s2k_path, 'mcf.dat')

    cols = ('cal_id','cal_descrip','a0','a1','a2','a3','a4')
    mcf = pd.read_table(filename,header=None,names=cols)
    mcf = mcf.convert_objects(convert_numeric=True)

    return mcf


def read_txf(filename=False):
    """Read the TXF (textual calibration) file"""

    if not filename:
        filename = os.path.join(s2k_path, 'txf.dat')

    cols = ('txf_id','txf_descr','raw_fmt')
    txf = pd.read_table(filename,header=None,names=cols,usecols=[0,1,2])
    txf = txf.convert_objects(convert_numeric=True)

    return txf

def read_txp(filename=False):
    """Read the TXP (textual calibration definition) file"""

    import os

    if not filename:
        filename = os.path.join(s2k_path, 'txp.dat')

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


def load_images(filename=None, data=False, sourcepath=None):
    """Load a messagepack file containing all image meta-data"""

    if filename is None:
        if data:
            filename = os.path.join(common.tlm_path, 'all_images_data.msg')
        else:
            filename = os.path.join(common.tlm_path, 'all_images.msg')

    images = pd.read_msgpack(filename)

    if sourcepath is not None:
        tm.pkts.filename = tm.pkts.filename.apply( lambda f: os.path.join(sourcepath, os.path.basename(f)) )

    return images

if __name__ == "__main__":

    print('WARNING: this module cannot be called interactively')
