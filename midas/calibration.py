#!/usr/bin/python
# -*- coding: utf-8 -*-
"""calibration.py

Mark S. Bentley (mark@lunartech.org), 2017

A module containing calibration related routines for MIDAS data.
"""

import os
from midas import ros_tm, common
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import logging
log = logging.getLogger(__name__)


def calibrate_xy(scan_file, gwy_path=common.gwy_path, outpath='.', printdata=False, restore=False,
        radius=100, colour='white', process_gwy=False, overwrite=False, **kwargs):
    """Accepts an image (scan) name, displays it and allows the user to click calibration positions, which
    are logged to a filename in CSV format.

    Left clicking in the plot adds a new point. Existing points can be dragged to
    a new location with the left button, and a right click deletes a point."""

    class Calibrate:

        def __init__(self, scan_file, gwy_path, outpath, printdata, restore, radius, colour, process_gwy, overwrite, **kwargs):

            tolerance = 10
            self.xy = []
            self.row = []
            self.current_row = 1
            self.process_gwy = process_gwy
            self.gwy_file = os.path.join(gwy_path, scan_file+'.gwy')

            if restore:
                overwrite = True

            csvfile = os.path.join(outpath, scan_file+'_cal.csv')

            if (os.path.isfile(csvfile)) and (not overwrite) and (not restore):
                log.error('CSV file %s already exists. Set overwrite=True or change the output path' % csvfile)
                return None
            elif restore and (not os.path.isfile(csvfile)):
                log.error('Cannot read CSV file %s for restore' % csvfile)
                return None

            if process_gwy:
                if not os.path.isfile(self.gwy_file):
                    log.error('Gwyddion file %s not found' % self.gwy_file)
                    return None

            image = ros_tm.load_images(data=True).query('scan_file==@scan_file').squeeze()
            if len(image)==0:
                log.error('could not find image %s' % scan_file)
                return None

            self.fig, self.ax = ros_tm.show(image, units='real', planesub='poly', title=True, cbar=True,
                        shade=False, show_fscans=False, show=False, rect=None)

            self.points = self.ax.scatter([], [], s=radius, facecolors='none', edgecolors=colour,
                        picker=tolerance, animated=True, **kwargs)

            if restore:
                row, x, y = np.loadtxt(csvfile, delimiter=',', unpack=True)
                self.row = row.astype(int).tolist()
                self.xy = zip(x.tolist(),y.tolist())
                self.update(blit=False)
                log.info('Data restored from CSV file %s' % csvfile)

            try:
                self.f = open(csvfile, 'w')
            except IOError as (errno, strerror):
                print "ERROR: I/O error({0}): {1}".format(errno, strerror)
                return None

            self.update_title()

            connect = self.fig.canvas.mpl_connect
            self.cid = connect('button_press_event', self.onclick)
            self.close_cid = connect('close_event', self.onclose)
            self.draw_cid = connect('draw_event', self.grab_background)
            self.key_cid = connect('key_press_event', self.onkey)

        def onkey(self, event):

            if event.key=='up':
                self.current_row -= 1
                if self.current_row < 1:
                    self.current_row = 1
            elif event.key=='down':
                self.current_row += 1
            else:
                return

            self.update()

        def onclick(self, event):

            if self.fig.canvas.toolbar._active is not None:
                return

            contains, info = self.points.contains(event)

            if contains:
                i = info['ind'][0]
                if event.button == 1: # left button
                    self.start_drag(i)
                elif event.button == 3: # right button
                    self.delete_point(i)
            else:
                self.add_point(event)

        def update_title(self):
            self.fig.canvas.set_window_title('Current row: %d' % self.current_row)

        def update(self, blit=True):
            self.points.set_offsets(self.xy)
            self.update_title()
            if blit:
                self.blit()

        def add_point(self, event):
            self.xy.append([event.xdata, event.ydata])
            self.row.append(self.current_row)
            if printdata:
                print 'INFO: added point: x = %3.3f, y = %3.3f to row %d' % (event.xdata, event.ydata, self.current_row)
            self.update()

        def delete_point(self, i):
            if printdata:
                print 'INFO: removed point: x = %3.3f, y = %3.3f from row %d' % (self.xy[i][0], self.xy[i][1], self.row[i])
            self.xy.pop(i)
            self.row.pop(i)
            self.update()

        def start_drag(self, i):
            self.drag_i = i
            connect = self.fig.canvas.mpl_connect
            cid1 = connect('motion_notify_event', self.drag_update)
            cid2 = connect('button_release_event', self.end_drag)
            self.drag_cids = [cid1, cid2]

        def drag_update(self, event):
            self.xy[self.drag_i] = [event.xdata, event.ydata]
            self.update()

        def end_drag(self, event):
            if printdata:
                log.info('point moved to: x = %3.3f, y = %3.3f on row %d' % (self.xy[self.drag_i][0], self.xy[self.drag_i][1], self.row[self.drag_i]))
            for cid in self.drag_cids:
                self.fig.canvas.mpl_disconnect(cid)

        def safe_draw(self):
            canvas = self.fig.canvas
            canvas.mpl_disconnect(self.draw_cid)
            canvas.draw()
            self.draw_cid = canvas.mpl_connect('draw_event', self.grab_background)

        def onclose(self, event):
            disconnect = self.fig.canvas.mpl_disconnect
            disconnect(self.cid)
            disconnect(self.close_cid)
            disconnect(self.draw_cid)
            disconnect(self.key_cid)
            self.writedata()

            if self.process_gwy:
                coeffs = xyz_to_coeff()
                if coeffs is not None:
                    # gwy_utils.polynomial_distort(self.gwy_file, channel=None, new_chan='corrected', coeffs=coeffs)
                    pass


        def grab_background(self, event=None):
            self.points.set_visible(False)
            self.safe_draw()
            self.background = self.fig.canvas.copy_from_bbox(self.fig.bbox)
            self.points.set_visible(True)
            self.blit()

        def blit(self):
            self.fig.canvas.restore_region(self.background)
            self.ax.draw_artist(self.points)
            self.fig.canvas.blit(self.fig.bbox)

        def writedata(self):
            for row, (x,y) in sorted(zip(self.row, self.xy), key=lambda k: [k[0],k[1]]):
                self.f.write('%d, %3.3f, %3.3f\n' % (row, x, y))
            self.f.close()

    cal = Calibrate(scan_file, gwy_path=gwy_path, outpath=outpath, printdata=printdata, restore=restore,
        radius=radius, colour=colour, process_gwy=process_gwy, overwrite=overwrite, **kwargs)

    plt.show(block=True)

    return


def read_all_caldata(calpath='.', filespec='SCAN*_cal.csv', recursive=False):
    """Reads all calibration files matching filespec in calpath and returns a
    pandas dataframe with row and column indices, x and y differences, and
    the corresponding scan filename."""

    calfiles = common.select_files(filespec, directory=calpath, recursive=recursive)

    caldata = pd.DataFrame()
    scan_file = []

    for calfile in calfiles:
        calframe = cal_stats(read_caldata(calfile, format='pandas'))
        if calframe is None:
            continue
        calframe['scan_file'] = scan_from_calfile(calfile)
        caldata = caldata.append(calframe)

    return caldata


def summarise_xy(image_fields=None, *kwargs):
    """Simple wrapper to run read_all_caldata() and take the output and
    summarise key statistics per scan. Input parameters are identical
    to those of read_all_caldata()"""

    images = ros_tm.load_images(data=False)

    if image_fields is None:
        image_fields = ['scan_file', 'x_step', 'y_step', 'fast_dir', 'x_closed', 'y_closed']
    else:
        if 'scan_file' not in image_fields:
                image_fields.append('scan_file')

    caldata = read_all_caldata(*kwargs)
    mean = caldata.groupby('scan_file')['xdiff', 'ydiff'].mean()
    mean.columns = ['xmean', 'ymean']
    std = caldata.groupby('scan_file')['xdiff', 'ydiff'].std()
    std.columns = ['xstd', 'ystd']
    summary = mean.join(std)
    summary = summary.merge(images[image_fields], how='left', right_on='scan_file', left_index=True)
    summary.set_index('scan_file', inplace=True)

    return summary


def read_caldata(calfile, format='numpy'):
    """Loads and returns calibration data from a CSV file. By default
    a numpy structured array is returned, corresponding exactly to
    the input format. If format=='array' a structured 2D array is returned with
    the X and Y values, respectively. If format=='pandas' a DataFrame
    is returned with X and Y values as separate Series, along with
    row and column Series."""

    caldata = np.loadtxt(calfile, delimiter=',', dtype={
        'names': ('row', 'x', 'y'),
        'formats': ('i2', 'f2', 'f2')})
    caldata.sort(order=['row', 'x'])

    rows = np.unique(caldata['row'])
    num_rows = len(rows)
    num_pts = []

    for row in range(1, num_rows+1):
        num_pts.append(len(caldata[caldata['row'] == row]))
    log.debug('row lengths in data: ' + ", ".join(str(val) for val in list(set(num_pts))))
    if len(set(num_pts)) > 1:
        log.error('not all rows in %s are the same length' % os.path.basename(calfile))
        return None

    num_cols = num_pts[0]
    log.debug('calibration file has %d rows and %d columns' % (num_rows, num_cols))

    if format == 'numpy':
        pass
    elif format == 'array':
        caldata = caldata[['x', 'y']].reshape(num_rows, num_cols)
    elif format == 'pandas':
        caldata = pd.DataFrame(caldata)
        caldata['col'] = None
        for row in rows:
            caldata['col'].loc[caldata[caldata.row == row].index] = range(1, len(caldata[caldata.row == row])+1)
            caldata = caldata[['row', 'col', 'x', 'y']]
    else:
        log.error('invalid format specification - should be one of numpy, array, or pandas')
        return None

    return caldata


def cal_stats(caldata, verbose=False):
    """Calculates simple statistical data about the calibration data, read with
    read_caldata(). This routine accepts either read_caldata(format='array') or
    format='pandas' data types, and the return is different for each case:

    format='array': returns two tuples (xmean, xstd), (ymean, ystd)
    format='pandas': returns a copy of the DataFrame with xdiff and ydiff added

    Output is logged if verbose=True"""

    if type(caldata) == np.ndarray:

        if len(caldata.shape) == 1:
            log.error('cal_stats works with data from read_caldata with format="array" or "pandas" only')
            return None

        mean_x = np.diff(caldata['x'], axis=1).mean()
        std_x = np.std(np.diff(caldata['x'], axis=1))
        mean_y = np.diff(caldata['y'], axis=0).mean()
        std_y = np.std(np.diff(caldata['y'], axis=0))

        caldata = (mean_x, std_x), (mean_y, std_y)

    elif type(caldata) == pd.DataFrame:
        caldata['xdiff'] = caldata.groupby('row').x.diff()
        caldata['ydiff'] = caldata.groupby('col').y.diff()
        mean_x = caldata.xdiff.mean()
        std_x = caldata.xdiff.std()
        mean_y = caldata.ydiff.mean()
        std_y = caldata.ydiff.std()

    if verbose:
        log.info(u'Mean X separation: %3.2f µm (SD %3.2f µm)' % (mean_x, std_x))
        log.info(u'Mean Y separation: %3.2f µm (SD %3.2f µm)' % (mean_y, std_y))

    return caldata


def scan_from_calfile(calfile):
    """Accepts a calibration filename in the standard format (including path)
    and returns the corresponding scan name (with no path)"""

    return os.path.basename(calfile).lower().split('_cal.csv')[0].upper()


def show_calfile(calfile, use_image=False):
    """Plots the points in the given calibration (CSV) file. If use_image=True
    the original scan will also be displayed"""

    scan_file = scan_from_calfile(calfile)

    caldata = read_caldata(calfile)
    fig, ax = plt.subplots()

    ax.set_xlabel('X (microns)')
    ax.set_ylabel('Y (microns)')
    ax.set_title('Calibration data for %s' % scan_file, fontsize=10)


    if use_image:
        ros_tm.show(scan_file, units='real', planesub='poly', title=None, cbar=True,
            show=False, fig=fig, ax=ax)
        colour = 'cyan'
    else:
        colour = 'black'
        ax.invert_yaxis()
        # ax.set_ylim(top=0.)
        # ax.set_xlim(left=0.)

    ax.scatter(caldata['x'], caldata['y'], s=3, c=colour)

    plt.show()

    return


def fit_row_col(caldata):
    """Performs a linear fit to each independant row and colum"""

    pass

    return None


def calfile_to_coeff(calfile, scan_file=None):
    """Accepts a calibration file produced by calibrate_xy() and produced a set of
    normalised coefficients suitable for applying a polynomial distortion correction.

    If scan_file=None the calibration filename is assumed to be of the form:

    scan_file_cal.csv

    and scan_file will be extracted and used. Otherwise specify via scan_file="""

    caldata = load_caldata(calfile)

    D = np.array([x*0+1, x, x**2, x**3, y, y*x, y*x**2, y**2, y**2*x, y**3]).T
    B = z
    coeff, r, rank, s = np.linalg.lstsq(D, B)

    return coeff
