#!/usr/bin/python
"""
followup.py

Mark S. Bentley (mark@lunartech.org), 2014

This module deals with the setup of followup images for common. It accepts images read
by ros_tm.get_images() and allows the user to select areas for follow-up. Meta data
are then used to create the parameters necessary to zoom on this region, wit a variety
of options for the image resolution etc.
"""

from midas import ros_tm, common, scanning
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import  RectangleSelector
from matplotlib.patches import Rectangle
from pylab import rcParams
from matplotlib import gridspec
# from datetime import timedelta



class followup:

    def update_display(self):
        """Update the position of the rectangle on the plot to show the position of the scan"""

        # Display using a rectangle patch (unfilled)
        if self.area: self.area.remove()
        self.area = self.ax1.add_patch(Rectangle((self.x_orig, self.y_orig), self.width, self.height, fill=False, linewidth=2, color='w'))

        # Also display a zoomed version of the ROI on ax2
        self.ax2.set_xlim(self.x_orig, self.x_orig+self.width)
        self.ax2.set_ylim(self.y_orig+self.height, self.y_orig)

        max_step_nm = max( self.x_step*self.xcal*1.e3, self.y_step*self.ycal*1.e3 )
        zretract_nm = max_step_nm * self.safety
        self.zret = int(np.around(zretract_nm / common.zcal))

        duration = int(scanning.calc_duration(self.x_pix, self.y_pix, 1, self.zret, zsettle=50, xysettle=50, zstep=4, avg=1))
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)

        displayText = 'Origin: (%i,%i), centre: (%i, %i)\nWidth: %i pix = %i DAC (%3.2f microns)\nHeight: %i pix = %i DAC (%3.2f microns)\n\
X step size: %i (%3.2f nm)\nY step size: %i (%3.2f nm)\nDuration: %02d:%02d:%02d' % \
            (self.x_orig, self.y_orig, self.x_orig+self.width/2, self.y_orig+self.height/2,
            self.x_pix, self.width, self.width*self.xcal, self.y_pix, self.height, self.height*self.ycal,
            self.x_step, self.x_step*self.xcal*1.e3,self.y_step, self.y_step*self.ycal*1.e3, hours, minutes, seconds)
        self.textbox.set_text(displayText)

        plt.draw()


    def __init__(self, image, safety=2.0):
        """Accepts an image dataframe, displays it, and allows the user to interactively
        define a rectangle defining the new scan region."""

        self.x_orig = 32000
        self.y_orig = 32000
        self.width = 0
        self.height = 0
        self.area = False
        self.x_step = 0
        self.y_step = 0
        self.xcal = self.ycal = 0
        self.x_pix = 32
        self.y_pix = 32
        self.zret = 0
        self.safety = safety

        rcParams['toolbar'] = 'None' # hide the MPL toolbar


        def selector(event):

            # Arrow keys shift the defined origin by mutliples of the X/Y step
            if event.key in ['left'] and selector.RS.active:
                self.x_orig -= self.x_step
                self.update_display()
            if event.key in ['right'] and selector.RS.active:
                self.x_orig += self.x_step
                self.update_display()
            if event.key in ['up'] and selector.RS.active:
                self.y_orig -= self.y_step
                self.update_display()
            if event.key in ['down'] and selector.RS.active:
                self.y_orig += self.y_step
                self.update_display()

            # A/D change the width by 32 pixels, X/W the height
            if event.key in ['a' or 'A'] and selector.RS.active:
                if self.x_pix > 32:
                    self.x_pix = self.x_pix - 32
                    self.x_orig += 16 * self.x_step
                    self.width = self.x_pix * self.x_step
                    self.update_display()
            if event.key in ['d' or 'D'] and selector.RS.active:
                if self.x_pix < 512:
                    self.x_pix = self.x_pix + 32
                    self.x_orig -= 16 * self.x_step
                    self.width = self.x_pix * self.x_step
                    self.update_display()
            if event.key in ['x' or 'X'] and selector.RS.active:
                if self.y_pix > 32:
                    self.y_pix -= 32
                    self.y_orig += 16 * self.y_step
                    self.height = self.y_pix * self.y_step
                    self.update_display()
            if event.key in ['w' or 'W'] and selector.RS.active:
                if self.y_pix < 512:
                    self.y_pix += 32
                    self.y_orig -= 16 * self.y_step
                    self.height = self.y_pix * self.y_step
                    self.update_display()

            # Also change the step size, maintaining approximately the same area
            if event.key in ['1'] and selector.RS.active:
                if self.x_step > 1:
                    self.x_step -= 1
                    width = self.x_pix * self.x_step
                    self.x_orig += (self.width-width)/2
                    self.width = width
                    self.update_display()
            if event.key in ['2'] and selector.RS.active:
                if self.x_step < 64:
                    self.x_step += 1
                    width = self.x_pix * self.x_step
                    self.x_orig -= (width-self.width)/2
                    self.width = width
                    self.update_display()
            if event.key in ['3'] and selector.RS.active:
                if self.y_step > 1:
                    self.y_step -= 1
                    height = self.y_pix * self.y_step
                    self.y_orig += (self.height-height)/2
                    self.height = height
                    self.update_display()
            if event.key in ['4'] and selector.RS.active:
                if self.y_step < 64:
                    self.y_step += 1
                    height = self.y_pix * self.y_step
                    self.y_orig += (height-self.height)/2
                    self.height = height
                    self.update_display()
            if event.key in ['enter' or '5'] and selector.RS.active:
                print(self.x_orig, self.y_orig, self.x_orig+self.width/2, self.y_orig+self.height/2, self.x_pix, self.y_pix, self.x_step, self.y_step, self.zret)



        def onselect(eclick, erelease):
            """Uses the start and end points of the bounding box to select zoom region"""

            xpos = min(eclick.xdata,erelease.xdata)
            ypos = min(eclick.ydata,erelease.ydata)

            width = abs(erelease.xdata - eclick.xdata)
            height = abs(erelease.ydata - eclick.ydata)

            if (width==0 or height==0):
                width=self.width
                height=self.height
                xpos = xpos-width/2.
                ypos = ypos-height/2.

            if image.x_closed:
                xpos_dac = common.closed_to_open(xpos, ypos)[0]
            if image.y_closed:
                ypos_dac = common.closed_to_open(xpos, ypos)[1]

            # Now have raw DAC values for X and Y origin and width in open/closed steps
            self.x_orig = int(round(xpos))
            self.y_orig = int(round(ypos))

            # Number of pixels must be a multiple of 32 - choose nearest
            self.x_pix = int(round(float(width)/self.x_step/32.,0)) * 32
            self.y_pix = int(round(float(height)/self.y_step/32.,0)) * 32
            self.width = self.x_pix * self.x_step
            self.height = self.y_pix * self.y_step

            # Draw a rectangle representing this scan on the original image
            self.update_display()

        ###########################
        # APPLICATION STARTS HERE #
        ###########################

        # Set up a matplotlib figure with 4 subplots
        # Use GridSpec to set up alignment
        gs = gridspec.GridSpec(2, 2, height_ratios=[4, 1])
        fig = plt.figure(facecolor='white')
        fig.canvas.set_window_title('MIDAS Scan Followup')
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(gs[1,:])
        ax3.axis('off')
        fig.subplots_adjust(hspace=0.0, wspace=0.1, left=0.0, bottom=0.0, right=0.97, top=0.96)

        # Load an image, subtract plane, display with pixel units
        self.fig, self.ax1 = ros_tm.show(image, units='dac', planesub='plane', title=False, fig=fig, ax=ax1)
        plt.setp( self.ax1.xaxis.get_majorticklabels(), rotation=70 )
        self.ax1.xaxis.get_major_formatter().set_useOffset(False)
        self.ax1.yaxis.get_major_formatter().set_useOffset(False)

        self.fig, self.ax2 = ros_tm.show(image, units='dac', planesub='plane', title=False, fig=fig, ax=ax2)
        plt.setp( self.ax2.xaxis.get_majorticklabels(), rotation=70 )
        self.ax2.xaxis.get_major_formatter().set_useOffset(False)
        self.ax2.yaxis.get_major_formatter().set_useOffset(False)

        # Display key parameters in a text field on ax3
        ax3.xaxis.set_visible(False)
        ax3.yaxis.set_visible(False)
        self.textbox = ax3.text(0.05, 0.95, '', fontsize=12, verticalalignment='top')

        # Set up a selector and connect to to the event handler
        rectprops = dict(facecolor='red', edgecolor='white', alpha=1.0, fill=False, linewidth=2.0)
        selector.RS = RectangleSelector(self.ax1, onselect, drawtype='box', rectprops=rectprops)
        plt.connect('key_press_event', selector)

        # Set default parameters based on the image size
        width = ax1.get_xlim()[1]-ax1.get_xlim()[0]
        height = ax1.get_ylim()[1]-ax1.get_ylim()[0]
        self.step = max(width,height) * (2./100.)
        self.width = self.height = 10.*self.step

        self.x_orig = image.x_orig
        self.y_orig = image.y_orig

        self.x_step = image.x_step
        self.y_step = image.y_step

        self.xcal = common.xycal['closed'] if image.x_closed else common.xycal['open']
        self.xcal /= 1000.
        self.ycal = common.xycal['closed'] if image.y_closed else common.xycal['open']
        self.ycal /= 1000.

        self.update_display()
        plt.show()
