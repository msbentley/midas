#!/usr/bin/python
"""midas_daily.py - run daily data retrieval and extraction tasks"""

import os, sys, subprocess, tempfile, time
import pandas as pd
import numpy as np
from midas import common, dds_utils, ros_tm
from datetime import date, datetime, timedelta

log_dir = '/var/www/html/'
event_dir = '/var/www/html/events/'
image_dir = '/home/midas/images/'
tlm_dir = '/home/midas/tlm/'
kernel_dir = os.path.expanduser('~/Copy/midas/spice')
tempdir = tempfile.gettempdir()

isofmt = '%Y-%m-%dT%H%M%SZ'

def run_daily():

    new_data = False

    # set display to virtual desktop 1, served by xvfb
    os.environ["DISPLAY"]=":1"

    # Dump all messages to a log file (append)
    sys.stdout = open(os.path.join(log_dir,'log.txt'), 'a')
    print('\n\nMIDAS daily start: %s' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    # Download any new SPICE kernels (spawn as background job)
    print('\nINFO: Downloading new or updated SPICE kernels...')
    status = subprocess.call( os.path.join(kernel_dir, 'get_kernels.sh'), shell=True )

    # Download any new NAVCAM images (via rsync)
    print('\nINFO: Downloading new NAVCAM images')
    # status = subprocess.call( '/home/mbentley/navcam/get_navcam.sh', shell=True )
    get_navcam()

    # Check for new auto-pushed files from FDyn
    print('\nINFO: Retrieving FDyn auto-pushed files')
    dds_utils.get_fdyn_files()

    # Check for new OFPM-pushed files from SGS
    print('\nINFO: Retrieving RSGS OFPM pushed files\n')
    dds_utils.get_sgs_files()

    # Perform a pull of the ROS_SGS Git arcchive
    print('\nINFO: Pulling the ROS_SGS git archive\n')
    tunnel = open_rostun()
    time.sleep(30)
    os.chdir(common.ros_sgs_path)
    pull = subprocess.call( 'proxychains git pull', shell=True )

    # Check for new completed observations and extract BCR and PNG images
    obs_filenames = dds_utils.get_new_observations(tlm_dir, mtpstp_dir=False, max_retry=10, retry_delay=5)

    for obs_file in obs_filenames:

        new_data = True
        print('\nProcessing TLM file: %s' % obs_file)

        obs_path,obs_fname = os.path.split(obs_file)
        tm = ros_tm.tm(obs_file) # open TM file

        # Extract and save images in various formats
        images = tm.get_images() # extract images
        if images is not None:
            ros_tm.save_bcr(images,os.path.join(image_dir, 'bcr/'), write_meta=True) # save images as BCRs + meta data
            ros_tm.save_gwy(images,os.path.join(image_dir, 'gwy/'), save_png=True, pngdir=os.path.join(image_dir, 'png/')) # and Gwyddion files

        # Extract and save event data as an HTML
        events = tm.get_events(info=True, html=os.path.join(event_dir,os.path.splitext(obs_fname)[0]+'_events.html'))

        # Check the sequence counter and generate a report per-observation
        missing_pkts = tm.check_seq(html=os.path.join(event_dir,os.path.splitext(obs_fname)[0]+'_seq_check.html'))

        # Attempt to free some memory (though still have to wait for garbage collection)
        del(tm)

    print('\nRequesting latest event data')
    # Request event data for the previous day (0-24)
    yesterday = date.today() - timedelta(days=1)
    start_time = datetime.combine(yesterday, datetime.min.time())
    end_time = start_time + timedelta(days=1)

    # event_file = 'TLM__1079_'+start_time.date().isoformat()
    tm_file, apid = dds_utils.get_data(start_time.isoformat(), end_time.isoformat(), outputpath=tempdir, apid=1079, delfiles=False)

    if tm_file: # file exists, i.e. MIDAS was on and generating events yesterday

        tm_file = tm_file[0]
        ev = ros_tm.tm(tm_file)
        events = ev.get_events(info=True, html=os.path.join(event_dir,'latest.html'))
        os.remove(tm_file)

    else:
        print('INFO: no event data available for %s' % (yesterday))

    if new_data:
        print('\nGenerating meta-data spreadsheet for all images')
        # Read ALL TLM files so far and extract image meta-data to an XLS and CSV spreadsheet

        # M.S.Bentley 02/09/2014 - the server has only 1GB of memory and indexing ALL packets
        # causes problems. Breaking down indexing here per file, to see if that helps...

        import glob
        tm_files = sorted(glob.glob(os.path.join(tlm_dir,'TLM__MD*.DAT')))

        # remove the cruise phase data until calculation of exposure time is fixed!
        tm_files.remove('TLM__MD_M000_S000_ALL_CRUISE_PHASE_____COUNT_00.DAT')

        tm = ros_tm.tm()
        for f in tm_files:
            tm.get_pkts(f, append=True)
            tm.pkts = tm.pkts[tm.pkts.apid==1084]

        images = tm.get_images(info_only=True)

        # Tidy up the metadata a little

        # remove the path from the absolute source filename
        images['filename'] = images['filename'].apply( lambda name: os.path.basename(name) )

        # Format the duration into a string
        images['duration'] = images['duration'].apply( lambda dur: "%s" % timedelta( seconds = dur / np.timedelta64(1, 's')) if not pd.isnull(dur) else '' )

        # rename the filename column to be clear that this is the TLM source file
        images.rename(columns={'filename':'tlm_file'}, inplace=True)

        # Add an index per scan (rather than per channel)
        # images.start_time

        # Write to XLS and CSV format
        images.to_excel(os.path.join(image_dir,'all_images.xls'), sheet_name='MIDAS images')
        images.to_csv(os.path.join(image_dir,'all_images.csv'))

        # Update the list of exposures
        print('INFO: updating table of exposures')
        for f in tm_files:
            tm.get_pkts(f, append=True)
            tm.pkts = tm.pkts[ ((tm.pkts.apid==1079) & ( (tm.pkts.sid==42553) | (tm.pkts.sid==42554) )) |
                ((tm.pkts.apid==1076) & (tm.pkts.sid==2)) ]
        exposures = tm.get_exposures(html=os.path.join(log_dir,'exposures.html'))

    tunnel.kill()

    print('MIDAS daily end: %s' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))



def regenerate(from_index=False, what='all', files='TLM__MD*.DAT'):
    """Regenerate images and/or event and packet loss data for all TM files"""

    what_types = ['all', 'images', 'events', 'exposures']

    what=what.lower()
    if what not in what_types:
        print('ERROR: what= must be one of %s' ", ".join(what_types) )
        return False

    tm = ros_tm.tm()

    if what=='all' or what=='images':
        # Load packet index, either from a pickle or individual TLM files
        if from_index:
            tm.load_index(os.path.join(tlm_dir,'packet_index.pkl'))
        else:
            import glob
            tm_files = sorted(glob.glob(os.path.join(tlm_dir,files)))
            if len(tm_files)==0:
                print('ERROR: no files matching pattern')
                return False
            for f in tm_files:
                # tm.get_pkts(f), append=True)
                # tm.pkts = tm.pkts[ (tm.pkts.apid==1084) & ((tm.pkts.sid==129) | (tm.pkts.sid==130)) ]
		tm=ros_tm.tm(f)

                # Extract image data
                images = tm.get_images(info_only=False)

                # Save BCR and GWY files
		if type(images)!=bool:
	                ros_tm.save_bcr(images,os.path.join(image_dir, 'bcr/'), write_meta=True) # save images as BCRs + meta data
        	        ros_tm.save_gwy(images,os.path.join(image_dir, 'gwy/'), save_png=True, pngdir=os.path.join(image_dir, 'png/')) # and Gwyddion files

        # images.drop('data', inplace=True, axis=1)

    if what=='all' or what=='meta':
        if from_index:
            tm.load_index(os.path.join(tlm_dir,'packet_index.pkl'))
        else:
            import glob
            tm_files = sorted(glob.glob(os.path.join(tlm_dir,files)))
            if len(tm_files)==0:
                print('ERROR: no files matchin pattern')
                return False
            for f in tm_files:
                tm.get_pkts(f, append=True)
                tm.pkts = tm.pkts[ (tm.pkts.apid==1084) & ((tm.pkts.sid==129) | (tm.pkts.sid==130)) ]

        # Extract image data
        images = tm.get_images(info_only=True)

        # Save the two meta-data spreadsheets
        images['filename'] = images['filename'].apply( lambda name: os.path.basename(name) )
        images['duration'] = images['duration'].apply( lambda dur: "%s" % timedelta( seconds = dur / np.timedelta64(1, 's')) if not pd.isnull(dur) else '' )
        images.rename(columns={'filename':'tlm_file'}, inplace=True)
        images.to_excel(os.path.join(image_dir,'all_images.xls'), sheet_name='MIDAS images')
        images.to_csv(os.path.join(image_dir,'all_images.csv'))

    if what=='exposures' or what=='all':

        del(tm)
        tm = ros_tm.tm()

        if from_index:
            tm.load_index(os.path.join(tlm_dir,'packet_index.pkl'))
        else:
            import glob
            tm_files = sorted(glob.glob(os.path.join(tlm_dir,'TLM__MD*.DAT')))
            for f in tm_files:
                tm.get_pkts(f, append=True)
                tm.pkts = tm.pkts[tm.pkts.sid.isin([2, 42553, 42554])]

        exposures = tm.get_exposures(html=os.path.join(log_dir,'exposures.html'))

    if what=='all' or what=='events':
        # Now process the events and
        del(tm)
        tm = ros_tm.tm()
        # Load packet index, either from a pickle or individual TLM files
        if from_index:
            tm.load_index(os.path.join(tlm_dir,'packet_index.pkl'))
        else:
            import glob
            tm_files = sorted(glob.glob(os.path.join(tlm_dir,'TLM__MD*.DAT')))
            for f in tm_files:
                tm.get_pkts(f)
                # tm.pkts = tm.pkts[tm.pkts.apid==1079]
                obs_path,obs_fname = os.path.split(f)
                events = tm.get_events(info=True, html=os.path.join(event_dir,os.path.splitext(obs_fname)[0]+'_events.html'))
                missing_pkts = tm.check_seq(html=os.path.join(event_dir,os.path.splitext(obs_fname)[0]+'_seq_check.html'))



def open_rostun():

    import subprocess
    tunnel = subprocess.Popen( ['ssh', '-N', 'rosetta_temp@ssh.esac.esa.int', '-D1080'] )

    return tunnel



def get_navcam():

    import pexpect
    p=pexpect.spawn('rsync -rv --ignore-existing -e ssh midas_navcam@ssh.esac.esa.int:/lhome/midas_navcam/ /media/data/navcam')
    p.expect('assword')
    p.sendline('OlERlE8k')
    p.expect(pexpect.EOF, timeout=5*60)
    # print(p.before)

if __name__ == "__main__":

    # Called interactively - run run_daily()
    run_daily()
    # regenerate(what='images')
