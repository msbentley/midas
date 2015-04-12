#!/usr/bin/python
"""midas_daily.py - run daily data retrieval and extraction tasks"""

import os, sys, subprocess, tempfile, time
import pandas as pd
import numpy as np
from midas import common, dds_utils, ros_tm
from datetime import date, datetime, timedelta

log_dir = '/var/www/html/'
event_dir = '/var/www/html/events/'
commanding_dir = '/var/www/html/commanding'
image_dir = '/home/midas/images/'
tlm_dir = '/home/midas/tlm/'
kernel_dir = os.path.expanduser('~/Copy/midas/spice')
tempdir = tempfile.gettempdir()

isofmt = '%Y-%m-%dT%H%M%SZ'

def run_daily():

    new_data = False

    # set display to virtual desktop 1, served by xvfb
    os.environ["DISPLAY"] = ":1"

    # Dump all messages to a log file (append)
    sys.stdout = open(os.path.join(log_dir,'log.txt'), 'a')
    print('\n\nMIDAS daily start: %s' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    # Download any new SPICE kernels (spawn as background job)
    print('\nINFO: Downloading new or updated SPICE kernels...')
    status = subprocess.call( os.path.join(kernel_dir, 'get_kernels.sh'), shell=True )

    # Download any new NAVCAM images (via rsync)
    print('\nINFO: Downloading new NAVCAM images')
    get_navcam()

    # Check for new auto-pushed files from FDyn
    print('\nINFO: Retrieving FDyn auto-pushed files')
    dds_utils.get_fdyn_files()

    # Check for new OFPM-pushed files from SGS
    print('\nINFO: Retrieving RSGS OFPM pushed files\n')
    dds_utils.get_sgs_files()

    # Perform a pull of the ROS_SGS Git archive
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


    if new_data:
        print('\n\nGenerating meta-data spreadsheet for all images')
        # Read ALL TLM files so far and extract image meta-data to an XLS and CSV spreadsheet

        # M.S.Bentley 02/09/2014 - the server has only 1GB of memory and indexing ALL packets
        # causes problems. Breaking down indexing here per file, to see if that helps...

        import glob
        tm_files = sorted(glob.glob(os.path.join(tlm_dir,'TLM__MD*.DAT')))

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

        # Write to XLS, CSV and msgpack format
        images.to_excel(os.path.join(image_dir,'all_images.xls'), sheet_name='MIDAS images')
        images.to_csv(os.path.join(image_dir,'all_images.csv'))
        images.to_msgpack(os.path.join(tlm_dir, 'all_images.msg'))

        # Update the list of exposures
        print('\n\nINFO: updating table of exposures\n')
        for f in tm_files:
            tm.get_pkts(f, append=True)
            tm.pkts = tm.pkts[ ((tm.pkts.apid==1079) & ( (tm.pkts.sid==42553) | (tm.pkts.sid==42554) )) |
                ((tm.pkts.apid==1076) & (tm.pkts.sid==2)) ]
        exposures = tm.get_exposures(html=os.path.join(log_dir,'exposures.html'))

        # (Re-)build the packet index
        print('\n\nUpdating packet index\n')
        build_pkt_index()

        # Generate a list of html files corresponding to each ITL/EVF pair
        print('\n\nGenerating commanding summaries\n')
        generate_timelines()

        print('\n\Requesting latest time correlation packet (TCP)\n')
        tcorr = dds_utils.get_timecorr(outputpath=tlm_dir)

        # Use this to write a binary msgpack with all image data
        print('\n\nINFO: updating binary image index\n')
        tm = ros_tm.tm()
        tm.query_index(what='science')
        tm.pkts = tm.pkts[ (tm.pkts.sid==129) | (tm.pkts.sid==130) ]
        tm.get_images().to_hdf(os.path.join(tlm_dir, 'all_images_data.h5'), mode='w', key='images', format='f', complib='blosc', complevel=5)


    tunnel.kill()

    print('MIDAS daily end: %s' % (datetime.now().strftime('%Y-%m-%d %H:%M:%S')))


def show_scans():
    """Uses ros_tm.locate_scans() to iterate through all segments having image scans and plots the scans
    per segment"""

    images = ros_tm.load_images(data=False)
    segments = sorted(images.wheel_pos.unique())

    for seg in segments:
        pass

    return



def regenerate(what='all', files='TLM__MD_M*.DAT', from_index=False):
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
            tm.query_index(what='science')
            # Save BCR and GWY files
            if type(images)!=bool:
                ros_tm.save_bcr(images,os.path.join(image_dir, 'bcr/'), write_meta=True) # save images as BCRs + meta data
                ros_tm.save_gwy(images,os.path.join(image_dir, 'gwy/'), save_png=True, pngdir=os.path.join(image_dir, 'png/')) # and Gwyddion files

        else:
            import glob
            tm_files = sorted(glob.glob(os.path.join(tlm_dir,files)))
            if len(tm_files)==0:
                print('ERROR: no files matching pattern')
                return False
            for f in tm_files:
                tm=ros_tm.tm(f)
                images = tm.get_images(info_only=False)

                # Save BCR and GWY files
                if type(images)!=bool:
                    ros_tm.save_bcr(images,os.path.join(image_dir, 'bcr/'), write_meta=True) # save images as BCRs + meta data
                    ros_tm.save_gwy(images,os.path.join(image_dir, 'gwy/'), save_png=True, pngdir=os.path.join(image_dir, 'png/')) # and Gwyddion files

    if what=='all' or what=='meta':
        if from_index:
            tm.query_index(what='science')
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
        images.to_msgpack(os.path.join(tlm_dir, 'all_images.msg'))


    if what=='exposures' or what=='all':

        del(tm)
        tm = ros_tm.tm()

        if from_index:
            tm.query_index()
        else:
            import glob
            tm_files = sorted(glob.glob(os.path.join(tlm_dir,'TLM__MD_M*.DAT')))
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
            tm.query_index(what='events')
        else:
            import glob
            tm_files = sorted(glob.glob(os.path.join(tlm_dir,'TLM__MD_M*.DAT')))
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



def build_pkt_index(files='TLM__MD_M*.DAT', tm_index_file='tlm_packet_index.hd5'):
    """Builds an HDF5 (pandas/PyTables) table of the packet index (tm.pkts). This can be used for
    on-disk queries to retrieve a selection of packets as input to ros_tm.tm()."""

    import glob
    from pandas import HDFStore

    tm = ros_tm.tm()

    store = HDFStore(os.path.join(tlm_dir,tm_index_file), 'w', complevel=9, complib='blosc')
    table = 'pkts'

    tm_files = sorted(glob.glob(os.path.join(tlm_dir,files)))

    longest_filename = len(max(tm_files, key=len))

    if len(tm_files)==0:
        print('ERROR: no files matching pattern')
        return False

    for f in tm_files:
        tm = ros_tm.tm(f)
        # data_columns determines which columns can be queried - here use OBT for time slicing, APID for choosing data
        # source and filename to allow selection by MTP or STP.

        try:
            nrows = store.get_storer().nrows
        except:
            nrows = 0

        tm.pkts.index = pd.Series(tm.pkts.index) + nrows
        store.append(table, tm.pkts, format='table', min_itemsize={'filename': longest_filename}, data_columns=['obt','apid','filename'])

    store.close()


def generate_timelines(case='P'):

    from midas import planning

    itl_files = ros_tm.select_files(wildcard='ITLS_MD_*%c_RSUXPIYZ.itl' % case.upper(), directory=common.ros_sgs_path, recursive=True)
    evf_files = ros_tm.select_files(wildcard='EVF__MD_*%c_RSUXPIYZ.evf' % case.upper(), directory=common.ros_sgs_path, recursive=True)

    if len(itl_files) != len(evf_files):
        print('ERROR: number of ITL and EVF files does not match!')
        return

    for itl, evf in zip(itl_files, evf_files):

        htmlfile = os.path.join(commanding_dir,os.path.basename(itl).split('.')[-2]+'.html')

        try:
            planning.resolve_time(itl_file=itl, evf_file=evf, html=htmlfile)
        except:
            continue
    return


if __name__ == "__main__":

    # Called interactively - run run_daily()
    run_daily()
    # regenerate(what='images')
