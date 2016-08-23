#!/usr/bin/python
"""mesh_utils.py

Mark S. Bentley (mark@lunartech.org), 2015

A module containing a collection of utilities to work with MIDAS/AFM data as 3D
objects or meshes. A variety of external libraries may be used so check for
dependencies!"""

import sys
import numpy as np
import pandas as pd
from midas import ros_tm, common


def vtk_to_stl(vtkfile, stlfile):
    """Convert StructuredGrid in a .vtk files to an STL file"""

    try:
        import vtk
    except ImportError:
        print('ERROR: VTK module not found!')
        return None

    reader = vtk.vtkStructuredGridReader()
    reader.SetFileName(vtkfile)

    normals = vtk.vtkPolyDataNormals()
    normals.FlipNormalsOn()

    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputConnection(reader.GetOutputPort())

    # surface_filter.SetInputConnection(normals.GetOutput())

    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputConnection(surface_filter.GetOutputPort())



    writer = vtk.vtkSTLWriter()
    writer.SetFileName(stlfile)
    writer.SetInputConnection(triangle_filter.GetOutputPort())
    writer.Write()

    return


def image_to_vtk(images, vtkfile=None, scaling=1.0):
    """Accepts one or more MIDAS images and converts to VTK files. Images can
    be given as a Series (single image), a DataFrame (multiple images) or a
    list of scan file names.

    By default the files are named <scan_file>.vtk but for single files
    this can be overridden with the vtkfile= parameter.

    scaling=1.0 will scale the height to produce a 1:1:1 image"""

    if type(images) == pd.Series:
        images = pd.DataFrame(columns=images.to_dict().keys()).append(images)
    elif (type(images) == str) or (type(images) == list):
        images = ros_tm.load_images(data=True).query('scan_file==@images')

    if len(images)==0:
        print('ERROR: no matching images found!')
        return None

    for idx, image in images.iterrows():

        if len(images)>1 or vtkfile is None:
            vtkfile = image.scan_file + '.vtk'

        array_to_vtk(image['data']*common.zcal, vtkfile,
            xlen=image.xlen_um*1.e3, ylen=image.ylen_um*1.e3,
            name = image.scan_file)

    return




def array_to_vtk(data, vtkfile, xlen=1.0, ylen=1.0, name=''):
    """Writes a 2D numpy array to a structured grid legacy ASCII VTK file. """

    if type(data) != np.ndarray:
        print('ERROR: data must be in a numpy ndarray!')
        return None

    if len(data.shape) != 2:
        print('ERROR: data must be a 2D array!')
        return None

    # fldata = data.view('float')
    # fldata[:] = data
    fldata = np.array(data, dtype=float)

    with open(vtkfile, 'w') as f:
        f.write('# vtk DataFile Version 2.0\n')
        f.write(name+'\n')
        f.write('ASCII\n')
        f.write('DATASET STRUCTURED_GRID\n')
        f.write('DIMENSIONS %d %d 1\n' % (fldata.shape))
        f.write('POINTS %d float\n' % data.size)

        x = np.linspace(0, xlen, num=fldata.shape[1])
        y = np.linspace(0, ylen, num=fldata.shape[0])
        for (xidx,yidx), value in np.ndenumerate(np.rot90(fldata, k=-1)):
            f.write('%f %f %f\n' % (x[xidx], y[yidx], value))

    return
