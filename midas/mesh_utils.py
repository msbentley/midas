#!/usr/bin/python
"""mesh_utils.py

Mark S. Bentley (mark@lunartech.org), 2015

A module containing a collection of utilities to work with MIDAS/AFM data as 3D
objects or meshes. A variety of external libraries may be used so check for
dependencies!"""

import sys


def vtk_to_stl(vtkfile, stlfile):
    """Convert StructuredGrid in a .vtk files to an STL file"""

    try:
        import vtk
    except ImportError:
        print('ERROR: VTK module not found!')
        return None

    reader = vtk.vtkStructuredGridReader()
    reader.SetFileName(vtkfile)

    surface_filter = vtk.vtkDataSetSurfaceFilter()
    surface_filter.SetInputConnection(reader.GetOutputPort())

    triangle_filter = vtk.vtkTriangleFilter()
    triangle_filter.SetInputConnection(surface_filter.GetOutputPort())

    writer = vtk.vtkSTLWriter()
    writer.SetFileName(stlfile)
    writer.SetInputConnection(triangle_filter.GetOutputPort())
    writer.Write()

    return
