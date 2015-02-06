# MIDAS 

This repo contains various python scripts that can be used to load and manipulate MIDAS telemetry and image files.

## Key modules

* ros_tm
 * Primary module for loading TLM files and extracting events, images, data.
 * The tm object is used to load one or more TLM files
* dds_utils
 * Generates queries for MIDAS (and other) data in the ESA DDS
 * Key functions are get_data and get_data_since
* planning
 * Module for generating ITL (commanding) files
* eps_utils
 * This module contains a generic EPS parser and utilities to run the EPS
 * Also plot utilities for EPS output files
* midas
 * Constants and generic routines
* midas_daily
 * Server scripts for auto-generating event logs, images etc.