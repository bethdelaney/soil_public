"""
get_sentinel2_timeseries.py
Authors: Beth Delaney, Matt Payne
Script to access Sentinel-2 spectral reflectance values by querying GEE servers.
"""

import logging
import os
import sys

import ee
import folium
import geojson
import geopandas as gpd
import json

import matplotlib.pyplot as plt

def main(project_name: str, aoi_path: str) -> None:
    """
    
    Parameters
    ----------
    project_name : str
        name of the Earth Engine (EE) project to initialise under
    aoi_path : str
        path to the aoi to inspect with
    
    Returns
    -------
    None
    """ 

    # get logger to print stdout and stderr to
    logger = logging.getLogger(__name__)

    # initialise EE Python API
    initialise(project_name=project_name)

    # convert vector AOI to GEE compliant geometry
    polygon_ee = convert_to_ee_geometry(gdf=gpd.read_file(aoi_path))



    return

def initialise(project_name: str) -> None:
    """
    Initialises programmatic access to Earth Engine via the Python API, once per session.

    Parameters
    ----------
    project_name : str
        name of the Google Cloud Project with the GEE API enabled.

    Returns
    -------
    None
    """
    
    ee.Authenticate()
    ee.Initialize(project=project_name)

    return

def convert_to_ee_geometry(gdf: gpd.GeoDataFrame) -> ee.Geometry.Polygon:
    """

    Converts a GeoPandas GeoDataFrame to an GEE compliant geometry.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        containing the geometry information we are interested in

    Returns
    -------
    ee.Geometry.Polygon
        GEE compliant geometry
    """

    polygon_geometry = gdf.geometry[0]
    polygon_geojson = polygon_geometry.__geo_interface__

    
    # Extract 2D coordinates by stripping the third dimension (altitude)
    polygon_2d_coords = [
        [(x, y) for x, y, _ in ring] for ring in polygon_geojson['coordinates']
    ]

    return ee.Geometry.Polygon(polygon_2d_coords)


if __name__ == "__main__":
    # if called from main, run
    logging.basicConfig(level=logging.INFO, filename=sys.argv[1], filemode="w", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main(project_name=sys.argv[2], aoi_path=sys.argv[3])