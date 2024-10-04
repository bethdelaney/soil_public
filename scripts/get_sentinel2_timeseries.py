"""
get_sentinel2_timeseries.py
Authors: Beth Delaney, Matt Payne
Script to access Sentinel-2 spectral reflectance values by querying GEE servers.
"""

from datetime import datetime
import logging
from typing import Tuple
import os
import sys

import ee
import folium
import geojson
import geopandas as gpd
import json

import matplotlib.pyplot as plt

def main(project_name: str, aoi_path: str, start_date: str, end_date: str) -> None:
    """
    
    Parameters
    ----------
    project_name : str
        name of the Earth Engine (EE) project to initialise under
    aoi_path : str
        path to the aoi to inspect with
    start_date : str
        the start date, in the format "YYYY-MM-DD"
    end_date : str
        the end date, in the format "YYYY-MM-DD"
    
    Returns
    -------
    None
    """ 

    # get logger to print stdout and stderr to
    logger = logging.getLogger(__name__)

    check_dates(start_date, end_date)

    # initialise EE Python API
    initialise(project_name=project_name)

    # convert vector AOI to GEE compliant geometry
    polygon_ee = convert_to_ee_geometry(gdf=gpd.read_file(aoi_path))

    query_sentinel2_archive(aoi=polygon_ee, date_range=(start_date, end_date))

    return

def check_dates(start_date: str, end_date: str) -> None:
    """

    Checks the dates are in the correct format, "YYYY-MM-DD"

    Parameters
    ----------
    start_date : str
        the start date, in the format "YYYY-MM-DD"
    end_date : str
        the start date, in the format "YYYY-MM-DD"
    """

    logger = logging.getLogger(__name__)

    # using strftime to check whether the supplied string is a date in the correct format
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"incorrect start date format supplied, should be 'YYYY-MM-DD', got: {e}")
        raise

    try:
        datetime.strptime(end_date, "%Y-%m-%d")
    except ValueError as e:
        logger.error(f"incorrect end date format supplied, should be 'YYYY-MM-DD', got: {e}")
        raise

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

def query_sentinel2_archive(aoi: ee.Geometry.Polygon, date_range: Tuple[str, str]) -> None:
    """
    
    Query the Sentinel-2 Archive with an AOI and date range.


    Parameters
    ----------
    aoi : ee.Geometry.Polygon
        the AOI to query with
    date_range : Tuple[str, str]
        a Tuple of the start and end dates, in the format "YYYY-MM-DD"
    """

    logger = logging.getLogger(__name__)

    s2 = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED") \
        .filterDate(date_range[0], date_range[1]) \
        .filterBounds(aoi) \
        .sort("system:time_start") \
        .getInfo()
        #.map(convert_dn_to_reflectance)
    
    logger.info(s2)

    return

def convert_dn_to_reflectance(image: ee.Image) -> float:
    """

    Uses Sentinel-2 metadata to obtain scale factor to convert digital numbers of the image to actual reflectance.

    Parameters
    ----------
    image : ee.Image
        a Sentinel-2 image to retrieve metadata on

    Returns
    -------
    float
        reflectance values

    Notes
    -----
    This function only works when used on non-harmonised Sentinel-2 imagery.
    This function can be used in conjunction with `map` and an `ImageCollection`.
    """

    # get a list of scale factors
    scale_factor = image.get("REFLECTANCE_MULTI_BAND_")

    # apply scaling factor and convert to reflectance
    reflectance = image.multiply(scale_factor).toFloat()

    return reflectance

if __name__ == "__main__":
    # if called from main, run
    logging.basicConfig(level=logging.INFO, filename=sys.argv[1], filemode="w", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main(project_name=sys.argv[2], aoi_path=sys.argv[3], start_date=sys.argv[4], end_date=sys.argv[5])