"""
get_sentinel2_timeseries.py
Authors: Beth Delaney, Matt Payne
Script to access Sentinel-2 spectral reflectance values by querying GEE servers, iterating through each polygon sequentially.
"""

from datetime import datetime
import logging
import os
import sys
import time
import traceback
from typing import Optional

import ee
import geemap
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def main(project_name: str, aoi_path: str, start_date: str, end_date: str, out_directory: Optional[str]=None) -> None:
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
    out_directory : str, optional, by default None
        if supplied, the path of the directory to save images and csvs
    
    Returns
    -------
    None
    """

    # get logger to print stdout and stderr to
    logger = logging.getLogger(__name__)

    # check the start and end dates format are correct
    check_dates(start_date, end_date)

    # initialise EE Python API
    initialise(project_name=project_name)

    # read in shapefile
    gdf=gpd.read_file(aoi_path)

    # iterate over each polygon in the shapefile
    for index, row in gdf.iterrows():
        try:

            # convert vector AOI to GEE compliant geometry
            polygon_ee = convert_to_ee_geometry(gdf=row)

            # query the S2 Archive
            s2 = query_sentinel2_archive(aoi=polygon_ee, start_date=start_date, end_date=end_date)
    
            # if this argument is passed, then save indices and true colour image as png
            if out_directory and s2 is not None:
                save_image_thumbnails(s2.first(), out_directory)

                # get mean index values over time, from the AOI centroid and write to CSV
                extract_index_timeseries(s2, polygon_ee, out_directory)
            
            # add delay to prevent GEE server rate limiting being applied
            time.sleep(seconds=5)
                
        except Exception as e: # should make this more specific when I know the likely exceptions raised
            # configure traceback
            error_message = traceback.format_exc()
            logger.error(f"Could not process polygon {index}: {e} \n {error_message}")

    return

def extract_index_timeseries(image_collection: ee.ImageCollection, aoi: ee.Geometry.Polygon, out_directory: str) -> None:
    """

    Converts index values for every image in `image_collection`, converts these to a Pandas DataFrame and writes these to CSV.

    Parameters
    ----------
    image_collection : ee.ImageCollection
        Sentinel-2 ImageCollection with indices computed.
    aoi : ee.Geometry.Polygon
        GEE compliant geometry.
    out_directory : str
        absolute path to the out directory to write files to.
    """

    logger = logging.getLogger(__name__)

    # get the mean values per image of the pixels overlapped by the AOI for each Image in the ImageCollection
    reduced_collection = image_collection.map(lambda image: reduce_image_mean(image, aoi))

    # execute the computation on GEE servers
    sampled_values = reduced_collection.getInfo()["features"]

    logger.info(f"sampled_values : {sampled_values}")

    # use list comprehension to compactly construct a Pandas DataFrame from a dictionary of `sampled_values`  
    df = pd.DataFrame([{
        "feature_id": feature["id"],
        **feature["properties"]
    } for feature in sampled_values])

    # convert the list of dictionaries to a DataFrame for easier access
    df["datetime"] = pd.to_datetime(df["time"], unit="ms")
    df.drop(["time"], axis=1, inplace=True) # drop time in ms

    logger.info(df)

    df.to_csv(os.path.join(out_directory, "indices_values.csv"), index=False)

    return

def reduce_image_mean(image: ee.image, aoi: ee.Geometry.Polygon) -> ee.dictionary:
    """

    Calculate the mean for each band in `image` of the pixels within the geometry by reducing the image.

    Parameters
    ----------
    image : ee.image
        The image to reduce on.
    aoi : ee.Geometry.Polygon
        the geometry for the region to reduce within.

    Returns
    -------
    ee.Feature
        An GEE feature containing the mean value of all the pixels within a geometry.
    """

    reduced_values = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi,
        scale=10,
        crs="EPSG:32632",
        bestEffort=True # if too many pixels at scale arg, then try again with coarser scale
    )

    # get the date and time in ms
    reduced_values = reduced_values.set("time", image.get("system:time_start"))

    # assign tile_id
    reduced_values = reduced_values.set("tile_id", image.get("MGRS_TILE"))

    # assign id
    reduced_values = reduced_values.set("system:index", image.get("system:index"))

    # convert from dictonary to ee.Feature. The parent code is `map`, which expects an `ee.Image` or `ee.Feature` to be returned.
    return ee.Feature(None, reduced_values)

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

    logger = logging.getLogger(__name__)

    polygon_geometry = gdf.geometry
    polygon_geojson = polygon_geometry.__geo_interface__

    # extract coordinates from the geojson
    try:
        polygon_2d_coords = [
            [(x, y) for x, y in ring] for ring in polygon_geojson['coordinates']
        ]
    except ValueError:
        # if ValueError is raised becsause too many values per vertex are present in the json, strip away altitude (3rd dimension)
        polygon_2d_coords = [
                [(x, y) for x, y, _ in ring] for ring in polygon_geojson['coordinates']
            ]
        
    logger.info(f"polygon_2d_coords : {polygon_2d_coords}")

    return ee.Geometry.Polygon(polygon_2d_coords)

def query_sentinel2_archive(aoi: ee.Geometry.Polygon, start_date: str, end_date: str) -> ee.ImageCollection:
    """
    
    Query the Sentinel-2 Archive with an AOI and date range.


    Parameters
    ----------
    aoi : ee.Geometry.Polygon
        the AOI to query with
    start_date: str
        the start date, in the format "YYYY-MM-DD"
    end_date : str
        the end date, in the format "YYYY-MM-DD"
    
    Returns
    -------
    ee.ImageCollection
        ImageCollection of S2 images
    """

    logger = logging.getLogger(__name__)

    logger.info(f"Start Date: {start_date}")
    logger.info(f"End Date: {end_date}")

    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .sort("system:time_start")
        .filterMetadata('CLOUDY_PIXEL_PERCENTAGE', 'less_than', 20) \
        .map(lambda image: image.clip(aoi))
        .map(compute_indices)
        )
    
    # check if an empty ImgCol was returned
    if s2.size().getInfo() == 0:
        logger.warning("No images found for given query date and AOI")
        return None

    # prepare collection for duplicate removal by appending date
    logger.info(f"s2 type before duplicate date removal: {type(s2)}")
    s2 = s2.map(prepare_collection_for_duplicate_removal)
    # removal duplicates using the `date` property
    s2 = ee.ImageCollection(s2.distinct("date"))

    logger.info(f"s2 type after duplicate date removal: {type(s2)}")
    
    logger.info(s2.first().getInfo())

    # apply QC using "COPERNICUS/S2_CLOUD_PROBABILITY" as Hollstein req. L1C band B10
    s2_cloud_proba_col = (
        ee.ImageCollection("COPERNICUS/S2_CLOUD_PROBABILITY")
        .filterBounds(aoi)
        .filterDate(start_date, end_date)
        .sort("system:time_start")
        .map(lambda image: image.clip(aoi))
        )
    s2_with_cloud_proba = s2.map(lambda image: join_cloud_proba(image, s2_cloud_proba_col))

    return s2_with_cloud_proba

def prepare_collection_for_duplicate_removal(img: ee.image) -> ee.image:
    """
    Assigns a `date` property to each Sentinel-2 image in an `ee.ImageCollection`, for use in excluding duplicates using the built-in function `ImageCollection.distinct()`.

    Parameters
    ----------
    img : ee.image
        The Sentinel-2 image to assess

    Returns
    -------
    ee.image
        The Sentinel-2 image with `date` property appended
    
    Notes
    -----
    This function is used on an ImageCollection via `ImageCollection.map(prepare_collection_for_duplicate_removal)`.
    """

    logger = logging.getLogger(__name__)

    # get system:index to convert to date
    index = ee.String(img.get("system:index"))

    # convert to date, by slicing the system:index string
    date = ee.Date(index.slice(0, 8)).format("YYYYMMdd").slice(0, 8) # double slice because .format returns 14 characters, bad hack "solution" I know...

    return img.set("date", date)


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

def save_image_thumbnails(image: ee.Image, out_path_prefix: str) -> None:
    """

    Saves thumbnails for multiple indices and RGB as PNGs:
        - Normalised Difference Vegetation Index (NDVI)
        - Normalised Burn Ratio (NBR)
        - Normalised Difference Water Index (NDWI)
        - Soil Adjusted Vegetation Index (SAVI)
        - True Colour (RGB)

    Parameters
    ----------
    image : ee.Image
        an earth engine image, containing the precomputed indices and S2 RGB bands.
    out_path : str
        the path of the directory to write to.
    """

    logger = logging.getLogger(__name__)

    vis_params_dict = {
        "NDVI": {
            "max": 1,
            "min": 0,
            "palette": ["white", "green"]
        },
        "NDWI": {
            "max": 0.5,
            "min": -1,
            "palette": ["white", "blue"]
        },
        "NBR": {
            "max": 1,
            "min": -1,
            "palette": ["white", "red"]
        },
        "SAVI": {
            "max": 1,
            "min": -1,
            "palette": ["white", "orange"]
        },
        "RGB": {
            "max": 3000,
            "min": 0,
            "bands": ["B4", "B3", "B2"],
            "gamma": 1.4
        }
    }

    for image_name, vis_params in vis_params_dict.items():
        # wrap in try except block, this is where problems will occur as `geemap` will be calling EE servers
        try:
            # logic to account for RGB
            if image_name == "RGB":
                image_to_save = image.select(vis_params["bands"])
            else:
                image_to_save = image.select(image_name)

            # construct out_path
            out_path = os.path.join(out_path_prefix, f"{image_name}.png")

            # save the thumbnail
            geemap.get_image_thumbnail(image_to_save,
                                       out_img=out_path,
                                       vis_params=vis_params,
                                       format="png")
            logger.info(f"saved {image_name} to {out_path}")

        except Exception as e:
        #except ee.ee_exception.EEException() as e: # did not like except ee.ee_exception.EEException as e as "TypeError: catching classes that do not inherit from BaseException is not allowed"
            if "Image.select: Parameter 'input' is required" in str(e):
                logger.warning(f"No images found for {image_name}. Skipping thumbnail generation.")
            else:
                logger.error("Error generating thumbnail:")
                logger.exception(e)

    return

def compute_indices(image: ee.Image) -> ee.Image:
    """
    Computes the following indices on `image` and adds the indices back to `image`:
        - Normalised Difference Vegetation Index (NDVI)
        - Normalised Burn Ratio (NBR)
        - Normalised Difference Water Index (NDWI)
        - Soil Adjusted Vegetation Index (SAVI)

    Parameters
    ----------
    image : ee.Image
        Image to compute indices upon

    Returns
    -------
    ee.Image
        Image with indices appended
    """

    ndvi = image.normalizedDifference(["B8", "B4"]).rename("NDVI")
    nbr = image.normalizedDifference(["B12", "B11"]).rename("NBR")
    ndwi = image.normalizedDifference(["B3", "B8"]).rename("NDWI")
    # as SAVI is not a normalised Difference index, we use the expression function instead
    savi = image.expression(
        "(1 + L) * (NIR - RED) / (NIR + RED + L)", {
            "NIR": image.select("B8"),
            "RED": image.select("B4"),
            "L": 0.5
        }
    ).rename("SAVI")

    return image.addBands(ndvi).addBands(nbr).addBands(ndwi).addBands(savi)

def join_cloud_proba(img: ee.image, cloud_proba_col: ee.ImageCollection) -> ee.image:
    """
    Joins a Sentinel-2 image with its corresponding cloud probability mask.    

    Parameters
    ----------
    img : ee.image
        The image to compute the join on
    cloud_prob_col : ee.image_collection
        The S2_CLOUD_PROBABILITY ImageCollection to match.
    
    Returns
    -------
    img: ee.image
        The image with the cloud probability score per pixel added as a band
        
    Notes
    -----
    This function is used on an ImageCollection via `ImageCollection.map(lambda img: join_cloud_proba(img, cloud_prob_col))
    """

    # get the cloud probability image that matches `img`
    cloud_proba_img = ee.Image(cloud_proba_col.filter(ee.Filter.eq("system:index", img.get("system:index"))).first())
    
    return img.addBands(cloud_proba_img.rename("cloud_probability"))

if __name__ == "__main__":
    # if called from main, run
    logging.basicConfig(level=logging.INFO, filename=sys.argv[1], filemode="w", format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    main(project_name=sys.argv[2],
         aoi_path=sys.argv[3],
         start_date=sys.argv[4],
         end_date=sys.argv[5],
         out_directory=sys.argv[6])