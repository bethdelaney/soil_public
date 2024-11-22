"""
hollstein.py
Script for applying Hollstein et al. (2016) concepts for cloud, shadow, snow, water, cirrus and water masking to a Decision Tree.
Original Author: Rodrigo E. Principe
Using code (under MIT Licence) from a deprecated version of `geetools`: https://github.com/gee-community/geetools/tree/27d600b19bf179a734b69b39ddd3226b70079a4c
"""

import ee
from typing import List

def difference(img: ee.image, a: str, b: str):
    """
    Calculates the difference between two bands in `img`.

    Parameters
    ----------
    img: ee.image
        the image to process
    a: str
        The name of the first band.
    b: str
        The name of the second band.

    Returns
    -------
    An `ee.image` with the difference between the specified bands.
    """

    difference_image = img.select(a).subtract(img.select(b))

    return difference_image

def ratio(img: ee.image, a: str, b: str):
    """
    Calculates the ratio between two bands in `img`.

    Parameters
    ----------
    img: ee.image
        the image to process
    a: str
        The name of the first band.
    b: str
        The name of the second band.

    Returns
    -------
    An `ee.image` with the ratio between the specified bands.
    """

    ratio_image = img.select(a).divide(img.select(b))

    return ratio_image

def hollstein_S2(img: ee.image, options: List=["cloud", "snow", "shadow", "water", "cirrus"], name: str='hollstein', updateMask: bool=True, addBands: bool=True) -> ee.image:
    """

    Applies the Hollstein et al. (2016) cloud, shadow, snow, water, cirrus and water
    to a Sentinel-2 image.

    Parameters
    ----------
    img : ee.image
        The ee.Image to process.
    options : List, optional
        A list of strings specifying the masks to compute, by default ["cloud", "snow", "shadow", "water", "cirrus"]
    name : str, optional
        The name to give the resulting mask band, by default 'hollstein'
    updateMask : bool, optional
        Whether to update the image mask with the result, by default True
    addBands : bool, optional
        Whether to add the result bands to the image, by default True

    Returns
    -------
    ee.image
        An ee.Image with the mask applied and/or added as bands.
    """
    if options is None:
        options = ['cloud', 'snow', 'shadow', 'water', 'cirrus']

    def compute_dt(img):
        # 1
        b3 = img.select('B3').lt(3190)

        # 2
        b8a = img.select('B8A').lt(1660)
        r511 = ratio(img, 'B5', 'B11').lt(4.33)

        # 3
        s1110 = difference(img, 'B11', 'B10').lt(2550)
        b3_3 = img.select('B3').lt(5250)
        r210 = ratio(img, 'B2','B10').lt(14.689)
        s37 = difference(img, 'B3', 'B7').lt(270)

        # 4
        r15 = ratio(img, 'B1', 'B5').lt(1.184)
        s67 = difference(img, 'B6', 'B7').lt(-160)
        b1 = img.select('B1').lt(3000)
        r29 =  ratio(img, 'B2', 'B9').lt(0.788)
        s911 = difference(img, 'B9', 'B11').lt(210)
        s911_2 = difference(img, 'B9', 'B11').lt(-970)

        snow = {'snow':[['1',0], ['22',0], ['34',0]]}
        cloud = {'cloud-1':[['1',0], ['22',1],['33',1],['44',1]],
                 'cloud-2':[['1',0], ['22',1],['33',0],['45',0]]}
        cirrus = {'cirrus-1':[['1',0], ['22',1],['33',1],['44',0]],
                  'cirrus-2':[['1',1], ['21',0],['32',1],['43',0]]}
        shadow = {'shadow-1':[['1',1], ['21',1],['31',1],['41',0]],
                  'shadow-2':[['1',1], ['21',1],['31',0],['42',0]],
                  'shadow-3':[['1',0], ['22',0],['34',1],['46',0]]}
        water = {'water':[['1',1], ['21',1],['31',0],['42',1]]}

        all = {'cloud':cloud,
               'snow': snow,
               'shadow':shadow,
               'water':water,
               'cirrus':cirrus}

        final = {}

        for option in options:
            final.update(all[option])

        dtf = binary(
            {'1':b3,
             '21':b8a, '22':r511,
             '31':s37, '32':r210, '33':s1110, '34':b3_3,
             '41': s911_2, '42':s911, '43':r29, '44':s67, '45':b1, '46':r15
             }, final, name)

        results = dtf

        if updateMask and addBands:
            return img.addBands(results).updateMask(results.select(name))
        elif addBands:
            return img.addBands(results)
        elif updateMask:
            return img.updateMask(results.select(name))

    return compute_dt(img)

def binary(conditions, classes, mask_name='dt_mask'):

    cond = ee.Dictionary(conditions)
    paths = ee.Dictionary(classes)

    def C(condition, bool):
        # b = ee.Number(bool)
        return ee.Image(ee.Algorithms.If(bool, ee.Image(condition),
                                         ee.Image(condition).Not()))

    # function to iterate over the path (classes)
    def overpath(key, path):
        v = ee.List(path) # the path is a list of lists
        # define an intial image = 1 with one band with the name of the class
        ini = ee.Image.constant(1).select([0], [key])

        # iterate over the path (first arg is a pair: [cond, bool])
        def toiterate(pair, init):
            init = ee.Image(init)
            pair = ee.List(pair)
            boolean = pair.get(1)
            condition_key = pair.get(0)  # could need var casting
            condition = cond.get(condition_key)
            final_condition = C(condition, boolean)
            return ee.Image(init.And(ee.Image(final_condition)))

        result = ee.Image(v.iterate(toiterate, ini))
        return result

    new_classes = ee.Dictionary(paths.map(overpath))

    # UNIFY CLASSES. example: {'snow-1':x, 'snow-2':y} into {'snow': x.and(y)}
    new_classes_list = new_classes.keys()

    def mapclasses(el):
        return ee.String(el).split('-').get(0)

    repeated = new_classes_list.map(mapclasses)

    unique = remove_duplicates(repeated)

    # CREATE INITIAL DICT
    def createinitial(baseclass, ini):
        ini = ee.Dictionary(ini)
        i = ee.Image.constant(0).select([0], [baseclass])
        return ini.set(baseclass, i)

    ini = ee.Dictionary(unique.iterate(createinitial, ee.Dictionary({})))

    def unify(key, init):
        init = ee.Dictionary(init)
        baseclass = ee.String(key).split('-').get(0)
        mask_before = ee.Image(init.get(baseclass))
        mask = new_classes.get(key)
        new_mask = mask_before.Or(mask)
        return init.set(baseclass, new_mask)

    new_classes_unique = ee.Dictionary(new_classes_list.iterate(unify, ini))

    masks = new_classes_unique.values() # list of masks

    # Return an Image with one band per option

    def tomaskimg(mask, ini):
        ini = ee.Image(ini)
        return ini.addBands(mask)

    mask_img = ee.Image(masks.slice(1).iterate(tomaskimg,
                                               ee.Image(masks.get(0))))
    # print(mask_img)

    init = ee.Image.constant(0).rename(mask_name)

    def iterate_results(mask, ini):
        ini = ee.Image(ini)
        return ini.Or(mask)

    result = masks.iterate(iterate_results, init)

    not_mask = ee.Image(result).Not()

    return mask_img.addBands(not_mask)

def remove_duplicates(eelist):
    """ Remove duplicated values from a EE list object """
    newlist = ee.List([])
    def wrap(element, init):
        init = ee.List(init)
        contained = init.contains(element)
        return ee.Algorithms.If(contained, init, init.add(element))
    return ee.List(eelist.iterate(wrap, newlist))