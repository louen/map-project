import concurrent.futures
import itertools
import multiprocessing
import random
import time
from random import choice

import cv2 as cv
import numpy as np

from geometry import project
from tiles import Tiles


def get_tile(tiles: Tiles, x0, y0, x, y, z, output):
    """Function called to get tile data for a given thread, writes to output"""
    tile_size = Tiles.TILE_SIZE
    data = tiles.get_tile(x, y, z)

    if data is not None:
        assert(data.shape == (tile_size, tile_size, 3) and data.dtype == np.uint8)

        start_x = (x-x0) * tile_size
        start_y = (y-y0) * tile_size

        output[start_y:start_y + tile_size, start_x: start_x + tile_size, :] = data
        return True
    else:
        return False



def render_tiles(tiles, x0, y0, x1, y1, z, interactive=True):
    """Multithreaded rendering of a set of tiles from x0,y0 to x1,y1 inclusive at zoom level z"""
    assert(x0 < x1)
    assert(y0 < y1)

    thread_count =  multiprocessing.cpu_count() - (1 if interactive else 0)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=thread_count)

    size_x = (x1 - x0 + 1) * Tiles.TILE_SIZE 
    size_y = (y1 - y0 + 1) * Tiles.TILE_SIZE 

    # Final picture big enough for all the tiles
    result = np.zeros((size_y, size_x, 3), dtype=np.uint8)

    # Setup the args for get_tile to be run in parallel
    args = [(get_tile, tiles, x0, y0, x, y, z, result)
            for (x, y) in itertools.product(range(x0, x1+1), range(y0, y1+1))]

    if interactive:
        cv.namedWindow("Render", cv.WINDOW_KEEPRATIO)
        cv.waitKey(0)

    start_time = time.time()
    futures = [executor.submit(*arg) for arg in args]
    print(
        f"Started downloading {(x1-x0) * (y1-y0)} tiles on {thread_count} threads ({result.shape[0]}x{result.shape[1]} pixels)")

    if interactive:
        for _ in concurrent.futures.as_completed(futures):
            cv.imshow("Render", result)
            cv.waitKey(1)
    else:
        concurrent.futures.wait(futures)


    success = sum([func.result() for func in futures])

    end_time = time.time()
    print(f"Download complete in {end_time-start_time:.3}s {success}/{len(futures)}")
    if interactive:
        cv.imshow("Render", result)
        cv.waitKey(0)
    return (result, success == len(futures))


def aabb_extents(aabb):
    """Compute the extends of an aabb"""
    return aabb[1] - aabb[0]

def aabb_ratio(aabb):
    """Compute the aspect ratio of an aabb"""
    extents = aabb_extents(aabb)
    return extents[0] / extents[1]


def get_render_params( latlongs, max_zoom:int, width:int, margin:int, aspect_ratio:float = 1.0 ):
    """
        Get the rendering params so that the image fits the following criteria
        - contains all the GPS points given in latlongs
        - is at `width` pixels wide 
        - has a margin of `margin` pixels around the points of interest
        - has the given aspect ratio
    """
    # project all in mercator
    projs = np.array([ project(*latlong) for latlong in latlongs])
    # AABB of the target points
    aabb = np.array([projs.min(axis=0), projs.max(axis=0)])
    

    # TODO: handle the case where the aabb actually strides the 180° meridian

    # Find the apropriate zoom level 
    max_extents = max (aabb_extents(aabb).max(), 0.0001)
    z = np.ceil(np.log2(( width - 2 * margin) / (max_extents * Tiles.TILE_SIZE)))
    z = int(min(z, max_zoom))


    # Now we determined the zoom level have three different yardsticks.
    # - "natural" map units where x, y range from 0 to 1
    # - tiles with x,y range from 0 to 2^z
    # - pixels with x,y range from 0 to 2^z * (tile width)

    tiles_per_unit = 2**z
    pixel_per_unit = tiles_per_unit * Tiles.TILE_SIZE

    # Extend the area of interest by the margin 
    margin_ext = np.ones(2) * (margin / pixel_per_unit)
    aabb[0] -= margin_ext
    aabb[1] += margin_ext

    # aspect ratio = width / height
    # expand the aabb to fit the target aspect ratio
    extents = aabb_extents(aabb) 
    ratio = aabb_ratio(aabb) 

    if aspect_ratio > ratio:
        # widen
        new_width = extents[1] *  aspect_ratio
        width_extension = new_width - extents[0]
        aabb[0][0] -=  0.5* width_extension
        aabb[1][0] += 0.5 * width_extension
    elif aspect_ratio < ratio:
        # heighten
        new_height = extents[0] / aspect_ratio
        height_extension = new_height - extents[1]
        aabb[0][1] -= 0.5 * height_extension
        aabb[1][1] += 0.5 * height_extension

    # AABB in pixels of the whole map
    aabb_pixels = aabb * pixel_per_unit
    aabb_tiles = aabb * tiles_per_unit

    # Compute the tiles covering the area
    topleft_tile = np.floor(aabb_tiles[0]).astype(int)
    bottomright_tile = np.ceil(aabb_tiles[1]).astype(int)

    topleft_pixel = np.floor(aabb_pixels[0] - topleft_tile *  Tiles.TILE_SIZE).astype(int)
    bottomright_pixel = np.floor(aabb_pixels[1] - topleft_tile * Tiles.TILE_SIZE).astype(int)


    return topleft_tile, bottomright_tile, z, topleft_pixel, bottomright_pixel

# Heightmap computation. TODO: move this to a file and test
#img, _ = render_tiles(toner, tl[0], tl[1], br[0], br[1], z, True)

#img, _ = render_tiles(height, tl[0], tl[1], br[0], br[1], z, True)

#height = (img[:,:,2] * 256 + img[:,:,1] + img[:,:,0] / 256) - 32768

#_, thres = cv.threshold(height, 0, 1.0, type=cv.THRESH_BINARY)

# height_pos = thres * height
# height_pos = cv.normalize(height_pos, height_pos, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)
# height_neg = -(1.0 - thres) * height
# height_neg = cv.normalize(height_neg, height_neg, alpha=0, beta=255, norm_type=cv.NORM_MINMAX, dtype=cv.CV_8U)

# height_pos = cv.applyColorMap(height_pos, cv.COLORMAP_HOT)
# height_neg = cv.applyColorMap(height_neg, cv.COLORMAP_OCEAN)  #* np.repeat(np.expand_dims(1.0 - thres,2), 3, axis = 2)

# cv.imshow("Render", height_neg + height_pos)



#img = cv.rectangle(img, pix, pix + np.array([5000,5000]), (0,0,255), 3)
#cv.imshow("Render", img)
# cv.waitKey(0)
