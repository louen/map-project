import numpy as np
import cv2 as cv

import multiprocessing
import concurrent.futures
import itertools
import time

from geometry import project
from tiles import Tiles



def get_tile(tiles: Tiles, x0, y0, x, y, z, output):
    tile_size = Tiles.TILE_SIZE
    """Function called to get tile data for a given thread, writes to output"""
    data = tiles.get_tile(x, y, z)
    assert(data.shape == (tile_size, tile_size, 3) and data.dtype == np.uint8)

    start_x = (x-x0) * tile_size
    start_y = (y-y0) * tile_size

    output[start_y:start_y + tile_size, start_x: start_x + tile_size, :] = data


def render_tiles(tiles, x0, y0, x1, y1, z, interactive=True):
    """Multithreaded rendering of a set of tiles from x0,y0 to x1,y1 inclusive at zoom level z"""
    assert(x0 < x1)
    assert(y0 < y1)

    thread_count =  multiprocessing.cpu_count() - (1 if interactive else 0)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=thread_count)

    size_x = (x1 - x0 + 1) * Tiles.TILE_SIZE 
    size_y = (y1 - y0 + 1) * Tiles.TILE_SIZE 

    result = np.zeros((size_y, size_x, 3), dtype=np.uint8)

    args = [(get_tile, tiles, x0, y0, x, y, z, result)
            for (x, y) in itertools.product(range(x0, x1+1), range(y0, y1+1))]

    if interactive:
        cv.namedWindow("Render", cv.WINDOW_KEEPRATIO)

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

    end_time = time.time()
    print(f"Download complete in {end_time-start_time:.3}s")
    if interactive:
        cv.imshow("Render", result)
        cv.waitKey(0)
    return result


terrain = Tiles(
    "https://stamen-tiles-b.a.ssl.fastly.net/terrain-background/{z}/{x}/{y}.png", 18)


LA_coords = (34.042379, -118.256078)
LA_proj = np.array(project(LA_coords[0],LA_coords[1]))

zoom = 13

LA_proj = (np.floor(2**zoom * LA_proj)).astype(int)

render_tiles(terrain, LA_proj[0]-10, LA_proj[1]-10, LA_proj[0]+10, LA_proj[1]+10, zoom, True)
