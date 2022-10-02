import numpy as np
import cv2 as cv

import multiprocessing
import concurrent.futures
import itertools
import time

from tiles import Tiles

tile_size = 256


def get_tile(tiles:Tiles, x0, y0, x, y, z, output):
    """Function called to get tile data for a given thread, writes to output"""
    data = tiles.get_tile(x,y,z)
    assert(data.shape == (tile_size, tile_size,3) and data.dtype == np.uint8)

    start_x = (x-x0) * tile_size
    start_y = (y-y0) * tile_size

    

    output[ start_y:start_y + tile_size, start_x: start_x + tile_size,:] = data


def render(tiles, x0, y0, x1,y1, z):
    """Multithreaded rendering of a set of tiles from x0,y0 to x1,y1 inclusive at zoom level z"""
    assert(x0 < x1)
    assert(y0 < y1)

    thread_count = multiprocessing.cpu_count()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers= thread_count-1)

    size_x = (x1 - x0 +1) * tile_size 
    size_y = (y1 - y0 +1) * tile_size

    result = np.zeros((size_y,size_x,3),dtype=np.uint8)

    args = [ (get_tile, tiles, x0, y0,x,y,z,result) for (x,y) in itertools.product(range(x0,x1+1), range(y0,y1+1)) ]

    futures =[executor.submit(*arg) for arg in args]
    
    cv.namedWindow("Render", cv.WINDOW_KEEPRATIO)
    #cv.imshow("Render",result)

    for _ in concurrent.futures.as_completed(futures):
        cv.imshow("Render", result)
        cv.waitKey(10)

    concurrent.futures.wait(futures)

    print("Done")
    cv.imshow("Render", result)
    cv.waitKey(0)




terrain = Tiles("https://stamen-tiles-b.a.ssl.fastly.net/terrain-background/{z}/{x}/{y}.png",18)

render(terrain, 6,20,20,30,6)