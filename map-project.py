import numpy as np
import cv2 as cv

import argparse
import random
import itertools

from tiles import Tiles
from renderer import get_render_params, render_tiles

#LA = (33.840752, -118.051364)
#SF = (37.921138, -122.578390)
#JT = (33.683071, -115.806942)

# Some reference :
# elevation tiles and normal maps :
# https://www.mapzen.com/blog/elevation/

# Terrain tiles without background :
# https://leaflet-extras.github.io/leaflet-providers/preview/#filter=Stamen.TerrainBackground -

# Tangram heightmapper
# https://www.mapzen.com/blog/tangram-heightmapper/


def main():
    parser = argparse.ArgumentParser(
        description='Produce maps from gps coordinates')
    parser.add_argument('gps', metavar='GPS', type=float,
                        nargs='+',  help='gps coordinates of interest')
    parser.add_argument('--output', type=str, help='output folder')
    parser.add_argument('--interactive', action=argparse.BooleanOptionalAction,
                        default=False, help='Display maps as they are downloaded')
    parser.add_argument('--width', type=int,
                        help='width of the final picture', default=2048)
    parser.add_argument('--height', type=int,
                        help='height of the final picture', default=2048)
    parser.add_argument(
        '--margin', type=int, help='margin in pixels around area of interest', default=200)

    args = parser.parse_args()

    if len(args.gps) % 2 != 0:
        print("Error: need two coordinates per point")
        exit(1)

    terrain = Tiles(
        "https://stamen-tiles-{r0}.a.ssl.fastly.net/terrain-background/{z}/{x}/{y}.png", 18, 
        [lambda: random.choice(['a', 'b', 'c', 'd'])])

    toner = Tiles(
        "https://stamen-tiles.a.ssl.fastly.net/toner/{z}/{x}/{y}.png", 15
    )

    height = Tiles(
        "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png", 15
    )

    max_zoom = min(terrain.max_zoom, height.max_zoom)
    gps = zip(args.gps[::2], args.gps[1::2])

    # Compute render parameters
    topleft_tile, bottomright_tile, z, topleft_pixel, bottomright_pixel = get_render_params(
        gps, max_zoom, args.width, args.margin, args.width / args.height)

    # Render image
    img, result = render_tiles(
        terrain, topleft_tile[0], topleft_tile[1], bottomright_tile[0], bottomright_tile[1], z, True)

    # Crop and resize to specified dimensions
    crop = img[topleft_pixel[1]:bottomright_pixel[1],
               topleft_pixel[0]: bottomright_pixel[0], :]
    crop = cv.resize(crop, (args.width, args.height), cv.INTER_AREA)
    cv.imshow("Render", crop)
    cv.waitKey(0)


if __name__ == "__main__":
    main()
