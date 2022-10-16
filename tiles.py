import enum
import numpy as np
import cv2 as cv

import urllib.request
import pathlib
import hashlib


cache_folder = "tilecache"
pathlib.Path(cache_folder).mkdir(exist_ok=True)


def url_to_image(url: str):
    """Synchronously load a url in an opencv image"""
    h = hashlib.sha1(url.encode(encoding='UTF-8'))
    cache_file = pathlib.Path(cache_folder).joinpath(f"{h.hexdigest()}.png")
    if cache_file.exists():
        image = np.asarray(bytearray(open(cache_file, "rb").read()))
    else:
        try:
            resp = urllib.request.urlopen(url)
        except urllib.request.HTTPError as e:
            print(f"url {url} responded with status code {e.code}")
            return
        image = np.asarray(bytearray(resp.read()), dtype="uint8")

    image = cv.imdecode(image, cv.IMREAD_COLOR)

    # Debug some common errors
    if image.shape[:2] != (Tiles.TILE_SIZE, Tiles.TILE_SIZE):
        print(f"{url} image size is {image.shape[0]}x{image.shape[1]}")
    if image.max == 0:
        print(f"{url} image is completely black")

    if not cache_file.exists():
        cv.imwrite(str(cache_file), image)
    return image


class Tiles:
    """Tiles from a given data end point."""
    TILE_SIZE = 256 # size of a tile, in pixels

    def __init__(self, url_template, max_zoom, replacements=[]):
        """URL templates requires {x}, {y} and {z} plus optional {rN} replacement strings
        with lambda functions."""
        self.url_template = url_template
        self.max_zoom = max_zoom
        self.replacements = replacements

    def check_coords(self, x, y, z):
        if z < 0 or z > self.max_zoom:
            return False
        if x < 0 or z > 2**z:
            return False
        if y < 0 or y > 2**z:
            return False
        return True

    def get_url(self, x, y, z):
        assert(self.check_coords(x, y, z))
        url = self.url_template.replace('{x}', f'{x}').replace('{y}', f'{y}').replace('{z}', f'{z}')
        for i, f in enumerate(self.replacements):
            string = '{r'+str(i)+'}'
            url = url.replace(string, f())
        return url

    def get_tile(self, x, y, z):
        return url_to_image(self.get_url(x, y, z))
