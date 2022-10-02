import numpy as np
import cv2 as cv

import urllib.request
import pathlib 
import hashlib


cache_folder = "tilecache"
pathlib.Path(cache_folder).mkdir(exist_ok=True) 


def url_to_image(url:str):
    """Synchronously load a url in an opencv image"""
    h = hashlib.sha1(url.encode(encoding='UTF-8'))
    cache_file = pathlib.Path(cache_folder).joinpath(f"{h.hexdigest()}.png")
    if cache_file.exists():
        image = np.asarray(bytearray(open(cache_file, "rb").read()))
    else:
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv.imdecode(image, cv.IMREAD_COLOR)

    if not cache_file.exists():
        cv.imwrite(str(cache_file), image)
    return image

class Tiles:
    def __init__(self, url_template,max_zoom):
        self.url_template = url_template
        self.max_zoom = max_zoom

    def check_coords(self,x,y,z):
        if z < 0 or z > self.max_zoom:
            return False
        if x < 0 or z > 2**z:
            return False
        if y < 0 or y > 2**z:
            return False
        return True

    def get_url(self, x, y, z):
        assert(self.check_coords(x,y,z))
        return self.url_template.replace('{x}', f'{x}').replace('{y}',f'{y}').replace('{z}',f'{z}')
    
    def get_tile(self,x,y,z):
        return url_to_image(self.get_url(x,y,z))
