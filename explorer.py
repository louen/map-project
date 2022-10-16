import cv2 as cv
import numpy as np
import random

from tiles import Tiles, url_to_image 

z = 0  # zoom level
x = 0
y = 0

max_zoom = 15  # stamen is 18


worldmap = url_to_image(
    "https://stamen-tiles-a.a.ssl.fastly.net/terrain-background/0/0/0.png")

stencil = url_to_image(
    
)


while True:
    normal_url = f"https://s3.amazonaws.com/elevation-tiles-prod/normal/{z}/{x}/{y}.png"
    normal_img = url_to_image(normal_url)
    if (normal_img.shape[0:2] != (256, 256)):
        print(f"WARNING: normal map mismatch {normal_img.shape[0:2]}")
        normal_img = cv.resize(normal_img, (256, 256))

    elevation_url = f"https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
    elevation_img = url_to_image(elevation_url)

    s = random.choice(['a', 'b', 'c', 'd'])
    terrain_url = f"https://stamen-tiles-{s}.a.ssl.fastly.net/terrain-background/{z}/{x}/{y}.png"
    terrain_img = url_to_image(terrain_url)

    size = 2**z

    # Create a mavigation map with a red box showing where the map is
    tile_width = max(256 // size, 1)
    corner_0 = (x * tile_width, y * tile_width)
    corner_1 = ((x+1) * tile_width, (y+1) * tile_width)

    navmap = worldmap.copy()
    cv.rectangle(navmap, corner_0, corner_1, (0, 0, 255), 1)

    # 4 picture display
    display = np.concatenate((np.concatenate((elevation_img, navmap), axis=1), np.concatenate(
        (normal_img, terrain_img), axis=1)), axis=0)

    cv.imshow("MainWindow", display)
    key = cv.waitKeyEx(0)

    if key == ord('a'):
        x = (x-1) % size
    elif key == ord('d'):
        x = (x+1) % size
    elif key == ord('s'):
        y = (y+1) % size
    elif key == ord('w'):
        y = (y - 1) % size

    elif key == ord('z') and z < max_zoom:
        z += 1
        x *= 2
        y *= 2

    elif key == ord('x') and z > 0:
        z -= 1
        x //= 2
        y //= 2

    elif key == ord('d'):
        d = (d+1) % 3

    elif key == 27:  # esc
        break

    print(f"x = {x}, y = {y}, z = {z} ({key})")
