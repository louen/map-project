import numpy as np

def project(latitude:float, longitude:float):
    """ return the Mercator coordinates in 0,1 from latitude and longiture"""
    siny = np.sin(( latitude * np.pi) / 180.0)

    # Truncating to 0.9999 effectively limits latitude to 89.189. This is
    # about a third of a tile past the edge of the world tile.
    siny = np.clip(siny,-0.9999, +0.9999) 

    x = (0.5 + longitude / 360.0)
    y = (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi))

    return (x,y)