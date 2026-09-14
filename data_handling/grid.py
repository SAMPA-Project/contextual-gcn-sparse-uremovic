
import numpy as np
import json
#from scipy import stats

from pyproj import Proj, transform

def latlon2epsg32633(xi, yi):
    inproj = Proj('wgs84')
    outproj = Proj('epsg:32633')

    #yi, xi = 14.5, 46
    xo, yo = transform(inproj, outproj, xi, yi)
    return (xo, yo)

RESOLUTION = 60
X0, X1, Y0, Y1 = 344690, 655338, 5010816, 5213848 # y raste Dwn->Up, x raste L->R

def __normalize_grid(grid):
    ax = (0, 1)
    mean = np.mean(grid, axis=ax)
    std = np.std(grid, axis=ax)
    return (grid - mean) / std

def __grid_data(metadata):
    dtm = 'data/meteo/contextual/dtm.npy'
    sentinel = 'data/meteo/contextual/sentinel2_data.npy'

    with open(metadata, 'r') as fp:
        metadata = json.load(fp)
    
    myKeys = list(metadata.keys())
    myKeys.sort()
    metadata_sorted = {i: metadata[i] for i in myKeys}
    metadata = metadata_sorted

    dtm = np.load(dtm)
    dtm[dtm < -100] = 0
    dtm = __normalize_grid(dtm)
    sentinel = np.load(sentinel)

    scl = sentinel[:,:,11]
    ndvi = __normalize_grid(sentinel[:,:,12])

    return (metadata, dtm, ndvi, scl)

def full_grid(resolution, metadata):
    _, dtm, ndvi, scl = __grid_data(metadata)
    f=resolution//RESOLUTION
    rasters = np.stack([dtm, ndvi, scl], axis=-1)[::f, ::f, :]
    return (X0, X1, Y0, Y1), rasters

def load_NxN_data(N, resolution_m, metadata='data/meteo/contextual/stations.json'):
    '''
        order (3x3 example):
        [0 1 2
        3 4 5 
        6 7 8]
    '''
    metadata, dtm, ndvi, scl = __grid_data(metadata)
    #print([item[1]['idx'] for item in metadata.items()])

    sizey, sizex = dtm.shape
    # import matplotlib.pyplot as plt
    # extent = [X0, X1, Y0, Y1]
    # fig, ax = plt.subplots(1, 1)
    #ax.imshow(dtm, extent=extent)
    res = resolution_m // RESOLUTION
    pixel_data = np.zeros(shape=(len(metadata.keys()), N*N, 3))
    for i, station in enumerate(metadata.keys()):
        try:
            x, y = metadata[station]['x'], metadata[station]['y']
        except KeyError:
            x, y = latlon2epsg32633(metadata[station]['lat'], metadata[station]['lon'])
        #ax.scatter(x, y, s=100, c='red', marker='o', zorder=9)
        #ax.text(x, y, metadata[station]['id'], size='large', zorder=10, horizontalalignment='center', verticalalignment='center')
        top2down = int((Y1 - y) / (Y1 - Y0) * sizey)
        right2left = int((X1 - x) / (X1 - X0) * sizex)
        idx = 0
        for j in range(-N//2+1, N//2+1, 1):
            for k in range(-N//2+1, N//2+1, 1):
                #ax.text(X0 + (sizex-right2left+k*res) * RESOLUTION, Y1 - (top2down+j*res) * RESOLUTION, 'j={},k={}'.format(j, k), horizontalalignment='center', verticalalignment='center', size='large', zorder=10)
                #ax.scatter(X0 + (sizex-right2left+k*res) * RESOLUTION, Y1 - (top2down+j*res) * RESOLUTION, s=50, c='blue', marker='o', zorder=9)
                pixel_data[i][idx][0] = dtm[top2down+j*res][sizex-right2left+k*res]
                pixel_data[i][idx][1] = ndvi[top2down+j*res][sizex-right2left+k*res]
                pixel_data[i][idx][2] = scl[top2down+j*res][sizex-right2left+k*res]
                idx += 1
    
    #plt.show()
    return pixel_data # (stations, N*N, 3)

if __name__ == '__main__':
    load_NxN_data(5, 480)
    load_NxN_data(3, 960)