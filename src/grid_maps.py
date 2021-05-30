"""
The grid_maps module contains functions which can be used for occupancy grid mapping.

Copyright (C) 2021 Marcin Kolakowski
"""

import numpy as np

from src.geometry import sort_clockwise
from matplotlib import pyplot as plt

class GridMap:

    def __init__(self, size=(0, 0), resolution=0.05):
        self.size = size
        self.width = size[0]
        self.height = size[1]
        self.resolution = resolution

        self.xc = None
        self.yc = None
        self.map = None

        self.l_free = log_odds(0.4)
        self.l_nd = log_odds(0.5)
        self.l_occ = log_odds(0.99)

        self.pose = np.r_[0, 0, 0]

        self.scan = np.array([]).reshape(-1, 2)

    def update_size(self, new_size):
        self.size = new_size
        self.width = new_size[0]
        self.height = new_size[1]

    def from_scan(self, s, pose=np.r_[0, 0, 0]):
        """Create occupancy grid map from a scan

        Parameters
        ----------
        s: ndarray
            scan in xy coordinates
        pose: ndarray
            pose [x,y, heading]

        """
        self.scan = s[:, :2]
        x, y = s[:, 0] + pose[0], s[:, 1] + pose[1]
        self.pose = pose
        cs = self.resolution
        self.xc = np.arange((np.floor(x.min() / cs) - 3) * cs, (np.ceil(x.max() / cs) + 3) * cs, cs)
        self.yc = np.arange((np.floor(y.min() / cs) - 3) * cs, (np.ceil(y.max() / cs) + 3) * cs, cs)

        xx, yy = np.meshgrid(self.xc, self.yc)
        xy = np.array([xx - self.pose[0], yy - self.pose[1]])
        self.map = np.zeros(xx.shape) + log_odds(0.5)
        R = np.array([[0, -1], [1, 0]])

        for p in s[:, :2]:
            v = p / np.linalg.norm(p)
            vp = R.dot(v)
            c = np.abs(np.tensordot(xy, vp, axes=((0), (0))))
            r = np.tensordot(xy, v, axes=((0), (0)))

            free = (c < cs / 2) & (r > 0) & (r <= np.linalg.norm(p))
            occ = (c < cs / 2) & (np.abs(r - np.linalg.norm(p)) < cs / 2)
            self.map[free] += self.l_free - self.l_nd
            self.map[occ] += self.l_occ - self.l_nd

        self.map = prob(self.map)
        self.map[self.map < 0.4] = 0.001
        self.map[self.map > 0.6] = 1

    def merge(self, map2):
        """Merge a grid map with another one

        Parameters
        ----------
        map2:

        Returns
        -------

        """
        xcm = np.r_[self.xc, map2.xc]
        ycm = np.r_[self.yc, map2.yc]

        self.expand_map(xcm.min(), xcm.max(), ycm.min(), ycm.max())
        map2.expand_map(xcm.min(), xcm.max(), ycm.min(), ycm.max())

        self.map = prob(log_odds(self.map) + log_odds(map2.map) - self.l_nd)

    def expand_map(self, x_min, x_max, y_min, y_max):
        cs = self.resolution
        new_xc = np.arange((np.floor(x_min / cs) - 3) * cs, (np.ceil(x_max / cs) + 3) * cs, cs)
        new_yc = np.arange((np.floor(y_min / cs) - 3) * cs, (np.ceil(y_max / cs) + 3) * cs, cs)
        x_start = int((self.xc[0] - new_xc[0]) / cs)
        y_start = int((self.yc[0] - new_yc[0]) / cs)
        xx, yy = np.meshgrid(new_xc, new_yc)
        new_map = np.zeros(xx.shape) + 0.5

        new_map[y_start:y_start + self.map.shape[0], x_start:x_start + self.map.shape[1]] = self.map.copy()
        self.map = new_map
        self.xc = new_xc
        self.yc = new_yc

    def get_occupied_xy(self):
        xx, yy = np.meshgrid(self.xc, self.yc)
        xo = xx[self.map == 1]
        yo = yy[self.map == 1]

        xyo = sort_clockwise(np.c_[xo, yo], middle_point=self.pose[:2])
        a0 = np.arctan2(xyo[:, 1] - self.pose[0], xyo[:, 0] - self.pose[0])
        return xyo, a0

    def transform(self, angle, t):
        # self.scan = rotation_matrix(angle).dot(self.scan.T).T + t
        self.from_scan(self.scan)

    def normalize_maps(self, map0):
        xcm = np.r_[self.xc, map0.xc]
        ycm = np.r_[self.yc, map0.yc]

        self.expand_map(xcm.min(), xcm.max(), ycm.min(), ycm.max())
        map0.expand_map(xcm.min(), xcm.max(), ycm.min(), ycm.max())

    def sample_map(self, sampling_pose, resolution):
        cs = self.resolution
        xx, yy = np.meshgrid(self.xc, self.yc)
        xy = np.array([xx - sampling_pose[0], yy - sampling_pose[1]])
        scan_sample = []
        R = np.array([[0, -1], [1, 0]])
        for a in np.arange(-np.pi, np.pi, resolution):
            v= np.r_[np.cos(a), np.sin(a)]
            vp = R.dot(v)
            c = np.abs(np.tensordot(xy, vp, axes=((0), (0))))
            r = np.tensordot(xy, v, axes=((0), (0)))

            intersects_cell = np.abs(c) < cs / 2
            # print(intersects_cell)
            # plt.figure()
            # plt.imshow(np.abs(r)*(self.map>0.6), cmap='hot')
            # # plt.imshow(self.map[intersects_cell])
            # plt.show()
            ix = np.unravel_index(np.argmin(r*(intersects_cell & (self.map>0.6))), r.shape)
            scan_sample.append([xx[ix], yy[ix]])

        return np.array(scan_sample)
    # def sample_grid_map(self, angle_res):
    #
    #     xx, yy = np.meshgrid(self.xc, self.yc)
    #     xo = xx[self.map == 1]
    #     yo = yy[self.map == 1]
    #
    #     # move 0 0 of the map to the robot location
    #     fm_c = fmap - p
    #
    #     # get angle and distance for each of the map points
    #     d = np.linalg.norm(fm_c, axis=1)
    #     a = np.arctan2(fm_c[:, 1], fm_c[:, 0])
    #
    #     # aggregate by angle and get a set of closest points (simulates a LiDAR)
    #     df = pd.DataFrame(np.c_[fmap, a, d], columns=['x', 'y', 'angle', 'r'])
    #     df = df.sort_values(by='angle').reset_index(drop=True)
    #     angle_bins = np.arange(-np.pi, np.pi + 0.00001, resolution)
    #     df['angle_binned'] = np.digitize(df['angle'], angle_bins)
    #
    #     pred_scan = df.loc[df.groupby('angle_binned')['r'].idxmin(skipna=True)][['x', 'y']].values


def log_odds(p):
    return np.log(p / (1 - p))


def prob(logodds):
    return 1 - 1 / (1 + np.exp(logodds))


def occ_grid_bresenham(x0, y0, x1, y1, m):
    """Fills in occupancy grid map for a single range measurement using an implementation of Bresenham's algorithm

    Parameters
    ----------
    x0: float
        origin of the map (0,0 point)
    y0: float
        origin of the map (0,0 point)
    x1: float
        measured point location x coordinate
    y1: float
        measured point location y coordinate
    m: ndarray
        occupancy grid mpa

    Returns
    -------
    None - the method modifies the supplied map `m`

    """
    steep = np.abs(y1 - y0) > np.abs(x1 - x0)

    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = y1 - y0

    if dx == 0:
        gradient = 1
    else:
        gradient = dy / dx

    x0p, y0p = np.floor(x0).astype('int'), np.floor(y0).astype('int')
    x1p, y1p = np.floor(x1).astype('int'), np.floor(y1).astype('int')

    if steep:
        m[x0p, y0p] += 100
        m[x1p, y1p] += 1

    else:
        m[y0p, x0p] += 100
        m[y1p, x1p] += 100

    if steep:
        a = (y1 - y0) / (x1 - x0)
        b = y0 - a * x0

        x = np.arange(x0p + 1, x1p)
        y = a * x + b

        y = np.floor(y).astype(int)

        m[x, y] -= 0.5
    else:

        a = (y1 - y0) / (x1 - x0)
        b = y0 - a * x0

        x = np.arange(x0p + 1, x1p)
        y = a * x + b

        y = np.floor(y).astype(int)

        m[y, x] -= 0.5


def bin_scan(s, step, retbins=False):
    """Bin the scan points

    Parameters
    ----------
    s: ndarray
        scan in xy format
    step: float
        bin size in meters
    retbins: bool
        return x, y bins if True
    Returns
    -------
    xy_b: ndarray
        filled coordinates of the scan bins
    (x_bins_c, y_bins_c): tuple
        coordinates of bins centers in ndarrays
    """
    x, y = s[:, 0], s[:, 1]

    bins_x = np.arange(int(x.min() / step) * step - 2.5 * step, int(x.max() / step) * step + 2.5 * step + 0.001, step)
    bins_y = np.arange(int(y.min() / step) * step - 2.5 * step, int(y.max() / step) * step + 2.5 * step + 0.001, step)

    x_bins_c = np.arange(int(x.min() / step) * step - 2 * step, int(x.max() / step) * step + 2 * step + 0.001, step)
    y_bins_c = np.arange(int(y.min() / step) * step - 2 * step, int(y.max() / step) * step + 2 * step + 0.001, step)

    xy_b = np.c_[np.digitize(s[:, 0], bins_x), np.digitize(s[:, 1], bins_y)]

    xy_b, cnts = np.unique(xy_b, axis=0, return_counts=True)

    if retbins:
        return xy_b, (x_bins_c, y_bins_c), (bins_x, bins_y)
    return xy_b, (x_bins_c, y_bins_c)


def occupancy_grid_map(scan, cell_size):
    xy_b, _, (bins_x, bins_y) = bin_scan(scan, cell_size, retbins=True)
    x0, y0 = np.digitize(0, bins_x), np.digitize(0, bins_y)

    occ_grid = np.ones((bins_y.size, bins_x.size)) * 0.5

    for (x1, y1) in xy_b:
        occ_grid_bresenham(x0, y0, x1, y1, occ_grid)
    occ_grid[y0, x0] = 0
    occ_grid[occ_grid < 0] = 0
    occ_grid[occ_grid >= 1] = 1

    return occ_grid[::-1]
