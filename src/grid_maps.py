"""
The grid_maps module contains functions which can be used for occupancy grid mapping.

Copyright (C) 2021 Marcin Kolakowski
"""

import numpy as np


def log_odds(p):
    """ Compute log odds

    Parameters
    ----------
    p: float, ndarray
        probability in (0,1) range

    Returns
    -------
    lp: float, ndarray
        log odds
    """
    return np.log(p / (1 - p))


def prob(logodds):
    """ Compute probability

    Parameters
    ----------

    logodds: float, ndarray
        log odds

    Returns
    -------
    p: float, ndarray
        probability in (0,1) range
    """
    return 1 - 1 / (1 + np.exp(logodds))


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


def points2gridmap(size, res, pose, scan, p_free=0.4, p_nd=0.5, p_occ=0.99):
    """Convert points to gridmap

    The gridmap is square-shaped

    Parameters
    ----------
    size: float
        gridmap dimensions
    res: float
        gridmap cell size
    pose: ndarray
        robot pose
    scan: ndarray
        registered scan in x,y coordinates
    p_free: float
        probability of free cell
    p_nd: float
        probability of no data
    p_occ: float
        probability ofoccupied cell

    Returns
    -------
    gridmap: ndarray
        gridmap
    """
    l_free = log_odds(p_free)
    l_nd = log_odds(p_nd)
    l_occ = log_odds(p_occ)

    if type(size) is tuple:
        gridmap = np.zeros([int(np.ceil(size[0] / res)), int(np.ceil(size[1] / res))]) + l_nd
    else:
        gridmap = np.zeros([int(np.ceil(size / res)), int(np.ceil(size / res))]) + l_nd


    pix = world2map(pose, gridmap, res)
    ix = world2map(scan, gridmap, res)

    for pt in ix:
        try:
            bix = np.array(list(bresenham(pix[0], pix[1], pt[0], pt[1])))
            gridmap[bix[:-1, 0], bix[:-1, 1]] += l_free - l_nd
            gridmap[bix[-1][0], bix[-1][1]] += l_occ - l_nd
        except:
            pass

    return prob(gridmap)


def world2map(pose, gridmap, map_res):
    """Convert x-y locations to map coordinate system

    Parameters
    ----------
    pose: ndarray
        robot pose in HC / points to be converted
    gridmap: ndarray
        gridmap
    map_res: float
        gridmap cell size

    Returns
    -------
    ci: ndarray
        coordinates of cells corresponding to points
    """
    origin = np.array(gridmap.shape) / 2
    new_pose = np.round(pose[:2] / map_res).T + origin

    return new_pose.astype(int)


def map2world(pose_m, gridmap, map_res):
    """ Convert cell coordinates to Euclidean x-y

    Parameters
    ----------
    pose_m: ndarray
        points in map coordinates
    gridmap: ndarray
        gridmap
    map_res: float
        gridmap cell size

    Returns
    -------
    xy: ndarray
        points in Euclidean x-y coordinates
    """
    origin = np.array(gridmap.shape) / 2
    pose_w = (pose_m - origin) * map_res

    return pose_w


def merge_maps(g1, g2, p_nd=0.5):
    """ Merge maps

    Parameters
    ----------
    g1: ndarray        map 1
    g2: ndarray
        map 2
    p_nd: float
        probability of no data

    Returns
    -------
    g: ndarray
        merged map
    """
    gridmap = prob(log_odds(g1) + log_odds(g2) - log_odds(p_nd))

    return gridmap


def bresenham(x0, y0, x1, y1):
    """ Get cells, which are crossed by a line

    Implementation of Bresenham's algorithm

    Parameters
    ----------
    x0: int
    y0: int
    x1: int
    y1: int

    Returns
    -------
    D: iterator
        coordinates of cells crossed by the line segment
    """
    dx = x1 - x0
    dy = y1 - y0

    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    dx = abs(dx)
    dy = abs(dy)

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0

    D = 2 * dy - dx
    y = 0

    for x in range(dx + 1):
        yield x0 + x * xx + y * yx, y0 + x * xy + y * yy
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy


def init_gridmap(size, res, p_nd=0.5):
    """ Initialize an empty gridmap

    Parameters
    ----------
    size: float
        gridmap dimensions
    res: float
        gridmap cell size
    p_nd: float
        probability of no data

    Returns
    -------
    gridmap: ndarray
        empty gridmap
        \
    """
    l_nd = log_odds(p_nd)

    gridmap = np.zeros([int(np.ceil(size / res)), int(np.ceil(size / res))]) + l_nd
    return prob(gridmap)


def map_corr(m1, m2):
    """ Calculate correlation between to maps

    Parameters
    ----------
    m1: ndarray
        map 1
    m2: ndarray
        map 2

    Returns
    -------
    corr: tuple
        percentages of common points for both maps
    """
    same_points = (np.abs(m1 - m2) < 0.2) & (m1 != 0.5) & (m2 != 0.5)
    return np.r_[same_points.sum() / (m1 != 0.5).sum(), same_points.sum() / (m2 != 0.5).sum()]


def sample_gridmap(gridmap, pose, size, res, lidar_range, angle_res):
    """

    Parameters
    ----------
    gridmap: ndarray
        map
    pose: ndarray
        robot pose
    size: float
        gridmap dimensions
    res: float
        gridmap cell size
    lidar_range: float
        max LiDAR range
    angle_res: float
        LiDAR angle resolution

    Returns
    -------
    sample: ndarray
        locations of points, which would be detected by the LiDAR in the given robot pose
    """
    map_sample = np.zeros(gridmap.shape) + 0.5
    # map_sample = gridmap.copy()

    pose_m = world2map(pose, gridmap, res)

    angles = np.arange(pose[2], pose[2] + 2 * np.pi, angle_res)

    far_points = np.c_[np.cos(angles), np.sin(angles)] * lidar_range + pose[:2]
    # print(far_points.shape)
    far_points_m = world2map(far_points.T, gridmap, res)

    obst_points_m = []
    for p in far_points_m:
        bp = np.array(list(bresenham(pose_m[0], pose_m[1], p[0], p[1])))
        bp = bp[(bp < gridmap.shape[0]).all(axis=1)]
        # print(bp.max())
        mv = gridmap[bp[:, 0], bp[:, 1]]
        first_occ_ix = min(np.argwhere(mv == 1))[0] + 1
        # print(first_occ_ix)
        # map_sample[bp[:first_occ_ix,0], bp[:first_occ_ix,1]] = 2
        obst_points_m.append(bp[first_occ_ix])

    obst_points = obst_points_m
    map2world(bp[first_occ_ix], gridmap, res)
    return np.vstack(obst_points)


def is_inside_map(pose, gridmap, res):
    """ Check if the given pose is inside the mapped environment

    Parameters
    ----------
    pose: ndarray
        robot pose
    gridmap: ndarray
        map
    res: float
        gridmap cell size

    Returns
    -------
    inside: bool
        True if the point is insde map
    """
    pose_m = world2map(pose, gridmap, res)
    inside = (gridmap[pose_m[0]:, pose_m[1]] == 1).any() & (gridmap[:pose_m[0], pose_m[1]] == 1).any() & (
            gridmap[pose_m[0], pose_m[1]:] == 1).any() & (gridmap[pose_m[0], :pose_m[1]] == 1).any()

    return inside
