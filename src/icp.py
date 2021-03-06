import numpy as np
from scipy.optimize import least_squares

import src.feature_detection as features
import src.hc as hc
import time

def match_scans(s1, s2, ic, sp, iters=5):
    """Match scans using the ICP method

    Parameters
    ----------
    s1: ndarray
        scan 1 in format [x,y, angle],
    s2: ndarray
        scan 2 in format [x,y, angle],
    ic: ndarray
        initial constarints between the scans (might be based on odometry) [tx, ty, theta]
    sp: dict
        slam parameters
    iters: int
        number of ICP iterations, default:5

    Returns
    -------
    x: ndarray
        transformation [tx, ty, theta] matching the scans
    """
    x = ic

    # Extract line features from the scans
    lp1 = features.get_line_features(s1[:, :2], s1[:, 2], sp["extr_angle_range"], sp["extr_split_th"],
                                     sp["extr_min_len"], sp["extr_min_points"], sp["mrg_max_dist"],
                                     sp["mrg_a_tol"], sp["mrg_b_tol"], sp["mrg_fit_tol"], corners=[])
    for i in range(iters):
        # transform second scan using the result from the last iteration
        # H = transformation_matrix(x[2], x[:2])
        H = hc.translation(x[:2]).dot(hc.rotation(x[2]))
        s2t = hc.hc_ec(H.dot(hc.ec_hc(s2[:, :2])))

        lp2 = features.get_line_features(s2t[:, :2], s2[:, 2], sp["extr_angle_range"], sp["extr_split_th"],
                                         sp["extr_min_len"], sp["extr_min_points"], sp["mrg_max_dist"],
                                         sp["mrg_a_tol"], sp["mrg_b_tol"], sp["mrg_fit_tol"], corners=[])

        # get corresponding line segments and points, which are close to each other
        corr_points = corresponding_line_points(lp1, lp2, H,
                                                an_th=sp['an_th'], d_th=sp['d_th'], corr_points_th=sp['corr_points_th'])

        optres = least_squares(icp_err_fun, x, args=(corr_points,))
        x = optres.x
        # print(x)

    return x

def transformation_matrix(angle, t):
    """Get transformation matrix in HC taking into account LiDAR displacement

    Parameters
    ----------
    angle: float
        angle in radians
    t: ndarray
        translation [tx, ty]

    Returns
    -------
    H: ndarray
        transformation matric in HC
    """
    # lidar movement due to rotation
    Tr = hc.translation(np.r_[0.14, -0.01])
    T = hc.translation(t)

    R = hc.rotation(angle)
    H = T.dot(np.linalg.inv(Tr).dot(R.dot(Tr)))

    return H


def corresponding_line_points(lp1, lp2, H, an_th=0.1, d_th=0.05, corr_points_th=0.1):
    """

    Parameters
    ----------
    lp1
    lp2
    H
    an_th
    d_th
    corr_points_th

    Returns
    -------

    """
    lines1, points1 = lp1
    lines2, points2 = lp2

    # get bearing and range to the detected walls
    br1 = bearing_range_lines(np.r_[0, 0, 0], lines1)
    br2 = bearing_range_lines(np.r_[0, 0, 0], lines2)

    # find corrspeonding walls
    corr_points = []

    # look for distances
    for b, p in zip(br1, points1):
        # distance between from points
        ldif = np.c_[angle_diff(br2[:,0], b[0]), np.abs(br2[:,1] - b[1])]

        # corresponding lines mask
        corr = ((ldif[:, 0] < an_th) & (ldif[:, 1] < d_th))

        # put corresponding points into tuples
        if corr.any():
            p2 = np.vstack([points2[i] for i in np.argwhere(corr).T[0]])  # combine points from lines
            cd = np.apply_along_axis(cp_dist, 1, p2, p)  # get closest distances
            p2 = p2[cd < corr_points_th]  # choose nearest points

            if p2.size:
                p2 = hc.hc_ec(np.linalg.inv(H).dot(hc.ec_hc(p2)))  # transform them back
                corr_points.append((p, p2))

    return corr_points


def bearing_range_lines(x, lines):
    """Get bearing and range to lines

    Parameters
    ----------
    x: ndarray
        LiDAR location [x,y]
    lines: ndarray
        line parameters as returned by `corresponding_line_points`

    Returns
    -------
    br: ndarray
        bearing and range to lines
    """
    r = features.dist_from_lines(x[:2], lines)
    cp = features.closest_point_on_line(lines, x[:2]) - x[:2]
    bearing = np.arctan2(cp[:, 1], cp[:, 0])

    return np.c_[bearing, r]


def cp_dist(a, p):
    """ Closest point distance

    Parameters
    ----------
    a: ndarray
        point cloud
    p: ndarray
        point

    Returns
    -------
    cp: float
        closest point distance
    """
    return np.min(np.linalg.norm(a - p, axis=1))


def icp_err_fun(x, corr_points):
    """ Function error for LS ICP fitting

    Parameters
    ----------
    x: ndarray
        searched vector [tx, ty, theta]
    corr_points: list
        list of tuples containing arrays with corresponding points

    Returns
    -------

    """
    H = hc.translation(x[:2]).dot(hc.rotation(x[2]))

    cper = []
    for p1, p2 in corr_points:
        cper.append(np.mean(np.apply_along_axis(cp_dist, 1, hc.hc_ec(H.dot(hc.ec_hc(p2))), p1)))
    cper = np.hstack(cper)

    return cper


def angle_diff(a1, a2):
    """Get the difference between two angle values taking into account the periodicity

    Parameters
    ----------
    a1: ndarray
        angles 1
    a2: ndarray
        angles 2

    Returns
    -------
    ad: ndarray
        angle difference
    """
    return np.pi - abs(abs(a1 - a2) - np.pi)


def transform_lp(lp, x):
    pass