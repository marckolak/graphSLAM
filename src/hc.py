"""
This module implements homogenous coordinates transformations and operations.
"""
import numpy as np


def ec_hc(x):
    """Convert euclidean x-y coordinates to homogenous

    Parameters
    ----------
    x: ndarray
        x-y coordinates

    Returns
    -------
    x_hc: ndarray
        points in hc_coordinates
    """
    if x.ndim == 1:
        return np.r_[x, 1].T
    else:
        return np.hstack([x, np.ones((x.shape[0], 1))]).T


def hc_ec(x_hc):
    """Convert homogenous coordinates to euclidean x-y

    Parameters
    ----------
    x_hc: ndarray
        points in hc_coordinates

    Returns
    -------
    x: ndarray
        x-y coordinates
    """
    if x_hc.ndim == 1:
        return x_hc[:2] / x_hc[2]
    else:
        return (x_hc[:2] / x_hc[2]).T


def translation(t):
    """Get translation matrix

    Parameters
    ----------
    t: ndarray
        translation vector [tx,ty]

    Returns
    -------
    h: ndarray
        transformation matrix
    """
    h = np.identity(3)
    h[:2, 2] = t
    return h


def rotation(angle):
    """Get translation matrix

    Parameters
    ----------
    angle: float
        rotation angle in radians

    Returns
    -------
    h: ndarray
        transformation matrix
    """
    h = np.identity(3)
    h[:2, :2] = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return h


def rigid_body_transformation(angle, t):
    """Get translation matrix

    Parameters
    ----------
    angle: float
        rotation angle in radians
    t: ndarray
        translation vector [tx,ty]

    Returns
    -------
    h: ndarray
        transformation matrix
    """
    h = np.identity(3)
    h[:2, :2] = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    h[:2, 2] = t

    return h


def skew(x):
    """Compute skew matrix for vector x

    Parameters
    ----------
    x: ndarray
        point in homogenous coordinates

    Returns
    -------
    s: ndarray
        skew matrix
    """

    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def line_jp(x1, x2):
    """Get a line joining two points

    Parameters
    ----------
    x1: ndarray
        point in homogenous coordinates
    x2: ndarray
        point in homogenous coordinates

    Returns
    -------
    l: ndarray
        line in homogenous coordinates
    """

    s = skew(x1)
    return s.dot(x2)


def line_si(l):
    """Get slope and intercept of the line

    Parameters
    ----------
    l: ndarray
        line in homogenous coordinates

    Returns
    -------
    slope: float
        slope parameter
    intercept: float
        intercept parameter
    """

    slope = -l[0] / l[1]
    intercept = -l[2] / l[1]
    return slope, intercept


def normalize_line(l):
    """Normalize line
    Normalization is needed when computing distance from a line

    Parameters
    ----------
    l: ndarray
        line in homogenous coordinates

    Returns
    -------
    ln: ndarray
        normalized line in homogenous coordinates
    """
    return l / np.linalg.norm(l[:2])


def point_line_dist(ln, x):
    """Calculate distance of points from a line

    Parameters
    ----------
    ln: ndarray
        normalized line in homogenous coordinates
    x: ndarray
        points in hc

    Returns
    -------
    d: ndarray
        distances between the points and the line
    """

    return ln.dot(x)


def transform_line(l, H):
    """Apply transformation to a line

    Parameters
    ----------
    l: ndarray
        line
    H: ndarray
        transformation matrix

    Returns
    -------
    lt: ndarray
        transformed line
    """
    return np.linalg.inv(H.T).dot(l)


def t2v(T):
    x = T[0, 2]
    y = T[1, 2]
    theta = np.arctan2(T[1, 0], T[0, 0])
    v = np.array([x, y, theta])
    return v


def v2t(pose):
    c = np.cos(pose[2])
    s = np.sin(pose[2])
    T = np.array([[c, -s, pose[0]], [s, c, pose[1]], [0, 0, 1]])
    return T
