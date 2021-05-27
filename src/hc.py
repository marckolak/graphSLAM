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
