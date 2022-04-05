"""
The geometry module contains functions for basic power related operations like log-linear
conversion and average value calculation

Copyright (C) 2021 Marcin Kolakowski
"""
import numpy as np

def sort_clockwise(xy, clockwise=True, middle_point=None):
    """ Sort point counterclockwise

    The function computes the middle point (median if not specified) of the set of points and sorts the points clockwise
    or counterclockwise.

    Parameters
    ----------
    xy: ndarray
        points' coordinates [x,y]
    clockwise: bool
        if True sort clockwise, False for counterclockwise
    middle_point: ndarray, None
        middle point [x,y], if None - use median point
    Returns
    -------
    xys: ndarray
        sorted points' coordinates [x,y]
    """
    # calc middle point if not specified
    if middle_point is None:
        middle_point = np.median(xy[:, :2], axis=0)

    xys = xy.copy()

    # get angles for points around middle point
    angles = np.arctan2(xys[:, 0] - middle_point[0], xys[:, 1] - middle_point[1])

    # sort angles counterclockwise (-180, 180)
    si = np.argsort(angles)

    # choose 'clockwiseness'
    if clockwise:
        xys = xys[si]
    else:
        xys = xys[si[::-1]]

    return xys