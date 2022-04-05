"""
The pypositioning.algorithms.slam.feature_detection module contains functions allowing for feature detection in LiDAR
scans.

Copyright (C) 2021 Marcin Kolakowski
"""
import itertools

import numpy as np



def get_line_features(s, a, extr_angle_range=np.pi / 2, extr_split_th=0.1, extr_min_len=0.2,
                      extr_min_points=2, mrg_max_dist=1, mrg_a_tol=0.9, mrg_b_tol=0.5, mrg_fit_tol=0.5, corners=[]):
    lines, points = extract_lines(s, a, angle_dist=extr_angle_range, split_th=extr_split_th,
                                  min_len=extr_min_len, min_points=extr_min_points, corners=corners)
    lines, points = merge_lines(lines, points, max_dist=mrg_max_dist, a_tol=mrg_a_tol, b_tol=mrg_b_tol,
                                fit_tol=mrg_fit_tol)

    return lines, points


def extract_lines(s, angles, angle_dist=np.pi / 6, split_th=0.05, min_len=0.2, min_points=5, corners=[]):
    lines = []
    line_points = []

    for i1 in np.arange(0, 2 * np.pi, angle_dist):
        i2 = (i1 + angle_dist)
        ix = np.argwhere((i1 < angles) & (angles < i2))
        p = s[ix.T[0]]
        polys, points, lenghts = split_into_lines(p, split_th, min_len, min_points, corners)
        polys = [polys[i] for i in np.argwhere(np.array(lenghts) > min_len).T[0]]
        points = [points[i] for i in np.argwhere(np.array(lenghts) > min_len).T[0]]

        lines += polys
        line_points += points

    return np.array(lines), line_points


def merge_lines(segments, line_points, max_dist, a_tol, b_tol, fit_tol, a_as_angle=False):
    end_pts = []
    for i in range(len(line_points)):
        end_pts.append(np.c_[line_points[i][0], line_points[i][-1]].T)

    end_pts = np.array(end_pts)

    # lines[:, 0] = np.abs(lines[:, 0])
    # print(lines)
    i = 0
    to_merge = []
    same_line = []
    merging = False

    angles = np.arctan2(segments[:, 0], np.ones(segments[:, 0].shape))
    if a_as_angle:
        segments[:, 0] = angles
        # print(segments[:, 0])

    # print(angles)
    while i < len(segments):

        # print('dist: ', np.linalg.norm(end_pts[i,1] - end_pts[i+1,0]), 'coeff dist: ', np.abs(lines[i]-lines[i+1]))

        # check if the points are mergable
        # print(np.linalg.norm(end_pts[i, 1] - end_pts[(i + 1) % 1, 0]), np.abs(
        #         angles[i] - angles[(i + 1) % len(lines)]), lines[i, 1] - lines[(i + 1) % len(lines), 1])

        end_pts_dist = np.linalg.norm(end_pts[i, 1] - end_pts[(i + 1) % len(segments), 0])
        angle_diff = np.abs(np.abs(angles[i]) - np.abs(angles[(i + 1) % len(segments)]))
        mergable = False
        # print(np.abs(np.abs(angles[i]) - np.abs(angles[(i + 1) % len(segments)])) )
        if end_pts_dist < max_dist and angle_diff < a_tol:
            # print('angles close', np.abs(angles[i]))
            if 0.3 * np.pi < np.abs(angles[i]) < 0.7 * np.pi:  # steep lines
                evals = np.polyval(segments[(i + 1) % len(segments)], line_points[i][:, 0])
                fit_err = np.linalg.norm(np.c_[line_points[i][:, 0], evals] - line_points[i], axis=1).mean()
                if fit_err < fit_tol:
                    mergable = True
            else:  # not steep lines
                if np.abs(segments[i, 1] - segments[(i + 1) % len(segments), 1]) < b_tol:
                    mergable = True

        if mergable:
            if merging:  # if already merging append next index to list
                same_line.append((i + 1) % len(segments))
            else:  # if not merging start a new list
                same_line = [i, (i + 1) % len(segments)]
                merging = True

            i = i + 1  # check the next segment

        else:  # segment not mergable
            if merging:  # stop merging
                merging = False
                to_merge.append(same_line)  # append meergable indices to final to merge list
                # do not increment - check for the same index if its mergable with the next segment
            else:
                i = i + 1
    # print(to_merge)
    # merge lines:
    lines_am = []
    points_am = []
    for m in to_merge:
        pm = np.vstack([line_points[i] for i in m])

        p = pair2line(pm[0], pm[-1])
        # p = np.polyfit(pm[:, 0], pm[:, 1], 1)
        points_am.append(pm)
        lines_am.append(p)

    # get indices of segments, which were merged
    merge_ind = []
    for mi in to_merge:
        merge_ind += mi

        # append not merged lines
    for i in range(len(segments)):
        if i not in merge_ind:
            lines_am.append(segments[i])
            points_am.append(line_points[i])

    return np.vstack(lines_am), points_am


def find_corners(lines, points):
    corners = []
    for i in range(len(lines)):
        for j in [k for k in range(len(lines)) if k != i]:
            are_perpendicular = np.radians(75) < np.abs(
                np.arctan2(lines[i][0], 1) - np.arctan2(lines[j][0], 1)) < np.radians(105)

            end_pts_dits = np.array([np.linalg.norm(points[i][i1] - points[j][i2]) for i1, i2 in
                         itertools.product([0, -1], [0, -1])])

            are_close = end_pts_dits.min()<0.3
            print(np.abs(np.arctan2(lines[i][0], 1) - np.arctan2(lines[i][0], 1)))
            if are_close and are_perpendicular:
                ipx = (lines[j][1]- lines[i][1])/(lines[i][0] - lines[j][0])
                ipy = lines[i][0]*ipx+lines[i][1]

                corners.append([ipx, ipy])

    return np.array(corners)


def split_into_lines(p, split_th, min_len, min_points, corners):
    if p.shape[0] < min_points:
        return [], [], []
    # get line params between two edge points
    pol = pair2line(p[0], p[-1], a_as_angle=False)

    # calculate distance
    d = dist_from_line(p, pol)

    if d.max() > split_th:

        ix_max = np.argmax(d)  # split point

        # invoke for split set of points
        pol1, p1, l1 = split_into_lines(p[:ix_max], split_th, min_len, min_points, corners)
        pol2, p2, l2 = split_into_lines(p[ix_max:], split_th, min_len, min_points, corners)

        # print(len(pol1), len(pol2))
        if len(pol1) and len(pol2):
            # if np.radians(70) < np.abs(np.arctan2(pol1[-1][0], 1) - np.arctan2(pol2[0][0], 1)) < np.radians(110):
            corners.append(p[ix_max])

        # concatenate results
        pol = pol1 + pol2
        pout = p1 + p2
        lengths = l1 + l2
        return pol, pout, lengths

    else:  # if no distinct split point - segment is whole
        line_length = np.linalg.norm(p[0] - p[-1])
        points_N = p.shape[0]
        ipd = np.linalg.norm(np.diff(p, axis=0), axis=1)
        ppm = points_N / line_length
        # print(ppm, np.quantile(ipd,0.9))

        if line_length > 0 and points_N > 0 and ppm > 0:
            return [pol], [p], [line_length]
        else:
            return [], [], []


def dist_from_line(xy, sl_in):
    """ Calculate distance from a line

    Parameters
    ----------
    xy: ndarray
        point coordinates
    sl_in: ndarray
        slope and intercept of the line

    Returns
    -------
    d: ndarray
        distances of points from xy from the line
    """
    if xy.ndim >= 2:
        return np.abs(xy[:, 0] * sl_in[0] - xy[:, 1] + sl_in[1]) / np.sqrt(1 + sl_in[0] ** 2)
    else:
        return np.abs(xy[0] * sl_in[0] - xy[1] + sl_in[1]) / np.sqrt(1 + sl_in[0] ** 2)


def dist_from_lines(xy, lines):
    """ Calculate distance from a line

    Parameters
    ----------
    xy: ndarray
        point coordinates
    sl_in: ndarray
        slope and intercept of the line

    Returns
    -------
    d: ndarray
        distances of points from xy from the line
    """

    return np.abs(xy[0] * lines[:, 0] - xy[1] + lines[:, 1]) / np.sqrt(1 + lines[:, 0] ** 2)


def pair2line(p1, p2, a_as_angle=False):
    """

    Parameters
    ----------
    p1
    p2

    Returns
    -------

    """
    if p2[0] == p1[0]:
        p2[0] += 0.0001
        p1[0] -= 0.0001

    a = (p2[1] - p1[1]) / (p2[0] - p1[0])
    b = p1[1] - a * p1[0]

    if a_as_angle:
        a = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    return np.array((a, b))


def closest_point(xy, p):
    x = (xy[0] + p[0] * xy[0] - p[0] * p[1]) / (p[0] ** 2 + 1)
    y = (p[0] * (xy[0] + p[0] * xy[0]) + p[1]) / (p[0] ** 2 + 1)
    return np.r_[x, y]


def closest_point_on_line(p, xy):
    x = (xy[0] + p[:, 0] * xy[1] - p[:, 0] * p[:, 1]) / (p[:, 0] ** 2 + 1)
    y = (p[:, 0] * (xy[0] + p[:, 0] * xy[1]) + p[:, 1]) / (p[:, 0] ** 2 + 1)
    return np.c_[x, y]


