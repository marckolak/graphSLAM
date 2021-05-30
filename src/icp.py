import numpy as np
from scipy.optimize import least_squares

import src.feature_detection as features
import src.hc as hc


def match_scans(s1, s2, ic, sp, iters=5):
    x = ic
    for i in range(iters):
        # transform second scan using the result from the last iteration
        H = transformation_matrix(x[2], x[:2])
        s2t = hc.hc_ec(H.dot(hc.ec_hc(s2[:, :2])))

        # Extract line features from the scans
        lp1 = features.get_line_features(s1[:, :2], s1[:, 2], sp["extr_angle_range"], sp["extr_split_th"],
                                         sp["extr_min_len"], sp["extr_min_points"], sp["mrg_max_dist"],
                                         sp["mrg_a_tol"], sp["mrg_b_tol"], sp["mrg_fit_tol"], corners=[])

        lp2 = features.get_line_features(s2t[:, :2], s2[:, 2], sp["extr_angle_range"], sp["extr_split_th"],
                                         sp["extr_min_len"], sp["extr_min_points"], sp["mrg_max_dist"],
                                         sp["mrg_a_tol"], sp["mrg_b_tol"], sp["mrg_fit_tol"], corners=[])

        # get corresponding line segments and points, which are close to each other
        corr_points = corresponding_line_points(lp1, lp2, H)

        optres = least_squares(icp_err_fun, x, args=(corr_points,))

        x = optres.x
        # print(x)

    return x

def transformation_matrix(angle, t):
    Tr = hc.translation(np.r_[0.14, -0.01])
    T = hc.translation(t)

    R = hc.rotation(angle)
    H = T.dot(np.linalg.inv(Tr).dot(R.dot(Tr)))

    return H


def corresponding_line_points(lp1, lp2, H, an_th=0.1, d_th=0.05, corr_points_th=0.1):
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
        ldif = np.abs(br2 - b)
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
    r = features.dist_from_lines(x[:2], lines)
    cp = features.closest_point_on_line(lines, x[:2]) - x[:2]
    bearing = np.arctan2(cp[:, 1], cp[:, 0])

    return np.c_[bearing, r]


def cp_dist(a, p):
    return np.min(np.linalg.norm(a - p, axis=1))


def icp_err_fun(x, corr_points):
    H = transformation_matrix(x[2], x[:2])

    cper = []
    for p1, p2 in corr_points:
        cper.append(np.mean(np.apply_along_axis(cp_dist, 1, hc.hc_ec(H.dot(hc.ec_hc(p2))), p1)))


    cper = np.hstack(cper)

    return cper
