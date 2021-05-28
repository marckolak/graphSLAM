import pickle

import numpy as np

import src.wild_thumper as wd
from src.grid_maps import bin_scan
from src.geometry import sort_clockwise

ROT_SPEED_RIGHT = np.radians(90) / 1.1
ROT_SPEED_LEFT = np.radians(90) / 1.2
LIN_SPEED = 0.4

# mot = slam.load_motion_file('slam_2021_01_27_17_54_31_motion.txt')
mot = wd.load_motion_file('../examples/slam_may2021/may_2021_motion.txt')

scans, scans_ts = wd.load_scans('../examples/slam_may2021/may_2021_scans.txt', min_size=100, d_limit=(0.2, 15))

scans, scans_ts = wd.select_static_scans(mot, scans, scans_ts)

poses = []
for i in range(len(scans_ts)):
    poses.append(wd.get_constraint(scans_ts[i], scans_ts[0], mot, LIN_SPEED, ROT_SPEED_LEFT, ROT_SPEED_RIGHT, h0=0))

controls = []
controls.append(wd.get_controls(scans_ts[0], scans_ts[0] - 10000, mot, LIN_SPEED, ROT_SPEED_LEFT, ROT_SPEED_RIGHT, h0=0,
                                control_format=True))
for i in range(1, len(scans_ts)):
    controls.append(
        wd.get_controls(scans_ts[i], scans_ts[i - 1], mot, LIN_SPEED, ROT_SPEED_LEFT, ROT_SPEED_RIGHT, h0=0,
                        control_format=True))

# scans_binned = []
# for s in scans:
#     sb, (bcx, bcy) = bin_scan(s, 0.04)
#     sbs = sort_clockwise(np.c_[bcx[sb[:, 0]], bcy[sb[:, 1]]], middle_point=np.array([0, 0]))
#
#     angles = np.arctan2(sbs[:, 1], sbs[:, 0])
#     angles[angles < 0] = angles[angles < 0] + np.pi * 2
#     sbs = np.c_[sbs, angles]
#     sbs = sbs[np.argsort(sbs[:, 2])]
#     scans_binned.append(sbs)
#
# scans = scans_binned

with open('measured_scan_may2021.pickle', 'wb') as f:
    pickle.dump((scans, poses, controls), f)
