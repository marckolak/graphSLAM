import pickle

import numpy as np

import wild_thumper as wd

# speed values for motor speed set to 0.2 (see more at https://github.com/marckolak/wtController)
ROT_SPEED_RIGHT = np.radians(90) / 1.1
ROT_SPEED_LEFT = np.radians(90) / 1.2
LIN_SPEED = 0.4

# load motion file
mot = wd.load_motion_file('./mapping_motion.txt')

# load scan files
scans, scans_ts = wd.load_scans('./mapping_scan.txt', min_size=100, d_limit=(0.2, 15))

# extract scans taken in static positions
scans, scans_ts = wd.select_static_scans(mot, scans, scans_ts)

# estimate poses based on odometry
poses = []
for i in range(len(scans_ts)):
    poses.append(wd.get_constraint(scans_ts[i], scans_ts[0], mot, LIN_SPEED, ROT_SPEED_LEFT, ROT_SPEED_RIGHT, h0=0))

# retrieve controls sent to the robot
controls = []
controls.append(wd.get_controls(scans_ts[0], scans_ts[0] - 10000, mot, LIN_SPEED, ROT_SPEED_LEFT, ROT_SPEED_RIGHT, h0=0,
                                control_format=True))
for i in range(1, len(scans_ts)):
    controls.append(
        wd.get_controls(scans_ts[i], scans_ts[i - 1], mot, LIN_SPEED, ROT_SPEED_LEFT, ROT_SPEED_RIGHT, h0=0,
                        control_format=True))

# pickle the preprocessed scans
with open('mapping.pickle', 'wb') as f:
    pickle.dump((scans, poses, controls), f)
