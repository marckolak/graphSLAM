## Example illustrating icp scan matching

import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from src.grid_maps import GridMap
from src.scan_processing import scan2xy
import src.feature_detection as features
import src.hc as hc
import src.icp as icp

with open('measured_scan_may2021.pickle', 'rb') as f:
    (scans, poses, controls) = pickle.load(f)

poses = np.vstack(poses)
scans = scans[1:]
poses = poses[1:]
controls = controls[1:]
controls = np.vstack(controls)
controls[:,0] = controls[:,0]*0.7
controls[:,1] = controls[:,1] *1


sp = {
    "extr_angle_range": np.pi / 3,
    "extr_split_th": 0.1,
    "extr_min_len": 0.6,
    "extr_min_points": 10,

    "mrg_max_dist": -0.2,
    "mrg_a_tol": 0.1,
    "mrg_b_tol": 0.1,
    "mrg_fit_tol": 0.1,

    "association_th": [0.3],


}

from progress.bar import Bar
bar = Bar('Processing', max=20)
for i in range(20):
    # Do some work
    bar.next()

bar.finish()

bar = Bar('Processing', max=len(scans))