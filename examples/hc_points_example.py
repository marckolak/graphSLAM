import numpy as np
import matplotlib.pyplot as plt

import src.hc as hc

# points
x = np.array([[0,0], [2,0], [2,1], [0,1], [0,0]])

# convert to hc
x_h = hc.ec_hc(x)

# transformations
R = hc.rotation(np.radians(60))
T = hc.translation(np.array([4,4]))
H = hc.rigid_body_transformation(np.radians(60), np.array([4,4]))

# plot examples
plt.figure()
plt.axes().set_aspect('equal')
plt.plot(x_h[0], x_h[1])
plt.plot(R.dot(x_h)[0], R.dot(x_h)[1], label='R')
plt.plot(T.dot(x_h)[0], T.dot(x_h)[1], label='T')

plt.plot(T.dot(R.dot(x_h))[0], T.dot(R.dot(x_h))[1], label='RT')

plt.plot(R.dot(T.dot(x_h))[0], R.dot(T.dot(x_h))[1], label='TR')
plt.plot(H.dot(x_h)[0], H.dot(x_h)[1], linestyle='--', label='H', color='k')

plt.grid()
plt.legend()
plt.show()