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



### lines example

x1h = hc.ec_hc(np.r_[0, 0]) + np.random.randn() * 0.001
x2h = hc.ec_hc(np.r_[2, 0]) + np.random.randn() * 0.001
x3h = hc.ec_hc(np.r_[2, 1]) + np.random.randn() * 0.001
x4h = hc.ec_hc(np.r_[0, 1]) + np.random.randn() * 0.001

l1 = hc.line_jp(x1h, x2h)
l2 = hc.line_jp(x2h, x3h)
l3 = hc.line_jp(x3h, x4h)
l4 = hc.line_jp(x4h, x1h)

lines = np.hstack([l1, l2, l3, l4]).reshape(-1, 3).T

plt.figure()
plt.axes().set_aspect('equal')

for l in lines.T:
    p = hc.line_si(l)
    plt.plot(np.r_[-20, 20], np.polyval(p, np.r_[-20, 20]))

plt.gca().set_prop_cycle(None)
for l in hc.transform_line(lines, hc.translation(np.r_[5, 5])).T:
    p = hc.line_si(l)
    plt.plot(np.r_[-20, 20], np.polyval(p, np.r_[-20, 20]), linestyle='--')

plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.show()
