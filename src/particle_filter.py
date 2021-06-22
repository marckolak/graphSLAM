import numpy as np
from numpy.random import uniform
from src.grid_maps import is_inside_map, world2map
import scipy
from src.icp import  cp_dist
import src.hc as hc

def predict(particles,c, rot_noise, d_noise):

    d_n = (c[0] + np.random.randn(particles.shape[0], 1) * d_noise).ravel()
    r_n = (c[1] + np.random.randn(particles.shape[0], 1) * rot_noise).ravel()
    particles = particles + np.c_[np.cos(particles[:,2])*d_n, np.sin(particles[:,2])*d_n, r_n]

    return particles


def init_uniform_particles(x_range, y_range, head_range, N):

    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(head_range[0], head_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles


def init_uniform_particles_inside(x_range, y_range, head_range, N, gridmap, res):

    particles = []

    generated = False
    particles_left = N
    while particles_left >0:
        part = init_uniform_particles(x_range, y_range, head_range, particles_left)

        inside_parts = np.apply_along_axis(is_inside_map, 1, part, gridmap, res)

        part = part[inside_parts]
        particles.append(part)
        particles_left = particles_left - len(part)
        # print(inside_parts)

    return np.vstack(particles)



def update(particles, s, gridmap, res):

    sh = hc.ec_hc(s)
    hits = []
    for p in particles:
        H = hc.translation(p[:2]).dot(hc.rotation(p[2]))
        shct = world2map(H.dot(sh), gridmap, res)
        if shct.max() < gridmap.shape[0]:
            hit = (gridmap[shct[:,0], shct[:,1]]==1).sum()
        else:
            hit = 1
        hits.append(hit)

    hits = (np.array(hits)/10)**2
    hits = hits + 1e-40
    weights = hits/ hits.sum()

    return weights


def resample(particles, weights):
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # avoid round-off error
    indexes = np.searchsorted(cumulative_sum, np.random.rand(N))

    # resample according to indexes
    particles[:] = particles[indexes]
    weights.fill(1.0 / N)

    return particles



def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    mean = np.average(pos, weights=weights, axis=0)
    var  = np.average((pos - mean)**2, weights=weights, axis=0)
    return mean, var