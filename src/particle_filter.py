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



def update(particles, weights, s, gridmap, res):

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

    return hits
