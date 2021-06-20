from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from src.grid_maps import points2gridmap, map_corr
from src.hc import v2t, t2v, translation, rotation, ec_hc
from src.icp import match_scans


class Graph:

    def __init__(self, x=None, nodes={}, edges=[], lut=None):
        self.x = x
        self.nodes = nodes
        self.edges = edges
        self.lut = lut

    def add_edge(self, etype, fromNode, toNode, measurement):

        Edge = namedtuple(
            'Edge', ['Type', 'fromNode', 'toNode', 'measurement'])

        edge = Edge(etype, fromNode, toNode, measurement)
        self.edges.append(edge)

    def init_nodes(self, poses):

        nodes = {}

        for p, i in zip(poses, range(len(poses))):
            nodes[i] = p

        lut = {}
        x = []
        offset = 0
        for nodeId in nodes:
            lut.update({nodeId: offset})
            offset = offset + len(nodes[nodeId])
            x.append(nodes[nodeId])
        x = np.concatenate(x, axis=0)

        self.lut = lut
        self.x = x
        self.nodes = nodes

    def init_edges(self):
        self.edges = []

        for i in range(len(self.nodes) - 1):
            node_from = self.nodes[i]
            node_to = self.nodes[i + 1]

            T1 = v2t(node_from)
            T2 = v2t(node_to)

            H = np.linalg.inv(T1).dot(T2)

            self.add_edge('O', i, i + 1, t2v(H))

    def add_icp_edge(self, fromNode, toNode, transformation):

        self.add_edge('I', fromNode, toNode, transformation)

    def get_poses(self):
        poses = []

        for nodeId in self.nodes:
            dimension = len(self.nodes[nodeId])
            offset = self.lut[nodeId]

            if dimension == 3:
                pose = self.x[offset:offset + 3]
                poses.append(pose)

        return poses

    def plot(self):
        plt.figure(figsize=(25, 25))
        plt.axes().set_aspect('equal')
        plt.grid()

        icpEdgesX, icpEdgesY = [], []

        for e in self.edges:
            if e.Type == 'I':
                pts = np.r_[self.nodes[e.fromNode][:2], self.nodes[e.toNode][:2]].reshape(-1, 2)
                # print(e.fromNode, e.toNode, pts)
                icpEdgesX.append(pts[:, 0])
                icpEdgesY.append(pts[:, 1])

        icpEdgesX = np.vstack(icpEdgesX)
        icpEdgesY = np.vstack(icpEdgesY)

        for x, y, i in zip(icpEdgesX, icpEdgesY, range(len(icpEdgesY))):
            plt.plot(x, y, linewidth=1, c='darkred')

        poses = self.get_poses()
        if len(poses):
            poses = np.vstack(poses)
            plt.plot(poses.T[0], poses.T[1], 'b*', markersize=15)

            for p, i in zip(poses, range(len(poses))):
                plt.text(p[0] + 0.1, p[1] + 0.1, str(int(i)))

        plt.show()

    def get_edge_nodelist(self, edgeType):

        node_pairs = []

        for edge in self.edges:
            if edge.Type == edgeType:
                node_pairs.append([edge.fromNode, edge.toNode])

        return np.array(node_pairs)

    def update_graph(self, x):

        self.x = x.copy();
        for i in range(len(self.nodes)):
            self.nodes[i] = self.x[i * 3:3 * i + 3]


def global_error(x, g):
    x = np.r_[np.zeros(3), x]
    global_error = []
    for edge in g.edges:

        if edge.Type == 'I':
            fromIdx = g.lut[edge.fromNode]
            toIdx = g.lut[edge.toNode]

            # get node state for the current edge
            x1 = x[fromIdx:fromIdx + 3]
            x2 = x[toIdx:toIdx + 3]

            Hij = np.linalg.inv(v2t(x1)).dot(v2t(x2))
            Zij = v2t(edge.measurement)

            err = t2v(np.linalg.inv(Zij).dot(Hij))

            global_error.append(err)

    global_error = np.hstack(global_error)

    return global_error


def run_graphSlam(graph):
    optres = least_squares(global_error, graph.x[3:], args=(graph,))

    return np.r_[np.zeros(3), optres.x]


def find_icp_candidates(graph, icp_max_dist, min_corr, scans):
    poses = np.vstack(graph.get_poses())
    nlist = graph.get_edge_nodelist('I')

    nlist = np.apply_along_axis(np.sort, 1, nlist)

    # check distance between the nodes

    pose_dists = np.apply_along_axis(pose_dist, 1, poses[:, :2], poses[:, :2])
    candidates_ix = np.unique(
        np.apply_along_axis(np.sort, 1, np.argwhere((pose_dists < icp_max_dist) & (pose_dists > 0))), axis=0)

    candidates_ix = candidates_ix[~np.apply_along_axis(edge_exists, 1, candidates_ix, nlist.tolist())]

    gridmaps = []
    for p, s in zip(poses, scans):
        H = translation(p[:2]).dot(rotation(p[2]))
        st = H.dot(ec_hc(s[:, :2]))
        #     r = np.linalg.norm(st[:,:2]-p[:2], axis=1)
        gmap = points2gridmap(20, 0.05, p, st)
        gridmaps.append(gmap)

    map_corrs = []
    for (nF, nT) in candidates_ix:
        map_corrs.append(map_corr(gridmaps[nF], gridmaps[nT]))

    # print(map_corrs, candidates_ix)
    icp_candidates = candidates_ix[(np.vstack(map_corrs) > min_corr).all(axis=1)]

    return icp_candidates


def pose_dist(p, poses):
    return np.linalg.norm(poses - p, axis=1)


def edge_exists(e, nlist):
    return e.tolist() in nlist


def create_icp_edges(graph, scans, icp_pairs, sp):
    poses = graph.get_poses()

    for (i, j) in icp_pairs:
        T1 = v2t(poses[i])

        T2 = v2t(poses[j])

        H = np.linalg.inv(T1).dot(T2)

        ic = t2v(H)

        s1 = scans[i]
        s2 = scans[j]

        try:
            t = match_scans(s1, s2, ic, sp, iters=5)

            graph.add_icp_edge(i, j, t)
        except:
            print("Error for scan {}".format(i))
            t = np.r_[1, 1, 1] * np.nan
