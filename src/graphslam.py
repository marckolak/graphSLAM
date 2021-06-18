from collections import namedtuple
import numpy as np
from  src.hc import v2t, t2v
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


class Graph:

    def __init__(self, x=None, nodes={}, edges=[], lut=None):
        self.x = x
        self.nodes=nodes
        self.edges = edges
        self.lut = lut



    def add_edge(self, etype, fromNode, toNode, measurement):

        Edge = namedtuple(
            'Edge', ['Type', 'fromNode', 'toNode', 'measurement'])

        edge = Edge(etype, fromNode, toNode, measurement)
        self.edges.append(edge)

    def init_nodes(self, poses):

        nodes = {}

        for p, i  in zip(poses, range(len(poses))):
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
        self.edges=[]

        for i in range(len(self.nodes)-1):
            node_from = self.nodes[i]
            node_to = self.nodes[i+1]

            T1 = v2t(node_from)
            T2 = v2t(node_to)

            H = np.linalg.inv(T1).dot(T2)

            self.add_edge('O', i, i+1, t2v(H))


    def add_icp_edge(self, fromNode, toNode, transformation):

        self.add_edge('I', fromNode,toNode, transformation)

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
        plt.figure(figsize=(10,10))
        plt.axes().set_aspect('equal')
        plt.grid()

        poses = self.get_poses()
        if len(poses):
            poses = np.vstack(poses)
            plt.plot(poses.T[0], poses.T[1], '*', markersize=15)

        icpEdgesX, icpEdgesY = [], []

        for e in self.edges:
            if e.Type == 'I':
                pts = np.r_[self.nodes[e.fromNode][:2], self.nodes[e.toNode][:2]].reshape(-1,2)
                icpEdgesX.append(pts[:, 0])
                icpEdgesY.append(pts[:, 1])

        icpEdgesX = np.vstack(icpEdgesX)
        icpEdgesY = np.vstack(icpEdgesY)

        plt.plot(icpEdgesX, icpEdgesY, linewidth=2)


        plt.show()




def global_error(x, g):

    x = np.r_[np.zeros(3), x]
    global_error = []
    for edge in g.edges:

        if edge.Type=='I':
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

    return np.r_[np.zeros(3),optres.x]
