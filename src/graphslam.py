from collections import namedtuple
import numpy as np



class Graph:

    def __init__(self, x=None, nodes={}, edges=[], lut=None):
        self.x = x
        self.nodes=nodes
        self.edges = edges
        self.lut = lut



    def add_edge(self, etype, fromNode, toNode, measurement):

        Edge = namedtuple(
            'Edge', ['Type', 'fromNode', 'toNode', 'measurement', 'information'])

        edge = Edge(etype, fromNode, toNode, measurement)


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