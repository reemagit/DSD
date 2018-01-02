#!/usr/bin/python
# This file is supposed to replace the dsdcore.py file in the original cDSD package

from __future__ import division
import numpy as np
import operator
from numpy.linalg import inv
from scipy.spatial.distance import pdist, squareform

def build_transition_matrix(adjacency_graph):
    degs = adjacency_graph.sum(axis=1)

    transition = adjacency_graph / degs[:,None]
    transition[degs==0,:] = 0
    transition[degs==0,degs==0] = 1
    return transition

def calc_hescotts(transition, iters, v=True, n=None):
    # params v and n are not used, left for compatibility with original dsdcore script
    nRw = iters
    p = transition
    n = p.shape[0]
    c = np.eye(n)
    c0 = np.eye(n)
    for i in xrange(nRw):
        c = np.dot(c, p) + c0
    return c

def calc_dsd(transition):
    return squareform(pdist(transition,metric='cityblock'))              


def add_self_edges(adjacency_graph, base_weight=1):    
    n = np.size(adjacency_graph[0])
    ident = np.identity(n)*base_weight
    return np.add(adjacency_graph,ident)

    
