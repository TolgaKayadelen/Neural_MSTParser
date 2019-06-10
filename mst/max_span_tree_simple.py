# -*- coding: utf-8 -*-

"""MST decoder for returning the maximum spanning tree of a sentence."""

import numpy as np
from collections import defaultdict
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx

import logging
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)



class MST:
    """Chu-Liu-Edmonds decoder for finding the maximum spanning tree."""
    
    def __init__(self, scores):
        """Initialize this class with a score matrix.
        
        Args:
            scores: np.array, scores[i][j] represents the weight from node j (dep)
                to node i (head).
        
        """
        self.length = scores.shape[0]
        #self.scores = scores * (1 - np.eye(self.length))
        self.scores = scores
        #print("scores: {}".format(self.scores))
        self.heads = np.argmax(self.scores, axis=1)
        self.heads[0] = 0 # root has no haed.
        logging.info("Initial heads: {}".format(self.heads))
        self.tokens = np.arange(1, self.length)
        #print("tokens: {}".format(self.tokens))
        # find the tokens whose head is the root (i.e node 0)
        self.roots = np.where(self.heads[self.tokens] == 0)[0] + 1
        #print("Initial roots: {}".format(self.roots))
        
    
    def Decode(self):
        
        # first deal with roots.
        # in case where no token points to the root or more than one token
        # point to the root, recalculate the score matrix to have exactly 
        # one token pointing to the root. 
        if len(self.roots) < 1:
            logging.info("No token is pointing to the root, choosing one")
            # the score for each token selecting the root as head.
            root_scores = self.scores[self.tokens, 0]
            #print("root_scores: {}".format(root_scores))
            # the score for each token pointing to its head
            head_scores = self.scores[self.tokens, self.heads[self.tokens]]
            #print("head_scores: {}".format(head_scores))
            # find the most likely token that can be pointing to the root.
            new_root = self.tokens[np.argmax(root_scores / head_scores)]
            logging.info("New root is {}".format(new_root))
            # change the head of the new root token to be 0.
            self.heads[new_root] = 0
            #print("new_heads: {}".format(self.heads))
        
        elif len(self.roots) > 1:
            logging.info("Multiple tokens are pointing to root, choosing one")
            # get the score for each token pointing to the head
            root_scores = self.scores[self.roots, 0]
            #print("root_scores: {}".format(root_scores))
            # turn the head scores for the multiple tokens pointing to root to 0.
            self.scores[self.roots, 0] = 0
            # pick new heads for these tokens.
            new_heads = np.argmax(self.scores[self.roots][:, self.tokens], axis=1) + 1
            #print("New candidate heads for the roots: {}".format(new_heads))
            # find the new root.
            new_root = self.roots[np.argmin(self.scores[self.roots, new_heads] / root_scores)]
            logging.info("New root is {}".format(new_root))
            # assign the new heads back into the matrix.
            self.heads[self.roots] = new_heads
            # change the head of the new root token to be 0.
            self.heads[new_root] = 0
            #print("new_heads: {}".format(self.heads))
        
        vertices = set((0,))
        edges = defaultdict(set)
        for dep, head in enumerate(self.heads[self.tokens]):
            vertices.add(dep+1)
            edges[head].add(dep+1)
        #print("vertices: {}".format(vertices))
        #print("edges: {}".format(edges))
        
        # Identify cycles and contract.
        for cycle in self._GetCycle(vertices, edges):
            logging.info("Found cycle! - {}".format(cycle))
            dependents = set()
            to_visit = set(cycle)
            while len(to_visit) > 0:
                node = to_visit.pop()
                logging.info("Contraction, visiting node: {}".format(node))
                if node not in dependents:
                    dependents.add(node)
                    to_visit.update(edges[node])
            cycle = np.array(list(cycle))
            old_heads = self.heads[cycle]
            old_scores = self.scores[cycle, old_heads]
            non_heads = np.array(list(dependents))
            self.scores[np.repeat(cycle, len(non_heads)),
                        np.repeat([non_heads], len(cycle), axis=0).flatten()] = 0
            new_heads = np.argmax(self.scores[cycle][:, self.tokens], axis=1) + 1
            new_scores = self.scores[cycle, new_heads] / old_scores
            change = np.argmax(new_scores)
            changed_cycle = cycle[change]
            old_head = old_heads[change]
            new_head = new_heads[change]
            self.heads[changed_cycle] = new_head
            edges[new_head].add(changed_cycle)
            edges[old_head].remove(changed_cycle)
        
        logging.info("Final Heads! {}".format(self.heads))
        return self.heads
    
    
    def _GetCycle(self, vertices, edges):
        """Given a graph as (vertices, edges) finds and returns cycles in it.
        https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
        
        Args:
            vertices: list of nodes.
            edges: defaultdict(set), edges representing head-dep relations, 
                i.e. edges[head] = dep.
        Returns:
            list of cycles.
        """
        _index = [0]
        _stack = []
        _indices = {}
        _lowlinks = {}
        _onstack = defaultdict(lambda: False)
        _SCCs = []
        
        def _strongconnect(v):
            _indices[v] = _index[0]
            _lowlinks[v] = _index[0]
            _index[0] += 1
            _stack.append(v)
            _onstack[v] = True
            
            for w in edges[v]:
                if w not in _indices:
                    _strongconnect(w)
                    _lowlinks[v] = min(_lowlinks[v], _lowlinks[w])
                elif _onstack[w]:
                    _lowlinks[v] = min(_lowlinks[v], _indices[w])
            
            if _lowlinks[v] == _indices[v]:
                SCC = set()
                while True:
                    w = _stack.pop()
                    _onstack[w] = False
                    SCC.add(w)
                    if not (w != v):
                        break
                _SCCs.append(SCC)
        
        for v in vertices:
            if v not in _indices:
                _strongconnect(v)
        
        return [SCC for SCC in _SCCs if len(SCC) > 1]


if __name__ == "__main__":
    #this matrix has multiple tokens pointing to the head
    test_2 = np.array([
       [-1., 9., 10., 9.],
       [9., -1., 30., 11.],
       [10., 20., -1., 0.],
       [9., 3., 30., -1.]
       ])
    decoder = MST(test_2)
    decoder.Decode()
    
    
    
