#!/usr/bin/env python
#-*-coding: utf-8 -*-

import sys
import os
import networkx as nx

if __name__=='__main__':
    if len(sys.argv) != 2:
        print 'python script.py networkFile'
        sys.exit(0)
    networkFile = sys.argv[1]
    G = nx.read_weighted_edgelist(networkFile, delimiter='\t')
    components = list(nx.connected_component_subgraphs(G))
    res = nx.Graph()
    for sub in components:
        if sub.number_of_nodes() > res.number_of_nodes():
            res = sub
    nx.write_edgelist(res, os.path.splitext(networkFile)[0]+'BiggestComponent.txt', delimiter='\t', data=['weight'])
