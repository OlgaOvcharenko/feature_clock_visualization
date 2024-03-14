import csv
import random
import sys
from joblib import Parallel, delayed
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import distance_matrix

from mnist_umap_phate import hdbscan_low_dim


class Graph:
    """https://stackabuse.com/courses/graphs-in-python-theory-and-implementation/lessons/minimum-spanning-trees-prims-algorithm/"""
    
    def __init__(self, num_of_nodes):
        self.m_num_of_nodes = num_of_nodes
        self.m_graph = [[0 for _ in range(num_of_nodes)] 
                    for _ in range(num_of_nodes)]
        
    def _euclidian_distance(self, a, b):
        return np.sqrt(np.sum(np.square(a - b)))
        
    def add_all_nodes(self, data):
        def inner_loop(a, b, i, j):
            self._add_edge(i, j, self._euclidian_distance(a, b))

        for i in range(len(data)-1):
            # combined_res = Parallel(n_jobs=-1)\
            #     (delayed(inner_loop)\
            #     (data[i], data[j], i, j) for j in range(i+1, len(data)))
            for j in range(i+1, len(data)):
                inner_loop(data[i], data[j], i, j)

            

    def _add_edge(self, node1, node2, weight):
        self.m_graph[node1][node2] = weight
        self.m_graph[node2][node1] = weight


    def prims_mst(self):
        # Defining a really big number, that'll always be the highest weight in comparisons
        postitive_inf = float('inf')

        # This is a list showing which nodes are already selected 
        # so we don't pick the same node twice and we can actually know when stop looking
        selected_nodes = [False for node in range(self.m_num_of_nodes)]

        # Matrix of the resulting MST
        result = [[0 for column in range(self.m_num_of_nodes)] 
                    for row in range(self.m_num_of_nodes)]
        
        indx = 0
        # for i in range(self.m_num_of_nodes):
        #     print(self.m_graph[i])
        
        # print(selected_nodes)

        # While there are nodes that are not included in the MST, keep looking:
        while(False in selected_nodes):
            # We use the big number we created before as the possible minimum weight
            minimum = postitive_inf

            # The starting node
            start = 0

            # The ending node
            end = 0

            for i in range(self.m_num_of_nodes):
                # If the node is part of the MST, look its relationships
                if selected_nodes[i]:
                    for j in range(self.m_num_of_nodes):
                        # If the analyzed node have a path to the ending node AND its not included in the MST (to avoid cycles)
                        if (not selected_nodes[j] and self.m_graph[i][j]>0):  
                            # If the weight path analized is less than the minimum of the MST
                            if self.m_graph[i][j] < minimum:
                                # Defines the new minimum weight, the starting vertex and the ending vertex
                                minimum = self.m_graph[i][j]
                                start, end = i, j
            
            # Since we added the ending vertex to the MST, it's already selected:
            selected_nodes[end] = True

            # Filling the MST Adjacency Matrix fields:
            result[start][end] = minimum
            
            if minimum == postitive_inf:
                result[start][end] = 0

            # print("(%d.) %d - %d: %d" % (indx, start, end, result[start][end]))
            indx += 1
            
            result[end][start] = result[start][end]
        
        return result
    
    def print_MST(result):
        # node1, node2, weight in result:
        for i in range(len(result)):
            for j in range(0+i, len(result)):
                if result[i][j] != 0:
                    print("%d - %d: %d" % (i, j, result[i][j]))

def plot_MST(graph, data):
    for i in range(len(graph)):
        for j in range(0+i, len(graph)):
            if graph[i][j] != 0:
                plt.plot([data[i][0], data[j][0]],[data[i][1], data[j][1]])

def plot_MSTs(graphs, data):
    for g in graphs:
        plot_MST(g, data)
        plt.show()
    # plt.savefig("plots/test_mst.png")

def get_cluster_MST(standard_embedding, in_cluster_vector):
    graph = Graph(num_of_nodes=sum(in_cluster_vector))
    data = np.array(standard_embedding)[in_cluster_vector]
    print(data.shape)
    print(sum(in_cluster_vector))
    graph.add_all_nodes(data)
    res = graph.prims_mst()
    return res

# with open('data/phate_mnist.csv', newline='') as csvfile:
#     reader = csv.reader(csvfile, delimiter=',')
    
#     standard_embedding = []
#     for lines in reader:
#         for row in lines:
#             vals = [float(t) for t in row.strip('][').split(' ') if t.replace(" ", "") != ""]
#             standard_embedding.append(vals)
# print("Read PHATE")

# cluster_labels = hdbscan_low_dim(standard_embedding)
# standard_embedding = standard_embedding[:100]
# cluster_labels = cluster_labels[:100]

# print("Start Prim")
# combined_res = get_cluster_MST(standard_embedding, cluster_labels == 0)

# plot_MSTs([combined_res], standard_embedding)
