from joblib import Parallel, delayed
import matplotlib
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
from MST_prim import Graph, plot_MSTs
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import phate
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
from shapely.ops import unary_union
import csv
import pandas as pd
import matplotlib.colors as mcolors
from mycolorpy import colorlist as mcp


def read_mnist_data():
    mnist = fetch_openml('mnist_784', version=1)
    mnist.target = mnist.target.astype(int)
    return mnist

def calculate_dim_red(data, method: str = 'phate'):
    assert method in ['phate', 'tsne', 'umap'], f'Dimenionality reduction {method} is not supported.'
    dr_model = phate.PHATE() if method == 'phate' else umap.UMAP(random_state=7, min_dist=0.9, n_neighbors=30, n_components=2)
    standard_embedding = dr_model.fit_transform(data)
    return standard_embedding

def hdbscan_low_dim(standard_embedding):
    # kmeans_labels = cluster.KMeans(n_clusters=10).fit_predict(mnist.data)
    labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500).fit_predict(standard_embedding)
    return labels

def draw_clusters(ax, data):
    num_clusters = data['cluster'].nunique()
    colors = mcp.gen_color(cmap="Spectral", n=num_clusters)
    print(num_clusters)
    for i in range(num_clusters-1):
        points = data[data['cluster'] == i]
        if points.shape[0] > 2:
            hull = ConvexHull(points[['emb1', 'emb2']].to_numpy())
            vert = np.append(hull.vertices, hull.vertices[0])

            a, b = points['emb1'].iloc[vert], points['emb2'].iloc[vert]
            ax.plot(a, b, '--', c=colors[i], alpha=0.5)
            ax.fill(a, b, alpha=0.5, c=colors[i])

def plot(mnist, emb_df, method, standard_embedding, combined_res):
    fig, ax = plt.subplots(1, figsize=(7, 5))
    
    draw_clusters(ax, emb_df)

    colors = mcp.gen_color(cmap="Spectral", n=len(mnist.target.unique()))
    j = 0
    for i in mnist.target.unique():
        plt.scatter(emb_df['emb1'][mnist.target==i], emb_df['emb2'][mnist.target==i], c=colors[j], s=10, label=str(i))
        j += 1
        
    plot_MSTs(ax, combined_res, standard_embedding)

    plt.xlabel(f"{method} 1")
    plt.ylabel(f"{method} 2")
    ax.legend()
    plt.tight_layout()
    plt.show()

def get_cluster_MST(standard_embedding, in_cluster_vector):
    graph = Graph(num_of_nodes=sum(in_cluster_vector))
    graph.add_all_nodes(np.array(standard_embedding)[in_cluster_vector])
    res = graph.prims_mst()
    return res

mnist = read_mnist_data()

read = True
if read:
    with open('data/phate_mnist.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        
        standard_embedding = []
        for lines in reader:
            for row in lines:
                vals = [float(t) for t in row.strip('][').split(' ') if t.replace(" ", "") != ""]
                standard_embedding.append(vals)
else:
    standard_embedding = calculate_dim_red(mnist.data)

write = False
if write:
    with open('data/phate_mnist.csv', 'w') as f:
        # create the csv writer
        writer = csv.writer(f)
        # write a row to the csv file
        writer.writerow(standard_embedding)

cluster_labels = hdbscan_low_dim(standard_embedding)

ix = 100
mnist.data = mnist.data[:ix]
mnist.target = mnist.target[:ix]
standard_embedding = standard_embedding[:ix]
cluster_labels = cluster_labels[:ix]

mnist.data['cluster'] = cluster_labels
emb_df = pd.DataFrame(standard_embedding, columns =['emb1', 'emb2'])
emb_df['cluster'] = cluster_labels

clusters_graphs = dict()
combined_res = Parallel(-1)\
    (delayed(get_cluster_MST)\
    (standard_embedding, cluster_labels == label) for label in np.unique(cluster_labels[0]))

plot(mnist, emb_df, 'phate', standard_embedding, combined_res)
