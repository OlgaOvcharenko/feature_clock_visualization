from joblib import Parallel, delayed
import matplotlib
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
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
import scanpy as sp
from skspatial.objects import Line
from skspatial.plotting import plot_2d


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
    labels = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=30).fit_predict(standard_embedding)
    return labels

def draw_clusters(ax, data):
    num_clusters = data['cluster'].nunique()
    colors = mcp.gen_color(cmap="Spectral", n=num_clusters)
    for i in data['cluster'].unique():
        if i >= 0:
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

def main():
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


def read_data(path):
    return sp.read(path)  # anndata

def plot_scd(data, projected_points, min_point, max_point):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    
    draw_clusters(ax, data)

    sc = plt.scatter(data["emb1"], data["emb2"], c=data["label"], cmap="viridis", vmin=data["label"].min(), vmax=data["label"].max(), s=5, alpha=0.5)
    fig.colorbar(sc)

    # Global projection
    plt.plot([min_point, max_point + 1], [min_point, max_point + 1], c="grey", alpha=0.4)
    plt.scatter(projected_points[:, 0], projected_points[:, 1], c="black", s=1)

    plt.xlabel(f"UMAP 1")
    plt.ylabel(f"UMAP 2")
    plt.title("Malignant cells")
    plt.tight_layout()
    plt.show()

def project_line(data):
    min_point = data["emb1"].min() if data["emb1"].min() < data["emb2"].min() else data["emb2"].min()
    max_point = data["emb1"].max() if data["emb1"].max() > data["emb2"].max() else data["emb2"].max()
    # line = Line.from_points(point_a=[min_point, min_point], point_b=[max_point, max_point])
    line = Line.from_points(point_a=[0, 0], point_b=[1, 1])

    # line = Line.from_points(point_a=[0, 0], point_b=[10, 0])
    # point = (100, 1)
    # print(point)
    # print(line.project_point(point))
    # return
    
    projected_points = []
    for i in range(data.shape[0]):
        point = (data["emb1"][i], data["emb2"][i])
        projected_points.append(line.project_point(point))
    []
    return np.array(projected_points), min_point, max_point

def try_scd():
    file_name = 'data/neftel_malignant.h5ad'
    X = read_data(file_name)
    standard_embedding = X.obsm['X_umap']
    cluster_labels = hdbscan_low_dim(standard_embedding)

    mes = np.stack((X.obs.MESlike1, X.obs.MESlike2))
    npc = np.stack((X.obs.NPClike1, X.obs.NPClike2))
    mes_max = np.max(mes, axis=0)
    npc_max = np.max(npc, axis=0)

    res_vect = np.stack((X.obs.AClike, X.obs.OPClike, mes_max, npc_max))
    res_labels = np.max(res_vect, axis=0)

    data = pd.DataFrame(standard_embedding, columns=["emb1", "emb2"])
    data["cluster"] = cluster_labels
    data["label"] = res_labels

    projected_points, min_point, max_point = project_line(data)

    plot_scd(data, projected_points, min_point, max_point)

try_scd()
