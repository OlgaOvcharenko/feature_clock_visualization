from joblib import Parallel, delayed
import matplotlib
import statsmodels.api as sm
from matplotlib import pyplot as plt, patches
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn import preprocessing
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
from sklearn.linear_model import LinearRegression
from mycolorpy import colorlist as mcp
import scanpy as sp
from skspatial.objects import Line
from skspatial.plotting import plot_2d
import math


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
    labels = hdbscan.HDBSCAN(min_samples=5, min_cluster_size=40).fit_predict(standard_embedding)
    return labels

def draw_clusters(ax, data, alpha):
    num_clusters = data['cluster'].nunique()
    colors = mcp.gen_color(cmap="Spectral", n=num_clusters)
    for i in data['cluster'].unique():
        if i >= 0:
            points = data[data['cluster'] == i]
            if points.shape[0] > 2:
                hull = ConvexHull(points[['emb1', 'emb2']].to_numpy())
                vert = np.append(hull.vertices, hull.vertices[0])

                a, b = points['emb1'].iloc[vert], points['emb2'].iloc[vert]
                ax.plot(a, b, '--', c=colors[i], alpha=alpha)
                ax.fill(a, b, alpha=alpha, c=colors[i])

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

def project_line(data, point_a: list=[0, 0], point_b: list=[1, 1]):
    # min_point = data["emb1"].min() if data["emb1"].min() < data["emb2"].min() else data["emb2"].min()
    # max_point = data["emb1"].max() if data["emb1"].max() > data["emb2"].max() else data["emb2"].max()
    line = Line.from_points(point_a=point_a, point_b=point_b)

    projected_points = []
    for i in range(data.shape[0]):
        point = (data["emb1"][i], data["emb2"][i])
        new_point = line.project_point(point)
        new_point = rotate(new_point)
        projected_points.append(new_point[0])
    
    return np.array(projected_points)

def get_slope_from_angle(angle: float):
    res = []
    if 0 < angle < 90:
        res = [math.tan(math.radians(angle)) * 100, 100]
    elif 90 < angle < 180:
        res = [-math.tan(math.radians(180-angle)) * 100, 100]
    elif angle == 0 or angle % 180 == 0:
        res = [100, 0]
    elif angle % 90 == 0:
        res = [0, 100]
    elif angle > 180:
        raise Exception("Projection angle greater than 180.")
    return res

def rotate(point: list):
    val = math.sqrt(math.pow(point[0], 2) + math.pow(point[1], 2))
    x_new = val if point[0] >= 0 else -val
    y_new = 0
    return[x_new, y_new]

def get_importance(X, y, significance: float = 0.05):
    # lm = LinearRegression()
    # lm.fit(X, y)
    # print(lm.coef_)

    coefs, pvals, is_significant = [], [], []
    for i in range(y.shape[1]):
        # X = sm.add_constant(X)
        lm = sm.OLS(y[:, i], X).fit()
        pval = np.array(lm.pvalues)
        coefs.append(np.array(lm.params))
        pvals.append(pval)
        is_significant.append(pval <= significance)
        
    return np.array(coefs), np.array(pvals), np.array(is_significant)

def get_center(data):
    return data["emb1"].mean(), data["emb2"].mean()

def get_min_max(data):
    return data["emb1"].min(), data["emb1"].max(), data["emb2"].min(), data["emb2"].max()

def plot_central(data, angles_shift, angles, coefs, pvals, is_significant, labels, draw_clusters: bool = False):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    colors = list(mcolors.TABLEAU_COLORS.keys())
    alpha = 0.4

    if draw_clusters:
        draw_clusters(ax, data, alpha)
    sc = plt.scatter(data["emb1"], data["emb2"], c=data["label"], cmap="viridis", 
                     vmin=data["label"].min(), vmax=data["label"].max(), s=3, alpha=alpha)
    fig.colorbar(sc)

    x_center, y_center = get_center(data)
    x_min, x_max, y_min, y_max = get_min_max(data)
    radius = abs(min(x_max-x_min, y_max-y_min)) * 0.5
    circle = patches.Circle((x_center, y_center), radius=radius, edgecolor='gray', facecolor='aliceblue', linewidth=1, alpha=0.4) 
    ax.add_patch(circle)
    ax.axis('equal')

    # TODO try -radius, radius
    coefs_scaled = -radius +  (coefs - coefs.min()) * ((2 * radius) / (coefs.max()-coefs.min()))
    
    arrows = []
    for a, a_s, c, s in zip(angles, list(range(0, 360, angles_shift)), coefs_scaled, is_significant):
        a, a_s = math.radians(a), math.radians(a_s)
        x_add, y_add = math.cos(a_s) * radius, math.sin(a_s) * radius
        plt.plot((x_center, x_center+x_add), (y_center, y_center+y_add), c='gray', linestyle="--", alpha=0.7, linewidth=1)

        # plot contributions
        ind = abs(c).argsort(axis=None)[::-1]
        x_add_coefs, y_add_coefs = math.cos(a_s) * c[ind], math.sin(a_s) * c[ind]
        
        for is_s, x_c, y_c, i in zip(s[ind], x_add_coefs, y_add_coefs, ind):
            if is_s:
                col = colors[i]
                lbl = labels[i]
                arrows.append(plt.arrow(x_center, y_center, x_c, y_c, width=0.1, color=col, label=lbl))
    plt.legend(arrows, labels)
    plt.xlabel(f"UMAP 1")
    plt.ylabel(f"UMAP 2")
    plt.title("Malignant cells")
    plt.tight_layout()
    plt.show()


def try_scd():
    # read data
    file_name = 'data/neftel_malignant.h5ad'
    X = read_data(file_name)

    # cut necessary 

    # obs = ['genes_expressed', 
    #        'MESlike2', 'MESlike1', 'AClike', 'OPClike', 'NPClike1', 'NPClike2', 
    #        'G1S', 'G2M']

    obs = ['MESlike2', 'MESlike1', 'AClike', 'OPClike', 'NPClike1', 'NPClike2']
    
    new_data = X.obs[obs].dropna()
    X_new = sp.AnnData(new_data)

    # compute umap
    sp.pp.neighbors(X_new)
    sp.tl.umap(X_new, min_dist=2)
    # sp.pl.umap(X_new)

    # get clusters
    standard_embedding = X_new.obsm['X_umap']
    cluster_labels = hdbscan_low_dim(standard_embedding)

    # get labels
    mes = np.stack((X_new.X[:, 0], X_new.X[:,1]))
    npc = np.stack((X_new.X[:, 4], X_new.X[:, 5]))
    mes_max = np.max(mes, axis=0)
    npc_max = np.max(npc, axis=0)
    res_vect = np.stack((X_new.X[:, 2], X_new.X[:, 3], mes_max, npc_max))
    res_labels = np.max(res_vect, axis=0)

    # make data with labels and clusters
    data = pd.DataFrame(standard_embedding, columns=["emb1", "emb2"])
    data["cluster"] = cluster_labels
    data["label"] = res_labels

    angles_shift = 15
    angles = list(range(0, 180, angles_shift))
    projections = [project_line(data, point_a=[0, 0], point_b=get_slope_from_angle(angle)) for angle in angles]
    projections = np.array(projections).T

    coefs, pvals, is_significant = get_importance(new_data.to_numpy(), projections)
    plot_central(data, angles_shift, angles, coefs, pvals, is_significant, obs)

try_scd()
