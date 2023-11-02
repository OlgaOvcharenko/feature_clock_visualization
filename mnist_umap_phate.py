from joblib import Parallel, delayed
import matplotlib
import statsmodels.api as sm
from matplotlib import pyplot as plt, patches
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import umap
import hdbscan
import phate
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import numpy as np
import csv
import pandas as pd
import matplotlib.colors as mcolors
from sklearn.linear_model import LinearRegression
from mycolorpy import colorlist as mcp
import scanpy as sp
from skspatial.objects import Line
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
    labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=30).fit_predict(standard_embedding)
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

def project_line(data, angle: float, point_a: list=[0, 0], point_b: list=[1, 1]):
    line = Line.from_points(point_a=point_a, point_b=point_b)

    projected_points = []
    for i in range(data.shape[0]):
        point = (data["emb1"][i], data["emb2"][i])
        new_point = line.project_point(point)
        new_point = rotate(new_point, angle)
        projected_points.append(new_point[0])
    
    return np.array(projected_points)

def get_slope_from_angle(angle: float):
   return [math.cos(math.radians(angle)) * 100, math.sin(math.radians(angle)) * 100]

def rotate(point: list, angle: float):
    # val = math.sqrt(math.pow(point[0], 2) + math.pow(point[1], 2))
    # x_new = val if (point[1] >= 0 and point[0] >= 0) else -val
    # y_new = 0
    x_new = point[0] * math.cos(math.radians(angle)) + point[1] * math.sin(math.radians(angle))
    y_new = point[1] * math.cos(math.radians(angle)) - point[0] * math.sin(math.radians(angle))
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

def plot_central(data, angles_shift, angles, coefs, is_significant, labels, draw_clusters: bool = False):
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

    c_min, c_max = coefs.min(), 0

    if coefs.shape[0] > 1:
        c_max = coefs.max()
    
    elif c_min > 0:
        c_max = c_min
        c_min = 0
    
    else:
        c_max = 0

    # coefs_scaled = (((coefs - c_min) / (c_max - c_min)) * (radius - (- radius))) - radius 
    coefs_scaled = coefs * (radius / max(abs(c_max), abs(c_min)))
    
    for a_s in list(range(0, 360, angles_shift)):
        a_s = math.radians(a_s)
        x_add, y_add = math.cos(a_s) * radius, math.sin(a_s) * radius
        plt.plot((x_center, x_center+x_add), (y_center, y_center+y_add), c='gray', linestyle="--", alpha=0.7, linewidth=1)

    arrows = []
    for a, c, s in zip(angles, coefs_scaled, is_significant):
        a = math.radians(a)
        x_add, y_add = math.cos(a) * radius, math.sin(a) * radius

        # plot contributions
        ind = abs(c).argsort(axis=None)[::-1]
        x_add_coefs, y_add_coefs = math.cos(a) * c[ind], math.sin(a) * c[ind]
        
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

def get_plot(plt, data, draw_hulls: bool = True):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    alpha = 0.4

    if draw_hulls:
        draw_clusters(ax, data, alpha)
    sc = plt.scatter(data["emb1"], data["emb2"], c=data["label"], cmap="viridis", 
                     vmin=data["label"].min(), vmax=data["label"].max(), s=3, alpha=alpha)
    fig.colorbar(sc)
    return fig, ax

def finish_plot():
    plt.xlabel(f"UMAP 1")
    plt.ylabel(f"UMAP 2")
    plt.title("Malignant cells")
    plt.tight_layout()
    plt.show()

def plot_small(fig, ax, data, angles_shift, angles, coefs, is_significant, labels):
    colors = list(mcolors.TABLEAU_COLORS.keys())
    x_center, y_center = get_center(data)
    x_min, x_max, y_min, y_max = get_min_max(data)
    radius = abs(min(x_max-x_min, y_max-y_min)) * 0.5
    circle = patches.Circle((x_center, y_center), radius=radius, edgecolor='gray', facecolor='aliceblue', linewidth=1, alpha=0.4) 
    ax.add_patch(circle)
    ax.axis('equal')

    # coefs_scaled = -radius +  (coefs - coefs.min()) * ((2 * radius) / (coefs.max()-coefs.min()))
    c_min, c_max = coefs.min(), 0

    if coefs.shape[0] > 1:
        c_max = coefs.max()
    
    elif c_min > 0:
        c_max = c_min
        c_min = 0
    
    else:
        c_max = 0
    coefs_scaled = coefs * (radius / max(abs(c_max), abs(c_min)))

    for a_s in list(range(0, 360, angles_shift)):
        a_s = math.radians(a_s)
        x_add, y_add = math.cos(a_s) * radius, math.sin(a_s) * radius
        plt.plot((x_center, x_center+x_add), (y_center, y_center+y_add), c='gray', linestyle="--", alpha=0.7, linewidth=1)

    
    arrows = []
    for a, c, s in zip(angles, coefs_scaled, is_significant):
        a = math.radians(a)
        x_add, y_add = math.cos(a) * radius, math.sin(a) * radius

        # plot contributions
        ind = abs(c).argsort(axis=None)[::-1]
        x_add_coefs, y_add_coefs = math.cos(a) * c[ind], math.sin(a) * c[ind]
        
        for is_s, x_c, y_c, i in zip(s[ind], x_add_coefs, y_add_coefs, ind):
            if is_s:
                col = colors[i]
                lbl = labels[i]
                arrows.append(plt.arrow(x_center, y_center, x_c, y_c, width=0.05, color=col, label=lbl))

    return arrows

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

    print(data["cluster"].unique())
    print(data["cluster"].nunique())

    # actual data
    angles_shift = 1
    angles = list(range(0, 180, angles_shift))  # FIXME fix angle
    projections = [project_line(data, angle, point_a=[0, 0], point_b=get_slope_from_angle(angle)) for angle in angles]
    projections = np.array(projections).T
    
    plot_central_clock = True
    if plot_central_clock:
        coefs, _, is_significant = get_importance(new_data.to_numpy(), projections)
        plot_central(data, 45, angles, coefs, is_significant, obs)

    plot_small_clock = True
    if plot_small_clock:
        dist_clusters = data["cluster"].unique()
        dist_clusters.sort()
        dist_clusters = dist_clusters[1:]
        arrows_all = []

        fig, ax = get_plot(plt, data, True)

        for cl in dist_clusters:
            ind = (data["cluster"] == cl).values.reshape((data.shape[0], 1))

            data_cl = data[ind]
            new_data_cl = new_data[ind]
            projections_cl = projections[ind[:, 0], :]

            coefs, _, is_significant = get_importance(new_data_cl.to_numpy(), projections_cl)
            arrows = plot_small(fig, ax, data_cl, 45, angles, coefs, is_significant, obs)
            arrows_all.extend(arrows)
            
        plt.legend(arrows_all, obs)
        
        finish_plot()


try_scd()
