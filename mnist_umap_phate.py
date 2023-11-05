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
from sklearn.cluster import DBSCAN
import matplotlib.patheffects as pe


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

def dbscan_low_dim(standard_embedding):
    labels = DBSCAN(min_samples=10).fit_predict(standard_embedding)
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
                ax.plot(a, b, '--', c="gray", alpha=alpha)
                ax.fill(a, b, alpha=alpha, c="gray")

def read_data(path):
    return sp.read_h5ad(path)  

def project_line(data, angle: float, point_a: list=[0, 0], point_b: list=[1, 1]):
    line = Line.from_points(point_a=point_a, point_b=point_b)

    projected_points = []
    for i in range(data.shape[0]):
        point = (data["emb1"][i], data["emb2"][i])
        new_point = line.project_point(point)
        new_point = rotate(new_point, angle)
        projected_points.append(new_point[0])
    
    return np.array(projected_points)

def get_plot(plt, data, draw_hulls: bool = True):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    alpha = 0.4

    if draw_hulls:
        draw_clusters(ax, data, alpha)
    sc = plt.scatter(data["emb1"], data["emb2"], c=data["label"], cmap="viridis", 
                     vmin=data["label"].min(), vmax=data["label"].max(), s=3, alpha=alpha)
    fig.colorbar(sc)
    return fig, ax

def get_slope_from_angle(angle: float):
   return [math.cos(math.radians(angle)) * 100, math.sin(math.radians(angle)) * 100]

def rotate(point: list, angle: float):
    x_new = point[0] * math.cos(math.radians(angle)) + point[1] * math.sin(math.radians(angle))
    y_new = point[1] * math.cos(math.radians(angle)) - point[0] * math.sin(math.radians(angle))
    return[x_new, y_new]

def get_importance(X, y, significance: float = 0.05, univar: bool = False):
    # lm = LinearRegression()
    # lm.fit(X, y)
    # print(lm.coef_)

    coefs, pvals, is_significant = [], [], []
    if univar:
        for i in range(y.shape[1]):
            coefs_a, pvals_a, is_significant_a = [], [], []
            for j in range(X.shape[1]):
                lm = sm.OLS(y[:, i], X[:, j]).fit()
                coefs_a.append(lm.params[0])
                pvals_a.append(lm.pvalues[0])
                is_significant_a.append(lm.pvalues[0] <= significance)

            coefs.append(np.array(coefs_a))
            pvals.append(pvals_a)
            is_significant.append(is_significant_a)

    else:
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

def plot_all(data, col_label,label):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    colors = list(mcolors.TABLEAU_COLORS.keys())

    sc = plt.scatter(data["emb1"], data["emb2"], c=col_label, cmap="viridis", 
                     vmin=col_label.min(), vmax=col_label.max(), s=3)
    fig.colorbar(sc)
    plt.xlabel(f"UMAP 1")
    plt.ylabel(f"UMAP 2")
    plt.title(f"Malignant cells + {label}")
    plt.tight_layout()
    plt.show()

def plot_central(data, angles_shift, angles, coefs, is_significant, labels, draw_clusters: bool = False, windrose: bool = True, biggest_arrow: bool = True):
    fig, ax = plt.subplots(1, figsize=(12, 8))
    colors = list(mcolors.TABLEAU_COLORS.keys())
    alpha = 0.2

    sc = plt.scatter(data["emb1"], data["emb2"], c=data["label"], cmap="viridis", 
                     vmin=data["label"].min(), vmax=data["label"].max(), s=3, alpha=alpha, zorder = 0)
    fig.colorbar(sc)

    x_center, y_center = get_center(data)
    # x_min, x_max, y_min, y_max = get_min_max(data)
    # radius = abs(min(x_max-x_min, y_max-y_min)) * 0.5
    radius = np.abs(coefs).max()

    c_min, c_max = coefs.min(), 0
    if coefs.shape[0] > 1:
        c_max = coefs.max()
    
    elif c_min > 0:
        c_max = c_min
        c_min = 0
    
    else:
        c_max = 0

     # Add circles
    annotate = 0.3
    num_circles = math.floor(radius / annotate) + 1

    coefs_scaled = coefs * (radius / max(abs(c_max), abs(c_min)))
    
    for a_s in list(range(0, 360, angles_shift)):
        a_s = math.radians(a_s)
        x_add, y_add = math.cos(a_s) * (num_circles-1) * annotate, math.sin(a_s) * (num_circles-1) * annotate
        plt.plot((x_center, x_center+x_add), (y_center, y_center+y_add), c='gray', linestyle="--", alpha=0.7, linewidth=1, zorder=10)
    
    if not windrose:
        arrows = []

        if not biggest_arrow:
            for a, c, s in zip(angles, coefs_scaled, is_significant):
                a = math.radians(a)

                # plot contributions
                ind = abs(c).argsort(axis=None)[::-1]
                x_add_coefs, y_add_coefs = math.cos(a) * c[ind], math.sin(a) * c[ind]
                
                for is_s, x_c, y_c, i in zip(s[ind], x_add_coefs, y_add_coefs, ind):
                    if is_s:
                        col = colors[i]
                        lbl = labels[i]
                        arrows.append(plt.arrow(x_center, y_center, x_c, y_c, width=0.01, color=col, label=lbl, zorder=15))

            plt.legend(arrows, labels)
        
        else:
            arrows_ind = np.argmax(np.abs(coefs_scaled), axis=0)
            
            for i in range(coefs_scaled.shape[1]):
                a = math.radians(angles[arrows_ind[i]])

                # plot contributions
                x_c, y_c = math.cos(a) * coefs_scaled[arrows_ind[i], i], math.sin(a) * coefs_scaled[arrows_ind[i], i]
                
                col = colors[i]
                lbl = labels[i]
                arrows.append(plt.arrow(x_center, y_center, x_c, y_c, width=0.01, color=col, label=lbl, zorder=15))

            plt.legend(arrows, labels)

    else:
        points_x, points_y = [], []
        for a, c in zip(angles, coefs_scaled):
            a = math.radians(a)
            x_add, y_add = math.cos(a) * radius, math.sin(a) * radius

            x_add_coefs, y_add_coefs = math.cos(a) * c, math.sin(a) * c

            points_x.append(x_center + x_add_coefs)
            points_y.append(y_center + y_add_coefs)
        
        points_x = np.array(points_x)
        points_y = np.array(points_y)

        order_ix = np.argsort((abs(points_x).mean(axis=0) + abs(points_y).mean(axis=0)) / 2)[::-1]

        for i in order_ix:
            ax.plot(points_x[:, i], points_y[:, i], '-', c=colors[i], alpha=1, label=labels[i])
            # ax.fill(points_x[:, i], points_y[:, i], alpha=0.7, c=colors[i])

        plt.legend()
    
    for i in list(range(1, num_circles))[::-1]:
        radius_circle = round(i * annotate, 1)

        if i == num_circles-1:
            circle = patches.Circle((x_center, y_center), radius=radius_circle, edgecolor='gray', linewidth=1, linestyle="--", facecolor=(0.941, 0.973, 1.0, 0.5), fill=True, zorder=10) 
        else:
            circle = patches.Circle((x_center, y_center), radius=radius_circle, edgecolor='gray', linewidth=1, linestyle="--", fill=False, zorder=12) 

        ax.add_patch(circle)
        ax.axis('equal')

        circle_an = math.radians(45)
        x_ann, y_ann = x_center + math.cos(circle_an) * radius_circle, y_center + math.sin(circle_an) * radius_circle
        ax.annotate(str(round(radius_circle / (radius / max(abs(c_max), abs(c_min))), 2)), xy=(x_ann, y_ann), ha="right", color="gray",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=20)

    plt.xlabel(f"UMAP 1")
    plt.ylabel(f"UMAP 2")
    plt.title("Malignant cells")
    plt.tight_layout()
    plt.show()

def finish_plot():
    plt.xlabel(f"UMAP 1")
    plt.ylabel(f"UMAP 2")
    plt.title("Malignant cells")
    plt.tight_layout()
    plt.show()

def plot_small(fig, ax, data, angles_shift, angles, coefs, is_significant, labels, biggest_arrow: bool = True, scale_circle: float = 1.0, annotate: float = 0.2):
    colors = list(mcolors.TABLEAU_COLORS.keys())
    x_center, y_center = get_center(data)

    x_min, x_max, y_min, y_max = get_min_max(data)
    # scale_circle = abs(max(x_max-x_min, y_max-y_min)) * 0.2
    radius = np.abs(coefs).max() * scale_circle

    c_min, c_max = coefs.min(), 0

    if coefs.shape[0] > 1:
        c_max = coefs.max()
    
    elif c_min > 0:
        c_max = c_min
        c_min = 0
    
    else:
        c_max = 0
    
    # Add circles
    # annotate = 0.2
    num_circles = math.floor(radius / annotate) + 1

    coefs_scaled = coefs * (radius / max(abs(c_max), abs(c_min)))
    
    for a_s in list(range(0, 360, angles_shift)):
        a_s = math.radians(a_s)
        x_add, y_add = math.cos(a_s) * (num_circles-1) * annotate, math.sin(a_s) * (num_circles-1) * annotate
        plt.plot((x_center, x_center+x_add), (y_center, y_center+y_add), c='gray', linestyle="--", alpha=0.4, linewidth=1, zorder=10)
    
    arrows = []
    if not biggest_arrow:
        for a, c, s in zip(angles, coefs_scaled, is_significant):
            a = math.radians(a)

            # plot contributions
            ind = abs(c).argsort(axis=None)[::-1]
            x_add_coefs, y_add_coefs = math.cos(a) * c[ind], math.sin(a) * c[ind]
            
            for is_s, x_c, y_c, i in zip(s[ind], x_add_coefs, y_add_coefs, ind):
                if is_s:
                    col = colors[i]
                    lbl = labels[i]
                    arrows.append(plt.arrow(x_center, y_center, x_c, y_c, width=0.01, color=col, label=lbl, zorder=15))

        plt.legend(arrows, labels)
    
    else:
        arrows_ind = np.argmax(np.abs(coefs_scaled), axis=0)
        
        for i in range(coefs_scaled.shape[1]):
            a = math.radians(angles[arrows_ind[i]])

            # plot contributions
            x_c, y_c = math.cos(a) * coefs_scaled[arrows_ind[i], i], math.sin(a) * coefs_scaled[arrows_ind[i], i]
            
            col = colors[i]
            lbl = labels[i]
            arrows.append(plt.arrow(x_center, y_center, x_c, y_c, width=0.01, color=col, label=lbl, zorder=15))

        plt.legend(arrows, labels)

    print(list(range(1, num_circles))[::-1])
    for i in list(range(1, num_circles))[::-1]:
        radius_circle = round(i * annotate, 1)

        if i == num_circles-1:
            circle = patches.Circle((x_center, y_center), radius=radius_circle, edgecolor='gray', linewidth=1, linestyle="--", facecolor=(0.941, 0.973, 1.0, 0.5), fill=True, zorder=10) 
        else:
            circle = patches.Circle((x_center, y_center), radius=radius_circle, edgecolor='gray', linewidth=1, linestyle="--", fill=False, zorder=12) 

        ax.add_patch(circle)
        ax.axis('equal')

        circle_an = math.radians(45)
        x_ann, y_ann = x_center + math.cos(circle_an) * radius_circle, y_center + math.sin(circle_an) * radius_circle
        ax.annotate(str(round(radius_circle / (radius / max(abs(c_max), abs(c_min))), 2)), xy=(x_ann, y_ann), ha="right", color="gray",
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=20)


    return arrows

def try_scd():
    # read data
    file_name = 'data/neftel_malignant.h5ad'
    X = read_data(file_name)

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
    ac, opc = X_new.X[:, 2], X_new.X[:, 3]
    mes_max = np.max(mes, axis=0)
    npc_max = np.max(npc, axis=0)
    res_vect = np.stack((ac, opc, mes_max, npc_max))
    res_labels = np.max(res_vect, axis=0)

    # make data with labels and clusters
    data = pd.DataFrame(standard_embedding, columns=["emb1", "emb2"])

    for col in data.columns:
        data[col] = (data[col] - data[col].mean()) / data[col].std()

    for col in new_data.columns:
        new_data[col] = (new_data[col] - new_data[col].mean()) / new_data[col].std()


    data["cluster"] = cluster_labels
    data["label"] = res_labels

    # for col, label in zip([X_new.X[:, 0], X_new.X[:, 1], X_new.X[:, 2], X_new.X[:, 3], X_new.X[:, 4], X_new.X[:, 5]], obs):
    #     plot_all(data, col, label)

    # actual data
    angles_shift = 5
    angles = list(range(0, 180, angles_shift))  # FIXME fix angle
    projections = [project_line(data, angle, point_a=[0, 0], point_b=get_slope_from_angle(angle)) for angle in angles]
    projections = np.array(projections).T
    
    plot_central_clock = False
    if plot_central_clock:
        coefs, _, is_significant = get_importance(new_data.to_numpy(), projections, univar=True)
        coefs = (coefs - coefs.mean()) / coefs.std()
        plot_central(data, 45, angles, coefs, is_significant, obs, windrose=False)

    plot_small_clock = True
    if plot_small_clock:
        dist_clusters = data["cluster"].unique()
        dist_clusters.sort()
        dist_clusters = dist_clusters[1:] # FIXME
        arrows_all = []

        fig, ax = get_plot(plt, data, True)

        scale_circle = [1.0, 0.25, 0.3]
        annotate = [0.2, 0.1, 0.1]
        for a, scale, cl in zip(annotate, scale_circle, dist_clusters):
            ind = (data["cluster"] == cl).values.reshape((data.shape[0], 1))

            data_cl = data[ind]
            new_data_cl = new_data[ind]
            projections_cl = projections[ind[:, 0], :]

            coefs, _, is_significant = get_importance(new_data_cl.to_numpy(), projections_cl, univar=True)
            coefs = (coefs - coefs.mean()) / coefs.std()
            arrows = plot_small(fig, ax, data_cl, 45, angles, coefs, is_significant, obs, scale_circle=scale, annotate=a)
            arrows_all.extend(arrows)
            
        plt.legend(arrows_all, obs)
        
        finish_plot()


try_scd()
