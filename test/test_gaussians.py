import sys
import matplotlib
sys.path.append('src/')

import numpy as np
import scanpy as sp
from src.nonlinear_clock.plot import NonLinearClock
import scanpy.external as sce
import phate
import pandas as pd
import scipy.stats as ss
import matplotlib.pyplot as plt 

np.random.seed(42)

def create_gm(mean_stds: np.array):
    xs, ys = [], []
    for ix, val in enumerate(mean_stds):
        x = np.random.multivariate_normal(val[0], val[1], val[2])
        xs.extend(x)
        ys.extend(np.full((val[2],), ix))

    xs, ys = np.array(xs), np.array(ys)

    # plt.scatter(xs[:, 0], xs[:, 1], c=ys, cmap=matplotlib.colors.ListedColormap(["red", "blue", "pink", "green"]))
    # plt.axis('equal')
    # plt.grid()
    # plt.show()
    return xs, ys

def setup_data(method = "umap"):
    mean_stds_nrow = [
        [[0, 0], [[6, -3], [-3, 3.5]], 200],
        [[5, 25], [[11, 0], [0, 4]], 300],
        [[30, -5], [[1, 0], [0, 1]], 500],
        [[35, 40], [[2.5, 6], [6, 75]], 250],
    ]

    X, labels = create_gm(mean_stds=mean_stds_nrow)
    
    new_data = pd.DataFrame(X)
    
    for col in new_data.columns:
        new_data[col] = (new_data[col] - \
           new_data[col].mean()) / new_data[col].std()
    
    X_new = sp.AnnData(new_data)

    # compute umap
    sp.pp.neighbors(X_new)

    if new_data.shape[1] <= 2:
        standard_embedding = X

    elif method == "umap":
        sp.tl.umap(X_new, min_dist=0.01, spread=0)

        # get clusters
        standard_embedding = X_new.obsm['X_umap']

    elif method == "tsne":
        sp.tl.tsne(X_new)

        # get clusters
        standard_embedding = X_new.obsm['X_tsne']

    elif method == "phate":
        sce.tl.phate(X_new, k=5, a=20, t=150)
        standard_embedding = X_new.obsm['X_phate']
    
    else:
        raise Exception("Low dimensional data or dimensionality reduction method is not specified.")


    return new_data, list(new_data.columns), standard_embedding, labels


def test_umap():
    X_new, obs, standard_embedding, labels = setup_data()

    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, "UMAP")
    plot_inst.plot_clocks(plot_title="Malignant cells", 
                          plot_big_clock=True, 
                          plot_small_clock=True,
                          standartize_data=True,
                          biggest_arrow_method=True,
                          univar_importance=True,
                          save_path_big="plots/new/big_1.png",
                          save_path_small="plots/new/small_1.png"
                          )

def test_between_arrow():
    X_new, obs, standard_embedding, labels = setup_data()

    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, "UMAP")
    plot_inst.plot_clocks(plot_title="Genes", 
                          plot_big_clock=True, 
                          plot_small_clock=True,
                          plot_between_cluster=True,
                          standartize_data=True,
                          biggest_arrow_method=True,
                          univar_importance=True,
                          save_path_big="plots/new/big_1.png",
                          save_path_small="plots/new/small_1.png",
                          save_path_between="plots/new/between_1.png"
                          )

def test_between_umap():
    X_new, obs, standard_embedding, labels = setup_data(method="umap")

    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, "UMAP")
    plot_inst.plot_clocks(plot_title="Genes", 
                          plot_big_clock=True, 
                          plot_small_clock=True,
                          plot_between_cluster=True,
                          standartize_data=True,
                          standartize_coef=False,
                          biggest_arrow_method=False,
                          univar_importance=True,
                          save_path_big="plots/new/gene_big_1_circle.png",
                          save_path_small="plots/new/gene_small_1_circle.png",
                          save_path_between="plots/new/gene_between_1_circle.png"
                          )


def test_between_tsne():
    X_new, obs, standard_embedding, labels = setup_data(method="tsne")

    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, "tsne")
    plot_inst.plot_clocks(plot_title="Genes", 
                          plot_big_clock=True, 
                          plot_small_clock=True,
                          plot_between_cluster=True,
                          standartize_data=True,
                          standartize_coef=False,
                          biggest_arrow_method=False,
                          univar_importance=True,
                          save_path_big="plots/new/gene_big_1_circle_tsne.png",
                          save_path_small="plots/new/gene_small_1_circle_tsne.png",
                          save_path_between="plots/new/gene_between_1_circle_tsne.png"
                          )

def test_between_phate():
    X_new, obs, standard_embedding, labels = setup_data(method="phate")

    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, "phate")
    plot_inst.plot_clocks(plot_title="Genes", 
                          plot_big_clock=True, 
                          plot_small_clock=True,
                          plot_between_cluster=True,
                          standartize_data=True,
                          standartize_coef=False,
                          biggest_arrow_method=False,
                          univar_importance=True,
                          save_path_big="plots/new/gene_big_1_circle_phate.png",
                          save_path_small="plots/new/gene_small_1_circle_phate.png",
                          save_path_between="plots/new/gene_between_1_circle_phate.png"
                          )

test_between_umap()
