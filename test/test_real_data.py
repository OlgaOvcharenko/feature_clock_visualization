import numpy as np
import scanpy as sp
from src.nonlinear_clock.plot import NonLinearClock
import scanpy.external as sce
import phate
import pandas as pd

def read_data(path):
    return pd.read_csv(path, sep=";", header=0)

def setup_data(method = "umap"):
    file_name = '/Users/olga_ovcharenko/Documents/ETH/FS23/ResearchProject/non_lin_visualization/data/E-GEOD-46817-query-results.csv'
    X = read_data(file_name)

    obs = [ 
           "A2058",   "A375",   "C32",  "Malme3M",  "SKMEL28",  "SKMEL5",  "WM2664"
           ]
    
    new_data = X[obs].dropna()
    for col in new_data.columns:
        new_data[col] = (new_data[col] - \
           new_data[col].mean()) / new_data[col].std()


    X_new = sp.AnnData(new_data)

    # compute umap
    sp.pp.neighbors(X_new)
    if method == "umap":
        sp.tl.umap(X_new, min_dist=10, spread=10)

        # get clusters
        standard_embedding = X_new.obsm['X_umap']

    elif method == "tsne":
        sp.tl.tsne(X_new)

        # get clusters
        standard_embedding = X_new.obsm['X_tsne']

    elif method == "phate":
        sce.tl.phate(X_new, k=5, a=20, t=150)
        standard_embedding = X_new.obsm['X_phate']

    # get labels
    mes = np.stack((X_new.X[:, 0], X_new.X[:,1]))
    npc = np.stack((X_new.X[:, 4], X_new.X[:, 5]))
    ac, opc = X_new.X[:, 2], X_new.X[:, 3]
    mes_max = np.max(mes, axis=0)
    npc_max = np.max(npc, axis=0)
    res_vect = np.stack((ac, opc, mes_max, npc_max))
    res_labels = np.max(res_vect, axis=0)

    return new_data, obs, standard_embedding, res_labels


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

# test_umap()

test_between_umap()
# test_between_tsne()
# test_between_phate()
