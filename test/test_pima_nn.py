from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from src.nonlinear_clock.plot import NonLinearClock
import umap
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p




def read_data(path):
    return pd.read_csv(path, header=0)


def setup_pima_data(method="tsne", drop_labels=True, file: str=""):
    file_name = "/Users/olga_ovcharenko/Documents/ETH/FS23/ResearchProject/non_lin_visualization/data/diabetes.csv"
    X = read_data(file_name)
    X.rename(columns={"DiabetesPedigreeFunction": "Pedigree"}, inplace=True)
    X = X.dropna()

    labels = X["Outcome"]
    if drop_labels:
        X.drop(columns=["Outcome"], inplace=True)
    obs = list(X.columns)

    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()

    # compute umap
    if method == "umap":
        file_name2 = file
        Y = read_data(file_name2)
        Y = Y.dropna()

        reducer = umap.UMAP(min_dist=0.2, n_neighbors=30, random_state=42)
        standard_embedding = reducer.fit_transform(Y)

    elif method == "tsne":
        raise NotImplementedError()

    elif method == "phate":
        raise NotImplementedError()
    
    # get clusters
    clusters = HDBSCAN(min_samples=12).fit_predict(X)
    # clusters = KMeans(n_clusters=3, n_init="auto", max_iter=1000).fit_predict(X)

    return X, obs, standard_embedding, labels, clusters

def print_pima_all(file, dataset_i):
    X_new, obs, standard_embedding, labels, clusters = setup_pima_data(method="umap", drop_labels=False, file=file)
    dpi = 1000
    # fig_size = (2.375, 2.375)
    fig_size = (3.2325, 2.9)
    fig, axi = plt.subplots(
        3,
        3,
        num=None,
        figsize=fig_size,
        dpi=dpi,
        facecolor="w",
        edgecolor="k",
    )
    for i, o in enumerate(obs):
        im = axi[i % 3, i // 3].scatter(
            standard_embedding[:, 0],
            standard_embedding[:, 1],
            marker=".",
            s=1.3,
            c=X_new[o],
            cmap="Spectral",
        )
        axi[i % 3, i // 3].set_yticks([])
        axi[i % 3, i // 3].set_xticks([])
        axi[i % 3, i // 3].yaxis.set_label_coords(x=-0.01, y=0.5)
        axi[i % 3, i // 3].xaxis.set_label_coords(x=0.5, y=-0.02)
        axi[i % 3, i // 3].set_title(o, size=8, pad=-14)

    axi[1, 0].set_ylabel("UMAP2", size=8)
    axi[2, 1].set_xlabel("UMAP1", size=8)
    
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.95,
        bottom=0.05,  # wspace=0.21, hspace=0.33
    )
    cbar = fig.colorbar(im, ax=axi.ravel().tolist(), pad=0.1)
    cbar.ax.tick_params(labelsize=7) 
    # for ax in axi:
    #     for a in ax:
    #         a.axis('off') 
    plt.savefig(f"plots/paper/pima_network/plot_pimaAll_nn_{dataset_i}.pdf")


def test_between_all_1():
    X_new, obs, standard_embedding, labels, clusters = setup_pima_data(method="umap", file="/Users/olga_ovcharenko/Documents/ETH/FS23/ResearchProject/non_lin_visualization/data/latent_space_1.csv")

    fig, ax = plt.subplots(1, figsize=(3.2325, 3.2325)) #(2.375, 2.375)
    plt.tight_layout()

    sc = ax.scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.3)
    colors = [
        'tab:pink', 'tab:green', 'tab:blue', 'tab:olive', 'tab:orange',
        'tab:purple', 'tab:cyan', 'tab:red', 'tab:brown']
    
    plot_inst = NonLinearClock(
        X_new, obs, standard_embedding, labels, 
        method="UMAP", cluster_labels=clusters, color_scheme=colors)
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=False,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=ax,
        scale_circle=2,
        move_circle=[0, 0],
        annotate=0.6,
        arrow_width=0.1,
    )

    hatches = [plt.plot([],marker="", ls="")[0]]*4 + arrows[0:3] + [sc.legend_elements()[0][0]] + arrows[3:6] + [sc.legend_elements()[0][1]] + arrows[6:] + [plt.plot([],marker="", ls="")[0]]*2
    labels = ["Factors:", " ", "", "Labels: "] + arrow_labels[0:3] + ["No diabetes"] + arrow_labels[3:6] + ["Diabetes"] + arrow_labels[6:] + [" ", " "]
    leg = ax.legend(
        hatches,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.09),
        fontsize=7,
        ncol=4,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.5,
        handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),},
    )

    for vpack in leg._legend_handle_box.get_children()[:1]:
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("UMAP2", size=8)
    ax.set_xlabel("UMAP1", size=8)
    ax.set_title("First hidden layer", size=8)
    ax.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax.xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.05,
        right=0.96,
        top=0.73,
        bottom=0.05,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/pima_network/pima_global_nn_1.pdf")

def test_between_all_2():
    X_new, obs, standard_embedding, labels, clusters = setup_pima_data(method="umap", file = "/Users/olga_ovcharenko/Documents/ETH/FS23/ResearchProject/non_lin_visualization/data/latent_space_2.csv")

    fig, ax = plt.subplots(1, figsize=(3.2325, 3.2325)) #(2.375, 2.375)
    plt.tight_layout()

    sc = ax.scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.3)

    colors = [
        'tab:pink', 'tab:green', 'tab:blue', 'tab:olive', 'tab:orange',
        'tab:purple', 'tab:cyan', 'tab:red', 'tab:brown']
    
    plot_inst = NonLinearClock(
        X_new, obs, standard_embedding, labels, method="UMAP", 
        cluster_labels=clusters, color_scheme=colors)
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=False,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=ax,
        scale_circle=3,
        move_circle=[0, 0],
        annotate=0.9,
        arrow_width=0.1
    )

    hatches = [plt.plot([],marker="", ls="")[0]]*3 + arrows[0:2] + [sc.legend_elements()[0][0]] + arrows[2:] + [sc.legend_elements()[0][1]]
    labels = ["Factors:", " ", "Labels: "] + arrow_labels[0:2] + ["No diabetes"] + arrow_labels[2:] + ["Diabetes"]
    leg = ax.legend(
        hatches, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.09),
        fontsize=7,
        ncol=3,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.5,
        handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),},
    )

    for vpack in leg._legend_handle_box.get_children()[:1]:
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("UMAP2", size=8)
    ax.set_xlabel("UMAP1", size=8)
    ax.set_title("Second hidden layer", size=8)
    ax.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax.xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.05,
        right=0.99,
        top=0.77,
        bottom=0.05,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/pima_network/pima_global_nn_2.pdf")


# print_pima_all("/Users/olga_ovcharenko/Documents/ETH/FS23/ResearchProject/non_lin_visualization/data/latent_space_1.csv", 1)
test_between_all_1()
# print_pima_all("/Users/olga_ovcharenko/Documents/ETH/FS23/ResearchProject/non_lin_visualization/data/latent_space_2.csv", 2)
test_between_all_2()