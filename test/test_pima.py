from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from src.plot import NonLinearClock
import umap
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans


def read_data(path):
    return pd.read_csv(path, header=0)


def setup_pima_data(method="tsne", drop_labels=True):
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
        reducer = umap.UMAP(min_dist=0.5, n_neighbors=10, random_state=42)
        standard_embedding = reducer.fit_transform(X)

    elif method == "tsne":
        raise NotImplementedError()

    elif method == "phate":
        raise NotImplementedError()
    
    # get clusters
    clusters = HDBSCAN(min_samples=12).fit_predict(X)
    # clusters = KMeans(n_clusters=3, n_init="auto", max_iter=1000).fit_predict(X)

    return X, obs, standard_embedding, labels, clusters

def print_pima_all():
    X_new, obs, standard_embedding, labels, clusters = setup_pima_data(method="umap", drop_labels=False)
    dpi = 1000
    fig_size = (3.33, 3.33)
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
        axi[i % 3, i // 3].scatter(
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

    plt.savefig("plots/paper/pima/plot_pimaAll.pdf")


def test_between_all():
    X_new, obs, standard_embedding, labels, clusters = setup_pima_data(method="umap")

    fig, ax = plt.subplots(1, figsize=(3.33, 3.33))
    plt.tight_layout()
    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, method="UMAP", cluster_labels=clusters)
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=ax,
        scale_circle=1,
        move_circle=[0, 0],
        annotate=0.6,
        arrow_width=1.5
    )
    ax.legend(
        arrows,
        arrow_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.07),
        fontsize=7,
        ncol=4,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.5,
    )

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("UMAP2", size=8)
    ax.set_xlabel("UMAP1", size=8)
    ax.set_title("Diabetis", size=8)
    ax.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax.xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.79,
        bottom=0.05,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/pima/pima_global.pdf")

    # Local
    fig, ax = plt.subplots(1, figsize=(3.33, 3.33))
    arrows, arrow_labels = plot_inst.plot_local_clocks(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=ax,
        scale_circles=[3, 0.5, 0.5],
        move_circles=[[0, 0], [0, 0], [0, 0]],
        annotates=[0.5, 0.5, 0.5],
        arrow_width=0.03,
    )
    ax.legend(
        arrows,
        arrow_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.07),
        fontsize=7,
        ncol=4,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.5,
    )

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("UMAP2", size=8)
    ax.set_xlabel("UMAP1", size=8)
    ax.set_title("Diabetis", size=8)
    ax.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax.xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.79,
        bottom=0.05,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/pima/pima_local.pdf")

    # # Between
    # fig, ax = plt.subplots(1, figsize=(3.33, 3.33))
    # arrows, arrow_labels = plot_inst.plot_between_clock(
    #     standartize_data=True,
    #     standartize_coef=True,
    #     univar_importance=True,
    #     ax=ax,
    #     scale_circles=[1, 1.5],
    #     move_circles=[[0, 0], [0.7, 0]],
    #     annotates=[0.3, 0.2],
    #     arrow_width=0.03,
    # )
    # ax.legend(
    #     arrows,
    #     arrow_labels,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, 1.07),
    #     fontsize=7,
    #     ncol=3,
    #     markerscale=0.6,
    #     handlelength=1.5,
    #     columnspacing=0.8,
    #     handletextpad=0.5,
    # )

    # ax.set_yticks([])
    # ax.set_xticks([])
    # ax.set_ylabel("UMAP2", size=8)
    # ax.set_xlabel("UMAP1", size=8)
    # ax.set_title("Malignant cells", size=8)
    # ax.yaxis.set_label_coords(x=-0.01, y=0.5)
    # ax.xaxis.set_label_coords(x=0.5, y=-0.02)
    # plt.subplots_adjust(
    #     left=0.05,
    #     right=0.95,
    #     top=0.79,
    #     bottom=0.05,  # wspace=0.21, hspace=0.33
    # )
    # plt.savefig("plots/paper/pima/pima_between.pdf")

# print_pima_all()
test_between_all()