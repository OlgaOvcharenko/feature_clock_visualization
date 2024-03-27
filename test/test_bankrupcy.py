from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from src.nonlinear_clock.plot import NonLinearClock
import umap
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans


def read_data(path):
    return pd.read_csv(path, header=0)


def setup_bankrupcy_data(method="tsne", drop_labels=True):
    file_name = "/Users/olga_ovcharenko/Documents/ETH/FS23/ResearchProject/non_lin_visualization/data/bankrupcy.csv"
    X = read_data(file_name)

    names = dict()
    for name in list(X.columns):
        names[name] = str.strip(name)
    X.rename(columns=names, inplace=True)
    X = X.dropna()
    
    labels = X["Bankrupt"]
    if drop_labels:
        X.drop(columns=["Bankrupt", "Net Income Flag"], inplace=True)
    obs = list(X.columns)
    
    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()
    
    # compute umap
    if method == "umap":
        reducer = umap.UMAP(min_dist=0.2, n_neighbors=30, random_state=42)
        if not drop_labels:
            K = X.drop(columns=["Bankrupt"], inplace=False)
            standard_embedding = reducer.fit_transform(K)
        else:
            standard_embedding = reducer.fit_transform(X)

    elif method == "tsne":
        raise NotImplementedError()

    elif method == "phate":
        raise NotImplementedError()
    
    # get clusters
    clusters = HDBSCAN(min_samples=12).fit_predict(X)
    # clusters = KMeans(n_clusters=3, n_init="auto", max_iter=1000).fit_predict(X)

    return X, obs, standard_embedding, labels, clusters

def print_bankrupcy_all():
    X_new, obs, standard_embedding, labels, clusters = setup_bankrupcy_data(method="umap", drop_labels=False)
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
        right=1,
        top=0.95,
        bottom=0.05,  # wspace=0.21, hspace=0.33
    )
    cbar = fig.colorbar(im, ax=axi.ravel().tolist(), pad=0.1)
    cbar.ax.tick_params(labelsize=7) 
    # for ax in axi:
    #     for a in ax:
    #         a.axis('off') 
    plt.savefig("plots/paper/bankrupcy/plot_bankrupcyAll.pdf")


def test_between_all():
    X_new, obs, standard_embedding, labels, clusters = setup_bankrupcy_data(method="umap")

    fig, ax = plt.subplots(1, figsize=(3.33, 2.8))
    plt.tight_layout()
    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, method="UMAP", cluster_labels=clusters)
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=False,
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
    plt.savefig("plots/paper/bankrupcy/bankrupcy_global.pdf")

    # Local
    fig, ax = plt.subplots(1, figsize=(2.375, 2.375))
    arrows, arrow_labels = plot_inst.plot_local_clocks(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=ax,
        scale_circles=[3, 0.5, 0.5],
        move_circles=[[0, 0], [0, 0], [0, 0]],
        annotates=[0.5, 0.5, 0.5],
        arrow_width=0.05,
    )
    # ax.legend(
    #     arrows,
    #     arrow_labels,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, 1.07),
    #     fontsize=7,
    #     ncol=4,
    #     markerscale=0.6,
    #     handlelength=1.5,
    #     columnspacing=0.8,
    #     handletextpad=0.5,
    # )

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
    plt.savefig("plots/paper/bankrupcy/bankrupcy_local.pdf")

    # Between
    fig, ax = plt.subplots(1, figsize=(3.33, 2.8))
    arrows, arrow_labels = plot_inst.plot_between_clock(
        standartize_data=True,
        standartize_coef=True,
        univar_importance=True,
        ax=ax,
        scale_circles=[1],
        move_circles=[[0, 0]],
        annotates=[0.6],
        arrow_width=0.05,
    )
    # ax.legend(
    #     arrows,
    #     arrow_labels,
    #     loc="lower center",
    #     bbox_to_anchor=(0.5, 1.07),
    #     fontsize=7,
    #     ncol=4,
    #     markerscale=0.6,
    #     handlelength=1.5,
    #     columnspacing=0.8,
    #     handletextpad=0.5,
    # )

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("UMAP2", size=8)
    ax.set_xlabel("UMAP1", size=8)
    ax.set_title("Malignant cells", size=8)
    ax.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax.xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.79,
        bottom=0.05,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/bankrupcy/bankrupcy_between.pdf")


def test_between_all_3():
    X_new, obs, standard_embedding, labels, clusters = setup_bankrupcy_data(method="umap")

    fig, axi = plt.subplots(1, 3, figsize=(7.125-0.66, 2.375))
    plt.tight_layout()

    sc = axi[0].scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.3)
    
    legend1 = axi[0].legend(
        handles = sc.legend_elements()[0],
        loc="upper center",
        # bbox_to_anchor=(0.0, 0.0),
        fontsize=7,
        ncol=3,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.1,
        labels=["No bankrupcy", "bankrupcy"])
    
    axi[0].add_artist(legend1)

    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, method="UMAP", cluster_labels=clusters)
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=False,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=axi[0],
        scale_circle=1.2,
        move_circle=[0, 0],
        annotate=5.0,
        arrow_width=0.1,
        plot_scatter=False
    )

    axi[0].set_yticks([])
    axi[0].set_xticks([])
    axi[0].set_ylabel("UMAP2", size=8)
    axi[0].set_xlabel("UMAP1", size=8)
    axi[0].set_title("Global clock", size=8)
    axi[0].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[0].xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.79,
        bottom=0.05,  # wspace=0.21, hspace=0.33
    )

    # Local
    # arrows, arrow_labels = plot_inst.plot_local_clocks(
    #     standartize_data=True,
    #     standartize_coef=True,
    #     biggest_arrow_method=True,
    #     univar_importance=False,
    #     ax=axi[1],
    #     scale_circles=[3, 1, 0.5],
    #     move_circles=[[0, 0], [0, 0], [0, 0]],
    #     annotates=[1.0, 1.0, 0.8],
    #     arrow_width=0.08,
    # )

    # axi[1].set_yticks([])
    # axi[1].set_xticks([])
    # axi[1].set_ylabel("UMAP2", size=8)
    # axi[1].set_xlabel("UMAP1", size=8)
    # axi[1].set_title("Local clock", size=8)
    # axi[1].yaxis.set_label_coords(x=-0.01, y=0.5)
    # axi[1].xaxis.set_label_coords(x=0.5, y=-0.02)

    # # Between
    # _, _ = plot_inst.plot_between_clock(
    #     standartize_data=True,
    #     standartize_coef=True,
    #     univar_importance=True,
    #     ax=axi[2],
    #     scale_circles=[1.25],
    #     move_circles=[[0, 0]],
    #     annotates=[1.1],
    #     arrow_width=0.08,
    # )
    # axi[2].legend(
    #     arrows,
    #     arrow_labels,
    #     loc="lower center",
    #     bbox_to_anchor=(-0.84, 1.12),
    #     fontsize=7,
    #     ncol=8,
    #     markerscale=0.6,
    #     handlelength=1.5,
    #     columnspacing=0.8,
    #     handletextpad=0.5,
    # )

    # axi[2].set_yticks([])
    # axi[2].set_xticks([])
    # axi[2].set_ylabel("UMAP2", size=8)
    # axi[2].set_xlabel("UMAP1", size=8)
    # axi[2].set_title("Inter-cluster clock", size=8)
    # axi[2].yaxis.set_label_coords(x=-0.01, y=0.5)
    # axi[2].xaxis.set_label_coords(x=0.5, y=-0.02)

    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.79,
        bottom=0.07,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/bankrupcy/bankrupcy_3.pdf")

# print_bankrupcy_all()
test_between_all_3()
# teaser()
