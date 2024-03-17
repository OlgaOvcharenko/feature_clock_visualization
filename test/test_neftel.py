from matplotlib import pyplot as plt
import numpy as np
import scanpy as sp
from src.plot import NonLinearClock
import scanpy.external as sce
import phate


def read_data(path):
    return sp.read_h5ad(path)


def setup_neftel_data(method="tsne"):
    file_name = "/Users/olga_ovcharenko/Documents/ETH/FS23/ResearchProject/non_lin_visualization/data/neftel_malignant.h5ad"
    X = read_data(file_name)

    obs = [
        "MESlike2",
        "MESlike1",
        "AClike",
        "OPClike",
        "NPClike1",
        "NPClike2",
        "G1S",
        "G2M",
        "genes_expressed",
    ]

    new_data = X.obs[obs].dropna()
    for col in new_data.columns:
        new_data[col] = (new_data[col] - new_data[col].mean()) / new_data[col].std()

    X_new = sp.AnnData(new_data)

    # compute umap
    sp.pp.neighbors(X_new)
    if method == "umap":
        sp.tl.umap(X_new, min_dist=2)

        # get clusters
        standard_embedding = X_new.obsm["X_umap"]

    elif method == "tsne":
        sp.tl.tsne(X_new)

        # get clusters
        standard_embedding = X_new.obsm["X_tsne"]

    elif method == "phate":
        sce.tl.phate(X_new, k=5, a=20, t=150)
        standard_embedding = X_new.obsm["X_phate"]
        print(standard_embedding)
        print(X_new)

    # get labels
    mes = np.stack((X_new.X[:, 0], X_new.X[:, 1]))
    npc = np.stack((X_new.X[:, 4], X_new.X[:, 5]))
    ac, opc = X_new.X[:, 2], X_new.X[:, 3]
    mes_max = np.max(mes, axis=0)
    npc_max = np.max(npc, axis=0)
    res_vect = np.stack((ac, opc, mes_max, npc_max))
    res_labels = np.max(res_vect, axis=0)

    return new_data, obs, standard_embedding, res_labels


def test_umap():
    X_new, obs, standard_embedding, labels = setup_neftel_data(method="umap")

    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, "UMAP")


def test_between_arrow():
    X_new, obs, standard_embedding, labels = setup_neftel_data()

    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, "UMAP")


def test_between_tsne():
    X_new, obs, standard_embedding, labels = setup_neftel_data(method="tsne")

    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, "tsne")


def test_between_phate():
    X_new, obs, standard_embedding, labels = setup_neftel_data(method="phate")

    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, "phate")


def print_neftel_all():
    X_new, obs, standard_embedding, labels = setup_neftel_data(method="umap")
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

    plt.savefig("plots/paper/plot_neftelAll.pdf")


def test_between_all():
    X_new, obs, standard_embedding, labels = setup_neftel_data(method="umap")

    fig, ax = plt.subplots(1, figsize=(3.33, 3.33))
    plt.tight_layout()
    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, "UMAP")
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=True,
        ax=ax,
        scale_circle=2,
        move_circle=[0, 0],
        annotate=0.6,
    )
    ax.legend(
        arrows,
        arrow_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.07),
        fontsize=7,
        ncol=3,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.5,
    )

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
    plt.savefig("plots/paper/neftel_global.pdf")

    # Local
    fig, ax = plt.subplots(1, figsize=(3.33, 3.33))
    arrows, arrow_labels = plot_inst.plot_local_clocks(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=True,
        ax=ax,
        scale_circles=[1, 0.25, 0.25],
        move_circles=[[0, 0], [0.3, 0.3], [-0.1, -0.4]],
        annotates=[0.3, 0.3, 0.3],
    )
    ax.legend(
        arrows,
        arrow_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.07),
        fontsize=7,
        ncol=3,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.5,
    )

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
    plt.savefig("plots/paper/neftel_local.pdf")

    # Between
    fig, ax = plt.subplots(1, figsize=(3.33, 3.33))
    arrows, arrow_labels = plot_inst.plot_between_clock(
        standartize_data=True,
        standartize_coef=True,
        univar_importance=True,
        ax=ax,
        scale_circles=[1, 1.5],
        move_circles=[[0, 0], [0.7, 0]],
        annotates=[0.3, 0.2],
    )
    ax.legend(
        arrows,
        arrow_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.07),
        fontsize=7,
        ncol=3,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.5,
    )

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
    plt.savefig("plots/paper/neftel_between.pdf")


test_between_all()
