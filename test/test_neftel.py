from matplotlib import gridspec, pyplot as plt
import numpy as np
import scanpy as sp
from src.nonlinear_clock.plot import NonLinearClock
import scanpy.external as sce
import phate
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches

def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p


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
            # vmin=0, vmax=1
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
    plt.savefig("plots/paper/neftel/plot_neftelAll.pdf")


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
    plt.savefig("plots/paper/neftel/neftel_global.pdf")

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
        arrow_width=0.01,
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
    plt.savefig("plots/paper/neftel/neftel_local.pdf")

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
        arrow_width=0.03,
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
    plt.savefig("plots/paper/neftel/neftel_between.pdf")

def test_all_in_row():
    X_new, obs, standard_embedding, labels = setup_neftel_data(method="umap")

    fig, axi = plt.subplots(1, 3, figsize=(7.125-0.66, 2.375))
    plt.tight_layout()
    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, "UMAP")
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=True,
        ax=axi[0],
        scale_circle=2,
        move_circle=[0, 0],
        annotate=0.6,
        arrow_width=0.05
    )

    axi[0].set_yticks([])
    axi[0].set_xticks([])
    axi[0].set_ylabel("UMAP2", size=8)
    axi[0].set_xlabel("UMAP1", size=8)
    axi[0].set_title("Global clock", size=8)
    axi[0].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[0].xaxis.set_label_coords(x=0.5, y=-0.02)

    # Local
    arrows, arrow_labels = plot_inst.plot_local_clocks(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=True,
        ax=axi[1],
        scale_circles=[1, 0.4, 0.25],
        move_circles=[[-0.2, 0], [0.4, 0.3], [-0.1, -0.4]],
        annotates=[0.5, 0.5, 0.3],
        arrow_width=0.05
    )

    axi[1].set_yticks([])
    axi[1].set_xticks([])
    axi[1].set_ylabel("UMAP2", size=8)
    axi[1].set_xlabel("UMAP1", size=8)
    axi[1].set_title("Local clock", size=8)
    axi[1].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[1].xaxis.set_label_coords(x=0.5, y=-0.02)

    # Between
    arrows, arrow_labels = plot_inst.plot_between_clock(
        standartize_data=True,
        standartize_coef=True,
        univar_importance=True,
        ax=axi[2],
        scale_circles=[1, 1.5],
        move_circles=[[0, 0], [0.7, 0]],
        annotates=[0.3, 0.2],
        arrow_width=0.05
    )
    axi[2].legend(
        arrows,
        arrow_labels,
        loc="lower center",
        bbox_to_anchor=(-0.84, 1.12),
        fontsize=7,
        ncol=9,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.5,
    )

    axi[2].set_yticks([])
    axi[2].set_xticks([])
    axi[2].set_ylabel("UMAP2", size=8)
    axi[2].set_xlabel("UMAP1", size=8)
    axi[2].set_title("Inter-cluster clock", size=8)
    axi[2].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[2].xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.79,
        bottom=0.07,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/neftel/neftel_3.pdf")



def test_experiment1():
    X_new, obs, standard_embedding, labels = setup_neftel_data(method="umap")

    fig, ax = plt.subplots(1, figsize=(3.4775/2, 3.4775/2))
    plt.tight_layout()
    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, "UMAP")
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=False,
        univar_importance=True,
        ax=ax,
        scale_circle=2,
        move_circle=[0, 0],
        annotate=0.9,
        arrow_width=0.03,
        angle_shift=5
    )

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("UMAP2", size=8)
    ax.set_xlabel("UMAP1", size=8)
    # ax.set_title("Malignant cells", size=8)
    ax.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax.xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.95,
        bottom=0.08,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/neftel/neftel_all_only_clock.pdf")

    fig, ax = plt.subplots(1, figsize=(3.4775/2, 3.4775/2))
    plt.tight_layout()
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=True,
        ax=ax,
        scale_circle=2,
        move_circle=[0, 0],
        annotate=0.9,
        arrow_width=0.05
    )
    
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("UMAP2", size=8)
    ax.set_xlabel("UMAP1", size=8)
    # ax.set_title("Malignant cells", size=8)
    ax.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax.xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.95,
        bottom=0.08,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/neftel/neftel_biggest_only_clock.pdf")


def test_experiment2():
    X_new, obs, standard_embedding, labels = setup_neftel_data(method="umap")

    fig, ax = plt.subplots(1, figsize=(3.4775/2, 3.4775/2))
    plt.tight_layout()
    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, "UMAP")
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=False,
        univar_importance=True,
        ax=ax,
        scale_circle=2,
        move_circle=[0, 0],
        annotate=0.9,
        arrow_width=0.01,
        angle_shift=1
    )

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("UMAP2", size=8)
    ax.set_xlabel("UMAP1", size=8)
    # ax.set_title("Malignant cells", size=8)
    ax.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax.xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.95,
        bottom=0.08,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/neftel/neftel_all_1_degree.pdf")
    
    fig, ax = plt.subplots(1, figsize=(3.4775/2, 3.4775/2))
    plt.tight_layout()
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=False,
        univar_importance=True,
        ax=ax,
        scale_circle=2,
        move_circle=[0, 0],
        annotate=0.9,
        arrow_width=0.01,
        angle_shift=5
    )

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("UMAP2", size=8)
    ax.set_xlabel("UMAP1", size=8)
    # ax.set_title("Malignant cells", size=8)
    ax.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax.xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.95,
        bottom=0.08,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/neftel/neftel_all_5_degree.pdf")

    
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=False,
        univar_importance=True,
        ax=ax,
        scale_circle=2,
        move_circle=[0, 0],
        annotate=0.9,
        arrow_width=0.01,
        angle_shift=15
    )

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("UMAP2", size=8)
    ax.set_xlabel("UMAP1", size=8)
    # ax.set_title("Malignant cells", size=8)
    ax.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax.xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.08,
        right=0.98,
        top=0.95,
        bottom=0.08,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/neftel/neftel_all_15_degree.pdf")


def test_all_4_in_row():
    X_new, obs, standard_embedding, labels = setup_neftel_data(method="umap")

    # fig, axi = plt.subplots(1, 3, figsize=(7.125-0.66, 2.375))
    # plt.tight_layout()

    dpi = 1000
    fig_size = ((7.125-0.17), ((7.125-0.17)/1.8)/1.618)

    fig = plt.figure(constrained_layout=True, figsize=fig_size, dpi=dpi, facecolor="w",edgecolor="k",)
    spec2 = gridspec.GridSpec(ncols=4, nrows=1, figure=fig, 
                     left=0.02, right=1, top=0.82, bottom=0.06, wspace=0.15)
    ax1 = fig.add_subplot(spec2[0])
    ax2 = fig.add_subplot(spec2[1])
    ax3 = fig.add_subplot(spec2[2])

    spec23 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=spec2[3], wspace=0.05)
    ax4_11 = fig.add_subplot(spec23[0, 0])
    ax4_12 = fig.add_subplot(spec23[0, 1])
    ax4_13 = fig.add_subplot(spec23[0, 2])
    ax4_21 = fig.add_subplot(spec23[1, 0])
    ax4_22 = fig.add_subplot(spec23[1, 1])
    ax4_23 = fig.add_subplot(spec23[1, 2])
    ax4_31 = fig.add_subplot(spec23[2, 0])
    ax4_32 = fig.add_subplot(spec23[2, 1])
    ax4_33 = fig.add_subplot(spec23[2, 2])


    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, "UMAP")
    arrows1, arrow_labels1 = plot_inst.plot_global_clock(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=True,
        ax=ax1,
        scale_circle=2,
        move_circle=[0, 0],
        annotate=0.6,
        arrow_width=0.05
    )

    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_ylabel("UMAP2", size=8)
    ax1.set_xlabel("UMAP1", size=8)
    ax1.set_title("Global clock", size=8)
    ax1.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax1.xaxis.set_label_coords(x=0.5, y=-0.02)

    # Local
    arrows2, arrow_labels2 = plot_inst.plot_local_clocks(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=True,
        ax=ax2,
        scale_circles=[1.5, 0.9],
        move_circles=[[-0.6, 0.1], [0.6, -0.2]],
        annotates=[0.5, 0.5],
        arrow_width=0.05
    )

    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_ylabel("UMAP2", size=8)
    ax2.set_xlabel("UMAP1", size=8)
    ax2.set_title("Local clock", size=8)
    ax2.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax2.xaxis.set_label_coords(x=0.5, y=-0.02)

    # Between
    arrows3, arrow_labels3 = plot_inst.plot_between_clock(
        standartize_data=True,
        standartize_coef=True,
        univar_importance=True,
        ax=ax3,
        scale_circles=[1.5, 1.5],
        move_circles=[[0, 0], [0.7, 0]],
        annotates=[0.4, 0.2],
        arrow_width=0.05
    )

    arrows_dict = {}
    for i, val in enumerate(arrow_labels3):
        arrows_dict[val] = arrows3[i]
    for i, val in enumerate(arrow_labels1):
        arrows_dict[val] = arrows1[i]
    for i, val in enumerate(arrow_labels2):
        arrows_dict[val] = arrows2[i]
    
    hatches = [plt.plot([],marker="", ls="")[0]] + list(arrows_dict.values())
    labels = ["Factors:"] + list(arrows_dict.keys())

    leg = ax3.legend(
        hatches,
        labels,
        loc="lower center",
        bbox_to_anchor=(-0.13, 1.1),
        fontsize=7,
        ncol=10,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.5,
        handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),},
    )
    for vpack in leg._legend_handle_box.get_children()[:1]:
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)

    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.set_ylabel("UMAP2", size=8)
    ax3.set_xlabel("UMAP1", size=8)
    ax3.set_title("Inter-cluster clock", size=8)
    ax3.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax3.xaxis.set_label_coords(x=0.5, y=-0.02)

    X_new, obs, standard_embedding, labels = setup_neftel_data(method="umap")
    
    for (i, o), axi in zip(enumerate(obs), [ax4_11, ax4_12, ax4_13, ax4_21, ax4_22, ax4_23, ax4_31, ax4_32, ax4_33]):
        im = axi.scatter(
            standard_embedding[:, 0],
            standard_embedding[:, 1],
            marker=".",
            s=1.3,
            c=X_new[o],
            cmap="Spectral",
            # vmin=0, vmax=1
        )
        axi.set_yticks([])
        axi.set_xticks([])
        # axi.yaxis.set_label_coords(x=-0.01, y=0.5)
        # axi.xaxis.set_label_coords(x=0.5, y=-0.02)
        
        if o == "genes_expressed":
            o = "genes_exp."
        axi.set_title(o, size=5, pad=-14)

    ax4_21.set_ylabel("UMAP2", size=8)
    ax4_32.set_xlabel("UMAP1", size=8)

    ax4_21.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax4_32.xaxis.set_label_coords(x=0.55, y=-0.07)

    cbar = fig.colorbar(im, ax=[ax4_11, ax4_12, ax4_13, ax4_21, ax4_22, ax4_23, ax4_31, ax4_32, ax4_33], 
                        pad=0.02, aspect=40)
    cbar.ax.tick_params(labelsize=5, pad=0.2, length=0.8, grid_linewidth=0.1) #labelrotation=90,
    cbar.outline.set_visible(False)

    plt.savefig("plots/paper/neftel/neftel_3.pdf")


# test_between_all()
test_all_4_in_row()
# print_neftel_all()
# test_experiment2()