from matplotlib import cm, gridspec, pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from src.feature_clock.plot import NonLinearClock
import umap
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches


def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    p = mpatches.FancyArrow(
        0, 0.5 * height, width, 0, length_includes_head=True, head_width=0.75 * height
    )
    return p


def read_data(path):
    return pd.read_csv(path, header=0)


def setup_melody_data(method="tsne", drop_labels=True):
    file_name = "feature_clock_visualization/data/melody.csv"
    X = read_data(file_name)
    # X.rename(columns={"DiabetesPedigreeFunction": "Pedigree"}, inplace=True)
    # X.drop(columns=["Genre"], inplace=True)
    X["Genre"], _ = pd.factorize(X["Genre"])
    X = X.dropna()

    X.rename(
        columns={
            "LyricalContent": "Lyrical Content",
            "ReleasedYear": "Year of Release",
            "NumInstruments": "Num of Instruments",
            "SongLength": "Song Length",
        },
        inplace=True,
    )

    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()

    labels = X["Popularity"]
    if drop_labels:
        X.drop(columns=["Popularity"], inplace=True)
    obs = list(X.columns)

    # compute umap
    if method == "umap":
        reducer = umap.UMAP(random_state=42)
        if not drop_labels:
            K = X.drop(columns=["Popularity"], inplace=False)
            standard_embedding = reducer.fit_transform(K)
        else:
            standard_embedding = reducer.fit_transform(X)

    elif method == "tsne":
        raise NotImplementedError()

    elif method == "phate":
        raise NotImplementedError()

    # get clusters
    # clusters = HDBSCAN(min_samples=10, min_cluster_size=30).fit_predict(X)
    clusters = labels > 0.5
    print(clusters)

    return X, obs, standard_embedding, labels, clusters


def print_melody_all():
    X_new, obs, standard_embedding, labels, clusters = setup_melody_data(
        method="umap", drop_labels=False
    )
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
    plt.savefig("plots/paper/melody/plot_melodyAll_teaser.pdf")


def test_between_all():
    X_new, obs, standard_embedding, labels, clusters = setup_melody_data(method="umap")

    fig, ax = plt.subplots(1, figsize=(3.33, 2.8))
    plt.tight_layout()
    plot_inst = NonLinearClock(
        X_new, obs, standard_embedding, labels, method="UMAP", cluster_labels=clusters
    )
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=False,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=ax,
        scale_circle=1,
        move_circle=[0, 0],
        annotate=0.6,
        arrow_width=1.5,
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
    ax.set_title("Melody popularity", size=8)
    ax.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax.xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.79,
        bottom=0.05,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/melody/melody_global.pdf")

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

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("UMAP2", size=8)
    ax.set_xlabel("UMAP1", size=8)
    ax.set_title("Melody popularity", size=8)
    ax.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax.xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.79,
        bottom=0.05,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/melody/melody_local.pdf")

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
    ax.set_title("Malignant cells", size=8)
    ax.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax.xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.79,
        bottom=0.05,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/melody/melody_between.pdf")


def test_between_all_3():
    X_new, obs, standard_embedding, labels, clusters = setup_melody_data(method="umap")

    colors = [
        "tab:pink",
        "tab:green",
        "tab:blue",
        "tab:red",
        "tab:orange",
        "tab:purple",
        "tab:cyan",
        "tab:olive",
        "tab:brown",
    ]

    fig_size = ((7.125 - 0.17), ((7.125 - 0.17) / 1.8) / 1.618)

    fig = plt.figure(
        constrained_layout=True,
        figsize=fig_size,
        dpi=1000,
        facecolor="w",
        edgecolor="k",
    )
    spec2 = gridspec.GridSpec(
        ncols=4,
        nrows=1,
        figure=fig,
        left=0.02,
        right=1,
        top=0.77,
        bottom=0.06,
        wspace=0.15,
    )
    ax1 = fig.add_subplot(spec2[0])
    ax2 = fig.add_subplot(spec2[1])
    ax3 = fig.add_subplot(spec2[2])
    axi = [ax1, ax2, ax3]

    spec23 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=spec2[3], wspace=0.05)
    ax4_11 = fig.add_subplot(spec23[0, 0])
    ax4_12 = fig.add_subplot(spec23[0, 1])
    ax4_13 = fig.add_subplot(spec23[0, 2])
    ax4_21 = fig.add_subplot(spec23[1, 0])
    ax4_22 = fig.add_subplot(spec23[1, 1])
    ax4_23 = fig.add_subplot(spec23[1, 2])
    ax4_32 = fig.add_subplot(spec23[2, 1])

    # Local
    scs = []
    for val, i in zip([-1, 0, 1], [0, 2, 8]):
        if val == -1:
            sc = axi[0].scatter(
                standard_embedding[clusters == val, 0],
                standard_embedding[clusters == val, 1],
                marker=".",
                color="gray",
                alpha=0.1,
                s=30,
            )
            sc = axi[1].scatter(
                standard_embedding[clusters == val, 0],
                standard_embedding[clusters == val, 1],
                marker=".",
                color="gray",
                alpha=0.1,
                s=30,
            )
            axi[2].scatter(
                standard_embedding[clusters == val, 0],
                standard_embedding[clusters == val, 1],
                marker=".",
                color="gray",
                alpha=0.1,
                s=30,
            )

        else:
            sc = axi[0].scatter(
                standard_embedding[clusters == val, 0],
                standard_embedding[clusters == val, 1],
                marker=".",
                color=matplotlib.colormaps["Paired"].colors[i],
                alpha=0.3,
                s=30,
            )

            sc = axi[1].scatter(
                standard_embedding[clusters == val, 0],
                standard_embedding[clusters == val, 1],
                marker=".",
                color=matplotlib.colormaps["Paired"].colors[i],
                alpha=0.3,
                s=30,
            )

            axi[2].scatter(
                standard_embedding[clusters == val, 0],
                standard_embedding[clusters == val, 1],
                marker=".",
                color=matplotlib.colormaps["Paired"].colors[i],
                alpha=0.3,
                s=30,
            )

        scs.append(sc)

    plot_inst = NonLinearClock(
        X_new,
        obs,
        standard_embedding,
        labels,
        method="UMAP",
        cluster_labels=clusters,
        color_scheme=colors,
    )

    arrows1, arrow_labels1 = plot_inst.plot_global_clock(
        standartize_data=False,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=axi[0],
        scale_circle=2.5,
        move_circle=[0, 0],
        annotate=1.5,
        arrow_width=0.2,
        plot_scatter=False,
    )

    axi[0].set_yticks([])
    axi[0].set_xticks([])
    axi[0].set_ylabel("UMAP2", size=8)
    axi[0].set_xlabel("UMAP1", size=8)
    axi[0].set_title("Global clock", size=8)
    axi[0].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[0].xaxis.set_label_coords(x=0.5, y=-0.02)

    arrows2, arrow_labels2 = plot_inst.plot_local_clocks(
        standartize_data=False,
        standartize_coef=False,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=axi[1],
        scale_circles=[
            1.5,
            1.3,
        ],
        move_circles=[[1.5, 1], [-1, -1]],
        annotates=[1, 1, 1],
        arrow_width=0.15,
        clocks_labels=["Unpopular", "Popular"],
        plot_scatter=False,
        plot_hulls=False,
    )

    axi[1].set_yticks([])
    axi[1].set_xticks([])
    axi[1].set_ylabel("UMAP2", size=8)
    axi[1].set_xlabel("UMAP1", size=8)
    axi[1].set_title("Local clock", size=8)
    axi[1].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[1].xaxis.set_label_coords(x=0.5, y=-0.02)

    arrows3, arrow_labels3 = plot_inst.plot_between_clock(
        standartize_data=False,
        standartize_coef=True,
        univar_importance=True,
        ax=axi[2],
        scale_circles=[
            2,
        ],
        move_circles=[[0, 0]],
        annotates=[
            1.0,
        ],
        arrow_width=0.15,
        plot_scatter=False,
        plot_hulls=False,
        clocks_labels=["Unpopular", "Popular"],
    )

    arrows_dict = {}
    for i, val in enumerate(arrow_labels3):
        arrows_dict[val] = arrows3[i]
    for i, val in enumerate(arrow_labels1):
        arrows_dict[val] = arrows1[i]
    for i, val in enumerate(arrow_labels2):
        arrows_dict[val] = arrows2[i]

    hatches = [plt.plot([], marker="", ls="")[0]] + list(arrows_dict.values())
    labels = ["Factors:"] + list(arrows_dict.keys())

    hatches = (
        [plt.plot([], marker="", ls="")[0]] * 2
        + [list(arrows_dict.values())[0]]
        + [scs[1]]
        + [list(arrows_dict.values())[1]]
        + [scs[2]]
        + [list(arrows_dict.values())[2]]
        + [plt.plot([], marker="", ls="")[0]]
        + [list(arrows_dict.values())[3]]
        + [plt.plot([], marker="", ls="")[0]]
        + [list(arrows_dict.values())[4]]
        + [plt.plot([], marker="", ls="")[0]]
        + [list(arrows_dict.values())[5]]
        + [plt.plot([], marker="", ls="")[0]]
    )

    labels = (
        ["Factors:", "Labels:"]
        + [list(arrows_dict.keys())[0]]
        + ["Unpopular"]
        + [list(arrows_dict.keys())[1]]
        + ["Popular"]
        + [list(arrows_dict.keys())[2]]
        + [""]
        + [list(arrows_dict.keys())[3]]
        + [""]
        + [list(arrows_dict.keys())[4]]
        + [""]
        + [list(arrows_dict.keys())[5]]
        + [""]
    )

    leg = axi[2].legend(
        hatches,
        labels,
        loc="lower center",
        bbox_to_anchor=(-0.13, 1.1),
        fontsize=7,
        ncol=7,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.5,
        handler_map={
            mpatches.FancyArrow: HandlerPatch(patch_func=make_legend_arrow),
        },
    )
    for vpack in leg._legend_handle_box.get_children()[:1]:
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)
    for lh in leg.legendHandles:
        lh.set_alpha(1)

    axi[2].set_yticks([])
    axi[2].set_xticks([])
    axi[2].set_ylabel("UMAP2", size=8)
    axi[2].set_xlabel("UMAP1", size=8)
    axi[2].set_title("Inter-group clock", size=8)
    axi[2].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[2].xaxis.set_label_coords(x=0.5, y=-0.02)

    # cbar = fig.colorbar(sc0, ax=axi[0], pad=0.02, aspect=40) # ticks=[100, 50, 0]
    # cbar.ax.tick_params(labelsize=5, length=0.8, pad=0.2, grid_linewidth=0.08) #labelrotation=90,
    # cbar.solids.set(alpha=1)
    # cbar.set_label('Song popularity', size=6, rotation=270, labelpad=8)
    # cbar.outline.set_visible(False)

    X_new, obs, standard_embedding, labels, clusters = setup_melody_data(
        method="umap", drop_labels=False
    )

    for (i, o), axi in zip(
        enumerate(obs), [ax4_11, ax4_12, ax4_13, ax4_21, ax4_22, ax4_23, ax4_32]
    ):
        im = axi.scatter(
            standard_embedding[:, 0],
            standard_embedding[:, 1],
            marker=".",
            s=1.3,
            c=X_new[o],
            cmap=cm.coolwarm,
            # vmin=0, vmax=1
        )
        axi.set_yticks([])
        axi.set_xticks([])
        # axi.yaxis.set_label_coords(x=-0.01, y=0.5)
        # axi.xaxis.set_label_coords(x=0.5, y=-0.02)

        if o == "Song Length":
            o = "Song Len."
        elif o == "Num of Instruments":
            o = "Num Inst."
        elif o == "Lyrical Content":
            o = "Lyrical C."
        elif o == "Year of Release":
            o = "Rel. Year"
        axi.set_title(o, size=5, pad=-14)

    ax4_21.set_ylabel("UMAP2", size=8)
    ax4_32.set_xlabel("UMAP1", size=8)

    ax4_21.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax4_32.xaxis.set_label_coords(x=0.55, y=-0.07)

    cbar = fig.colorbar(
        im,
        ax=[ax4_11, ax4_12, ax4_13, ax4_21, ax4_22, ax4_23, ax4_32],
        pad=0.02,
        aspect=40,
    )
    cbar.ax.tick_params(
        labelsize=5, pad=0.2, length=0.8, grid_linewidth=0.1
    )  # labelrotation=90,
    cbar.outline.set_visible(False)
    plt.savefig("plots/paper/melody/melody_3.pdf")


# print_melody_all()
test_between_all_3()
