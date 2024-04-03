from matplotlib import pyplot as plt
from matplotlib.legend_handler import HandlerPatch
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
        reducer = umap.UMAP(min_dist=0.2, n_neighbors=30, random_state=42)
        if not drop_labels:
            K = X.drop(columns=["Outcome"], inplace=False)
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

def print_pima_all():
    X_new, obs, standard_embedding, labels, clusters = setup_pima_data(method="umap", drop_labels=False)
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
    names = {'SkinThickness': 'a', 'Insulin': 'd', 'BMI': 'g', 'Age': 'b', 'Pregnancies': 'e', "Pedigree": "h", 'BloodPressure': 'c', 'Glucose': 'f', "Outcome": "i"}
    for i, o in enumerate(list(names.keys())):
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
        axi[i % 3, i // 3].set_title(names[o], size=8, pad=-14)

    # axi[1, 0].set_ylabel("UMAP2", size=8)
    # axi[2, 1].set_xlabel("UMAP1", size=8)

    plt.subplots_adjust(
        left=0.05,
        right=1,
        top=0.95,
        bottom=0.05,  # wspace=0.21, hspace=0.33
    )
    # cbar = fig.colorbar(im, ax=axi.ravel().tolist(), pad=0.1)
    # cbar.ax.tick_params(labelsize=7) 
    for ax in axi:
        for a in ax:
            a.axis('off') 
    plt.savefig("plots/paper/pima/plot_pimaAll_teaser.pdf")


def test_between_all():
    X_new, obs, standard_embedding, labels, clusters = setup_pima_data(method="umap")

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
    plt.savefig("plots/paper/pima/pima_global.pdf")

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
    plt.savefig("plots/paper/pima/pima_local.pdf")

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
    plt.savefig("plots/paper/pima/pima_between.pdf")


def test_between_all_3():
    X_new, obs, standard_embedding, labels, clusters = setup_pima_data(method="umap")
    colors = [
        'tab:pink', 'tab:green', 'tab:blue', 'tab:orange',
        'tab:purple', 'tab:cyan', 'tab:red', 'tab:brown', 'tab:olive']

    fig, axi = plt.subplots(1, 3, figsize=((7.125-0.17), ((7.125-0.17)/1.8)/1.618))
    plt.tight_layout()
    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, method="UMAP", cluster_labels=labels, color_scheme=colors)
    
    sc = axi[0].scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.2)
    # legend1 = axi[0].legend(
    #     handles = sc.legend_elements()[0],
    #     loc="upper center",
    #     # bbox_to_anchor=(0.0, 0.0),
    #     fontsize=7,
    #     ncol=3,
    #     markerscale=0.6,
    #     handlelength=1.5,
    #     columnspacing=0.8,
    #     handletextpad=0.1,
    #     labels=["Healthy", "Diabetis"],
    #     alignment="left" )
    # axi[0].add_artist(legend1)

    arrows1, arrow_labels1 = plot_inst.plot_global_clock(
        standartize_data=False,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=axi[0],
        scale_circle=2.5,
        move_circle=[0, 0],
        annotate=1.0,
        arrow_width=0.08
    )
    axi[0].set_yticks([])
    axi[0].set_xticks([])
    axi[0].set_ylabel("UMAP2", size=8)
    axi[0].set_xlabel("UMAP1", size=8)
    axi[0].set_title("Global clock", size=8)
    axi[0].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[0].xaxis.set_label_coords(x=0.5, y=-0.02)

    # Local
    sc = axi[1].scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.2)
    # legend2 = axi[1].legend(
    #     handles = sc.legend_elements()[0],
    #     loc="upper center",
    #     # bbox_to_anchor=(0.0, 0.0),
    #     fontsize=7,
    #     ncol=3,
    #     markerscale=0.6,
    #     handlelength=1.5,
    #     columnspacing=0.8,
    #     handletextpad=0.1,
    #     labels=["Healthy", "Diabetis"])
    # axi[1].add_artist(legend2)

    arrows2, arrow_labels2 = plot_inst.plot_local_clocks(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=axi[1],
        scale_circles=[1.5, 1.5],
        move_circles=[[-2.2, 0], [2.5, 0]],
        annotates=[0.9, 0.9],
        arrow_width=0.08,
        plot_scatter=False,
        plot_hulls=False
    )

    axi[1].set_yticks([])
    axi[1].set_xticks([])
    axi[1].set_ylabel("UMAP2", size=8)
    axi[1].set_xlabel("UMAP1", size=8)
    axi[1].set_title("Local clock", size=8)
    axi[1].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[1].xaxis.set_label_coords(x=0.5, y=-0.02)

    # Between
    sc = axi[2].scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.2)
    # legend3 = axi[2].legend(
    #     handles = sc.legend_elements()[0],
    #     labels=["Healthy", "Diabetis"],
    #     loc="upper center",
    #     # bbox_to_anchor=(0.0, 0.0),
    #     fontsize=7,
    #     ncol=3,
    #     markerscale=0.6,
    #     handlelength=1.5,
    #     columnspacing=0.8,
    #     handletextpad=0.1,)
    # axi[2].add_artist(legend3)

    arrows3, arrow_labels3 = plot_inst.plot_between_clock(
        standartize_data=True,
        standartize_coef=True,
        univar_importance=False,
        ax=axi[2],
        scale_circles=[4],
        move_circles=[[0, 0]],
        annotates=[0.7],
        arrow_width=0.08,
        plot_scatter=False,
        plot_hulls=False
    )

    arrows_dict = {}
    for i, val in enumerate(arrow_labels3):
        arrows_dict[val] = arrows3[i]
    for i, val in enumerate(arrow_labels1):
        arrows_dict[val] = arrows1[i]
    for i, val in enumerate(arrow_labels2):
        arrows_dict[val] = arrows2[i]
    
    # hatches = [plt.plot([],marker="", ls="")[0]]*2 + list(arrows_dict.values()) + sc.legend_elements()[0]
    # labels = ["Factors:", "Labels:"] + list(arrows_dict.keys()) + ["Healthy", "Diabetis"]

    hatches = [plt.plot([],marker="", ls="")[0]]*2 + \
        [list(arrows_dict.values())[0]] + [sc.legend_elements()[0][0]] + \
        [list(arrows_dict.values())[1]] + [sc.legend_elements()[0][1]] + \
        list(arrows_dict.values())[2:]
    
    labels = ["Factors:", "Labels:"] + [list(arrows_dict.keys())[0]] + ["Healthy"] + \
        [list(arrows_dict.keys())[1]] + ["Diabetis"] + \
        list(arrows_dict.keys())[2:]


    leg = axi[1].legend(
        hatches,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.12),
        fontsize=7,
        ncol=9,
        markerscale=0.6,
        handlelength=1.3,
        columnspacing=0.8,
        handletextpad=0.5,
        handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),},
        # markerfirst=False 
    )

    for vpack in leg._legend_handle_box.get_children()[:1]:
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)

    axi[2].set_yticks([])
    axi[2].set_xticks([])
    axi[2].set_ylabel("UMAP2", size=8)
    axi[2].set_xlabel("UMAP1", size=8)
    axi[2].set_title("Inter-cluster clock", size=8)
    axi[2].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[2].xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.75,
        bottom=0.06,  
        wspace=0.1, 
        # hspace=0.1
    )
    plt.savefig("plots/paper/pima/pima_3.pdf")


def teaser():
    X_new, obs, standard_embedding, labels, clusters = setup_pima_data(method="umap")

    fig, axi = plt.subplots(1, 1, figsize=(2.375, 2.375))
    plt.tight_layout()
    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, method="UMAP", cluster_labels=clusters)
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=False,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=axi,
        scale_circle=2,
        move_circle=[0, 0],
        annotate=0.8,
        arrow_width=0.08,
    )

    axi.set_yticks([])
    axi.set_xticks([])
    # axi[1].set_ylabel("UMAP2", size=8)
    # axi[1].set_xlabel("UMAP1", size=8)
    # axi[1].set_title("Diabetis", size=8)
    # axi[1].yaxis.set_label_coords(x=-0.01, y=0.5)
    # axi[1].xaxis.set_label_coords(x=0.5, y=-0.02)

    
    names_dict = {}
    letters = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
    for a, l in zip(arrow_labels, letters):
        names_dict[a] = l
    print(names_dict)
    leg = axi.legend(
        arrows,
        letters,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.88),
        fontsize=7,
        ncol=8,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.5,
       handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),},
    )

    # axi[0].set_yticks([])
    # axi[0].set_xticks([])
    # axi[0].set_ylabel("UMAP2", size=8)
    # axi[0].set_xlabel("UMAP1", size=8)
    # axi[0].set_title("Diabetis", size=8)
    # axi[0].yaxis.set_label_coords(x=-0.01, y=0.5)
    # axi[0].xaxis.set_label_coords(x=0.5, y=-0.02)
    axi.axis('off')
    plt.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.98,
        bottom=0.01,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/pima/pima_general_clock.pdf")

# print_pima_all()
test_between_all_3()
# teaser()
