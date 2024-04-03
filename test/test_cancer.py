from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from src.nonlinear_clock.plot import NonLinearClock
import umap
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
from sklearn import preprocessing

def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p



def read_data(path):
    return pd.read_csv(path, header=0)


def setup_cancer_data(method="tsne", drop_labels=True):
    file_name = "/Users/olga_ovcharenko/Documents/ETH/FS23/ResearchProject/non_lin_visualization/data/Cancer_Data.csv"
    X = read_data(file_name)

    X.drop(columns=["id"], inplace=True)
    X = X.dropna()
    X.diagnosis = X.diagnosis.map({'M': 0, 'B': 1})
    
    labels = X["diagnosis"]
    if drop_labels:
        X.drop(columns=["diagnosis"], inplace=True)
    obs = list(X.columns)
    
    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()
    
    # compute umap
    if method == "umap":
        reducer = umap.UMAP(min_dist=0.2, n_neighbors=30, random_state=42)
        if not drop_labels:
            K = X.drop(columns=["diagnosis"], inplace=False)
            standard_embedding = reducer.fit_transform(K)
        else:
            standard_embedding = reducer.fit_transform(X)

    elif method == "tsne":
        raise NotImplementedError()

    elif method == "phate":
        raise NotImplementedError()
    
    # get clusters
    # clusters = HDBSCAN(min_samples=12).fit_predict(X)
    clusters = KMeans(n_clusters=2, n_init="auto", max_iter=1000, random_state=42).fit_predict(X)

    return X, obs, standard_embedding, labels, clusters


def test_between_all_3():
    colors = [
        'tab:pink', 'darkorchid', 'tab:green', 'cornflowerblue', 'mediumslateblue', 'crimson', 
        'tab:blue', 'tab:olive', 'coral', 'black', 'blueviolet', 'brown', 
        'darkgoldenrod', 'tab:orange', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 
        'cadetblue', 'chartreuse', 'tab:purple', 'tab:cyan', 'tab:red', 'tab:brown', 'darkblue', 
        'darkolivegreen', 'darkorange', 'lawngreen', 'darkred', "orangered"]

    X_new, obs, standard_embedding, labels, clusters = setup_cancer_data(method="umap")

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
        labels=["Malignant", "Benign"])
    
    axi[0].add_artist(legend1)

    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, method="UMAP", cluster_labels=clusters, color_scheme=colors)
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=False,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=axi[0],
        scale_circle=1.2,
        move_circle=[0, 0],
        annotate=1.0,
        arrow_width=0.1,
        plot_scatter=False,
        plot_top_k=5
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
    arrows1, arrow_labels1 = plot_inst.plot_local_clocks(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=False,
        univar_importance=False,
        ax=axi[1],
        scale_circles=[0.5, 0.5],
        move_circles=[[-0.4, 0], [0.5, 0]],
        annotates=[50.0, 50.0],
        arrow_width=0.08,
        plot_top_k=5,
        plot_hulls=True,
        plot_scatter=False
    )

    sc2 = axi[1].scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.3)
    
    legend2 = axi[1].legend(
        handles = sc2.legend_elements()[0],
        loc="upper center",
        # bbox_to_anchor=(0.0, 0.0),
        fontsize=7,
        ncol=3,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.1,
        labels=["Malignant", "Benign"])
    
    axi[1].add_artist(legend2)

    axi[1].set_yticks([])
    axi[1].set_xticks([])
    axi[1].set_ylabel("UMAP2", size=8)
    axi[1].set_xlabel("UMAP1", size=8)
    axi[1].set_title("Local clock", size=8)
    axi[1].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[1].xaxis.set_label_coords(x=0.5, y=-0.02)

    # Between
    sc3 = axi[2].scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.3)
    
    legend3 = axi[2].legend(
        handles = sc3.legend_elements()[0],
        loc="upper center",
        # bbox_to_anchor=(0.0, 0.0),
        fontsize=7,
        ncol=3,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.1,
        labels=["Malignant", "Benign"])
    
    axi[2].add_artist(legend3)
    
    arrows2, arrow_labels2 = plot_inst.plot_between_clock(
        standartize_data=False,
        standartize_coef=True,
        univar_importance=False,
        ax=axi[2],
        scale_circles=[1.25],
        move_circles=[[0, 0]],
        annotates=[1.1],
        arrow_width=0.08,
        plot_top_k=5,
        plot_scatter=False
    )

    arrows_dict = {}
    for i, val in enumerate(arrow_labels):
        arrows_dict[val] = arrows[i]
    for i, val in enumerate(arrow_labels1):
        arrows_dict[val] = arrows1[i]
    for i, val in enumerate(arrow_labels2):
        arrows_dict[val] = arrows2[i]

    axi[2].legend(
        list(arrows_dict.values()),
        list(arrows_dict.keys()),
        loc="lower center",
        bbox_to_anchor=(-0.84, 1.12),
        fontsize=7,
        ncol=5,
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
        left=0.03,
        right=0.99,
        top=0.75,
        bottom=0.07,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/cancer/cancer_3.pdf")


def test_between_all_3():
    colors = [
        'tab:pink', 'darkorchid', 'tab:green', 'cornflowerblue', 'mediumslateblue', 'crimson', 
        'tab:blue', 'tab:olive', 'coral', 'black', 'blueviolet', 'brown', 
        'darkgoldenrod', 'tab:orange', 'darkgreen', 'darkgrey', 'darkkhaki', 'darkmagenta', 
        'cadetblue', 'chartreuse', 'tab:purple', 'tab:cyan', 'tab:red', 'tab:brown', 'darkblue', 
        'darkolivegreen', 'darkorange', 'lawngreen', 'darkred', "orangered"]

    X_new, obs, standard_embedding, labels, clusters = setup_cancer_data(method="umap")

    fig, axi = plt.subplots(1, 3, figsize=((7.125-0.17), ((7.125-0.17)/1.8)/1.618))
    plt.tight_layout()

    sc = axi[0].scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.3)

    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, method="UMAP", cluster_labels=labels, color_scheme=colors)
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=False,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=True,
        ax=axi[0],
        scale_circle=1.2,
        move_circle=[0, 0],
        annotate=1.0,
        arrow_width=0.1,
        plot_scatter=False,
        plot_top_k=5
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
    arrows1, arrow_labels1 = plot_inst.plot_local_clocks(
        standartize_data=False,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=axi[1],
        scale_circles=[1, 1],
        move_circles=[[-0.4, 0], [0.5, 0]],
        annotates=[40, 40],
        arrow_width=0.08,
        plot_top_k=5,
        plot_hulls=True,
        plot_scatter=False
    )

    sc2 = axi[1].scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.3)

    axi[1].set_yticks([])
    axi[1].set_xticks([])
    axi[1].set_ylabel("UMAP2", size=8)
    axi[1].set_xlabel("UMAP1", size=8)
    axi[1].set_title("Local clock", size=8)
    axi[1].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[1].xaxis.set_label_coords(x=0.5, y=-0.02)

    # Between
    sc3 = axi[2].scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.3)
    
    arrows2, arrow_labels2 = plot_inst.plot_between_clock(
        standartize_data=True,
        standartize_coef=True,
        univar_importance=False,
        ax=axi[2],
        scale_circles=[1.25],
        move_circles=[[0, 0]],
        annotates=[1.1],
        arrow_width=0.08,
        plot_top_k=5,
        plot_scatter=False
    )

    arrows_dict = {}
    for i, val in enumerate(arrow_labels):
        arrows_dict[val] = arrows[i]
    for i, val in enumerate(arrow_labels1):
        arrows_dict[val] = arrows1[i]
    for i, val in enumerate(arrow_labels2):
        arrows_dict[val] = arrows2[i]

    
    hatches = [plt.plot([],marker="", ls="")[0]]*3 + \
        list(arrows_dict.values())[0:2] + [sc.legend_elements()[0][0]] + \
        list(arrows_dict.values())[2:4] + [sc.legend_elements()[0][1]] + \
        list(arrows_dict.values())[4:6] + [plt.plot([],marker="", ls="")[0]] + \
        list(arrows_dict.values())[6:8] + [plt.plot([],marker="", ls="")[0]] + \
        list(arrows_dict.values())[8:]  + [plt.plot([],marker="", ls="")[0]]
    
    labels = ["Factors:", " ","Labels:"] + \
        list(arrows_dict.keys())[0:2] + ["Malignant"] + \
        list(arrows_dict.keys())[2:4] + ["Benign"] + \
        list(arrows_dict.keys())[4:6] + [" "] + \
        list(arrows_dict.keys())[6:8] + [" "] + \
        list(arrows_dict.keys())[8:]  + [" "]
    
    leg = axi[2].legend(
        hatches,
        labels,
        loc="lower center",
        bbox_to_anchor=(-0.84, 1.12),
        fontsize=7,
        ncol=6,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.5,
        handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),},
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
        right=0.99,
        top=0.7,
        bottom=0.06,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/cancer/cancer_3.pdf")

# print_cancer_all()
test_between_all_3()
# teaser()
# import matplotlib.colors as mcolors
# print(list(mcolors.TABLEAU_COLORS.keys()))
# print(list(mcolors.CSS4_COLORS.keys()))
