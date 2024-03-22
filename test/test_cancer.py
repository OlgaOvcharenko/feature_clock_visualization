from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from src.plot import NonLinearClock
import umap
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans


def read_data(path):
    return pd.read_csv(path, header=0)


def setup_cancer_data(method="tsne", drop_labels=True):
    file_name = "/Users/olga_ovcharenko/Documents/ETH/FS23/ResearchProject/non_lin_visualization/data/Cancer_Data.csv"
    X = read_data(file_name)
    # names = dict()
    # for name in list(X.columns):
    #     names[name] = str.strip(name)
    # X.rename(columns=names, inplace=True)

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
    clusters = HDBSCAN(min_samples=12).fit_predict(X)
    # clusters = KMeans(n_clusters=3, n_init="auto", max_iter=1000).fit_predict(X)

    return X, obs, standard_embedding, labels, clusters


def test_between_all_3():
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

    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, method="UMAP", cluster_labels=clusters)
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

    print(len(arrows))

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
    arrows, arrow_labels = plot_inst.plot_local_clocks(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=axi[1],
        scale_circles=[0.5, 5],
        move_circles=[[0, 0], [0, 0]],
        annotates=[1.0, 5.0],
        arrow_width=0.08,
        plot_top_k=5,
        plot_hulls=True
    )

    axi[1].set_yticks([])
    axi[1].set_xticks([])
    axi[1].set_ylabel("UMAP2", size=8)
    axi[1].set_xlabel("UMAP1", size=8)
    axi[1].set_title("Local clock", size=8)
    axi[1].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[1].xaxis.set_label_coords(x=0.5, y=-0.02)
    print(len(arrows))

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
    plt.savefig("plots/paper/cancer/cancer_3.pdf")

# print_cancer_all()
test_between_all_3()
# teaser()
