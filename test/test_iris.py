from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from src.nonlinear_clock.plot import NonLinearClock
import umap
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans
from sklearn import decomposition  # decomposition.PCA
from sklearn import manifold 


def read_data(path):
    return pd.read_csv(path, header=0)


def setup_iris_data(method="tsne", drop_labels=True):
    file_name = "/Users/olga_ovcharenko/Documents/ETH/FS23/ResearchProject/non_lin_visualization/data/Iris.csv"
    X = read_data(file_name)
    X.rename(columns={"SepalLengthCm": "SepalLength", 
                      "SepalWidthCm": "SepalWidth", 
                      "PetalLengthCm": "PetalLength", 
                      "PetalWidthCm": "PetalWidth"}, inplace=True)
    
    X.drop(columns={"Id"}, inplace=True)
    X = X.dropna()

    X["Species"] = pd. factorize(X["Species"])[0]
    labels = X["Species"]
    if drop_labels:
        X.drop(columns=["Species"], inplace=True)
    obs = list(X.columns)

    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()

    # compute umap
    if method == "umap":
        reducer = umap.UMAP(min_dist=0.2, n_neighbors=30, random_state=42)
        if not drop_labels:
            K = X.drop(columns=["Species"], inplace=False)
            standard_embedding = reducer.fit_transform(K)
        else:
            standard_embedding = reducer.fit_transform(X)

    elif method == "tsne":
        tsne = manifold.TSNE(n_components = 2, learning_rate = 'auto', random_state = 42, n_iter=1000)
        standard_embedding = tsne.fit_transform(X)

    elif method == "phate":
        raise NotImplementedError()
    
    elif method == "pca":
        pca = decomposition.PCA(n_components = 2)
        standard_embedding = pca.fit_transform(X)
        
        return X, obs, standard_embedding, labels, pca.components_
    
    # get clusters
    clusters = HDBSCAN(min_samples=12).fit_predict(X)
    # clusters = KMeans(n_clusters=3, n_init="auto", max_iter=1000).fit_predict(X)

    return X, obs, standard_embedding, labels, clusters

def print_iris_all():
    X_new, obs, standard_embedding, labels, clusters = setup_iris_data(method="tsne", drop_labels=True)
    dpi = 1000
    # fig_size = (2.375, 2.375)
    fig_size = (3.2325, 2.9)
    fig, axi = plt.subplots(
        2,
        2,
        num=None,
        figsize=fig_size,
        dpi=dpi,
        facecolor="w",
        edgecolor="k",
    )
    for i, o in enumerate(obs):
        if o == "Species":
            break

        im = axi[i % 2, i // 2].scatter(
            standard_embedding[:, 0],
            standard_embedding[:, 1],
            marker=".",
            s=5,
            c=X_new[o],
            cmap="jet",
        )
        axi[i % 2, i // 2].set_yticks([])
        axi[i % 2, i // 2].set_xticks([])
        axi[i % 2, i // 2].yaxis.set_label_coords(x=-0.01, y=0.5)
        axi[i % 2, i // 2].xaxis.set_label_coords(x=0.5, y=-0.02)
        axi[i % 2, i // 2].set_title(o, size=8, pad=-14)

    axi[1, 0].set_ylabel("t-SNE2", size=8)
    axi[1, 0].set_xlabel("t-SNE1", size=8)

    plt.subplots_adjust(
        left=0.05,
        right=0.98,
        top=0.95,
        bottom=0.05,  # wspace=0.21, hspace=0.33
    )
    cbar = fig.colorbar(im, ax=axi.ravel().tolist(), pad=0.1)
    cbar.ax.tick_params(labelsize=7) 
    # for ax in axi:
    #     for a in ax:
    #         a.axis('off') 

    # axi[1][2].set_visible(False)
    plt.savefig("plots/paper/iris/plot_irisAll.pdf")


def test_pca():
    dpi = 1000
    fig_size = (3.2325, 3.2325)
    fig, axi = plt.subplots(
        1,
        1,
        num=None,
        figsize=fig_size,
        dpi=dpi,
        facecolor="w",
        edgecolor="k",
    )

    df = pd.read_csv("/Users/olga_ovcharenko/Documents/ETH/FS23/ResearchProject/non_lin_visualization/data/Iris.csv")
    labels = df.Species.map({"Iris-setosa":0, "Iris-versicolor":1, "Iris-virginica":2}).to_numpy()
    
    df.rename(columns={"SepalLengthCm": "SepalLength", 
                      "SepalWidthCm": "SepalWidth", 
                      "PetalLengthCm": "PetalLength", 
                      "PetalWidthCm": "PetalWidth"}, inplace=True)
    
    df.drop(columns={"Id"}, inplace=True)
    df = df.dropna()

    X = df.drop("Species", axis=1).to_numpy()
    pca = decomposition.PCA(n_components=2)
    pcaT = pca.fit_transform(X)

    C = pca.components_

    sc = axi.scatter(pcaT[:,0], pcaT[:,1], marker= '.', c=labels, cmap="tab10")
    
    legend1 = axi.legend(
        handles = sc.legend_elements()[0],
        loc="upper center",
        # bbox_to_anchor=(0.0, 0.0),
        fontsize=7,
        ncol=3,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.1,
        labels=["Setosa", "Versicolor", "Virginica"])
    
    axi.add_artist(legend1)

    colors = ["tab:orange", "tab:green", "tab:red", "tab:gray"]
    labels = ["SepalLength", "SepalWidth", "PetalLength","PetalWidth"]

    for i in range(0, 4):
        axi.arrow(-1.5, 0, C[0][i], C[1][i], 
                    color = colors[i], width = 0.04,
                    label = labels[i]
                    )

    axi.set_yticks([])
    axi.set_xticks([])
    axi.set_ylabel("PCA2", size=8)
    axi.set_xlabel("PCA1", size=8)
    axi.yaxis.set_label_coords(x=-0.01, y=.5)
    axi.xaxis.set_label_coords(x=0.5, y=-0.02)


    axi.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=7,
        ncol=2,
        markerscale=0.6,
        handlelength=1.3,
        columnspacing=0.8,
        handletextpad=0.1,
    )

    plt.subplots_adjust(
        left=0.05, right=0.99, top=0.86, bottom=0.05, wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/iris/plot_biplot.pdf")


def test_between_all():
    X_new, obs, standard_embedding, labels, clusters = setup_iris_data(method="tsne")

    fig, ax = plt.subplots(1, figsize=(3.2325, 3.2325))
    plt.tight_layout()

    sc = ax.scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="tab10", alpha=0.5, zorder=0, s=3)
    
    legend1 = ax.legend(
        handles = sc.legend_elements()[0],
        loc="upper center",
        # bbox_to_anchor=(0.0, 0.0),
        fontsize=7,
        ncol=3,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.1,
        labels=["Setosa", "Versicolor", "Virginica"])
    ax.add_artist(legend1)

    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, method="tsne", cluster_labels=clusters)
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=False,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=ax,
        scale_circle=5,
        move_circle=[-8.0, 0],
        annotate=2.5,
        arrow_width=0.3
    )

    ax.legend(
        arrows,
        arrow_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        fontsize=7,
        ncol=3,
        markerscale=0.6,
        handlelength=1.3,
        columnspacing=0.8,
        handletextpad=0.1,
    )

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylabel("t-SNE2", size=8)
    ax.set_xlabel("t-SNE1", size=8)
    # ax.set_title("Iris dataset", size=8)
    ax.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax.xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.92,
        bottom=0.05,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/iris/iris_global.pdf")

    # # Local
    # fig, ax = plt.subplots(1, figsize=(2.375, 2.375))
    # arrows, arrow_labels = plot_inst.plot_local_clocks(
    #     standartize_data=True,
    #     standartize_coef=True,
    #     biggest_arrow_method=True,
    #     univar_importance=False,
    #     ax=ax,
    #     scale_circles=[3, 0.5, 0.5],
    #     move_circles=[[0, 0], [0, 0], [0, 0]],
    #     annotates=[0.5, 0.5, 0.5],
    #     arrow_width=0.05,
    # )
    # # ax.legend(
    # #     arrows,
    # #     arrow_labels,
    # #     loc="lower center",
    # #     bbox_to_anchor=(0.5, 1.07),
    # #     fontsize=7,
    # #     ncol=4,
    # #     markerscale=0.6,
    # #     handlelength=1.5,
    # #     columnspacing=0.8,
    # #     handletextpad=0.5,
    # # )

    # ax.set_yticks([])
    # ax.set_xticks([])
    # ax.set_ylabel("UMAP2", size=8)
    # ax.set_xlabel("UMAP1", size=8)
    # ax.set_title("Diabetis", size=8)
    # ax.yaxis.set_label_coords(x=-0.01, y=0.5)
    # ax.xaxis.set_label_coords(x=0.5, y=-0.02)
    # plt.subplots_adjust(
    #     left=0.05,
    #     right=0.95,
    #     top=0.79,
    #     bottom=0.05,  # wspace=0.21, hspace=0.33
    # )
    # plt.savefig("plots/paper/iris/iris_local.pdf")

    # # Between
    # fig, ax = plt.subplots(1, figsize=(3.33, 2.8))
    # arrows, arrow_labels = plot_inst.plot_between_clock(
    #     standartize_data=True,
    #     standartize_coef=True,
    #     univar_importance=True,
    #     ax=ax,
    #     scale_circles=[1],
    #     move_circles=[[0, 0]],
    #     annotates=[0.6],
    #     arrow_width=0.05,
    # )
    # # ax.legend(
    # #     arrows,
    # #     arrow_labels,
    # #     loc="lower center",
    # #     bbox_to_anchor=(0.5, 1.07),
    # #     fontsize=7,
    # #     ncol=4,
    # #     markerscale=0.6,
    # #     handlelength=1.5,
    # #     columnspacing=0.8,
    # #     handletextpad=0.5,
    # # )

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
    # plt.savefig("plots/paper/iris/iris_between.pdf")


def test_between_all_3():
    X_new, obs, standard_embedding, labels, clusters = setup_iris_data(method="umap")

    fig, axi = plt.subplots(1, 3, figsize=(7.125-0.66, 2.375))
    plt.tight_layout()
    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, method="UMAP", cluster_labels=clusters)
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=False,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=axi[0],
        scale_circle=1.2,
        move_circle=[0, 0],
        annotate=1.0,
        arrow_width=0.08
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

    axi[0].set_yticks([])
    axi[0].set_xticks([])
    axi[0].set_ylabel("UMAP2", size=8)
    axi[0].set_xlabel("UMAP1", size=8)
    axi[0].set_title("Diabetis", size=8)
    axi[0].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[0].xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.79,
        bottom=0.05,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/iris/iris_global.pdf")

    # Local
    # fig, ax = plt.subplots(1, figsize=(3.33, 3.33))
    arrows, arrow_labels = plot_inst.plot_local_clocks(
        standartize_data=True,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=axi[1],
        scale_circles=[3, 1, 0.5],
        move_circles=[[0, 0], [0, 0], [0, 0]],
        annotates=[1.0, 1.0, 0.8],
        arrow_width=0.08,
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

    axi[1].set_yticks([])
    axi[1].set_xticks([])
    axi[1].set_ylabel("UMAP2", size=8)
    axi[1].set_xlabel("UMAP1", size=8)
    axi[1].set_title("Diabetis", size=8)
    axi[1].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[1].xaxis.set_label_coords(x=0.5, y=-0.02)
    # plt.subplots_adjust(
    #     left=0.05,
    #     right=0.95,
    #     top=0.79,
    #     bottom=0.05,  # wspace=0.21, hspace=0.33
    # )
    # plt.savefig("plots/paper/iris/iris_local.pdf")

    # Between
    # fig, ax = plt.subplots(1, figsize=(3.33, 2.8))
    _, _ = plot_inst.plot_between_clock(
        standartize_data=True,
        standartize_coef=True,
        univar_importance=True,
        ax=axi[2],
        scale_circles=[1.25],
        move_circles=[[0, 0]],
        annotates=[1.1],
        arrow_width=0.08,
    )
    axi[2].legend(
        arrows,
        arrow_labels,
        loc="lower center",
        bbox_to_anchor=(-0.84, 1.12),
        fontsize=7,
        ncol=8,
        markerscale=0.6,
        handlelength=1.5,
        columnspacing=0.8,
        handletextpad=0.5,
    )

    axi[2].set_yticks([])
    axi[2].set_xticks([])
    axi[2].set_ylabel("UMAP2", size=8)
    axi[2].set_xlabel("UMAP1", size=8)
    axi[2].set_title("Diabetis", size=8)
    axi[2].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[2].xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.79,
        bottom=0.07,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/iris/iris_3.pdf")


def teaser():
    X_new, obs, standard_embedding, labels, clusters = setup_iris_data(method="umap")

    fig, axi = plt.subplots(1, 1, figsize=(2.375, 2.375))
    plt.tight_layout()
    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, method="UMAP", cluster_labels=clusters)
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=False,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=axi,
        scale_circle=1.2,
        move_circle=[0, 0],
        annotate=1.0,
        arrow_width=0.08
    )

    axi.set_yticks([])
    axi.set_xticks([])
    # axi[1].set_ylabel("UMAP2", size=8)
    # axi[1].set_xlabel("UMAP1", size=8)
    # axi[1].set_title("Diabetis", size=8)
    # axi[1].yaxis.set_label_coords(x=-0.01, y=0.5)
    # axi[1].xaxis.set_label_coords(x=0.5, y=-0.02)
    plt.subplots_adjust(
        left=0.05,
        right=0.95,
        top=0.79,
        bottom=0.05,  # wspace=0.21, hspace=0.33
    )

    # axi.legend(
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

    # axi[0].set_yticks([])
    # axi[0].set_xticks([])
    # axi[0].set_ylabel("UMAP2", size=8)
    # axi[0].set_xlabel("UMAP1", size=8)
    # axi[0].set_title("Diabetis", size=8)
    # axi[0].yaxis.set_label_coords(x=-0.01, y=0.5)
    # axi[0].xaxis.set_label_coords(x=0.5, y=-0.02)
    axi.axis('off')
    plt.subplots_adjust(
        left=0.0,
        right=0.95,
        top=0.95,
        bottom=0.1,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/iris/iris_general_clock.pdf")

# print_iris_all()
# print_iris_pca()
# test_pca()
# test_between_all_3()
# teaser()
    
test_between_all()
