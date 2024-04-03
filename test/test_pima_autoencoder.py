from matplotlib import gridspec, pyplot as plt
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
    
    file_name2 = file
    Y = read_data(file_name2)
    Y.drop(columns=["target"], inplace=True)
    print(Y.shape)

    Y = np.array(Y)

    standard_embedding = Y

    # compute umap
    if method == "umap":
        reducer = umap.UMAP(min_dist=0.2, n_neighbors=30, random_state=42)
        standard_embedding = reducer.fit_transform(Y)

    elif method == "tsne":
        raise NotImplementedError()

    elif method == "phate":
        raise NotImplementedError()
    
    for i in range(standard_embedding.shape[1]):
        standard_embedding[:, i] = (standard_embedding[:, i] - standard_embedding[:, i].mean()) / standard_embedding[:, i].std()
    

    standard_embedding[:, 0] = 3 * standard_embedding[:, 0]
    standard_embedding[:, 1] = 1 * standard_embedding[:, 1]

    # get clusters
    clusters = HDBSCAN(min_samples=12).fit_predict(X)
    # clusters = KMeans(n_clusters=3, n_init="auto", max_iter=1000).fit_predict(X)

    return X, obs, standard_embedding, labels, clusters

def print_pima_all(file, dataset_i):
    X_new, obs, standard_embedding, labels, clusters = setup_pima_data(method="none", drop_labels=False, file=file)
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


def test_between_all_2():
    X_new, obs, standard_embedding, labels, clusters = setup_pima_data(method="", file = "/Users/olga_ovcharenko/Documents/ETH/FS23/ResearchProject/non_lin_visualization/data/emb_6.csv")

    fig_size = ((7.125-0.17)/2, ((7.125-0.17)/2.5)/1.618)

    fig = plt.figure(constrained_layout=True, figsize=fig_size, dpi=1000, facecolor="w",edgecolor="k",)
    spec2 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, 
                     left=0.04, right=1.04, top=0.565, bottom=0.07)
    ax1 = fig.add_subplot(spec2[0])

    spec23 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=spec2[1], wspace=0.05, hspace=0.33)
    ax2_11 = fig.add_subplot(spec23[0, 0])
    ax2_12 = fig.add_subplot(spec23[0, 1])
    ax2_13 = fig.add_subplot(spec23[0, 2])
    ax2_21 = fig.add_subplot(spec23[1, 0])
    ax2_22 = fig.add_subplot(spec23[1, 1])
    ax2_23 = fig.add_subplot(spec23[1, 2])
    ax2_31 = fig.add_subplot(spec23[2, 0])
    ax2_32 = fig.add_subplot(spec23[2, 1])
    ax2_33 = fig.add_subplot(spec23[2, 2])

    plt.tight_layout()

    sc = ax1.scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.2)

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
        ax=ax1,
        scale_circle=3,
        move_circle=[0, 2],
        annotate=2.5,
        arrow_width=0.1
    )

    hatches = [plt.plot([],marker="", ls="")[0]]*4 + arrows[0:3] + \
        [sc.legend_elements()[0][0]] + arrows[3:6] + \
        [sc.legend_elements()[0][1]] + arrows[6:] + [plt.plot([],marker="", ls="")[0]]*2
    labels = ["Factors:", " ", " ", "Labels: "] + arrow_labels[0:3] + \
        ["No diabetes"] + arrow_labels[3:6] + \
        ["Diabetes"] + arrow_labels[6:] + [" ", " "]
    leg = ax1.legend(
        hatches, labels,
        loc="lower center",
        bbox_to_anchor=(1.02, 1.09),
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

    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_ylabel("Dim2", size=8)
    ax1.set_xlabel("Dim1", size=8)
    # ax1.set_title("Second hidden layer", size=8)
    ax1.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax1.xaxis.set_label_coords(x=0.5, y=-0.02)


    # second plot
    axis_array = [ax2_11, ax2_12, ax2_13, ax2_21, ax2_22, ax2_23, ax2_31, ax2_32, ax2_33]

    X_new, obs, standard_embedding, labels, clusters = setup_pima_data(method="", drop_labels=False, file = "/Users/olga_ovcharenko/Documents/ETH/FS23/ResearchProject/non_lin_visualization/data/emb_6.csv")
    standard_embedding[:,0], standard_embedding[:,1] = 1 * standard_embedding[:,0], 7 * standard_embedding[:,1]
    for (i, o), axi in zip(enumerate(obs), axis_array):
        if o == "Species":
            break

        im = axi.scatter(
            standard_embedding[:, 0],
            standard_embedding[:, 1],
            marker=".",
            s=1,
            c=X_new[o],
            cmap="jet",
            alpha=0.7
        )
        axi.set_yticks([])
        axi.set_xticks([])
        axi.yaxis.set_label_coords(x=-0.01, y=0.5)
        axi.xaxis.set_label_coords(x=0.5, y=-0.02)
        if o == "SkinThickness":
            o = "SkinThick."
        elif o == "BloodPressure":
            o = "BloodPr."
        axi.set_title(o, size=5, pad=-9)

    ax2_21.set_ylabel("Dim2", size=8)
    ax2_32.set_xlabel("Dim1", size=8)

    ax2_21.yaxis.set_label_coords(0, 0.5)
    ax2_32.xaxis.set_label_coords(0.5, -0.07)

    cbar = fig.colorbar(im, ax=axis_array, pad=0.02, ticks=[-1, 0, 1], aspect=40)
    cbar.ax.tick_params(labelsize=5, pad=0.2, length=0.8, grid_linewidth=0.1) #labelrotation=90,
    cbar.outline.set_visible(False)

    plt.savefig("plots/paper/pima_network/pima_global_autoencoder.pdf")



def test_between_all_new():
    X_new, obs, standard_embedding, labels, clusters = setup_pima_data(method="", file = "/Users/olga_ovcharenko/Documents/ETH/FS23/ResearchProject/non_lin_visualization/data/emb_1_e-5.csv")

    fig_size = ((7.125-0.17)/2, ((7.125-0.17)/2.5)/1.618)

    fig = plt.figure(constrained_layout=True, figsize=fig_size, dpi=1000, facecolor="w",edgecolor="k",)
    spec2 = gridspec.GridSpec(ncols=2, nrows=1, figure=fig, 
                     left=0.04, right=1.04, top=0.565, bottom=0.07)
    ax1 = fig.add_subplot(spec2[0])

    spec23 = gridspec.GridSpecFromSubplotSpec(3, 3, subplot_spec=spec2[1], wspace=0.05, hspace=0.33)
    ax2_11 = fig.add_subplot(spec23[0, 0])
    ax2_12 = fig.add_subplot(spec23[0, 1])
    ax2_13 = fig.add_subplot(spec23[0, 2])
    ax2_21 = fig.add_subplot(spec23[1, 0])
    ax2_22 = fig.add_subplot(spec23[1, 1])
    ax2_23 = fig.add_subplot(spec23[1, 2])
    ax2_31 = fig.add_subplot(spec23[2, 0])
    ax2_32 = fig.add_subplot(spec23[2, 1])
    ax2_33 = fig.add_subplot(spec23[2, 2])

    plt.tight_layout()

    sc = ax1.scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.2)

    colors = [
        'tab:pink', 'tab:green', 'tab:blue', 'tab:orange',
        'tab:purple', 'tab:cyan', 'tab:red', 'tab:brown']
    
    plot_inst = NonLinearClock(
        X_new, obs, standard_embedding, labels, method="UMAP", 
        cluster_labels=clusters, color_scheme=colors)
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=False,
        standartize_coef=True,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=ax1,
        scale_circle=4,
        move_circle=[7, 1],
        annotate=1.8,
        arrow_width=0.1
    )

    hatches = [plt.plot([],marker="", ls="")[0]]*4 + arrows[0:3] + \
        [sc.legend_elements()[0][0]] + arrows[3:6] + \
        [sc.legend_elements()[0][1]] + arrows[6:] + [plt.plot([],marker="", ls="")[0]]*2
    labels = ["Factors:", " ", " ", "Labels: "] + arrow_labels[0:3] + \
        ["No diabetes"] + arrow_labels[3:6] + \
        ["Diabetes"] + arrow_labels[6:] + [" ", " "]
    leg = ax1.legend(
        hatches, labels,
        loc="lower center",
        bbox_to_anchor=(1.02, 1.09),
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

    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_ylabel("Dim2", size=8)
    ax1.set_xlabel("Dim1", size=8)
    # ax1.set_title("Second hidden layer", size=8)
    ax1.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax1.xaxis.set_label_coords(x=0.5, y=-0.02)


    # second plot
    axis_array = [ax2_11, ax2_12, ax2_13, ax2_21, ax2_22, ax2_23, ax2_31, ax2_32, ax2_33]

    X_new, obs, standard_embedding, labels, clusters = setup_pima_data(method="", drop_labels=False, file = "/Users/olga_ovcharenko/Documents/ETH/FS23/ResearchProject/non_lin_visualization/data/emb_1_e-5.csv")
    standard_embedding[:,0], standard_embedding[:,1] = 1 * standard_embedding[:,0], 7 * standard_embedding[:,1]
    for (i, o), axi in zip(enumerate(obs), axis_array):
        if o == "Species":
            break

        im = axi.scatter(
            standard_embedding[:, 0],
            standard_embedding[:, 1],
            marker=".",
            s=1,
            c=X_new[o],
            cmap="jet",
            alpha=0.7
        )
        axi.set_yticks([])
        axi.set_xticks([])
        axi.yaxis.set_label_coords(x=-0.01, y=0.5)
        axi.xaxis.set_label_coords(x=0.5, y=-0.02)
        if o == "SkinThickness":
            o = "SkinThick."
        elif o == "BloodPressure":
            o = "BloodPr."
        axi.set_title(o, size=5, pad=-9)

    ax2_21.set_ylabel("Dim2", size=8)
    ax2_32.set_xlabel("Dim1", size=8)

    ax2_21.yaxis.set_label_coords(0, 0.5)
    ax2_32.xaxis.set_label_coords(0.5, -0.07)

    cbar = fig.colorbar(im, ax=axis_array, pad=0.02, ticks=[-1, 0, 1], aspect=40)
    cbar.ax.tick_params(labelsize=5, pad=0.2, length=0.8, grid_linewidth=0.1) #labelrotation=90,
    cbar.outline.set_visible(False)

    plt.savefig("plots/paper/pima_network/pima_global_autoencoder_new.pdf")

# test_between_all_2()
test_between_all_new()
