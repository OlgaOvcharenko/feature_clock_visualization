from matplotlib import cm, gridspec, pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from src.feature_clock.plot import NonLinearClock
import umap
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
from sklearn import manifold, preprocessing


def make_legend_arrow(
        legend,
        orig_handle,
        xdescent,
        ydescent,
        width,
        height,
        fontsize):
    p = mpatches.FancyArrow(
        0,
        0.5 *
        height,
        width,
        0,
        length_includes_head=True,
        head_width=0.75 *
        height)
    return p


def read_data(path):
    return pd.read_csv(path, header=0)


def setup_support_data(method="tsne", drop_labels=True):
    file_name = "feature_clock_visualization/data/support2.csv"
    X = read_data(file_name)
    X.drop(columns=["id"], inplace=True)

    X.drop(
        columns=[
            "death",
            "sfdm2",
            "d.time",
            "surv2m",
            "surv6m"],
        inplace=True)

    for c in X.columns:
        if c in ["sex", "dzgroup", "dzclass", "income", "race", "ca", "dnr"]:
            X[c] = X[c].fillna(X[c].mode())
        elif c == "sfdm2":
            pass
        else:
            X[c] = X[c].fillna(X[c].mean())

    for c in ["sex", "dzgroup", "dzclass", "income", "race", "ca", "dnr"]:
        X[c], _ = pd.factorize(X[c])

    # X['sfdm2'] = X['sfdm2'].map({"no(M2 and SIP pres)": 0, "adl>=4 (>=5 if sur)": 1, "SIP>=30": 2, "Coma or Intub": 4, "<2 mo. follow-up": 3})
    # X = X[X.sfdm2 != 4]
    X = X.dropna(subset=["hospdead"])

    labels = X["hospdead"]

    if drop_labels:
        X.drop(columns=["hospdead"], inplace=True)

    X.rename(
        columns={
            "dzclass": "Disease",
            "avtisst": "Avg. TISS score",
            "aps": "APACHE3 score",
            "totcst": "Cost/charges ratio",
            "diabetes": "Diabetes",
            "dementia": "Dementia",
            "adls": "ADL family",
            "adlp": "ADL patient",
            "bili": "Bilirubin",
            "slos": "Days in study",
            "prg2m": "2 month survival",
            "totmcst": "Micro cost",
            "scoma": "Coma score",
            "sps": "Physiology score",
            "hday": "Study enter",
            "dnr": "Resuscitate order",
            "charges": "Charges",
            "age": "Age",
            "dnrday": "DNR order day",
        },
        inplace=True,
    )

    obs = list(X.columns)

    # X.drop(columns=["diabetes", "ca", "bun", "glucose", "sex"], inplace=True)
    obs = list(X.columns)

    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()
        # print([X[col].min(), X[col].max()])

    # compute umap
    if method == "umap":
        reducer = umap.UMAP(min_dist=0.2, n_neighbors=30, random_state=42)
        if not drop_labels:
            K = X.drop(columns=["hospdead"], inplace=False)
            standard_embedding = reducer.fit_transform(K)
        else:
            standard_embedding = reducer.fit_transform(X)

    elif method == "tsne":
        tsne = manifold.TSNE(
            n_components=2,
            random_state=42,
            perplexity=50,
            learning_rate="auto")
        standard_embedding = tsne.fit_transform(X)

    elif method == "phate":
        raise NotImplementedError()

    standard_embedding = np.array(standard_embedding)
    labels = np.array(labels)
    for i in range(standard_embedding.shape[1]):
        standard_embedding[:, i] = (
            standard_embedding[:, i] - standard_embedding[:, i].mean()
        ) / standard_embedding[:, i].std()

    # get clusters
    clusters = HDBSCAN(min_samples=12).fit_predict(X)
    # clusters = KMeans(n_clusters=2, n_init="auto", max_iter=1000, random_state=42).fit_predict(X)

    return X, obs, standard_embedding, labels, clusters


def test_between_all_3_5():
    colors = [
        "aliceblue",
        "antiquewhite",
        "aqua",
        "aquamarine",
        "azure",
        "beige",
        "bisque",
        "black",
        "blanchedalmond",
        "blue",
        "blueviolet",
        "brown",
        "burlywood",
        "cadetblue",
        "chartreuse",
        "chocolate",
        "coral",
        "cornflowerblue",
        "cornsilk",
        "crimson",
        "cyan",
        "darkblue",
        "darkcyan",
        "darkgoldenrod",
        "darkgray",
        "darkgreen",
        "darkgrey",
        "darkkhaki",
        "darkmagenta",
        "darkolivegreen",
        "darkorange",
        "darkorchid",
        "darkred",
        "darksalmon",
        "darkseagreen",
        "darkslateblue",
        "darkslategray",
        "darkslategrey",
        "darkturquoise",
        "darkviolet",
        "deeppink",
        "deepskyblue",
        "dimgray",
        "dimgrey",
        "dodgerblue",
        "firebrick",
        "floralwhite",
        "forestgreen",
        "fuchsia",
        "gainsboro",
        "ghostwhite",
        "gold",
        "goldenrod",
        "gray",
        "green",
        "greenyellow",
        "grey",
        "honeydew",
        "hotpink",
        "indianred",
        "indigo",
        "ivory",
        "khaki",
        "lavender",
        "lavenderblush",
        "lawngreen",
        "lemonchiffon",
        "lightblue",
        "lightcoral",
        "lightcyan",
        "lightgoldenrodyellow",
        "lightgray",
        "lightgreen",
        "lightgrey",
        "lightpink",
        "lightsalmon",
        "lightseagreen",
        "lightskyblue",
        "lightslategray",
        "lightslategrey",
        "lightsteelblue",
        "lightyellow",
        "lime",
        "limegreen",
        "linen",
        "magenta",
        "maroon",
        "mediumaquamarine",
        "mediumblue",
        "mediumorchid",
        "mediumpurple",
        "mediumseagreen",
        "mediumslateblue",
        "mediumspringgreen",
        "mediumturquoise",
        "mediumvioletred",
        "midnightblue",
        "mintcream",
        "mistyrose",
        "moccasin",
        "navajowhite",
        "navy",
        "oldlace",
        "olive",
        "olivedrab",
        "orange",
        "orangered",
        "orchid",
        "palegoldenrod",
        "palegreen",
        "paleturquoise",
        "palevioletred",
        "papayawhip",
        "peachpuff",
        "peru",
        "pink",
        "plum",
        "powderblue",
        "purple",
        "rebeccapurple",
        "red",
        "rosybrown",
        "royalblue",
        "saddlebrown",
        "salmon",
        "sandybrown",
        "seagreen",
        "seashell",
        "sienna",
        "silver",
        "skyblue",
        "slateblue",
        "slategray",
        "slategrey",
        "snow",
        "springgreen",
        "steelblue",
        "tan",
        "teal",
        "thistle",
        "tomato",
        "turquoise",
        "violet",
        "wheat",
        "white",
        "whitesmoke",
        "yellow",
        "yellowgreen",
    ]
    # tab20 = [
    #     '#1f77b4', '#aec7e8',
    #     '#ff7f0e', '#ffbb78',
    #     '#2ca02c', '#98df8a',
    #     '#d62728', '#ff9896',
    #     '#9467bd', '#c5b0d5',
    #     '#8c564b', '#c49c94',
    #     '#e377c2', '#f7b6d2',
    #     '#7f7f7f', '#c7c7c7',
    #     '#bcbd22', '#dbdb8d',
    #     '#17becf', '#9edae5']

    tab20 = [
        "#1f77b4",
        "#ff7f0e",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "navy",
        "#8c564b",
        "dimgray",
        "#e377c2",
        "darkolivegreen",
        "#bcbd22",
        "#17becf",
        "chartreuse",
    ]

    X_new, obs, standard_embedding, labels, clusters = setup_support_data(
        method="tsne")

    i = 0
    for k in range(len(obs)):
        if obs[k] in [
            "surv2m",
            "surv6m",
            "avtisst",
            "dzclass",
            "aps",
            "death",
            "sps",
            "dnr",
            "adlsc",
            "prg6m",
            "prg2m",
            "d.time",
            "num.co",
        ]:
            colors[k] = tab20[i]
            if obs[k] == "surv6m":
                colors[k] = "darkred"
            if obs[k] == "num.co":
                colors[k] = "orangered"
            if obs[k] == "adlsc":
                colors[k] = "peru"
            i += 1

    fig, axi = plt.subplots(
        1, 3, figsize=((7.125 - 0.17), ((7.125 - 0.17) / 1.8) / 1.618)
    )
    plt.tight_layout()

    scs = []
    # for val, i in zip([0, 1, 2, 3], [0, 5, 2, 1]):
    for val, i in zip([0, 3, 1, 2], [0, 2, 5, 1]):
        sc = axi[0].scatter(
            standard_embedding[labels == val, 0],
            standard_embedding[labels == val, 1],
            marker=".",
            color=matplotlib.colormaps["Accent"].colors[i],
            alpha=0.3,
        )
        scs.append(sc)
        axi[1].scatter(
            standard_embedding[labels == val, 0],
            standard_embedding[labels == val, 1],
            marker=".",
            color=matplotlib.colormaps["Accent"].colors[i],
            alpha=0.3,
        )
        axi[2].scatter(
            standard_embedding[labels == val, 0],
            standard_embedding[labels == val, 1],
            marker=".",
            color=matplotlib.colormaps["Accent"].colors[i],
            alpha=0.3,
        )

    plot_inst = NonLinearClock(
        X_new,
        obs,
        standard_embedding,
        labels,
        method="UMAP",
        cluster_labels=labels,
        color_scheme=colors,
    )
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=False,
        standartize_coef=False,
        biggest_arrow_method=True,
        univar_importance=True,
        ax=axi[0],
        scale_circle=3,
        move_circle=[0, 0],
        annotate=0.8,
        arrow_width=0.1,
        plot_scatter=False,
        plot_top_k=5,
    )

    axi[0].set_yticks([])
    axi[0].set_xticks([])
    axi[0].set_ylabel("t-SNE2", size=8)
    axi[0].set_xlabel("t-SNE1", size=8)
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
        standartize_coef=False,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=axi[1],
        scale_circles=[2.2, 2, 2.5, 0.2],
        # move_circles=[[-1.5, 0.3], [0, 1.5], [2.3, -1], [-0.8, -0.2]],
        move_circles=[[-1.8, 0.3], [-0.8, 0], [2.5, -0.8], [-0.0, 0]],
        annotates=[0.4, 0.4, 0.4, 0.4],
        arrow_width=0.08,
        plot_top_k=5,
        plot_hulls=False,
        plot_scatter=False,
    )

    # sc2 = axi[1].scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.2)

    axi[1].set_yticks([])
    axi[1].set_xticks([])
    axi[1].set_ylabel("t-SNE2", size=8)
    axi[1].set_xlabel("t-SNE1", size=8)
    axi[1].set_title("Local clock", size=8)
    axi[1].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[1].xaxis.set_label_coords(x=0.5, y=-0.02)

    # Between
    # sc3 = axi[2].scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.2)

    arrows2, arrow_labels2 = plot_inst.plot_between_clock(
        standartize_data=True,
        standartize_coef=False,
        univar_importance=False,
        ax=axi[2],
        scale_circles=[6, 10, 0],
        # move_circles=[[1.8, 1.1], [0.1, -0.8], [0.0, 0.0]],
        move_circles=[[0.7, -1.5], [0.4, 1.3], [0, 0]],
        annotates=[0.5, 0.5, 0.5],
        arrow_width=0.08,
        plot_top_k=5,
        plot_scatter=False,
        plot_hulls=False,
    )

    arrows_dict = {}
    for i, val in enumerate(arrow_labels):
        arrows_dict[val] = arrows[i]
    for i, val in enumerate(arrow_labels1):
        arrows_dict[val] = arrows1[i]
    for i, val in enumerate(arrow_labels2):
        arrows_dict[val] = arrows2[i]

    print(len(list(arrows_dict.keys())))

    hatches = (
        [plt.plot([], marker="", ls="")[0]] * 3
        + list(arrows_dict.values())[0:2]
        + [scs[0]]
        + list(arrows_dict.values())[2:4]
        + [scs[3]]
        + list(arrows_dict.values())[4:6]
        + [scs[1]]
        + list(arrows_dict.values())[6:8]
        + [scs[2]]
        + list(arrows_dict.values())[8:10]
        + [plt.plot([], marker="", ls="")[0]]
        + list(arrows_dict.values())[10:12]
        + [plt.plot([], marker="", ls="")[0]]
        + list(arrows_dict.values())[12:14]
        + [plt.plot([], marker="", ls="")[0]]
        + list(arrows_dict.values())[14:]
        + [plt.plot([], marker="", ls="")[0]]
    )

    labels = (
        ["Factors:", " ", "Labels (Disability):"]
        + list(arrows_dict.keys())[0:2]
        + ["Level 1"]
        + list(arrows_dict.keys())[2:4]
        + ["Level 2"]
        + list(arrows_dict.keys())[4:6]
        + ["Level 3"]
        + list(arrows_dict.keys())[6:8]
        + ["Level 4"]
        + list(arrows_dict.keys())[8:10]
        + [" "]
        + list(arrows_dict.keys())[10:12]
        + [" "]
        + list(arrows_dict.keys())[12:14]
        + [" "]
        + list(arrows_dict.keys())[14:]
        + [" "]
    )

    leg = axi[2].legend(
        hatches,
        labels,
        loc="lower center",
        bbox_to_anchor=(-0.84, 1.12),
        fontsize=7,
        ncol=9,
        markerscale=1,
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

    axi[2].set_yticks([])
    axi[2].set_xticks([])
    axi[2].set_ylabel("t-SNE2", size=8)
    axi[2].set_xlabel("t-SNE1", size=8)
    axi[2].set_title("Inter-group clock", size=8)
    axi[2].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[2].xaxis.set_label_coords(x=0.5, y=-0.02)

    plt.subplots_adjust(
        left=0.02,
        right=0.99,
        top=0.7,
        bottom=0.06,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/support/support_3.pdf")


def test_between_all_3():
    colors = [
        "aliceblue",
        "antiquewhite",
        "aqua",
        "aquamarine",
        "azure",
        "beige",
        "bisque",
        "black",
        "blanchedalmond",
        "blue",
        "blueviolet",
        "brown",
        "burlywood",
        "cadetblue",
        "chartreuse",
        "chocolate",
        "coral",
        "cornflowerblue",
        "cornsilk",
        "crimson",
        "cyan",
        "darkblue",
        "darkcyan",
        "darkgoldenrod",
        "darkgray",
        "darkgreen",
        "darkgrey",
        "darkkhaki",
        "darkmagenta",
        "darkolivegreen",
        "darkorange",
        "darkorchid",
        "darkred",
        "darksalmon",
        "darkseagreen",
        "darkslateblue",
        "darkslategray",
        "darkslategrey",
        "darkturquoise",
        "darkviolet",
        "deeppink",
        "deepskyblue",
        "dimgray",
        "dimgrey",
        "dodgerblue",
        "firebrick",
        "floralwhite",
        "forestgreen",
        "fuchsia",
        "gainsboro",
        "ghostwhite",
        "gold",
        "goldenrod",
        "gray",
        "green",
        "greenyellow",
        "grey",
        "honeydew",
        "hotpink",
        "indianred",
        "indigo",
        "ivory",
        "khaki",
        "lavender",
        "lavenderblush",
        "lawngreen",
        "lemonchiffon",
        "lightblue",
        "lightcoral",
        "lightcyan",
        "lightgoldenrodyellow",
        "lightgray",
        "lightgreen",
        "lightgrey",
        "lightpink",
        "lightsalmon",
        "lightseagreen",
        "lightskyblue",
        "lightslategray",
        "lightslategrey",
        "lightsteelblue",
        "lightyellow",
        "lime",
        "limegreen",
        "linen",
        "magenta",
        "maroon",
        "mediumaquamarine",
        "mediumblue",
        "mediumorchid",
        "mediumpurple",
        "mediumseagreen",
        "mediumslateblue",
        "mediumspringgreen",
        "mediumturquoise",
        "mediumvioletred",
        "midnightblue",
        "mintcream",
        "mistyrose",
        "moccasin",
        "navajowhite",
        "navy",
        "oldlace",
        "olive",
        "olivedrab",
        "orange",
        "orangered",
        "orchid",
        "palegoldenrod",
        "palegreen",
        "paleturquoise",
        "palevioletred",
        "papayawhip",
        "peachpuff",
        "peru",
        "pink",
        "plum",
        "powderblue",
        "purple",
        "rebeccapurple",
        "red",
        "rosybrown",
        "royalblue",
        "saddlebrown",
        "salmon",
        "sandybrown",
        "seagreen",
        "seashell",
        "sienna",
        "silver",
        "skyblue",
        "slateblue",
        "slategray",
        "slategrey",
        "snow",
        "springgreen",
        "steelblue",
        "tan",
        "teal",
        "thistle",
        "tomato",
        "turquoise",
        "violet",
        "wheat",
        "white",
        "whitesmoke",
        "yellow",
        "yellowgreen",
    ]
    # tab20 = [
    #     '#1f77b4', '#aec7e8',
    #     '#ff7f0e', '#ffbb78',
    #     '#2ca02c', '#98df8a',
    #     '#d62728', '#ff9896',
    #     '#9467bd', '#c5b0d5',
    #     '#8c564b', '#c49c94',
    #     '#e377c2', '#f7b6d2',
    #     '#7f7f7f', '#c7c7c7',
    #     '#bcbd22', '#dbdb8d',
    #     '#17becf', '#9edae5']

    tab20 = [
        "#1f77b4",
        "#bcbd22",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "orangered",
        "#d62728",
        "darkolivegreen",
        "#17becf",
        "navy",
    ]

    X_new, obs, standard_embedding, labels, clusters = setup_support_data(
        method="tsne")

    i = 0
    for k in range(len(obs)):
        if obs[k] in [
            "Disease",
            "Avg. TISS score",
            "APACHE3 score",
            "Cost/charges ratio",
            "Diabetes",
            "Dementia",
            "ADL family",
            "ADL patient",
            "Bilirubin",
            "Coma score",
            "Physiology score",
            "2 month survival",
            "Study enter",
            "Micro cost",
        ]:

            if obs[k] == "APACHE3 score":
                colors[k] = "purple"
            elif obs[k] == "Study enter":
                colors[k] = "peru"
            elif obs[k] == "Physiology score":
                colors[k] = "dimgray"
            elif obs[k] == "Cost/charges ratio":
                colors[k] = "#ff7f0e"
            else:
                colors[k] = tab20[i]
                i += 1

    fig = plt.figure(
        constrained_layout=True,
        figsize=((7.125 - 0.17), ((7.125 - 0.17) / 1.65) / 1.618),
        dpi=1000,
        facecolor="w",
        edgecolor="k",
    )
    spec2 = gridspec.GridSpec(
        ncols=4,
        nrows=1,
        figure=fig,
        left=0.02,
        right=0.99,
        top=0.67,
        bottom=0.06)
    ax1 = fig.add_subplot(spec2[0])
    ax3 = fig.add_subplot(spec2[2])
    ax4 = fig.add_subplot(spec2[3])
    axi = [ax1, ax3, ax4]

    spec23 = gridspec.GridSpecFromSubplotSpec(
        2, 2, subplot_spec=spec2[1], hspace=0.18, wspace=0.05
    )  # wspace=-0.5, hspace=-0.9)
    ax21 = fig.add_subplot(spec23[0, 0])
    ax22 = fig.add_subplot(spec23[0, 1])
    ax23 = fig.add_subplot(spec23[1, 0])
    ax24 = fig.add_subplot(spec23[1, 1])

    scs = []
    for val, i in zip([0, 1], [2, 8]):
        sc = axi[0].scatter(
            standard_embedding[labels == val, 0],
            standard_embedding[labels == val, 1],
            marker=".",
            color=matplotlib.colormaps["Paired"].colors[i],
            alpha=0.2,
            s=15,
        )
        scs.append(sc)
        axi[1].scatter(
            standard_embedding[labels == val, 0],
            standard_embedding[labels == val, 1],
            marker=".",
            color=matplotlib.colormaps["Paired"].colors[i],
            alpha=0.2,
            s=15,
        )
        axi[2].scatter(
            standard_embedding[labels == val, 0],
            standard_embedding[labels == val, 1],
            marker=".",
            color=matplotlib.colormaps["Paired"].colors[i],
            alpha=0.2,
            s=15,
        )

    plot_inst = NonLinearClock(
        X_new,
        obs,
        standard_embedding,
        labels,
        method="UMAP",
        cluster_labels=labels,
        color_scheme=colors,
    )
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=False,
        standartize_coef=False,
        biggest_arrow_method=True,
        univar_importance=True,
        ax=axi[0],
        scale_circle=2,
        move_circle=[0, 0],
        annotate=0.8,
        arrow_width=0.1,
        plot_scatter=False,
        plot_top_k=4,
    )

    axi[0].set_yticks([])
    axi[0].set_xticks([])
    axi[0].set_ylabel("t-SNE2", size=8)
    axi[0].set_xlabel("t-SNE1", size=8)
    axi[0].set_title("Global clock", size=8)
    axi[0].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[0].xaxis.set_label_coords(x=0.5, y=-0.02)

    # Local
    arrows1, arrow_labels1 = plot_inst.plot_local_clocks(
        standartize_data=False,
        standartize_coef=False,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=axi[1],
        scale_circles=[1.5, 1.5],
        move_circles=[[-0.3, 0.5], [0.0, -0.5]],
        clocks_labels=["Survival", "Death"],
        clocks_labels_angles=[45, 90],
        annotates=[0.3, 0.3],
        arrow_width=0.08,
        plot_top_k=4,
        plot_hulls=False,
        plot_scatter=False,
    )

    axi[1].set_yticks([])
    axi[1].set_xticks([])
    axi[1].set_ylabel("t-SNE2", size=8)
    axi[1].set_xlabel("t-SNE1", size=8)
    axi[1].set_title("Local clock", size=8)
    axi[1].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[1].xaxis.set_label_coords(x=0.5, y=-0.02)

    # Between
    arrows2, arrow_labels2 = plot_inst.plot_between_clock(
        standartize_data=True,
        standartize_coef=False,
        univar_importance=False,
        ax=axi[2],
        scale_circles=[2],
        move_circles=[[0, 0]],
        annotates=[0.5],
        clocks_labels=["Survival", "Death"],
        clocks_labels_angles=[90],
        arrow_width=0.08,
        plot_top_k=4,
        plot_scatter=False,
        plot_hulls=False,
    )

    arrows_dict = {}
    for i, val in enumerate(arrow_labels):
        arrows_dict[val] = arrows[i]
    for i, val in enumerate(arrow_labels1):
        arrows_dict[val] = arrows1[i]
    for i, val in enumerate(arrow_labels2):
        arrows_dict[val] = arrows2[i]

    print(len(list(arrows_dict.keys())))
    print((list(arrows_dict.keys())))

    hatches = (
        [plt.plot([], marker="", ls="")[0]] * 4
        + list(arrows_dict.values())[0:3]
        + [scs[0]]
        + list(arrows_dict.values())[3:6]
        + [scs[1]]
        + list(arrows_dict.values())[6:9]
        + [plt.plot([], marker="", ls="")[0]]
        + list(arrows_dict.values())[9:12]
        + [plt.plot([], marker="", ls="")[0]]
    )
    labels = (
        ["Factors:", " ", " ", "Labels (Death in hospital):"]
        + list(arrows_dict.keys())[0:3]
        + ["Survival"]
        + list(arrows_dict.keys())[3:6]
        + ["Death"]
        + list(arrows_dict.keys())[6:9]
        + [""]
        + list(arrows_dict.keys())[9:12]
        + [""]
    )
    leg = axi[2].legend(
        hatches,
        labels,
        loc="lower center",
        bbox_to_anchor=(-1.29, 1.12),
        fontsize=7,
        ncol=5,
        markerscale=1,
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
    axi[2].set_ylabel("t-SNE2", size=8)
    axi[2].set_xlabel("t-SNE1", size=8)
    axi[2].set_title("Inter-group clock", size=8)
    axi[2].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[2].xaxis.set_label_coords(x=0.5, y=-0.02)

    # fourth plot
    X_new, obs, standard_embedding, labels, clusters = setup_support_data(
        method="tsne", drop_labels=True
    )

    standard_embedding[:, 0], standard_embedding[:, 1] = (
        5 * standard_embedding[:, 0],
        1 * standard_embedding[:, 1],
    )
    for (
        i, o), axi_i in zip(
        enumerate(arrow_labels), [
            ax21, ax22, ax23, ax24]):
        if i == 0:
            im = axi_i.scatter(
                standard_embedding[:, 0],
                standard_embedding[:, 1],
                marker=".",
                s=1,
                c=X_new[o],
                cmap=cm.coolwarm,
                alpha=0.8,
            )
        else:
            axi_i.scatter(
                standard_embedding[:, 0],
                standard_embedding[:, 1],
                marker=".",
                s=1,
                c=X_new[o],
                cmap=cm.coolwarm,
                alpha=0.8,
            )

        axi_i.set_yticks([])
        axi_i.set_xticks([])

        names = {
            "Disease": "Disease",
            "Avg. TISS score": "Avg. TISS",
            "APACHE3 score": "APACHE3",
            "Cost/charges ratio": "Cost/charges",
        }

        axi_i.set_title(names[o], size=7, pad=-14)

    ax21.set_ylabel("t-SNE2", size=8)
    ax23.set_xlabel("t-SNE1", size=8)

    ax21.yaxis.set_label_coords(0, -0.1)
    ax23.xaxis.set_label_coords(1.1, -0.04)

    cbar = fig.colorbar(im, ax=[ax21, ax22, ax23, ax24], pad=0.03, aspect=40)
    cbar.ax.tick_params(
        labelsize=5, pad=0.2, length=0.8, grid_linewidth=0.1
    )  # labelrotation=90,
    cbar.outline.set_visible(False)

    plt.savefig("plots/paper/support/support_3.pdf")


def test_between_all_3_5():
    colors = [
        "aliceblue",
        "antiquewhite",
        "aqua",
        "aquamarine",
        "azure",
        "beige",
        "bisque",
        "black",
        "blanchedalmond",
        "blue",
        "blueviolet",
        "brown",
        "burlywood",
        "cadetblue",
        "chartreuse",
        "chocolate",
        "coral",
        "cornflowerblue",
        "cornsilk",
        "crimson",
        "cyan",
        "darkblue",
        "darkcyan",
        "darkgoldenrod",
        "darkgray",
        "darkgreen",
        "darkgrey",
        "darkkhaki",
        "darkmagenta",
        "darkolivegreen",
        "darkorange",
        "darkorchid",
        "darkred",
        "darksalmon",
        "darkseagreen",
        "darkslateblue",
        "darkslategray",
        "darkslategrey",
        "darkturquoise",
        "darkviolet",
        "deeppink",
        "deepskyblue",
        "dimgray",
        "dimgrey",
        "dodgerblue",
        "firebrick",
        "floralwhite",
        "forestgreen",
        "fuchsia",
        "gainsboro",
        "ghostwhite",
        "gold",
        "goldenrod",
        "gray",
        "green",
        "greenyellow",
        "grey",
        "honeydew",
        "hotpink",
        "indianred",
        "indigo",
        "ivory",
        "khaki",
        "lavender",
        "lavenderblush",
        "lawngreen",
        "lemonchiffon",
        "lightblue",
        "lightcoral",
        "lightcyan",
        "lightgoldenrodyellow",
        "lightgray",
        "lightgreen",
        "lightgrey",
        "lightpink",
        "lightsalmon",
        "lightseagreen",
        "lightskyblue",
        "lightslategray",
        "lightslategrey",
        "lightsteelblue",
        "lightyellow",
        "lime",
        "limegreen",
        "linen",
        "magenta",
        "maroon",
        "mediumaquamarine",
        "mediumblue",
        "mediumorchid",
        "mediumpurple",
        "mediumseagreen",
        "mediumslateblue",
        "mediumspringgreen",
        "mediumturquoise",
        "mediumvioletred",
        "midnightblue",
        "mintcream",
        "mistyrose",
        "moccasin",
        "navajowhite",
        "navy",
        "oldlace",
        "olive",
        "olivedrab",
        "orange",
        "orangered",
        "orchid",
        "palegoldenrod",
        "palegreen",
        "paleturquoise",
        "palevioletred",
        "papayawhip",
        "peachpuff",
        "peru",
        "pink",
        "plum",
        "powderblue",
        "purple",
        "rebeccapurple",
        "red",
        "rosybrown",
        "royalblue",
        "saddlebrown",
        "salmon",
        "sandybrown",
        "seagreen",
        "seashell",
        "sienna",
        "silver",
        "skyblue",
        "slateblue",
        "slategray",
        "slategrey",
        "snow",
        "springgreen",
        "steelblue",
        "tan",
        "teal",
        "thistle",
        "tomato",
        "turquoise",
        "violet",
        "wheat",
        "white",
        "whitesmoke",
        "yellow",
        "yellowgreen",
    ]
    # tab20 = [
    #     '#1f77b4', '#aec7e8',
    #     '#ff7f0e', '#ffbb78',
    #     '#2ca02c', '#98df8a',
    #     '#d62728', '#ff9896',
    #     '#9467bd', '#c5b0d5',
    #     '#8c564b', '#c49c94',
    #     '#e377c2', '#f7b6d2',
    #     '#7f7f7f', '#c7c7c7',
    #     '#bcbd22', '#dbdb8d',
    #     '#17becf', '#9edae5']

    tab20 = [
        "#1f77b4",
        "#ff7f0e",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "navy",
        "#8c564b",
        "dimgray",
        "#e377c2",
        "darkolivegreen",
        "#bcbd22",
        "#17becf",
        "chartreuse",
    ]

    X_new, obs, standard_embedding, labels, clusters = setup_support_data(
        method="tsne")

    i = 0
    for k in range(len(obs)):
        if obs[k] in [
            "surv2m",
            "surv6m",
            "avtisst",
            "dzclass",
            "aps",
            "death",
            "sps",
            "dnr",
            "adlsc",
            "prg6m",
            "prg2m",
            "d.time",
            "num.co",
        ]:
            colors[k] = tab20[i]
            if obs[k] == "surv6m":
                colors[k] = "darkred"
            if obs[k] == "num.co":
                colors[k] = "orangered"
            if obs[k] == "adlsc":
                colors[k] = "peru"
            i += 1

    fig, axi = plt.subplots(
        1, 3, figsize=((7.125 - 0.17), ((7.125 - 0.17) / 1.8) / 1.618)
    )
    plt.tight_layout()

    scs = []
    # for val, i in zip([0, 1, 2, 3], [0, 5, 2, 1]):
    for val, i in zip([0, 3, 1, 2], [0, 2, 5, 1]):
        sc = axi[0].scatter(
            standard_embedding[labels == val, 0],
            standard_embedding[labels == val, 1],
            marker=".",
            color=matplotlib.colormaps["Accent"].colors[i],
            alpha=0.3,
        )
        scs.append(sc)
        axi[1].scatter(
            standard_embedding[labels == val, 0],
            standard_embedding[labels == val, 1],
            marker=".",
            color=matplotlib.colormaps["Accent"].colors[i],
            alpha=0.3,
        )
        axi[2].scatter(
            standard_embedding[labels == val, 0],
            standard_embedding[labels == val, 1],
            marker=".",
            color=matplotlib.colormaps["Accent"].colors[i],
            alpha=0.3,
        )

    plot_inst = NonLinearClock(
        X_new,
        obs,
        standard_embedding,
        labels,
        method="UMAP",
        cluster_labels=labels,
        color_scheme=colors,
    )
    arrows, arrow_labels = plot_inst.plot_global_clock(
        standartize_data=False,
        standartize_coef=False,
        biggest_arrow_method=True,
        univar_importance=True,
        ax=axi[0],
        scale_circle=3,
        move_circle=[0, 0],
        annotate=0.8,
        arrow_width=0.1,
        plot_scatter=False,
        plot_top_k=5,
    )

    axi[0].set_yticks([])
    axi[0].set_xticks([])
    axi[0].set_ylabel("t-SNE2", size=8)
    axi[0].set_xlabel("t-SNE1", size=8)
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
        standartize_coef=False,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=axi[1],
        scale_circles=[2.2, 2, 2.5, 0.2],
        # move_circles=[[-1.5, 0.3], [0, 1.5], [2.3, -1], [-0.8, -0.2]],
        move_circles=[[-1.8, 0.3], [-0.8, 0], [2.5, -0.8], [-0.0, 0]],
        annotates=[0.4, 0.4, 0.4, 0.4],
        arrow_width=0.08,
        plot_top_k=5,
        plot_hulls=False,
        plot_scatter=False,
    )

    # sc2 = axi[1].scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.2)

    axi[1].set_yticks([])
    axi[1].set_xticks([])
    axi[1].set_ylabel("t-SNE2", size=8)
    axi[1].set_xlabel("t-SNE1", size=8)
    axi[1].set_title("Local clock", size=8)
    axi[1].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[1].xaxis.set_label_coords(x=0.5, y=-0.02)

    # Between
    # sc3 = axi[2].scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.2)

    arrows2, arrow_labels2 = plot_inst.plot_between_clock(
        standartize_data=True,
        standartize_coef=False,
        univar_importance=False,
        ax=axi[2],
        scale_circles=[6, 10, 0],
        # move_circles=[[1.8, 1.1], [0.1, -0.8], [0.0, 0.0]],
        move_circles=[[0.7, -1.5], [0.4, 1.3], [0, 0]],
        annotates=[0.5, 0.5, 0.5],
        arrow_width=0.08,
        plot_top_k=5,
        plot_scatter=False,
        plot_hulls=False,
    )

    axi[2].set_yticks([])
    axi[2].set_xticks([])
    axi[2].set_ylabel("t-SNE2", size=8)
    axi[2].set_xlabel("t-SNE1", size=8)
    axi[2].set_title("Inter-group clock", size=8)
    axi[2].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[2].xaxis.set_label_coords(x=0.5, y=-0.02)

    arrows_dict = {}
    for i, val in enumerate(arrow_labels):
        arrows_dict[val] = arrows[i]
    for i, val in enumerate(arrow_labels1):
        arrows_dict[val] = arrows1[i]
    for i, val in enumerate(arrow_labels2):
        arrows_dict[val] = arrows2[i]

    print(len(list(arrows_dict.keys())))

    hatches = (
        [plt.plot([], marker="", ls="")[0]] * 3
        + list(arrows_dict.values())[0:2]
        + [scs[0]]
        + list(arrows_dict.values())[2:4]
        + [scs[3]]
        + list(arrows_dict.values())[4:6]
        + [scs[1]]
        + list(arrows_dict.values())[6:8]
        + [scs[2]]
        + list(arrows_dict.values())[8:10]
        + [plt.plot([], marker="", ls="")[0]]
        + list(arrows_dict.values())[10:12]
        + [plt.plot([], marker="", ls="")[0]]
        + list(arrows_dict.values())[12:14]
        + [plt.plot([], marker="", ls="")[0]]
        + list(arrows_dict.values())[14:]
        + [plt.plot([], marker="", ls="")[0]]
    )

    labels = (
        ["Factors:", " ", "Labels (Disability):"]
        + list(arrows_dict.keys())[0:2]
        + ["Level 1"]
        + list(arrows_dict.keys())[2:4]
        + ["Level 2"]
        + list(arrows_dict.keys())[4:6]
        + ["Level 3"]
        + list(arrows_dict.keys())[6:8]
        + ["Level 4"]
        + list(arrows_dict.keys())[8:10]
        + [" "]
        + list(arrows_dict.keys())[10:12]
        + [" "]
        + list(arrows_dict.keys())[12:14]
        + [" "]
        + list(arrows_dict.keys())[14:]
        + [" "]
    )

    leg = axi[2].legend(
        hatches,
        labels,
        loc="lower center",
        bbox_to_anchor=(-0.84, 1.12),
        fontsize=7,
        ncol=9,
        markerscale=1,
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

    plt.subplots_adjust(
        left=0.02,
        right=0.99,
        top=0.7,
        bottom=0.06,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/support/support_3.pdf")


def test_local_hdbscan():
    colors = [
        "aliceblue",
        "antiquewhite",
        "aqua",
        "aquamarine",
        "azure",
        "beige",
        "bisque",
        "black",
        "blanchedalmond",
        "blue",
        "blueviolet",
        "brown",
        "burlywood",
        "cadetblue",
        "chartreuse",
        "chocolate",
        "coral",
        "cornflowerblue",
        "cornsilk",
        "crimson",
        "cyan",
        "darkblue",
        "darkcyan",
        "darkgoldenrod",
        "darkgray",
        "darkgreen",
        "darkgrey",
        "darkkhaki",
        "darkmagenta",
        "darkolivegreen",
        "darkorange",
        "darkorchid",
        "darkred",
        "darksalmon",
        "darkseagreen",
        "darkslateblue",
        "darkslategray",
        "darkslategrey",
        "darkturquoise",
        "darkviolet",
        "deeppink",
        "deepskyblue",
        "dimgray",
        "dimgrey",
        "dodgerblue",
        "firebrick",
        "floralwhite",
        "forestgreen",
        "fuchsia",
        "gainsboro",
        "ghostwhite",
        "gold",
        "goldenrod",
        "gray",
        "green",
        "greenyellow",
        "grey",
        "honeydew",
        "hotpink",
        "indianred",
        "indigo",
        "ivory",
        "khaki",
        "lavender",
        "lavenderblush",
        "lawngreen",
        "lemonchiffon",
        "lightblue",
        "lightcoral",
        "lightcyan",
        "lightgoldenrodyellow",
        "lightgray",
        "lightgreen",
        "lightgrey",
        "lightpink",
        "lightsalmon",
        "lightseagreen",
        "lightskyblue",
        "lightslategray",
        "lightslategrey",
        "lightsteelblue",
        "lightyellow",
        "lime",
        "limegreen",
        "linen",
        "magenta",
        "maroon",
        "mediumaquamarine",
        "mediumblue",
        "mediumorchid",
        "mediumpurple",
        "mediumseagreen",
        "mediumslateblue",
        "mediumspringgreen",
        "mediumturquoise",
        "mediumvioletred",
        "midnightblue",
        "mintcream",
        "mistyrose",
        "moccasin",
        "navajowhite",
        "navy",
        "oldlace",
        "olive",
        "olivedrab",
        "orange",
        "orangered",
        "orchid",
        "palegoldenrod",
        "palegreen",
        "paleturquoise",
        "palevioletred",
        "papayawhip",
        "peachpuff",
        "peru",
        "pink",
        "plum",
        "powderblue",
        "purple",
        "rebeccapurple",
        "red",
        "rosybrown",
        "royalblue",
        "saddlebrown",
        "salmon",
        "sandybrown",
        "seagreen",
        "seashell",
        "sienna",
        "silver",
        "skyblue",
        "slateblue",
        "slategray",
        "slategrey",
        "snow",
        "springgreen",
        "steelblue",
        "tan",
        "teal",
        "thistle",
        "tomato",
        "turquoise",
        "violet",
        "wheat",
        "white",
        "whitesmoke",
        "yellow",
        "yellowgreen",
    ]

    tab20 = [
        "#1f77b4",
        "#bcbd22",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "orangered",
        "#d62728",
        "darkolivegreen",
        "#17becf",
        "navy",
    ]

    color_dict = {
        "Disease": "#1f77b4",
        "Coma score": "#bcbd22",
        "Micro cost": "#9467bd",
        "Avg. TISS score": "#8c564b",
        "Diabetes": "#e377c2",
        "Dementia": "orangered",
        "2 month survival": "#d62728",
        "Bilirubin": "darkolivegreen",
        "ADL patient": "#17becf",
        "ADL family": "navy",
    }

    X_new, obs, standard_embedding, labels, clusters = setup_support_data(
        method="tsne")

    i = 0
    for k in range(len(obs)):
        if obs[k] in [
            "Disease",
            "Avg. TISS score",
            "APACHE3 score",
            "Cost/charges ratio",
            "Diabetes",
            "Dementia",
            "ADL family",
            "ADL patient",
            "Bilirubin",
            "Coma score",
            "Physiology score",
            "2 month survival",
            "Study enter",
            "Micro cost",
        ]:

            if obs[k] == "APACHE3 score":
                colors[k] = "purple"
            elif obs[k] == "Study enter":
                colors[k] = "peru"
            elif obs[k] == "Physiology score":
                colors[k] = "dimgray"
            elif obs[k] == "Cost/charges ratio":
                colors[k] = "#ff7f0e"
            else:
                colors[k] = tab20[i]
                i += 1
    fig_size = ((7.125 - 0.17) / 2, ((7.125 - 0.17) / 2.3) / 1.618)
    fig = plt.figure(
        constrained_layout=True,
        figsize=fig_size,
        dpi=1000,
        facecolor="w",
        edgecolor="k",
    )
    spec2 = gridspec.GridSpec(
        ncols=3,
        nrows=1,
        figure=fig,
        left=0.04,
        right=0.99,
        top=0.63,
        bottom=0.06)
    ax1 = fig.add_subplot(spec2[0])
    ax2 = fig.add_subplot(spec2[1])

    spec23 = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=spec2[2], hspace=0.25, wspace=0.08
    )  # wspace=-0.5, hspace=-0.9)
    ax11 = fig.add_subplot(spec23[0, 0])
    ax12 = fig.add_subplot(spec23[0, 1])
    ax13 = fig.add_subplot(spec23[0, 2])
    ax21 = fig.add_subplot(spec23[1, 0])
    ax23 = fig.add_subplot(spec23[1, 1])
    ax22 = fig.add_subplot(spec23[1, 2])

    scs = []
    for val, i in zip([-1, 0, 1], [-1, 0, 6]):
        if val == -1:
            sc = ax1.scatter(
                standard_embedding[clusters == val, 0],
                standard_embedding[clusters == val, 1],
                marker=".",
                color="gray",
                alpha=0.2,
                s=13,
            )
            sc = ax2.scatter(
                standard_embedding[clusters == val, 0],
                standard_embedding[clusters == val, 1],
                marker=".",
                color="gray",
                alpha=0.2,
                s=13,
            )
        else:
            sc = ax1.scatter(
                standard_embedding[clusters == val, 0],
                standard_embedding[clusters == val, 1],
                marker=".",
                color=matplotlib.colormaps["Paired"].colors[i],
                alpha=0.2,
                s=13,
            )
            sc = ax2.scatter(
                standard_embedding[clusters == val, 0],
                standard_embedding[clusters == val, 1],
                marker=".",
                color=matplotlib.colormaps["Paired"].colors[i],
                alpha=0.2,
                s=13,
            )
        scs.append(sc)

    plot_inst = NonLinearClock(
        X_new,
        obs,
        standard_embedding,
        clusters,
        method="UMAP",
        cluster_labels=clusters,
        color_scheme=colors,
    )

    # Local
    arrows1, arrow_labels1 = plot_inst.plot_local_clocks(
        standartize_data=False,
        standartize_coef=False,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=ax1,
        scale_circles=[2.5, 15],
        move_circles=[[0, 0], [0, 0]],
        clocks_labels=["0", "1"],
        clocks_labels_angles=[45, 315],
        annotates=[0.5, 0.5],
        arrow_width=0.08,
        plot_top_k=4,
        plot_hulls=False,
        plot_scatter=False,
    )

    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_ylabel("t-SNE2", size=8)
    ax1.set_xlabel("t-SNE1", size=8)
    ax1.set_title("Local clock", size=8)
    ax1.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax1.xaxis.set_label_coords(x=0.5, y=-0.02)

    # Between
    arrows2, arrow_labels2 = plot_inst.plot_between_clock(
        standartize_data=False,
        standartize_coef=False,
        univar_importance=False,
        ax=ax2,
        scale_circles=[1],
        move_circles=[[0, 0]],
        annotates=[0.5],
        arrow_width=0.08,
        plot_top_k=4,
        plot_scatter=False,
        plot_hulls=False,
    )

    ax2.set_yticks([])
    ax2.set_xticks([])
    ax2.set_ylabel("t-SNE2", size=8)
    ax2.set_xlabel("t-SNE1", size=8)
    ax2.set_title("Inter-group clock", size=8)
    ax2.yaxis.set_label_coords(x=-0.01, y=0.5)
    ax2.xaxis.set_label_coords(x=0.5, y=-0.02)

    arrows_dict = {}
    for i, val in enumerate(arrow_labels1):
        arrows_dict[val] = arrows1[i]
    for i, val in enumerate(arrow_labels2):
        arrows_dict[val] = arrows2[i]

    dict_vals = list(arrows_dict.values())
    dict_keys = list(arrows_dict.keys())
    tmp = dict_vals[-2]
    dict_vals[-2] = dict_vals[-1]
    dict_vals[-1] = tmp
    tmp = dict_keys[-2]
    dict_keys[-2] = dict_keys[-1]
    dict_keys[-1] = tmp

    print(dict_keys)
    print(len(dict_keys))

    hatches = (
        [plt.plot([], marker="", ls="")[0]] * 3
        + dict_vals[0:2]
        + [scs[1]]
        + dict_vals[2:4]
        + [scs[2]]
        + dict_vals[4:6]
        + [scs[0]]
    )

    labels = (
        ["Factors:", " ", "Labels:"]
        + dict_keys[0:2]
        + ["0"]
        + dict_keys[2:4]
        + ["1"]
        + dict_keys[4:6]
        + ["No class"]
    )

    leg = ax12.legend(
        hatches,
        labels,
        loc="lower center",
        bbox_to_anchor=(-3.9, 1.35),
        fontsize=7,
        ncol=4,
        markerscale=1,
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

    # fourth plot
    standard_embedding[:, 0], standard_embedding[:, 1] = (
        20 * standard_embedding[:, 0],
        1 * standard_embedding[:, 1],
    )
    for (i, o), axi_i in zip(
        enumerate(dict_keys), [ax11, ax12, ax13, ax21, ax22, ax23]
    ):
        if i == 0:
            im = axi_i.scatter(
                standard_embedding[:, 0],
                standard_embedding[:, 1],
                marker=".",
                s=1,
                c=X_new[o],
                cmap=cm.coolwarm,
                alpha=0.8,
            )
        else:
            axi_i.scatter(
                standard_embedding[:, 0],
                standard_embedding[:, 1],
                marker=".",
                s=1,
                c=X_new[o],
                cmap=cm.coolwarm,
                alpha=0.8,
            )

        axi_i.set_yticks([])
        axi_i.set_xticks([])

        names = {
            "Disease": "Dis.",
            "ADL family": "ADL f.",
            "ADL patient": "ADL p.",
            "Dementia": "Dem.",
            "Diabetes": "Diab.",
            "Charges": "Charges",
            "Study enter": "St. e.",
        }

        axi_i.set_title(names[o], size=5, pad=-14)

    ax21.set_ylabel("t-SNE2", size=8)
    ax22.set_xlabel("t-SNE1", size=8)

    ax21.yaxis.set_label_coords(0, 1.15)
    ax22.xaxis.set_label_coords(-0.5, -0.04)

    # ticks=[0, 5, 10, 15])
    cbar = fig.colorbar(
        im,
        ax=[ax11, ax12, ax13, ax21, ax22, ax23],
        pad=0.03,
        aspect=40,
    )
    cbar.ax.tick_params(
        labelsize=5, pad=0.2, length=0.8, grid_linewidth=0.1
    )  # labelrotation=90,
    cbar.outline.set_visible(False)

    plt.savefig("plots/paper/support/support_local_hdbscan.pdf")


test_between_all_3()
test_local_hdbscan()
