from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from src.nonlinear_clock.plot import NonLinearClock
import umap
from sklearn.cluster import HDBSCAN
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
from sklearn import manifold, preprocessing

def make_legend_arrow(legend, orig_handle,
                      xdescent, ydescent,
                      width, height, fontsize):
    p = mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.75*height )
    return p



def read_data(path):
    return pd.read_csv(path, header=0)


def setup_support_data(method="tsne", drop_labels=True):
    file_name = "/Users/olga_ovcharenko/Documents/ETH/FS23/ResearchProject/non_lin_visualization/data/support2.csv"
    X = read_data(file_name)

    X.drop(columns=["id"], inplace=True)

    X.drop(columns=["hospdead"], inplace=True)

    for c in X.columns:
        if c in ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca', 'dnr']:
            X[c] = X[c].fillna(X[c].mode())
        elif c == 'sfdm2':
            pass
        else: 
            X[c] = X[c].fillna(X[c].mean())

    for c in ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca', 'dnr']:
        X[c], _ = pd.factorize(X[c])
    
    X['sfdm2'] = X['sfdm2'].map({"no(M2 and SIP pres)": 0, "adl>=4 (>=5 if sur)": 1, "SIP>=30": 2, "Coma or Intub": 4, "<2 mo. follow-up": 3})
    X = X[X.sfdm2 != 4]
    X = X.dropna(subset=['sfdm2'])
    
    labels = X["sfdm2"]

    if drop_labels:
        X.drop(columns=["sfdm2"], inplace=True)
    obs = list(X.columns)

    # X.drop(columns=["diabetes", "ca", "bun", "glucose", "sex"], inplace=True)
    obs = list(X.columns)

    for col in X.columns:
        X[col] = (X[col] - X[col].mean()) / X[col].std()
    
    # compute umap
    if method == "umap":
        reducer = umap.UMAP(min_dist=0.2, n_neighbors=30, random_state=42)
        if not drop_labels:
            K = X.drop(columns=["sfdm2"], inplace=False)
            standard_embedding = reducer.fit_transform(K)
        else:
            standard_embedding = reducer.fit_transform(X)

    elif method == "tsne":
        tsne = manifold.TSNE(n_components = 2, random_state=42, perplexity=1, learning_rate='auto')
        standard_embedding = tsne.fit_transform(X)

    elif method == "phate":
        raise NotImplementedError()
    
    standard_embedding = np.array(standard_embedding)
    labels = np.array(labels)
    for i in range(standard_embedding.shape[1]):
        standard_embedding[:, i] = (standard_embedding[:, i] - standard_embedding[:, i].mean()) / standard_embedding[:, i].std()
    
    # get clusters
    clusters = HDBSCAN(min_samples=12).fit_predict(X)
    # clusters = KMeans(n_clusters=2, n_init="auto", max_iter=1000, random_state=42).fit_predict(X)

    return X, obs, standard_embedding, labels, clusters


def test_between_all_3():
    colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beige', 'bisque', 'black', 
              'blanchedalmond', 'blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse', 
              'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 
              'darkcyan', 'darkgoldenrod', 'darkgray', 'darkgreen', 'darkgrey', 'darkkhaki', 
              'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darksalmon', 
              'darkseagreen', 'darkslateblue', 'darkslategray', 'darkslategrey', 'darkturquoise', 
              'darkviolet', 'deeppink', 'deepskyblue', 'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 
              'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod', 'gray', 
              'green', 'greenyellow', 'grey', 'honeydew', 'hotpink', 'indianred', 'indigo', 'ivory', 
              'khaki', 'lavender', 'lavenderblush', 'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 
              'lightcyan', 'lightgoldenrodyellow', 'lightgray', 'lightgreen', 'lightgrey', 'lightpink', 
              'lightsalmon', 'lightseagreen', 'lightskyblue', 'lightslategray', 'lightslategrey', 
              'lightsteelblue', 'lightyellow', 'lime', 'limegreen', 'linen', 'magenta', 'maroon', 
              'mediumaquamarine', 'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen', 
              'mediumslateblue', 'mediumspringgreen', 'mediumturquoise', 'mediumvioletred', 'midnightblue', 
              'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 
              'orange', 'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise', 'palevioletred', 
              'papayawhip', 'peachpuff', 'peru', 'pink', 'plum', 'powderblue', 'purple', 'rebeccapurple', 'red', 
              'rosybrown', 'royalblue', 'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna', 
              'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow', 'springgreen', 'steelblue', 
              'tan', 'teal', 'thistle', 'tomato', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke', 'yellow', 
              'yellowgreen']
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
        '#1f77b4', 
        '#ff7f0e', 
        '#d62728', '#ff9896', 
        '#9467bd', 'navy', 
        '#8c564b', 'dimgray', 
        '#e377c2', 'darkolivegreen', 
        '#bcbd22', 
        '#17becf', "chartreuse"]

    
    X_new, obs, standard_embedding, labels, clusters = setup_support_data(method="tsne")

    i = 0
    for k in range(len(obs)): 
        if obs[k] in ['surv2m', 'surv6m', 'avtisst', 'dzclass', 'aps', 'death', 'sps', 'dnr', 'adlsc', 'prg6m', 'prg2m', 'd.time', 'num.co']:
            colors[k] = tab20[i]
            if  obs[k] == "surv6m":
                colors[k] = "darkred"
            if  obs[k] == "num.co":
                colors[k] = "orangered"
            if obs[k] == "adlsc":
                colors[k] = "peru"
            i += 1

    # print(np.unique(labels, return_counts=True))
    fig, axi = plt.subplots(1, 3, figsize=((7.125-0.17), ((7.125-0.17)/1.8)/1.618))
    plt.tight_layout()

    # sc = axi[0].scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap="Accent", zorder=0, alpha=0.2)
    # print(sc.legend_elements())

    scs = []
    # for val, i in zip([0, 1, 2, 3], [0, 5, 2, 1]):
    for val, i in zip([0, 3, 1, 2], [0, 2, 5, 1]):
        sc = axi[0].scatter(standard_embedding[labels == val,0], standard_embedding[labels == val,1], marker= '.', color=matplotlib.colormaps["Accent"].colors[i], alpha=0.3)
        scs.append(sc)
        axi[1].scatter(standard_embedding[labels == val,0], standard_embedding[labels == val,1], marker= '.', color=matplotlib.colormaps["Accent"].colors[i], alpha=0.3)
        axi[2].scatter(standard_embedding[labels == val,0], standard_embedding[labels == val,1], marker= '.', color=matplotlib.colormaps["Accent"].colors[i], alpha=0.3)

    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, method="UMAP", cluster_labels=labels, color_scheme=colors)
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
        plot_top_k=5
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
        plot_scatter=False
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
        plot_hulls=False
    )

    arrows_dict = {}
    for i, val in enumerate(arrow_labels):
        arrows_dict[val] = arrows[i]
    for i, val in enumerate(arrow_labels1):
        arrows_dict[val] = arrows1[i]
    for i, val in enumerate(arrow_labels2):
        arrows_dict[val] = arrows2[i]

    print(len(list(arrows_dict.keys())))
    
    hatches = [plt.plot([],marker="", ls="")[0]]*3 + \
        list(arrows_dict.values())[0:2]   + [scs[0]] + \
        list(arrows_dict.values())[2:4]   + [scs[3]] + \
        list(arrows_dict.values())[4:6]   + [scs[1]] + \
        list(arrows_dict.values())[6:8]   + [scs[2]] + \
        list(arrows_dict.values())[8:10]  + [plt.plot([],marker="", ls="")[0]] + \
        list(arrows_dict.values())[10:12] + [plt.plot([],marker="", ls="")[0]] + \
        list(arrows_dict.values())[12:14] + [plt.plot([],marker="", ls="")[0]] + \
        list(arrows_dict.values())[14:]   + [plt.plot([],marker="", ls="")[0]]
    
    labels = ["Factors:", " ", "Labels (Disability):"] + \
        list(arrows_dict.keys())[0:2]   + ["Level 1"] + \
        list(arrows_dict.keys())[2:4]   + ["Level 2"] + \
        list(arrows_dict.keys())[4:6]   + ["Level 3"] + \
        list(arrows_dict.keys())[6:8]   + ["Level 4"] + \
        list(arrows_dict.keys())[8:10]  + [" "] + \
        list(arrows_dict.keys())[10:12] + [" "] + \
        list(arrows_dict.keys())[12:14] + [" "] + \
        list(arrows_dict.keys())[14:]   + [" "]

    
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
        handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow),},
    )
    for vpack in leg._legend_handle_box.get_children()[:1]:
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)

    axi[2].set_yticks([])
    axi[2].set_xticks([])
    axi[2].set_ylabel("t-SNE2", size=8)
    axi[2].set_xlabel("t-SNE1", size=8)
    axi[2].set_title("Inter-cluster clock", size=8)
    axi[2].yaxis.set_label_coords(x=-0.01, y=0.5)
    axi[2].xaxis.set_label_coords(x=0.5, y=-0.02)

    plt.subplots_adjust(
        left=0.02,
        right=0.99,
        top=0.7,
        bottom=0.06,  # wspace=0.21, hspace=0.33
    )
    plt.savefig("plots/paper/support/support_3.pdf")

test_between_all_3()
# en([mpl.colors.rgb2hex(val) for val in mpl.colormaps["tab20"].colors])
['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']