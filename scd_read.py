import scanpy as sp
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib.colors as colors
import matplotlib.cm as cmx


def read_data(path):
    # X = sp.read_h5ad(file_name)  # anndata
    return sp.read(file_name)  # anndata

def umap_scanpy(X: sp.AnnData, groups_plots: list()):
    sp.tl.umap(X, min_dist=200, n_components=2)
    sp.pl.umap(X, color=groups_plots)

def tsne(X):
    sp.tl.tsne(X, n_pcs=2, perplexity=10)

def tsne(X):
    X_tsne = TSNE(n_components=2, init="random", perplexity=10)
    X_low_dim = X_tsne.fit_transform(X)
    return X_low_dim

def plot_tsne(X_low_dim):
    for g in groups_plots:
        uniq = list(set(X.obs[g]))
        hot = plt.get_cmap('hot')
        cNorm  = colors.Normalize(vmin=0, vmax=len(uniq))
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=hot)


        plt.figure(figsize=(6, 5))

        for i in range(len(uniq)):
            indx = X.obs[g] == uniq[i]
            plt.scatter(X_low_dim[indx, 0], X_low_dim[indx, 1], s=15, color=scalarMap.to_rgba(i), label=uniq[i])

        plt.legend()
        plt.show()



file_name = 'data/neftel_malignant.h5ad'
X = read_data(file_name)


groups_plots = ['histology', 'cross_section']
umap_scanpy(X, groups_plots)

