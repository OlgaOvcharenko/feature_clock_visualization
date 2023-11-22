import numpy as np
import scanpy as sp
from plot import NonLinearClock

def read_data(path):
    return sp.read_h5ad(path) 

def setup_neftel_data():
    file_name = 'data/neftel_malignant.h5ad'
    X = read_data(file_name)

    obs = [ 
           'MESlike2', 'MESlike1', 'AClike', 'OPClike', 'NPClike1', 'NPClike2', 
        #    'G1S', 'G2M', 'genes_expressed'
           ]
    
    new_data = X.obs[obs].dropna()
    X_new = sp.AnnData(new_data)

    # compute umap
    sp.pp.neighbors(X_new)
    sp.tl.umap(X_new, min_dist=2)

    # get clusters
    standard_embedding = X_new.obsm['X_umap']

    # get labels
    mes = np.stack((X_new.X[:, 0], X_new.X[:,1]))
    npc = np.stack((X_new.X[:, 4], X_new.X[:, 5]))
    ac, opc = X_new.X[:, 2], X_new.X[:, 3]
    mes_max = np.max(mes, axis=0)
    npc_max = np.max(npc, axis=0)
    res_vect = np.stack((ac, opc, mes_max, npc_max))
    res_labels = np.max(res_vect, axis=0)

    return new_data, obs, standard_embedding, res_labels


def test_umap():
    X_new, obs, standard_embedding, labels = setup_neftel_data()

    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, "UMAP")
    plot_inst.plot_clocks(plot_title="Malignant cells", 
                          plot_big_clock=True, 
                          plot_small_clock=True,
                          standartize_data=True,
                          biggest_arrow_method=True,
                          univar_importance=True,
                          save_path_big="plots/new/big_1.png",
                          save_path_small="plots/new/small_1.png"
                          )

test_umap()
