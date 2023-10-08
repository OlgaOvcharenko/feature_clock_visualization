from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import umap
import hdbscan
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import phate


mnist = fetch_openml('mnist_784', version=1)
mnist.target = mnist.target.astype(int)

# standard_embedding = umap.UMAP(random_state=7, min_dist=0.9, n_neighbors=30, n_components=2).fit_transform(mnist.data)

phate_op = phate.PHATE()
standard_embedding = phate_op.fit_transform(mnist.data)


plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=mnist.target.astype(int), s=0.1, cmap='Spectral')
plt.show()

kmeans_labels = cluster.KMeans(n_clusters=10).fit_predict(mnist.data)
plt.scatter(standard_embedding[:, 0], standard_embedding[:, 1], c=kmeans_labels, s=0.1, cmap='Spectral')
plt.show()

labels = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500,).fit_predict(standard_embedding)
clustered = (labels >= 0)
# plt.scatter(standard_embedding[~clustered, 0], standard_embedding[~clustered, 1], color=(0.5, 0.5, 0.5),ms=0.1, alpha=0.5)
plt.scatter(standard_embedding[clustered, 0], standard_embedding[clustered, 1], c=labels[clustered], s=0.1,cmap='Spectral');
plt.show()
