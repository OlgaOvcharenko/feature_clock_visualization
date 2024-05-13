<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="examples/iris/iris.png">
  <img alt="Feature Clock" src="examples/iris/iris.png">
</picture>

## A ***Global* Feature Clock** for the Iris Dataset 

Given [Iris flower dataset](https://www.kaggle.com/datasets/uciml/iris), we want to analyze two-dimensional t-SNE embeddings and see which features impact the low dimensional space.
Additionally, we are interested in the comparison of PCA and t-SNE.

First, we prepare the data by standartizing and compute t-SNE representations.

```python 
def setup_iris_data(file_name):
    '''
    Return original normalized data X, names of features obs, t-SNE embeddings standard_embedding, and original labels.
    '''
    X = read_data(file_name)
    X.drop(columns={"Id"}, inplace=True)

    for col in X.columns:
        if col != 'Species':
            X[col] = (X[col] - X[col].mean()) / X[col].std()

    labels = X["Species"]
    X.drop(columns=["Species"], inplace=True)
    obs = list(X.columns)

    # Calculate t-SNE 2D representations
    tsne = manifold.TSNE(n_components = 2, learning_rate = 'auto', random_state = 42, n_iter=1000, perplexity=17)
    standard_embedding = tsne.fit_transform(X)
    
    # Normalize t-SNE embeddings
    for i in range(standard_embedding.shape[1]):
        standard_embedding[:, i] = (standard_embedding[:, i] - standard_embedding[:, i].mean()) / standard_embedding[:, i].std()
    
    return X, obs, standard_embedding, labels
```

Now, we use high- and low-dimensional data to make a clock. For more details of how PCA loadings were calculated and plotted, see [test/test_iris.py](test/test_iris.py).

```python 
def make_clock(file_name, fig_size, dpi):
    '''
    Plot a clock and save into a file under file_name.
    '''
    fig = plt.figure(constrained_layout=True, figsize=fig_size, dpi=dpi, facecolor="w",edgecolor="k",)
    spec2 = gridspec.GridSpec(ncols=3, nrows=1, figure=fig, 
                     left=0.04, right=0.99, top=0.72, bottom=0.08)

    # Define ax object to put a clock
    ax3 = fig.add_subplot(spec2[2])
    ax3.set_yticks([])
    ax3.set_xticks([])
    ax3.set_ylabel("t-SNE2", size=8)
    ax3.set_xlabel("t-SNE1", size=8)

    ...
    
    # Read and prepare data
    iris_data_path = "iris.csv"
    X_new, obs, standard_embedding, labels = setup_iris_data(iris_data_path)
    
    # Make a scatter plot separately, outside the clock
    # Scatter can also be made in clock by setting plot_scatter=True
    sc = ax3.scatter(standard_embedding[:,0], standard_embedding[:,1], marker= '.', c=labels, cmap=colormap, norm=normalize, alpha=0.2, zorder=0, edgecolors='face')

    # Define a Feature Clock instance
    plot_inst = NonLinearClock(X_new, obs, standard_embedding, labels, method="umap", cluster_labels=labels, color_scheme=colors)

    # Using defined object to create a Global Clock
    # Returns matplotlib arrows and their labels that can be used for making the legend
    arrows_objects, arrows_labels = plot_inst.plot_global_clock(
        standartize_data=False, # already standartized data before
        standartize_coef=True, # standartize lin. reg.coefficients 
        biggest_arrow_method=True, # plot only one arrow per feature
        univar_importance=False, # calculate multivariate regression 
        ax=ax3, # ax object where clock is plotted 
        scale_circle=1.5, # make clock bigger
        move_circle=[-0.3, 0.0], # move clock along x/y axis
        annotate=0.5, # how often to add annotations within clock
        arrow_width=0.05, # how thick should be feature arrows
        plot_scatter=False, # do not plot default scatter
    )

    ...

    plt.savefig("iris_clock.pdf")

```

