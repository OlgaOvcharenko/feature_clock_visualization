<picture align="center">
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/OlgaOvcharenko/feature_clock_visualization/main/examples/support/support_3.jpg">
  <img alt="Feature Clock" src="https://raw.githubusercontent.com/OlgaOvcharenko/feature_clock_visualization/main/examples/support/support_3.jpg" width="80%" height="80%">
</picture>

-----------------

# Feature Clock: High-Dimensional Effects in Two-Dimensional Plots

 **Feature Clock**, provides novel visualizations that eliminate the need for multiple plots to inspect the influence of original variables in the latent space, and enhances the explainability and compactness of visualizations of embedded data.
 **Feature Clock** allows creation of three types of static visualizations, highlighting the contributions of the high-dimensional features to linear directions of the two-dimensional spaces produced by nonlinear dimensionality reduction. 

## Table of Contents

- [Global clock](#how-to-make-a-global-feature-clock)  
- [Local clock](#how-to-make-a-local-feature-clock) 
- [Inter-group clock](#how-to-make-a-inter-group-feature-clock) 

## How to create a clock?

First, we need to create an instance of a `NonlinearClock` class. This class is a level of abstraction, used to store low- and high-dimensional data, calculate clocks and store intermediates. 
An instance of `NonlinearClock` can be resused to plot ***global*** / ***local*** / ***inter-group*** clocks.

```python
clock_instance = NonLinearClock(
    high_dim_data=X_org, 
    low_dim_data=X_emb, 
    observations=obs_names, 
    labels=labels, 
    method="UMAP", 
    cluster_labels=hdbscan_labels, 
    color_scheme=colors)

```

| Argument| Details| Format|
| --- | --- | --- |
| **high_dim_data** | Original numerical high-dimensional data of size with n observations (rows) and d features (cols). It is recommended to standardize data. | `numpy.array`
| **low_dim_data** | (Optional) Two-dimensional data of size with n observations (rows) and 2 features (cols) that is created using t-SNE, UMAP, PCA or any other dimensionlaity reduction technique (optionally, standartized). If not given, specify **method** argument. | `numpy.array`
| **observations** | List of names of all d features. List should contain d names, one per column, the order should correspond the order of columns in **high_dim_data**. | `list[str]`
| **labels** | (Optional, necessary only for the ***Inter-group* Clock**) Original labels of the data (if exist), size of n observations and 1 column. | `numpy.array`
| **method** | (Optional) Nonlinear dimensionality reduction method that should be used to compute **low_dim_data** if it is not given as input. Supported methods: `UMAP`, `tSNE`, `PHATE`. | `str`
| **cluster_labels** | (Optional) The labels with which ***Global*** and ***Local* Clocks** are created. User can either use original labels (same data as **labels**), or use any clustering algorithm to calculate clusters. If not specified,by default, HDBSCAN clustering algorihtm with high-dimensional data is used to define groups. | `numpy.array`
| **color_scheme** | (Optional) Specify colors that should be used for plotting. Number of colors should equal number of columns in high-dimentional data. By default, `TABLEAU` is used for datasets with less than 10 features, and `XKCD_COLORS` otherwise. | `list[str]`

## How to make a ***Global* Clock**?

***Global* Feature Clock** shows the effects of features in changing two-dimensional coordinates of the dataset as a whole.  
We construct a clock using all data points to visualize the impact of the high-dimensional features in the two- dimensional space. 
We apply linear regression to estimate linear directions of impact of each feature.
Since we approximate the path on a manifold with a line, this clock can miss some information.
After (creating a clock instance)(#how-to-create-a-clock), user can plot a ***Global* Feature Clock**.


```python
arrows, arrow_labels = clock_instance.plot_global_clock(
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
        plot_top_k=4
    )
```

| Argument| Details| Format|
| --- | --- | --- |
| **ax** | Matplotlib Axes object that is used to plot the clock. | `matplotlib.axes`
| **standartize_data** | (Optional) Standartize high- and low-dimensional data. Default: `True`. | `bool`
| **standartize_coef** | (Optional) Standartize coefficients of the linear regression model that is fitted by method. Leads to better compatibility and interpretability of coefficients, especially if data is not standartized. Default: `True`. | `bool`
| **feature_significance** | (Optional) Significance level for the t-test that is applied to test feature contributions (coefficients of the linear regression) equal 0. We visualize only statistically significant features.Default: `0.05`.| `float`
| **univar_importance** | (Optional) Fit univariate linear regression (`True`), one model per feature. Otherwise, calculate multivatiate regression (`False`) using all features. Default: `False` | `bool`
| **biggest_arrow_method** | (Optional) Show only the biggest vector of change per feature (`True`), or feature impact in all directions withing semicircle (`False`). Default: `True`. | `bool`
| **angle_shift** | (Optional) If compute feature impact in all directions withing semicircle(`biggest_arrow_method=False`), specify by how many degrees should radius be iterated. Default: `5`. | `float`
| **scale_circle** | (Optional) Make bigger or smaller the actual clock that is plotted by method. Circle size is defined by distance between minimum and maximum points. `1` means actual size, `> 1` makes clock bigger in plot, `< 1` - smaller. Default: `1.0`. | `float`
| **move_circle** | (Optional) How to move clock along x- or y-axis. List that contains two values. Default: `[0.0, 0.0]`. | `list[float]`
| **arrow_width** | (Optional) How thick arrows showing contributions should be. Default: `0.1`. | `float`
| **annotate** | (Optional) How often annotation within a clock should be plotted. The bigger value, the more annotations. Default: `0.6`. | `float`
| **plot_scatter** | (Optional) Plot the low-dimansional data scatter plot (`True`), or not (`False`). Default: `True`. | `bool`
| **plot_top_k** | (Optional) Visualize only ***k*** features with the biggest impact. Handy for datasets with more than 10 features. Should be smaller or equal number of features in the original dataset. `0` - visualize all. Default: `0`. | `int`
| Output **arrows** | Arrow objects created by library that can later be used by user to create a legend. | `list[matplotlib.pyplot.arrow]`
| Output **arrow_labels** | Feature names for each arrow that can later be used by user to create a legend. | `list[str]`


## How to make a ***Local* Clocks**?

***Local* Feature Clocks** helps to explore data at a finer granularity within a group, and enable an easier analysis of a selection of neighboring points. 
Class labels or unsupervised clustering of high-dimensional data determine the points used for a single clock. 
Analyzing original labels gives a perspective on what drives low-dimensional data point coordinates within a particular class or cluster. 
We default clustering to HDBSCAN, but users can define their own clusters via any clustering algorithm. 
Importantly, HDBSCAN also identifies outliers that are not assigned to any cluster. 
For each cluster or class, we apply the same method and create a single clock signifying changes within selected points.

After (creating a clock instance)(#how-to-create-a-clock), user can plot a ***Local* Feature Clock**.

```python
arrows, arrow_labels = clock_instance.plot_local_clocks(
        standartize_data=False,
        standartize_coef=False,
        biggest_arrow_method=True,
        univar_importance=False,
        ax=axi[1],
        scale_circles=[1.5, 1.5],
        move_circles=[[-0.3, 0.5], [0.0, -0.5]],
        clocks_labels=["Survival", "Death"],
        clocks_labels_angles = [45, 90],
        annotates=[0.3, 0.3],
        arrow_width=0.08,
        plot_top_k=4,
        plot_hulls=False,
        plot_scatter=False
    )
```

| Argument| Details| Format|
| --- | --- | --- |
| **ax** | Matplotlib Axes object that is used to plot the clock. | `matplotlib.axes`
| **standartize_data** | (Optional) Standartize high- and low-dimensional data. Default: `True`. | `bool`
| **standartize_coef** | (Optional) Standartize coefficients of the linear regression model that is fitted by method. Leads to better compatibility and interpretability of coefficients, especially if data is not standartized. Default: `True`. | `bool`
| **feature_significance** | (Optional) Significance level for the t-test that is applied to test feature contributions (coefficients of the linear regression) equal 0. We visualize only statistically significant features. Default: `0.05`.| `float`
| **univar_importance** | (Optional) Fit univariate linear regression (`True`), one model per feature. Otherwise, calculate multivatiate regression (`False`) using all features. Default: `False` | `bool`
| **biggest_arrow_method** | (Optional) Show only the biggest vector of change per feature (`True`), or feature impact in all directions withing semicircle (`False`). Default: `True`. | `bool`
| **angle_shift** | (Optional) If compute feature impact in all directions withing semicircle(`biggest_arrow_method=False`), specify by how many degrees should radius be iterated. Default: `5`. | `float`
| **scale_circles** | (Optional) Make bigger or smaller the actual clocks that is plotted by method. Circle size is defined by distance between minimum and maximum points. `1` means actual size, `> 1` makes clock bigger in plot, `< 1` - smaller. One value for each clock, size of list should be the same as number of clocks/groups. Default: `[1.0, 1.0, ...]`. | `list[float]`
| **move_circles** | (Optional) How to move clock salong x- or y-axis. List that contains lists of two values, one list per clock. Size of list should be the same as number of clocks/groups. Default: `[[0.0, 0.0], [0.0, 0.0], ...]`. | `list[list[float]]`
| **arrow_width** | (Optional) How thick arrows showing contributions should be.Default: `0.1`. | `float`
| **annotate** | (Optional) How often annotation within a clock should be plotted. The bigger value, the more annotations. One value per clock, size of list should be the same as number of clocks/groups. Default: `[0.1, 0.1, ...]`. | `list[float]`
| **plot_scatter** | (Optional) Plot the low-dimansional data scatter plot (`True`), or not (`False`). Default: `True`. | `bool`
| **plot_hulls** | (Optional) Plot the hulls around each group for easier perception of a group (`True`), or not (`False`). Default: `True`. | `bool`
| **plot_top_k** | (Optional) Visualize only ***k*** features with the biggest impact. Handy for datasets with more than 10 features. Should be smaller or equal number of features in the original dataset. `0` - visualize all. Default: `0`. | `int`
| **clocks_labels** | (Optional) Annotate each clock with a name, e.g., class name. Otherwise, sequential numbers are used (1, 2, ...). Default: `[1, 2, ...]`. | `list[str]`
| **clocks_labels_angles** | (Optional) Where on clock/circle to put the label, in degrees (top - 90, bottom - 270, right - 0, left - 180). Default: `[90, 90, ...]`. | `list[float]`
| Output **arrows** | Arrow objects (for each clock) created by library that can later be used by user to create a legend. | `list[list[matplotlib.pyplot.arrow]]`
| Output **arrow_labels** | Feature names (for each clock) for each arrow that can later be used by user to create a legend. | `list[list[str]]`

## How to make a ***Inter-group* Clock**?

***Inter-group* Feature Clock** helps inspect how variables change between groups, either clusters or classes. 
We fit a binary logistic regression with high-dimensional observations as predictor variables and visualize statistically significant coefficients on a single line that connects the group centers. 
In a multi-group setting, there is a space limitation for plotting all pairs of groups and their clocks. 
Therefore, we build a minimum spanning tree (MST) between group centers in low-dimensional space and plot the inter-group clocks only for a trajectory on the MST.

After (creating a clock instance)(#how-to-create-a-clock), user can plot a ***Inter-group* Feature Clock**.

```python
arrows, arrow_labels = clock_instance.plot_between_clock(
        standartize_data=True,
        standartize_coef=False,
        univar_importance=False,
        ax=axi[2],
        scale_circles=[6, 10],
        move_circles=[[1.8, 1.1], [0.1, -0.8]],
        move_circles=[[0.7, -1.5], [0, 0]],
        annotates=[0.5,0.5],
        arrow_width=0.08,
        plot_top_k=5,
        plot_scatter=False,
        plot_hulls=False
    )
```

| Argument| Details| Format|
| --- | --- | --- |
| **ax** | Matplotlib Axes object that is used to plot the clock. | `matplotlib.axes`
| **standartize_data** | (Optional) Standartize high- and low-dimensional data. Default: `True`. | `bool`
| **standartize_coef** | (Optional) Standartize coefficients of the linear regression model that is fitted by method. Leads to better compatibility and interpretability of coefficients, especially if data is not standartized. Default: `True`. | `bool`
| **feature_significance** | (Optional) Significance level for the t-test that is applied to test feature contributions (coefficients of the linear regression) equal 0. We visualize only statistically significant features. Default: `0.05`.| `float`
| **univar_importance** | (Optional) Fit univariate linear regression (`True`), one model per feature. Otherwise, calculate multivatiate regression (`False`) using all features. Default: `False` | `bool`
| **biggest_arrow_method** | (Optional) Show only the biggest vector of change per feature (`True`), or feature impact in all directions withing semicircle (`False`). Default: `True`. | `bool`
| **angle_shift** | (Optional) If compute feature impact in all directions withing semicircle(`biggest_arrow_method=False`), specify by how many degrees should radius be iterated. Default: `5`. | `float`
| **scale_circles** | (Optional) Make bigger or smaller the actual clocks that is plotted by method. Circle size is defined by distance between minimum and maximum points. `1` means actual size, `> 1` makes clock bigger in plot, `< 1` - smaller. One value for each clock, size of list should be the same as number of clocks/groups. Default: `[1.0, 1.0, ...]`. | `list[float]`
| **move_circles** | (Optional) How to move clock salong x- or y-axis. List that contains lists of two values, one list per clock. Size of list should be the same as number of clocks/groups. Default: `[[0.0, 0.0], [0.0, 0.0], ...]`. | `list[list[float]]`
| **arrow_width** | (Optional) How thick arrows showing contributions should be.Default: `0.1`. | `float`
| **annotate** | (Optional) How often annotation within a clock should be plotted. The bigger value, the more annotations. One value per clock, size of list should be the same as number of clocks/groups. Default: `[0.1, 0.1, ...]`. | `list[float]`
| **plot_scatter** | (Optional) Plot the low-dimansional data scatter plot (`True`), or not (`False`). Default: `True`. | `bool`
| **plot_hulls** | (Optional) Plot the hulls around each group for easier perception of a group (`True`), or not (`False`). Default: `True`. | `bool`
| **plot_top_k** | (Optional) Visualize only ***k*** features with the biggest impact. Handy for datasets with more than 10 features. Should be smaller or equal number of features in the original dataset. `0` - visualize all. Default: `0`. | `int`
| **clocks_labels** | (Optional) Annotate each clock with a name, e.g., class name. Otherwise, sequential numbers are used (1, 2, ...). Default: `[1, 2, ...]`. | `list[str]`
| **clocks_labels_angles** | (Optional) Where on clock/circle to put the label, in degrees (top - 90, bottom - 270, right - 0, left - 180). Default: `[90, 90, ...]`. | `list[float]`
| Output **arrows** | Arrow objects (for each clock) created by library that can later be used by user to create a legend. | `list[list[matplotlib.pyplot.arrow]]`
| Output **arrow_labels** | Feature names (for each clock) for each arrow that can later be used by user to create a legend. | `list[list[str]]`

## Tutorial
There is a simple [tutorial](https://github.com/OlgaOvcharenko/feature_clock_visualization/blob/main/tutorials/iris.md) for the [Iris flower dataset](https://www.kaggle.com/datasets/uciml/iris).
For more examples (e.g., [melody popularity](https://github.com/OlgaOvcharenko/feature_clock_visualization/blob/main/test/test_melody.py), [glioblastoma cell states](https://github.com/OlgaOvcharenko/feature_clock_visualization/blob/main/test/test_neftel.py), [diabetis by Pima Indians](https://github.com/OlgaOvcharenko/feature_clock_visualization/blob/main/test/test_pima.py)),  see [*test/*](https://github.com/OlgaOvcharenko/feature_clock_visualization/tree/main/test).

<hr>

[Go to Top](#table-of-contents)
