from sklearn.discriminant_analysis import StandardScaler
import statsmodels.api as sm
from matplotlib import pyplot as plt, patches
import numpy as np
import matplotlib.pyplot as plt
import umap
import hdbscan
import phate
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from skspatial.objects import Line
import math
import matplotlib.patheffects as pe
from sklearn.manifold import TSNE
from src.graph import Graph
import warnings


class NonLinearClock():
    def __init__(self, high_dim_data: np.array, 
                 observations: list, 
                 low_dim_data: np.ndarray = None, 
                 high_dim_labels: pd.DataFrame = None,
                 method: str = ""):
        
        self.high_dim_data = high_dim_data
        self.observations = observations
        self.low_dim_data = pd.DataFrame(low_dim_data, columns=["emb1", "emb2"])
            
        self.cluster_labels = self._compute_HDBSCAN_hulls() \
            if self.low_dim_data is not None else None
        
        self.high_dim_labels = high_dim_labels

        self.projections = None
        self.angles = None

        self.method = method

        if (self.method == "" and self.low_dim_data is not None) or (len(self.method) > 0 and self.low_dim_data is None):
            raise Exception("Low dimensional data or dimensionality reduction method is not specified.")
    
    def compute_UMAP(self, *args):
        self.method = "UMAP"
        um = umap.UMAP(args)
        self.low_dim_data = um.fit_transform(self.high_dim_data)
        return self.low_dim_data

    def compute_PHATE(self, *args):
        self.method = "PHATE"
        ph = phate.PHATE(args)
        self.low_dim_data = ph.fit_transform(self.high_dim_data)
        return self.low_dim_data
    
    def compute_tSNE(self, *args):
        self.method = "tSNE"
        tsne = TSNE(args)
        self.low_dim_data = tsne.fit_transform(self.high_dim_data)
        return self.low_dim_data

    def _compute_HDBSCAN_hulls(self):
        hdb = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=30)
        labels = hdb.fit_predict(self.low_dim_data)
        return labels
    
    def _standartize_data(self):
        for col in self.low_dim_data.columns:
            self.low_dim_data[col] = (self.low_dim_data[col] - \
                self.low_dim_data[col].mean()) / self.low_dim_data[col].std()

        for col in self.high_dim_data.columns:
            self.high_dim_data[col] = (self.high_dim_data[col] - \
                self.high_dim_data[col].mean()) / self.high_dim_data[col].std()

    def _prepare_data(self, standartize_data: bool = True):
        if self.low_dim_data is None:
            raise Exception("Low dimensional data is not initialized. Either give low_dim_data in constructor, or call compute_UMAP(), compute_TSNE(), compute_PHATE().")
        
        if self.method == "":
            raise Exception("Low dimensional data is initialized but dimensionality reduction method is not specified.")
        
        if standartize_data:
            self._standartize_data()
        
        self.low_dim_data["cluster"] = self.cluster_labels
        self.low_dim_data["label"] = self.high_dim_labels
    
    def _rotate(self, point: list, angle: float):
        x_new = point[0] * math.cos(math.radians(angle)) + point[1] * math.sin(math.radians(angle))
        y_new = point[1] * math.cos(math.radians(angle)) - point[0] * math.sin(math.radians(angle))
        return[x_new, y_new]
    
    def _project_line(self, data, angle: float, point_a: list=[0, 0], point_b: list=[1, 1]):
        line = Line.from_points(point_a=point_a, point_b=point_b)

        projected_points = []
        for i in range(data.shape[0]):
            point = (data["emb1"][i], data["emb2"][i])
            new_point = line.project_point(point)
            new_point = self._rotate(new_point, angle)
            projected_points.append(new_point[0])
        
        return np.array(projected_points)

    def _get_slope_from_angle(self, angle: float):
        return [math.cos(math.radians(angle)) * 100, math.sin(math.radians(angle)) * 100]
    
    def _create_angles_projections(self, angle_shift: int = 5):
        self.angles = list(range(0, 180, angle_shift))
        projections = [
            self._project_line(self.low_dim_data, angle, 
                                point_a=[0, 0], 
                                point_b=self._get_slope_from_angle(angle)) for angle in self.angles
            ]
        self.projections = np.array(projections).T
    
    def _get_importance(self, X, y, significance: float = 0.05, univar: bool = False):
        coefs, pvals, is_significant = [], [], []
        for i in range(y.shape[1]):
            if univar:
                coefs_a, pvals_a, is_significant_a = [], [], []
                for j in range(X.shape[1]):
                    lm = sm.OLS(y[:, i], X[:, j]).fit()
                    coefs_a.append(lm.params[0])
                    pvals_a.append(lm.pvalues[0])
                    is_significant_a.append(lm.pvalues[0] <= significance)

                coefs.append(np.array(coefs_a))
                pvals.append(pvals_a)
                is_significant.append(is_significant_a)

            else:
                lm = sm.OLS(y[:, i], X).fit()
                pval = np.array(lm.pvalues)
                coefs.append(np.array(lm.params))
                pvals.append(pval)
                is_significant.append(pval <= significance)
        return np.array(coefs), np.array(pvals), np.array(is_significant)
    
    def _plot_big_clock(self, 
                       plot_title: str,
                       standartize_coef: bool = True,
                       univar_importance: bool = True,
                       feature_significance: float = 0.05,
                       biggest_arrow: bool = True,
                       save_path: str = ""):
        coefs, _, is_significant = self._get_importance(self.high_dim_data.to_numpy(), self.projections, univar=univar_importance, significance=feature_significance)
            
        if standartize_coef:
            coefs = (coefs - coefs.mean(axis=0)) / coefs.std(axis=0)
            if np.isnan(coefs).any():
                coefs = np.nan_to_num(coefs)
                warnings.warn("NaNs were introduced after standartizing coefficients and were replaced by 0.")
        
        self._plot_central(self.low_dim_data, 45, self.angles, coefs, is_significant, self.observations, windrose=False, biggest_arrow=biggest_arrow, plot_title=plot_title, save_path=save_path)

    def _get_center(self, data):
        return data["emb1"].mean(), data["emb2"].mean()
    
    def _add_circles_lines(self, num_circles, annotate, angles_shift, x_center, y_center):
        # Add circles
        for a_s in list(range(0, 360, angles_shift)):
            a_s = math.radians(a_s)
            x_add, y_add = math.cos(a_s) * (num_circles-1) * annotate, math.sin(a_s) * (num_circles-1) * annotate
            plt.plot((x_center, x_center+x_add), (y_center, y_center+y_add), c='gray', linestyle="--", alpha=0.7, linewidth=1, zorder=10)
    
    def _get_cmin_cmax(self, coefs):
        c_min, c_max = coefs.min(), 0
        if coefs.shape[0] > 1:
            c_max = coefs.max()
        elif c_min > 0:
            c_max = c_min
            c_min = 0
        else:
            c_max = 0
        return c_min, c_max

    def _add_not_windrose(self, biggest_arrow, coefs_scaled, is_significant, colors, labels, x_center, y_center):
        arrows = []

        if not biggest_arrow:
            for a, c, s in zip(self.angles, coefs_scaled, is_significant):
                a = math.radians(a)

                # Plot contributions
                ind = abs(c).argsort(axis=None)[::-1]
                x_add_coefs, y_add_coefs = math.cos(a) * c[ind], math.sin(a) * c[ind]
                
                for is_s, x_c, y_c, i in zip(s[ind], x_add_coefs, y_add_coefs, ind):
                    # if is_s:
                    col = colors[i]
                    lbl = labels[i]
                    arrows.append(plt.arrow(x_center, y_center, x_c, y_c, width=0.01, color=col, label=lbl, zorder=15))
                
            plt.legend(arrows, labels)
        
        else:
            arrows_ind = np.argmax(np.abs(coefs_scaled), axis=0)
            
            for i in range(coefs_scaled.shape[1]):
                a = math.radians(self.angles[arrows_ind[i]])

                # Plot contributions
                x_c, y_c = math.cos(a) * coefs_scaled[arrows_ind[i], i], math.sin(a) * coefs_scaled[arrows_ind[i], i]
                col = colors[i]
                lbl = labels[i]
                arrows.append(plt.arrow(x_center, y_center, x_c, y_c, width=0.01, color=col, label=lbl, zorder=15))

            plt.legend(arrows, labels)
    
    def _add_windrose(self, ax, coefs_scaled, radius, x_center, y_center, colors, labels):
        points_x, points_y = [], []
        for a, c in zip(self.angles, coefs_scaled):
            a = math.radians(a)
            x_add, y_add = math.cos(a) * radius, math.sin(a) * radius
            x_add_coefs, y_add_coefs = math.cos(a) * c, math.sin(a) * c
            points_x.append(x_center + x_add_coefs)
            points_y.append(y_center + y_add_coefs)
        
        points_x = np.array(points_x)
        points_y = np.array(points_y)
        order_ix = np.argsort((abs(points_x).mean(axis=0) + abs(points_y).mean(axis=0)) / 2)[::-1]
        for i in order_ix:
            ax.plot(points_x[:, i], points_y[:, i], '-', c=colors[i], alpha=1, label=labels[i])
            # ax.fill(points_x[:, i], points_y[:, i], alpha=0.7, c=colors[i])

        plt.legend()

    def _make_circles(self, ax, num_circles, annotate, x_center, y_center, radius, c_max, c_min):
        for i in list(range(1, num_circles))[::-1]:
            radius_circle = round(i * annotate, 1)

            if i == num_circles-1:
                circle = patches.Circle((x_center, y_center), radius=radius_circle, edgecolor='gray', linewidth=1, linestyle="--", facecolor=(0.941, 0.973, 1.0, 0.5), fill=True, zorder=10) 
            else:
                circle = patches.Circle((x_center, y_center), radius=radius_circle, edgecolor='gray', linewidth=1, linestyle="--", fill=False, zorder=12) 

            ax.add_patch(circle)
            ax.axis('equal')

            circle_an = math.radians(45)
            x_ann, y_ann = x_center + math.cos(circle_an) * radius_circle, y_center + math.sin(circle_an) * radius_circle
            ax.annotate(str(round(radius_circle / (radius / max(abs(c_max), abs(c_min))), 2)), xy=(x_ann, y_ann), ha="right", color="gray",
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=20)

    def _add_circles(self, ax, num_circles, annotate, x_center, y_center, radius, c_min, c_max):
        for i in list(range(1, num_circles))[::-1]:
            radius_circle = round(i * annotate, 1)

            if i == num_circles-1:
                circle = patches.Circle((x_center, y_center), radius=radius_circle, edgecolor='gray', linewidth=1, linestyle="--", facecolor=(0.941, 0.973, 1.0, 0.5), fill=True, zorder=10) 
            else:
                circle = patches.Circle((x_center, y_center), radius=radius_circle, edgecolor='gray', linewidth=1, linestyle="--", fill=False, zorder=12) 

            ax.add_patch(circle)
            ax.axis('equal')

            circle_an = math.radians(45)
            x_ann, y_ann = x_center + math.cos(circle_an) * radius_circle, y_center + math.sin(circle_an) * radius_circle
            ax.annotate(str(round(radius_circle / (radius / max(abs(c_max), abs(c_min))), 2)), xy=(x_ann, y_ann), ha="right", color="gray",
                        path_effects=[pe.withStroke(linewidth=2, foreground="white")], zorder=20)

    def _plot_central(self, data, angles_shift, angles, coefs, is_significant, labels, plot_title: str, windrose: bool = True, biggest_arrow: bool = True, save_path: str = ""):
        fig, ax = plt.subplots(1, figsize=(12, 8))
        colors = list(mcolors.TABLEAU_COLORS.keys())
        alpha = 0.2
        annotate = 0.3

        # Scatter plot
        sc = plt.scatter(data["emb1"], data["emb2"], color='black', s=3, alpha=alpha, zorder=0)
        # fig.colorbar(sc)

        # Add circles lines
        x_center, y_center = self._get_center(data)
        radius = np.abs(coefs).max() * 2 #scale
        c_min, c_max = self._get_cmin_cmax(coefs=coefs)
        num_circles = math.floor(radius / annotate) + 1
        coefs_scaled = coefs * (radius / max(abs(c_max), abs(c_min)))
        self._add_circles_lines(num_circles, annotate, angles_shift, x_center, y_center)

        if not windrose:
            self._add_not_windrose(biggest_arrow, coefs_scaled, is_significant, colors, labels, x_center, y_center)

        else:
            self._add_windrose(ax, coefs_scaled, radius, x_center, y_center, colors, labels)
        
        self._add_circles(ax, num_circles, annotate, x_center, y_center, radius, c_min, c_max)

        self._finish_plot(plot_title=plot_title, save_path=save_path)
    
    def _draw_clusters(self, ax, data, alpha):
        for i in data['cluster'].unique():
            if i >= 0:
                points = data[data['cluster'] == i]
                if points.shape[0] > 2:
                    hull = ConvexHull(points[['emb1', 'emb2']].to_numpy())
                    vert = np.append(hull.vertices, hull.vertices[0])

                    a, b = points['emb1'].iloc[vert], points['emb2'].iloc[vert]
                    ax.plot(a, b, '--', c="gray", alpha=alpha)
                    ax.fill(a, b, alpha=alpha, c="gray")
    
    def _get_plot(self, plt, data, draw_hulls: bool = True):
        fig, ax = plt.subplots(1, figsize=(12, 8))
        alpha = 0.2

        if draw_hulls:
            self._draw_clusters(ax, data, alpha)
        # c=data["label"], cmap="viridis",  vmin=data["label"].min(), vmax=data["label"].max(), 
        sc = plt.scatter(data["emb1"], data["emb2"], color='black', s=3, alpha=alpha)
        # fig.colorbar(sc)
        return fig, ax
    
    def _plot_small(self, fig, ax, data, angles_shift, angles, coefs, is_significant, labels, plot_title: str, biggest_arrow: bool = True, scale_circle: float = 1.0, annotate: float = 0.2):
        colors = list(mcolors.TABLEAU_COLORS.keys())
        x_center, y_center = self._get_center(data)
        
        radius = np.abs(coefs).max() * scale_circle
        self._get_cmin_cmax(coefs)

        # Add circles lines
        c_min, c_max = self._get_cmin_cmax(coefs)
        num_circles = math.floor(radius / annotate) + 1
        coefs_scaled = coefs * (radius / max(abs(c_max), abs(c_min)))
        self._add_circles_lines(num_circles, annotate, angles_shift, x_center, y_center)

        arrows = []
        if not biggest_arrow:
            for a, c, s in zip(angles, coefs_scaled, is_significant):
                a = math.radians(a)

                # Plot contributions
                ind = abs(c).argsort(axis=None)[::-1]
                x_add_coefs, y_add_coefs = math.cos(a) * c[ind], math.sin(a) * c[ind]

                for is_s, x_c, y_c, i in zip(s[ind], x_add_coefs, y_add_coefs, ind):
                    if is_s:
                        col = colors[i]
                        lbl = labels[i]
                        arrows.append(plt.arrow(x_center, y_center, x_c, y_c, width=0.01, color=col, label=lbl, zorder=15))

            plt.legend(arrows, labels)
        
        else:
            arrows_ind = np.argmax(np.abs(coefs_scaled), axis=0)
            
            for i in range(coefs_scaled.shape[1]):
                a = math.radians(angles[arrows_ind[i]])

                # Plot contributions
                x_c, y_c = math.cos(a) * coefs_scaled[arrows_ind[i], i], math.sin(a) * coefs_scaled[arrows_ind[i], i]
                
                col = colors[i]
                lbl = labels[i]
                arrows.append(plt.arrow(x_center, y_center, x_c, y_c, width=0.01, color=col, label=lbl, zorder=15))

            plt.legend(arrows, labels)
        
        self._add_circles(ax, num_circles, annotate, x_center, y_center, radius, c_min, c_max)

        return arrows

    def _plot_small_clock(self, standartize_coef, univar_importance, feature_significance, biggest_arrow, plot_title, save_path):
        dist_clusters = self.low_dim_data["cluster"].unique()
        dist_clusters.sort()
        dist_clusters = dist_clusters[1:] # FIXME
        arrows_all = []

        fig, ax = self._get_plot(plt, self.low_dim_data, True)

        # FIXME
        scale_circle = [0.25, 0.25, 0.25]
        annotate = [0.2, 0.1, 0.1]

        # for a, scale, cl in zip(annotate, scale_circle, dist_clusters):
        for cl in dist_clusters:
            a = 1.0
            scale = 0.05

            ind = (self.low_dim_data["cluster"] == cl).values.reshape((self.low_dim_data.shape[0], 1))

            data_cl = self.low_dim_data[ind]
            new_data_cl = self.high_dim_data[ind]
            projections_cl = self.projections[ind[:, 0], :]

            coefs, _, is_significant = self._get_importance(new_data_cl.to_numpy(), projections_cl, univar=univar_importance, significance=feature_significance)
            if standartize_coef:
                coefs = (coefs - coefs.mean(axis=0)) / coefs.std(axis=0)
                if np.isnan(coefs).any():
                    coefs = np.nan_to_num(coefs)
                    warnings.warn("NaNs were introduced after standartizing coefficients and were replaced by 0.")
            
            arrows = self._plot_small(fig, ax, data_cl, 45, self.angles, coefs, is_significant, self.observations, scale_circle=scale, annotate=a, biggest_arrow=biggest_arrow, plot_title=plot_title)
            arrows_all.extend(arrows)
            
        plt.legend(arrows_all, self.observations)
        self._finish_plot(plot_title=plot_title, save_path=save_path)

    def _finish_plot(self, plot_title, save_path):
        plt.xlabel(f"{self.method} 1")
        plt.ylabel(f"{self.method} 2")
        plt.title(plot_title)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()

    def _get_cluster_centers(self, dist_clusters):
        cl_means = dict()
        for cl in dist_clusters:
            cluster_ind = self.low_dim_data["cluster"] == cl
            data_dim1 = self.low_dim_data["emb1"].loc[cluster_ind]
            data_dim2 = self.low_dim_data["emb2"].loc[cluster_ind]
            cl_means[cl] = np.array([data_dim1.mean(), data_dim2.mean()])
        return cl_means

    def _build_adj_matrix(self, cl_means):
        adj_matrix = []
        for val1 in cl_means.values():
            adj_row = []
            for val2 in cl_means.values():
                dist = np.linalg.norm(val1-val2)
                adj_row.append(dist)
            adj_matrix.append(adj_row)
        return adj_matrix

    def _build_MST(self, n_clusters, adj_matrix):
        grapf_cl = Graph(n_clusters)
        grapf_cl.graph = adj_matrix
        mst = grapf_cl.primMST()
        return mst
    
    def _get_angle_to_x(self, points_a, points_b):
        m1 = (points_a[1] - points_b[1]) / (points_a[0] - points_b[0])
        y = min(points_a[1], points_b[1])
        m2 = (y - y) / (points_a[0] - points_b[0])
        return math.atan((m1 - m2) / (1 + (m1*m2)))
    
    def _plot_between(self, data, cl_means, mst, coefs, labels, angles, plot_title: str, save_path: str = ""):
        fig, ax = plt.subplots(1, figsize=(12, 8))
        colors = list(mcolors.TABLEAU_COLORS.keys())
        alpha = 0.2

        # Scale coefficients
        radius = np.abs(coefs).max()
        c_min, c_max = self._get_cmin_cmax(coefs=coefs)
        coefs_scaled = coefs * (radius / max(abs(c_max), abs(c_min)))

        # Scatter plot
        sc = plt.scatter(data["emb1"], data["emb2"], color='black', s=3, alpha=alpha, zorder=0)
        self._draw_clusters(ax, data, alpha)
        
        # Add lines
        for val in mst:
            p_a, p_b = cl_means[val[0]], cl_means[val[1]]
            plt.plot((p_a[0], p_b[0]), (p_a[1], p_b[1]), c='gray', linestyle="--", alpha=0.7, linewidth=1, zorder=10)
        
        # Add arrows
        arrows_ind = np.argmax(np.abs(coefs_scaled), axis=0)
        arrows = []

        for j in range(len(mst)):
            for i in range(coefs_scaled.shape[1]):
                a = math.radians(angles[j])

                # Plot contributions
                x_c, y_c = math.cos(a) * coefs_scaled[j, i], math.sin(a) * coefs_scaled[j, i]
                col = colors[i]
                lbl = labels[i]

                arrows.append(plt.arrow(cl_means[mst[j][0]][0], cl_means[mst[j][0]][1], x_c, y_c, width=0.01, color=col, label=lbl, zorder=15))

        plt.legend(arrows, labels)
        self._finish_plot(plot_title=plot_title, save_path=save_path)
    
    
    def _plot_between_clusters(self, standartize_coef, univar_importance, feature_significance, biggest_arrow, plot_title, save_path):
        dist_clusters = self.low_dim_data["cluster"].unique()
        dist_clusters.sort()
        dist_clusters = dist_clusters[1:] if dist_clusters[0] == -1 else dist_clusters

        cl_means = self._get_cluster_centers(dist_clusters)
        adj_matrix = self._build_adj_matrix(cl_means)
        mst = self._build_MST(len(dist_clusters), adj_matrix)

        mst_proj, angles = [], []
        for val in mst:
            angle = math.degrees(self._get_angle_to_x(cl_means[val[0]], cl_means[val[1]]))
            proj = self._project_line(self.low_dim_data, 
                                angle, 
                                point_a=cl_means[val[0]], 
                                point_b=cl_means[val[1]])
            mst_proj.append(proj)
            angles.append(angle)

        self.mst_proj = np.array(mst_proj).T
        coefs, _, is_significant = self._get_importance(self.high_dim_data.to_numpy(), self.mst_proj, univar=univar_importance, significance=feature_significance)
        
        if standartize_coef:
            coefs = (coefs - coefs.mean(axis=0)) / coefs.std(axis=0)
            if np.isnan(coefs).any():
                coefs = np.nan_to_num(coefs)
                warnings.warn("NaNs were introduced after standartizing coefficients and were replaced by 0.")

        self._plot_between(self.low_dim_data, cl_means, mst, coefs, self.observations, angles, plot_title=plot_title, save_path=save_path)


    def plot_clocks(self, 
                  plot_title: str,
                  plot_big_clock: bool = True,
                  plot_small_clock: bool = True,
                  plot_between_cluster: bool = True,
                  standartize_data: bool = True, 
                  standartize_coef: bool = True, 
                  biggest_arrow_method: bool = True, 
                  angle_shift: int = 5, 
                  univar_importance: bool = True,
                  feature_significance: float = 0.05,
                  save_path_big: str = "",
                  save_path_small: str = "",
                  save_path_between: str = ""):
        self._prepare_data(standartize_data=standartize_data)
        self._create_angles_projections(angle_shift=angle_shift)

        print(self.low_dim_data["cluster"].nunique())

        if plot_big_clock:
            self._plot_big_clock(standartize_coef=standartize_coef, 
                                 univar_importance=univar_importance, 
                                 feature_significance=feature_significance, 
                                 biggest_arrow=biggest_arrow_method, 
                                 plot_title=plot_title, 
                                 save_path=save_path_big)

        if plot_small_clock:
            self._plot_small_clock(standartize_coef=standartize_coef, 
                                   univar_importance=univar_importance, 
                                   feature_significance=feature_significance, 
                                   biggest_arrow=biggest_arrow_method, 
                                   plot_title=plot_title, 
                                   save_path=save_path_small)
        
        if plot_between_cluster:
            self._plot_between_clusters(standartize_coef=standartize_coef, 
                                   univar_importance=univar_importance, 
                                   feature_significance=feature_significance, 
                                   biggest_arrow=biggest_arrow_method, 
                                   plot_title=plot_title, 
                                   save_path=save_path_between)
