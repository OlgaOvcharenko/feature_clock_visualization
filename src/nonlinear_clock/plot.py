import copy
import matplotlib
from sklearn.discriminant_analysis import StandardScaler
import statsmodels.api as sm
from matplotlib import pyplot as plt, patches
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import HDBSCAN
import phate
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
from skspatial.objects import Line
import math
import matplotlib.patheffects as pe
from sklearn.manifold import TSNE
from src.nonlinear_clock.graph import Graph
import warnings
from sklearn.preprocessing import StandardScaler


class NonLinearClock:
    def __init__(
        self,
        high_dim_data: np.array,
        observations: list,
        low_dim_data: np.ndarray = None,
        high_dim_labels: pd.DataFrame = None,
        method: str = "",
        cluster_labels: np.ndarray = None,
        color_scheme: list = list(mcolors.TABLEAU_COLORS.keys()),
    ):
        self.is_projected = False

        self.high_dim_data = high_dim_data
        self.observations = observations
        low_dim_data = np.array(low_dim_data)
        cluster_labels = np.array(cluster_labels)
        self.low_dim_data = pd.DataFrame(low_dim_data, columns=["emb1", "emb2"])

        if cluster_labels is not None:
            self.cluster_labels = cluster_labels
        elif self.low_dim_data is not None:
            self.cluster_labels = self._compute_HDBSCAN_hulls()
        else:
            self.cluster_labels = None

        self.high_dim_labels = high_dim_labels

        self.projections = None
        self.angles = None

        self.method = method

        self.shift = 0

        self.colors = dict()

        if len(observations) >= 10 and color_scheme is list(
            mcolors.TABLEAU_COLORS.keys()
        ):
            color_scheme = list(mcolors.XKCD_COLORS.keys())

        for i, obs in enumerate(observations):
            self.colors[obs] = color_scheme[i]

        if (self.method == "" and self.low_dim_data is not None) or (
            len(self.method) > 0 and self.low_dim_data is None
        ):
            raise Exception(
                "Low dimensional data or dimensionality reduction method is not specified."
            )

        if cluster_labels is not None or self.low_dim_data is not None:
            self._prepare_data(standartize_data=False)

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
        hdb = HDBSCAN(min_samples=10, min_cluster_size=30)
        labels = hdb.fit_predict(self.low_dim_data)
        return labels

    def _standartize_data(self):
        for col in self.low_dim_data.columns:
            self.low_dim_data[col] = (
                self.low_dim_data[col] - self.low_dim_data[col].mean()
            ) / self.low_dim_data[col].std()

        for col in self.high_dim_data.columns:
            self.high_dim_data[col] = (
                self.high_dim_data[col] - self.high_dim_data[col].mean()
            ) / self.high_dim_data[col].std()

    def _prepare_data(self, standartize_data: bool = True):
        if self.low_dim_data is None:
            raise Exception(
                "Low dimensional data is not initialized. Either give low_dim_data in constructor, or call compute_UMAP(), compute_TSNE(), compute_PHATE()."
            )

        if self.method == "":
            raise Exception(
                "Low dimensional data is initialized but dimensionality reduction method is not specified."
            )

        if standartize_data:
            self._standartize_data()

        self.low_dim_data["cluster"] = self.cluster_labels
        self.low_dim_data["label"] = self.high_dim_labels

    def _rotate(self, point: list, angle: float):
        x_new = point[0] * math.cos(math.radians(angle)) + point[1] * math.sin(
            math.radians(angle)
        )
        y_new = point[1] * math.cos(math.radians(angle)) - point[0] * math.sin(
            math.radians(angle)
        )
        return [x_new, y_new]

    def _project_line(
        self, data, angle: float, point_a: list = [0, 0], point_b: list = [1, 1]
    ):
        line = Line.from_points(point_a=point_a, point_b=point_b)

        projected_points = []
        for i in range(data.shape[0]):
            point = (data["emb1"][i], data["emb2"][i])
            new_point = line.project_point(point)
            new_point = self._rotate(new_point, angle)
            projected_points.append(new_point[0])

        return np.array(projected_points)

    def _get_slope_from_angle(self, angle: float):
        return [
            math.cos(math.radians(angle)) * 100,
            math.sin(math.radians(angle)) * 100,
        ]

    def _create_angles_projections(self, angle_shift: int = 5):
        self.angles = list(range(0, 180, angle_shift))
        projections = [
            self._project_line(
                self.low_dim_data,
                angle,
                point_a=[0, 0],
                point_b=self._get_slope_from_angle(angle),
            )
            for angle in self.angles
        ]
        self.projections = np.array(projections).T

    def _create_biggest_projections(self):
        self.angles = [0, 90]
        projections = [
            self._project_line(
                self.low_dim_data,
                angle,
                point_a=[0, 0],
                point_b=self._get_slope_from_angle(angle),
            )
            for angle in self.angles
        ]
        self.projections = np.array(projections).T

    def _get_importance(self, X, y, significance: float = 0.05, univar: bool = False):
        coefs, pvals, is_significant, std_x, std_y = [], [], [], [], []
        for i in range(y.shape[1]):
            if univar:
                coefs_a, pvals_a, is_significant_a = [], [], []
                for j in range(X.shape[1]):
                    lm = sm.OLS(y[:, i], X[:, j]).fit()
                    coefs_a.append(lm.params[0])
                    pvals_a.append(lm.pvalues[0])
                    is_significant_a.append(lm.pvalues[0] <= significance)

                coefs.append(np.array(coefs_a))
                std_x.append(np.ones((len(coefs_a),)))
                std_y.append(np.ones((len(coefs_a),)))
                pvals.append(pvals_a)
                is_significant.append(is_significant_a)

                # FIXME add std X, y

            else:
                lm = sm.OLS(y[:, i], X).fit()
                pval = np.array(lm.pvalues)
                coefs.append(np.array(lm.params))
                pvals.append(pval)
                is_significant.append(pval <= significance)

                std_x.append(X.std(axis=0))
                std_y.append(y[:, i].std())
        return (
            np.array(coefs),
            np.array(pvals),
            np.array(is_significant),
            np.array(std_x),
            np.array(std_y),
        )

    def _plot_big_clock(
        self,
        ax: matplotlib.axis.Axis,
        standartize_coef: bool = True,
        univar_importance: bool = True,
        feature_significance: float = 0.05,
        biggest_arrow: bool = True,
        scale_circle: float = 1,
        move_circle: list = [0, 0],
        annotate: float = 0.3,
        arrow_width: float = 0.1,
        plot_scatter: bool = True,
        plot_top_k: int = 0,
    ):
        coefs, _, is_significant, std_x, std_y = self._get_importance(
            self.high_dim_data.to_numpy(),
            self.projections,
            univar=univar_importance,
            significance=feature_significance,
        )

        if standartize_coef:
            coefs = coefs / std_x  # FIXME
            if np.isnan(coefs).any():
                coefs = np.nan_to_num(coefs)
                warnings.warn(
                    "NaNs were introduced after standartizing coefficients and were replaced by 0."
                )

        if biggest_arrow:
            is_significant = np.logical_or(is_significant[0], is_significant[1])
            coefs_points = [[x, y] for x, y in zip(coefs[0], coefs[1])]
            coefs_new = np.array(
                [math.sqrt((x * x) + (y * y)) for x, y in zip(coefs[0], coefs[1])]
            )

            if plot_top_k != 0:
                min_ix = list(
                    np.argsort(coefs_new)[0 : len(coefs_new) - plot_top_k]
                )  # FISME * is_significant
                is_significant[min_ix] = False

            coefs = coefs_new

            arrows, arrow_labels = self._plot_central(
                ax,
                self.low_dim_data,
                45,
                coefs,
                is_significant,
                self.observations,
                biggest_arrow=biggest_arrow,
                scale_circle=scale_circle,
                move_circle=move_circle,
                annotate=annotate,
                arrow_width=arrow_width,
                plot_scatter=plot_scatter,
                points=coefs_points,
            )

        else:
            if plot_top_k != 0:
                min_ix = list(
                    np.argsort(np.max(np.abs(coefs * is_significant), axis=0))[
                        0 : coefs.shape[1] - plot_top_k
                    ]
                )
                is_significant[:, min_ix] = False

            arrows, arrow_labels = self._plot_central(
                ax,
                self.low_dim_data,
                45,
                coefs,
                is_significant,
                self.observations,
                biggest_arrow=biggest_arrow,
                scale_circle=scale_circle,
                move_circle=move_circle,
                annotate=annotate,
                arrow_width=arrow_width,
                plot_scatter=plot_scatter,
            )
        return arrows, arrow_labels

    def _get_center(self, data):
        return data["emb1"].mean(), data["emb2"].mean()

    def _add_circles_lines(
        self, ax, num_circles, annotate, angles_shift, x_center, y_center
    ):
        # Add circles
        for a_s in list(range(0, 360, angles_shift)):
            a_s = math.radians(a_s)
            x_add, y_add = (
                math.cos(a_s) * (num_circles - 1) * annotate,
                math.sin(a_s) * (num_circles - 1) * annotate,
            )
            ax.plot(
                (x_center, x_center + x_add),
                (y_center, y_center + y_add),
                c="gray",
                linestyle="--",
                alpha=0.7,
                linewidth=1,
                zorder=10,
            )

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

    def _add_biggest_contributions(
        self,
        ax,
        coefs_scaled,
        is_significant,
        labels,
        x_center,
        y_center,
        arrow_width,
        points,
    ):
        arrows, labels_all = [], []
        ix_col_by_len = np.argsort(coefs_scaled)[::-1]

        for i in ix_col_by_len:
            if is_significant[i]:
                # Plot contributions
                col = self.colors[self.observations[i]]
                lbl = labels[i]
                arrows.append(
                    ax.arrow(
                        x_center,
                        y_center,
                        points[i][0],
                        points[i][1],
                        width=arrow_width,
                        color=col,
                        label=lbl,
                        zorder=15,
                    )
                )
                labels_all.append(lbl)

        return arrows, labels_all

    def _add_circle_contributions(
        self, ax, coefs_scaled, is_significant, labels, x_center, y_center, arrow_width
    ):
        arrows, labels_all = [], []
        for a, c, s in zip(self.angles, coefs_scaled, is_significant):
            a = math.radians(a)

            # Plot contributions
            ind = abs(c).argsort(axis=None)[::-1]
            x_add_coefs, y_add_coefs = math.cos(a) * c[ind], math.sin(a) * c[ind]

            for is_s, x_c, y_c, i in zip(s[ind], x_add_coefs, y_add_coefs, ind):
                # if is_s: does not make sense since we want to see circles
                col = self.colors[self.observations[i]]
                lbl = labels[i]
                arrows.append(
                    ax.arrow(
                        x_center,
                        y_center,
                        x_c,
                        y_c,
                        width=arrow_width,
                        color=col,
                        label=lbl,
                        zorder=15,
                    )
                )
                labels_all.append(lbl)

        return arrows, labels_all

    def _make_circles(
        self, ax, num_circles, annotate, x_center, y_center, radius, c_max, c_min
    ):
        for i in list(range(1, num_circles))[::-1]:
            radius_circle = round(i * annotate, 1)
            if i == num_circles - 1:
                circle = patches.Circle(
                    (x_center, y_center),
                    radius=radius_circle,
                    edgecolor="gray",
                    linewidth=1,
                    linestyle="--",
                    facecolor=(0.941, 0.973, 1.0, 0.5),
                    fill=True,
                    zorder=10,
                )
            else:
                circle = patches.Circle(
                    (x_center, y_center),
                    radius=radius_circle,
                    edgecolor="gray",
                    linewidth=1,
                    linestyle="--",
                    fill=False,
                    zorder=12,
                )

            ax.add_patch(circle)
            ax.axis("equal")

            circle_an = math.radians(45)
            x_ann, y_ann = (
                x_center + math.cos(circle_an) * radius_circle,
                y_center + math.sin(circle_an) * radius_circle,
            )
            ax.annotate(
                str(round(radius_circle / (radius / max(abs(c_max), abs(c_min))), 2)),
                xy=(x_ann, y_ann),
                ha="right",
                color="gray",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                zorder=20,
            )

    def _add_circles(
        self,
        ax,
        num_circles,
        annotate,
        x_center,
        y_center,
        radius,
        c_min,
        c_max,
        ann_sign=[1, 1],
    ):
        for i in list(range(1, num_circles))[::-1]:
            radius_circle = round(i * annotate, 1)
            if i == num_circles - 1:
                circle = patches.Circle(
                    (x_center, y_center),
                    radius=radius_circle,
                    edgecolor="gray",
                    linewidth=1,
                    linestyle="--",
                    facecolor=(0.941, 0.973, 1.0, 0.5),
                    fill=True,
                    zorder=10,
                )
            else:
                circle = patches.Circle(
                    (x_center, y_center),
                    radius=radius_circle,
                    edgecolor="gray",
                    linewidth=1,
                    linestyle="--",
                    fill=False,
                    zorder=12,
                )

            ax.add_patch(circle)
            ax.axis("equal")

            circle_an = math.radians(45)
            x_ann, y_ann = (
                x_center + math.cos(circle_an) * radius_circle * ann_sign[0],
                y_center + math.sin(circle_an) * radius_circle * ann_sign[1],
            )
            ax.annotate(
                str(round(radius_circle / (radius / max(abs(c_max), abs(c_min))), 2)),
                xy=(x_ann, y_ann),
                ha="right",
                color="gray",
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
                fontsize=7,
                zorder=20,
            )

    def _plot_central(
        self,
        ax: matplotlib.axis.Axis,
        data,
        angles_shift,
        coefs,
        is_significant,
        labels,
        biggest_arrow: bool = True,
        scale_circle: float = 1,
        move_circle: list = [0, 0],
        annotate: float = 0.3,
        arrow_width: float = 0.1,
        plot_scatter: bool = True,
        points: list = [],
    ):
        alpha = 0.1

        if plot_scatter:
            # Scatter plot
            ax.scatter(
                data["emb1"], data["emb2"], color="gray", s=1, alpha=alpha, zorder=0
            )

        # Add circles lines
        x_center, y_center = self._get_center(data)
        x_center, y_center = x_center + move_circle[0], y_center + move_circle[1]
        radius = np.abs(coefs).max() * scale_circle
        c_min, c_max = self._get_cmin_cmax(coefs=coefs)
        num_circles = math.floor(radius / annotate) + 1
        coefs_scaled = coefs * (radius / max(abs(c_max), abs(c_min)))
        self._add_circles_lines(
            ax, num_circles, annotate, angles_shift, x_center, y_center
        )

        ann_sign = [1, 1]  # Chnage angle of the annotation

        if not biggest_arrow:
            arrows, arrow_labels = self._add_circle_contributions(
                ax,
                coefs_scaled,
                is_significant,
                labels,
                x_center,
                y_center,
                arrow_width,
            )
        else:
            if len(points) != len(labels):
                raise Exception(f"Points for biggest arrow method are not computed.")

            tmp_points = []
            counts = np.zeros((4,))
            for point in points:
                new_x = point[0] * (radius / max(abs(c_max), abs(c_min)))
                new_y = point[1] * (radius / max(abs(c_max), abs(c_min)))
                tmp_points.append([new_x, new_y])

                if new_x >= 0 and new_y >= 0:
                    counts[0] += 1
                elif new_x < 0 and new_y >= 0:
                    counts[1] += 1
                elif new_x < 0 and new_y < 0:
                    counts[2] += 1
                elif new_x >= 0 and new_y < 0:
                    counts[3] += 1
            points = copy.deepcopy(tmp_points)

            # Get quarter for annotation
            tmp_quater_max = np.argmin(counts)
            if tmp_quater_max == 1:
                ann_sign = [-1, 1]
            elif tmp_quater_max == 2:
                ann_sign = [-1, -1]
            elif tmp_quater_max == 3:
                ann_sign = [1, -1]

            arrows, arrow_labels = self._add_biggest_contributions(
                ax,
                coefs_scaled,
                is_significant,
                labels,
                x_center,
                y_center,
                arrow_width,
                points=points,
            )

        # num_circles - one circle per annotation
        self._add_circles(
            ax,
            num_circles,
            annotate,
            x_center,
            y_center,
            radius,
            c_min,
            c_max,
            ann_sign=ann_sign,
        )
        return arrows, arrow_labels

    def _draw_clusters(self, ax, data, alpha):
        for i in data["cluster"].unique():
            if i >= 0:
                points = data[data["cluster"] == i]
                if points.shape[0] > 2:
                    hull = ConvexHull(points[["emb1", "emb2"]].to_numpy())
                    vert = np.append(hull.vertices, hull.vertices[0])

                    a, b = points["emb1"].iloc[vert], points["emb2"].iloc[vert]
                    ax.plot(a, b, "--", c="gray", alpha=alpha)
                    ax.fill(a, b, alpha=alpha, c="gray")

    def _get_plot(self, ax, data, draw_hulls: bool = True, plot_scatter: bool = True):
        alpha = 0.1

        if draw_hulls:
            self._draw_clusters(ax, data, alpha)
        if plot_scatter:
            ax.scatter(data["emb1"], data["emb2"], color="gray", s=2, alpha=alpha)

    def _plot_small(
        self,
        ax,
        data,
        angles_shift,
        angles,
        coefs,
        is_significant,
        labels,
        biggest_arrow: bool = True,
        scale_circle: float = 1.0,
        annotate: float = 0.2,
        move_circle: list = [0, 0],
        arrow_width: float = 0.01,
        points: list = [],
    ):
        x_center, y_center = self._get_center(data)

        x_center += move_circle[0]
        y_center += move_circle[1]

        radius = np.abs(coefs).max() * scale_circle
        self._get_cmin_cmax(coefs)

        # Add circles lines
        c_min, c_max = self._get_cmin_cmax(coefs)
        num_circles = math.floor(radius / annotate) + 1
        coefs_scaled = coefs * (radius / max(abs(c_max), abs(c_min)))
        self._add_circles_lines(
            ax, num_circles, annotate, angles_shift, x_center, y_center
        )

        arrows, labels_all = [], []
        ann_sign = [1, 1]

        if not biggest_arrow:
            for a, c, s in zip(angles, coefs_scaled, is_significant):
                a = math.radians(a)

                # Plot contributions
                ind = abs(c).argsort(axis=None)[::-1]
                x_add_coefs, y_add_coefs = math.cos(a) * c[ind], math.sin(a) * c[ind]

                for is_s, x_c, y_c, i in zip(s[ind], x_add_coefs, y_add_coefs, ind):
                    if is_s:
                        col = self.colors[self.observations[i]]
                        lbl = labels[i]
                        arrows.append(
                            ax.arrow(
                                x_center,
                                y_center,
                                x_c,
                                y_c,
                                width=arrow_width,
                                color=col,
                                label=lbl,
                                zorder=15,
                            )
                        )

        else:
            if len(points) != len(labels):
                raise Exception(f"Points for biggest arrow method are not computed.")

            ix_col_by_len = np.argsort(coefs_scaled)[::-1]
            tmp_points = []
            counts = np.zeros((4,))
            for point in points:
                new_x = point[0] * (radius / max(abs(c_max), abs(c_min)))
                new_y = point[1] * (radius / max(abs(c_max), abs(c_min)))
                tmp_points.append([new_x, new_y])

                if new_x >= 0 and new_y >= 0:
                    counts[0] += 1
                elif new_x < 0 and new_y >= 0:
                    counts[1] += 1
                elif new_x < 0 and new_y < 0:
                    counts[2] += 1
                elif new_x >= 0 and new_y < 0:
                    counts[3] += 1
            points = copy.deepcopy(tmp_points)

            # Get quarter for annotation
            tmp_quater_max = np.argmin(counts)
            if tmp_quater_max == 1:
                ann_sign = [-1, 1]
            elif tmp_quater_max == 2:
                ann_sign = [-1, -1]
            elif tmp_quater_max == 3:
                ann_sign = [1, -1]

            for i in ix_col_by_len:
                # Plot contributions
                if is_significant[i]:
                    col = self.colors[self.observations[i]]
                    lbl = labels[i]
                    arrows.append(
                        ax.arrow(
                            x_center,
                            y_center,
                            points[i][0],
                            points[i][1],
                            width=arrow_width,
                            color=col,
                            label=lbl,
                            zorder=15,
                        )
                    )
                    labels_all.append(lbl)

        self._add_circles(
            ax,
            num_circles,
            annotate,
            x_center,
            y_center,
            radius,
            c_min,
            c_max,
            ann_sign=ann_sign,
        )

        return arrows, labels_all

    def _plot_small_clock(
        self,
        ax,
        standartize_coef,
        univar_importance,
        feature_significance,
        biggest_arrow,
        scale_circle,
        move_circle,
        annotate,
        arrow_width,
        plot_scatter,
        plot_hulls,
        plot_top_k,
    ):
        dist_clusters = self.low_dim_data["cluster"].unique()
        dist_clusters.sort()
        dist_clusters = dist_clusters[1:] if dist_clusters[0] == -1 else dist_clusters

        self._get_plot(ax, self.low_dim_data, plot_hulls, plot_scatter)

        all_coeffs, all_is_significant, all_points, all_std_x, all_std_y = (
            [],
            [],
            [],
            [],
            [],
        )
        arrows_dict = {}
        for cl in dist_clusters:
            ind = (self.low_dim_data["cluster"] == cl).values.reshape(
                (self.low_dim_data.shape[0], 1)
            )

            data_cl = self.low_dim_data[ind]
            new_data_cl = self.high_dim_data[ind]
            projections_cl = self.projections[ind[:, 0], :]

            coefs, _, is_significant, std_x, std_y = self._get_importance(
                new_data_cl.to_numpy(),
                projections_cl,
                univar=univar_importance,
                significance=feature_significance,
            )

            if biggest_arrow:
                is_significant = np.logical_or(is_significant[0], is_significant[1])
                coefs_points = [[x, y] for x, y in zip(coefs[0], coefs[1])]
                coefs_new = np.array(
                    [math.sqrt((x * x) + (y * y)) for x, y in zip(coefs[0], coefs[1])]
                )

                if plot_top_k != 0:
                    min_ix = list(
                        np.argsort(coefs_new)[0 : len(coefs_new) - plot_top_k]
                    )
                    is_significant[min_ix] = False

                coefs = coefs_new * is_significant
                all_points.append(coefs_points)

            else:
                if plot_top_k != 0:
                    min_ix = list(
                        np.argsort(np.max(np.abs(coefs * is_significant), axis=0))[
                            0 : coefs.shape[1] - plot_top_k
                        ]
                    )
                    is_significant[:, min_ix] = False

            all_coeffs.append(coefs)
            all_is_significant.append(is_significant)
            all_std_x.append(std_x)
            all_std_y.append(std_y)

        for i, cl in enumerate(dist_clusters):
            scale = scale_circle[i]
            a = annotate[i]

            ind = (self.low_dim_data["cluster"] == cl).values.reshape(
                (self.low_dim_data.shape[0], 1)
            )

            coefs = all_coeffs[i]
            is_significant = all_is_significant[i]
            data_cl = self.low_dim_data[ind]
            std_coef = all_std_x[i][0]
            std_coef_y = all_std_y[i][0]

            if standartize_coef:
                all_points[i] = all_points[i] / std_coef[:, None]  # FIXME
                all_points[i] = all_points[i] / std_coef[:, None]  # FIXME
                coefs = coefs / std_coef  # FIXME
                if np.isnan(coefs).any():
                    coefs = np.nan_to_num(coefs)
                    warnings.warn(
                        "NaNs were introduced after standartizing coefficients and were replaced by 0."
                    )

            if is_significant.sum() != 0:
                arrows, arrow_labels = self._plot_small(
                    ax,
                    data_cl,
                    45,
                    self.angles,
                    coefs,
                    is_significant,
                    self.observations,
                    scale_circle=scale,
                    annotate=a,
                    biggest_arrow=biggest_arrow,
                    move_circle=move_circle[i],
                    arrow_width=arrow_width,
                    points=all_points[i] if len(all_points) > 0 else [],
                )

                for arrow, lbl in zip(arrows, arrow_labels):
                    if arrows_dict.get(lbl) is None:
                        arrows_dict[lbl] = arrow

            else:
                warnings.warn(f"Cluster {i} has no significant features.")

        return list(arrows_dict.values()), list(arrows_dict.keys())

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
                dist = np.linalg.norm(val1 - val2)
                adj_row.append(dist)
            adj_matrix.append(adj_row)
        return adj_matrix

    def _build_MST(self, n_clusters, adj_matrix):
        grapf_cl = Graph(n_clusters)
        grapf_cl.graph = adj_matrix
        mst = grapf_cl.primMST()
        print(f"MST constructed for the inter-cluster clock ([[node1, node2, distance], ...]): \n {mst}")
        return mst

    def _get_angle_to_x(self, points_a, points_b):
        m1 = (points_a[1] - points_b[1]) / (points_a[0] - points_b[0])
        y = min(points_a[1], points_b[1])
        m2 = (y - y) / (points_a[0] - points_b[0])
        return math.atan((m1 - m2) / (1 + (m1 * m2)))

    def _plot_between(
        self,
        ax,
        data,
        cl_means,
        mst,
        coefs,
        labels,
        angles,
        scale_circle,
        move_circle,
        annotate,
        arrow_width,
        plot_scatter,
        plot_hulls,
    ):
        alpha = 0.1

        # Scatter plot
        if plot_scatter:
            ax.scatter(
                data["emb1"], data["emb2"], color="gray", s=2, alpha=alpha, zorder=0
            )

        if plot_hulls:
            self._draw_clusters(ax, data, alpha)

        # Add lines and clock
        clock_centers = []
        coefs_scaled = copy.deepcopy(coefs)
        for i, val in enumerate(mst):
            p_a, p_b = cl_means[val[0]], cl_means[val[1]]

            # FIXME
            # ax.plot(
            #     (p_a[0], p_b[0]),
            #     (p_a[1], p_b[1]),
            #     c="gray",
            #     linestyle="--",
            #     alpha=0.7,
            #     linewidth=1,
            #     zorder=10,
            # )

            # Add clock
            x_center, y_center = (
                np.mean((p_a[0], p_b[0])) + move_circle[i][0],
                np.mean((p_a[1], p_b[1])) + move_circle[i][1],
            )
            clock_centers.append([x_center, y_center])

            radius = (math.dist(p_a, p_b) / 2) * scale_circle[i]
            c_min, c_max = self._get_cmin_cmax(coefs=coefs[i])
            num_circles = math.floor(radius / annotate[i]) + 1
            self._add_circles_lines(
                ax, num_circles, annotate[i], 90, x_center, y_center
            )

            # num_circles - one circle per annotation
            self._add_circles(
                ax, num_circles, annotate[i], x_center, y_center, radius, c_min, c_max
            )

            coefs_scaled[i] = coefs_scaled[i] * (radius / max(abs(c_max), abs(c_min)))

        # Add arrows
        arrows = {}
        for j in range(len(mst)):
            for i in abs(coefs_scaled[j]).argsort(axis=None)[::-1]:
                # Add actual arrow
                a = math.radians(angles[j])
                x_center, y_center = clock_centers[min(mst[j][0], mst[j][1])]

                if coefs_scaled[j, i] != 0:
                    # Plot contributions
                    x_c, y_c = (
                        math.cos(a) * coefs_scaled[j, i],
                        math.sin(a) * coefs_scaled[j, i],
                    )
                    lbl = labels[i]
                    col = self.colors[lbl]
                    arrow = ax.arrow(
                        x_center,
                        y_center,
                        x_c,
                        y_c,
                        width=arrow_width,
                        color=col,
                        label=lbl,
                        zorder=15,
                    )

                    if arrows.get(lbl) is None:
                        arrows[lbl] = arrow

        return list(arrows.values()), list(arrows.keys())

    def _plot_between_clusters(
        self,
        ax,
        standartize_coef,
        univar_importance,
        feature_significance,
        scale_circle,
        move_circle,
        annotate,
        arrow_width,
        plot_scatter,
        plot_hulls,
        plot_top_k,
    ):
        dist_clusters = self.low_dim_data["cluster"].unique()
        dist_clusters.sort()
        dist_clusters = dist_clusters[1:] if dist_clusters[0] == -1 else dist_clusters

        cl_means = self._get_cluster_centers(dist_clusters)
        adj_matrix = self._build_adj_matrix(cl_means)
        mst = self._build_MST(len(dist_clusters), adj_matrix)

        mst_proj, angles = [], []
        for val in mst:
            angle = math.degrees(
                self._get_angle_to_x(cl_means[val[0]], cl_means[val[1]])
            )
            proj = self._project_line(
                self.low_dim_data,
                angle,
                point_a=cl_means[val[0]],
                point_b=cl_means[val[1]],
            )
            mst_proj.append(proj)
            angles.append(angle)

        self.mst_proj = np.array(mst_proj).T
        coefs, _, is_significant, std_x, std_y = self._get_importance(
            self.high_dim_data.to_numpy(),
            self.mst_proj,
            univar=univar_importance,
            significance=feature_significance,
        )

        if plot_top_k != 0:
            min_ix = np.argsort(coefs * is_significant, axis=1)[::-1][:, 0:plot_top_k]
            for i in range(is_significant.shape[0]):
                for j in range(is_significant.shape[1]):
                    if j not in min_ix[i]:
                        is_significant[i, j] = False
        if is_significant.sum() == 0:
            return [], []

        if standartize_coef:
            coefs = coefs / std_x
            if np.isnan(coefs).any():
                coefs = np.nan_to_num(coefs)
                warnings.warn(
                    "NaNs were introduced after standartizing coefficients and were replaced by 0."
                )

        for c, is_s in zip(coefs, is_significant):
            c[is_s == False] = 0

        arrows, labels = self._plot_between(
            ax,
            self.low_dim_data,
            cl_means,
            mst,
            coefs,
            self.observations,
            angles,
            scale_circle,
            move_circle,
            annotate,
            arrow_width,
            plot_scatter,
            plot_hulls,
        )

        return arrows, labels

    def get_num_clusters(self):
        return self.low_dim_data["cluster"].nunique()

    def plot_global_clock(
        self,
        ax: matplotlib.axis.Axis,
        standartize_data: bool = True,
        standartize_coef: bool = True,
        biggest_arrow_method: bool = True,
        angle_shift: int = 5,
        univar_importance: bool = True,
        feature_significance: float = 0.05,
        scale_circle: float = 1,
        move_circle: list = [0, 0],
        annotate: float = 0.6,
        arrow_width: float = 0.1,
        plot_scatter: bool = True,
        plot_top_k: int = 0,
    ):
        if not self.is_projected or self.shift != angle_shift:
            self.is_projected = True
            self.shift = angle_shift
            self._prepare_data(standartize_data=standartize_data)

            if biggest_arrow_method:
                self._create_biggest_projections()
            else:
                self._create_angles_projections(angle_shift=angle_shift)

        if plot_top_k < 0:
            raise Exception(
                f"Invalid value for plot_top_k: {plot_top_k} (0 - plot all, ncol > k > 0)."
            )

        if plot_top_k > self.high_dim_data.shape[1]:
            raise Exception(
                f"Invalid value for plot_top_k: {plot_top_k} (0 - plot all, ncol > k > 0). plot_top_k greater than umber of columns."
            )

        return self._plot_big_clock(
            ax=ax,
            standartize_coef=standartize_coef,
            univar_importance=univar_importance,
            feature_significance=feature_significance,
            biggest_arrow=biggest_arrow_method,
            scale_circle=scale_circle,
            move_circle=move_circle,
            annotate=annotate,
            arrow_width=arrow_width,
            plot_scatter=plot_scatter,
            plot_top_k=plot_top_k,
        )

    def plot_local_clocks(
        self,
        ax: matplotlib.axis.Axis,
        standartize_data: bool = True,
        standartize_coef: bool = True,
        biggest_arrow_method: bool = True,
        angle_shift: int = 5,
        univar_importance: bool = True,
        feature_significance: float = 0.05,
        scale_circles: list = [],
        move_circles: list = [],
        annotates: list = [],
        arrow_width: float = 0.1,
        plot_scatter: bool = True,
        plot_hulls: bool = True,
        plot_top_k: int = 0,
    ):
        if not self.is_projected or self.shift != angle_shift:
            self.is_projected = True
            self.shift = angle_shift
            self._prepare_data(standartize_data=standartize_data)

            if biggest_arrow_method:
                self._create_biggest_projections()
            else:
                self._create_angles_projections(angle_shift=angle_shift)

        n_clusters = self.low_dim_data["cluster"].nunique()
        if n_clusters != len(scale_circles) != len(move_circles) != len(annotates):
            raise Exception(
                f"Length of annotates, move_circles, scale_circles should be equal number of clusters {n_clusters}."
            )

        if len(scale_circles) == len(move_circles) == len(annotates) == 0:
            for _ in range(n_clusters):
                scale_circles.append(1)
                move_circles.append([0, 0])
                annotates.append(0.3)

        if plot_top_k < 0:
            raise Exception(
                f"Invalid value for plot_top_k: {plot_top_k} (0 - plot all, ncol > k > 0)."
            )

        if plot_top_k > self.high_dim_data.shape[1]:
            raise Exception(
                f"Invalid value for plot_top_k: {plot_top_k} (0 - plot all, ncol > k > 0). plot_top_k greater than umber of columns."
            )

        arrows_all, arrow_labels_all = self._plot_small_clock(
            ax=ax,
            standartize_coef=standartize_coef,
            univar_importance=univar_importance,
            feature_significance=feature_significance,
            biggest_arrow=biggest_arrow_method,
            scale_circle=scale_circles,
            move_circle=move_circles,
            annotate=annotates,
            arrow_width=arrow_width,
            plot_scatter=plot_scatter,
            plot_hulls=plot_hulls,
            plot_top_k=plot_top_k,
        )
        return arrows_all, arrow_labels_all

    def plot_between_clock(
        self,
        ax: matplotlib.axis.Axis,
        standartize_data: bool = True,
        standartize_coef: bool = True,
        angle_shift: int = 5,
        univar_importance: bool = True,
        feature_significance: float = 0.05,
        scale_circles: list = [],
        move_circles: list = [],
        annotates: list = [],
        arrow_width: float = 0.03,
        plot_scatter: bool = True,
        plot_hulls: bool = True,
        plot_top_k: int = 0,
    ):

        if not self.is_projected or self.shift != angle_shift:
            self.is_projected = True
            self.shift = angle_shift
            self._prepare_data(standartize_data=standartize_data)
            self._create_angles_projections(angle_shift=angle_shift)

        if plot_top_k < 0:
            raise Exception(
                f"Invalid value for plot_top_k: {plot_top_k} (0 - plot all, ncol > k > 0)."
            )

        if plot_top_k > self.high_dim_data.shape[1]:
            raise Exception(
                f"Invalid value for plot_top_k: {plot_top_k} (0 - plot all, ncol > k > 0). plot_top_k greater than umber of columns."
            )

        n_clusters = self.low_dim_data["cluster"].nunique() - 1
        if n_clusters != len(scale_circles) != len(move_circles) != len(annotates):
            raise Exception(
                f"Length of annotates, move_circles, scale_circles should be equal number of clusters {n_clusters}."
            )

        if len(scale_circles) == len(move_circles) == len(annotates) == 0:
            for _ in range(n_clusters):
                scale_circles.append(1)
                move_circles.append([0, 0])
                annotates.append(0.3)

        arrows, labels = self._plot_between_clusters(
            ax=ax,
            standartize_coef=standartize_coef,
            univar_importance=univar_importance,
            feature_significance=feature_significance,
            scale_circle=scale_circles,
            move_circle=move_circles,
            annotate=annotates,
            arrow_width=arrow_width,
            plot_scatter=plot_scatter,
            plot_hulls=plot_hulls,
            plot_top_k=plot_top_k,
        )

        return arrows, labels
