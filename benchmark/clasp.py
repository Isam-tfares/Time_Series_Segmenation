import os
import warnings
from queue import PriorityQueue

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError

from claspy.clasp import ClaSPEnsemble
from claspy.utils import check_input_time_series, check_excl_radius
from claspy.window_size import map_window_size_methods


class BinaryClaSPSegmentation:

    def __init__(self, n_segments="learn", n_estimators=10, window_size="suss", k_neighbours=3,
                 distance="znormed_euclidean_distance", score="roc_auc",
                 early_stopping=True, validation="significance_test", threshold=1e-15, excl_radius=5,
                 n_jobs=-1, random_state=2357):
        self.n_segments = n_segments
        self.n_estimators = n_estimators
        self.window_size = window_size
        self.k_neighbours = k_neighbours
        self.distance = distance
        self.validation = validation
        self.threshold = threshold
        self.score = score
        self.early_stopping = early_stopping
        self.excl_radius = excl_radius
        self.n_jobs = os.cpu_count() if n_jobs < 1 else n_jobs
        self.random_state = random_state
        self.is_fitted = False

        check_excl_radius(k_neighbours, excl_radius)

    def _cp_is_valid(self, candidate, change_points):
       
        for change_point in [0] + change_points + [self.n_timepoints]:
            left_begin = max(0, change_point - self.min_seg_size)
            right_end = min(self.n_timepoints, change_point + self.min_seg_size)
            if candidate in range(left_begin, right_end): return False

        return True

    def _local_segmentation(self, lbound, ubound, change_points):
       
        if ubound - lbound < 2 * self.min_seg_size: return

        clasp = ClaSPEnsemble(
            n_estimators=self.n_estimators,
            window_size=self.window_size,
            k_neighbours=self.k_neighbours,
            distance=self.distance,
            score=self.score,
            early_stopping=self.early_stopping,
            excl_radius=self.excl_radius,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        ).fit(self.time_series[lbound:ubound], validation=self.validation, threshold=self.threshold)

        cp = clasp.split(validation=self.validation, threshold=self.threshold)
        if cp is None: return
        score = clasp.profile[cp]

        if not self._cp_is_valid(lbound + cp, change_points): return

        self.clasp_tree.append(((lbound, ubound), clasp))
        self.queue.put((-score, len(self.clasp_tree) - 1))

    def _check_is_fitted(self):
      
        if not self.is_fitted:
            raise NotFittedError(
                "BinaryClaSPSegmentation object is not fitted yet. Please fit the object before using this method.")

    def fit(self, time_series):
       
        check_input_time_series(time_series)

        if isinstance(self.window_size, str):
            self.window_size = max(1, map_window_size_methods(self.window_size)(time_series) // 2)

        self.min_seg_size = self.window_size * self.excl_radius

        if time_series.shape[0] < 2 * self.min_seg_size:
            warnings.warn(
                "Time series must at least have 2*min_seg_size data points for segmentation. Try setting "
                "a smaller window size.")
            self.n_segments = 1

        self.time_series = time_series
        self.n_timepoints = time_series.shape[0]

        self.queue = PriorityQueue()
        self.clasp_tree = []

        if self.n_segments == "learn":
            self.n_segments = time_series.shape[0] // self.min_seg_size

        if self.n_segments > 1:
            prange = 0, time_series.shape[0]
            clasp = ClaSPEnsemble(
                n_estimators=self.n_estimators,
                window_size=self.window_size,
                k_neighbours=self.k_neighbours,
                distance=self.distance,
                score=self.score,
                early_stopping=self.early_stopping,
                excl_radius=self.excl_radius,
                n_jobs=self.n_jobs,
                random_state=self.random_state
            ).fit(time_series, validation=self.validation, threshold=self.threshold)

            cp = clasp.split(validation=self.validation, threshold=self.threshold)

            if cp is not None:
                self.clasp_tree.append((prange, clasp))
                self.queue.put((-clasp.profile[cp], len(self.clasp_tree) - 1))

            profile = clasp.profile
        else:
            profile = np.full(shape=self.n_timepoints - self.window_size + 1, fill_value=-np.inf, dtype=np.float64)

        change_points = []
        scores = []

        for idx in range(self.n_segments - 1):
            # happens if no valid change points exist anymore
            if self.queue.empty() is True: break

            priority, clasp_tree_idx = self.queue.get()
            (lbound, ubound), clasp = self.clasp_tree[clasp_tree_idx]
            cp = lbound + clasp.split(validation=self.validation, threshold=self.threshold)

            profile[lbound:ubound - self.window_size + 1] = np.max(
                [profile[lbound:ubound - self.window_size + 1], clasp.profile], axis=0)

            change_points.append(cp)
            scores.append(-priority)

            if len(change_points) == self.n_segments - 1: break

            lrange, rrange = (lbound, cp), (cp, ubound)

            for prange in (lrange, rrange):
                self._local_segmentation(*prange, change_points)

        sorted_cp_args = np.argsort(change_points)
        self.change_points, self.scores = np.asarray(change_points)[sorted_cp_args], np.asarray(scores)[sorted_cp_args]

        profile[np.isinf(profile)] = np.nan
        self.profile = pd.Series(profile).interpolate(limit_direction="both").to_numpy()

        self.is_fitted = True
        return self

    def predict(self, sparse=True):
      
        self._check_is_fitted()

        if sparse is True:
            return self.change_points

        seg_idx = np.concatenate(([0], self.change_points, [self.time_series.shape[0]]))
        return [self.time_series[seg_idx[idx]:seg_idx[idx + 1]] for idx in range(len(seg_idx) - 1)]

    def fit_predict(self, time_series, sparse=True):
       
        return self.fit(time_series).predict(sparse)

    def plot(self, gt_cps=None, heading=None, ts_name=None, fig_size=(20, 10), font_size=26, file_path=None):
        
        self._check_is_fitted()
        fig, (ax1, ax2) = plt.subplots(2, sharex=True, gridspec_kw={"hspace": .05}, figsize=fig_size)

        if gt_cps is not None:
            segments = [0] + gt_cps.tolist() + [self.time_series.shape[0]]
            for idx in np.arange(0, len(segments) - 1):
                ax1.plot(np.arange(segments[idx], segments[idx + 1]), self.time_series[segments[idx]:segments[idx + 1]])

            ax2.plot(np.arange(self.profile.shape[0]), self.profile, color="black")
        else:
            ax1.plot(np.arange(self.time_series.shape[0]), self.time_series)
            ax2.plot(np.arange(self.profile.shape[0]), self.profile, color="black")

        if heading is not None:
            ax1.set_title(heading, fontsize=font_size)

        if ts_name is not None:
            ax1.set_ylabel(ts_name, fontsize=font_size)

        ax2.set_xlabel("split point", fontsize=font_size)
        ax2.set_ylabel("ClaSP Score", fontsize=font_size)

        for ax in (ax1, ax2):
            for tick in ax.xaxis.get_major_ticks():
                tick.label1.set_fontsize(font_size)

            for tick in ax.yaxis.get_major_ticks():
                tick.label1.set_fontsize(font_size)

        if gt_cps is not None:
            for idx, true_cp in enumerate(gt_cps):
                ax1.axvline(x=true_cp, linewidth=2, color="r", label=f"True Change Point" if idx == 0 else None)
                ax2.axvline(x=true_cp, linewidth=2, color="r", label="True Change Point" if idx == 0 else None)

        for idx, found_cp in enumerate(self.change_points):
            ax1.axvline(x=found_cp, linewidth=2, color="g", label="Predicted Change Point" if idx == 0 else None)
            ax2.axvline(x=found_cp, linewidth=2, color="g", label="Predicted Change Point" if idx == 0 else None)

        ax1.legend(prop={"size": font_size})

        if file_path is not None:
            plt.savefig(file_path, bbox_inches="tight")

        return ax1, ax2