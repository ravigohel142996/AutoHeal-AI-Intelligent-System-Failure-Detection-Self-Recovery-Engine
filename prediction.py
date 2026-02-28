"""
Failure probability prediction using Random Forest.

The predictor is trained on labelled history where labels are derived from
anomaly detector scores (threshold-based).  It outputs a continuous failure
probability in [0, 1], and maintains a rolling-window estimate used by the
dashboard to visualise trend.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import (
    RANDOM_FOREST_MAX_DEPTH,
    RANDOM_FOREST_N_ESTIMATORS,
    RANDOM_FOREST_RANDOM_STATE,
    ROLLING_WINDOW,
    MetricSnapshot,
)

# Anomaly score threshold used to derive binary training labels
ANOMALY_LABEL_THRESHOLD: float = 0.50
# Minimum samples needed before fitting
MIN_FIT_SAMPLES: int = 30


class FailurePredictor:
    """
    Random Forest classifier that estimates the probability of imminent
    system failure from raw metric snapshots.

    Labels are derived automatically from anomaly detector scores so no
    human annotation is required during simulation.

    Parameters
    ----------
    n_estimators : int
        Number of trees in the Random Forest.
    max_depth : int
        Maximum tree depth (prevents overfitting on small datasets).
    random_state : int
        Reproducibility seed.
    rolling_window : int
        Window size for computing the rolling average failure probability.
    """

    def __init__(
        self,
        n_estimators: int = RANDOM_FOREST_N_ESTIMATORS,
        max_depth: int = RANDOM_FOREST_MAX_DEPTH,
        random_state: int = RANDOM_FOREST_RANDOM_STATE,
        rolling_window: int = ROLLING_WINDOW,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._random_state = random_state
        self._rolling_window = rolling_window
        self._model: Optional[RandomForestClassifier] = None
        self._is_fitted: bool = False

        self._probability_history: List[float] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    @property
    def probability_history(self) -> List[float]:
        return list(self._probability_history)

    def fit(
        self,
        history: List[MetricSnapshot],
        anomaly_scores: List[float],
    ) -> None:
        """
        Fit (or re-fit) the Random Forest on labelled metric history.

        Parameters
        ----------
        history : list of MetricSnapshot
            Feature samples.
        anomaly_scores : list of float
            Corresponding anomaly scores used to derive binary labels.

        Raises
        ------
        ValueError
            If input lengths mismatch or too few samples provided.
        """
        if len(history) != len(anomaly_scores):
            raise ValueError(
                "history and anomaly_scores must have the same length."
            )
        if len(history) < MIN_FIT_SAMPLES:
            raise ValueError(
                f"Need at least {MIN_FIT_SAMPLES} samples to fit; "
                f"got {len(history)}."
            )

        X = np.array([s.as_feature_vector() for s in history])
        y = np.array(
            [1 if score >= ANOMALY_LABEL_THRESHOLD else 0
             for score in anomaly_scores]
        )

        # Guard against single-class datasets (model would not learn)
        if len(np.unique(y)) < 2:
            # Flip one label to ensure both classes are present
            y[0] = 1 - y[0]

        self._model = RandomForestClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            random_state=self._random_state,
        )
        self._model.fit(X, y)
        self._is_fitted = True

    def predict_proba(self, snapshot: MetricSnapshot) -> float:
        """
        Return failure probability in [0, 1] for a single snapshot.

        Returns 0.0 if the model has not yet been fitted.
        """
        if not self._is_fitted or self._model is None:
            return 0.0

        x = np.array(snapshot.as_feature_vector()).reshape(1, -1)
        prob = self._model.predict_proba(x)[0]
        # Index 1 corresponds to the 'failure' class
        classes = list(self._model.classes_)
        failure_idx = classes.index(1) if 1 in classes else -1
        if failure_idx < 0:
            return 0.0
        raw = float(prob[failure_idx])
        self._probability_history.append(raw)
        return raw

    def rolling_failure_probability(self) -> float:
        """
        Return the rolling-window average failure probability.

        Uses the last ``rolling_window`` predictions stored internally.
        Returns 0.0 if no predictions have been made yet.
        """
        if not self._probability_history:
            return 0.0
        window = self._probability_history[-self._rolling_window:]
        return float(np.mean(window))

    def feature_importances(self) -> Optional[pd.Series]:
        """
        Return a Series of feature importances if the model is fitted.
        """
        if not self._is_fitted or self._model is None:
            return None
        names = [
            "cpu_usage", "memory_usage", "disk_io", "network_latency",
            "error_rate", "service_availability", "response_time",
        ]
        return pd.Series(
            self._model.feature_importances_,
            index=names,
        ).sort_values(ascending=False)

    def build_probability_series(
        self,
        history: List[MetricSnapshot],
    ) -> Tuple[List[int], List[float]]:
        """
        Score an entire history batch and return (steps, probabilities).

        Useful for populating chart data after model fitting.
        """
        if not self._is_fitted or self._model is None:
            return [], []

        steps = [s.step for s in history]
        X = np.array([s.as_feature_vector() for s in history])
        proba_matrix = self._model.predict_proba(X)
        classes = list(self._model.classes_)
        failure_idx = classes.index(1) if 1 in classes else -1
        if failure_idx < 0:
            return steps, [0.0] * len(steps)

        probs = [float(row[failure_idx]) for row in proba_matrix]
        return steps, probs
