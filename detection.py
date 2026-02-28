"""
Anomaly detection layer using Isolation Forest.

The detector is trained in an online/batch fashion: once the history buffer
contains enough samples it fits (or re-fits) an Isolation Forest and scores
every new snapshot.  Anomaly scores are mapped to a [0, 1] range where values
closer to 1 indicate higher anomaly likelihood.
"""

from typing import List, Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from config import (
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_N_ESTIMATORS,
    ISOLATION_FOREST_RANDOM_STATE,
    MetricSnapshot,
)

# Minimum history length required before the model can be fitted
MIN_FIT_SAMPLES: int = 20


class AnomalyDetector:
    """
    Wraps an Isolation Forest to detect anomalous metric snapshots.

    The model is lazily fitted the first time enough history has accumulated
    and can be re-fitted at any point to incorporate new distribution shifts.

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies in the training data.
    n_estimators : int
        Number of trees in the Isolation Forest.
    random_state : int
        Reproducibility seed.
    """

    def __init__(
        self,
        contamination: float = ISOLATION_FOREST_CONTAMINATION,
        n_estimators: int = ISOLATION_FOREST_N_ESTIMATORS,
        random_state: int = ISOLATION_FOREST_RANDOM_STATE,
    ) -> None:
        self._contamination = contamination
        self._n_estimators = n_estimators
        self._random_state = random_state
        self._model: Optional[IsolationForest] = None
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def fit(self, history: List[MetricSnapshot]) -> None:
        """
        Fit the Isolation Forest on the provided history.

        Parameters
        ----------
        history : list of MetricSnapshot
            Training samples.  Requires at least ``MIN_FIT_SAMPLES`` entries.

        Raises
        ------
        ValueError
            If fewer than ``MIN_FIT_SAMPLES`` snapshots are provided.
        """
        if len(history) < MIN_FIT_SAMPLES:
            raise ValueError(
                f"Need at least {MIN_FIT_SAMPLES} samples to fit; "
                f"got {len(history)}."
            )

        X = np.array([s.as_feature_vector() for s in history])
        self._model = IsolationForest(
            n_estimators=self._n_estimators,
            contamination=self._contamination,
            random_state=self._random_state,
        )
        self._model.fit(X)
        self._is_fitted = True

    def score(self, snapshot: MetricSnapshot) -> float:
        """
        Return anomaly score in [0, 1] for a single snapshot.

        A score near 1 means highly anomalous; near 0 means normal.
        Returns 0.0 if the model has not yet been fitted.

        Parameters
        ----------
        snapshot : MetricSnapshot
            Snapshot to score.
        """
        if not self._is_fitted or self._model is None:
            return 0.0

        x = np.array(snapshot.as_feature_vector()).reshape(1, -1)
        # decision_function returns negative values for anomalies;
        # map to [0, 1] via sigmoid-like normalisation
        raw_score = self._model.decision_function(x)[0]
        # Invert and normalise: anomalies have lower (more negative) raw scores
        anomaly_score = float(1.0 / (1.0 + np.exp(5.0 * raw_score)))
        return float(np.clip(anomaly_score, 0.0, 1.0))

    def score_batch(self, history: List[MetricSnapshot]) -> List[float]:
        """
        Score a sequence of snapshots.

        Returns a list of anomaly scores aligned with the input list.
        """
        if not self._is_fitted or self._model is None:
            return [0.0] * len(history)

        X = np.array([s.as_feature_vector() for s in history])
        raw_scores = self._model.decision_function(X)
        scores = 1.0 / (1.0 + np.exp(5.0 * raw_scores))
        return [float(np.clip(v, 0.0, 1.0)) for v in scores]
