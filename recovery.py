"""
Recovery engine.

Evaluates the current system state and, when risk exceeds the configured
threshold, selects and executes the most appropriate recovery action.

Each action:
  - Has a quantified cost
  - Modifies simulator state via a bias injection
  - Returns a detailed RecoveryEvent for logging/audit
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from config import (
    HEALTH_WEIGHTS,
    METRIC_BASELINES,
    RECOVERY_ACTIONS,
    RECOVERY_RISK_THRESHOLD,
    MetricSnapshot,
    RecoveryEvent,
)


def compute_health_score(snapshot: MetricSnapshot) -> float:
    """
    Calculate a composite health score in [0, 1].

    Each metric is normalised against its baseline range and then combined
    using the weights defined in ``HEALTH_WEIGHTS``.  A score of 1.0 means
    all metrics are at their ideal baseline midpoints; 0.0 means critical.

    Parameters
    ----------
    snapshot : MetricSnapshot
        Current metric readings.

    Returns
    -------
    float
        Health score in [0, 1].
    """
    values = {
        "cpu_usage":            snapshot.cpu_usage,
        "memory_usage":         snapshot.memory_usage,
        "disk_io":              snapshot.disk_io,
        "network_latency":      snapshot.network_latency,
        "error_rate":           snapshot.error_rate,
        "service_availability": snapshot.service_availability,
        "response_time":        snapshot.response_time,
    }

    score_components: List[float] = []
    for metric, weight in HEALTH_WEIGHTS.items():
        lo, hi = METRIC_BASELINES[metric]
        raw = values[metric]

        if metric == "service_availability":
            # Higher is better; normalise so hi -> 1, lo -> 0
            normalised = (raw - lo) / max(hi - lo, 1e-9)
        else:
            # Lower is better; normalise so lo -> 1, hi -> 0
            normalised = 1.0 - (raw - lo) / max(hi - lo, 1e-9)

        score_components.append(weight * float(np.clip(normalised, 0.0, 1.0)))

    return float(np.clip(sum(score_components), 0.0, 1.0))


class RecoveryEngine:
    """
    Monitors system health and triggers corrective actions when required.

    The engine selects recovery actions by scoring each candidate against the
    current degraded metrics – actions that directly address the worst offenders
    are prioritised.  A simple cost-adjusted score prevents the most expensive
    actions from being selected when cheaper alternatives are adequate.

    Parameters
    ----------
    risk_threshold : float
        Failure probability above which recovery is triggered.
    """

    def __init__(self, risk_threshold: float = RECOVERY_RISK_THRESHOLD) -> None:
        self._risk_threshold = risk_threshold
        self._actions: List[Dict] = RECOVERY_ACTIONS
        self._last_action_step: int = -1
        self._cooldown_steps: int = 5   # minimum steps between consecutive actions

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def should_trigger(self, failure_probability: float) -> bool:
        """Return True if risk exceeds the configured threshold."""
        return failure_probability >= self._risk_threshold

    def select_action(
        self,
        snapshot: MetricSnapshot,
        step: int,
    ) -> Optional[Dict]:
        """
        Choose the best recovery action for the current system state.

        Selection is based on how well each action's target deltas compensate
        for the observed metric deviations, adjusted by action cost.  Returns
        None if the cooldown period has not elapsed.

        Parameters
        ----------
        snapshot : MetricSnapshot
            Current metrics.
        step : int
            Current simulation step (used for cooldown enforcement).
        """
        if step - self._last_action_step < self._cooldown_steps:
            return None

        metric_values = {
            "cpu_usage":            snapshot.cpu_usage,
            "memory_usage":         snapshot.memory_usage,
            "disk_io":              snapshot.disk_io,
            "network_latency":      snapshot.network_latency,
            "error_rate":           snapshot.error_rate,
            "service_availability": snapshot.service_availability,
            "response_time":        snapshot.response_time,
        }

        best_action: Optional[Dict] = None
        best_score: float = -np.inf

        for action in self._actions:
            relevance = 0.0
            for metric, delta in action["targets"].items():
                lo, hi = METRIC_BASELINES[metric]
                current = metric_values.get(metric, 0.0)
                midpoint = (lo + hi) / 2.0

                if metric == "service_availability":
                    deviation = max(0.0, midpoint - current)
                else:
                    deviation = max(0.0, current - midpoint)

                # Positive delta on availability or negative delta on others
                # are both "helpful"
                if metric == "service_availability":
                    helpful = delta > 0
                else:
                    helpful = delta < 0

                if helpful:
                    relevance += deviation * abs(delta)

            # Cost penalty: prefer low-cost actions when relevance is similar
            adjusted = relevance / (action["cost"] + 0.1)

            if adjusted > best_score:
                best_score = adjusted
                best_action = action

        self._last_action_step = step
        return best_action

    def execute(
        self,
        action: Dict,
        snapshot: MetricSnapshot,
        failure_probability: float,
        step: int,
    ) -> Tuple[RecoveryEvent, Dict[str, float]]:
        """
        Execute a recovery action and compute the metric impact delta.

        Parameters
        ----------
        action : dict
            Action definition from ``RECOVERY_ACTIONS``.
        snapshot : MetricSnapshot
            Pre-recovery metric snapshot.
        failure_probability : float
            Current failure probability (used in event record).
        step : int
            Current simulation step.

        Returns
        -------
        (RecoveryEvent, dict)
            The audit record and the raw bias deltas to apply to the simulator.
        """
        health_before = compute_health_score(snapshot)
        targets: Dict[str, float] = action["targets"]

        # Scale impact by actual severity – if metrics are already healthy,
        # recovery impact is proportionally smaller
        metric_values = {
            "cpu_usage":            snapshot.cpu_usage,
            "memory_usage":         snapshot.memory_usage,
            "disk_io":              snapshot.disk_io,
            "network_latency":      snapshot.network_latency,
            "error_rate":           snapshot.error_rate,
            "service_availability": snapshot.service_availability,
            "response_time":        snapshot.response_time,
        }

        scaled_targets: Dict[str, float] = {}
        for metric, delta in targets.items():
            lo, hi = METRIC_BASELINES[metric]
            midpoint = (lo + hi) / 2.0
            current = metric_values.get(metric, midpoint)

            if metric == "service_availability":
                deviation = max(0.0, midpoint - current) / max(midpoint, 1e-9)
            else:
                deviation = max(0.0, current - midpoint) / max(hi - midpoint, 1e-9)

            # Scale factor: full impact when deviation is high, diminished otherwise
            scale = 0.5 + 0.5 * float(np.clip(deviation, 0.0, 1.0))
            scaled_targets[metric] = delta * scale

        event = RecoveryEvent(
            step=step,
            action_name=action["name"],
            action_label=action["label"],
            trigger_score=health_before,
            failure_probability=failure_probability,
            impact_delta=dict(scaled_targets),
            cost=action["cost"],
        )

        return event, scaled_targets

    def compute_stability_index(
        self,
        health_scores: List[float],
    ) -> float:
        """
        Compute a stability index from the health score history.

        The stability index rewards both a high mean health score and low
        variance – a system oscillating between 0.4 and 0.9 is less stable
        than one that holds at 0.75.

        Returns a value in [0, 1].
        """
        if not health_scores:
            return 0.0
        arr = np.array(health_scores)
        mean_score = float(np.mean(arr))
        std_score = float(np.std(arr))
        # Penalise variance; scale factor of 2 empirically balances the terms
        stability = mean_score * (1.0 - 2.0 * std_score)
        return float(np.clip(stability, 0.0, 1.0))
