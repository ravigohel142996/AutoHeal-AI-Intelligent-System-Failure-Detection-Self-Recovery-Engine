"""
Infrastructure simulation engine.

Drives a time-step based synthetic environment whose metric evolution is
governed by configurable baselines, Gaussian noise, and occasional anomaly
injections.  Each call to ``step()`` returns a new ``MetricSnapshot`` and
optionally modifies an active recovery bias that persists for a configurable
number of future steps.
"""

import random
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import (
    ANOMALY_INJECT_PROB,
    ANOMALY_MAGNITUDE_RANGE,
    METRIC_BASELINES,
    NOISE_STDDEV,
    RECOVERY_PERSISTENCE,
    SIMULATION_SEED,
    MetricSnapshot,
    RecoveryEvent,
)

METRIC_NAMES: List[str] = list(METRIC_BASELINES.keys())


class InfrastructureSimulator:
    """
    Simulates a multi-metric cloud infrastructure environment.

    The simulator maintains an internal state vector (current metric values)
    and advances it one step at a time.  Recovery actions can inject a
    persistent bias that decays over subsequent steps.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    anomaly_prob : float
        Per-step probability that a synthetic anomaly spike is injected.
    noise_stddev : float
        Standard deviation of baseline Gaussian noise added each step.
    """

    def __init__(
        self,
        seed: int = SIMULATION_SEED,
        anomaly_prob: float = ANOMALY_INJECT_PROB,
        noise_stddev: float = NOISE_STDDEV,
    ) -> None:
        self._rng = np.random.RandomState(seed)
        random.seed(seed)

        self._anomaly_prob = anomaly_prob
        self._noise_stddev = noise_stddev

        # Current metric state – initialise to midpoint of each baseline
        self._state: Dict[str, float] = {
            name: (lo + hi) / 2.0
            for name, (lo, hi) in METRIC_BASELINES.items()
        }

        # Recovery bias: additive correction applied each step, decays
        self._recovery_bias: Dict[str, float] = {n: 0.0 for n in METRIC_NAMES}
        self._bias_decay: float = 1.0 - RECOVERY_PERSISTENCE

        self._step_index: int = 0
        self._history: List[MetricSnapshot] = []
        self._recovery_log: List[RecoveryEvent] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def history(self) -> List[MetricSnapshot]:
        return list(self._history)

    @property
    def recovery_log(self) -> List[RecoveryEvent]:
        return list(self._recovery_log)

    def current_snapshot(self) -> Optional[MetricSnapshot]:
        """Return the most recently produced snapshot, or None."""
        return self._history[-1] if self._history else None

    def step(self) -> MetricSnapshot:
        """
        Advance simulation by one time-step.

        Returns
        -------
        MetricSnapshot
            Metric values for the current step.
        """
        inject_anomaly = self._rng.random() < self._anomaly_prob
        anomaly_target: Optional[str] = None

        if inject_anomaly:
            anomaly_target = random.choice(METRIC_NAMES)

        new_state: Dict[str, float] = {}
        for name, value in self._state.items():
            lo, hi = METRIC_BASELINES[name]

            # Mean-reversion nudge toward midpoint
            midpoint = (lo + hi) / 2.0
            reversion = 0.05 * (midpoint - value)

            # Gaussian noise
            noise = self._rng.normal(0.0, self._noise_stddev)

            # Anomaly spike on a randomly chosen metric
            spike = 0.0
            if inject_anomaly and name == anomaly_target:
                mag = self._rng.uniform(*ANOMALY_MAGNITUDE_RANGE)
                # Spike direction: push metrics that should be low upward
                # and service_availability downward
                if name == "service_availability":
                    spike = -mag
                else:
                    spike = mag

            # Recovery bias (decays each step)
            bias = self._recovery_bias[name]

            raw = value + reversion + noise + spike + bias
            new_state[name] = float(np.clip(raw, lo - 0.05, hi + 0.50))

        # Clamp to absolute [0, 1] domain
        for name in new_state:
            new_state[name] = float(np.clip(new_state[name], 0.0, 1.0))

        self._state = new_state

        # Decay recovery bias
        for name in self._recovery_bias:
            self._recovery_bias[name] *= self._bias_decay

        snapshot = MetricSnapshot(
            step=self._step_index,
            cpu_usage=new_state["cpu_usage"],
            memory_usage=new_state["memory_usage"],
            disk_io=new_state["disk_io"],
            network_latency=new_state["network_latency"],
            error_rate=new_state["error_rate"],
            service_availability=new_state["service_availability"],
            response_time=new_state["response_time"],
        )

        self._history.append(snapshot)
        self._step_index += 1
        return snapshot

    def apply_recovery_bias(self, targets: Dict[str, float]) -> None:
        """
        Inject a persistent metric correction from a recovery action.

        Parameters
        ----------
        targets : dict
            Mapping of metric name -> signed delta to add.  Positive values
            push the metric up; negative values push it down.
        """
        for metric, delta in targets.items():
            if metric in self._recovery_bias:
                self._recovery_bias[metric] += delta

    def log_recovery_event(self, event: RecoveryEvent) -> None:
        self._recovery_log.append(event)

    def to_dataframe(self) -> pd.DataFrame:
        """Return the full metric history as a tidy DataFrame."""
        if not self._history:
            return pd.DataFrame()
        records = [
            {
                "step": s.step,
                "cpu_usage": s.cpu_usage,
                "memory_usage": s.memory_usage,
                "disk_io": s.disk_io,
                "network_latency": s.network_latency,
                "error_rate": s.error_rate,
                "service_availability": s.service_availability,
                "response_time": s.response_time,
            }
            for s in self._history
        ]
        return pd.DataFrame(records)
