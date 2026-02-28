"""
Configuration constants and data model definitions for AutoHeal AI.

All tuneable parameters live here. No magic numbers elsewhere in the codebase.
"""

from dataclasses import dataclass, field
from typing import Dict, List


# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------

SIMULATION_SEED: int = 42
DEFAULT_STEPS: int = 120          # time-steps in one simulation run
ANOMALY_INJECT_PROB: float = 0.08  # probability of a synthetic spike each step
NOISE_STDDEV: float = 0.03        # baseline Gaussian noise on each metric

# Metric baseline ranges (min, max) – normal operating window
METRIC_BASELINES: Dict[str, tuple] = {
    "cpu_usage":          (0.25, 0.65),
    "memory_usage":       (0.30, 0.70),
    "disk_io":            (0.10, 0.50),
    "network_latency":    (0.05, 0.30),
    "error_rate":         (0.00, 0.05),
    "service_availability": (0.90, 1.00),
    "response_time":      (0.05, 0.40),
}

# Spike magnitude range applied during anomaly injection
ANOMALY_MAGNITUDE_RANGE: tuple = (0.25, 0.50)

# ---------------------------------------------------------------------------
# Detection parameters
# ---------------------------------------------------------------------------

ISOLATION_FOREST_CONTAMINATION: float = 0.10
ISOLATION_FOREST_N_ESTIMATORS: int = 100
ISOLATION_FOREST_RANDOM_STATE: int = 42

# ---------------------------------------------------------------------------
# Prediction parameters
# ---------------------------------------------------------------------------

RANDOM_FOREST_N_ESTIMATORS: int = 100
RANDOM_FOREST_RANDOM_STATE: int = 42
RANDOM_FOREST_MAX_DEPTH: int = 8

# Rolling window used to compute the failure-probability trend
ROLLING_WINDOW: int = 10

# ---------------------------------------------------------------------------
# Health scoring
# ---------------------------------------------------------------------------

# Weight vector for the weighted health score (must sum to 1.0)
HEALTH_WEIGHTS: Dict[str, float] = {
    "cpu_usage":            0.20,
    "memory_usage":         0.20,
    "disk_io":              0.10,
    "network_latency":      0.10,
    "error_rate":           0.20,
    "service_availability": 0.10,
    "response_time":        0.10,
}

assert abs(sum(HEALTH_WEIGHTS.values()) - 1.0) < 1e-9, (
    f"HEALTH_WEIGHTS must sum to 1.0; got {sum(HEALTH_WEIGHTS.values()):.6f}"
)

# Threshold above which the recovery engine is triggered
RECOVERY_RISK_THRESHOLD: float = 0.60

# ---------------------------------------------------------------------------
# Recovery parameters
# ---------------------------------------------------------------------------

# Each recovery action is parameterised here – no magic numbers in recovery.py
RECOVERY_ACTIONS: List[Dict] = [
    {
        "name": "scale_resources",
        "label": "Scale Resources",
        "cost": 0.8,
        "targets": {"cpu_usage": -0.18, "memory_usage": -0.15},
        "description": "Provision additional compute capacity to absorb load.",
    },
    {
        "name": "restart_service",
        "label": "Restart Service",
        "cost": 0.5,
        "targets": {"error_rate": -0.60, "service_availability": 0.08,
                    "response_time": -0.20},
        "description": "Graceful service restart to clear degraded state.",
    },
    {
        "name": "reduce_load",
        "label": "Reduce Load",
        "cost": 0.4,
        "targets": {"cpu_usage": -0.12, "network_latency": -0.10,
                    "response_time": -0.12},
        "description": "Activate rate limiting to shed non-critical traffic.",
    },
    {
        "name": "allocate_backup",
        "label": "Allocate Backup",
        "cost": 0.9,
        "targets": {"service_availability": 0.06, "disk_io": -0.15,
                    "memory_usage": -0.10},
        "description": "Redirect workload to standby replica nodes.",
    },
    {
        "name": "optimize_traffic",
        "label": "Optimize Traffic Routing",
        "cost": 0.6,
        "targets": {"network_latency": -0.15, "response_time": -0.15,
                    "error_rate": -0.10},
        "description": "Re-balance requests across healthy endpoints.",
    },
]

# How strongly a recovery action reshapes the simulation going forward
RECOVERY_PERSISTENCE: float = 0.40


# ---------------------------------------------------------------------------
# UI / display
# ---------------------------------------------------------------------------

CHART_HEIGHT: int = 280
HISTORY_DISPLAY_STEPS: int = 80   # how many most-recent steps shown in charts


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class MetricSnapshot:
    """Immutable snapshot of all monitored metrics at a single time-step."""
    step: int
    cpu_usage: float
    memory_usage: float
    disk_io: float
    network_latency: float
    error_rate: float
    service_availability: float
    response_time: float

    def as_feature_vector(self) -> List[float]:
        return [
            self.cpu_usage,
            self.memory_usage,
            self.disk_io,
            self.network_latency,
            self.error_rate,
            self.service_availability,
            self.response_time,
        ]


@dataclass
class RecoveryEvent:
    """Record of a single recovery action execution."""
    step: int
    action_name: str
    action_label: str
    trigger_score: float
    failure_probability: float
    impact_delta: Dict[str, float] = field(default_factory=dict)
    cost: float = 0.0
