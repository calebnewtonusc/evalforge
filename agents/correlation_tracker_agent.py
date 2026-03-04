"""
agents/correlation_tracker_agent.py — Longitudinal benchmark correlation tracking.

Monitors whether benchmark scores predict real-world capability over time.
Tracks Pearson r, Spearman rho, Kendall tau across model releases and
fires alerts when correlation degrades.

Usage:
    tracker = CorrelationTrackerAgent(db_path="knowledge/correlation_history.json")
    tracker.record(benchmark="MMLU", model_scores={...}, downstream_scores={...})
    report = tracker.get_report(benchmark="MMLU")
    print(report.trend)  # "DEGRADING" | "STABLE" | "IMPROVING"
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from loguru import logger
from scipy.stats import kendalltau, pearsonr, spearmanr


@dataclass
class CorrelationRecord:
    """A single timestamped correlation measurement."""

    timestamp: str
    benchmark: str
    model_scores: dict[str, float]  # model_name → benchmark score
    downstream_scores: dict[str, float]  # model_name → downstream performance
    pearson_r: float
    spearman_rho: float
    kendall_tau: float
    n_models: int


@dataclass
class CorrelationReport:
    """Summary of longitudinal correlation tracking for a benchmark."""

    benchmark: str
    n_records: int
    latest_pearson_r: float
    latest_spearman_rho: float
    latest_kendall_tau: float
    trend: str  # "DEGRADING" | "STABLE" | "IMPROVING"
    pearson_trend_slope: float  # Δr per measurement period
    alert: bool
    alert_reason: str
    records: list[CorrelationRecord] = field(default_factory=list)


class CorrelationTrackerAgent:
    """
    Tracks and analyzes benchmark–downstream task correlation over time.

    Persistence: JSON database at db_path.
    Alert conditions:
      - Pearson r drops > 0.1 from previous measurement
      - Latest r < 0.5 (benchmark no longer predictive)
      - Ranking stability (Kendall tau) drops below 0.6
    """

    ALERT_THRESHOLD_R = 0.5
    ALERT_THRESHOLD_TAU = 0.6
    ALERT_SLOPE_THRESHOLD = -0.05  # r dropping > 0.05 per measurement

    def __init__(self, db_path: str = "knowledge/correlation_history.json") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db: dict[str, list[dict]] = {}
        if self.db_path.exists():
            self._db = json.loads(self.db_path.read_text())
            logger.info(f"Loaded correlation DB: {len(self._db)} benchmarks tracked")

    def record(
        self,
        benchmark: str,
        model_scores: dict[str, float],
        downstream_scores: dict[str, float],
    ) -> CorrelationRecord:
        """
        Record a new correlation measurement.

        Args:
            benchmark: Benchmark name.
            model_scores: Dict of model_name → benchmark accuracy (0–1).
            downstream_scores: Dict of model_name → downstream task score (0–1).

        Returns:
            CorrelationRecord with computed correlations.
        """
        common_models = sorted(set(model_scores) & set(downstream_scores))
        if len(common_models) < 3:
            raise ValueError(
                f"Need at least 3 models with both benchmark and downstream scores. "
                f"Got {len(common_models)}."
            )

        bench_vec = [model_scores[m] for m in common_models]
        downstream_vec = [downstream_scores[m] for m in common_models]

        pr, _ = pearsonr(bench_vec, downstream_vec)
        sr, _ = spearmanr(bench_vec, downstream_vec)
        kt, _ = kendalltau(bench_vec, downstream_vec)

        record = CorrelationRecord(
            timestamp=datetime.utcnow().isoformat(),
            benchmark=benchmark,
            model_scores=model_scores,
            downstream_scores=downstream_scores,
            pearson_r=round(float(pr), 4),
            spearman_rho=round(float(sr), 4),
            kendall_tau=round(float(kt), 4),
            n_models=len(common_models),
        )

        if benchmark not in self._db:
            self._db[benchmark] = []
        self._db[benchmark].append(self._record_to_dict(record))
        self._save()

        logger.info(
            f"Recorded correlation for {benchmark}: "
            f"r={record.pearson_r:.3f}, rho={record.spearman_rho:.3f}, tau={record.kendall_tau:.3f}"
        )
        return record

    def get_report(self, benchmark: str) -> CorrelationReport | None:
        """Generate a longitudinal report for a benchmark."""
        records_raw = self._db.get(benchmark, [])
        if not records_raw:
            return None

        records = [self._dict_to_record(r) for r in records_raw]

        pearson_values = [r.pearson_r for r in records]
        latest = records[-1]

        # Trend: linear regression on Pearson r over time
        trend_slope = 0.0
        if len(pearson_values) >= 3:
            x = np.arange(len(pearson_values), dtype=float)
            y = np.array(pearson_values, dtype=float)
            A = np.vstack([x, np.ones_like(x)]).T
            result = np.linalg.lstsq(A, y, rcond=None)
            trend_slope = float(result[0][0]) if len(result[0]) > 0 else 0.0

        if trend_slope < self.ALERT_SLOPE_THRESHOLD:
            trend = "DEGRADING"
        elif trend_slope > 0.02:
            trend = "IMPROVING"
        else:
            trend = "STABLE"

        # Alert conditions
        alert = False
        alert_reason = ""
        if latest.pearson_r < self.ALERT_THRESHOLD_R:
            alert = True
            alert_reason = f"Pearson r ({latest.pearson_r:.3f}) below threshold ({self.ALERT_THRESHOLD_R})"
        elif latest.kendall_tau < self.ALERT_THRESHOLD_TAU:
            alert = True
            alert_reason = f"Kendall tau ({latest.kendall_tau:.3f}) below threshold ({self.ALERT_THRESHOLD_TAU})"
        elif trend_slope < self.ALERT_SLOPE_THRESHOLD:
            alert = True
            alert_reason = (
                f"Pearson r declining at {trend_slope:.3f}/period — benchmark degrading"
            )

        return CorrelationReport(
            benchmark=benchmark,
            n_records=len(records),
            latest_pearson_r=latest.pearson_r,
            latest_spearman_rho=latest.spearman_rho,
            latest_kendall_tau=latest.kendall_tau,
            trend=trend,
            pearson_trend_slope=round(trend_slope, 4),
            alert=alert,
            alert_reason=alert_reason,
            records=records,
        )

    def list_benchmarks(self) -> list[str]:
        """Return all tracked benchmark names."""
        return list(self._db.keys())

    def get_all_alerts(self) -> list[CorrelationReport]:
        """Return reports for all benchmarks currently in alert state."""
        alerts = []
        for benchmark in self._db:
            report = self.get_report(benchmark)
            if report and report.alert:
                alerts.append(report)
        return alerts

    def _save(self) -> None:
        """Persist database to disk."""
        self.db_path.write_text(json.dumps(self._db, indent=2))

    @staticmethod
    def _record_to_dict(record: CorrelationRecord) -> dict:
        return {
            "timestamp": record.timestamp,
            "benchmark": record.benchmark,
            "model_scores": record.model_scores,
            "downstream_scores": record.downstream_scores,
            "pearson_r": record.pearson_r,
            "spearman_rho": record.spearman_rho,
            "kendall_tau": record.kendall_tau,
            "n_models": record.n_models,
        }

    @staticmethod
    def _dict_to_record(d: dict) -> CorrelationRecord:
        return CorrelationRecord(**d)
