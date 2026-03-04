"""
core/irt_models.py — Item Response Theory implementation for EvalForge.

Implements 1PL (Rasch), 2PL, and 3PL IRT models for:
  - Item parameter estimation (discrimination a, difficulty b, guessing c)
  - Ability estimation (theta) for each model
  - Test information function
  - Item fit statistics

Reference:
  Hambleton, Swaminathan & Rogers (1991): Fundamentals of Item Response Theory
  Lord (1980): Applications of Item Response Theory to Practical Testing Problems
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class ItemParameters:
    """IRT parameters for a single benchmark item."""

    item_id: str
    discrimination_a: float  # Discrimination — typically 0.5–3.0
    difficulty_b: (
        float  # Difficulty on latent ability scale (theta) — typically -3 to +3
    )
    guessing_c: float = 0.0  # Guessing parameter — typically 0.0–0.33
    quality_flags: list[str] = field(default_factory=list)
    information_peak_theta: float = 0.0  # Theta at which item provides max info

    @property
    def quality_flag(self) -> str:
        if self.quality_flags:
            return self.quality_flags[0]
        return "OK"


@dataclass
class TestInformation:
    """Summary statistics for the full test."""

    n_items: int
    reliability_estimate: float  # Coefficient omega / alpha estimate
    peak_theta: float  # Theta at which test provides max info
    effective_n_items: int  # Items contributing > 10% of max info
    standard_error_at_theta: dict[float, float] = field(default_factory=dict)


@dataclass
class IRTCalibrationResult:
    """Full IRT calibration output."""

    item_parameters: list[ItemParameters]
    model_ability_estimates: dict[str, float]  # model_name → theta
    test_information: TestInformation
    model_fit: dict[str, float]  # item_id → chi-square p-value


# ---------------------------------------------------------------------------
# Core IRT math
# ---------------------------------------------------------------------------


def p_correct_1pl(theta: float, b: float) -> float:
    """1PL (Rasch) probability of correct response."""
    return 1.0 / (1.0 + math.exp(-(theta - b)))


def p_correct_2pl(theta: float, a: float, b: float) -> float:
    """2PL probability of correct response."""
    return 1.0 / (1.0 + math.exp(-a * (theta - b)))


def p_correct_3pl(theta: float, a: float, b: float, c: float) -> float:
    """3PL probability of correct response (includes guessing parameter)."""
    return c + (1.0 - c) / (1.0 + math.exp(-a * (theta - b)))


def item_information_2pl(theta: float, a: float, b: float) -> float:
    """Item information function at ability level theta (2PL)."""
    p = p_correct_2pl(theta, a, b)
    q = 1.0 - p
    return a**2 * p * q


def item_information_3pl(theta: float, a: float, b: float, c: float) -> float:
    """Item information function at ability level theta (3PL)."""
    p_star = p_correct_2pl(theta, a, b)  # 2PL component
    p = p_correct_3pl(theta, a, b, c)
    q = 1.0 - p
    return (a**2 * q * (p_star**2)) / (p * (1.0 - c) ** 2)


def test_information(
    theta: float,
    item_params: list[ItemParameters],
    model: str = "2pl",
) -> float:
    """Sum of item information functions at ability level theta."""
    total = 0.0
    for ip in item_params:
        if model == "2pl":
            total += item_information_2pl(theta, ip.discrimination_a, ip.difficulty_b)
        elif model == "3pl":
            total += item_information_3pl(
                theta, ip.discrimination_a, ip.difficulty_b, ip.guessing_c
            )
    return total


def standard_error(theta: float, item_params: list[ItemParameters]) -> float:
    """Standard error of measurement at ability level theta."""
    info = test_information(theta, item_params)
    return 1.0 / math.sqrt(max(info, 1e-8))


# ---------------------------------------------------------------------------
# Parameter estimation
# ---------------------------------------------------------------------------


class IRTCalibrator:
    """
    Estimates IRT parameters from response matrices.

    The response matrix is:
        rows = models/examinees
        columns = items
        values = 0 (incorrect) or 1 (correct)
    """

    def __init__(
        self,
        model: str = "2pl",
        n_ability_levels: int = 41,
        theta_range: tuple[float, float] = (-4.0, 4.0),
    ) -> None:
        if model not in ("1pl", "2pl", "3pl"):
            raise ValueError(f"model must be '1pl', '2pl', or '3pl'; got {model!r}")
        self.model = model
        self.theta_grid = np.linspace(theta_range[0], theta_range[1], n_ability_levels)

    def calibrate(
        self,
        response_matrix: np.ndarray,
        item_ids: list[str],
        model_names: list[str],
    ) -> IRTCalibrationResult:
        """
        Estimate IRT parameters from response matrix.

        Args:
            response_matrix: (n_models, n_items) binary array.
            item_ids: List of item identifiers.
            model_names: List of model names (row names).

        Returns:
            IRTCalibrationResult with item and ability estimates.
        """
        if response_matrix.shape != (len(model_names), len(item_ids)):
            raise ValueError(
                f"Matrix shape {response_matrix.shape} doesn't match "
                f"({len(model_names)}, {len(item_ids)})"
            )

        # Step 1: Estimate model abilities using raw score → theta mapping
        raw_scores = response_matrix.mean(axis=1)
        ability_estimates: dict[str, float] = {}
        for i, name in enumerate(model_names):
            # Logit transformation of raw score (clamped to avoid ±inf)
            score = np.clip(raw_scores[i], 0.01, 0.99)
            ability_estimates[name] = math.log(score / (1.0 - score))

        # Step 2: Estimate item parameters via marginal maximum likelihood (simplified EM)
        theta_estimates = np.array([ability_estimates[m] for m in model_names])
        item_params = []
        for j, item_id in enumerate(item_ids):
            responses = response_matrix[:, j]
            params = self._estimate_item_params(responses, theta_estimates, item_id)
            item_params.append(params)

        # Step 3: Compute test information and reliability
        theta_test_grid = np.linspace(-3, 3, 61)
        test_info_values = [
            test_information(th, item_params, self.model) for th in theta_test_grid
        ]
        peak_theta = float(theta_test_grid[int(np.argmax(test_info_values))])
        # Count grid points where information exceeds 10% of peak — this measures
        # the breadth of the theta range that the test covers, not the number of items.
        theta_range_coverage = sum(
            1 for info in test_info_values if info > 0.1 * max(test_info_values)
        )

        # Reliability via Spearman-Brown from median SEM.
        # Correct formula: r = 1 - SEM^2 / Var(theta). Without dividing by the
        # observed theta variance, the formula is not scale-invariant.
        median_sem = float(
            np.median([standard_error(th, item_params) for th in theta_test_grid])
        )
        theta_variance = (
            float(np.var(theta_estimates)) if len(theta_estimates) > 1 else 1.0
        )
        reliability = max(0.0, 1.0 - median_sem**2 / max(theta_variance, 1e-8))

        test_info = TestInformation(
            n_items=len(item_ids),
            reliability_estimate=round(reliability, 3),
            peak_theta=round(peak_theta, 2),
            effective_n_items=theta_range_coverage,
            standard_error_at_theta={
                round(th, 1): round(standard_error(th, item_params), 3)
                for th in [-2.0, -1.0, 0.0, 1.0, 2.0]
            },
        )

        # Step 4: Item fit (simplified: use point-biserial correlation)
        model_fit: dict[str, float] = {}
        for j, item_id in enumerate(item_ids):
            responses = response_matrix[:, j]
            r, p_val = pearsonr(theta_estimates, responses)
            model_fit[item_id] = round(float(p_val), 4)

        return IRTCalibrationResult(
            item_parameters=item_params,
            model_ability_estimates={
                k: round(v, 3) for k, v in ability_estimates.items()
            },
            test_information=test_info,
            model_fit=model_fit,
        )

    def _estimate_item_params(
        self, responses: np.ndarray, thetas: np.ndarray, item_id: str
    ) -> ItemParameters:
        """
        Estimate 2PL parameters for a single item via optimization.
        Returns ItemParameters with quality flags.
        """
        p_bar = np.mean(responses)
        # Guard against perfect items (all correct / all incorrect)
        if p_bar >= 0.99:
            return ItemParameters(
                item_id=item_id,
                discrimination_a=0.0,
                difficulty_b=-4.0,
                quality_flags=["CEILING"],
            )
        if p_bar <= 0.01:
            return ItemParameters(
                item_id=item_id,
                discrimination_a=0.0,
                difficulty_b=4.0,
                quality_flags=["FLOOR"],
            )

        # MLE via scipy.optimize
        def neg_log_likelihood(params: list[float]) -> float:
            a, b = params
            a = max(0.01, a)  # discrimination must be positive
            eps = 1e-8
            ll = 0.0
            for r_i, theta_i in zip(responses, thetas):
                p = p_correct_2pl(float(theta_i), a, b)
                p = np.clip(p, eps, 1 - eps)
                ll += r_i * math.log(p) + (1 - r_i) * math.log(1 - p)
            return -ll

        # Initial guess: a=1, b from proportion correct
        b_init = -math.log(p_bar / (1 - p_bar))
        result = minimize(
            neg_log_likelihood,
            x0=[1.0, b_init],
            method="L-BFGS-B",
            bounds=[(0.01, 5.0), (-4.0, 4.0)],
        )

        a_est, b_est = result.x
        flags: list[str] = []
        if a_est < 0.3:
            flags.append("LOW_DISCRIMINATION")
        if b_est > 3.0:
            flags.append("FLOOR")
        if b_est < -3.0:
            flags.append("CEILING")
        if not flags:
            flags.append("OK")

        # Compute information peak
        info_peak_theta = float(
            self.theta_grid[
                np.argmax(
                    [
                        item_information_2pl(th, float(a_est), float(b_est))
                        for th in self.theta_grid
                    ]
                )
            ]
        )

        return ItemParameters(
            item_id=item_id,
            discrimination_a=round(float(a_est), 3),
            difficulty_b=round(float(b_est), 3),
            quality_flags=flags,
            information_peak_theta=round(info_peak_theta, 2),
        )

    def get_items_to_replace(
        self, result: IRTCalibrationResult, max_to_replace: int = 20
    ) -> list[str]:
        """
        Identify items that should be replaced based on IRT quality flags.

        Priority:
          1. LOW_DISCRIMINATION items (not informative)
          2. CEILING/FLOOR items (not discriminating in useful ability range)
        """
        # An item is flagged if ANY of its quality_flags is not "OK".
        # Checking only flags[0] misses items with multiple non-OK flags.
        flagged = [
            ip
            for ip in result.item_parameters
            if ip.quality_flags and not all(flag == "OK" for flag in ip.quality_flags)
        ]

        # Sort: low discrimination first, then extremes.
        # Use the first non-OK flag for priority ordering.
        def _first_bad_flag(ip):
            for f in ip.quality_flags:
                if f != "OK":
                    return f
            return "OK"

        priority = {"LOW_DISCRIMINATION": 0, "FLOOR": 1, "CEILING": 1}
        flagged.sort(key=lambda ip: priority.get(_first_bad_flag(ip), 2))
        return [ip.item_id for ip in flagged[:max_to_replace]]
