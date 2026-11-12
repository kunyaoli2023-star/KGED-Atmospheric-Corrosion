"""One-standard-error one-standard-error reduction for the adaptive archive.

The four grouped validation axes are already fitted by the discovery engine.
This module changes only selection, never fitting or access to field responses.
The standard error is a development-set heuristic, not an independent-data CI.
"""
from __future__ import annotations

import math
from collections import defaultdict


AXES = ("shape_group", "T_C", "h_um", "C_wt")
POLICY = "balanced_grouped_cv_one_se_then_k_bic_ast_v1"


def _score(row):
    score = row["score"]
    if score.get("fidelity") != "final":
        raise ValueError("ONE_SE_SELECTION_REQUIRES_FINAL_GROUPED_CV")
    value = float(score["cv_log_rmse"])
    if not math.isfinite(value) or value < 0:
        raise ValueError("ONE_SE_SELECTION_REQUIRES_FINITE_NONNEGATIVE_CV")
    return value


def balanced_cv_se(row):
    """Transfer the v2 per-axis fold-SE rule to four validation axes.

    Folds across axes overlap. Thus this is a declared selection tolerance,
    not a calibrated uncertainty interval or a significance test.
    """
    groups = defaultdict(list)
    for fold in row.get("fold_rows", []):
        value = float(fold["rmse"])
        if not math.isfinite(value) or value < 0:
            raise ValueError("ONE_SE_SELECTION_REQUIRES_FINITE_FOLD_RMSE")
        groups[fold["axis"]].append(value)
    if set(groups) != set(AXES) or any(len(groups[axis]) < 2 for axis in AXES):
        raise ValueError("ONE_SE_SELECTION_REQUIRES_ALL_FOUR_AXIS_FOLD_RECORDS")
    variance = 0.0
    for axis in AXES:
        values = groups[axis]
        mean = sum(values) / len(values)
        sample_var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        variance += sample_var / len(values)
    return math.sqrt(variance) / len(AXES)


def training_bic(row):
    """Comparable tie-breaker from the same experimental training fit."""
    score = row["score"]
    n = len(row.get("full_predictions", []))
    k = int(score["parameter_count"])
    rmse = float(score["train_log_rmse"])
    if n <= k or rmse <= 0 or not math.isfinite(rmse):
        raise ValueError("ONE_SE_BIC_REQUIRES_FINITE_TRAINING_FIT")
    return n * math.log(rmse * rmse) + k * math.log(n)


def select(archive, _legacy_tolerance=None):
    """Select the simplest member within one SE of the best grouped CV.

    The optional second argument preserves the engine's existing call site;
    it does not influence this predeclared v3 rule.
    """
    if not archive:
        raise RuntimeError("NO_ADMISSIBLE_COMPLETE_EQUATION")
    for row in archive:
        score = _score(row)
        balanced_cv_se(row)
        groups = defaultdict(list)
        for fold in row["fold_rows"]:
            groups[fold["axis"]].append(float(fold["rmse"]))
        mean = sum(sum(groups[axis]) / len(groups[axis]) for axis in AXES) / len(AXES)
        if not math.isclose(score, mean, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError("ONE_SE_SELECTION_CV_FOLD_MISMATCH")
    best = min(archive, key=lambda row: (_score(row), row["equation_key"]))
    threshold = _score(best) + balanced_cv_se(best)
    choices = [row for row in archive if _score(row) <= threshold + 1e-12]
    selected = min(choices, key=lambda row: (
        int(row["score"]["parameter_count"]), training_bic(row),
        int(row["score"]["canonical_ast_nodes"]), _score(row), row["equation_key"]
    ))
    return selected, choices


def receipt(archive, selected, choices):
    best = min(archive, key=lambda row: (_score(row), row["equation_key"]))
    return {
        "selection_policy": POLICY,
        "balanced_cv_minimum_equation_key": best["equation_key"],
        "best_cv_log_rmse": _score(best),
        "best_cv_standard_error": balanced_cv_se(best),
        "one_se_threshold": _score(best) + balanced_cv_se(best),
        "one_se_pool_size": len(choices),
        "one_se_equation_keys": [row["equation_key"] for row in choices],
        "retained_equation_key": selected["equation_key"],
        "retained_parameter_count": selected["score"]["parameter_count"],
        "retained_ast_nodes": selected["score"]["canonical_ast_nodes"],
        "retained_training_bic": training_bic(selected),
        "validation_axes": list(AXES),
        "field_targets_used": False,
        "uncertainty_scope": "Development grouped-fold dispersion only; overlapping axes, not an independent CI.",
    }
