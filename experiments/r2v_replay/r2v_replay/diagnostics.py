from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .consistency_scorers import DynamicsConsistencyScorer
from .encoders import TransitionEncoder
from .rarity_scorers import DiffusionConfig, DiffusionRarityScorer, KNNRarityScorer
from .replay_dataset import ReplayDataset
from .risk_scorers import OODRiskScorer
from .selector import R2VSelector, SelectedSubset, SelectionScores, composition_by_label
from .utility_scorers import TaskUtilityScorer


RARE_VALUABLE = {"rare_valuable_positive", "rare_valuable_zero_precursor"}
RARE_ANY = RARE_VALUABLE | {"rare_useless"}
RARITY_COLUMNS = {
    "diffusion": "diffusion_rarity",
    "knn": "knn_rarity",
}


def run_level0_diagnostic(dataset: ReplayDataset, cfg: dict, output_dir: str | Path, seed: int = 0) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    real_mask = dataset.real_mask()

    encoder = TransitionEncoder(input_mode=cfg.get("encoder", {}).get("rarity_input", "obs_action_next"))
    z = encoder.fit_transform(dataset, mask=real_mask)
    z_real = z[real_mask]

    knn_k = int(cfg.get("scorers", {}).get("knn_k", 10))
    knn = KNNRarityScorer(k=knn_k).fit(z_real)
    knn_raw = knn.score(z)
    knn_rarity = _rank_within_masks(knn_raw, real_mask)

    diff_cfg_raw = cfg.get("scorers", {}).get("diffusion", {})
    diff_cfg = DiffusionConfig(seed=seed, **diff_cfg_raw)
    diffusion = DiffusionRarityScorer(diff_cfg)
    diffusion_real, diffusion_real_components = diffusion.cross_fit_score(z_real)
    diffusion.fit(z_real)
    diffusion_rarity = np.zeros(len(dataset), dtype=np.float32)
    diffusion_rarity[real_mask] = diffusion_real
    diffusion_components = _empty_component_frame(len(dataset), diffusion_real_components)
    for name, values in diffusion_real_components.items():
        diffusion_components[name][real_mask] = values
    invalid_mask = ~real_mask
    if invalid_mask.any():
        invalid_components = diffusion.score_components(z[invalid_mask])
        for name, values in invalid_components.items():
            diffusion_components[name][invalid_mask] = values
        diffusion_rarity[invalid_mask] = invalid_components["rank01_mean"]

    utility_cfg = cfg.get("scorers", {}).get("utility", {})
    utility = TaskUtilityScorer(**utility_cfg).score(dataset)

    risk = OODRiskScorer(k=knn_k).fit(z_real).score(z)

    consistency_cfg = cfg.get("scorers", {}).get("consistency", {})
    consistency = DynamicsConsistencyScorer(seed=seed, **consistency_cfg).fit(dataset, mask=real_mask).score(dataset)

    score_table = dataset.to_frame()
    score_table["knn_rarity_raw"] = knn_raw
    score_table["knn_rarity"] = knn_rarity
    score_table["diffusion_rarity"] = diffusion_rarity
    score_table["utility_total"] = utility.total
    for name, values in utility.components.items():
        score_table[f"utility_{name}"] = values
    score_table["ood_risk"] = risk
    score_table["dynamics_error"] = consistency.dynamics_error
    score_table["reward_error"] = consistency.reward_error

    selector_cfg = cfg.get("selector", {})
    baselines, selector_results, stress_results = build_baseline_indices(score_table, selector_cfg, real_mask)
    metrics = compute_metrics(
        score_table,
        baselines,
        selector_results,
        stress_results,
        rare_topk_ratio=float(selector_cfg.get("rare_topk_ratio", 0.10)),
    )
    compositions = {
        name: composition_by_label(dataset.labels, np.asarray(indices, dtype=np.int64)) for name, indices in baselines.items()
    }

    score_table.to_csv(output_dir / "score_table.csv", index=False)
    pd.DataFrame(diffusion_components).assign(index=np.arange(len(dataset))).to_csv(
        output_dir / "diffusion_score_components.csv", index=False
    )
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "composition_summary.json").write_text(json.dumps(compositions, indent=2), encoding="utf-8")
    selector_summary = _selector_summary(selector_results, stress_results)
    (output_dir / "selector_summary.json").write_text(json.dumps(selector_summary, indent=2), encoding="utf-8")
    write_funnel_tables(score_table, baselines, selector_results, stress_results, output_dir)
    write_failure_tables(score_table, baselines["r2v_diffusion_full"], output_dir)
    return {"metrics": metrics, "compositions": compositions}


def build_baseline_indices(
    score_table: pd.DataFrame, selector_cfg: dict, eligible_mask: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, SelectedSubset], dict[str, SelectedSubset]]:
    eligible_indices = np.where(eligible_mask)[0]
    rare_topk_ratio = float(selector_cfg.get("rare_topk_ratio", 0.10))
    selected_budget_ratio = float(selector_cfg.get("selected_budget_ratio", 0.05))
    rare_n = max(1, int(np.ceil(len(eligible_indices) * rare_topk_ratio)))
    budget_n = max(1, int(np.ceil(rare_n * selected_budget_ratio)))
    rng = np.random.default_rng(0)

    def top_by(column: str) -> np.ndarray:
        values = score_table[column].to_numpy()[eligible_indices]
        return eligible_indices[np.argsort(-values)[:budget_n]]

    baselines: dict[str, np.ndarray] = {
        "uniform_random": rng.choice(eligible_indices, size=budget_n, replace=False),
        "reward_only": top_by("reward"),
        "rtg_only": top_by("return_to_go"),
        "progress_only": top_by("utility_progress"),
        "knn_rarity_only": top_by("knn_rarity"),
        "diffusion_rarity_only": top_by("diffusion_rarity"),
    }
    selector_results: dict[str, SelectedSubset] = {}
    stress_results: dict[str, SelectedSubset] = {}

    for scorer_name, rarity_column in RARITY_COLUMNS.items():
        rarity_values = score_table[rarity_column].to_numpy()[eligible_indices]
        rare_top = eligible_indices[np.argsort(-rarity_values)[:rare_n]]
        baselines[f"random_from_{scorer_name}_rare_topk"] = rng.choice(rare_top, size=budget_n, replace=False)

        no_risk = _select_r2v(score_table, selector_cfg, eligible_indices, rarity_column, use_risk=False)
        full = _select_r2v(score_table, selector_cfg, eligible_indices, rarity_column, use_risk=True)
        selector_results[f"r2v_{scorer_name}_no_risk"] = no_risk
        selector_results[f"r2v_{scorer_name}_full"] = full
        baselines[f"r2v_{scorer_name}_no_risk"] = no_risk.indices
        baselines[f"r2v_{scorer_name}_full"] = full.indices

        if (~eligible_mask).any():
            stress_indices = np.arange(len(score_table))
            stress_results[f"r2v_{scorer_name}_no_risk_stress"] = _select_r2v(
                score_table, selector_cfg, stress_indices, rarity_column, use_risk=False
            )
            stress_results[f"r2v_{scorer_name}_full_stress"] = _select_r2v(
                score_table, selector_cfg, stress_indices, rarity_column, use_risk=True
            )

    return baselines, selector_results, stress_results


def compute_metrics(
    score_table: pd.DataFrame,
    baselines: dict[str, np.ndarray],
    selector_results: dict[str, SelectedSubset],
    stress_results: dict[str, SelectedSubset],
    rare_topk_ratio: float,
) -> dict[str, float]:
    labels = score_table["label_for_eval_only"].to_numpy()
    real_mask = ~score_table["optional_invalid"].to_numpy(dtype=bool)
    rare_any = np.isin(labels, list(RARE_ANY)) & real_mask
    rare_valuable = np.isin(labels, list(RARE_VALUABLE)) & real_mask
    common = (labels == "common_zero") & real_mask

    metrics: dict[str, float] = {}
    for scorer_name, rarity_column in RARITY_COLUMNS.items():
        values = score_table[rarity_column].to_numpy()
        top10 = _top_fraction(values, rare_topk_ratio, eligible_mask=real_mask)
        top20 = _top_fraction(values, 0.20, eligible_mask=real_mask)
        scorer_metrics = {
            f"h1_{scorer_name}_auroc_common_vs_rare_any": _safe_auc(rare_any[common | rare_any], values[common | rare_any]),
            f"h1_{scorer_name}_auprc_common_vs_rare_any": _safe_ap(rare_any[common | rare_any], values[common | rare_any]),
            f"h1_{scorer_name}_recall_top10_rare_any": _recall(rare_any, top10),
            f"h1_{scorer_name}_recall_top10_rare_valuable": _recall(rare_valuable, top10),
            f"h1_{scorer_name}_recall_top20_rare_valuable": _recall(rare_valuable, top20),
            f"h1_{scorer_name}_enrichment_top10_rare_valuable": _enrichment(rare_valuable, top10),
            f"h2_{scorer_name}_rare_useless_fraction_top10": float(np.mean(labels[top10] == "rare_useless")),
            f"h2_{scorer_name}_valuable_precision_top10": float(np.mean(np.isin(labels[top10], list(RARE_VALUABLE)))),
            f"h2_{scorer_name}_rare_useless_enrichment_vs_uniform": _enrichment(
                (labels == "rare_useless") & real_mask, top10
            ),
        }
        metrics.update(scorer_metrics)
        if scorer_name == "diffusion":
            metrics.update({key.replace("_diffusion", "", 1): value for key, value in scorer_metrics.items()})

        candidate = selector_results[f"r2v_{scorer_name}_full"].candidate_indices
        candidate_mask = _mask_from_indices(len(labels), candidate)
        zero_precursor = (labels == "rare_valuable_zero_precursor") & real_mask
        no_risk_mask = _mask_from_indices(len(labels), selector_results[f"r2v_{scorer_name}_no_risk"].indices)
        full_mask = _mask_from_indices(len(labels), selector_results[f"r2v_{scorer_name}_full"].indices)
        metrics[f"h3_{scorer_name}_zero_precursor_available_in_rarity_top10"] = float(
            (zero_precursor & candidate_mask).sum()
        )
        metrics[f"h3_{scorer_name}_zero_precursor_candidate_recall"] = _recall(zero_precursor, candidate_mask)
        metrics[f"h3_{scorer_name}_zero_precursor_retention_no_risk_from_candidate"] = _conditional_fraction(
            zero_precursor & candidate_mask, no_risk_mask
        )
        metrics[f"h3_{scorer_name}_zero_precursor_retention_full_from_candidate"] = _conditional_fraction(
            zero_precursor & candidate_mask, full_mask
        )
        metrics[f"h4_{scorer_name}_precursor_false_rejection_full_vs_no_risk"] = _false_rejection_rate(
            zero_precursor, no_risk_mask, full_mask
        )

    for name, indices in baselines.items():
        idx = np.asarray(indices, dtype=np.int64)
        metrics[f"{name}_selection_count"] = float(len(idx))
        metrics[f"{name}_valuable_precision"] = _mean_mask(np.isin(labels[idx], list(RARE_VALUABLE)))
        metrics[f"{name}_rare_useless_fraction"] = _mean_mask(labels[idx] == "rare_useless")
        metrics[f"{name}_zero_precursor_recall"] = _recall(
            (labels == "rare_valuable_zero_precursor") & real_mask, _mask_from_indices(len(labels), idx)
        )
        metrics[f"{name}_positive_reward_fraction"] = _mean_mask(score_table["reward"].to_numpy()[idx] > 0)

    invalid = labels == "optional_invalid"
    if invalid.any():
        risk_components = {
            "ood_risk": score_table["ood_risk"].to_numpy(),
            "dynamics_error": score_table["dynamics_error"].to_numpy(),
            "reward_error": score_table["reward_error"].to_numpy(),
        }
        for name, values in risk_components.items():
            metrics[f"h4_invalid_auroc_{name}"] = _safe_auc(invalid, values)
        combined_risk = risk_components["ood_risk"] + risk_components["dynamics_error"] + risk_components["reward_error"]
        metrics["h4_invalid_auroc_combined_risk"] = _safe_auc(invalid, combined_risk)
        for scorer_name in RARITY_COLUMNS:
            no_key = f"r2v_{scorer_name}_no_risk_stress"
            full_key = f"r2v_{scorer_name}_full_stress"
            no_invalid = _selection_fraction(invalid, stress_results[no_key].indices)
            full_invalid = _selection_fraction(invalid, stress_results[full_key].indices)
            metrics[f"h4_{scorer_name}_invalid_fraction_no_risk_stress"] = no_invalid
            metrics[f"h4_{scorer_name}_invalid_fraction_full_stress"] = full_invalid
            metrics[f"h4_{scorer_name}_invalid_selection_reduction_stress"] = no_invalid - full_invalid
    else:
        metrics["h4_invalid_auroc_combined_risk"] = float("nan")
    return metrics


def write_funnel_tables(
    score_table: pd.DataFrame,
    baselines: dict[str, np.ndarray],
    selector_results: dict[str, SelectedSubset],
    stress_results: dict[str, SelectedSubset],
    output_dir: Path,
) -> None:
    labels = score_table["label_for_eval_only"].to_numpy()
    stage_rows = []
    composition_rows = []
    for scorer_name in RARITY_COLUMNS:
        stages = {
            "candidate_top10": selector_results[f"r2v_{scorer_name}_full"].candidate_indices,
            "r2v_no_risk": selector_results[f"r2v_{scorer_name}_no_risk"].indices,
            "r2v_full": selector_results[f"r2v_{scorer_name}_full"].indices,
        }
        for stage, indices in stages.items():
            stage_rows.append(_stage_summary(score_table, scorer_name, stage, indices))
            for label, count in composition_by_label(labels, np.asarray(indices, dtype=np.int64)).items():
                composition_rows.append({"scorer": scorer_name, "stage": stage, "label": label, "count": count})

    pd.DataFrame(stage_rows).to_csv(output_dir / "candidate_funnel_summary.csv", index=False)
    pd.DataFrame(composition_rows).to_csv(output_dir / "candidate_composition_by_scorer.csv", index=False)

    baseline_rows = []
    for name, indices in baselines.items():
        baseline_rows.append(_stage_summary(score_table, "baseline", name, indices))
    pd.DataFrame(baseline_rows).to_csv(output_dir / "selection_funnel_by_baseline.csv", index=False)

    risk_rows = _risk_rows(score_table, selector_results, stress_results)
    pd.DataFrame(risk_rows).to_csv(output_dir / "risk_rejection_summary.csv", index=False)


def write_failure_tables(score_table: pd.DataFrame, selected_indices: np.ndarray, output_dir: Path) -> None:
    selected = score_table.iloc[selected_indices].copy()
    selected.sort_values("utility_total", ascending=False).head(50).to_csv(output_dir / "top_selected_examples.csv", index=False)
    selected[selected["label_for_eval_only"] == "rare_useless"].head(50).to_csv(
        output_dir / "selected_rare_useless.csv", index=False
    )
    real = score_table[~score_table["optional_invalid"]]
    rejected = real[real["label_for_eval_only"] == "rare_valuable_zero_precursor"].copy()
    rejected = rejected[~rejected.index.isin(selected_indices)]
    rejected.sort_values("diffusion_rarity", ascending=False).head(50).to_csv(
        output_dir / "rejected_valuable_precursors.csv", index=False
    )
    real.sort_values("diffusion_rarity", ascending=False).head(50).to_csv(output_dir / "top_failure_cases.csv", index=False)


def _select_r2v(
    score_table: pd.DataFrame,
    selector_cfg: dict,
    eligible_indices: np.ndarray,
    rarity_column: str,
    use_risk: bool,
) -> SelectedSubset:
    cfg = dict(selector_cfg)
    if not use_risk:
        cfg["gamma"] = 0.0
        cfg["eta"] = 0.0
        cfg["rho"] = 0.0
    selector = R2VSelector(**cfg)
    selection_scores = SelectionScores(
        indices=eligible_indices,
        rarity=score_table[rarity_column].to_numpy()[eligible_indices],
        utility=score_table["utility_total"].to_numpy()[eligible_indices],
        ood_risk=score_table["ood_risk"].to_numpy()[eligible_indices],
        dynamics_error=score_table["dynamics_error"].to_numpy()[eligible_indices],
        reward_error=score_table["reward_error"].to_numpy()[eligible_indices],
    )
    return selector.select(selection_scores)


def _stage_summary(score_table: pd.DataFrame, scorer: str, stage: str, indices: np.ndarray) -> dict[str, float | str]:
    labels = score_table["label_for_eval_only"].to_numpy()
    idx = np.asarray(indices, dtype=np.int64)
    return {
        "scorer": scorer,
        "stage": stage,
        "selection_count": float(len(idx)),
        "rare_any_count": float(np.isin(labels[idx], list(RARE_ANY)).sum()),
        "valuable_count": float(np.isin(labels[idx], list(RARE_VALUABLE)).sum()),
        "zero_precursor_count": float((labels[idx] == "rare_valuable_zero_precursor").sum()),
        "rare_useless_count": float((labels[idx] == "rare_useless").sum()),
        "common_count": float((labels[idx] == "common_zero").sum()),
        "valuable_precision": _mean_mask(np.isin(labels[idx], list(RARE_VALUABLE))),
        "rare_useless_fraction": _mean_mask(labels[idx] == "rare_useless"),
        "positive_reward_fraction": _mean_mask(score_table["reward"].to_numpy()[idx] > 0),
    }


def _risk_rows(
    score_table: pd.DataFrame,
    selector_results: dict[str, SelectedSubset],
    stress_results: dict[str, SelectedSubset],
) -> list[dict[str, float | str]]:
    labels = score_table["label_for_eval_only"].to_numpy()
    invalid = labels == "optional_invalid"
    rows: list[dict[str, float | str]] = []
    if invalid.any():
        risk_components = {
            "ood_risk": score_table["ood_risk"].to_numpy(),
            "dynamics_error": score_table["dynamics_error"].to_numpy(),
            "reward_error": score_table["reward_error"].to_numpy(),
        }
        for name, values in risk_components.items():
            rows.append({"metric": "invalid_auroc", "scorer": "risk", "component": name, "value": _safe_auc(invalid, values)})
        combined = risk_components["ood_risk"] + risk_components["dynamics_error"] + risk_components["reward_error"]
        rows.append({"metric": "invalid_auroc", "scorer": "risk", "component": "combined", "value": _safe_auc(invalid, combined)})
        for scorer_name in RARITY_COLUMNS:
            no_key = f"r2v_{scorer_name}_no_risk_stress"
            full_key = f"r2v_{scorer_name}_full_stress"
            rows.append(
                {
                    "metric": "invalid_fraction_no_risk_stress",
                    "scorer": scorer_name,
                    "component": "selection",
                    "value": _selection_fraction(invalid, stress_results[no_key].indices),
                }
            )
            rows.append(
                {
                    "metric": "invalid_fraction_full_stress",
                    "scorer": scorer_name,
                    "component": "selection",
                    "value": _selection_fraction(invalid, stress_results[full_key].indices),
                }
            )

    zero_precursor = labels == "rare_valuable_zero_precursor"
    for scorer_name in RARITY_COLUMNS:
        no_mask = _mask_from_indices(len(labels), selector_results[f"r2v_{scorer_name}_no_risk"].indices)
        full_mask = _mask_from_indices(len(labels), selector_results[f"r2v_{scorer_name}_full"].indices)
        rows.append(
            {
                "metric": "precursor_false_rejection_full_vs_no_risk",
                "scorer": scorer_name,
                "component": "selection",
                "value": _false_rejection_rate(zero_precursor, no_mask, full_mask),
            }
        )
    return rows


def _selector_summary(
    selector_results: dict[str, SelectedSubset], stress_results: dict[str, SelectedSubset]
) -> dict[str, dict[str, list[int]]]:
    summary = {}
    for name, subset in {**selector_results, **stress_results}.items():
        summary[name] = {
            "candidate_indices": subset.candidate_indices.tolist(),
            "selected_indices": subset.indices.tolist(),
        }
    return summary


def _empty_component_frame(n: int, components: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.zeros(n, dtype=np.float32) for name in components}


def _top_fraction(values: np.ndarray, ratio: float, eligible_mask: np.ndarray | None = None) -> np.ndarray:
    eligible = np.arange(len(values)) if eligible_mask is None else np.where(eligible_mask)[0]
    n = max(1, int(np.ceil(len(eligible) * ratio)))
    mask = np.zeros(len(values), dtype=bool)
    mask[eligible[np.argsort(-values[eligible])[:n]]] = True
    return mask


def _mask_from_indices(n: int, indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[np.asarray(indices, dtype=np.int64)] = True
    return mask


def _recall(target: np.ndarray, selected: np.ndarray) -> float:
    denom = int(target.sum())
    if denom == 0:
        return float("nan")
    return float((target & selected).sum() / denom)


def _enrichment(target: np.ndarray, selected: np.ndarray) -> float:
    base = float(np.mean(target))
    if base <= 0:
        return float("nan")
    return float(np.mean(target[selected]) / base)


def _conditional_fraction(available: np.ndarray, selected: np.ndarray) -> float:
    denom = int(available.sum())
    if denom == 0:
        return float("nan")
    return float((available & selected).sum() / denom)


def _selection_fraction(target: np.ndarray, indices: np.ndarray) -> float:
    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) == 0:
        return float("nan")
    return float(np.mean(target[idx]))


def _false_rejection_rate(target: np.ndarray, no_risk_selected: np.ndarray, full_selected: np.ndarray) -> float:
    base = target & no_risk_selected
    denom = int(base.sum())
    if denom == 0:
        return float("nan")
    return float((base & ~full_selected).sum() / denom)


def _mean_mask(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.mean(values))


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true.astype(int))) < 2:
        return float("nan")
    return float(roc_auc_score(y_true.astype(int), scores))


def _safe_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true.astype(int))) < 2:
        return float("nan")
    return float(average_precision_score(y_true.astype(int), scores))


def _rank_within_masks(values: np.ndarray, primary_mask: np.ndarray) -> np.ndarray:
    ranked = np.zeros(len(values), dtype=np.float32)
    ranked[primary_mask] = _rank01(values[primary_mask])
    if (~primary_mask).any():
        ranked[~primary_mask] = _rank01(values[~primary_mask])
    return ranked


def _rank01(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 1:
        return np.ones_like(values, dtype=np.float32)
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, len(values), dtype=np.float32)
    return ranks
