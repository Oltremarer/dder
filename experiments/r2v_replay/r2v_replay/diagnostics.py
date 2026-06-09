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
from .selector import R2VSelector, SelectionScores, composition_by_label
from .utility_scorers import TaskUtilityScorer


RARE_VALUABLE = {"rare_valuable_positive", "rare_valuable_zero_precursor"}
RARE_ANY = RARE_VALUABLE | {"rare_useless"}


def run_level0_diagnostic(dataset: ReplayDataset, cfg: dict, output_dir: str | Path, seed: int = 0) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    real_mask = dataset.real_mask()

    encoder = TransitionEncoder(input_mode=cfg.get("encoder", {}).get("rarity_input", "obs_action_next"))
    z = encoder.fit_transform(dataset, mask=real_mask)
    z_real = z[real_mask]

    knn_k = int(cfg.get("scorers", {}).get("knn_k", 10))
    knn = KNNRarityScorer(k=knn_k).fit(z_real)
    knn_rarity = knn.score(z)

    diff_cfg_raw = cfg.get("scorers", {}).get("diffusion", {})
    diff_cfg = DiffusionConfig(seed=seed, **diff_cfg_raw)
    diffusion = DiffusionRarityScorer(diff_cfg).fit(z_real)
    diffusion_rarity = diffusion.score(z)

    utility_cfg = cfg.get("scorers", {}).get("utility", {})
    utility = TaskUtilityScorer(**utility_cfg).score(dataset)

    risk = OODRiskScorer(k=knn_k).fit(z_real).score(z)

    consistency_cfg = cfg.get("scorers", {}).get("consistency", {})
    consistency = DynamicsConsistencyScorer(seed=seed, **consistency_cfg).fit(dataset, mask=real_mask).score(dataset)

    score_table = dataset.to_frame()
    score_table["knn_rarity"] = knn_rarity
    score_table["diffusion_rarity"] = diffusion_rarity
    score_table["utility_total"] = utility.total
    for name, values in utility.components.items():
        score_table[f"utility_{name}"] = values
    score_table["ood_risk"] = risk
    score_table["dynamics_error"] = consistency.dynamics_error
    score_table["reward_error"] = consistency.reward_error

    selector_cfg = cfg.get("selector", {})
    selector = R2VSelector(**selector_cfg)
    selection_scores = SelectionScores(
        indices=np.arange(len(dataset)),
        rarity=diffusion_rarity,
        utility=utility.total,
        ood_risk=risk,
        dynamics_error=consistency.dynamics_error,
        reward_error=consistency.reward_error,
    )
    r2v_diff = selector.select(selection_scores)
    selection_scores_knn = SelectionScores(
        indices=np.arange(len(dataset)),
        rarity=knn_rarity,
        utility=utility.total,
        ood_risk=risk,
        dynamics_error=consistency.dynamics_error,
        reward_error=consistency.reward_error,
    )
    r2v_knn = selector.select(selection_scores_knn)

    baselines = build_baseline_indices(score_table, selector_cfg)
    baselines["r2v_diffusion"] = r2v_diff.indices
    baselines["r2v_knn"] = r2v_knn.indices

    metrics = compute_metrics(score_table, baselines, rare_topk_ratio=float(selector_cfg.get("rare_topk_ratio", 0.10)))
    compositions = {
        name: composition_by_label(dataset.labels, np.asarray(indices, dtype=np.int64)) for name, indices in baselines.items()
    }

    score_table.to_csv(output_dir / "score_table.csv", index=False)
    pd.DataFrame([metrics]).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "composition_summary.json").write_text(json.dumps(compositions, indent=2), encoding="utf-8")
    selector_summary = {
        "diffusion_candidate_indices": r2v_diff.candidate_indices.tolist(),
        "diffusion_selected_indices": r2v_diff.indices.tolist(),
        "knn_selected_indices": r2v_knn.indices.tolist(),
    }
    (output_dir / "selector_summary.json").write_text(json.dumps(selector_summary, indent=2), encoding="utf-8")
    write_failure_tables(score_table, baselines["r2v_diffusion"], output_dir)
    return {"metrics": metrics, "compositions": compositions}


def build_baseline_indices(score_table: pd.DataFrame, selector_cfg: dict) -> dict[str, np.ndarray]:
    n = len(score_table)
    rare_topk_ratio = float(selector_cfg.get("rare_topk_ratio", 0.10))
    selected_budget_ratio = float(selector_cfg.get("selected_budget_ratio", 0.05))
    rare_n = max(1, int(np.ceil(n * rare_topk_ratio)))
    budget_n = max(1, int(np.ceil(rare_n * selected_budget_ratio)))
    rng = np.random.default_rng(0)
    diff_top = np.argsort(-score_table["diffusion_rarity"].to_numpy())[:rare_n]
    return {
        "uniform_random": rng.choice(np.arange(n), size=budget_n, replace=False),
        "reward_only": np.argsort(-score_table["reward"].to_numpy())[:budget_n],
        "rtg_only": np.argsort(-score_table["return_to_go"].to_numpy())[:budget_n],
        "progress_only": np.argsort(-score_table["utility_progress"].to_numpy())[:budget_n],
        "knn_rarity_only": np.argsort(-score_table["knn_rarity"].to_numpy())[:budget_n],
        "diffusion_rarity_only": diff_top[:budget_n],
        "random_from_diffusion_rare_topk": rng.choice(diff_top, size=budget_n, replace=False),
    }


def compute_metrics(score_table: pd.DataFrame, baselines: dict[str, np.ndarray], rare_topk_ratio: float) -> dict[str, float]:
    labels = score_table["label_for_eval_only"].to_numpy()
    real_mask = ~score_table["optional_invalid"].to_numpy(dtype=bool)
    rare_any = np.isin(labels, list(RARE_ANY)) & real_mask
    rare_valuable = np.isin(labels, list(RARE_VALUABLE)) & real_mask
    common = (labels == "common_zero") & real_mask
    top10 = _top_fraction(score_table["diffusion_rarity"].to_numpy(), rare_topk_ratio)
    top20 = _top_fraction(score_table["diffusion_rarity"].to_numpy(), 0.20)

    metrics: dict[str, float] = {
        "h1_auroc_common_vs_rare_any": _safe_auc(rare_any[common | rare_any], score_table["diffusion_rarity"].to_numpy()[common | rare_any]),
        "h1_auprc_common_vs_rare_any": _safe_ap(rare_any[common | rare_any], score_table["diffusion_rarity"].to_numpy()[common | rare_any]),
        "h1_recall_top10_rare_any": _recall(rare_any, top10),
        "h1_recall_top10_rare_valuable": _recall(rare_valuable, top10),
        "h1_recall_top20_rare_valuable": _recall(rare_valuable, top20),
        "h1_enrichment_top10_rare_valuable": _enrichment(rare_valuable, top10),
        "h2_rare_useless_fraction_diffusion_top10": float(np.mean(labels[top10] == "rare_useless")),
        "h2_valuable_precision_diffusion_top10": float(np.mean(np.isin(labels[top10], list(RARE_VALUABLE)))),
        "h2_rare_useless_enrichment_vs_uniform": _enrichment(labels == "rare_useless", top10),
    }

    for name, indices in baselines.items():
        idx = np.asarray(indices, dtype=np.int64)
        metrics[f"{name}_valuable_precision"] = float(np.mean(np.isin(labels[idx], list(RARE_VALUABLE))))
        metrics[f"{name}_zero_precursor_recall"] = _recall(labels == "rare_valuable_zero_precursor", _mask_from_indices(len(labels), idx))
        metrics[f"{name}_positive_reward_fraction"] = float(np.mean(score_table["reward"].to_numpy()[idx] > 0))

    invalid = labels == "optional_invalid"
    if invalid.any():
        combined_risk = (
            score_table["ood_risk"].to_numpy()
            + score_table["dynamics_error"].to_numpy()
            + score_table["reward_error"].to_numpy()
        )
        metrics["h4_invalid_auroc_combined_risk"] = _safe_auc(invalid, combined_risk)
    else:
        metrics["h4_invalid_auroc_combined_risk"] = float("nan")
    return metrics


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


def _top_fraction(values: np.ndarray, ratio: float) -> np.ndarray:
    n = max(1, int(np.ceil(len(values) * ratio)))
    mask = np.zeros(len(values), dtype=bool)
    mask[np.argsort(-values)[:n]] = True
    return mask


def _mask_from_indices(n: int, indices: np.ndarray) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    mask[indices] = True
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


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true.astype(int))) < 2:
        return float("nan")
    return float(roc_auc_score(y_true.astype(int), scores))


def _safe_ap(y_true: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y_true.astype(int))) < 2:
        return float("nan")
    return float(average_precision_score(y_true.astype(int), scores))
