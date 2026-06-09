from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_level0(score_table_path: str | Path, composition_path: str | Path, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(score_table_path)

    _hist_by_label(df, "diffusion_rarity", output_dir / "score_hist_by_label.png")
    _scatter(df, "diffusion_rarity", "utility_total", output_dir / "utility_vs_rarity_scatter.png")
    _scatter(df, "utility_total", "ood_risk", output_dir / "risk_vs_utility_scatter.png")
    _composition_bar(df, output_dir / "rare_candidate_composition.png")
    _composition_bar(df.iloc[df["diffusion_rarity"].to_numpy().argsort()[::-1][: max(1, int(0.1 * len(df)))]], output_dir / "selected_composition_bar.png")
    _precision_proxy(df, output_dir / "precision_recall.png")


def _hist_by_label(df: pd.DataFrame, column: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, group in df.groupby("label_for_eval_only"):
        ax.hist(group[column], bins=30, alpha=0.45, label=label)
    ax.set_xlabel(column)
    ax.set_ylabel("count")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _scatter(df: pd.DataFrame, x: str, y: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    labels = df["label_for_eval_only"].astype("category")
    ax.scatter(df[x], df[y], c=labels.cat.codes, s=8, alpha=0.65)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _composition_bar(df: pd.DataFrame, path: Path) -> None:
    counts = df["label_for_eval_only"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _precision_proxy(df: pd.DataFrame, path: Path) -> None:
    ordered = df.sort_values("diffusion_rarity", ascending=False).reset_index(drop=True)
    is_valuable = ordered["label_for_eval_only"].isin(["rare_valuable_positive", "rare_valuable_zero_precursor"])
    cum_tp = is_valuable.cumsum()
    ranks = pd.Series(range(1, len(ordered) + 1))
    precision = cum_tp / ranks
    recall = cum_tp / max(1, int(is_valuable.sum()))
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision)
    ax.set_xlabel("recall proxy")
    ax.set_ylabel("precision")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
