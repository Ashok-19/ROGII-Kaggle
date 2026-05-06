from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "eda_findings"
FIG = OUT / "figures"
TAB = OUT / "tables"
BEAM_SAMPLE_EVERY = 8

@dataclass
class WellSummary:
    well: str
    n_rows: int
    known_len: int
    hidden_len: int
    hidden_fraction: float
    mask_is_suffix: bool
    md_min: float
    md_max: float
    tvt_min: float
    tvt_max: float
    tvt_hidden_delta_end: float
    tvt_hidden_delta_abs_max: float
    tvt_step_median: float
    tvt_step_p05: float
    tvt_step_p95: float
    gr_missing_rate_all: float
    gr_missing_rate_known: float
    gr_missing_rate_hidden: float
    gr_longest_nan_run: int
    typewell_rows: int
    typewell_tvt_min: float
    typewell_tvt_max: float
    typewell_gr_mean: float
    typewell_gr_std: float
    geology_labeled_rate: float
    prefix_typewell_rmse: float
    hidden_typewell_oracle_rmse: float
    prefix_hidden_oracle_rmse_gap: float
    last_known_rmse: float
    prefix_step20_rmse: float
    prefix_step100_rmse: float
    prefix_md_slope100_rmse: float
    prefix_z_slope100_rmse: float
    beam_conservative_rmse: float
    beam_loose_rmse: float


def longest_nan_run(values: pd.Series) -> int:
    is_nan = values.isna().to_numpy()
    best = 0
    cur = 0
    for flag in is_nan:
        if flag:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def suffix_mask(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    start = int(np.flatnonzero(mask)[0])
    return bool(mask[start:].all() and not mask[:start].any())


def recent_mean_diff(values: np.ndarray, window: int) -> float:
    values = values[-(window + 1) :]
    if len(values) < 2:
        return 0.0
    return float(np.diff(values).mean())


def recent_slope(y_values: np.ndarray, x_values: np.ndarray, window: int) -> float:
    y_values = y_values[-window:]
    x_values = x_values[-window:]
    if len(y_values) < 2:
        return 0.0
    centered_x = x_values - x_values.mean()
    denominator = float(np.dot(centered_x, centered_x))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(centered_x, y_values - y_values.mean()) / denominator)


def nearest_index(sorted_values: np.ndarray, target: float) -> int:
    idx = int(np.searchsorted(sorted_values, target, side="left"))
    if idx >= len(sorted_values):
        return len(sorted_values) - 1
    if idx > 0 and abs(sorted_values[idx - 1] - target) <= abs(sorted_values[idx] - target):
        return idx - 1
    return idx


def fill_and_smooth_gr(values: np.ndarray, fallback: float, radius: int) -> np.ndarray:
    series = pd.Series(values, dtype="float32").interpolate(limit_direction="both").fillna(fallback)
    if radius <= 0:
        return series.to_numpy(dtype=np.float32)
    return series.rolling(radius * 2 + 1, center=True, min_periods=1).mean().to_numpy(dtype=np.float32)


def beam_predict(
    gr_values: np.ndarray,
    tw_tvt: np.ndarray,
    tw_gr: np.ndarray,
    start_tvt: float,
    beam_size: int,
    move_cost: float,
    emit_scale: float,
    radius: int,
) -> np.ndarray:
    deltas = np.array([-1, 0, 1], dtype=np.int32)
    smoothed_gr = fill_and_smooth_gr(gr_values, float(np.nanmean(tw_gr)), radius)
    states = np.array([nearest_index(tw_tvt, start_tvt)], dtype=np.int32)
    costs = np.array([0.0], dtype=np.float64)
    kept_states: list[np.ndarray] = []
    kept_parents: list[np.ndarray] = []

    for gr_value in smoothed_gr:
        next_idx = (states[:, None] + deltas[None, :]).ravel()
        parents = np.repeat(np.arange(len(states), dtype=np.int32), len(deltas))
        base_costs = np.repeat(costs, len(deltas))
        step_costs = np.tile(np.abs(deltas), len(states)) * move_cost
        valid = (next_idx >= 0) & (next_idx < len(tw_tvt))
        next_idx = next_idx[valid]
        parents = parents[valid]
        total_costs = base_costs[valid] + step_costs[valid] + ((float(gr_value) - tw_gr[next_idx]) ** 2) / emit_scale

        order = np.lexsort((total_costs, next_idx))
        next_sorted = next_idx[order]
        first = np.r_[True, next_sorted[1:] != next_sorted[:-1]]
        best_order = order[first]
        best_idx = next_idx[best_order]
        best_costs = total_costs[best_order]
        best_parents = parents[best_order]

        keep = np.argsort(best_costs)[:beam_size]
        states = best_idx[keep].astype(np.int32)
        costs = best_costs[keep].astype(np.float64)
        kept_parents.append(best_parents[keep].astype(np.int32))
        kept_states.append(states.copy())

    pos = int(np.argmin(costs))
    path = np.empty(len(kept_states), dtype=np.float32)
    for i in range(len(kept_states) - 1, -1, -1):
        path[i] = tw_tvt[kept_states[i][pos]]
        pos = int(kept_parents[i][pos])
    return path


def rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    actual = np.asarray(actual, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    valid = np.isfinite(actual) & np.isfinite(pred)
    if not valid.any():
        return float("nan")
    actual = actual[valid]
    pred = pred[valid]
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def quantile_or_nan(values: np.ndarray, q: float) -> float:
    values = np.asarray(values)
    if len(values) == 0:
        return float("nan")
    return float(np.quantile(values, q))


def summarize_series(series: pd.Series) -> dict[str, float]:
    return {
        "min": float(series.min()),
        "p05": float(series.quantile(0.05)),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
        "mean": float(series.mean()),
    }


def analyze_well(horizontal_path: Path, do_beam: bool) -> tuple[WellSummary, pd.DataFrame, pd.DataFrame]:
    well = horizontal_path.name.split("__")[0]
    typewell_path = horizontal_path.with_name(f"{well}__typewell.csv")
    df = pd.read_csv(horizontal_path)
    tw = pd.read_csv(typewell_path)

    mask = df["TVT_input"].isna().to_numpy()
    mask_start = int(np.flatnonzero(mask)[0])
    known = df.iloc[:mask_start]
    hidden = df.iloc[mask_start:]
    last_known = known.iloc[-1]

    tw_tvt = tw["TVT"].to_numpy(dtype=np.float32)
    tw_gr = tw["GR"].to_numpy(dtype=np.float32)
    gr_filled = df["GR"].interpolate(limit_direction="both")
    if gr_filled.isna().any():
        gr_filled = gr_filled.fillna(float(np.nanmean(tw_gr)))

    known_tvt = known["TVT_input"].to_numpy(dtype=np.float32)
    hidden_tvt = hidden["TVT"].to_numpy(dtype=np.float32)
    last_known_tvt = float(last_known["TVT_input"])
    hidden_actual_delta = hidden_tvt - np.float32(last_known_tvt)

    hidden_rows = np.arange(len(hidden), dtype=np.float32)
    step20 = recent_mean_diff(known_tvt, 20)
    step100 = recent_mean_diff(known_tvt, 100)
    md_slope100 = recent_slope(
        known_tvt,
        known["MD"].to_numpy(dtype=np.float32),
        100,
    )
    z_slope100 = recent_slope(
        known_tvt,
        known["Z"].to_numpy(dtype=np.float32),
        100,
    )

    pred_last = np.full(len(hidden), last_known_tvt, dtype=np.float32)
    pred_step20 = last_known_tvt + step20 * (hidden_rows + 1)
    pred_step100 = last_known_tvt + step100 * (hidden_rows + 1)
    pred_md_slope = last_known_tvt + md_slope100 * (
        hidden["MD"].to_numpy(dtype=np.float32) - float(last_known["MD"])
    )
    pred_z_slope = last_known_tvt + z_slope100 * (
        hidden["Z"].to_numpy(dtype=np.float32) - float(last_known["Z"])
    )

    prefix_tw_gr = np.interp(known_tvt, tw_tvt, tw_gr)
    hidden_tw_gr = np.interp(hidden_tvt, tw_tvt, tw_gr)
    prefix_resid = known["GR"].interpolate(limit_direction="both").to_numpy(dtype=np.float32) - prefix_tw_gr
    hidden_resid = hidden["GR"].interpolate(limit_direction="both").to_numpy(dtype=np.float32) - hidden_tw_gr

    if do_beam:
        beam_cons = beam_predict(
            hidden["GR"].to_numpy(dtype=np.float32),
            tw_tvt,
            tw_gr,
            last_known_tvt,
            beam_size=30,
            move_cost=20.0,
            emit_scale=144.0,
            radius=2,
        )
        beam_loose = beam_predict(
            hidden["GR"].to_numpy(dtype=np.float32),
            tw_tvt,
            tw_gr,
            last_known_tvt,
            beam_size=30,
            move_cost=8.0,
            emit_scale=64.0,
            radius=2,
        )
        beam_cons_rmse = rmse(hidden_tvt, beam_cons)
        beam_loose_rmse = rmse(hidden_tvt, beam_loose)
    else:
        beam_cons = np.full(len(hidden), np.nan, dtype=np.float32)
        beam_loose = np.full(len(hidden), np.nan, dtype=np.float32)
        beam_cons_rmse = float("nan")
        beam_loose_rmse = float("nan")

    tvt_steps = np.diff(df["TVT"].to_numpy(dtype=np.float32))
    gr_all = df["GR"]
    gr_known = known["GR"]
    gr_hidden = hidden["GR"]
    geology_labeled_rate = float(tw["Geology"].notna().mean()) if "Geology" in tw.columns else 0.0

    summary = WellSummary(
        well=well,
        n_rows=len(df),
        known_len=mask_start,
        hidden_len=len(hidden),
        hidden_fraction=float(len(hidden) / len(df)),
        mask_is_suffix=suffix_mask(mask),
        md_min=float(df["MD"].min()),
        md_max=float(df["MD"].max()),
        tvt_min=float(df["TVT"].min()),
        tvt_max=float(df["TVT"].max()),
        tvt_hidden_delta_end=float(hidden_actual_delta[-1]),
        tvt_hidden_delta_abs_max=float(np.max(np.abs(hidden_actual_delta))),
        tvt_step_median=float(np.median(tvt_steps)),
        tvt_step_p05=quantile_or_nan(tvt_steps, 0.05),
        tvt_step_p95=quantile_or_nan(tvt_steps, 0.95),
        gr_missing_rate_all=float(gr_all.isna().mean()),
        gr_missing_rate_known=float(gr_known.isna().mean()),
        gr_missing_rate_hidden=float(gr_hidden.isna().mean()),
        gr_longest_nan_run=longest_nan_run(gr_all),
        typewell_rows=len(tw),
        typewell_tvt_min=float(tw["TVT"].min()),
        typewell_tvt_max=float(tw["TVT"].max()),
        typewell_gr_mean=float(tw["GR"].mean()),
        typewell_gr_std=float(tw["GR"].std()),
        geology_labeled_rate=geology_labeled_rate,
        prefix_typewell_rmse=rmse(np.zeros_like(prefix_resid), prefix_resid),
        hidden_typewell_oracle_rmse=rmse(np.zeros_like(hidden_resid), hidden_resid),
        prefix_hidden_oracle_rmse_gap=rmse(np.zeros_like(hidden_resid), hidden_resid)
        - rmse(np.zeros_like(prefix_resid), prefix_resid),
        last_known_rmse=rmse(hidden_tvt, pred_last),
        prefix_step20_rmse=rmse(hidden_tvt, pred_step20),
        prefix_step100_rmse=rmse(hidden_tvt, pred_step100),
        prefix_md_slope100_rmse=rmse(hidden_tvt, pred_md_slope),
        prefix_z_slope100_rmse=rmse(hidden_tvt, pred_z_slope),
        beam_conservative_rmse=beam_cons_rmse,
        beam_loose_rmse=beam_loose_rmse,
    )

    row_sample_idx = np.linspace(0, len(hidden) - 1, min(len(hidden), 30), dtype=int)
    row_sample = pd.DataFrame(
        {
            "well": well,
            "hidden_pos": row_sample_idx,
            "hidden_frac": row_sample_idx / max(len(hidden) - 1, 1),
            "target_delta": hidden_actual_delta[row_sample_idx],
            "gr_missing": hidden["GR"].isna().to_numpy()[row_sample_idx],
            "gr": gr_filled.iloc[mask_start:].to_numpy(dtype=np.float32)[row_sample_idx],
            "beam_conservative_delta": beam_cons[row_sample_idx] - np.float32(last_known_tvt),
            "beam_loose_delta": beam_loose[row_sample_idx] - np.float32(last_known_tvt),
        }
    )

    baseline_rows = pd.DataFrame(
        {
            "well": well,
            "row_idx": hidden.index.to_numpy(dtype=np.int32),
            "actual_tvt": hidden_tvt,
            "last_known": pred_last,
            "prefix_step20": pred_step20,
            "prefix_step100": pred_step100,
            "prefix_md_slope100": pred_md_slope,
            "prefix_z_slope100": pred_z_slope,
            "beam_conservative": beam_cons,
            "beam_loose": beam_loose,
            "gr_missing": hidden["GR"].isna().to_numpy(),
            "hidden_frac": hidden_rows / max(len(hidden) - 1, 1),
        }
    )

    return summary, row_sample, baseline_rows


def save_hist(series: pd.Series, title: str, xlabel: str, path: Path, bins: int = 40) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(series.dropna(), bins=bins, color="#356859", alpha=0.88)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Well count")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)

    summaries: list[WellSummary] = []
    samples: list[pd.DataFrame] = []
    baseline_parts: list[pd.DataFrame] = []

    paths = sorted((DATA / "train").glob("*__horizontal_well.csv"))
    for i, path in enumerate(paths, start=1):
        if i % 50 == 0:
            print(f"processed {i}/{len(paths)} wells")
        do_beam = i == 1 or i % BEAM_SAMPLE_EVERY == 0
        summary, row_sample, baseline_rows = analyze_well(path, do_beam=do_beam)
        summaries.append(summary)
        samples.append(row_sample)
        baseline_parts.append(baseline_rows)

    wells = pd.DataFrame([asdict(item) for item in summaries])
    row_sample_df = pd.concat(samples, ignore_index=True)
    baseline_df = pd.concat(baseline_parts, ignore_index=True)

    wells.to_csv(TAB / "per_well_summary.csv", index=False)
    row_sample_df.to_csv(TAB / "hidden_row_sample.csv", index=False)

    baseline_cols = [
        "last_known",
        "prefix_step20",
        "prefix_step100",
        "prefix_md_slope100",
        "prefix_z_slope100",
        "beam_conservative",
        "beam_loose",
    ]
    baseline_metrics = []
    for col in baseline_cols:
        valid = baseline_df[col].notna()
        err = baseline_df.loc[valid, col].to_numpy(dtype=np.float64) - baseline_df.loc[valid, "actual_tvt"].to_numpy(dtype=np.float64)
        baseline_metrics.append(
            {
                "method": col,
                "row_rmse": float(np.sqrt(np.mean(err**2))),
                "row_mae": float(np.mean(np.abs(err))),
                "n_rows_scored": int(valid.sum()),
                "n_wells_scored": int(baseline_df.loc[valid, "well"].nunique()),
                "well_mean_rmse": float(
                    wells[
                        {
                            "last_known": "last_known_rmse",
                            "prefix_step20": "prefix_step20_rmse",
                            "prefix_step100": "prefix_step100_rmse",
                            "prefix_md_slope100": "prefix_md_slope100_rmse",
                            "prefix_z_slope100": "prefix_z_slope100_rmse",
                            "beam_conservative": "beam_conservative_rmse",
                            "beam_loose": "beam_loose_rmse",
                        }[col]
                    ].mean()
                ),
            }
        )
    baseline_metrics_df = pd.DataFrame(baseline_metrics).sort_values("row_rmse")
    baseline_metrics_df.to_csv(TAB / "baseline_metrics.csv", index=False)

    frac_frames = []
    frac_source = baseline_df.assign(
        frac_bin=pd.cut(baseline_df["hidden_frac"], np.linspace(0, 1, 21), include_lowest=True)
    )
    for frac_bin, g in frac_source.groupby("frac_bin", observed=True):
        frac_frames.append(
            {
                "frac_bin": str(frac_bin),
                "n": len(g),
                "beam_n": int(g["beam_conservative"].notna().sum()),
                "last_known_rmse": rmse(g["actual_tvt"], g["last_known"]),
                "beam_conservative_rmse": rmse(g["actual_tvt"], g["beam_conservative"]),
                "beam_loose_rmse": rmse(g["actual_tvt"], g["beam_loose"]),
                "prefix_step100_rmse": rmse(g["actual_tvt"], g["prefix_step100"]),
            }
        )
    rows_by_frac = pd.DataFrame(frac_frames)
    rows_by_frac.to_csv(TAB / "baseline_rmse_by_hidden_fraction.csv", index=False)

    summary_stats = {
        "train_horizontal_wells": int(len(paths)),
        "train_typewells": int(len(list((DATA / "train").glob("*__typewell.csv")))),
        "train_pngs": int(len(list((DATA / "train").glob("*.png")))),
        "visible_test_horizontal_wells": int(len(list((DATA / "test").glob("*__horizontal_well.csv")))),
        "beam_sample_every_nth_well": BEAM_SAMPLE_EVERY,
        "beam_sample_wells": int(wells["beam_conservative_rmse"].notna().sum()),
        "beam_sample_hidden_rows": int(baseline_df["beam_conservative"].notna().sum()),
        "all_train_masks_are_suffix": bool(wells["mask_is_suffix"].all()),
        "total_train_rows": int(wells["n_rows"].sum()),
        "total_hidden_rows": int(wells["hidden_len"].sum()),
        "total_known_rows": int(wells["known_len"].sum()),
        "hidden_rows": summarize_series(wells["hidden_len"]),
        "known_rows": summarize_series(wells["known_len"]),
        "hidden_fraction": summarize_series(wells["hidden_fraction"]),
        "gr_missing_rate_all": summarize_series(wells["gr_missing_rate_all"]),
        "gr_missing_rate_hidden": summarize_series(wells["gr_missing_rate_hidden"]),
        "prefix_typewell_rmse": summarize_series(wells["prefix_typewell_rmse"]),
        "hidden_typewell_oracle_rmse": summarize_series(wells["hidden_typewell_oracle_rmse"]),
        "geology_labeled_rate": summarize_series(wells["geology_labeled_rate"]),
    }
    (TAB / "summary_stats.json").write_text(json.dumps(summary_stats, indent=2))

    save_hist(wells["hidden_len"], "Hidden/evaluation row count per training well", "Hidden rows", FIG / "hidden_length_distribution.png")
    save_hist(wells["hidden_fraction"], "Hidden/evaluation fraction per training well", "Hidden fraction", FIG / "hidden_fraction_distribution.png")
    save_hist(wells["gr_missing_rate_hidden"], "Hidden-zone GR missingness by well", "Missing GR fraction", FIG / "hidden_gr_missingness.png")
    save_hist(wells["prefix_typewell_rmse"], "Known-zone GR vs typewell GR alignment RMSE", "GR RMSE", FIG / "prefix_typewell_alignment_rmse.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(baseline_metrics_df["method"], baseline_metrics_df["row_rmse"], color="#c8553d")
    ax.set_title("Simple train-hidden baselines, row-weighted RMSE")
    ax.set_xlabel("RMSE on hidden suffix rows")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "baseline_rmse_bar.png", dpi=180)
    plt.close(fig)

    quant = (
        row_sample_df.groupby("hidden_frac")["target_delta"]
        .quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        .unstack()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(quant["hidden_frac"], quant[0.05], quant[0.95], color="#b8d8d8", alpha=0.45, label="5-95%")
    ax.fill_between(quant["hidden_frac"], quant[0.25], quant[0.75], color="#7a9e9f", alpha=0.55, label="25-75%")
    ax.plot(quant["hidden_frac"], quant[0.5], color="#213635", linewidth=2, label="median")
    ax.set_title("Target delta over normalized hidden interval")
    ax.set_xlabel("Hidden interval fraction")
    ax.set_ylabel("TVT - last known TVT")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "target_delta_by_hidden_fraction.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        wells["prefix_typewell_rmse"],
        wells["hidden_typewell_oracle_rmse"],
        s=np.clip(wells["hidden_len"] / 50, 8, 80),
        alpha=0.55,
        color="#5b6c8f",
    )
    lim = max(wells["prefix_typewell_rmse"].max(), wells["hidden_typewell_oracle_rmse"].max())
    ax.plot([0, lim], [0, lim], color="#333333", linewidth=1)
    ax.set_title("Known-zone typewell fit predicts hidden-zone typewell fit")
    ax.set_xlabel("Known-zone GR alignment RMSE")
    ax.set_ylabel("Hidden-zone oracle GR alignment RMSE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "prefix_vs_hidden_typewell_alignment.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    beam_wells = wells.dropna(subset=["beam_conservative_rmse"])
    ax.scatter(beam_wells["gr_missing_rate_hidden"], beam_wells["beam_conservative_rmse"], alpha=0.55, color="#7d4f50")
    ax.set_title("Beam baseline degrades with hidden GR missingness")
    ax.set_xlabel("Hidden GR missing fraction")
    ax.set_ylabel("Conservative beam RMSE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "beam_rmse_vs_gr_missingness.png", dpi=180)
    plt.close(fig)

    example_wells = (
        beam_wells.sort_values(["beam_conservative_rmse", "hidden_len"], ascending=[True, False])["well"].head(2).tolist()
        + beam_wells.sort_values(["beam_conservative_rmse", "hidden_len"], ascending=[False, False])["well"].head(2).tolist()
    )
    fig, axes = plt.subplots(len(example_wells), 1, figsize=(11, 2.8 * len(example_wells)), sharex=False)
    if len(example_wells) == 1:
        axes = [axes]
    for ax, well in zip(axes, example_wells):
        h = pd.read_csv(DATA / "train" / f"{well}__horizontal_well.csv")
        tw = pd.read_csv(DATA / "train" / f"{well}__typewell.csv")
        mask_start = int(np.flatnonzero(h["TVT_input"].isna().to_numpy())[0])
        hidden = h.iloc[mask_start:]
        ax.plot(tw["TVT"], tw["GR"], color="#999999", linewidth=1, label="typewell GR")
        ax.plot(hidden["TVT"], hidden["GR"].interpolate(limit_direction="both"), color="#1b998b", linewidth=1.2, label="horizontal hidden GR at true TVT")
        ax.set_title(f"{well}: hidden GR over true TVT vs typewell GR")
        ax.set_xlabel("TVT")
        ax.set_ylabel("GR")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(FIG / "typewell_alignment_examples.png", dpi=180)
    plt.close(fig)

    print(f"Wrote EDA artifacts to {OUT}")


if __name__ == "__main__":
    main()
