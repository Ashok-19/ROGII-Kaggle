"""Clean-room sequential likelihood-weighted particle filter for E013."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from numba import njit


@dataclass(frozen=True)
class LikelihoodPFConfig:
    n_particles: int = 384
    seed_ids: tuple[int, ...] = tuple(range(16))
    seed_likelihood_temperature: float = 5.0
    initial_position_spread: float = 4.5
    initial_rate_spread: float = 0.01
    typewell_grid_step: float = 0.2
    rate_momentum: float = 0.998
    rate_noise: float = 0.002
    position_noise: float = 0.005
    resample_position_noise: float = 0.1
    resample_rate_noise: float = 0.001
    resample_effective_fraction: float = 0.5
    gr_sigma_min: float = 10.0
    gr_sigma_max: float = 60.0
    state_boundary_padding: float = 100.0

    @classmethod
    def from_mapping(cls, raw: dict) -> "LikelihoodPFConfig":
        return cls(
            n_particles=int(raw["n_particles"]),
            seed_ids=tuple(int(x) for x in raw["seed_ids"]),
            seed_likelihood_temperature=float(raw["seed_likelihood_temperature"]),
            initial_position_spread=float(raw["initial_position_spread"]),
            initial_rate_spread=float(raw["initial_rate_spread"]),
            typewell_grid_step=float(raw["typewell_grid_step"]),
            rate_momentum=float(raw["rate_momentum"]),
            rate_noise=float(raw["rate_noise"]),
            position_noise=float(raw["position_noise"]),
            resample_position_noise=float(raw["resample_position_noise"]),
            resample_rate_noise=float(raw["resample_rate_noise"]),
            resample_effective_fraction=float(raw["resample_effective_fraction"]),
            gr_sigma_min=float(raw["gr_sigma_min"]),
            gr_sigma_max=float(raw["gr_sigma_max"]),
            state_boundary_padding=float(raw["state_boundary_padding"]),
        )


@njit(cache=True)
def _interp_uniform(values: np.ndarray, x: float, x0: float, step: float) -> float:
    pos = (x - x0) / step
    left = int(pos)
    if left < 0:
        return values[0]
    last = values.size - 1
    if left >= last:
        return values[last]
    frac = pos - left
    return values[left] * (1.0 - frac) + values[left + 1] * frac


@njit(cache=True, nogil=True)
def _simulate_seed_bank(
    md: np.ndarray,
    z: np.ndarray,
    gr: np.ndarray,
    tw_grid_gr: np.ndarray,
    tw_min: float,
    tw_step: float,
    gr_sigma: float,
    initial_u: float,
    initial_rate: float,
    seed_ids: np.ndarray,
    n_particles: int,
    initial_position_spread: float,
    initial_rate_spread: float,
    rate_momentum: float,
    rate_noise: float,
    position_noise: float,
    resample_position_noise: float,
    resample_rate_noise: float,
    resample_effective_fraction: float,
    state_boundary_padding: float,
    use_emission: bool,
) -> tuple[np.ndarray, np.ndarray]:
    n_rows = md.size
    n_seeds = seed_ids.size
    predictions = np.empty((n_seeds, n_rows), dtype=np.float64)
    log_likelihoods = np.zeros(n_seeds, dtype=np.float64)
    upper_tvt = tw_min + tw_grid_gr.size * tw_step

    for seed_pos in range(n_seeds):
        np.random.seed(int(seed_ids[seed_pos]))
        u = np.empty(n_particles, dtype=np.float64)
        rate = np.empty(n_particles, dtype=np.float64)
        weights = np.full(n_particles, 1.0 / n_particles, dtype=np.float64)
        for j in range(n_particles):
            u[j] = initial_u + initial_position_spread * np.random.randn()
            rate[j] = initial_rate + initial_rate_spread * np.random.randn()

        previous_md = md[0] - 1.0
        total_log_likelihood = 0.0
        for i in range(n_rows):
            delta_md = md[i] - previous_md
            if delta_md < 1.0:
                delta_md = 1.0
            for j in range(n_particles):
                rate[j] = rate_momentum * rate[j] + rate_noise * np.random.randn()
                u[j] += rate[j] * delta_md + position_noise * np.random.randn()
                tvt = u[j] - z[i]
                if tvt < tw_min - state_boundary_padding:
                    tvt = tw_min - state_boundary_padding
                elif tvt > upper_tvt + state_boundary_padding:
                    tvt = upper_tvt + state_boundary_padding
                u[j] = tvt + z[i]

            if use_emission:
                average_likelihood = 0.0
                for j in range(n_particles):
                    expected_gr = _interp_uniform(tw_grid_gr, u[j] - z[i], tw_min, tw_step)
                    standardized = (gr[i] - expected_gr) / gr_sigma
                    squared = standardized * standardized
                    if squared > 600.0:
                        squared = 600.0
                    likelihood = np.exp(-0.5 * squared)
                    if likelihood < 1e-300:
                        likelihood = 1e-300
                    average_likelihood += weights[j] * likelihood
                    weights[j] *= likelihood
                if average_likelihood < 1e-300:
                    average_likelihood = 1e-300
                total_log_likelihood += np.log(average_likelihood)
                weight_sum = weights.sum()
                if weight_sum > 0.0:
                    weights /= weight_sum
                else:
                    weights[:] = 1.0 / n_particles
            effective = 1.0 / np.sum(weights * weights)
            if effective < resample_effective_fraction * n_particles:
                cumulative = np.empty(n_particles, dtype=np.float64)
                running = 0.0
                for j in range(n_particles):
                    running += weights[j]
                    cumulative[j] = running
                start = np.random.uniform(0.0, 1.0 / n_particles)
                new_u = np.empty(n_particles, dtype=np.float64)
                new_rate = np.empty(n_particles, dtype=np.float64)
                cursor = 0
                for j in range(n_particles):
                    threshold = start + j / n_particles
                    while cursor < n_particles - 1 and cumulative[cursor] < threshold:
                        cursor += 1
                    new_u[j] = u[cursor] + resample_position_noise * np.random.randn()
                    new_rate[j] = rate[cursor] + resample_rate_noise * np.random.randn()
                u = new_u
                rate = new_rate
                weights[:] = 1.0 / n_particles

            estimate = 0.0
            for j in range(n_particles):
                estimate += weights[j] * (u[j] - z[i])
            predictions[seed_pos, i] = estimate
            previous_md = md[i]
        log_likelihoods[seed_pos] = total_log_likelihood
    return predictions, log_likelihoods


def _validate_frames(horizontal: pd.DataFrame, typewell: pd.DataFrame) -> None:
    required_h = {"MD", "Z", "GR", "TVT_input"}
    required_t = {"TVT", "GR"}
    if missing := required_h - set(horizontal.columns):
        raise ValueError(f"horizontal well missing columns: {sorted(missing)}")
    if missing := required_t - set(typewell.columns):
        raise ValueError(f"typewell missing columns: {sorted(missing)}")
    visible = horizontal["TVT_input"].notna().to_numpy()
    if visible.sum() < 3 or visible.all():
        raise ValueError("requires visible prefix and non-empty hidden suffix")
    first_hidden = int(np.flatnonzero(~visible)[0])
    if visible[first_hidden:].any():
        raise ValueError("TVT_input must be a contiguous visible prefix")


def predict_likelihood_pf(
    horizontal: pd.DataFrame,
    typewell: pd.DataFrame,
    config: LikelihoodPFConfig,
    *,
    use_emission: bool = True,
    seed_ids: Iterable[int] | None = None,
) -> np.ndarray:
    """Return a full-length TVT vector with visible rows preserved exactly."""
    _validate_frames(horizontal, typewell)
    hw = horizontal.copy(deep=True)
    tw = typewell.sort_values("TVT").reset_index(drop=True).copy(deep=True)
    tw_tvt = tw["TVT"].to_numpy(float)
    tw_gr = tw["GR"].ffill().bfill().to_numpy(float)
    if tw_tvt.size < 2 or not np.isfinite(tw_tvt).all() or not np.isfinite(tw_gr).all():
        raise ValueError("typewell support must be finite and non-degenerate")

    visible = hw["TVT_input"].notna().to_numpy()
    hidden = ~visible
    known = hw.loc[visible]
    hidden_rows = hw.loc[hidden]
    output = hw["TVT_input"].to_numpy(float).copy()

    typewell_at_visible = np.interp(known["TVT_input"].to_numpy(float), tw_tvt, tw_gr)
    visible_gr = known["GR"].fillna(0.0).to_numpy(float)
    gr_sigma = float(np.clip(np.nanstd(visible_gr - typewell_at_visible), config.gr_sigma_min, config.gr_sigma_max))

    tail = known.tail(30)
    delta_tvt = np.diff(tail["TVT_input"].to_numpy(float))
    delta_z = np.diff(tail["Z"].to_numpy(float))
    delta_md = np.diff(tail["MD"].to_numpy(float))
    valid = delta_md > 0
    initial_rate = float(np.median((delta_tvt + delta_z)[valid] / delta_md[valid])) if valid.sum() >= 3 else 0.0
    last = known.iloc[-1]
    initial_u = float(last["TVT_input"]) + float(last["Z"])

    tw_min = float(tw_tvt.min())
    grid = np.arange(tw_min, float(tw_tvt.max()) + config.typewell_grid_step, config.typewell_grid_step)
    grid_gr = np.interp(grid, tw_tvt, tw_gr)
    completed_gr = hw["GR"].interpolate(limit_direction="both").fillna(float(np.nanmean(tw_gr))).to_numpy(float)
    hidden_gr = completed_gr[hidden]

    ids = np.asarray(tuple(config.seed_ids if seed_ids is None else seed_ids), dtype=np.int64)
    if ids.size == 0 or len(set(ids.tolist())) != ids.size:
        raise ValueError("seed IDs must be non-empty and unique")
    seed_predictions, log_likelihoods = _simulate_seed_bank(
        hidden_rows["MD"].to_numpy(float),
        hidden_rows["Z"].to_numpy(float),
        hidden_gr,
        grid_gr,
        tw_min,
        config.typewell_grid_step,
        gr_sigma,
        initial_u,
        initial_rate,
        ids,
        config.n_particles,
        config.initial_position_spread,
        config.initial_rate_spread,
        config.rate_momentum,
        config.rate_noise,
        config.position_noise,
        config.resample_position_noise,
        config.resample_rate_noise,
        config.resample_effective_fraction,
        config.state_boundary_padding,
        use_emission,
    )
    centered = log_likelihoods - np.max(log_likelihoods)
    seed_weights = np.exp(centered / config.seed_likelihood_temperature)
    seed_weights /= seed_weights.sum()
    output[hidden] = np.sum(seed_weights[:, None] * seed_predictions, axis=0)
    if not np.isfinite(output).all():
        raise ValueError("particle filter produced non-finite predictions")
    return output
