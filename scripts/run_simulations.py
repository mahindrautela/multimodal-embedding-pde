#!/usr/bin/env python3
"""Main run: simulate all 54 problems for the test samples r = 0..99.

Uses the settings in benchmark/frozen_protocol.json and writes

  under --data (default: data/):
    trajectories/five_frame/<problem>.npy   float32 [samples, 5, 2, 64, 64]
    sheets/<problem>__r<rrr>.png            the image inputs
  under --out-dir (default: benchmark/results/):
    reference_terms.npz       per-sample squared distances and cosines between
                              all 54 problems (five frames and all 100 frames),
                              and the numerical error of every trajectory
    simulation_manifest.json  solver settings, checksums and value ranges

The full 100-frame arrays (17.7 GB in float32) are not kept. Instead the
SHA-256 of every 100-frame float32 trajectory is recorded, so a rerun can be
checked bit for bit on the same platform.

Numerical error of each trajectory (squared, summed over fields and pixels):
  linear PDEs     stored float32 result vs the exact grid solution
  nonlinear PDEs  stored float32 result vs a run with half the time step;
                  the difference between the two runs is kept as well
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import simulator as S
from common import (DATA_DIR, FAMILIES, GRID, LINEAR_PDES, NAMES, N_STATES, PROBLEMS, RESULTS,
                    load_protocol, observation_times, sheet_ranges, initial_fields)
from render import render_sheet, sheet_size


def squared_norm(x: np.ndarray) -> np.ndarray:
    """Sum of squares over (field, x, y) and, if present, time; one value per sample."""
    return np.sum(x**2, axis=tuple(range(1, x.ndim)))


def gram_terms(states: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Squared distances and cosines between all problems for one sample.

    states: [K, ...] values of the K problems. Computed in float64 through the
    Gram matrix G: |a - b|^2 = G_aa + G_bb - 2 G_ab.
    """
    flat = states.reshape(len(states), -1).astype(np.float64)
    G = flat @ flat.T
    diag = np.diag(G)
    squared = np.maximum(diag[:, None] + diag[None, :] - 2 * G, 0.0)
    norms = np.sqrt(diag)
    return squared, G / np.outer(norms, norms)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=RESULTS)
    parser.add_argument("--chunk", type=int, default=10, help="samples simulated together")
    parser.add_argument("--realizations", type=int, default=100, help="number of test samples")
    args = parser.parse_args()

    protocol = load_protocol()
    substeps = protocol["solver"]["substeps_per_output_interval"]
    frames = protocol["observation_indices"]
    ranges = sheet_ranges(protocol)
    times = observation_times()
    K, N = len(PROBLEMS), args.realizations
    five_dir = args.data / "trajectories" / "five_frame"
    sheet_dir = args.data / "sheets"
    for folder in (five_dir, sheet_dir, args.out_dir):
        folder.mkdir(parents=True, exist_ok=True)

    five_all = np.zeros((K, N, len(frames), 2, GRID, GRID), dtype=np.float32)
    sq5, sq100, cos5, cos100 = (np.zeros((N, K, K)) for _ in range(4))
    err5, err100 = np.zeros((N, K)), np.zeros((N, K))         # |stored - reference|^2
    tstep5, tstep100 = np.zeros((N, K)), np.zeros((N, K))     # |dt run - dt/2 run|^2 (nonlinear only)
    trajectory_sha = {}
    diagnostics = {name: {"min": np.inf, "max": -np.inf, "finite": True} for name in NAMES}
    out_of_range = np.zeros((K, N), dtype=np.int64)
    exact_times = np.arange(N_STATES) / (N_STATES - 1)
    t_start = time.time()

    for start in range(0, N, args.chunk):
        rs = list(range(start, min(N, start + args.chunk)))
        block = np.zeros((K, len(rs), N_STATES, 2, GRID, GRID), dtype=np.float32)
        initial = {family: initial_fields(family, rs) for family in FAMILIES}
        for k, ((pde, family, bc), name) in enumerate(zip(PROBLEMS, NAMES)):
            U0 = initial[family]
            X = S.simulate(pde, bc, U0, substeps=substeps)
            X32 = X.astype(np.float32)                        # what is stored and encoded
            block[k] = X32

            if pde in LINEAR_PDES:
                reference = S.exact_linear_reference(pde, bc, U0, exact_times)
            else:
                reference = S.simulate(pde, bc, U0, substeps=2 * substeps)
                difference = X - reference
                tstep100[rs, k] = squared_norm(difference)
                tstep5[rs, k] = squared_norm(difference[:, frames])
            error = X32.astype(np.float64) - reference
            err100[rs, k] = squared_norm(error)
            err5[rs, k] = squared_norm(error[:, frames])

            diag = diagnostics[name]
            diag["min"] = min(diag["min"], float(X32.min()))
            diag["max"] = max(diag["max"], float(X32.max()))
            diag["finite"] = diag["finite"] and bool(np.isfinite(X32).all())
            for i, r in enumerate(rs):
                trajectory_sha[f"{name}__r{r:03d}"] = hashlib.sha256(X32[i].tobytes()).hexdigest()
                five_all[k, r] = X32[i, frames]
                image, bad = render_sheet(X32[i, frames], times, ranges)
                out_of_range[k, r] = bad
                image.save(sheet_dir / f"{name}__r{r:03d}.png", optimize=False)

        for i, r in enumerate(rs):
            sq100[r], cos100[r] = gram_terms(block[:, i])
            sq5[r], cos5[r] = gram_terms(block[:, i][:, frames])
        if start == 0:
            # Compare the Gram-matrix distances with direct differences once.
            sub = block[:, 0][:, frames].reshape(K, -1).astype(np.float64)
            direct = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
            gram_check = float(np.abs(direct - sq5[0]).max() / max(direct.max(), 1e-300))
        del block
        print(f"samples {rs[0]}-{rs[-1]} done, {time.time() - t_start:.0f}s", flush=True)

    for k, name in enumerate(NAMES):
        np.save(five_dir / f"{name}.npy", five_all[k])
    five_sha = {name: hashlib.sha256((five_dir / f"{name}.npy").read_bytes()).hexdigest() for name in NAMES}
    np.savez_compressed(args.out_dir / "reference_terms.npz", config_names=np.array(NAMES),
                        sq5=sq5, sq100=sq100, cos5=cos5, cos100=cos100, err5=err5, err100=err100,
                        tstep5=tstep5, tstep100=tstep100, out_of_range=out_of_range)
    manifest = {
        "solver": {"name": S.SOLVER_NAME, "version": S.SOLVER_VERSION, "substeps_per_output_interval": substeps,
                   "internal_step": 1 / 99 / substeps, "scheme": "ETDRK4 with exact diffusion in the BC eigenbasis",
                   "numpy": np.__version__},
        "realizations": [0, N - 1], "configurations": NAMES,
        "array_shape_full": [K, N, N_STATES, 2, GRID, GRID],
        "array_shape_stored": [K, N, len(frames), 2, GRID, GRID],
        "dtype": "float32", "gram_identity_relative_check": gram_check,
        "five_frame_file_sha256": five_sha, "trajectory_sha256_float32_100_frames": trajectory_sha,
        "diagnostics": diagnostics,
        "contact_sheet": {"ranges": ranges, "out_of_range_pixels_total": int(out_of_range.sum()),
                          "sheets_with_out_of_range_pixels": int((out_of_range > 0).sum()),
                          "size": list(sheet_size())},
        "elapsed_seconds": time.time() - t_start,
    }
    (args.out_dir / "simulation_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    long_keys = ("trajectory_sha256_float32_100_frames", "five_frame_file_sha256", "configurations", "diagnostics")
    print(json.dumps({key: value for key, value in manifest.items() if key not in long_keys}, indent=1))


if __name__ == "__main__":
    main()
