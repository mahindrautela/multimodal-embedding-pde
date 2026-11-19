#!/usr/bin/env python3
"""Production run: all 54 configurations x 100 test realizations (r = 0..99).

Reads the frozen protocol (benchmark/frozen_protocol.json) and writes, under
--data (default /home/claude/data):

  trajectories/five_frame/<config>.npy   float32 [100, 5, 2, 64, 64]
  sheets/<config>__r<rrr>.png            five-frame contact sheets
and, under benchmark/results/:
  reference_terms.npz   per-realization squared distances and cosines
                        (five-frame and all-100-frame) for the 54x54 matrix,
                        plus per-trajectory numerical error terms
  simulation_manifest.json  solver settings, checksums, diagnostics

The full 100-frame array (17.7 GB as float32) is not kept on disk; the
SHA-256 of every 100-frame float32 trajectory is recorded so that a
regenerated array can be audited bit-for-bit on the same platform.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_decks as M  # noqa: E402
import render as R  # noqa: E402
import simulator as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LINEAR = {"diffusion", "advection_diffusion", "linear_exchange"}


def config_list():
    return [(p, f, b) for p in S.PDES for f in M.FAMILIES for b in S.BCS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/home/claude/data"))
    ap.add_argument("--chunk", type=int, default=10)
    ap.add_argument("--realizations", type=int, default=100)
    args = ap.parse_args()
    protocol = json.loads((ROOT / "benchmark" / "frozen_protocol.json").read_text())
    substeps = protocol["solver"]["substeps_per_output_interval"]
    frames = protocol["observation_indices"]
    ranges = [tuple(r) for r in protocol["contact_sheet"]["channel_ranges"]]
    times = [k / 99 for k in frames]
    configs = config_list()
    names = [M.config_id(*c) for c in configs]
    K, N = len(configs), args.realizations
    five_dir = args.data / "trajectories" / "five_frame"
    sheet_dir = args.data / "sheets"
    five_dir.mkdir(parents=True, exist_ok=True)
    sheet_dir.mkdir(parents=True, exist_ok=True)
    out_dir = ROOT / "benchmark" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    five_all = np.zeros((K, N, len(frames), 2, 64, 64), dtype=np.float32)
    sq5 = np.zeros((N, K, K))
    sq100 = np.zeros((N, K, K))
    cos5 = np.zeros((N, K, K))
    cos100 = np.zeros((N, K, K))
    err5 = np.zeros((N, K))      # ||stored - reference||^2 (five frames)
    err100 = np.zeros((N, K))    # same, all frames
    tstep5 = np.zeros((N, K))    # ||X_m - X_2m||^2 (nonlinear; zero for linear)
    tstep100 = np.zeros((N, K))
    traj_sha = {}
    diagnostics = {name: {"min": np.inf, "max": -np.inf, "finite": True} for name in names}
    out_of_range = np.zeros((K, N), dtype=np.int64)
    exact_times = np.arange(100) / 99.0
    t_start = time.time()
    for start in range(0, N, args.chunk):
        rs = list(range(start, min(N, start + args.chunk)))
        block = np.zeros((K, len(rs), 100, 2, 64, 64), dtype=np.float32)
        for k, (pde, fam, bc) in enumerate(configs):
            U0 = np.stack([M.initial_array(fam, M.draw_parameters(r)[fam]) for r in rs])
            X = S.simulate(pde, bc, U0, substeps=substeps)
            X32 = X.astype(np.float32)
            block[k] = X32
            if pde in LINEAR:
                ref = S.exact_linear_reference(pde, bc, U0, exact_times)
            else:
                ref = S.simulate(pde, bc, U0, substeps=2 * substeps)
                d = X - ref
                tstep100[rs, k] = np.sum(d**2, axis=(1, 2, 3, 4))
                tstep5[rs, k] = np.sum(d[:, frames]**2, axis=(1, 2, 3, 4))
            e = X32.astype(np.float64) - ref
            err100[rs, k] = np.sum(e**2, axis=(1, 2, 3, 4))
            err5[rs, k] = np.sum(e[:, frames]**2, axis=(1, 2, 3, 4))
            diag = diagnostics[names[k]]
            diag["min"] = min(diag["min"], float(X32.min()))
            diag["max"] = max(diag["max"], float(X32.max()))
            diag["finite"] = diag["finite"] and bool(np.isfinite(X32).all())
            for i, r in enumerate(rs):
                traj_sha[f"{names[k]}__r{r:03d}"] = hashlib.sha256(X32[i].tobytes()).hexdigest()
                five_all[k, r] = X32[i, frames]
                image, bad = R.render_sheet(X32[i, frames], times, ranges)
                out_of_range[k, r] = bad
                image.save(sheet_dir / f"{names[k]}__r{r:03d}.png", optimize=False)
        # Per-realization Gram matrices in float64 from the stored float32 values.
        for i, r in enumerate(rs):
            full = block[:, i].reshape(K, -1).astype(np.float64)
            sub = block[:, i][:, frames].reshape(K, -1).astype(np.float64)
            for target_sq, target_cos, mat in ((sq100, cos100, full), (sq5, cos5, sub)):
                G = mat @ mat.T
                dg = np.diag(G)
                sq = np.maximum(dg[:, None] + dg[None, :] - 2 * G, 0.0)
                target_sq[r] = sq
                norm = np.sqrt(dg)
                target_cos[r] = G / np.outer(norm, norm)
        # Exactness check of the Gram identity on the first realization of the chunk.
        if start == 0:
            sub = block[:, 0][:, frames].reshape(K, -1).astype(np.float64)
            direct = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
            gram_check = float(np.abs(direct - sq5[0]).max() / max(direct.max(), 1e-300))
        del block
        print(f"realizations {rs[0]}-{rs[-1]} done, {time.time() - t_start:.0f}s", flush=True)

    for k, name in enumerate(names):
        np.save(five_dir / f"{name}.npy", five_all[k])
    five_sha = {name: hashlib.sha256((five_dir / f"{name}.npy").read_bytes()).hexdigest() for name in names}
    np.savez_compressed(out_dir / "reference_terms.npz", config_names=np.array(names),
                        sq5=sq5, sq100=sq100, cos5=cos5, cos100=cos100, err5=err5, err100=err100,
                        tstep5=tstep5, tstep100=tstep100, out_of_range=out_of_range)
    manifest = {
        "solver": {"name": S.SOLVER_NAME, "version": S.SOLVER_VERSION, "substeps_per_output_interval": substeps,
                   "internal_step": 1 / 99 / substeps, "scheme": "ETDRK4 with exact diffusion in the BC eigenbasis",
                   "numpy": np.__version__},
        "realizations": [0, N - 1], "configurations": names,
        "array_shape_full": [K, N, 100, 2, 64, 64], "array_shape_stored": [K, N, len(frames), 2, 64, 64],
        "dtype": "float32", "gram_identity_relative_check": gram_check,
        "five_frame_file_sha256": five_sha, "trajectory_sha256_float32_100_frames": traj_sha,
        "diagnostics": diagnostics,
        "contact_sheet": {"ranges": ranges, "out_of_range_pixels_total": int(out_of_range.sum()),
                          "sheets_with_out_of_range_pixels": int((out_of_range > 0).sum()),
                          "size": list(R.sheet_size())},
        "elapsed_seconds": time.time() - t_start,
    }
    (out_dir / "simulation_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k not in
                      ("trajectory_sha256_float32_100_frames", "five_frame_file_sha256", "configurations",
                       "diagnostics")}, indent=1))


if __name__ == "__main__":
    main()
