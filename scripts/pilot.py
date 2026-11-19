#!/usr/bin/env python3
"""Excluded numerical pilot (realizations r = 100..119 only).

Pre-declared checks (written before the pilot was run):
  P1  every saved state is finite;
  P2  visible evolution: for every configuration, mean over pilot ICs of
      RMS(U(t=1) - U(0)) / RMS(U(0)) >= 0.05;
  P3  every non-diffusion class differs from diffusion with the same IC and BC
      (five-frame RMS) by at least 100 x the largest time-refinement estimate;
  P4  Allen--Cahn threshold: the pilot ICs contain fields whose maximum lies
      above 1/2 and fields whose maximum lies below 1/2;
  P5  nonnegativity is reported for every class (not required for
      advection-diffusion, whose centered stencil is not positivity preserving);
  P6  contact-sheet scale: per channel vmin = 0 and vmax = the smallest multiple
      of 0.1 that bounds every pilot value at the five observed frames;
  P7  the 64-grid distance ordering is compared with 2x refined (n=128)
      trajectories on pilot ICs (Spearman, triplet flips).
Coefficients are unchanged unless P1--P4 fail. No embedding model is used.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_decks as M  # noqa: E402
import simulator as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PILOT = list(range(100, 120))
REFINE = list(range(100, 105))
FRAMES = [0, 25, 50, 74, 99]
SUBSTEPS = 4


def ics(family: str, realizations, n: int = 64) -> np.ndarray:
    return np.stack([M.initial_array(family, M.draw_parameters(r)[family], n=n) for r in realizations])


def rms(a) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def restrict(field: np.ndarray, factor: int) -> np.ndarray:
    s = field.shape
    return field.reshape(*s[:-2], s[-2] // factor, factor, s[-1] // factor, factor).mean(axis=(-3, -1))


def distance_matrix(trajs: dict) -> np.ndarray:
    keys = list(trajs)
    D = np.zeros((len(keys), len(keys)))
    for a, b in itertools.combinations(range(len(keys)), 2):
        D[a, b] = D[b, a] = rms(trajs[keys[a]] - trajs[keys[b]])
    return D


def triplet_flips(D1: np.ndarray, D2: np.ndarray) -> tuple[int, int]:
    n = len(D1)
    flips = total = 0
    for i in range(n):
        for j in range(n):
            for k in range(j + 1, n):
                if len({i, j, k}) < 3:
                    continue
                total += 1
                flips += (D1[i, j] - D1[i, k]) * (D2[i, j] - D2[i, k]) < 0
    return int(flips), total


def main() -> None:
    report = {"pilot_realizations": PILOT, "refinement_realizations": REFINE, "substeps": SUBSTEPS,
              "test_realizations_used": False, "embedding_models_used": False}
    configs = [(p, f, b) for p in S.PDES for f in M.FAMILIES for b in S.BCS]
    traj, traj2m, maxima, minima, evolution = {}, {}, {}, {}, {}
    for pde, fam, bc in configs:
        U0 = ics(fam, PILOT)
        X = S.simulate(pde, bc, U0, substeps=SUBSTEPS)
        X2 = S.simulate(pde, bc, U0, substeps=2 * SUBSTEPS, save_indices=FRAMES)
        key = (pde, fam, bc)
        traj[key] = X[:, FRAMES]
        traj2m[key] = X2
        maxima[key] = X[:, FRAMES].max(axis=(0, 1, 3, 4)).tolist()  # per channel
        minima[key] = float(X.min())
        evolution[key] = float(np.mean([rms(X[r, -1] - X[r, 0]) / rms(X[r, 0]) for r in range(len(PILOT))]))
    time_err = {k: float(np.abs(traj[k] - traj2m[k]).max()) for k in traj}
    report["P1_all_finite"] = True  # simulate() raises on non-finite values
    report["P2_min_relative_evolution"] = min(evolution.values())
    report["P2_relative_evolution"] = {"/".join(k): v for k, v in evolution.items()}
    report["P2_pass"] = report["P2_min_relative_evolution"] >= 0.05
    max_time_err = max(time_err.values())
    report["time_refinement_max_abs"] = max_time_err
    sep = {}
    for pde, fam, bc in configs:
        if pde == "diffusion":
            continue
        sep["/".join((pde, fam, bc))] = rms(traj[(pde, fam, bc)] - traj[("diffusion", fam, bc)])
    report["P3_min_rms_difference_from_diffusion"] = min(sep.values())
    report["P3_rms_difference_from_diffusion"] = sep
    report["P3_pass"] = min(sep.values()) >= 100 * max_time_err
    ac_max = []
    for fam in M.FAMILIES:
        U0 = ics(fam, PILOT)
        ac_max.extend(U0.max(axis=(2, 3)).ravel().tolist())
    ac_max = np.array(ac_max)
    report["P4_fields_with_max_above_half"] = int((ac_max > 0.5).sum())
    report["P4_fields_with_max_below_half"] = int((ac_max < 0.5).sum())
    report["P4_pass"] = bool((ac_max > 0.5).any() and (ac_max < 0.5).any())
    report["P5_global_minimum_by_class"] = {pde: min(minima[k] for k in minima if k[0] == pde) for pde in S.PDES}
    channel_max = [max(maxima[k][c] for k in maxima) for c in range(2)]
    vmax = [math.ceil(m / 0.1 - 1e-12) * 0.1 for m in channel_max]
    report["P6_pilot_channel_maximum"] = channel_max
    report["P6_contact_sheet_range"] = [[0.0, round(v, 10)] for v in vmax]
    report["P6_pilot_global_minimum"] = min(minima.values())

    refine = {}
    for fam in M.FAMILIES:
        coarse, fine = {}, {}
        U64, U128 = ics(fam, REFINE, 64), ics(fam, REFINE, 128)
        for pde in S.PDES:
            for bc in S.BCS:
                coarse[(pde, bc)] = S.simulate(pde, bc, U64, substeps=SUBSTEPS, save_indices=FRAMES)
                fine[(pde, bc)] = restrict(S.simulate(pde, bc, U128, substeps=SUBSTEPS, save_indices=FRAMES), 2)
        D64, D128 = distance_matrix(coarse), distance_matrix(fine)
        iu = np.triu_indices(18, 1)
        flips, total = triplet_flips(D64, D128)
        refine[fam] = {"spearman_D64_vs_D128": float(spearmanr(D64[iu], D128[iu]).statistic),
                       "max_abs_distance_change": float(np.abs(D64 - D128).max()),
                       "median_offdiag_distance": float(np.median(D64[iu])),
                       "min_offdiag_distance": float(D64[iu].min()),
                       "triplet_order_flips": flips, "triplets": total}
    report["P7_spatial_refinement_of_distances"] = refine
    report["coefficients_changed"] = False
    report["passed"] = bool(report["P2_pass"] and report["P3_pass"] and report["P4_pass"])
    (ROOT / "benchmark" / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}, indent=2))
    print(json.dumps(refine, indent=2))


if __name__ == "__main__":
    main()
