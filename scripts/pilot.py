#!/usr/bin/env python3
"""Pilot run on setup samples r = 100..119 (writes benchmark/pilot_report.json).

The checks below were written down before the pilot was run. No test sample
(r = 0..99) and no embedding model is used.
  P1  every saved state is finite;
  P2  every problem visibly changes: mean over samples of
      RMS(U(t=1) - U(0)) / RMS(U(0)) >= 0.05;
  P3  every PDE differs from diffusion (same IC and BC) by at least 100 times
      the largest time-step error;
  P4  Allen-Cahn threshold: some initial fields peak above 1/2 and some below;
  P5  the smallest value of each PDE is reported (advection-diffusion may dip
      below zero because centered differences do not preserve positivity);
  P6  image color range per field: from 0 to the smallest multiple of 0.1
      above every pilot value at the five observed frames;
  P7  distances on the 64 grid are compared with distances from 128-grid runs
      averaged back to 64 (Spearman and triplet-order flips).
The coefficients stay unchanged unless P1-P4 fail.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import simulator as S
from common import BCS, BENCHMARK, FAMILIES, FRAMES, PDES, PROBLEMS, initial_fields, restrict, rms

PILOT = list(range(100, 120))
REFINE = list(range(100, 105))         # samples used for the grid-refinement check (P7)
SUBSTEPS = 4


def distance_matrix(trajectories: dict) -> np.ndarray:
    """RMS distance between every pair of trajectories."""
    keys = list(trajectories)
    D = np.zeros((len(keys), len(keys)))
    for a, b in itertools.combinations(range(len(keys)), 2):
        D[a, b] = D[b, a] = rms(trajectories[keys[a]] - trajectories[keys[b]])
    return D


def triplet_flips(D1: np.ndarray, D2: np.ndarray) -> tuple[int, int]:
    """Number of (anchor, j, k) triplets ordered differently by D1 and D2, and the total."""
    flips = total = 0
    n = len(D1)
    for i in range(n):
        for j, k in itertools.combinations([x for x in range(n) if x != i], 2):
            total += 1
            flips += (D1[i, j] - D1[i, k]) * (D2[i, j] - D2[i, k]) < 0
    return int(flips), total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=BENCHMARK / "pilot_report.json")
    args = parser.parse_args()
    report = {"pilot_realizations": PILOT, "refinement_realizations": REFINE, "substeps": SUBSTEPS,
              "test_realizations_used": False, "embedding_models_used": False}

    # Simulate every problem with the normal step and with half the step.
    initial = {family: initial_fields(family, PILOT) for family in FAMILIES}
    frames, half_step, maxima, minima, evolution = {}, {}, {}, {}, {}
    for key in PROBLEMS:
        pde, family, bc = key
        X = S.simulate(pde, bc, initial[family], substeps=SUBSTEPS)   # raises if not finite (P1)
        frames[key] = X[:, FRAMES]
        half_step[key] = S.simulate(pde, bc, initial[family], substeps=2 * SUBSTEPS, save_indices=FRAMES)
        maxima[key] = X[:, FRAMES].max(axis=(0, 1, 3, 4)).tolist()     # one value per field
        minima[key] = float(X.min())
        evolution[key] = float(np.mean([rms(X[r, -1] - X[r, 0]) / rms(X[r, 0]) for r in range(len(PILOT))]))

    report["P1_all_finite"] = True
    report["P2_min_relative_evolution"] = min(evolution.values())
    report["P2_relative_evolution"] = {"/".join(k): v for k, v in evolution.items()}
    report["P2_pass"] = report["P2_min_relative_evolution"] >= 0.05

    max_time_error = max(float(np.abs(frames[k] - half_step[k]).max()) for k in frames)
    report["time_refinement_max_abs"] = max_time_error
    separation = {"/".join(k): rms(frames[k] - frames[("diffusion", k[1], k[2])])
                  for k in PROBLEMS if k[0] != "diffusion"}
    report["P3_min_rms_difference_from_diffusion"] = min(separation.values())
    report["P3_rms_difference_from_diffusion"] = separation
    report["P3_pass"] = min(separation.values()) >= 100 * max_time_error

    peaks = np.array([peak for family in FAMILIES for peak in initial[family].max(axis=(2, 3)).ravel().tolist()])
    report["P4_fields_with_max_above_half"] = int((peaks > 0.5).sum())
    report["P4_fields_with_max_below_half"] = int((peaks < 0.5).sum())
    report["P4_pass"] = bool((peaks > 0.5).any() and (peaks < 0.5).any())

    report["P5_global_minimum_by_class"] = {pde: min(minima[k] for k in minima if k[0] == pde) for pde in PDES}

    channel_max = [max(maxima[k][c] for k in maxima) for c in range(2)]
    vmax = [math.ceil(m / 0.1 - 1e-12) * 0.1 for m in channel_max]
    report["P6_pilot_channel_maximum"] = channel_max
    report["P6_contact_sheet_range"] = [[0.0, round(v, 10)] for v in vmax]
    report["P6_pilot_global_minimum"] = min(minima.values())

    # P7: are the distances stable when the grid is refined?
    refinement = {}
    upper = np.triu_indices(18, 1)
    for family in FAMILIES:
        coarse, fine = {}, {}
        U64, U128 = initial_fields(family, REFINE, 64), initial_fields(family, REFINE, 128)
        for pde in PDES:
            for bc in BCS:
                coarse[(pde, bc)] = S.simulate(pde, bc, U64, substeps=SUBSTEPS, save_indices=FRAMES)
                fine[(pde, bc)] = restrict(S.simulate(pde, bc, U128, substeps=SUBSTEPS, save_indices=FRAMES), 2)
        D64, D128 = distance_matrix(coarse), distance_matrix(fine)
        flips, total = triplet_flips(D64, D128)
        refinement[family] = {"spearman_D64_vs_D128": float(spearmanr(D64[upper], D128[upper]).statistic),
                              "max_abs_distance_change": float(np.abs(D64 - D128).max()),
                              "median_offdiag_distance": float(np.median(D64[upper])),
                              "min_offdiag_distance": float(D64[upper].min()),
                              "triplet_order_flips": flips, "triplets": total}
    report["P7_spatial_refinement_of_distances"] = refinement
    report["coefficients_changed"] = False
    report["passed"] = bool(report["P2_pass"] and report["P3_pass"] and report["P4_pass"])
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}, indent=2))
    print(json.dumps(refinement, indent=2))


if __name__ == "__main__":
    main()
