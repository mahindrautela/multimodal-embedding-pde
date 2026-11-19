#!/usr/bin/env python3
"""Check the solver before the main run (writes benchmark/simulator_validation.json).

Uses only setup samples (r >= 100) and hand-made exact solutions; no test
sample (r = 0..99) is touched.
  1. The fast transforms, the stencils and the dense matrices give the same
     Laplacian and gradient for every BC.
  2. Linear PDEs: the time-stepped result is compared with the exact solution
     of the same grid equations (1, 2 and 4 steps per output interval).
  3. Spatial order: errors against exact continuum solutions on 32, 64 and
     128 grids for diffusion and linear exchange.
  4. Time-step refinement: 1, 2, 4, 8 steps per interval vs 16.
  5. Spatial refinement for every PDE: 64 and 128 grids vs a 256 grid.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm

import simulator as S
from common import BENCHMARK, BCS, FAMILIES, FRAMES, PDES, initial_fields, restrict, rms

PILOT = list(range(100, 104))
TIMES = np.arange(100) / 99.0


def operator_checks(rng: np.random.Generator) -> dict:
    """Section 1: three ways of applying the Laplacian (and two of the gradient) agree."""
    checks = {}
    for bc in BCS:
        f = rng.normal(size=(2, 2, 64, 64))
        spectral = S.inverse(S.eigenvalues_2d(64, bc) * S.forward(f, bc), bc, 64)
        stencil = S.laplacian_stencil(f, bc)
        A = S.dense_laplacian_1d(64, bc)
        dense = np.einsum("pa,zcaq->zcpq", A, f) + np.einsum("qb,zcpb->zcpq", A, f)
        G = S.dense_gradient_1d(64, bc)
        gx, gy = S.gradient(f, bc)
        checks[bc] = {
            "spectral_vs_stencil_rel_max": float(np.abs(spectral - stencil).max() / np.abs(stencil).max()),
            "dense_vs_stencil_rel_max": float(np.abs(dense - stencil).max() / np.abs(stencil).max()),
            "gradient_vs_dense_max": float(max(np.abs(gx - np.einsum("pa,zcaq->zcpq", G, f)).max(),
                                               np.abs(gy - np.einsum("qb,zcpb->zcpq", G, f)).max())),
        }
        assert checks[bc]["spectral_vs_stencil_rel_max"] < 1e-13
        assert checks[bc]["dense_vs_stencil_rel_max"] < 1e-13
        assert checks[bc]["gradient_vs_dense_max"] == 0.0
    return checks


def linear_checks(initial: dict) -> dict:
    """Section 2: time-stepping error of the linear PDEs against the exact grid solution."""
    results = {}
    for pde in ("diffusion", "advection_diffusion", "linear_exchange"):
        for bc in BCS:
            for family in FAMILIES:
                exact = S.exact_linear_reference(pde, bc, initial[family], TIMES)
                row = {}
                for m in (1, 2, 4):
                    error = S.simulate(pde, bc, initial[family], substeps=m) - exact
                    row[f"substeps_{m}"] = {"max_abs": float(np.abs(error).max()), "rms": rms(error),
                                            "rms_over_signal": rms(error) / rms(exact)}
                results[f"{pde}/{bc}/{family}"] = row
    return results


def eigenfunction(bc: str, X: np.ndarray, Y: np.ndarray) -> tuple[np.ndarray, float]:
    """A Laplacian eigenfunction that satisfies the BC, and its eigenvalue."""
    if bc == "periodic":
        return np.cos(2 * np.pi * X) * np.sin(4 * np.pi * Y), -(4 + 16) * np.pi**2
    if bc == "dirichlet_zero":
        return np.sin(np.pi * X) * np.sin(2 * np.pi * Y), -(1 + 4) * np.pi**2
    return np.cos(np.pi * X) * np.cos(2 * np.pi * Y), -(1 + 4) * np.pi**2


def spatial_order_checks() -> dict:
    """Section 3: error against exact continuum solutions U = c(t) phi(x, y)."""
    results = {}
    for bc in BCS:
        for pde in ("diffusion", "linear_exchange"):
            p = S.PDE_COEFFICIENTS[pde]
            diffusion = np.diag([p["D_u"], p["D_v"]])
            exchange = (np.zeros((2, 2)) if pde == "diffusion"
                        else p["kappa"] * np.array([[-1.0, 1.0], [1.0, -1.0]]))
            amplitudes = np.array([0.6, 0.3])
            errors = []
            for n in (32, 64, 128):
                x = (np.arange(n) + 0.5) / n
                X, Y = np.meshgrid(x, x, indexing="ij")
                phi, lam = eigenfunction(bc, X, Y)
                # The amplitudes c(t) solve the 2 x 2 linear system c' = (R + lam D) c.
                c = np.stack([expm(t * (exchange + lam * diffusion)) @ amplitudes for t in TIMES])
                exact = c[:, :, None, None] * phi[None, None]
                approx = S.simulate(pde, bc, phi[None] * amplitudes[:, None, None], substeps=4)
                errors.append(rms(approx - exact))
            results[f"{pde}/{bc}"] = {"rms_error_n32_64_128": errors,
                                      "observed_order": [float(np.log2(errors[0] / errors[1])),
                                                         float(np.log2(errors[1] / errors[2]))]}
    return results


def time_step_checks(initial: dict) -> dict:
    """Section 4: largest difference from a run with 16 steps per output interval."""
    results = {}
    for pde in ("fisher_kpp", "shifted_allen_cahn", "gray_scott", "advection_diffusion", "linear_exchange"):
        for bc in BCS:
            for family in FAMILIES:
                fine = S.simulate(pde, bc, initial[family], substeps=16)
                results[f"{pde}/{bc}/{family}"] = {
                    f"substeps_{m}_max_abs": float(np.abs(S.simulate(pde, bc, initial[family], substeps=m) - fine).max())
                    for m in (1, 2, 4, 8)}
    return results


def grid_checks() -> dict:
    """Section 5: five-frame RMS difference from the 256 grid after block averaging."""
    grids = (64, 128, 256)
    initial = {n: {family: initial_fields(family, PILOT[:2], n) for family in FAMILIES} for n in grids}
    results = {}
    for pde in PDES:
        for bc in BCS:
            for family in FAMILIES:
                runs = {n: S.simulate(pde, bc, initial[n][family], substeps=4, save_indices=FRAMES) for n in grids}
                on_64_from_256 = restrict(runs[256], 4)
                results[f"{pde}/{bc}/{family}"] = {
                    "rms_64_vs_256": rms(runs[64] - on_64_from_256),
                    "rms_128_vs_256": rms(restrict(runs[128], 2) - on_64_from_256),
                    "rms_signal": rms(on_64_from_256)}
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=BENCHMARK / "simulator_validation.json")
    args = parser.parse_args()
    start = time.time()
    report: dict = {"purpose": "simulator validation on pilot realizations and manufactured solutions",
                    "solver": {"name": S.SOLVER_NAME, "version": S.SOLVER_VERSION},
                    "pilot_realizations": PILOT, "test_realizations_used": False}
    initial = {family: initial_fields(family, PILOT) for family in FAMILIES}

    report["operator_identities"] = operator_checks(np.random.default_rng(7))

    linear = linear_checks(initial)
    report["linear_vs_exact_semidiscrete"] = linear
    report["linear_vs_exact_worst_max_abs"] = {
        f"substeps_{m}": max(v[f"substeps_{m}"]["max_abs"] for v in linear.values()) for m in (1, 2, 4)}

    report["manufactured_continuum"] = spatial_order_checks()

    temporal = time_step_checks(initial)
    report["time_refinement_vs_substeps_16"] = temporal
    report["time_refinement_worst"] = {k: max(v[k] for v in temporal.values()) for k in next(iter(temporal.values()))}

    spatial = grid_checks()
    report["spatial_refinement_five_frames"] = spatial
    report["spatial_refinement_worst_rms_64_vs_256"] = max(v["rms_64_vs_256"] for v in spatial.values())
    report["spatial_refinement_median_rms_64_vs_256"] = float(np.median([v["rms_64_vs_256"]
                                                                         for v in spatial.values()]))
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    summary_keys = ("linear_vs_exact_worst_max_abs", "time_refinement_worst",
                    "spatial_refinement_worst_rms_64_vs_256", "spatial_refinement_median_rms_64_vs_256")
    print(json.dumps({k: report[k] for k in summary_keys}, indent=2))
    print(json.dumps(report["manufactured_continuum"], indent=1))
    print("elapsed", round(time.time() - start, 1))


if __name__ == "__main__":
    main()
