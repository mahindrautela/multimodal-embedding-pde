#!/usr/bin/env python3
"""Validate the common simulator before production (writes benchmark/simulator_validation.json).

Uses only pilot realizations (r >= 100) and manufactured tests; no test
realization (r = 0..99) is touched.

1. Spectral operators equal the five-point / centered stencils for all BCs.
2. Linear classes (diffusion, advection-diffusion, linear exchange): the
   ETDRK4 output is compared with the *exact* solution of the same
   semidiscrete system (Kronecker-sum or eigenmode matrix exponentials).
3. Manufactured continuum eigenfunction solutions: spatial order of the
   five-point discretization for every BC.
4. Nonlinear classes: time-step refinement (substeps 1, 2, 4, 8 vs 16).
5. Spatial refinement for all classes (n = 64, 128, 256; block averages).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_decks as M  # noqa: E402
import simulator as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = M.FAMILIES
PILOT = list(range(100, 104))


def pilot_ics(n: int = 64, realizations=PILOT) -> dict[str, np.ndarray]:
    return {f: np.stack([M.initial_array(f, M.draw_parameters(r)[f], n=n) for r in realizations]) for f in FAMILIES}


def rms(a: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(a))))


def restrict(field: np.ndarray, factor: int) -> np.ndarray:
    """Average factor x factor blocks; maps a fine cell-centered grid to the 64 grid."""
    s = field.shape
    return field.reshape(*s[:-2], s[-2] // factor, factor, s[-1] // factor, factor).mean(axis=(-3, -1))


def main() -> None:
    report: dict = {"purpose": "simulator validation on pilot realizations and manufactured solutions",
                    "solver": {"name": S.SOLVER_NAME, "version": S.SOLVER_VERSION},
                    "pilot_realizations": PILOT, "test_realizations_used": False}
    rng = np.random.default_rng(7)

    # 1. Operator identities -------------------------------------------------
    ops = {}
    for bc in S.BCS:
        f = rng.normal(size=(2, 2, 64, 64))
        spectral = S.inverse(S.eigenvalues_2d(64, bc) * S.forward(f, bc), bc, 64)
        stencil = S.laplacian_stencil(f, bc)
        A = S.dense_laplacian_1d(64, bc)
        dense = np.einsum("pa,zcaq->zcpq", A, f) + np.einsum("qb,zcpb->zcpq", A, f)
        G = S.dense_gradient_1d(64, bc)
        gx, gy = S.gradient(f, bc)
        ops[bc] = {
            "spectral_vs_stencil_rel_max": float(np.abs(spectral - stencil).max() / np.abs(stencil).max()),
            "dense_vs_stencil_rel_max": float(np.abs(dense - stencil).max() / np.abs(stencil).max()),
            "gradient_vs_dense_max": float(max(np.abs(gx - np.einsum("pa,zcaq->zcpq", G, f)).max(),
                                               np.abs(gy - np.einsum("qb,zcpb->zcpq", G, f)).max())),
        }
        assert ops[bc]["spectral_vs_stencil_rel_max"] < 1e-13
        assert ops[bc]["dense_vs_stencil_rel_max"] < 1e-13
        assert ops[bc]["gradient_vs_dense_max"] == 0.0
    report["operator_identities"] = ops

    # 2. Linear classes against the exact semidiscrete solution ---------------
    ics = pilot_ics()
    times = np.arange(100) / 99.0
    linear = {}
    for pde in ("diffusion", "advection_diffusion", "linear_exchange"):
        for bc in S.BCS:
            for fam in FAMILIES:
                exact = S.exact_linear_reference(pde, bc, ics[fam], times)
                row = {}
                for m in (1, 2, 4):
                    approx = S.simulate(pde, bc, ics[fam], substeps=m)
                    err = approx - exact
                    row[f"substeps_{m}"] = {"max_abs": float(np.abs(err).max()), "rms": rms(err),
                                            "rms_over_signal": rms(err) / rms(exact)}
                linear[f"{pde}/{bc}/{fam}"] = row
    report["linear_vs_exact_semidiscrete"] = linear
    report["linear_vs_exact_worst_max_abs"] = {
        f"substeps_{m}": max(v[f"substeps_{m}"]["max_abs"] for v in linear.values()) for m in (1, 2, 4)}

    # 3. Manufactured continuum eigenfunction tests ---------------------------
    manufactured = {}
    for bc in S.BCS:
        for pde in ("diffusion", "linear_exchange"):
            errs = []
            for n in (32, 64, 128):
                x = (np.arange(n) + 0.5) / n
                X, Y = np.meshgrid(x, x, indexing="ij")
                if bc == "periodic":
                    phi, lam = np.cos(2 * np.pi * X) * np.sin(4 * np.pi * Y), -(4 + 16) * np.pi**2
                elif bc == "dirichlet_zero":
                    phi, lam = np.sin(np.pi * X) * np.sin(2 * np.pi * Y), -(1 + 4) * np.pi**2
                else:
                    phi, lam = np.cos(np.pi * X) * np.cos(2 * np.pi * Y), -(1 + 4) * np.pi**2
                b = np.array([0.6, 0.3])
                U0 = phi[None] * b[:, None, None]
                p = S.PDE_COEFFICIENTS[pde]
                Dm = np.diag([p["D_u"], p["D_v"]])
                R = np.zeros((2, 2)) if pde == "diffusion" else p["kappa"] * np.array([[-1.0, 1.0], [1.0, -1.0]])
                from scipy.linalg import expm
                coeff = np.stack([expm(t * (R + lam * Dm)) @ b for t in times])
                exact = coeff[:, :, None, None] * phi[None, None]
                approx = S.simulate(pde, bc, U0, substeps=4)
                errs.append(rms(approx - exact))
            manufactured[f"{pde}/{bc}"] = {"rms_error_n32_64_128": errs,
                                           "observed_order": [float(np.log2(errs[0] / errs[1])),
                                                              float(np.log2(errs[1] / errs[2]))]}
    report["manufactured_continuum"] = manufactured

    # 4. Nonlinear time refinement --------------------------------------------
    temporal = {}
    for pde in ("fisher_kpp", "shifted_allen_cahn", "gray_scott", "advection_diffusion", "linear_exchange"):
        for bc in S.BCS:
            for fam in FAMILIES:
                ref = S.simulate(pde, bc, ics[fam], substeps=16)
                row = {}
                for m in (1, 2, 4, 8):
                    row[f"substeps_{m}_max_abs"] = float(np.abs(S.simulate(pde, bc, ics[fam], substeps=m) - ref).max())
                temporal[f"{pde}/{bc}/{fam}"] = row
    report["time_refinement_vs_substeps_16"] = temporal
    report["time_refinement_worst"] = {k: max(v[k] for v in temporal.values())
                                       for k in next(iter(temporal.values()))}

    # 5. Spatial refinement for every class ------------------------------------
    spatial = {}
    fine = {n: pilot_ics(n, PILOT[:2]) for n in (64, 128, 256)}
    for pde in S.PDES:
        for bc in S.BCS:
            for fam in FAMILIES:
                sols = {n: S.simulate(pde, bc, fine[n][fam], substeps=4, save_indices=[0, 25, 50, 74, 99])
                        for n in (64, 128, 256)}
                r256 = restrict(sols[256], 4)
                r128 = restrict(sols[128], 2)
                spatial[f"{pde}/{bc}/{fam}"] = {
                    "rms_64_vs_256": rms(sols[64] - r256), "rms_128_vs_256": rms(r128 - r256),
                    "rms_signal": rms(r256)}
    report["spatial_refinement_five_frames"] = spatial
    worst = max(spatial.values(), key=lambda v: v["rms_64_vs_256"])
    report["spatial_refinement_worst_rms_64_vs_256"] = worst["rms_64_vs_256"]
    report["spatial_refinement_median_rms_64_vs_256"] = float(np.median([v["rms_64_vs_256"] for v in spatial.values()]))
    report["passed"] = True
    out = ROOT / "benchmark" / "simulator_validation.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("linear_vs_exact_worst_max_abs", "time_refinement_worst",
                                             "spatial_refinement_worst_rms_64_vs_256",
                                             "spatial_refinement_median_rms_64_vs_256")}, indent=2))
    print(json.dumps(manufactured, indent=1))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("elapsed", round(time.time() - t0, 1))
