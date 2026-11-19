#!/usr/bin/env python3
"""Check the exact semidiscrete linear references against full-matrix exponentials (n=8).

Writes benchmark/exact_reference_check.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import simulator as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rng = np.random.default_rng(1)
    n = 8
    U0 = rng.uniform(0, 0.8, size=(2, 2, n, n))
    times = np.array([0.0, 0.3, 1.0])
    report = {"grid": n, "times": times.tolist(), "max_abs_difference": {}}
    for pde in ("diffusion", "advection_diffusion", "linear_exchange"):
        p = S.PDE_COEFFICIENTS[pde]
        for bc in S.BCS:
            A = S.dense_laplacian_1d(n, bc)
            G = S.dense_gradient_1d(n, bc)
            I = np.eye(n)
            blocks = []
            for nu in (p["D_u"], p["D_v"]):
                Ax, Ay = nu * A, nu * A
                if pde == "advection_diffusion":
                    Ax, Ay = Ax - p["c_x"] * G, Ay - p["c_y"] * G
                blocks.append(np.kron(Ax, I) + np.kron(I, Ay))
            full = np.zeros((2 * n * n, 2 * n * n))
            full[:n * n, :n * n], full[n * n:, n * n:] = blocks
            if pde == "linear_exchange":
                full += np.kron(p["kappa"] * np.array([[-1.0, 1.0], [1.0, -1.0]]), np.eye(n * n))
            direct = np.stack([np.stack([(expm(t * full) @ u.ravel()).reshape(2, n, n) for t in times]) for u in U0])
            ref = S.exact_linear_reference(pde, bc, U0, times)
            report["max_abs_difference"][f"{pde}/{bc}"] = float(np.abs(ref - direct).max())
    report["worst"] = max(report["max_abs_difference"].values())
    assert report["worst"] < 1e-13
    (ROOT / "benchmark" / "exact_reference_check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
