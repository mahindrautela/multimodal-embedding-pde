#!/usr/bin/env python3
"""Numerical checks of the math behind the reference distance (writes benchmark/math_checks.json).

1. Heat calibration: the closed-form distances between heat solutions with
   nu = 0.01 and nu in {0.011, 0.02, 0.05} (the heat control in the paper).
2. The 64-point 1D Laplacian matrices have the eigenvalues used by the solver.
3. The exact linear solutions in simulator.py match a brute-force matrix
   exponential of the whole grid system (n = 8, all linear PDEs and BCs).
4. Two algebraic identities: squared RMS distance from Gram matrices, and
   squared chord distance between unit embeddings = 2 - 2 * score.
5. The error-margin rule for triplets never resolves a triplet the wrong way
   when the error of every item is known.

These are floating-point checks, not proofs.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import eigh, expm

import simulator as S
from controls import HEAT_ANCHOR, HEAT_CANDIDATES

REPO = Path(__file__).resolve().parents[1]
FIVE_FRAMES = [0, 25, 50, 74, 99]


def heat_calibration() -> list[dict]:
    """Exact RMS distance between two heat solutions u = exp(-2 pi^2 nu t) sin(pi x) sin(pi y).

    The factor 0.5 is the RMS of sin(pi x) sin(pi y) over the unit square.
    """
    rows = []
    for nu in HEAT_CANDIDATES:
        row = {"anchor_nu": HEAT_ANCHOR, "candidate_nu": nu}
        for name, frames in (("five_frames", FIVE_FRAMES), ("all_100_frames", np.arange(100))):
            t = np.asarray(frames) / 99.0
            delta = np.exp(-2 * np.pi**2 * HEAT_ANCHOR * t) - np.exp(-2 * np.pi**2 * nu * t)
            row[name] = float(0.5 * np.sqrt(np.mean(delta**2)))
        rows.append(row)
    return rows


def laplacian_eigenvalue_errors(n: int = 64) -> dict[str, float]:
    """Largest gap between the eigenvalues of the dense matrix and the closed-form formula."""
    errors = {}
    for bc in S.BCS:
        if bc == "periodic":
            expected = -4 * n**2 * np.sin(np.pi * np.arange(n) / n) ** 2
        else:
            k = np.arange(1, n + 1) if bc == "dirichlet_zero" else np.arange(n)
            expected = -4 * n**2 * np.sin(np.pi * k / (2 * n)) ** 2
        expected = np.sort(expected)
        actual = eigh(S.dense_laplacian_1d(n, bc), eigvals_only=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=3e-11)
        errors[bc] = float(np.max(np.abs(expected - actual)))
    return errors


def full_system_matrix(pde: str, bc: str, n: int) -> np.ndarray:
    """The whole linear grid system for both fields as one (2 n^2) x (2 n^2) matrix."""
    p = S.PDE_COEFFICIENTS[pde]
    A = S.dense_laplacian_1d(n, bc)
    G = S.dense_gradient_1d(n, bc)
    I = np.eye(n)
    full = np.zeros((2 * n * n, 2 * n * n))
    for c, nu in enumerate((p["D_u"], p["D_v"])):
        Ax, Ay = nu * A, nu * A
        if pde == "advection_diffusion":
            Ax, Ay = Ax - p["c_x"] * G, Ay - p["c_y"] * G
        block = slice(c * n * n, (c + 1) * n * n)
        full[block, block] = np.kron(Ax, I) + np.kron(I, Ay)
    if pde == "linear_exchange":
        full += np.kron(p["kappa"] * np.array([[-1.0, 1.0], [1.0, -1.0]]), np.eye(n * n))
    return full


def exact_reference_errors(rng: np.random.Generator, n: int = 8) -> dict[str, float]:
    """simulator.exact_linear_reference vs expm of the full system matrix."""
    U0 = rng.uniform(0, 0.8, size=(2, 2, n, n))
    times = np.array([0.0, 0.3, 1.0])
    errors = {}
    for pde in ("diffusion", "advection_diffusion", "linear_exchange"):
        for bc in S.BCS:
            full = full_system_matrix(pde, bc, n)
            direct = np.stack([np.stack([(expm(t * full) @ u.ravel()).reshape(2, n, n) for t in times])
                               for u in U0])
            exact = S.exact_linear_reference(pde, bc, U0, times)
            errors[f"{pde}/{bc}"] = float(np.abs(exact - direct).max())
    assert max(errors.values()) < 1e-13
    return errors


def identity_errors(rng: np.random.Generator) -> dict[str, float]:
    """Gram-matrix distance identity and the embedding chord identity."""
    profiles = rng.normal(size=(6, 7, 5, 2, 4, 4))
    flat = profiles.reshape(6, -1) / np.sqrt(np.prod(profiles.shape[1:]))
    gram = flat @ flat.T
    distances = np.linalg.norm(flat[:, None] - flat[None, :], axis=-1)
    from_gram = gram.diagonal()[:, None] + gram.diagonal()[None, :] - 2 * gram
    np.testing.assert_allclose(distances**2, from_gram, atol=2e-15)

    z = rng.normal(size=(6, 7, 13))                      # 6 problems, 7 samples, 13 dims
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    score = np.einsum("ird,jrd->ij", z, z) / z.shape[1]
    chord2 = np.mean(np.sum((z[:, None] - z[None, :]) ** 2, axis=-1), axis=-1)
    np.testing.assert_allclose(chord2, 2 - 2 * score, atol=2e-15)
    assert np.linalg.eigvalsh(score).min() >= -1e-12
    return {"gram_distance_identity": float(np.max(np.abs(distances**2 - from_gram))),
            "embedding_chord_identity": float(np.max(np.abs(chord2 - (2 - 2 * score))))}


def margin_rule_check(rng: np.random.Generator) -> int:
    """With known errors eps, every triplet the margin rule resolves has the correct order.

    Returns the number of resolved triplets.
    """
    true = rng.normal(size=(15, 20))
    approx = true + 0.002 * rng.normal(size=true.shape)
    eps = np.linalg.norm(true - approx, axis=1)
    exact_d = np.linalg.norm(true[:, None] - true[None, :], axis=-1)
    approx_d = np.linalg.norm(approx[:, None] - approx[None, :], axis=-1)
    assert np.all(np.abs(exact_d - approx_d) <= eps[:, None] + eps[None, :] + 1e-13)
    resolved = 0
    for i, j, k in itertools.permutations(range(len(true)), 3):
        if approx_d[i, k] - approx_d[i, j] > 2 * eps[i] + eps[j] + eps[k]:
            assert exact_d[i, j] < exact_d[i, k]
            resolved += 1
    assert resolved > 0
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=REPO / "benchmark" / "math_checks.json")
    args = parser.parse_args()
    rng = np.random.default_rng(20261118)
    report = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "heat_calibration": heat_calibration(),
        "n64_laplacian_eigenvalue_max_abs_error": laplacian_eigenvalue_errors(),
        "exact_linear_reference_vs_full_expm_n8": exact_reference_errors(rng),
        "identity_max_errors": identity_errors(rng),
        "margin_rule_resolved_triplets": margin_rule_check(rng),
        "passed": True,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
