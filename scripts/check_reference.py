"""Reproduce mathematical calibration values and finite-dimensional checks.

This script does not generate the production PDE dataset or run encoders.
It checks identities numerically; it is not interval arithmetic or formal proof.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import scipy
from scipy.linalg import eigh, expm

ROOT = Path(__file__).resolve().parents[1]


def laplacian_1d(n: int, bc: str) -> np.ndarray:
    mat = np.diag(np.full(n, -2.0))
    mat += np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    if bc == "periodic":
        mat[0, -1] = mat[-1, 0] = 1.0
    elif bc == "dirichlet":
        mat[0, 0] = mat[-1, -1] = -3.0
    elif bc == "neumann":
        mat[0, 0] = mat[-1, -1] = -1.0
    else:
        raise ValueError(bc)
    return mat * n**2


def mode_reference(initial, bc, times, diffusivities, reaction):
    """Exact semidiscrete linear evolution, apart from floating-point error."""
    n = initial.shape[-1]
    lam, vec = eigh(laplacian_1d(n, bc))
    coeff = np.einsum("xa,cxy,yb->cab", vec, initial, vec)
    states = []
    for t in times:
        evolved = np.empty_like(coeff)
        for a in range(n):
            for b in range(n):
                generator = (lam[a] + lam[b]) * np.diag(diffusivities) + reaction
                evolved[:, a, b] = expm(t * generator) @ coeff[:, a, b]
        states.append(np.einsum("xa,cab,yb->cxy", vec, evolved, vec))
    return np.array(states)


def main():
    rng = np.random.default_rng(20261118)
    report = {
        "status": "analytical calibration and numerical consistency checks only",
        "production_trajectories_generated": False,
        "embedding_inference_performed": False,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "formal_verification": False,
    }

    # Closed-form continuum heat calibration; uniform observation weights.
    calibration = []
    for nu in (0.011, 0.020, 0.050):
        row = {"anchor_nu": 0.01, "candidate_nu": nu}
        for name, idx in (("five_frames", [0, 25, 50, 74, 99]),
                          ("all_100_frames", np.arange(100))):
            t = np.asarray(idx) / 99.0
            delta = np.exp(-2 * np.pi**2 * 0.01 * t) - np.exp(-2 * np.pi**2 * nu * t)
            row[name] = float(0.5 * np.sqrt(np.mean(delta**2)))
        calibration.append(row)
    report["continuum_heat_calibration"] = calibration

    # Cell-centered stencils must match the exact eigenvalue formulas used.
    errors = {}
    n = 64
    for bc in ("periodic", "dirichlet", "neumann"):
        k = np.arange(1, n + 1) if bc == "dirichlet" else np.arange(n)
        denominator = n if bc == "periodic" else 2 * n
        expected = np.sort(-4 * n**2 * np.sin(np.pi * k / denominator)**2)
        actual = eigh(laplacian_1d(n, bc), eigvals_only=True)
        errors[bc] = float(np.max(np.abs(expected - actual)))
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=3e-11)
    report["n64_laplacian_eigenvalue_max_abs_error"] = errors

    # Noncommuting channel diffusion/reaction: mode evolution vs the full matrix.
    n = 6
    initial = rng.uniform(0.0, 0.8, size=(2, n, n))
    diff = np.array([0.01, 0.005])
    reaction = 0.5 * np.array([[-1.0, 1.0], [1.0, -1.0]])
    times = np.array([0.0, 0.25, 1.0])
    linear_errors = {}
    for bc in ("periodic", "dirichlet", "neumann"):
        L = laplacian_1d(n, bc)
        L2 = np.kron(L, np.eye(n)) + np.kron(np.eye(n), L)
        A = np.kron(np.diag(diff), L2) + np.kron(reaction, np.eye(n * n))
        direct = np.stack([(expm(t * A) @ initial.ravel()).reshape(initial.shape) for t in times])
        modes = mode_reference(initial, bc, times, diff, reaction)
        linear_errors[bc] = float(np.max(np.abs(direct - modes)))
        np.testing.assert_allclose(modes, direct, rtol=2e-12, atol=2e-13)
    report["linear_exchange_mode_vs_full_exponential_max_error"] = linear_errors

    # The physical and embedding identities use distinct normalizations.
    profiles = rng.normal(size=(6, 7, 5, 2, 4, 4))
    flat = profiles.reshape(6, -1) / np.sqrt(np.prod(profiles.shape[1:]))
    kernel = flat @ flat.T
    distances = np.linalg.norm(flat[:, None] - flat[None, :], axis=-1)
    identity = kernel.diagonal()[:, None] + kernel.diagonal()[None, :] - 2 * kernel
    np.testing.assert_allclose(distances**2, identity, atol=2e-15)
    z = rng.normal(size=(6, 7, 13))
    z /= np.linalg.norm(z, axis=-1, keepdims=True)
    score = np.einsum("ird,jrd->ij", z, z) / z.shape[1]
    chord2 = np.mean(np.sum((z[:, None] - z[None, :])**2, axis=-1), axis=-1)
    np.testing.assert_allclose(chord2, 2 - 2 * score, atol=2e-15)
    assert np.linalg.eigvalsh(score).min() >= -1e-12
    report["raw_kernel_identity_max_error"] = float(np.max(np.abs(distances**2 - identity)))
    report["paired_embedding_chord_max_error"] = float(np.max(np.abs(chord2 - (2 - 2 * score))))

    # Known perturbation norms supply genuine finite-dimensional error bounds.
    true = rng.normal(size=(15, 20))
    approximate = true + 0.002 * rng.normal(size=true.shape)
    eps = np.linalg.norm(true - approximate, axis=1)
    exact_d = np.linalg.norm(true[:, None] - true[None, :], axis=-1)
    approx_d = np.linalg.norm(approximate[:, None] - approximate[None, :], axis=-1)
    bound = eps[:, None] + eps[None, :]
    assert np.all(np.abs(exact_d - approx_d) <= bound + 1e-13)
    resolved = 0
    for i in range(len(true)):
        for j in range(len(true)):
            for k in range(len(true)):
                if len({i, j, k}) < 3:
                    continue
                if approx_d[i, k] - approx_d[i, j] > 2 * eps[i] + eps[j] + eps[k]:
                    assert exact_d[i, j] < exact_d[i, k]
                    resolved += 1
    assert resolved > 0
    report["resolved_random_ranking_margins_checked"] = resolved
    report["passed"] = True
    out = ROOT / "benchmark" / "math_checks.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
