#!/usr/bin/env python3
"""Common cell-centered finite-difference simulator for the six benchmark PDEs.

All six classes share one spatial discretization and one time integrator:

* Grid: n x n cell centers x_p=(p+1/2)/n on [0,1]^2, array axes (channel, x, y).
* Laplacian: centered five-point stencil with BC-specific ghost values
  (periodic wrap; Dirichlet ghost = -interior; Neumann ghost = interior).
  This matrix is diagonalized exactly by the real FFT (periodic), the
  orthonormal DST-II (Dirichlet) or the orthonormal DCT-II (Neumann).
* First derivatives (advection only): centered differences with the same
  ghost values.
* Time integration: ETDRK4 (Cox--Matthews), with the diffusion operator
  D*Laplacian treated exactly in its eigenbasis and every other term
  (advection, exchange, reactions) treated explicitly. The phi-function
  coefficients are evaluated by the Kassam--Trefethen contour integral.
  Each output interval 1/99 is divided into ``substeps`` equal steps, so the
  saved times are hit exactly.

Everything is computed in float64. Callers decide how to store outputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft

PDE_COEFFICIENTS = {
    "diffusion": {"D_u": 0.01, "D_v": 0.005},
    "advection_diffusion": {"D_u": 0.01, "D_v": 0.005, "c_x": 0.25, "c_y": 0.10},
    "linear_exchange": {"D_u": 0.01, "D_v": 0.005, "kappa": 0.5},
    "fisher_kpp": {"D_u": 0.01, "D_v": 0.005, "r": 1.0},
    "shifted_allen_cahn": {"D_u": 0.01, "D_v": 0.005, "a": 4.0},
    "gray_scott": {"D_u": 0.01, "D_v": 0.005, "F": 0.04, "k": 0.06},
}
PDES = list(PDE_COEFFICIENTS)
BCS = ["periodic", "dirichlet_zero", "neumann_zero"]
N_STATES = 100
T_FINAL = 1.0
SOLVER_NAME = "cc-fd-etdrk4"
SOLVER_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Spatial operators
# ---------------------------------------------------------------------------
def eigenvalues_1d(n: int, bc: str) -> np.ndarray:
    """Eigenvalues of the 1D cell-centered stencil, in transform-coefficient order."""
    h2 = float(n) ** 2
    if bc == "periodic":
        k = np.arange(n // 2 + 1)  # rfft order along the last axis
        return -4.0 * h2 * np.sin(np.pi * k / n) ** 2
    if bc == "dirichlet_zero":
        k = np.arange(1, n + 1)  # DST-II coefficient j <-> mode j+1
        return -4.0 * h2 * np.sin(np.pi * k / (2 * n)) ** 2
    if bc == "neumann_zero":
        k = np.arange(n)
        return -4.0 * h2 * np.sin(np.pi * k / (2 * n)) ** 2
    raise ValueError(bc)


def eigenvalues_2d(n: int, bc: str) -> np.ndarray:
    """2D eigenvalue array matching the layout of ``forward``'s output."""
    if bc == "periodic":
        kx = np.arange(n)
        lam_x = -4.0 * n**2 * np.sin(np.pi * kx / n) ** 2  # full fft order on axis -2
        lam_y = eigenvalues_1d(n, bc)  # rfft order on axis -1
        return lam_x[:, None] + lam_y[None, :]
    lam = eigenvalues_1d(n, bc)
    return lam[:, None] + lam[None, :]


def forward(field: np.ndarray, bc: str) -> np.ndarray:
    if bc == "periodic":
        return sfft.rfftn(field, axes=(-2, -1), norm="ortho")
    if bc == "dirichlet_zero":
        return sfft.dstn(field, type=2, axes=(-2, -1), norm="ortho")
    if bc == "neumann_zero":
        return sfft.dctn(field, type=2, axes=(-2, -1), norm="ortho")
    raise ValueError(bc)


def inverse(coeff: np.ndarray, bc: str, n: int) -> np.ndarray:
    if bc == "periodic":
        return sfft.irfftn(coeff, s=(n, n), axes=(-2, -1), norm="ortho")
    if bc == "dirichlet_zero":
        return sfft.idstn(coeff, type=2, axes=(-2, -1), norm="ortho")
    if bc == "neumann_zero":
        return sfft.idctn(coeff, type=2, axes=(-2, -1), norm="ortho")
    raise ValueError(bc)


def pad_ghost(field: np.ndarray, bc: str) -> np.ndarray:
    """Add one ghost layer on both spatial axes using the BC rule."""
    if bc == "periodic":
        return np.pad(field, [(0, 0)] * (field.ndim - 2) + [(1, 1), (1, 1)], mode="wrap")
    padded = np.pad(field, [(0, 0)] * (field.ndim - 2) + [(1, 1), (1, 1)], mode="edge")
    if bc == "dirichlet_zero":
        padded[..., 0, :] *= -1.0
        padded[..., -1, :] *= -1.0
        padded[..., :, 0] *= -1.0
        padded[..., :, -1] *= -1.0
        # Corner ghosts are never used by the five-point or centered stencils.
        return padded
    if bc == "neumann_zero":
        return padded
    raise ValueError(bc)


def laplacian_stencil(field: np.ndarray, bc: str) -> np.ndarray:
    """Direct five-point stencil (used only for validation)."""
    n = field.shape[-1]
    g = pad_ghost(field, bc)
    c = g[..., 1:-1, 1:-1]
    return (g[..., 2:, 1:-1] + g[..., :-2, 1:-1] + g[..., 1:-1, 2:] + g[..., 1:-1, :-2] - 4.0 * c) * n**2


def gradient(field: np.ndarray, bc: str) -> tuple[np.ndarray, np.ndarray]:
    """Centered first differences (d/dx along axis -2, d/dy along axis -1)."""
    n = field.shape[-1]
    g = pad_ghost(field, bc)
    dx = (g[..., 2:, 1:-1] - g[..., :-2, 1:-1]) * (n / 2.0)
    dy = (g[..., 1:-1, 2:] - g[..., 1:-1, :-2]) * (n / 2.0)
    return dx, dy


# ---------------------------------------------------------------------------
# Explicit terms G(U, grad U)
# ---------------------------------------------------------------------------
def explicit_term(pde: str, U: np.ndarray, bc: str, coefficients: dict | None = None) -> np.ndarray:
    """U has shape [..., 2, n, n]; returns G with the same shape."""
    p = PDE_COEFFICIENTS[pde] if coefficients is None else coefficients
    u, v = U[..., 0, :, :], U[..., 1, :, :]
    if pde == "diffusion":
        return np.zeros_like(U)
    if pde == "advection_diffusion":
        dx, dy = gradient(U, bc)
        return -p["c_x"] * dx - p["c_y"] * dy
    if pde == "linear_exchange":
        return np.stack([p["kappa"] * (v - u), p["kappa"] * (u - v)], axis=-3)
    if pde == "fisher_kpp":
        return p["r"] * U * (1.0 - U)
    if pde == "shifted_allen_cahn":
        return p["a"] * U * (1.0 - U) * (U - 0.5)
    if pde == "gray_scott":
        uv2 = u * v * v
        return np.stack([-uv2 + p["F"] * (1.0 - u), uv2 - (p["F"] + p["k"]) * v], axis=-3)
    raise ValueError(pde)


# ---------------------------------------------------------------------------
# ETDRK4
# ---------------------------------------------------------------------------
@dataclass
class ETDCoefficients:
    E: np.ndarray
    E2: np.ndarray
    Q: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    f3: np.ndarray


def etdrk4_coefficients(L: np.ndarray, h: float, contour_points: int = 64) -> ETDCoefficients:
    """Kassam--Trefethen contour-integral evaluation for a real diagonal operator L."""
    hL = h * L
    r = np.exp(1j * np.pi * (np.arange(1, contour_points + 1) - 0.5) / contour_points)
    LR = hL[..., None] + r
    expLR = np.exp(LR)
    Q = h * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=-1))
    f1 = h * np.real(np.mean((-4 - LR + expLR * (4 - 3 * LR + LR**2)) / LR**3, axis=-1))
    f2 = h * np.real(np.mean((2 + LR + expLR * (-2 + LR)) / LR**3, axis=-1))
    f3 = h * np.real(np.mean((-4 - 3 * LR - LR**2 + expLR * (4 - LR)) / LR**3, axis=-1))
    return ETDCoefficients(np.exp(hL), np.exp(hL / 2), Q, f1, f2, f3)


def simulate(pde: str, bc: str, U0: np.ndarray, substeps: int = 4, n_states: int = N_STATES,
             t_final: float = T_FINAL, coefficients: dict | None = None,
             save_indices: list[int] | None = None) -> np.ndarray:
    """Integrate a batch of initial conditions.

    U0: float array [B, 2, n, n] (or [2, n, n]). Returns float64
    [B, n_saved, 2, n, n] at t_k = k*t_final/(n_states-1) for the requested
    save indices (all states by default). Raises FloatingPointError on
    non-finite values; nothing is clipped.
    """
    squeeze = U0.ndim == 3
    U = np.array(U0[None] if squeeze else U0, dtype=np.float64)
    B, C, n, _ = U.shape
    p = PDE_COEFFICIENTS[pde] if coefficients is None else coefficients
    diffusivity = np.array([p["D_u"], p["D_v"]], dtype=np.float64)
    lam = eigenvalues_2d(n, bc)
    L = diffusivity[:, None, None] * lam[None]  # [2, kx, ky]
    h = t_final / (n_states - 1) / substeps
    co = etdrk4_coefficients(L, h)
    save = list(range(n_states)) if save_indices is None else list(save_indices)
    save_set = {k: i for i, k in enumerate(save)}
    out = np.empty((B, len(save), C, n, n), dtype=np.float64)

    def N_hat(v_hat: np.ndarray) -> np.ndarray:
        return forward(explicit_term(pde, inverse(v_hat, bc, n), bc, p), bc)

    v = forward(U, bc)
    if 0 in save_set:
        out[:, save_set[0]] = U
    linear_only = pde == "diffusion"
    for k in range(1, n_states):
        for _ in range(substeps):
            if linear_only:
                v = co.E * v
                continue
            Nv = N_hat(v)
            a = co.E2 * v + co.Q * Nv
            Na = N_hat(a)
            b = co.E2 * v + co.Q * Na
            Nb = N_hat(b)
            c = co.E2 * a + co.Q * (2 * Nb - Nv)
            Nc = N_hat(c)
            v = co.E * v + Nv * co.f1 + 2 * (Na + Nb) * co.f2 + Nc * co.f3
        if k in save_set:
            state = inverse(v, bc, n)
            if not np.isfinite(state).all():
                raise FloatingPointError(f"non-finite state for {pde}/{bc} at saved index {k}")
            out[:, save_set[k]] = state
    return out[0] if squeeze else out


# ---------------------------------------------------------------------------
# Exact semidiscrete references for the three linear classes
# ---------------------------------------------------------------------------
def dense_laplacian_1d(n: int, bc: str) -> np.ndarray:
    mat = np.diag(np.full(n, -2.0)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    if bc == "periodic":
        mat[0, -1] = mat[-1, 0] = 1.0
    elif bc == "dirichlet_zero":
        mat[0, 0] = mat[-1, -1] = -3.0
    elif bc == "neumann_zero":
        mat[0, 0] = mat[-1, -1] = -1.0
    else:
        raise ValueError(bc)
    return mat * n**2


def dense_gradient_1d(n: int, bc: str) -> np.ndarray:
    mat = (np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)) * (n / 2.0)
    if bc == "periodic":
        mat[0, -1] = -n / 2.0
        mat[-1, 0] = n / 2.0
    elif bc == "dirichlet_zero":
        mat[0, 0] = n / 2.0   # -(ghost=-u0)
        mat[-1, -1] = -n / 2.0
    elif bc == "neumann_zero":
        mat[0, 0] = -n / 2.0  # -(ghost=u0)
        mat[-1, -1] = n / 2.0
    else:
        raise ValueError(bc)
    return mat


def exact_linear_reference(pde: str, bc: str, U0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Exact solution of the declared semidiscrete linear system (no time-stepping error).

    U0: [B, 2, n, n]; returns [B, len(times), 2, n, n] in float64.
    Diffusion/advection-diffusion use the Kronecker-sum identity
    exp(t(A_x (+) A_y)) = exp(t A_x) (x) exp(t A_y) per channel; linear exchange
    uses the symmetric Laplacian eigenbasis and a 2x2 matrix exponential per mode.
    """
    from scipy.linalg import eigh, expm

    U0 = np.asarray(U0, dtype=np.float64)
    B, C, n, _ = U0.shape
    p = PDE_COEFFICIENTS[pde]
    diffusivity = [p["D_u"], p["D_v"]]
    out = np.empty((B, len(times), C, n, n))
    A = dense_laplacian_1d(n, bc)
    if pde in ("diffusion", "advection_diffusion"):
        G = dense_gradient_1d(n, bc)
        for c in range(C):
            Ax = diffusivity[c] * A
            Ay = diffusivity[c] * A
            if pde == "advection_diffusion":
                Ax = Ax - p["c_x"] * G
                Ay = Ay - p["c_y"] * G
            for ti, t in enumerate(times):
                Ex, Ey = expm(t * Ax), expm(t * Ay)
                out[:, ti, c] = Ex @ U0[:, c] @ Ey.T
        return out
    if pde == "linear_exchange":
        lam, vec = eigh(A)
        coeff = np.einsum("xa,zcxy,yb->zcab", vec, U0, vec, optimize=True)
        lam2 = lam[:, None] + lam[None, :]
        kappa = p["kappa"]
        for ti, t in enumerate(times):
            # Closed-form exp of the 2x2 generator t*(lam*diag(D) + R) for every mode.
            a11 = t * (lam2 * diffusivity[0] - kappa)
            a22 = t * (lam2 * diffusivity[1] - kappa)
            off = t * kappa
            mean = 0.5 * (a11 + a22)
            delta = np.sqrt((0.5 * (a11 - a22)) ** 2 + off**2)
            ch = np.cosh(delta)
            sh = np.where(delta > 0, np.sinh(delta) / np.where(delta > 0, delta, 1.0), 1.0)
            em = np.exp(mean)
            M11 = em * (ch + sh * (a11 - mean))
            M22 = em * (ch + sh * (a22 - mean))
            M12 = em * sh * off
            c0, c1 = coeff[:, 0], coeff[:, 1]
            evolved = np.stack([M11 * c0 + M12 * c1, M12 * c0 + M22 * c1], axis=1)
            out[:, ti] = np.einsum("xa,zcab,yb->zcxy", vec, evolved, vec, optimize=True)
        return out
    raise ValueError(f"{pde} has no exact linear reference")
