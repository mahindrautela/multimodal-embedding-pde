"""One finite-difference solver for all six PDEs and all three boundary conditions.

Every equation has the form  dU/dt = D * Laplacian(U) + G(U, grad U)  for the
two fields U = (u, v). The solver works as follows.

Grid
    n x n cell centers x_p = (p + 1/2)/n on [0, 1]^2. Arrays are ordered
    (field, x, y), with any batch dimensions in front.

Space
    Laplacian: the five-point stencil. At the boundary it uses one layer of
    "ghost" values: periodic wraps around, zero Dirichlet uses minus the
    neighbour, zero Neumann copies the neighbour. For each BC this matrix is
    diagonalized exactly by a fast transform (real FFT, DST-II or DCT-II), so
    diffusion can be solved exactly in that basis.
    Advection uses centered first differences with the same ghost values.

Time
    ETDRK4 (Cox & Matthews 2002) with the phi-functions evaluated by the
    contour integral of Kassam & Trefethen (2005). Diffusion is treated
    exactly; advection, exchange and reaction terms are treated explicitly.
    Each output interval 1/99 is split into `substeps` equal steps, so the
    saved times are hit exactly.

Everything is computed in float64; callers decide how to store results.
The file also provides exact solutions of the grid equations for the three
linear PDEs, used to measure the time-stepping error.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft
from scipy.linalg import eigh, expm

from make_decks import BOUNDARIES, PDE_SPECS

# Coefficients are defined once, in make_decks.py, and shared with the decks.
PDE_COEFFICIENTS = {pde: spec["parameters"] for pde, spec in PDE_SPECS.items()}
PDES = list(PDE_COEFFICIENTS)
BCS = list(BOUNDARIES)
N_STATES = 100
T_FINAL = 1.0
SOLVER_NAME = "cc-fd-etdrk4"
SOLVER_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
# Laplacian eigenvalues and the matching fast transforms
# ---------------------------------------------------------------------------
def eigenvalues_1d(n: int, bc: str) -> np.ndarray:
    """Eigenvalues of the 1D stencil, in the order of the transform coefficients."""
    h2 = float(n) ** 2
    if bc == "periodic":
        k = np.arange(n // 2 + 1)                  # rfft keeps the first n//2 + 1 modes
        return -4.0 * h2 * np.sin(np.pi * k / n) ** 2
    if bc == "dirichlet_zero":
        k = np.arange(1, n + 1)                    # DST-II coefficient j is mode j + 1
        return -4.0 * h2 * np.sin(np.pi * k / (2 * n)) ** 2
    if bc == "neumann_zero":
        k = np.arange(n)                           # DCT-II coefficient j is mode j
        return -4.0 * h2 * np.sin(np.pi * k / (2 * n)) ** 2
    raise ValueError(bc)


def eigenvalues_2d(n: int, bc: str) -> np.ndarray:
    """2D eigenvalues laid out like the output of `forward`."""
    if bc == "periodic":
        # Full FFT along x (axis -2), real FFT along y (axis -1).
        kx = np.arange(n)
        lam_x = -4.0 * n**2 * np.sin(np.pi * kx / n) ** 2
        lam_y = eigenvalues_1d(n, bc)
        return lam_x[:, None] + lam_y[None, :]
    lam = eigenvalues_1d(n, bc)
    return lam[:, None] + lam[None, :]


def forward(field: np.ndarray, bc: str) -> np.ndarray:
    """Grid values -> Laplacian eigen-coefficients (orthonormal transforms)."""
    if bc == "periodic":
        return sfft.rfftn(field, axes=(-2, -1), norm="ortho")
    if bc == "dirichlet_zero":
        return sfft.dstn(field, type=2, axes=(-2, -1), norm="ortho")
    if bc == "neumann_zero":
        return sfft.dctn(field, type=2, axes=(-2, -1), norm="ortho")
    raise ValueError(bc)


def inverse(coeff: np.ndarray, bc: str, n: int) -> np.ndarray:
    """Laplacian eigen-coefficients -> grid values."""
    if bc == "periodic":
        return sfft.irfftn(coeff, s=(n, n), axes=(-2, -1), norm="ortho")
    if bc == "dirichlet_zero":
        return sfft.idstn(coeff, type=2, axes=(-2, -1), norm="ortho")
    if bc == "neumann_zero":
        return sfft.idctn(coeff, type=2, axes=(-2, -1), norm="ortho")
    raise ValueError(bc)


# ---------------------------------------------------------------------------
# Stencils with ghost values
# ---------------------------------------------------------------------------
def pad_ghost(field: np.ndarray, bc: str) -> np.ndarray:
    """Add one layer of ghost values around both spatial axes."""
    widths = [(0, 0)] * (field.ndim - 2) + [(1, 1), (1, 1)]
    if bc == "periodic":
        return np.pad(field, widths, mode="wrap")
    padded = np.pad(field, widths, mode="edge")    # ghost = neighbour (Neumann)
    if bc == "dirichlet_zero":
        # ghost = -neighbour, so the value midway between them is zero.
        # The corner ghosts are never used by the stencils below.
        padded[..., 0, :] *= -1.0
        padded[..., -1, :] *= -1.0
        padded[..., :, 0] *= -1.0
        padded[..., :, -1] *= -1.0
        return padded
    if bc == "neumann_zero":
        return padded
    raise ValueError(bc)


def laplacian_stencil(field: np.ndarray, bc: str) -> np.ndarray:
    """Five-point Laplacian applied directly (used only to validate the transforms)."""
    n = field.shape[-1]
    g = pad_ghost(field, bc)
    center = g[..., 1:-1, 1:-1]
    return (g[..., 2:, 1:-1] + g[..., :-2, 1:-1] + g[..., 1:-1, 2:] + g[..., 1:-1, :-2] - 4.0 * center) * n**2


def gradient(field: np.ndarray, bc: str) -> tuple[np.ndarray, np.ndarray]:
    """Centered first differences: (d/dx along axis -2, d/dy along axis -1)."""
    n = field.shape[-1]
    g = pad_ghost(field, bc)
    dx = (g[..., 2:, 1:-1] - g[..., :-2, 1:-1]) * (n / 2.0)
    dy = (g[..., 1:-1, 2:] - g[..., 1:-1, :-2]) * (n / 2.0)
    return dx, dy


# ---------------------------------------------------------------------------
# The explicit part G(U, grad U) of each PDE
# ---------------------------------------------------------------------------
def explicit_term(pde: str, U: np.ndarray, bc: str) -> np.ndarray:
    """G for fields U of shape [..., 2, n, n]; returns the same shape."""
    p = PDE_COEFFICIENTS[pde]
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
# ETDRK4 time stepping
# ---------------------------------------------------------------------------
@dataclass
class ETDCoefficients:
    """Per-mode multipliers of one ETDRK4 step of size h (names as in Kassam & Trefethen)."""
    E: np.ndarray      # exp(h L)
    E2: np.ndarray     # exp(h L / 2)
    Q: np.ndarray
    f1: np.ndarray
    f2: np.ndarray
    f3: np.ndarray


def etdrk4_coefficients(L: np.ndarray, h: float, contour_points: int = 64) -> ETDCoefficients:
    """Evaluate the ETDRK4 phi-functions for a real diagonal operator L.

    The averages over points on a unit circle around each h*L avoid the
    cancellation that the direct formulas suffer from when h*L is small.
    """
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
             t_final: float = T_FINAL, save_indices: list[int] | None = None) -> np.ndarray:
    """Integrate a batch of initial conditions.

    U0 has shape [B, 2, n, n] (or [2, n, n] for one sample). Returns float64
    [B, n_saved, 2, n, n] at the times t_k = k * t_final / (n_states - 1) for
    the requested indices k (all states by default). Raises FloatingPointError
    if a saved state is not finite; nothing is ever clipped.
    """
    single = U0.ndim == 3
    U = np.array(U0[None] if single else U0, dtype=np.float64)
    batch, n_fields, n, _ = U.shape
    p = PDE_COEFFICIENTS[pde]

    # Diffusion operator in the eigenbasis: one diagonal per field.
    diffusivity = np.array([p["D_u"], p["D_v"]], dtype=np.float64)
    L = diffusivity[:, None, None] * eigenvalues_2d(n, bc)[None]
    h = t_final / (n_states - 1) / substeps
    c = etdrk4_coefficients(L, h)

    save = list(range(n_states)) if save_indices is None else list(save_indices)
    slot = {k: i for i, k in enumerate(save)}          # saved index -> output position
    out = np.empty((batch, len(save), n_fields, n, n), dtype=np.float64)

    def explicit_hat(v_hat: np.ndarray) -> np.ndarray:
        """G evaluated on the grid, returned in the eigenbasis."""
        return forward(explicit_term(pde, inverse(v_hat, bc, n), bc), bc)

    v = forward(U, bc)                                 # state in the eigenbasis
    if 0 in slot:
        out[:, slot[0]] = U
    for k in range(1, n_states):
        for _ in range(substeps):
            if pde == "diffusion":                     # no explicit part: exact update
                v = c.E * v
                continue
            Nv = explicit_hat(v)
            a = c.E2 * v + c.Q * Nv
            Na = explicit_hat(a)
            b = c.E2 * v + c.Q * Na
            Nb = explicit_hat(b)
            cc = c.E2 * a + c.Q * (2 * Nb - Nv)
            Nc = explicit_hat(cc)
            v = c.E * v + Nv * c.f1 + 2 * (Na + Nb) * c.f2 + Nc * c.f3
        if k in slot:
            state = inverse(v, bc, n)
            if not np.isfinite(state).all():
                raise FloatingPointError(f"non-finite state for {pde}/{bc} at saved index {k}")
            out[:, slot[k]] = state
    return out[0] if single else out


# ---------------------------------------------------------------------------
# Exact solutions of the grid equations for the three linear PDEs
# ---------------------------------------------------------------------------
def dense_laplacian_1d(n: int, bc: str) -> np.ndarray:
    """1D five-point Laplacian as a dense matrix, with the same ghost rules."""
    mat = np.diag(np.full(n, -2.0)) + np.diag(np.ones(n - 1), 1) + np.diag(np.ones(n - 1), -1)
    if bc == "periodic":
        mat[0, -1] = mat[-1, 0] = 1.0
    elif bc == "dirichlet_zero":
        mat[0, 0] = mat[-1, -1] = -3.0             # -2 - 1 from the ghost -u_0
    elif bc == "neumann_zero":
        mat[0, 0] = mat[-1, -1] = -1.0             # -2 + 1 from the ghost u_0
    else:
        raise ValueError(bc)
    return mat * n**2


def dense_gradient_1d(n: int, bc: str) -> np.ndarray:
    """1D centered first difference as a dense matrix, with the same ghost rules."""
    mat = (np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)) * (n / 2.0)
    if bc == "periodic":
        mat[0, -1] = -n / 2.0
        mat[-1, 0] = n / 2.0
    elif bc == "dirichlet_zero":
        mat[0, 0] = n / 2.0                        # -(ghost) with ghost = -u_0
        mat[-1, -1] = -n / 2.0
    elif bc == "neumann_zero":
        mat[0, 0] = -n / 2.0                       # -(ghost) with ghost = u_0
        mat[-1, -1] = n / 2.0
    else:
        raise ValueError(bc)
    return mat


def exact_linear_reference(pde: str, bc: str, U0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Exact solution of the grid equations (no time-stepping error).

    U0 has shape [B, 2, n, n]; returns float64 [B, len(times), 2, n, n].
    Diffusion and advection-diffusion act separately along x and y, so each
    field evolves as  exp(t A_x) U0 exp(t A_y)^T. Linear exchange couples the
    two fields; it is solved mode by mode in the Laplacian eigenbasis with a
    closed-form 2 x 2 matrix exponential.
    """
    U0 = np.asarray(U0, dtype=np.float64)
    batch, n_fields, n, _ = U0.shape
    p = PDE_COEFFICIENTS[pde]
    diffusivity = [p["D_u"], p["D_v"]]
    out = np.empty((batch, len(times), n_fields, n, n))
    A = dense_laplacian_1d(n, bc)

    if pde in ("diffusion", "advection_diffusion"):
        G = dense_gradient_1d(n, bc)
        for c in range(n_fields):
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
            # exp of the 2 x 2 matrix t * [[lam D_u - k, k], [k, lam D_v - k]] for every mode:
            # exp(M) = e^m [cosh(d) I + sinh(d)/d (M - m I)], m = trace/2, d^2 = ((a11-a22)/2)^2 + k^2.
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
