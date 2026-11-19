#!/usr/bin/env python3
"""Independent periodic cross-check against Exponax (Fourier pseudospectral ETDRK).

Run in an environment with jax and exponax installed. For periodic BCs and
the five classes Exponax provides directly (diffusion, advection-diffusion,
Fisher-KPP, Allen-Cahn via w = 2u - 1, Gray-Scott), both solvers start from
the same pilot initial arrays sampled at the cell centers of an n x n grid
(n = 64, 128, 256). Because the problem is periodic and translation
invariant, the half-cell shift between the two grid conventions is only a
relabelling. Exponax is spectrally accurate in space for these smooth fields,
so the difference measures the five-point finite-difference spatial error;
it should decrease by about 4x per refinement. Pilot realizations only.
Writes benchmark/exponax_crosscheck.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import exponax as ex  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_decks as M  # noqa: E402
import simulator as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FRAMES = [0, 25, 50, 74, 99]
SUB = 8  # Exponax steps per output interval


def exponax_run(pde: str, U0: np.ndarray, n: int) -> np.ndarray:
    p = S.PDE_COEFFICIENTS[pde]
    dt = 1.0 / 99 / SUB
    common = dict(num_spatial_dims=2, domain_extent=1.0, num_points=n, dt=dt)
    D = [p["D_u"], p["D_v"]]

    def roll(stepper, u0):
        u = jnp.asarray(u0)
        out = [np.asarray(u)]
        step = jax.jit(lambda v: stepper(v))
        for k in range(1, 100):
            for _ in range(SUB):
                u = step(u)
            if k in FRAMES:
                out.append(np.asarray(u))
        return np.stack(out)

    if pde == "gray_scott":
        st = ex.stepper.reaction.GrayScott(**common, diffusivity_1=D[0], diffusivity_2=D[1],
                                           feed_rate=p["F"], kill_rate=p["k"], order=4, dealiasing_fraction=1.0)
        return roll(st, U0)
    chans = []
    for c in range(2):
        u0 = U0[c:c + 1]
        if pde == "diffusion":
            st = ex.stepper.Diffusion(**common, diffusivity=D[c])
            chans.append(roll(st, u0))
        elif pde == "advection_diffusion":
            st = ex.stepper.AdvectionDiffusion(**common, velocity=jnp.array([p["c_x"], p["c_y"]]), diffusivity=D[c])
            chans.append(roll(st, u0))
        elif pde == "fisher_kpp":
            st = ex.stepper.reaction.FisherKPP(**common, diffusivity=D[c], reactivity=p["r"], order=4,
                                               dealiasing_fraction=1.0)
            chans.append(roll(st, u0))
        elif pde == "shifted_allen_cahn":
            st = ex.stepper.reaction.AllenCahn(**common, diffusivity=D[c], first_order_coefficient=p["a"] / 4,
                                               third_order_coefficient=-p["a"] / 4, order=4, dealiasing_fraction=1.0)
            chans.append((roll(st, 2 * u0 - 1) + 1) / 2)
    return np.concatenate(chans, axis=1)


def main() -> None:
    report = {"tool": "exponax", "version": getattr(ex, "__version__", "0.2.1"), "jax": jax.__version__,
              "pilot_realizations": [100, 101], "bc": "periodic", "steps_per_output": SUB, "rows": {}}
    for pde in ["diffusion", "advection_diffusion", "fisher_kpp", "shifted_allen_cahn", "gray_scott"]:
        for fam in M.FAMILIES:
            errs = []
            for n in (64, 128, 256):
                diffs, signal = [], []
                for r in (100, 101):
                    U0 = M.initial_array(fam, M.draw_parameters(r)[fam], n=n)
                    fd = S.simulate(pde, "periodic", U0, substeps=4, save_indices=FRAMES)
                    sp = exponax_run(pde, U0, n)
                    diffs.append(np.mean((fd - sp) ** 2))
                    signal.append(np.mean(sp**2))
                errs.append(float(np.sqrt(np.mean(diffs))))
            report["rows"][f"{pde}/{fam}"] = {"rms_fd_minus_spectral_n64_128_256": errs,
                                             "ratio_64_128": errs[0] / errs[1], "ratio_128_256": errs[1] / errs[2],
                                             "rms_signal": float(np.sqrt(np.mean(signal)))}
            print(pde, fam, errs, flush=True)
    rows = report["rows"].values()
    report["max_rms_difference_n64"] = max(r["rms_fd_minus_spectral_n64_128_256"][0] for r in rows)
    report["median_ratio_64_128"] = float(np.median([r["ratio_64_128"] for r in rows]))
    (ROOT / "benchmark" / "exponax_crosscheck.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
