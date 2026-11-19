#!/usr/bin/env python3
"""Compare our solver with Exponax, an independent spectral solver (periodic BC only).

Writes benchmark/exponax_crosscheck.json. Needs jax and exponax, e.g. in a
separate environment: pip install exponax.

Both solvers start from the same setup-sample initial fields (r = 100, 101)
on n x n grids, n = 64, 128, 256, for the five PDEs that Exponax provides
directly (Allen-Cahn through w = 2u - 1; linear exchange is not available).
Exponax is spectrally accurate in space for these smooth fields, so the
difference measures our finite-difference error. It should shrink about
4 times per grid refinement (second order). On a periodic grid, the half-cell
shift between the two grid conventions does not matter.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)        # must be set before Exponax is imported
import exponax as ex  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import simulator as S  # noqa: E402
from common import BENCHMARK, FAMILIES, FRAMES, initial_fields  # noqa: E402

STEPS = 8                                        # Exponax steps per output interval
SAMPLES = [100, 101]
PDES = ["diffusion", "advection_diffusion", "fisher_kpp", "shifted_allen_cahn", "gray_scott"]


def roll_out(stepper, u0: np.ndarray) -> np.ndarray:
    """Run an Exponax stepper to t = 1 and return the five observed frames."""
    u = jnp.asarray(u0)
    frames = [np.asarray(u)]
    step = jax.jit(lambda v: stepper(v))          # jit a plain function, not the module itself
    for k in range(1, 100):
        for _ in range(STEPS):
            u = step(u)
        if k in FRAMES:
            frames.append(np.asarray(u))
    return np.stack(frames)


def exponax_run(pde: str, U0: np.ndarray, n: int) -> np.ndarray:
    """Exponax solution [5, 2, n, n] for one initial condition [2, n, n]."""
    p = S.PDE_COEFFICIENTS[pde]
    grid = dict(num_spatial_dims=2, domain_extent=1.0, num_points=n, dt=1.0 / 99 / STEPS)
    diffusivity = [p["D_u"], p["D_v"]]
    if pde == "gray_scott":                       # the only coupled system
        stepper = ex.stepper.reaction.GrayScott(**grid, diffusivity_1=diffusivity[0], diffusivity_2=diffusivity[1],
                                                feed_rate=p["F"], kill_rate=p["k"], order=4,
                                                dealiasing_fraction=1.0)
        return roll_out(stepper, U0)
    fields = []
    for c in range(2):                            # the other PDEs act on each field separately
        u0 = U0[c:c + 1]
        if pde == "diffusion":
            fields.append(roll_out(ex.stepper.Diffusion(**grid, diffusivity=diffusivity[c]), u0))
        elif pde == "advection_diffusion":
            stepper = ex.stepper.AdvectionDiffusion(**grid, velocity=jnp.array([p["c_x"], p["c_y"]]),
                                                    diffusivity=diffusivity[c])
            fields.append(roll_out(stepper, u0))
        elif pde == "fisher_kpp":
            stepper = ex.stepper.reaction.FisherKPP(**grid, diffusivity=diffusivity[c], reactivity=p["r"],
                                                    order=4, dealiasing_fraction=1.0)
            fields.append(roll_out(stepper, u0))
        elif pde == "shifted_allen_cahn":
            # With w = 2u - 1, a u(1-u)(u-1/2) becomes (a/4)(w - w^3), Exponax's Allen-Cahn form.
            stepper = ex.stepper.reaction.AllenCahn(**grid, diffusivity=diffusivity[c],
                                                    first_order_coefficient=p["a"] / 4,
                                                    third_order_coefficient=-p["a"] / 4, order=4,
                                                    dealiasing_fraction=1.0)
            fields.append((roll_out(stepper, 2 * u0 - 1) + 1) / 2)
    return np.concatenate(fields, axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=BENCHMARK / "exponax_crosscheck.json")
    args = parser.parse_args()
    report = {"tool": "exponax", "version": importlib.metadata.version("exponax"), "jax": jax.__version__,
              "pilot_realizations": SAMPLES, "bc": "periodic", "steps_per_output": STEPS, "rows": {}}
    for pde in PDES:
        for family in FAMILIES:
            errors = []
            for n in (64, 128, 256):
                squared_diff, signal = [], []
                for U0 in initial_fields(family, SAMPLES, n):
                    ours = S.simulate(pde, "periodic", U0, substeps=4, save_indices=FRAMES)
                    theirs = exponax_run(pde, U0, n)
                    squared_diff.append(np.mean((ours - theirs) ** 2))
                    signal.append(np.mean(theirs**2))
                errors.append(float(np.sqrt(np.mean(squared_diff))))
            report["rows"][f"{pde}/{family}"] = {"rms_fd_minus_spectral_n64_128_256": errors,
                                                "ratio_64_128": errors[0] / errors[1],
                                                "ratio_128_256": errors[1] / errors[2],
                                                "rms_signal": float(np.sqrt(np.mean(signal)))}
            print(pde, family, errors, flush=True)
    rows = report["rows"].values()
    report["max_rms_difference_n64"] = max(r["rms_fd_minus_spectral_n64_128_256"][0] for r in rows)
    report["median_ratio_64_128"] = float(np.median([r["ratio_64_128"] for r in rows]))
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
