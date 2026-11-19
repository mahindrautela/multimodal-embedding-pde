#!/usr/bin/env python3
"""Create the six-PDE input decks and shared initial-condition parameter bank.

Default: 54 distribution-level JSON decks, three resolved JSON/text examples,
and a 3 x 100 parameter bank. --all-realizations writes all 5,400 resolved
JSON/text pairs. No trajectory simulation or embedding inference is performed.

Every realization uses PCG64(SeedSequence([20261118, realization])). The
resulting (2, 20) uniform array contains one independent row per channel.
The same row is mapped into all three IC families:
  0 amplitude; 1,2,3 first Gaussian x,y,sigma;
  4,5 integer Fourier frequencies; 6,7 phases; 8 first mixture weight;
  9,10,11 second Gaussian x,y,sigma; 12 second weight;
  13,14,15 third Gaussian x,y,sigma; 16 third weight; 17:20 reserved.
All floating generator parameters are rounded to eight decimals BEFORE use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

BASE_SEED = 20261118
N_TRAJECTORIES = 100
GRID_N = 64
N_STATES = 100
T_FINAL = 1.0
FRAME_INDICES = [0, 25, 50, 74, 99]
CHANNELS = ["u", "v"]
FAMILIES = ["single_gaussian", "low_frequency", "patch_mixture"]
BOUNDARIES = ["periodic", "dirichlet_zero", "neumann_zero"]
ETA_LATEX = (
    r"\eta(s)=\begin{cases}\exp\!\left(4-\frac{1}{s(1-s)}\right),"
    r"&0<s<1,\\0,&s\in\{0,1\}.\end{cases}"
)
PDE_SPECS = {
    "diffusion": {
        "parameters": {"D_u": 0.01, "D_v": 0.005},
        "latex": r"\partial_t u=D_u\Delta u,\qquad \partial_t v=D_v\Delta v.",
    },
    "advection_diffusion": {
        "parameters": {"D_u": 0.01, "D_v": 0.005, "c_x": 0.25, "c_y": 0.10},
        "latex": (
            r"\partial_t u=D_u\Delta u-c_x\partial_xu-c_y\partial_yu,\qquad "
            r"\partial_t v=D_v\Delta v-c_x\partial_xv-c_y\partial_yv."
        ),
    },
    "linear_exchange": {
        "parameters": {"D_u": 0.01, "D_v": 0.005, "kappa": 0.5},
        "latex": (
            r"\partial_t u=D_u\Delta u+\kappa(v-u),\qquad "
            r"\partial_t v=D_v\Delta v+\kappa(u-v)."
        ),
    },
    "fisher_kpp": {
        "parameters": {"D_u": 0.01, "D_v": 0.005, "r": 1.0},
        "latex": (
            r"\partial_t u=D_u\Delta u+r\,u(1-u),\qquad "
            r"\partial_t v=D_v\Delta v+r\,v(1-v)."
        ),
    },
    "shifted_allen_cahn": {
        "parameters": {"D_u": 0.01, "D_v": 0.005, "a": 4.0},
        "latex": (
            r"\partial_t u=D_u\Delta u+a\,u(1-u)(u-\tfrac12),\qquad "
            r"\partial_t v=D_v\Delta v+a\,v(1-v)(v-\tfrac12)."
        ),
    },
    "gray_scott": {
        "parameters": {"D_u": 0.01, "D_v": 0.005, "F": 0.04, "k": 0.06},
        "latex": (
            r"\partial_t u=D_u\Delta u-u v^2+F(1-u),\qquad "
            r"\partial_t v=D_v\Delta v+u v^2-(F+k)v."
        ),
    },
}
BC_LATEX = {
    "periodic": (
        r"U(t,0,y)=U(t,1,y),\quad \partial_xU(t,0,y)=\partial_xU(t,1,y),\\"
        r"U(t,x,0)=U(t,x,1),\quad \partial_yU(t,x,0)=\partial_yU(t,x,1)."
    ),
    "dirichlet_zero": r"U(t,x,y)=0\quad\text{for }(x,y)\in\partial\Omega.",
    "neumann_zero": r"\partial_{\mathbf n}U(t,x,y)=0\quad\text{for }(x,y)\in\partial\Omega.",
}
DOMAIN_LATEX = (
    r"\Omega=[0,1]^2,\qquad U=(u,v)^\top,\qquad "
    r"\Delta=\partial_{xx}+\partial_{yy}.\quad "
    r"x_p=(p+\tfrac12)/64,\ y_q=(q+\tfrac12)/64,\ p,q=0,\ldots,63."
)
TIME_LATEX = (
    r"0\leq t\leq1,\qquad t_n=n/99,\quad n=0,\ldots,99.\quad "
    r"\text{Observed frames: }n\in\{0,25,50,74,99\}."
)


def rounded(value: float) -> float:
    return round(float(value), 8)


def decimal(value: float) -> str:
    """The text deck records every resolved floating parameter explicitly."""
    return f"{value:.8f}"


def draw_parameters(realization: int) -> dict[str, list[dict[str, Any]]]:
    """Return all three families from one fixed latent draw."""
    if not isinstance(realization, (int, np.integer)) or realization < 0:
        raise ValueError("realization must be a nonnegative integer; 0..99 are test draws, >=100 are pilot draws")
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([BASE_SEED, realization])))
    latent = rng.uniform(size=(2, 20))
    result: dict[str, list[dict[str, Any]]] = {name: [] for name in FAMILIES}
    for row in latent:
        amplitude = rounded(0.2 + 0.6 * row[0])

        def gaussian(offset: int) -> dict[str, float]:
            return {
                "center_x": rounded(0.25 + 0.5 * row[offset]),
                "center_y": rounded(0.25 + 0.5 * row[offset + 1]),
                "sigma": rounded(0.08 + 0.10 * row[offset + 2]),
            }

        result["single_gaussian"].append({"amplitude": amplitude, **gaussian(1)})
        result["low_frequency"].append({
            "amplitude": amplitude,
            "kx": int(1 + np.floor(3 * row[4])),
            "ky": int(1 + np.floor(3 * row[5])),
            "phase_x": rounded(2 * np.pi * row[6]),
            "phase_y": rounded(2 * np.pi * row[7]),
        })
        components = []
        for offset, weight_index in [(1, 8), (9, 12), (13, 16)]:
            components.append({"weight": rounded(0.2 + 0.8 * row[weight_index]), **gaussian(offset)})
        result["patch_mixture"].append({"amplitude": amplitude, "components": components})
    return result


def eta(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    result = np.zeros_like(s)
    inside = (s > 0) & (s < 1)
    result[inside] = np.exp(4 - 1 / (s[inside] * (1 - s[inside])))
    return result


def _evaluate(family: str, parameters: list[dict[str, Any]], x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if family not in FAMILIES or len(parameters) != 2:
        raise ValueError("Provide a supported family and two channel parameter dictionaries")
    window = eta(x) * eta(y)
    channels = []
    for p in parameters:
        def gaussian(g: dict[str, Any]) -> np.ndarray:
            radius_sq = (x - g["center_x"]) ** 2 + (y - g["center_y"]) ** 2
            return np.exp(-radius_sq / (2 * g["sigma"] ** 2))
        if family == "single_gaussian":
            profile = gaussian(p)
        elif family == "low_frequency":
            profile = (1 + 0.5 * np.cos(2 * np.pi * p["kx"] * x + p["phase_x"])
                       * np.cos(2 * np.pi * p["ky"] * y + p["phase_y"])) / 1.5
        else:
            total_weight = sum(g["weight"] for g in p["components"])
            profile = sum(g["weight"] * gaussian(g) for g in p["components"]) / total_weight
        channels.append(p["amplitude"] * window * profile)
    return np.stack(channels)


def initial_array(family: str, parameters: list[dict[str, Any]], n: int = GRID_N) -> np.ndarray:
    """Evaluate a resolved IC on the common cell-centered grid; returns float64 [2,n,n]."""
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    coordinates = (np.arange(n, dtype=np.float64) + 0.5) / n
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    return _evaluate(family, parameters, x, y)


def gaussian_latex(g: dict[str, Any]) -> str:
    return (
        r"\exp\!\left[-\frac{(x-" + decimal(g["center_x"]) + r")^2+(y-"
        + decimal(g["center_y"]) + r")^2}{2(" + decimal(g["sigma"]) + r")^2}\right]"
    )


def resolved_ic_latex(family: str, parameters: list[dict[str, Any]]) -> str:
    expressions = [ETA_LATEX]
    for channel, p in zip(CHANNELS, parameters):
        amplitude = decimal(p["amplitude"])
        if family == "single_gaussian":
            profile = gaussian_latex(p)
        elif family == "low_frequency":
            profile = (
                r"\frac{1+0.5\cos(2\pi(" + str(p["kx"]) + r")x+" + decimal(p["phase_x"])
                + r")\cos(2\pi(" + str(p["ky"]) + r")y+" + decimal(p["phase_y"]) + r")}{1.5}"
            )
        else:
            terms = [decimal(g["weight"]) + r"\," + gaussian_latex(g) for g in p["components"]]
            denominator = "+".join(decimal(g["weight"]) for g in p["components"])
            profile = r"\frac{" + "+".join(terms) + "}{" + denominator + "}"
        expressions.append(channel + r"(0,x,y)=" + amplitude + r"\,\eta(x)\eta(y)\," + profile + ".")
    return "\n".join(expressions)


def distribution_ic_latex(family: str) -> str:
    common = (
        ETA_LATEX + "\n" + r"q(0,x,y)=\alpha_q\eta(x)\eta(y)g_q(x,y),\quad q\in\{u,v\},\quad "
        r"\alpha_q\sim\mathcal U[0.2,0.8]."
    )
    if family == "single_gaussian":
        profile = (
            r"g_q(x,y)=\exp\!\left[-\frac{(x-\mu_{x,q})^2+(y-\mu_{y,q})^2}{2\sigma_q^2}\right],\quad "
            r"\mu_{x,q},\mu_{y,q}\sim\mathcal U[0.25,0.75],\quad\sigma_q\sim\mathcal U[0.08,0.18]."
        )
    elif family == "low_frequency":
        profile = (
            r"g_q(x,y)=\frac{1+0.5\cos(2\pi k_{x,q}x+\phi_{x,q})\cos(2\pi k_{y,q}y+\phi_{y,q})}{1.5},\quad "
            r"k_{x,q},k_{y,q}\sim\operatorname{Unif}\{1,2,3\},\quad "
            r"\phi_{x,q},\phi_{y,q}\sim\mathcal U[0,2\pi]."
        )
    else:
        profile = (
            r"g_q(x,y)=\frac{\sum_{\ell=1}^{3}w_{q\ell}\exp\!\left[-\frac{(x-\mu_{x,q\ell})^2+"
            r"(y-\mu_{y,q\ell})^2}{2\sigma_{q\ell}^2}\right]}{\sum_{\ell=1}^{3}w_{q\ell}},\quad "
            r"w_{q\ell}\sim\mathcal U[0.2,1],\quad\mu_{x,q\ell},\mu_{y,q\ell}\sim\mathcal U[0.25,0.75],\quad "
            r"\sigma_{q\ell}\sim\mathcal U[0.08,0.18]."
        )
    return common + "\n" + profile + "\n" + r"\text{Parameters are independently sampled between channels and rounded to eight decimal places.}"


def coefficient_latex(parameters: dict[str, float]) -> str:
    return r",\quad ".join((r"\kappa" if name == "kappa" else name) + "=" + decimal(value)
                            for name, value in parameters.items()) + "."


def input_text(latex: dict[str, str]) -> str:
    headings = [("Equation", "pde"), ("Parameters", "parameters"), ("Initial condition", "ic"),
                ("Boundary", "bc"), ("Domain", "domain"), ("Time", "time")]
    return "\n\n".join(heading + ":\n" + latex[key] for heading, key in headings) + "\n"


def config_id(pde: str, family: str, bc: str) -> str:
    return f"{pde}__{family}__{bc}"


def make_deck(pde: str, family: str, bc: str, parameters: list[dict[str, Any]] | None = None,
              realization: int | None = None) -> dict[str, Any]:
    resolved = parameters is not None
    if resolved != (realization is not None):
        raise ValueError("Resolved parameters and realization index must be supplied together")
    pde_spec = PDE_SPECS[pde]
    latex = {"pde": pde_spec["latex"], "parameters": coefficient_latex(pde_spec["parameters"]),
             "ic": resolved_ic_latex(family, parameters) if resolved else distribution_ic_latex(family),
             "bc": BC_LATEX[bc], "domain": DOMAIN_LATEX, "time": TIME_LATEX}
    deck: dict[str, Any] = {
        "schema_version": "1.0",
        "configuration_id": config_id(pde, family, bc),
        "pde_class": pde,
        "ic_family": family,
        "bc_type": bc,
        "scope": "realized_initial_condition" if resolved else "initial_condition_distribution",
        "channel_order": CHANNELS,
        "coefficients": pde_spec["parameters"],
        "latex": latex,
        "domain": {"x": [0.0, 1.0], "y": [0.0, 1.0], "units": "dimensionless"},
        "grid": {"type": "cell_centered", "shape": [GRID_N, GRID_N], "spacing": 1 / GRID_N,
                 "axis_order": ["channel", "x", "y"], "coordinate_formula": "(index + 0.5) / 64"},
        "time": {"final_time": T_FINAL, "saved_states": N_STATES, "output_intervals": N_STATES - 1,
                 "saved_times": [float(n / 99) for n in range(N_STATES)],
                 "observation_indices": FRAME_INDICES,
                 "observation_times": [float(n / 99) for n in FRAME_INDICES],
                 "internal_solver_steps": "not specified; output intervals are not solver steps"},
        "initial_condition": {"parameter_precision_decimal_places": 8,
                              "parameter_bank": "../ic_parameter_bank.json",
                              "generator_parameters": parameters if resolved else None,
                              "distribution_latex": distribution_ic_latex(family)},
        "provenance": {"base_seed": BASE_SEED, "realization_index": realization,
                       "rng": "PCG64(SeedSequence([20261118, realization_index]))",
                       "shared_initial_arrays": "identical within each IC family across all PDEs and BCs",
                       "cross_family_coupling": "same explicit (2,20) latent draw; documented parameter mapping"},
        "solver_status": "not_run",
        "embedding_status": "not_run",
        "model_input_text": input_text(latex),
    }
    return deck


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def build(output: Path, all_realizations: bool) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    bank = {family: [] for family in FAMILIES}
    bounds = {family: {"minimum": float("inf"), "maximum": -float("inf"), "example_sha256_float64": None}
              for family in FAMILIES}
    boundary_zero = True
    replay_equal = True
    for r in range(N_TRAJECTORIES):
        draws = draw_parameters(r)
        replay_equal = replay_equal and draws == draw_parameters(r)
        for family in FAMILIES:
            parameters = draws[family]
            bank[family].append({"realization_index": r, "seed_entropy": [BASE_SEED, r],
                                 "channel_order": CHANNELS, "parameters": parameters})
            array = initial_array(family, parameters)
            if array.shape != (2, 64, 64) or not np.isfinite(array).all():
                raise AssertionError("Invalid initial array")
            if array.min() < 0 or array.max() > 0.8 + 1e-12:
                raise AssertionError("Initial-condition bound failed")
            replay_equal = replay_equal and np.array_equal(array, initial_array(family, draw_parameters(r)[family]))
            bounds[family]["minimum"] = min(bounds[family]["minimum"], float(array.min()))
            bounds[family]["maximum"] = max(bounds[family]["maximum"], float(array.max()))
            edge = np.linspace(0, 1, 17)
            boundary_zero = boundary_zero and bool(np.all(_evaluate(family, parameters, np.zeros_like(edge), edge) == 0))
            boundary_zero = boundary_zero and bool(np.all(_evaluate(family, parameters, np.ones_like(edge), edge) == 0))
            boundary_zero = boundary_zero and bool(np.all(_evaluate(family, parameters, edge, np.zeros_like(edge)) == 0))
            boundary_zero = boundary_zero and bool(np.all(_evaluate(family, parameters, edge, np.ones_like(edge)) == 0))
            if r == 0:
                bounds[family]["example_sha256_float64"] = hashlib.sha256(array.astype("<f8").tobytes()).hexdigest()
    if not boundary_zero or not replay_equal:
        raise AssertionError("Boundary or reproducibility check failed")

    bank_document = {"schema_version": "1.0", "base_seed": BASE_SEED,
                     "rng": "PCG64", "seed_sequence": "[base_seed, realization_index]",
                     "latent_shape": [2, 20], "floating_parameter_decimal_places": 8,
                     "latent_mapping": {
                         "0": "shared amplitude:0.2+0.6*z", "1,2,3": "first Gaussian center_x,center_y,sigma",
                         "4,5": "kx,ky:1+floor(3*z)", "6,7": "phase_x,phase_y:2*pi*z",
                         "8": "first mixture weight:0.2+0.8*z", "9,10,11": "second Gaussian center_x,center_y,sigma",
                         "12": "second mixture weight", "13,14,15": "third Gaussian center_x,center_y,sigma",
                         "16": "third mixture weight", "17,18,19": "reserved"},
                     "families": bank}
    write_json(output / "ic_parameter_bank.json", bank_document)

    configuration_files = []
    realization_files = []
    for pde in PDE_SPECS:
        for family in FAMILIES:
            for bc in BOUNDARIES:
                name = config_id(pde, family, bc)
                config_path = output / "configurations" / f"{name}.json"
                write_json(config_path, make_deck(pde, family, bc))
                configuration_files.append(str(config_path.relative_to(output)))
                indices = range(N_TRAJECTORIES) if all_realizations else ([0] if pde == "diffusion" and bc == "periodic" else [])
                for r in indices:
                    resolved = make_deck(pde, family, bc, bank[family][r]["parameters"], r)
                    base = output / "realizations" / f"{name}__r{r:03d}"
                    write_json(base.with_suffix(".json"), resolved)
                    base.with_suffix(".txt").write_text(resolved["model_input_text"], encoding="utf-8")
                    realization_files.append(str(base.with_suffix(".json").relative_to(output)))

    shape = [54, 100, 100, 2, 64, 64]
    total_bytes = int(np.prod(shape)) * 4
    specification = {
        "schema_version": "1.0", "status": "input decks only; simulations and embedding inference not run",
        "pde_classes": list(PDE_SPECS), "ic_families": FAMILIES, "boundary_conditions": BOUNDARIES,
        "number_of_configurations": 54, "trajectories_per_configuration": 100,
        "planned_total_trajectories": 5400, "planned_array_shape": shape,
        "planned_array_axes": ["configuration", "trajectory", "saved_state", "channel", "x", "y"],
        "planned_float32_bytes": total_bytes, "planned_float32_GiB": total_bytes / 2**30,
        "domain": [[0.0, 1.0], [0.0, 1.0]], "grid_type": "cell_centered", "channel_order": CHANNELS,
        "final_time": T_FINAL, "saved_times": [float(n / 99) for n in range(N_STATES)],
        "observation_indices": FRAME_INDICES,
        "test_realization_indices": {"first": 0, "last": 99},
        "pilot_realization_indices": {"first": 100, "last": None},
        "models": [
            {"checkpoint": "google/embeddinggemma-2", "embedding_dimension": 768, "revision": None},
            {"checkpoint": "Qwen/Qwen3-VL-Embedding-2B", "embedding_dimension": 2048, "revision": None},
            {"checkpoint": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct", "embedding_dimension": 1536, "revision": None},
        ],
        "input_modes": ["resolved_text", "five_frame_contact_sheet", "native_text_plus_image"],
        "reference": {"primary": "uniform RMS over matched trajectories, five observed frames, two fields and all pixels",
                      "secondary": "uniform RMS over all 100 saved frames",
                      "physical_cosine": "optional separate amplitude-normalized analysis"},
        "embedding_score": "mean normalized dot product of matched realization pairs",
        "primary_analysis": "within each IC family: 18 configurations; Spearman versus negative RMS and ordered-triplet accuracy",
        "uncertainty": "joint matched-realization bootstrap; conditional on the selected PDE/BC configurations",
        "ic_parameter_bank": "ic_parameter_bank.json",
        "input_deck_generation": "all_realizations" if all_realizations else "three_resolved_examples",
        "configuration_files": configuration_files, "resolved_json_files_written_this_run": realization_files,
        "model_input_policy": "Use model_input_text or the resolved .txt, never provenance labels or configuration IDs.",
        "initial_condition_policy": "Round generator parameters to eight decimals before evaluating; common arrays across PDEs/BCs.",
        "neumann_note": "Zero normal gradient; not automatically zero total advective flux.",
    }
    write_json(output / "spec.json", specification)
    resolved_count = 5400 if all_realizations else 3
    assert len(configuration_files) == 54 and len(realization_files) == resolved_count
    for name in realization_files:
        deck = json.loads((output / name).read_text(encoding="utf-8"))
        text = deck["model_input_text"]
        if any(label in text for label in list(PDE_SPECS) + FAMILIES + BOUNDARIES):
            raise AssertionError("Provenance label leaked into model input")
        if deck["configuration_id"] in text:
            raise AssertionError("Configuration ID leaked into model input")
        if deck["initial_condition"]["generator_parameters"] is None:
            raise AssertionError("Resolved example has no numeric IC parameters")

    checks = {
        "status": "passed", "numpy_version": np.__version__, "base_seed": BASE_SEED,
        "configuration_json_decks": len(configuration_files),
        "resolved_json_decks_written_this_run": len(realization_files),
        "resolved_text_decks_written_this_run": len(realization_files),
        "ic_parameter_records": {family: len(bank[family]) for family in FAMILIES},
        "initial_arrays_evaluated_for_checks": 300,
        "initial_array_shape": [2, 64, 64], "initial_array_dtype": "float64",
        "all_arrays_finite_and_between_0_and_0_8": True,
        "exact_zero_sampled_boundary_values": boundary_zero,
        "parameter_and_array_replay_identical": replay_equal,
        "resolved_model_inputs_exclude_provenance_labels": True,
        "ic_bounds": bounds,
        "simulations_run": False, "embedding_inference_run": False,
        "note": "Boundary derivatives vanish by construction; sampled boundary values are a numerical check. Hashes describe this NumPy/platform execution. Existing unrelated files are not deleted.",
    }
    write_json(output / "deck_checks.json", checks)
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "benchmark")
    parser.add_argument("--all-realizations", action="store_true", help="Write 5,400 resolved JSON/text pairs instead of three examples")
    args = parser.parse_args()
    print(json.dumps(build(args.output.resolve(), args.all_realizations), indent=2))


if __name__ == "__main__":
    main()
