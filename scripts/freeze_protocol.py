#!/usr/bin/env python3
"""Write benchmark/frozen_protocol.json after the pilot.

The protocol fixes everything the test analysis depends on before any test
sample (r = 0..99) is simulated or encoded: coefficients, solver settings,
image rendering, model versions and input formats, the encoded subset, the
controls, the error-margin rule and the analysis. It also stores the SHA-256
of every script.

The study's protocol already exists, so this script refuses to overwrite it
unless --force is given. Use --output to write a new protocol elsewhere.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import controls as C
import render as R
import simulator as S
from common import BENCHMARK, CHECKPOINTS, FRAMES, PROTOCOL_FILE, REPO


def build_protocol(pilot: dict) -> dict:
    import embed_models as E          # needs torch, so it is imported only when a protocol is built
    scripts = sorted((REPO / "scripts").glob("*.py"))
    return {
        "frozen_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "status": "frozen before any test realization was simulated or encoded",
        "pilot_report": "benchmark/pilot_report.json",
        "coefficients": S.PDE_COEFFICIENTS, "coefficients_changed_by_pilot": False,
        "solver": {"name": S.SOLVER_NAME, "version": S.SOLVER_VERSION, "grid": [64, 64],
                   "scheme": "ETDRK4 (Cox-Matthews; Kassam-Trefethen contour, 64 points); diffusion exact in "
                             "the BC eigenbasis (rFFT / DST-II / DCT-II); advection, exchange and reactions explicit",
                   "substeps_per_output_interval": pilot["substeps"], "internal_step": 1 / 99 / pilot["substeps"],
                   "float": "float64 internally, float32 stored"},
        "observation_indices": FRAMES,
        "test_realizations": [0, 99], "pilot_realizations": pilot["pilot_realizations"],
        "contact_sheet": {"layout": "2 channel rows (u, v) x 5 time columns", "panel_pixels": 64 * R.PANEL_SCALE,
                          "upsampling": "nearest-neighbour x2", "size_wh": list(R.sheet_size()),
                          "colormap": R.COLORMAP, "channel_ranges": pilot["P6_contact_sheet_range"],
                          "labels": "t=0.00, t=0.25, t=0.51, t=0.75, t=1.00 and u, v only",
                          "orientation": "x to the right, y upward", "clipping": "display only; arrays unmodified"},
        "models": {k: {"checkpoint": repo, "revision": revision} for k, (repo, revision) in CHECKPOINTS.items()},
        "model_inputs": {
            "embeddinggemma2": {"text_prefix": E.GEMMA_PREFIX, "image_prefix": None,
                                "combined": E.GEMMA_PREFIX + "{deck}\\n<|image|>", "dtype": "bfloat16",
                                "pooling": "checkpoint (mean) + normalize", "image_tokens": E.GEMMA_IMAGE_TOKENS},
            "qwen3vl2b": {"instruction": E.QWEN3_INSTRUCTION, "order": "image then text", "dtype": "bfloat16",
                          "min_pixels": E.Qwen3VLBackend.min_pixels, "max_pixels": E.Qwen3VLBackend.max_pixels,
                          "pooling": "last token"},
            "gme2b": {"instruction": E.GME_INSTRUCTION, "order": "image then text", "dtype": "bfloat16",
                      "min_pixels": E.GMEBackend.min_pixels, "max_pixels": E.GMEBackend.max_pixels,
                      "max_length": E.GMEBackend.max_length, "pooling": "last token",
                      "positions": "sequential, as in the reference gme_inference.py"},
            "normalization": "every embedding re-normalized in float32; full native dimension",
            "precision": "bfloat16 for all models (CPU with native AMX bf16); diagnostic: EmbeddingGemma 2 float32 "
                         "re-encoding of realization 0 for all 54 configurations and three modes, if time permits",
            "batch_size": 1},
        "embedding_subset": {
            "reason": "CPU-only environment (2 cores, no GPU); the full 48,600-input run is not feasible here",
            "main_realizations": [0, 9], "control_realizations": [0, 2],
            "rule": "first realizations in index order; fixed before any embedding of test inputs"},
        "controls": {"time_order": "panels reversed (index order 4,3,2,1,0) with header labels unchanged",
                     "first_frame": "single t=0 column sheet; one per IC family and realization, encoded twice",
                     "notation": "diffusion as second partials, terms reordered, coordinates renamed x->xi, y->zeta",
                     "mismatch": "sheet of configuration (p,f,b) with the deck of (p+1 mod 6, f, b), same realization",
                     "heat_calibration": {"anchor": C.HEAT_ANCHOR, "candidates": C.HEAT_CANDIDATES}},
        "baselines": {"text": "character 3-5-gram TF-IDF (sublinear tf, fitted on the analysed decks), cosine",
                      "image_pixels": "rendered sheet RGB pixels, flattened, L2-normalized",
                      "image_fourier": "|2D DFT| of each 64x64 field at the five frames, frequencies |k|<=8 per axis, "
                                       "L2-normalized",
                      "numeric_cosine": "physical cosine of trajectory profiles (amplitude removed)"},
        "analysis": {
            "reference": "five-frame RMS over matched realizations (primary); 100-frame RMS (secondary)",
            "primary": "within each IC family, 153 off-diagonal pairs of the 18x18 matrices: Spearman(S_emb, -D), "
                       "and ordered-triplet accuracy (reference ties excluded, embedding ties = 1/2)",
            "matched_subset": "embedding and reference matrices are computed on the same realization subset; "
                              "agreement with the 100-realization reference is reported separately",
            "breakdown": "triplets whose candidates share / do not share the anchor BC",
            "separation_rule": "triplet (i;j,k) resolved if |D_ik - D_ij| > 2 eps_i + eps_j + eps_k; eps from exact "
                               "semidiscrete references (linear classes; certified up to roundoff) or "
                               "||fl32(X_m)-X_2m|| + 2||X_m-X_2m|| (nonlinear; convergence-checked)",
            "bootstrap": {"replicates": 2000, "seed": 20261119, "unit": "realization index, resampled jointly"},
            "full_matrix": "54x54 matrix as secondary analysis"},
        "script_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in scripts},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pilot", type=Path, default=BENCHMARK / "pilot_report.json")
    parser.add_argument("--output", type=Path, default=PROTOCOL_FILE)
    parser.add_argument("--force", action="store_true", help="overwrite an existing protocol")
    args = parser.parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} already exists; it records the study as run. "
                         "Use --output for a new file or --force to replace it.")
    pilot = json.loads(args.pilot.read_text())
    if not pilot["passed"]:
        raise SystemExit("the pilot failed; revise the coefficients before freezing")
    protocol = build_protocol(pilot)
    args.output.write_text(json.dumps(protocol, indent=2) + "\n")
    print(json.dumps({k: protocol[k] for k in ("frozen_at", "solver", "contact_sheet")}, indent=2))


if __name__ == "__main__":
    main()
