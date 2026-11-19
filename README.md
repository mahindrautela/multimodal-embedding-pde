# Do Multimodal Embeddings Capture PDE Behavior?

## Main findings

On 54 two-dimensional PDE problems (six equations × three initial-condition
families × three boundary conditions, 100 simulated samples each), we compare
how three multimodal embedding models rank pairs of problems with how their
simulated solutions actually differ. On 10 samples per problem:

- **Images** of five solution snapshots give Spearman correlations of
  0.76–0.83 with the reference, below a plain pixel comparison (0.93).
- **Text descriptions** give 0.21–0.46, close to character n-grams (0.35).
  For two of the models, rewriting a description in equivalent notation moves
  its embedding further than switching to any other problem.
- **Text and image together** give 0.23–0.50, close to text alone.

The models recognize solutions that look alike much better than equations
that behave alike. All numbers come from `benchmark/results/analysis.json`.

## What is here

This repository holds the code and records of the study behind the ICML 2027
working manuscript. The manuscript itself is kept on Overleaf. Every number in
the paper is generated from the files in `benchmark/` by the scripts in
`scripts/`: `make_tables.py` writes the tables and number macros to
`sections/generated/`, and `make_figures.py` writes the figures to `figures/`.
Copy those two folders into the Overleaf project after rerunning the analysis.

**Done at full scale**

- One solver (`scripts/simulator.py`) for all six PDEs and three BCs, checked
  against exact grid solutions, exact continuum solutions, smaller time steps,
  finer grids, and an independent spectral solver (Exponax, periodic BC).
- A pilot on setup samples 100–119; the coefficients did not change.
- The frozen protocol (`benchmark/frozen_protocol.json`), written before any
  test sample was simulated or encoded.
- All 5,400 test trajectories (54 problems × samples 0–99), all 5,400 image
  inputs, and the five-frame and 100-frame reference distances with
  numerical error bounds.
- Model-free baselines (character TF–IDF, pixels, Fourier magnitudes,
  physical cosine) on all 100 samples.

**Done at a reduced scale fixed in advance**

- EmbeddingGemma 2, Qwen3-VL-Embedding-2B and GME-Qwen2-VL-2B with text,
  image and joint inputs on samples 0–9 of all 54 problems (1,620 inputs per
  model), and the controls (time reversal, notation rewrites, mismatched
  descriptions, first frame, heat equation) on samples 0–2.
- Reason: the machine had 2 CPU cores, 7 GB RAM and no GPU. The full
  48,600-input run would take about 30–40 hours there.

**Not yet done**

- The 100-sample embedding run (needs a GPU; one command, below).
- Human review of all code, proofs, citations and claims.

## Design at a glance

| Choice | Specification |
|---|---|
| PDEs | Diffusion, advection–diffusion, linear exchange, Fisher–KPP, shifted Allen–Cahn, Gray–Scott |
| Initial conditions | One smooth bump, bounded oscillations, three smooth patches |
| Boundary conditions | Periodic, zero Dirichlet, zero Neumann |
| Domain and time | [0,1]², 0 ≤ t ≤ 1 |
| Saved states | t_n = n/99 for n = 0,…,99; 4 ETDRK4 steps per interval |
| Observed frames | Indices 0, 25, 50, 74, 99 |
| Full array | [54,100,100,2,64,64] float32, 17.7 GB (regenerated, checksummed) |
| Stored array | the five observed frames, [54,100,5,2,64,64], 0.9 GB |
| Reference distance | RMS difference between solutions from the same initial fields |
| Embedding score | Mean dot product of unit embeddings over the same samples |
| Model inputs | Text description; 688×286 image of five snapshots; both together |
| Models | EmbeddingGemma 2; Qwen3-VL-Embedding-2B; GME-Qwen2-VL-2B-Instruct (pinned revisions) |

## Repository layout

```
benchmark/
  configurations/               54 problem descriptions (one per PDE, IC family and BC)
  resolved_decks.zip            5,400 descriptions with explicit initial fields (.json and the model .txt)
  ic_parameter_bank.json        random initial-condition parameters of samples 0-99
  spec.json, deck_checks.json   design summary and generator self-checks
  math_checks.json              numerical checks of the reference math
  simulator_validation.json     solver checks on setup samples
  exponax_crosscheck.json       comparison with an independent spectral solver
  pilot_report.json             pilot on setup samples 100-119
  frozen_protocol.json          everything fixed before the test run
  embedding_interface_checks.json  model input formats checked against the official code
  results/
    simulation_manifest.json    solver settings, checksums, value ranges
    reference_terms.npz         per-sample distances, cosines and error terms
    embeddings/*.npz, *.meta.json  all embeddings, with timings and token counts
    analysis.json               every reported statistic
    model_input_sheets_r000-r009.zip  the 540 images encoded in the main runs
scripts/                        all code (below)
data/                           regenerated arrays and images (not in git)
```

## The scripts

| Script | What it does |
|---|---|
| `common.py` | Shared paths, the list of 54 problems, and small helpers |
| `make_decks.py` | Draws the initial conditions and writes the problem descriptions |
| `simulator.py` | The finite-difference ETDRK4 solver and the exact linear solutions |
| `render.py` | Draws the five-snapshot image given to the models |
| `controls.py` | Inputs for the control experiments |
| `check_reference.py` | Numerical checks of the reference math |
| `validate_simulator.py` | Solver checks on setup samples |
| `exponax_crosscheck.py` | Comparison with Exponax (separate jax environment) |
| `pilot.py` | Pilot checks that fix the image color range |
| `freeze_protocol.py` | Writes the frozen protocol (refuses to overwrite it) |
| `run_simulations.py` | Main simulation run: arrays, images, reference terms |
| `embed_models.py` | The three embedding models behind one interface |
| `run_embeddings.py` | Encodes the inputs with one model (resumable) |
| `analysis.py` | All statistics in `analysis.json` |
| `make_tables.py`, `make_figures.py` | Paper tables, macros and figures |

## Reproduce

Python 3.13. `scripts/requirements.txt` lists the numerical packages and
`scripts/requirements-embeddings.txt` the exact versions used for the models.
Run the scripts from the repository root. Large outputs go to `data/` (change
it with `--data` or the `PDE_DATA_DIR` variable). The models are read from the
Hugging Face cache (or from `PDE_MODEL_ROOT`).

```bash
python scripts/make_decks.py --all-realizations    # descriptions (already included)
python scripts/check_reference.py                  # math checks
python scripts/validate_simulator.py               # solver checks (setup samples only)
python scripts/pilot.py                            # pilot, samples 100-119
python scripts/freeze_protocol.py --output new_protocol.json   # the study's protocol already exists
python scripts/run_simulations.py                  # 5,400 trajectories and images (~40 min, 2 cores)
python scripts/run_embeddings.py --model qwen3vl2b # main task on samples 0-9, controls on 0-2
#   (the same for gme2b and embeddinggemma2)
python scripts/exponax_crosscheck.py               # in a jax/exponax environment
python scripts/analysis.py                         # benchmark/results/analysis.json
python scripts/make_tables.py                      # sections/generated/
python scripts/make_figures.py                     # figures/
```

**Full-scale embedding run:** `run_embeddings.py --realizations 100` (and
`--control-realizations 100` if wanted). The analysis uses every sample that
is encoded for all 54 problems.

A model sees only the text description (`.txt`) and/or the image. PDE names,
problem names and solver settings never reach a model. All PDEs and BCs share
the same initial fields for a given IC family and sample.

## Integrity notes

- Test samples (0–99) were not used for any setup decision. The image color
  range and the solver settings came from the pilot.
- The analysis, table and figure scripts were written after the protocol was
  frozen and before any embedding output was inspected. `exponax_crosscheck.py`
  was added after the freeze; it uses only setup samples.
- After the study, the scripts were rewritten for readability. The rewrite was
  checked to give bit-identical descriptions, trajectories, images, reference
  terms, embeddings, `analysis.json` and paper tables. The script hashes in
  `frozen_protocol.json` describe the scripts at the freeze; the scripts as
  they were run are in commit `b01ce41`. The description JSON files no longer
  carry two outdated "not_run" status fields; the text given to the models is
  unchanged.
- An initial Qwen3-VL run used one CPU thread while the simulator used the
  other core. The thread count changed image embeddings slightly (cosine
  ≈ 0.9996), so those 420 outputs were discarded unseen and every model was
  encoded with two threads.
- No trajectory or encoding failed; nothing was clipped or dropped.

The RMS distance, the kernel identities, the ETDRK4 integrator and the heat
calibration are standard tools, and the paper describes them that way.
