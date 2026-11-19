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

## What this revision contains

This is the ICML 2027 working manuscript (supplied style unchanged) together
with the code and records of the executed study. The paper's numbers are
generated from the files in `benchmark/` by the scripts in `scripts/`.

**Completed at full scale**

- Common simulator (`scripts/simulator.py`) for all six PDEs and three BCs,
  validated against exact semidiscrete references, manufactured solutions,
  step halving, spatial refinement, and an independent Exponax periodic check.
- Excluded numerical pilot on realizations 100–119; coefficients unchanged.
- Frozen protocol (`benchmark/frozen_protocol.json`) written before any test
  realization was simulated or encoded.
- All 5,400 test trajectories (54 configurations × realizations 0–99), all
  5,400 contact sheets, and the five-frame / 100-frame reference matrices with
  numerical error bounds.
- Model-free baselines (character TF–IDF, rendered pixels, Fourier magnitudes,
  physical cosine) on all 100 realizations.

**Completed at reduced, predeclared scale**

- EmbeddingGemma 2, Qwen3-VL-Embedding-2B and GME-Qwen2-VL-2B in text, image
  and joint modes on realizations 0–9 of all 54 configurations (1,620 inputs
  per model), and the controls (time reversal, notation rewrites, mismatched
  decks, first frame, heat calibration) on realizations 0–2.
- Reason: the available machine had 2 CPU cores, 7 GB RAM and no GPU. The full
  48,600-input run would take roughly 30–40 hours there.

**Not yet done**

- The 100-realization embedding run (needs a GPU; one command, below).
- Human review of all code, proofs, citations and claims.

## Design at a glance

| Choice | Specification |
|---|---|
| PDE classes | Diffusion, advection–diffusion, linear exchange, Fisher–KPP, shifted Allen–Cahn, Gray–Scott |
| Initial conditions | One smooth bump, bounded oscillations, three smooth patches |
| Boundary conditions | Periodic, zero Dirichlet, zero Neumann |
| Domain and horizon | [0,1]²; dimensionless time 0 through 1 |
| Saved states | t_n=n/99 for n=0,…,99; solver uses 4 ETDRK4 steps per interval |
| Primary observed frames | Indices 0, 25, 50, 74, 99 |
| Full array | [54,100,100,2,64,64] float32, 17.7 GB (regenerated, checksummed) |
| Stored array | five observed frames, [54,100,5,2,64,64], 0.9 GB |
| Mathematical target | RMS difference between matched trajectory fields |
| Embedding score | Mean dot product of matched, unit-normalized embeddings |
| Input modes | Resolved deck; 688×286 contact sheet; native deck + image |
| Models | EmbeddingGemma 2; Qwen3-VL-Embedding-2B; GME-Qwen2-VL-2B-Instruct (pinned revisions) |

## Repository layout

```
main.tex, sections/            manuscript (sections/generated/ is written by scripts/make_tables.py)
figures/                       figures (scripts/make_figures.py)
benchmark/configurations/      54 configuration decks
benchmark/resolved_decks.zip   5,400 resolved JSON/text decks
benchmark/ic_parameter_bank.json
benchmark/simulator_validation.json, exact_reference_check.json, exponax_crosscheck.json
benchmark/pilot_report.json, frozen_protocol.json, embedding_interface_checks.json
benchmark/results/simulation_manifest.json   solver settings, checksums, diagnostics
benchmark/results/reference_terms.npz        per-realization squared distances, cosines, error terms
benchmark/results/embeddings/*.npz, *.meta.json   all computed embeddings + per-input metadata
benchmark/results/analysis.json              every reported statistic
benchmark/results/model_input_sheets_r000-r009.zip   the 540 contact sheets encoded in the main runs
scripts/                       all code (see below)
```

## Reproduce

Python 3.13; `scripts/requirements.txt` lists the numerical stack and
`scripts/requirements-embeddings.txt` the exact inference versions used (the
Exponax check runs in a separate jax/exponax environment).

```bash
python scripts/make_decks.py --all-realizations      # decks (already included)
python scripts/check_reference.py                    # calibration identities
python scripts/validate_simulator.py                 # simulator validation (pilot ICs only)
python scripts/check_exact_references.py
python scripts/pilot.py                              # excluded pilot, r = 100..119
python scripts/freeze_protocol.py                    # writes frozen_protocol.json
python scripts/run_simulations.py --data DATA        # 5,400 trajectories + sheets (~40 min, 2 cores)
python scripts/run_embeddings.py --model qwen3vl2b --tasks main --realizations 10
python scripts/run_embeddings.py --model qwen3vl2b --tasks reversal,notation,mismatch,firstframe,heat --control-realizations 3
#   (same for gme2b and embeddinggemma2; set PDE_MODEL_ROOT to the HF cache)
python scripts/exponax_crosscheck.py                 # in a jax/exponax environment
python scripts/analysis.py --data DATA               # writes benchmark/results/analysis.json
python scripts/make_tables.py
python scripts/make_figures.py DATA '[["qwen3vl2b","text","Qwen3-VL text"],["gme2b","image","GME image"],["qwen3vl2b","image","Qwen3-VL image"],["embeddinggemma2","image","EmbGemma 2 image"]]'
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

**Full-scale embedding run:** `run_embeddings.py --realizations 100` (and
`--control-realizations 100` if desired); the analysis picks up whatever
realizations are complete for every configuration and mode.

The model input is only `model_input_text` (or the `.txt` deck) and the
contact sheet; PDE names, configuration IDs and solver metadata never reach a
model. The simulator uses the same realized initial array for a given IC
family and index across all PDEs and BCs.

## Integrity notes

- Test realizations (0–99) were not used for any setup decision. Contact-sheet
  scales and solver settings came from the pilot.
- Analysis, table and figure scripts were written after the protocol freeze
  and before any embedding output was inspected. `exponax_crosscheck.py` and
  `check_exact_references.py` were also added after the freeze; they use only
  pilot or synthetic data. A `--float32` flag for the declared precision
  diagnostic was added to `run_embeddings.py` afterwards without changing its
  default path; every other script that existed at the freeze still matches
  the hash recorded in the protocol.
- An initial Qwen3-VL run used one CPU thread while the simulator occupied the
  other core. Thread count changed image embeddings slightly (cosine ≈ 0.9996),
  so those 420 outputs were discarded unseen and every model was encoded with
  two threads.
- No trajectory or encoding failed; nothing was clipped or dropped.

The RMS distance, kernel identities, ETDRK4 integrator and linear calibration
are classical tools, and the paper describes them that way.
