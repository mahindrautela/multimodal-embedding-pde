"""Shared paths, problem lists, and small helpers used by every script.

Naming used throughout the code:

* problem   one combination of PDE, initial-condition (IC) family and boundary
            condition (BC); there are 6 x 3 x 3 = 54 problems.
* sample    one random draw of the initial fields, indexed r. Samples 0..99
            are the test set; samples 100 and higher are used only for setup.
* deck      the plain-text problem description given to the text models.
* sheet     the 2 x 5 image of five snapshots given to the image models.
"""
from __future__ import annotations

import json
import os
import zipfile
from functools import lru_cache
from pathlib import Path

import numpy as np

import make_decks

# ---------------------------------------------------------------------------
# Locations
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
BENCHMARK = REPO / "benchmark"
RESULTS = BENCHMARK / "results"
EMBEDDINGS = RESULTS / "embeddings"
DECKS_ZIP = BENCHMARK / "resolved_decks.zip"
PROTOCOL_FILE = BENCHMARK / "frozen_protocol.json"

# Large regenerated data (trajectories and sheets) lives outside git.
# Override with the PDE_DATA_DIR environment variable or a --data argument.
DATA_DIR = Path(os.environ.get("PDE_DATA_DIR", REPO / "data"))

# ---------------------------------------------------------------------------
# The 54 problems, always in this order (PDE, then IC family, then BC)
# ---------------------------------------------------------------------------
PDES = list(make_decks.PDE_SPECS)
FAMILIES = list(make_decks.FAMILIES)
BCS = list(make_decks.BOUNDARIES)
LINEAR_PDES = {"diffusion", "advection_diffusion", "linear_exchange"}

PROBLEMS = [(pde, family, bc) for pde in PDES for family in FAMILIES for bc in BCS]
NAMES = [make_decks.config_id(*problem) for problem in PROBLEMS]

# Indices of the problems in each IC family (18 each).
FAMILY_INDEX = {family: [i for i, p in enumerate(PROBLEMS) if p[1] == family] for family in FAMILIES}

# Saved states and the five observed snapshots.
N_STATES = make_decks.N_STATES                  # 100 states at t_n = n / 99
FRAMES = list(make_decks.FRAME_INDICES)         # [0, 25, 50, 74, 99]
GRID = make_decks.GRID_N                        # 64

# Embedding checkpoints, pinned to the revisions used in the study.
CHECKPOINTS = {
    "embeddinggemma2": ("google/embeddinggemma-2", "f9972ac22e696a105c482202646979e33e4b5a1e"),
    "qwen3vl2b": ("Qwen/Qwen3-VL-Embedding-2B", "9f2f7e710d6d81056aa5c0a4f04764fec6bb7bda"),
    "gme2b": ("Alibaba-NLP/gme-Qwen2-VL-2B-Instruct", "9cfa6413f704a7c1cf5064d240748e10c876b286"),
}
MODELS = list(CHECKPOINTS)
MODES = ["text", "image", "both"]   # "both" = text and image in one joint input


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def problem_name(pde: str, family: str, bc: str) -> str:
    return make_decks.config_id(pde, family, bc)


def observation_times() -> list[float]:
    """Times of the five observed snapshots."""
    return [k / (N_STATES - 1) for k in FRAMES]


def load_protocol() -> dict:
    return json.loads(PROTOCOL_FILE.read_text())


def sheet_ranges(protocol: dict | None = None) -> list[tuple[float, float]]:
    """Fixed per-field color range of the sheets, chosen by the pilot."""
    protocol = protocol or load_protocol()
    return [tuple(r) for r in protocol["contact_sheet"]["channel_ranges"]]


def rms(values) -> float:
    """Root-mean-square of all entries."""
    return float(np.sqrt(np.mean(np.square(values))))


def restrict(field: np.ndarray, factor: int) -> np.ndarray:
    """Average factor x factor blocks, mapping a fine cell-centered grid onto a coarse one."""
    s = field.shape
    blocks = field.reshape(*s[:-2], s[-2] // factor, factor, s[-1] // factor, factor)
    return blocks.mean(axis=(-3, -1))


def initial_fields(family: str, samples, n: int = GRID) -> np.ndarray:
    """Initial arrays [len(samples), 2, n, n] for one IC family."""
    return np.stack([make_decks.initial_array(family, make_decks.draw_parameters(r)[family], n=n)
                     for r in samples])


# ---------------------------------------------------------------------------
# Reading released inputs
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _deck_archive() -> zipfile.ZipFile:
    return zipfile.ZipFile(DECKS_ZIP)


def deck_text(name: str, r: int) -> str:
    """The text description of sample r of a problem, read from resolved_decks.zip."""
    return _deck_archive().read(f"realizations/{name}__r{r:03d}.txt").decode("utf-8")


def five_frame_file(data: Path, name: str) -> Path:
    """Stored snapshots of one problem: float32 [100 samples, 5 frames, 2 fields, 64, 64]."""
    return data / "trajectories" / "five_frame" / f"{name}.npy"


def sheet_file(data: Path, name: str, r: int) -> Path:
    return data / "sheets" / f"{name}__r{r:03d}.png"
