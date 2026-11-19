#!/usr/bin/env python3
"""Encode descriptions and image sheets with one frozen model.

Tasks (any subset, run in this order):
  main        text, image, and text + image for samples 0..R-1 of all 54 problems
  reversal    sheets with the five snapshots in reverse order (image only)
  notation    descriptions rewritten in equivalent notation (text only)
  mismatch    a sheet paired with the description of the next PDE (text + image)
  firstframe  sheets that show only t = 0, one per IC family, each encoded twice
  heat        the four heat-equation descriptions (text only)
The control tasks use samples 0..Rc-1.

Results go to <out-dir>/<model>.npz (one array per input) and
<model>.meta.json (model settings, timings, token counts, failures).
Every input has a key "<task>|<mode>|<problem>|r<rrr>". The run can be
stopped and restarted: inputs that already have an embedding are skipped.

Needs the five-frame arrays and sheets written by run_simulations.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

import controls as C
import embed_models as E
from render import render_sheet
from common import (CHECKPOINTS, DATA_DIR, EMBEDDINGS, FAMILIES, NAMES, PDES, PROBLEMS, deck_text,
                    five_frame_file, load_protocol, observation_times, problem_name, sheet_file, sheet_ranges)

ALL_TASKS = "main,reversal,notation,mismatch,firstframe,heat"


class Store:
    """Embeddings in one .npz file plus a JSON file of metadata."""

    def __init__(self, path: Path):
        self.path = path
        self.meta_path = path.with_suffix(".meta.json")
        self.data: dict[str, np.ndarray] = {}
        self.meta: dict[str, dict] = {}
        if path.exists():
            with np.load(path) as stored:
                self.data = {key: stored[key] for key in stored.files}
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text())

    def save(self) -> None:
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(tmp, **self.data)
        tmp.replace(self.path)                          # never leave a half-written file
        self.meta_path.write_text(json.dumps(self.meta, indent=0) + "\n")


def build_jobs(tasks: list[str], n_main: int, n_control: int, data: Path) -> list[tuple[str, Callable[[], dict]]]:
    """List of (key, function that builds the input item)."""
    protocol = load_protocol()
    ranges = sheet_ranges(protocol)
    times = observation_times()
    arrays: dict[str, np.ndarray] = {}

    def five_frames(name: str, r: int) -> np.ndarray:
        """[5, 2, 64, 64] snapshots of one sample (memory-mapped)."""
        if name not in arrays:
            arrays[name] = np.load(five_frame_file(data, name), mmap_mode="r")
        return np.asarray(arrays[name][r])

    def sheet(name: str, r: int) -> Image.Image:
        return Image.open(sheet_file(data, name, r)).convert("RGB")

    jobs = []
    if "main" in tasks:
        for r in range(n_main):
            for name in NAMES:
                tag = f"{name}|r{r:03d}"
                jobs.append((f"main|text|{tag}", lambda n=name, r=r: {"text": deck_text(n, r)}))
                jobs.append((f"main|image|{tag}", lambda n=name, r=r: {"image": sheet(n, r)}))
                jobs.append((f"main|both|{tag}",
                             lambda n=name, r=r: {"text": deck_text(n, r), "image": sheet(n, r)}))

    controls = range(n_control)
    if "reversal" in tasks:
        for r in controls:
            for name in NAMES:
                jobs.append((f"reversal|image|{name}|r{r:03d}", lambda n=name, r=r: {
                    "image": render_sheet(five_frames(n, r)[C.TIME_REVERSAL], times, ranges)[0]}))
    if "notation" in tasks:
        for r in controls:
            for (pde, _, _), name in zip(PROBLEMS, NAMES):
                jobs.append((f"notation|text|{name}|r{r:03d}",
                             lambda n=name, p=pde, r=r: {"text": C.equivalent_deck(deck_text(n, r), p)}))
    if "mismatch" in tasks:
        for r in controls:
            for (pde, family, bc), name in zip(PROBLEMS, NAMES):
                partner = problem_name(PDES[C.mismatch_partner(PDES.index(pde))], family, bc)
                jobs.append((f"mismatch|both|{name}|r{r:03d}",
                             lambda n=name, q=partner, r=r: {"text": deck_text(q, r), "image": sheet(n, r)}))
    if "firstframe" in tasks:
        for r in controls:
            for family in FAMILIES:
                # The t = 0 fields are the same for every PDE and BC of a family.
                name = problem_name("diffusion", family, "periodic")
                for copy in (0, 1):
                    jobs.append((f"firstframe|image|{family}|r{r:03d}|copy{copy}", lambda n=name, r=r: {
                        "image": render_sheet(five_frames(n, r)[:1], times[:1], ranges)[0]}))
    if "heat" in tasks:
        for nu in [C.HEAT_ANCHOR] + C.HEAT_CANDIDATES:
            jobs.append((f"heat|text|nu={nu:.3f}|", lambda nu=nu: {"text": C.heat_deck(nu)}))
    return jobs


def record_for(backend, key: str, item: dict, seconds: float, position: int) -> dict:
    """Run record of one input. Token counts are kept for the first three inputs
    of a run and for every sample-0 text and text + image input."""
    info = {"seconds": round(seconds, 3)}
    if position < 3 or (key.startswith(("main|text|", "main|both|")) and key.endswith("r000")):
        info["tokens"] = int(backend.token_count(item))
        if item.get("image") is not None and hasattr(backend, "processed_image_size"):
            info["processed_image_size"] = list(backend.processed_image_size(item["image"]))
    if item.get("text") is not None:
        info["text_sha256"] = hashlib.sha256(item["text"].encode()).hexdigest()[:16]
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, choices=list(E.BACKENDS))
    parser.add_argument("--tasks", default=ALL_TASKS, help=f"comma-separated subset of {ALL_TASKS}")
    parser.add_argument("--realizations", type=int, default=10, help="samples for the main task")
    parser.add_argument("--control-realizations", type=int, default=3, help="samples for the control tasks")
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="folder written by run_simulations.py")
    parser.add_argument("--out-dir", type=Path, default=EMBEDDINGS)
    parser.add_argument("--save-every", type=int, default=30)
    parser.add_argument("--float32", action="store_true", help="precision check (EmbeddingGemma 2 only)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    store = Store(args.out_dir / (f"{args.model}_fp32.npz" if args.float32 else f"{args.model}.npz"))
    backend = E.load(args.model, float32=args.float32)
    repo, revision = CHECKPOINTS[args.model]
    store.meta.setdefault("model", {"key": args.model, "checkpoint": repo, "revision": revision,
                                    **backend.describe()})
    failures = store.meta.setdefault("failures", {})
    records = store.meta.setdefault("records", {})

    jobs = build_jobs(args.tasks.split(","), args.realizations, args.control_realizations, args.data)
    todo = [(key, build) for key, build in jobs if key not in store.data]
    print(f"{args.model}: {len(jobs)} inputs, {len(todo)} still to encode", flush=True)
    start = time.time()
    for i, (key, build) in enumerate(todo):
        item = build()
        try:
            t0 = time.time()
            embedding = backend.embed([item])[0]
            seconds = time.time() - t0
            if not np.isfinite(embedding).all():
                raise FloatingPointError("non-finite embedding")
        except Exception as exc:                          # recorded, never silently dropped
            failures[key] = repr(exc)
            print("FAILED", key, repr(exc), flush=True)
            continue
        failures.pop(key, None)                           # an earlier failure is resolved
        store.data[key] = embedding.astype(np.float32)
        records[key] = record_for(backend, key, item, seconds, i)
        if (i + 1) % args.save_every == 0:
            store.save()
            rate = (time.time() - start) / (i + 1)
            print(f"{i + 1}/{len(todo)}  {rate:.2f} s/input  about {rate * (len(todo) - i - 1) / 60:.0f} min left",
                  flush=True)
    store.save()
    print(f"done {args.model} in {(time.time() - start) / 60:.1f} min; failures: {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
