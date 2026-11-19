#!/usr/bin/env python3
"""Encode decks and contact sheets with one frozen model.

Tasks (any subset, in this order):
  main        resolved text, five-frame sheet, and both, for realizations
              0..R-1 of all 54 configurations (realization-major order)
  reversal    time-reversed sheets (image mode), realizations 0..Rc-1
  notation    notation-equivalent decks (text mode), realizations 0..Rc-1
  mismatch    sheet + deck of the next PDE class (combined mode), 0..Rc-1
  firstframe  first-frame-only sheets, one per IC family and realization
              (0..Rc-1), each encoded twice
  heat        the four analytical heat-calibration decks (text mode)

Embeddings are appended to benchmark/results/embeddings/<model>.npz-style
shards (resumable). Each record is keyed "<task>|<mode>|<config>|r<rrr>".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import controls as C  # noqa: E402
import embed_models as E  # noqa: E402
import make_decks as M  # noqa: E402
import render as R  # noqa: E402
import simulator as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [(p, f, b) for p in S.PDES for f in M.FAMILIES for b in S.BCS]
NAMES = [M.config_id(*c) for c in CONFIGS]


class Store:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, np.ndarray] = {}
        self.meta: dict[str, dict] = {}
        if path.exists():
            with np.load(path) as z:
                self.data = {k: z[k] for k in z.files}
        mpath = path.with_suffix(".meta.json")
        if mpath.exists():
            self.meta = json.loads(mpath.read_text())

    def save(self):
        tmp = self.path.with_name(self.path.stem + ".tmp.npz")
        np.savez(tmp, **self.data)
        tmp.replace(self.path)
        self.path.with_suffix(".meta.json").write_text(json.dumps(self.meta, indent=0) + "\n")


def deck_text(name: str, r: int, data: Path) -> str:
    return (data / "decks" / "realizations" / f"{name}__r{r:03d}.txt").read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(E.BACKENDS))
    ap.add_argument("--tasks", default="main,reversal,notation,mismatch,firstframe,heat")
    ap.add_argument("--realizations", type=int, default=10)
    ap.add_argument("--control-realizations", type=int, default=3)
    ap.add_argument("--data", type=Path, default=Path("/home/claude/data"))
    ap.add_argument("--save-every", type=int, default=30)
    ap.add_argument("--float32", action="store_true", help="precision diagnostic (EmbeddingGemma 2 only)")
    args = ap.parse_args()
    protocol = json.loads((ROOT / "benchmark" / "frozen_protocol.json").read_text())
    ranges = [tuple(r) for r in protocol["contact_sheet"]["channel_ranges"]]
    frames = protocol["observation_indices"]
    times = [k / 99 for k in frames]
    out = ROOT / "benchmark" / "results" / "embeddings"
    out.mkdir(parents=True, exist_ok=True)
    store = Store(out / (f"{args.model}_fp32.npz" if args.float32 else f"{args.model}.npz"))
    if args.float32:
        import torch
        assert args.model == "embeddinggemma2"
        torch.set_num_threads(int(__import__("os").environ.get("PDE_TORCH_THREADS", "2")))
        backend = E.GemmaBackend(dtype=torch.float32)
    else:
        backend = E.load(args.model)
    store.meta.setdefault("model", {"key": args.model, "checkpoint": E.CHECKPOINTS[args.model][0],
                                    "revision": E.CHECKPOINTS[args.model][1], **backend.describe()})
    five_cache: dict[str, np.ndarray] = {}

    def five(name: str) -> np.ndarray:
        if name not in five_cache:
            five_cache[name] = np.load(args.data / "trajectories" / "five_frame" / f"{name}.npy", mmap_mode="r")
        return five_cache[name]

    def sheet(name: str, r: int) -> Image.Image:
        return Image.open(args.data / "sheets" / f"{name}__r{r:03d}.png").convert("RGB")

    jobs = []  # (key, item-builder)
    tasks = args.tasks.split(",")
    if "main" in tasks:
        for r in range(args.realizations):
            for name in NAMES:
                jobs.append((f"main|text|{name}|r{r:03d}", lambda n=name, r=r: {"text": deck_text(n, r, args.data)}))
                jobs.append((f"main|image|{name}|r{r:03d}", lambda n=name, r=r: {"image": sheet(n, r)}))
                jobs.append((f"main|both|{name}|r{r:03d}",
                             lambda n=name, r=r: {"text": deck_text(n, r, args.data), "image": sheet(n, r)}))
    rc = range(args.control_realizations)
    if "reversal" in tasks:
        for r in rc:
            for name in NAMES:
                def build(n=name, r=r):
                    arr = np.asarray(five(n)[r])[C.TIME_REVERSAL]
                    return {"image": R.render_sheet(arr, times, ranges)[0]}
                jobs.append((f"reversal|image|{name}|r{r:03d}", build))
    if "notation" in tasks:
        for r in rc:
            for (pde, fam, bc), name in zip(CONFIGS, NAMES):
                jobs.append((f"notation|text|{name}|r{r:03d}",
                             lambda n=name, p=pde, r=r: {"text": C.equivalent_deck(deck_text(n, r, args.data), p)}))
    if "mismatch" in tasks:
        for r in rc:
            for (pde, fam, bc), name in zip(CONFIGS, NAMES):
                partner = M.config_id(S.PDES[C.mismatch_partner(S.PDES.index(pde))], fam, bc)
                jobs.append((f"mismatch|both|{name}|r{r:03d}",
                             lambda n=name, q=partner, r=r: {"text": deck_text(q, r, args.data), "image": sheet(n, r)}))
    if "firstframe" in tasks:
        for r in rc:
            for fam in M.FAMILIES:
                name = M.config_id("diffusion", fam, "periodic")  # identical t=0 array for every PDE/BC
                for copy in (0, 1):
                    def build(n=name, r=r):
                        arr = np.asarray(five(n)[r])[:1]
                        return {"image": R.render_sheet(arr, times[:1], ranges)[0]}
                    jobs.append((f"firstframe|image|{fam}|r{r:03d}|copy{copy}", build))
    if "heat" in tasks:
        for nu in [C.HEAT_ANCHOR] + C.HEAT_CANDIDATES:
            jobs.append((f"heat|text|nu={nu:.3f}|", lambda nu=nu: {"text": C.heat_deck(nu)}))

    todo = [(k, b) for k, b in jobs if k not in store.data]
    print(f"{args.model}: {len(jobs)} jobs, {len(todo)} to do", flush=True)
    t0 = time.time()
    failures = store.meta.setdefault("failures", {})
    for i, (key, build) in enumerate(todo):
        item = build()
        try:
            t1 = time.time()
            emb = backend.embed([item])[0]
            dt = time.time() - t1
            if not np.isfinite(emb).all():
                raise FloatingPointError("non-finite embedding")
        except Exception as exc:  # recorded, never silently dropped
            failures[key] = repr(exc)
            print("FAILED", key, repr(exc), flush=True)
            continue
        store.data[key] = emb.astype(np.float32)
        info = {"seconds": round(dt, 3)}
        if i < 3 or key.startswith(("main|text|", "main|both|")) and key.endswith("r000"):
            info["tokens"] = int(backend.token_count(item))
            if item.get("image") is not None and hasattr(backend, "processed_image_size"):
                info["processed_image_size"] = list(backend.processed_image_size(item["image"]))
        if item.get("text") is not None:
            info["text_sha256"] = hashlib.sha256(item["text"].encode()).hexdigest()[:16]
        store.meta.setdefault("records", {})[key] = info
        if (i + 1) % args.save_every == 0:
            store.save()
            rate = (time.time() - t0) / (i + 1)
            print(f"{i + 1}/{len(todo)}  {rate:.2f}s/item  eta {rate * (len(todo) - i - 1) / 60:.0f} min", flush=True)
    store.save()
    print(f"done {args.model} in {(time.time() - t0) / 60:.1f} min; failures: {len(failures)}", flush=True)


if __name__ == "__main__":
    main()
