#!/usr/bin/env python3
"""Make the paper's figures (default output folder: figures/).

  fig_configurations.pdf  u at t = 1 for the 18 problems of one IC family
  fig_matrices.pdf        reference -D next to four embedding similarity matrices
  example_sheet.png       one real image input (a copy of a sheet)

Needs the data written by run_simulations.py and the results in benchmark/results/.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (the backend must be chosen first)
import numpy as np  # noqa: E402

import analysis as A  # noqa: E402
from common import (BCS, DATA_DIR, EMBEDDINGS, FAMILY_INDEX, PDES, PROBLEMS, REPO, RESULTS,  # noqa: E402
                    five_frame_file, load_protocol, problem_name, sheet_file)

PDE_LABEL = {"diffusion": "Diffusion", "advection_diffusion": "Adv.–diff.", "linear_exchange": "Exchange",
             "fisher_kpp": "Fisher–KPP", "shifted_allen_cahn": "Allen–Cahn", "gray_scott": "Gray–Scott"}
PDE_SHORT = ["Diff", "AdvD", "Exch", "FKPP", "AC", "GS"]
DEFAULT_PANELS = [["qwen3vl2b", "text", "Qwen3-VL text"], ["gme2b", "image", "GME image"],
                  ["qwen3vl2b", "image", "Qwen3-VL image"], ["embeddinggemma2", "image", "EmbGemma 2 image"]]
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.5})


def fig_configurations(data: Path, out: Path, family="low_frequency", r=1, channel=0, frame=4) -> None:
    """Initial u, then u at t = 1 for every PDE (rows) and BC (columns), for one sample."""
    vmax = load_protocol()["contact_sheet"]["channel_ranges"][channel][1]
    fig, axes = plt.subplots(len(PDES), 4, figsize=(3.0, 4.3), gridspec_kw={"wspace": 0.06, "hspace": 0.06})
    initial = np.load(five_frame_file(data, problem_name("diffusion", family, "periodic")),
                      mmap_mode="r")[r, 0, channel]
    image = None
    for i, pde in enumerate(PDES):
        for j in range(4):
            ax = axes[i, j]
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:                               # first column: the shared initial field, top row only
                if i != 0:
                    ax.axis("off")
                    continue
                field = initial
            else:
                field = np.load(five_frame_file(data, problem_name(pde, family, BCS[j - 1])),
                                mmap_mode="r")[r, frame, channel]
            image = ax.imshow(np.asarray(field).T, origin="lower", cmap="viridis", vmin=0, vmax=vmax,
                              interpolation="nearest")
            if i == 0:
                ax.set_title(["Initial $u$", "Periodic", "Dirichlet", "Neumann"][j], fontsize=6.5, pad=2)
            if j == 3:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(PDE_LABEL[pde], fontsize=6.5, rotation=0, ha="left", va="center", labelpad=3)
    # Color bar in the empty part of the first column.
    top, bottom = axes[2, 0].get_position(), axes[4, 0].get_position()
    cax = fig.add_axes([top.x0 + 0.45 * top.width, bottom.y0, 0.1 * top.width, top.y1 - bottom.y0])
    bar = fig.colorbar(image, cax=cax, orientation="vertical")
    bar.ax.tick_params(labelsize=5, length=1.5, pad=1)
    bar.ax.yaxis.set_ticks_position("left")
    fig.savefig(out / "fig_configurations.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def fig_matrices(panels: list[list[str]], out: Path, family="low_frequency") -> None:
    """Reference -D and embedding similarities for one IC family, on the encoded samples."""
    analysis = json.loads((RESULTS / "analysis.json").read_text())
    # Problems of the family, grouped by PDE (rows/columns), BC order inside each group.
    idx = sorted(FAMILY_INDEX[family], key=lambda i: (PDES.index(PROBLEMS[i][0]), BCS.index(PROBLEMS[i][2])))
    first_model, first_mode, _ = panels[0]
    rs = analysis["models"][first_model][first_mode]["realizations"]
    ref = A.Reference(RESULTS / "reference_terms.npz")
    matrices = [("Reference $-D$", -ref.D(rs)[np.ix_(idx, idx)])]
    for model, mode, label in panels:
        emb = A.load_embeddings(EMBEDDINGS, model)
        S = A.embedding_similarity(emb, "main", mode, rs)
        rho = analysis["models"][model][mode]["per_family"][family]["spearman"]
        matrices.append((f"{label}\n($\\rho_s={rho:.2f}$)", S[np.ix_(idx, idx)]))

    fig, axes = plt.subplots(1, len(matrices), figsize=(6.75, 1.75), gridspec_kw={"wspace": 0.18})
    ticks = [1 + 3 * k for k in range(len(PDES))]
    for ax, (title, matrix) in zip(axes, matrices):
        shown = matrix.copy()
        np.fill_diagonal(shown, np.nan)              # hide the trivial diagonal
        ax.imshow(shown, cmap="magma", interpolation="nearest")
        ax.set_title(title, fontsize=6.5, pad=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels(PDE_SHORT, fontsize=5, rotation=90)
        ax.set_yticks([])
        for k in range(1, len(PDES)):                 # white lines between PDE blocks
            ax.axhline(3 * k - 0.5, color="w", lw=0.3)
            ax.axvline(3 * k - 0.5, color="w", lw=0.3)
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels([PDE_LABEL[p] for p in PDES], fontsize=5)
    fig.savefig(out / "fig_matrices.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=REPO / "figures")
    parser.add_argument("--panels", type=json.loads, default=DEFAULT_PANELS,
                        help='JSON list of [model, input, label] for the matrix figure')
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig_configurations(args.data, args.out_dir)
    fig_matrices(args.panels, args.out_dir)
    example = sheet_file(args.data, problem_name("gray_scott", "low_frequency", "dirichlet_zero"), 1)
    shutil.copyfile(example, args.out_dir / "example_sheet.png")
    print("figures written to", args.out_dir)


if __name__ == "__main__":
    main()
