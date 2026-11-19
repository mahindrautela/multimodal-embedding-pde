#!/usr/bin/env python3
"""Figures for the manuscript (written to figures/)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_decks as M  # noqa: E402
import simulator as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/home/claude/data")
PDE_LABEL = {"diffusion": "Diffusion", "advection_diffusion": "Adv.\u2013diff.", "linear_exchange": "Exchange",
             "fisher_kpp": "Fisher\u2013KPP", "shifted_allen_cahn": "Allen\u2013Cahn", "gray_scott": "Gray\u2013Scott"}
BC_LABEL = {"periodic": "Periodic", "dirichlet_zero": "Dirichlet", "neumann_zero": "Neumann"}
plt.rcParams.update({"font.size": 7, "font.family": "serif", "axes.linewidth": 0.5})


def fig_configurations(family="low_frequency", r=1, channel=0, frame=4):
    """u at t=1 for the 18 configurations of one IC family and realization."""
    vmax = json.loads((ROOT / "benchmark" / "frozen_protocol.json").read_text())["contact_sheet"]["channel_ranges"][channel][1]
    fig, axes = plt.subplots(len(S.PDES), 4, figsize=(3.0, 4.3), gridspec_kw={"wspace": 0.06, "hspace": 0.06})
    first = np.load(DATA / "trajectories" / "five_frame" / f"diffusion__{family}__periodic.npy", mmap_mode="r")[r, 0, channel]
    im = None
    for i, pde in enumerate(S.PDES):
        for j in range(4):
            ax = axes[i, j]
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:
                if i != 0:
                    ax.axis("off")
                    continue
                field = first
            else:
                bc = S.BCS[j - 1]
                field = np.load(DATA / "trajectories" / "five_frame" / f"{pde}__{family}__{bc}.npy",
                                mmap_mode="r")[r, frame, channel]
            im = ax.imshow(np.asarray(field).T, origin="lower", cmap="viridis", vmin=0, vmax=vmax,
                           interpolation="nearest")
            if i == 0:
                ax.set_title(["Initial $u$", "Periodic", "Dirichlet", "Neumann"][j], fontsize=6.5, pad=2)
            if j == 3:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(PDE_LABEL[pde], fontsize=6.5, rotation=0, ha="left", va="center", labelpad=3)
    pos = axes[2, 0].get_position()
    pos4 = axes[4, 0].get_position()
    cax = fig.add_axes([pos.x0 + 0.45 * pos.width, pos4.y0, 0.1 * pos.width, pos.y1 - pos4.y0])
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1)
    cb.ax.yaxis.set_ticks_position("left")
    fig.savefig(FIG / "fig_configurations.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def fig_matrices(selection, family="low_frequency"):
    """Reference -D and embedding similarities for one IC family (realizations 0-9)."""
    an = json.loads((ROOT / "benchmark" / "results" / "analysis.json").read_text())
    names = json.loads((ROOT / "benchmark" / "results" / "simulation_manifest.json").read_text())["configurations"]
    idx = [i for i, n in enumerate(names) if n.split("__")[1] == family]
    # order: BC-major within PDE
    idx = sorted(idx, key=lambda i: (S.PDES.index(names[i].split("__")[0]), S.BCS.index(names[i].split("__")[2])))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import analysis as A
    ref = A.Reference()
    rs = an["models"][selection[0][0]][selection[0][1]]["realizations"]
    D = ref.D(rs)
    panels = [("Reference $-D$", -D[np.ix_(idx, idx)], None)]
    for model, mode, label in selection:
        emb = A.load_embeddings(model)
        S_ = A.paired_similarity(lambda n, r: emb[f"main|{mode}|{n}|r{r:03d}"], rs)
        sp = an["models"][model][mode]["per_family"][family]["spearman"]
        panels.append((f"{label}\n($\\rho_s={sp:.2f}$)", S_[np.ix_(idx, idx)], sp))
    fig, axes = plt.subplots(1, len(panels), figsize=(6.75, 1.75), gridspec_kw={"wspace": 0.18})
    ticks = [1 + 3 * k for k in range(6)]
    for ax, (title, mat, _) in zip(axes, panels):
        m = mat.copy()
        np.fill_diagonal(m, np.nan)
        ax.imshow(m, cmap="magma", interpolation="nearest")
        ax.set_title(title, fontsize=6.5, pad=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels(["Diff", "AdvD", "Exch", "FKPP", "AC", "GS"], fontsize=5, rotation=90)
        ax.set_yticks([])
        for k in range(1, 6):
            ax.axhline(3 * k - 0.5, color="w", lw=0.3)
            ax.axvline(3 * k - 0.5, color="w", lw=0.3)
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels([PDE_LABEL[p] for p in S.PDES], fontsize=5)
    fig.savefig(FIG / "fig_matrices.pdf", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    fig_configurations()
    sel = json.loads(sys.argv[2]) if len(sys.argv) > 2 else None
    if sel:
        fig_matrices(sel)
    print("figures written")
