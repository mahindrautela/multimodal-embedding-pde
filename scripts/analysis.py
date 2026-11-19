#!/usr/bin/env python3
"""Primary, secondary and control analyses (writes benchmark/results/analysis.json).

Definitions follow benchmark/frozen_protocol.json. All matrices for a model
are computed on the realization subset that was actually encoded, and the
reference is recomputed on exactly the same subset ("matched"). Baselines
that need no model inference are also reported on all 100 realizations.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import controls as C  # noqa: E402
import make_decks as M  # noqa: E402
import simulator as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "benchmark" / "results"
EMB_DIR = Path(os.environ.get("PDE_EMB_DIR", str(RES / "embeddings")))
OUT_NAME = os.environ.get("PDE_ANALYSIS_OUT", "analysis.json")
CONFIGS = [(p, f, b) for p in S.PDES for f in M.FAMILIES for b in S.BCS]
NAMES = [M.config_id(*c) for c in CONFIGS]
LINEAR = {"diffusion", "advection_diffusion", "linear_exchange"}
NORM5 = 5 * 2 * 64 * 64
NORM100 = 100 * 2 * 64 * 64
MODELS = ["embeddinggemma2", "qwen3vl2b", "gme2b"]
MODES = ["text", "image", "both"]
FAMILY_INDEX = {f: [i for i, c in enumerate(CONFIGS) if c[1] == f] for f in M.FAMILIES}


# ---------------------------------------------------------------------------
# Reference quantities
# ---------------------------------------------------------------------------
class Reference:
    def __init__(self, path: Path = RES / "reference_terms.npz"):
        z = np.load(path)
        assert list(z["config_names"]) == NAMES
        self.sq5, self.sq100 = z["sq5"], z["sq100"]
        self.cos5, self.cos100 = z["cos5"], z["cos100"]
        self.err5, self.tstep5 = z["err5"], z["tstep5"]
        self.out_of_range = z["out_of_range"]

    def D(self, rs, frames: int = 5) -> np.ndarray:
        sq, norm = (self.sq5, NORM5) if frames == 5 else (self.sq100, NORM100)
        return np.sqrt(sq[rs].sum(0) / (len(rs) * norm))

    def S_phys(self, rs, frames: int = 5) -> np.ndarray:
        return (self.cos5 if frames == 5 else self.cos100)[rs].mean(0)

    def eps(self, rs) -> tuple[np.ndarray, np.ndarray]:
        """Per-configuration error bound on the stacked five-frame profile, and a 'certified' flag."""
        base = np.sqrt(self.err5[rs].sum(0) / (len(rs) * NORM5))
        step = np.sqrt(self.tstep5[rs].sum(0) / (len(rs) * NORM5))
        linear = np.array([c[0] in LINEAR for c in CONFIGS])
        return np.where(linear, base, base + 2 * step), linear


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def offdiag(idx):
    return [(a, b) for a, b in itertools.combinations(idx, 2)]


def spearman(S_emb: np.ndarray, D: np.ndarray, pairs) -> float:
    s = np.array([S_emb[a, b] for a, b in pairs])
    d = np.array([D[a, b] for a, b in pairs])
    if np.ptp(s) == 0 or np.ptp(d) == 0:
        return float("nan")
    return float(spearmanr(s, -d).statistic)


def triplets(idx):
    out = []
    for i in idx:
        others = [j for j in idx if j != i]
        for j, k in itertools.combinations(others, 2):
            out.append((i, j, k))
    return out


def triplet_accuracy(S_emb, D, trips, keep=None) -> tuple[float, int]:
    credit, n = 0.0, 0
    for t, (i, j, k) in enumerate(trips):
        if keep is not None and not keep[t]:
            continue
        dj, dk = D[i, j], D[i, k]
        if dj == dk:
            continue
        near, far = (j, k) if dj < dk else (k, j)
        si, sf = S_emb[i, near], S_emb[i, far]
        credit += 1.0 if si > sf else (0.5 if si == sf else 0.0)
        n += 1
    return (credit / n if n else float("nan")), n


def resolved_mask(D, eps, trips):
    return np.array([abs(D[i, k] - D[i, j]) > 2 * eps[i] + eps[j] + eps[k] for i, j, k in trips])


def bc_of(i):
    return CONFIGS[i][2]


def pde_of(i):
    return CONFIGS[i][0]


def evaluate(S_emb: np.ndarray, D: np.ndarray, eps=None, certified=None) -> dict:
    out = {"per_family": {}}
    for fam, idx in FAMILY_INDEX.items():
        pairs = offdiag(idx)
        trips = triplets(idx)
        same = [(a, b) for a, b in pairs if bc_of(a) == bc_of(b)]
        changed = [(a, b) for a, b in pairs if bc_of(a) != bc_of(b)]
        same_t = np.array([bc_of(j) == bc_of(i) and bc_of(k) == bc_of(i) for i, j, k in trips])
        changed_t = np.array([bc_of(j) != bc_of(i) and bc_of(k) != bc_of(i) for i, j, k in trips])
        row = {"spearman": spearman(S_emb, D, pairs),
               "spearman_same_bc_pairs": spearman(S_emb, D, same),
               "spearman_changed_bc_pairs": spearman(S_emb, D, changed)}
        row["triplet_accuracy"], row["triplets"] = triplet_accuracy(S_emb, D, trips)
        row["triplet_accuracy_same_bc"], _ = triplet_accuracy(S_emb, D, trips, same_t)
        row["triplet_accuracy_changed_bc"], _ = triplet_accuracy(S_emb, D, trips, changed_t)
        if eps is not None:
            res = resolved_mask(D, eps, trips)
            cert = res & np.array([certified[i] and certified[j] and certified[k] for i, j, k in trips])
            row["resolved_fraction"] = float(res.mean())
            row["triplet_accuracy_resolved"], row["resolved_triplets"] = triplet_accuracy(S_emb, D, trips, res)
            row["certified_fraction"] = float(cert.mean())
            row["triplet_accuracy_certified"], row["certified_triplets"] = triplet_accuracy(S_emb, D, trips, cert)
        out["per_family"][fam] = row
    keys = [k for k in next(iter(out["per_family"].values())) if not k.endswith("triplets")]
    out["mean"] = {k: float(np.nanmean([out["per_family"][f][k] for f in M.FAMILIES])) for k in keys}
    pairs54 = offdiag(range(len(NAMES)))
    out["full_54"] = {"spearman": spearman(S_emb, D, pairs54)}
    return out


def factor_means(S_emb: np.ndarray) -> dict:
    """Exploratory: mean off-diagonal similarity by shared factor, within IC families."""
    groups = {"same_pde_diff_bc": [], "diff_pde_same_bc": [], "diff_pde_diff_bc": []}
    for idx in FAMILY_INDEX.values():
        for a, b in offdiag(idx):
            key = ("same_pde_diff_bc" if pde_of(a) == pde_of(b) else
                   "diff_pde_same_bc" if bc_of(a) == bc_of(b) else "diff_pde_diff_bc")
            groups[key].append(S_emb[a, b])
    return {k: float(np.mean(v)) for k, v in groups.items()}


def bootstrap(fn, rs, B=2000, seed=20261119) -> dict:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        sample = rng.choice(rs, size=len(rs), replace=True)
        vals.append(fn(sample))
    vals = np.array(vals)
    vals = vals[np.isfinite(vals)]
    return {"lo": float(np.quantile(vals, 0.025)), "hi": float(np.quantile(vals, 0.975)), "replicates": int(len(vals))}


# ---------------------------------------------------------------------------
# Embeddings and baselines
# ---------------------------------------------------------------------------
def load_embeddings(model: str):
    path = EMB_DIR / f"{model}.npz"
    if not path.exists():
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def paired_similarity(get, rs, names=NAMES) -> np.ndarray:
    """S_ij = mean_r z_i^(r) . z_j^(r) for available realizations rs."""
    Z = np.stack([np.stack([get(n, r) for n in names]) for r in rs])  # [R, K, d]
    return np.einsum("rid,rjd->ij", Z, Z) / len(rs)


def available_realizations(emb, task, mode, max_r=100):
    rs = []
    for r in range(max_r):
        if all(f"{task}|{mode}|{n}|r{r:03d}" in emb for n in NAMES):
            rs.append(r)
    return rs


def tfidf_similarity(rs, data: Path, rewrite=False):
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = []
    for r in rs:
        for (pde, _, _), n in zip(CONFIGS, NAMES):
            t = (data / "decks" / "realizations" / f"{n}__r{r:03d}.txt").read_text()
            texts.append(C.equivalent_deck(t, pde) if rewrite else t)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True)
    X = vec.fit_transform(texts)
    K = len(NAMES)
    S_ = np.zeros((K, K))
    for i, r in enumerate(rs):
        block = X[i * K:(i + 1) * K]
        S_ += (block @ block.T).toarray()
    return S_ / len(rs)


def pixel_similarity(rs, data: Path):
    S_ = np.zeros((len(NAMES), len(NAMES)))
    for r in rs:
        P = np.stack([np.asarray(Image.open(data / "sheets" / f"{n}__r{r:03d}.png"), dtype=np.float32).ravel()
                      for n in NAMES])
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        S_ += P @ P.T
    return S_ / len(rs)


def fourier_features(arr: np.ndarray, kmax: int = 8) -> np.ndarray:
    """arr [..., 5, 2, 64, 64] -> magnitudes of low 2D DFT modes, flattened per trajectory."""
    F = np.abs(np.fft.rfft2(arr, axes=(-2, -1)))
    kx = np.r_[0:kmax + 1, -kmax:0]
    F = F[..., kx, :][..., :kmax + 1]
    return F.reshape(*arr.shape[:-4], -1)


def fourier_similarity(rs, data: Path):
    feats = np.stack([fourier_features(np.load(data / "trajectories" / "five_frame" / f"{n}.npy")[rs])
                      for n in NAMES], axis=1)  # [R, K, d]
    feats /= np.linalg.norm(feats, axis=-1, keepdims=True)
    return np.einsum("rid,rjd->ij", feats, feats) / len(rs)


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/home/claude/data"))
    ap.add_argument("--bootstrap", type=int, default=2000)
    args = ap.parse_args()
    ref = Reference()
    all_r = list(range(ref.sq5.shape[0]))
    out: dict = {"realizations_reference": len(all_r)}

    # --- Reference descriptives --------------------------------------------
    D_all, D100_all = ref.D(all_r), ref.D(all_r, 100)
    eps_all, cert = ref.eps(all_r)
    desc = {}
    for fam, idx in FAMILY_INDEX.items():
        pairs = offdiag(idx)
        d = np.array([D_all[a, b] for a, b in pairs])
        trips = triplets(idx)
        res = resolved_mask(D_all, eps_all, trips)
        certm = res & np.array([cert[i] and cert[j] and cert[k] for i, j, k in trips])
        desc[fam] = {"D_min": float(d.min()), "D_median": float(np.median(d)), "D_max": float(d.max()),
                     "spearman_D5_vs_D100": float(spearmanr(d, [D100_all[a, b] for a, b in pairs]).statistic),
                     "resolved_fraction": float(res.mean()), "certified_fraction": float(certm.mean()),
                     "unresolved_triplets": int((~res).sum()), "triplets": len(trips),
                     "eps_max": float(eps_all[idx].max())}
    out["reference"] = {"per_family": desc,
                        "eps_max_linear_certified": float(eps_all[cert].max()),
                        "eps_max_nonlinear_estimated": float(eps_all[~cert].max()),
                        "spearman_D5_vs_D100_full54": spearman(-D100_all, D_all, offdiag(range(54))),
                        "out_of_range_pixels_total": int(ref.out_of_range.sum())}
    out["reference"]["D_matrix_five_frame"] = D_all.tolist()
    out["reference"]["D_matrix_100_frame"] = D100_all.tolist()

    # --- Baselines on all realizations and on the embedding subset ---------
    def baseline_block(rs, tag):
        block = {}
        sims = {"tfidf_text": tfidf_similarity(rs, args.data),
                "pixels_image": pixel_similarity(rs, args.data),
                "fourier_image": fourier_similarity(rs, args.data),
                "physical_cosine": ref.S_phys(rs)}
        D = ref.D(rs)
        eps, certified = ref.eps(rs)
        for k, S_ in sims.items():
            block[k] = evaluate(S_, D, eps, certified)
            block[k]["factor_means"] = factor_means(S_)
        block["first_frame_image"] = {"spearman": "undefined (constant scores)", "triplet_accuracy": 0.5}
        return block, sims

    out["baselines_all100"], _ = baseline_block(all_r, "all")

    # --- Models ----------------------------------------------------------------
    out["models"] = {}
    subset = None
    for model in MODELS:
        emb = load_embeddings(model)
        if emb is None:
            continue
        mres = {}
        for mode in MODES:
            rs = available_realizations(emb, "main", mode)
            if not rs:
                continue
            subset = rs if subset is None else sorted(set(subset) & set(rs))
            S_ = paired_similarity(lambda n, r, mode=mode: emb[f"main|{mode}|{n}|r{r:03d}"], rs)
            D = ref.D(rs)
            eps, certified = ref.eps(rs)
            row = evaluate(S_, D, eps, certified)
            row["realizations"] = rs
            row["S_offdiag_range_within_families"] = [
                float(min(S_[a, b] for idx in FAMILY_INDEX.values() for a, b in offdiag(idx))),
                float(max(S_[a, b] for idx in FAMILY_INDEX.values() for a, b in offdiag(idx)))]
            row["spearman_vs_D100_mean"] = float(np.nanmean(
                [spearman(S_, ref.D(rs, 100), offdiag(idx)) for idx in FAMILY_INDEX.values()]))
            row["spearman_vs_D_all100_mean"] = float(np.nanmean(
                [spearman(S_, D_all, offdiag(idx)) for idx in FAMILY_INDEX.values()]))
            row["spearman_vs_physical_cosine_mean"] = float(np.nanmean(
                [spearman(S_, -ref.S_phys(rs), offdiag(idx)) for idx in FAMILY_INDEX.values()]))
            row["factor_means"] = factor_means(S_)

            def stat(sample, mode=mode):
                S_b = paired_similarity(lambda n, r: emb[f"main|{mode}|{n}|r{r:03d}"], sample)
                D_b = ref.D(sample)
                return np.nanmean([spearman(S_b, D_b, offdiag(idx)) for idx in FAMILY_INDEX.values()])
            if args.bootstrap:
                row["mean_spearman_ci95"] = bootstrap(stat, rs, B=args.bootstrap)
            mres[mode] = row
        # Controls
        ctrl = {}
        rc = available_realizations(emb, "reversal", "image")
        if rc:
            S_rev = paired_similarity(lambda n, r: emb[f"reversal|image|{n}|r{r:03d}"], rc)
            S_img = paired_similarity(lambda n, r: emb[f"main|image|{n}|r{r:03d}"], rc)
            D = ref.D(rc)
            ctrl["time_reversal"] = {
                "realizations": rc,
                "spearman_reversed_vs_ref": float(np.nanmean([spearman(S_rev, D, offdiag(i)) for i in FAMILY_INDEX.values()])),
                "spearman_original_vs_ref": float(np.nanmean([spearman(S_img, D, offdiag(i)) for i in FAMILY_INDEX.values()])),
                "spearman_reversed_vs_original": float(np.nanmean([spearmanr([S_rev[a, b] for a, b in offdiag(i)],
                                                                             [S_img[a, b] for a, b in offdiag(i)]).statistic
                                                                   for i in FAMILY_INDEX.values()])),
                "mean_self_similarity_original_vs_reversed": float(np.mean(
                    [emb[f"reversal|image|{n}|r{r:03d}"] @ emb[f"main|image|{n}|r{r:03d}"] for n in NAMES for r in rc]))}
        rc = available_realizations(emb, "notation", "text")
        if rc:
            S_not = paired_similarity(lambda n, r: emb[f"notation|text|{n}|r{r:03d}"], rc)
            S_txt = paired_similarity(lambda n, r: emb[f"main|text|{n}|r{r:03d}"], rc)
            D = ref.D(rc)
            top1, self_sim, nearest_other, closer = [], [], [], []
            for r in rc:
                orig = np.stack([emb[f"main|text|{n}|r{r:03d}"] for n in NAMES])
                for i, n in enumerate(NAMES):
                    zr = emb[f"notation|text|{n}|r{r:03d}"]
                    sims = orig @ zr
                    top1.append(int(np.argmax(sims) == i))
                    self_sim.append(float(sims[i]))
                    others = orig @ orig[i]
                    others[i] = -np.inf
                    nearest_other.append(float(others.max()))
                    # how many of the other 53 problems are more similar to the original than its rewrite
                    closer.append(int((others > float(orig[i] @ zr)).sum()))
            ctrl["notation"] = {
                "realizations": rc,
                "rewrite_retrieval_top1_among_54": float(np.mean(top1)),
                "mean_similarity_original_vs_rewrite": float(np.mean(self_sim)),
                "mean_similarity_original_vs_nearest_other_configuration": float(np.mean(nearest_other)),
                "median_other_problems_closer_than_rewrite": float(np.median(closer)),
                "fraction_rewrite_farther_than_all_53_others": float(np.mean(np.array(closer) == len(NAMES) - 1)),
                "spearman_rewritten_vs_ref": float(np.nanmean([spearman(S_not, D, offdiag(i)) for i in FAMILY_INDEX.values()])),
                "spearman_original_vs_ref": float(np.nanmean([spearman(S_txt, D, offdiag(i)) for i in FAMILY_INDEX.values()]))}
        rc = available_realizations(emb, "mismatch", "both")
        if rc:
            S_mis = paired_similarity(lambda n, r: emb[f"mismatch|both|{n}|r{r:03d}"], rc)
            S_both = paired_similarity(lambda n, r: emb[f"main|both|{n}|r{r:03d}"], rc)
            D = ref.D(rc)
            perm = [NAMES.index(M.config_id(S.PDES[C.mismatch_partner(S.PDES.index(p))], f, b)) for p, f, b in CONFIGS]
            D_deck = D[np.ix_(perm, perm)]
            ctrl["mismatch"] = {
                "realizations": rc,
                "spearman_vs_image_reference": float(np.nanmean([spearman(S_mis, D, offdiag(i)) for i in FAMILY_INDEX.values()])),
                "spearman_vs_deck_reference": float(np.nanmean([spearman(S_mis, D_deck, offdiag(i)) for i in FAMILY_INDEX.values()])),
                "spearman_matched_both_vs_ref": float(np.nanmean([spearman(S_both, D, offdiag(i)) for i in FAMILY_INDEX.values()]))}
        ff = [k for k in emb if k.startswith("firstframe|")]
        if ff:
            diffs = []
            for k in ff:
                if k.endswith("copy0"):
                    k1 = k[:-1] + "1"
                    if k1 in emb:
                        diffs.append(float(np.abs(emb[k] - emb[k1]).max()))
            ctrl["first_frame"] = {"pairs_checked": len(diffs), "max_abs_difference_between_copies": max(diffs) if diffs else None,
                                   "spearman": "undefined (constant scores within each IC family)",
                                   "triplet_accuracy": 0.5}
        hk = sorted(k for k in emb if k.startswith("heat|"))
        if hk:
            anchor = emb[f"heat|text|nu={C.HEAT_ANCHOR:.3f}|"]
            rows = []
            for nu in C.HEAT_CANDIDATES:
                rows.append({"nu": nu, "rms_five_frames": C.heat_rms(C.HEAT_ANCHOR, nu),
                             "similarity_to_anchor": float(anchor @ emb[f"heat|text|nu={nu:.3f}|"])})
            sims = [r_["similarity_to_anchor"] for r_ in rows]
            ctrl["heat_calibration"] = {"rows": rows, "ordering_preserved": bool(sims[0] > sims[1] > sims[2])}
        mres["controls"] = ctrl
        out["models"][model] = mres

    # --- Precision diagnostic: EmbeddingGemma 2 float32 vs bfloat16 on realization 0 ---
    fp32, bf16 = load_embeddings("embeddinggemma2_fp32"), load_embeddings("embeddinggemma2")
    if fp32 is not None and bf16 is not None:
        prec = {}
        for mode in MODES:
            rs = available_realizations(fp32, "main", mode)
            if not rs:
                continue
            cos = [float(fp32[f"main|{mode}|{n}|r{r:03d}"] @ bf16[f"main|{mode}|{n}|r{r:03d}"]) for n in NAMES for r in rs]
            S32 = paired_similarity(lambda n, r, mode=mode: fp32[f"main|{mode}|{n}|r{r:03d}"], rs)
            S16 = paired_similarity(lambda n, r, mode=mode: bf16[f"main|{mode}|{n}|r{r:03d}"], rs)
            D = ref.D(rs)
            prec[mode] = {"realizations": rs, "min_cosine_fp32_bf16": min(cos), "mean_cosine_fp32_bf16": float(np.mean(cos)),
                          "spearman_S32_vs_S16_mean": float(np.nanmean([spearmanr([S32[a, b] for a, b in offdiag(i)],
                                                                                  [S16[a, b] for a, b in offdiag(i)]).statistic
                                                                        for i in FAMILY_INDEX.values()])),
                          "spearman_ref_fp32": float(np.nanmean([spearman(S32, D, offdiag(i)) for i in FAMILY_INDEX.values()])),
                          "spearman_ref_bf16": float(np.nanmean([spearman(S16, D, offdiag(i)) for i in FAMILY_INDEX.values()]))}
        out["precision_diagnostic_embeddinggemma2"] = prec

    if subset:
        out["baselines_embedding_subset"], _ = baseline_block(subset, "subset")
        out["baselines_embedding_subset"]["realizations"] = subset
        D_sub = ref.D(subset)
        out["reference"]["subset_vs_all100_spearman_mean"] = float(np.mean(
            [spearmanr([D_sub[a, b] for a, b in offdiag(i)], [D_all[a, b] for a, b in offdiag(i)]).statistic
             for i in FAMILY_INDEX.values()]))
    (RES / OUT_NAME).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", RES / OUT_NAME)


if __name__ == "__main__":
    main()
