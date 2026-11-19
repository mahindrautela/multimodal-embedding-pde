#!/usr/bin/env python3
"""All analyses of the paper (writes benchmark/results/analysis.json).

Terms
  D        reference distance between two problems: RMS difference of their
           solutions over matched samples, five observed frames, both fields
           and all pixels (D100 uses all 100 saved frames).
  S        embedding score between two problems: the mean dot product of
           their unit embeddings over matched samples.
  eps      numerical error of each problem's stored solution, on the scale of D.

Metrics are computed inside each IC family (18 problems) and then averaged
over the three families:
  Spearman          rank correlation between S and -D over the 153 pairs
  triplet accuracy  for an anchor i and two others j, k: is the problem that
                    is closer in D also more similar in S? (ties in S count 1/2)
  resolved triplet  |D_ik - D_ij| > 2 eps_i + eps_j + eps_k, i.e. numerical
                    error cannot flip the reference order.

Every model result uses only the samples that were encoded, and the reference
is recomputed on the same samples. The baselines need no model, so they are
also reported on all 100 samples.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr

import controls as C
from common import (DATA_DIR, EMBEDDINGS, FAMILIES, FAMILY_INDEX, GRID, LINEAR_PDES, MODELS, MODES,
                    NAMES, N_STATES, PDES, PROBLEMS, RESULTS, deck_text, five_frame_file, problem_name,
                    sheet_file)

# Number of values in one stacked solution, used to turn sums of squares into RMS values.
SIZE5 = 5 * 2 * GRID * GRID
SIZE100 = N_STATES * 2 * GRID * GRID


# ---------------------------------------------------------------------------
# Reference quantities
# ---------------------------------------------------------------------------
class Reference:
    """Per-sample terms written by run_simulations.py."""

    def __init__(self, path: Path):
        terms = np.load(path)
        assert list(terms["config_names"]) == NAMES
        self.sq5, self.sq100 = terms["sq5"], terms["sq100"]
        self.cos5, self.cos100 = terms["cos5"], terms["cos100"]
        self.err5, self.tstep5 = terms["err5"], terms["tstep5"]
        self.out_of_range = terms["out_of_range"]

    def D(self, rs, frames: int = 5) -> np.ndarray:
        """54 x 54 RMS distance over the samples rs."""
        sq, size = (self.sq5, SIZE5) if frames == 5 else (self.sq100, SIZE100)
        return np.sqrt(sq[rs].sum(0) / (len(rs) * size))

    def physical_cosine(self, rs, frames: int = 5) -> np.ndarray:
        return (self.cos5 if frames == 5 else self.cos100)[rs].mean(0)

    def eps(self, rs) -> tuple[np.ndarray, np.ndarray]:
        """Error of each problem on the scale of D, and whether it is exact.

        Linear PDEs: distance to the exact grid solution (a true bound).
        Nonlinear PDEs: distance to the half-step run plus twice the step-halving
        difference (an estimate).
        """
        base = np.sqrt(self.err5[rs].sum(0) / (len(rs) * SIZE5))
        step = np.sqrt(self.tstep5[rs].sum(0) / (len(rs) * SIZE5))
        linear = np.array([pde in LINEAR_PDES for pde, _, _ in PROBLEMS])
        return np.where(linear, base, base + 2 * step), linear


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def pde_of(i: int) -> str:
    return PROBLEMS[i][0]


def bc_of(i: int) -> str:
    return PROBLEMS[i][2]


def pairs_within(idx) -> list[tuple[int, int]]:
    return list(itertools.combinations(idx, 2))


def triplets_within(idx) -> list[tuple[int, int, int]]:
    """(anchor, j, k) with j < k, both different from the anchor."""
    out = []
    for i in idx:
        others = [j for j in idx if j != i]
        out.extend((i, j, k) for j, k in itertools.combinations(others, 2))
    return out


def spearman(S: np.ndarray, D: np.ndarray, pairs) -> float:
    """Rank correlation between similarity S and -D over the given pairs."""
    s = np.array([S[a, b] for a, b in pairs])
    d = np.array([D[a, b] for a, b in pairs])
    if np.ptp(s) == 0 or np.ptp(d) == 0:
        return float("nan")
    return float(spearmanr(s, -d).statistic)


def mean_spearman(S: np.ndarray, D: np.ndarray) -> float:
    """Spearman within each IC family, averaged over the families."""
    return float(np.nanmean([spearman(S, D, pairs_within(idx)) for idx in FAMILY_INDEX.values()]))


def mean_rank_agreement(A: np.ndarray, B: np.ndarray, average=np.nanmean) -> float:
    """Spearman between two similarity (or two distance) matrices, per family, averaged."""
    values = []
    for idx in FAMILY_INDEX.values():
        pairs = pairs_within(idx)
        values.append(spearmanr([A[a, b] for a, b in pairs], [B[a, b] for a, b in pairs]).statistic)
    return float(average(values))


def triplet_accuracy(S, D, trips, keep=None) -> tuple[float, int]:
    """Share of triplets where S orders j, k like D; also returns the number used."""
    credit, n = 0.0, 0
    for t, (i, j, k) in enumerate(trips):
        if keep is not None and not keep[t]:
            continue
        dj, dk = D[i, j], D[i, k]
        if dj == dk:
            continue
        near, far = (j, k) if dj < dk else (k, j)
        s_near, s_far = S[i, near], S[i, far]
        credit += 1.0 if s_near > s_far else (0.5 if s_near == s_far else 0.0)
        n += 1
    return (credit / n if n else float("nan")), n


def resolved_mask(D, eps, trips) -> np.ndarray:
    """Triplets whose reference order cannot be flipped by numerical error."""
    return np.array([abs(D[i, k] - D[i, j]) > 2 * eps[i] + eps[j] + eps[k] for i, j, k in trips])


def evaluate(S: np.ndarray, D: np.ndarray, eps=None, exact=None) -> dict:
    """All per-family metrics of one similarity matrix, and their means."""
    out = {"per_family": {}}
    for family, idx in FAMILY_INDEX.items():
        pairs = pairs_within(idx)
        trips = triplets_within(idx)
        same_bc_pairs = [(a, b) for a, b in pairs if bc_of(a) == bc_of(b)]
        changed_bc_pairs = [(a, b) for a, b in pairs if bc_of(a) != bc_of(b)]
        same_bc = np.array([bc_of(j) == bc_of(i) and bc_of(k) == bc_of(i) for i, j, k in trips])
        changed_bc = np.array([bc_of(j) != bc_of(i) and bc_of(k) != bc_of(i) for i, j, k in trips])
        row = {"spearman": spearman(S, D, pairs),
               "spearman_same_bc_pairs": spearman(S, D, same_bc_pairs),
               "spearman_changed_bc_pairs": spearman(S, D, changed_bc_pairs)}
        row["triplet_accuracy"], row["triplets"] = triplet_accuracy(S, D, trips)
        row["triplet_accuracy_same_bc"], _ = triplet_accuracy(S, D, trips, same_bc)
        row["triplet_accuracy_changed_bc"], _ = triplet_accuracy(S, D, trips, changed_bc)
        if eps is not None:
            resolved = resolved_mask(D, eps, trips)
            # "certified": resolved and all three problems have an exact error bound
            certified = resolved & np.array([exact[i] and exact[j] and exact[k] for i, j, k in trips])
            row["resolved_fraction"] = float(resolved.mean())
            row["triplet_accuracy_resolved"], row["resolved_triplets"] = triplet_accuracy(S, D, trips, resolved)
            row["certified_fraction"] = float(certified.mean())
            row["triplet_accuracy_certified"], row["certified_triplets"] = triplet_accuracy(S, D, trips, certified)
        out["per_family"][family] = row
    metric_names = [k for k in out["per_family"][FAMILIES[0]] if not k.endswith("triplets")]
    out["mean"] = {k: float(np.nanmean([out["per_family"][f][k] for f in FAMILIES])) for k in metric_names}
    out["full_54"] = {"spearman": spearman(S, D, pairs_within(range(len(NAMES))))}
    return out


def factor_means(S: np.ndarray) -> dict:
    """Mean similarity of pairs that share the PDE, the BC, or neither (within IC families)."""
    groups = {"same_pde_diff_bc": [], "diff_pde_same_bc": [], "diff_pde_diff_bc": []}
    for idx in FAMILY_INDEX.values():
        for a, b in pairs_within(idx):
            if pde_of(a) == pde_of(b):
                groups["same_pde_diff_bc"].append(S[a, b])
            elif bc_of(a) == bc_of(b):
                groups["diff_pde_same_bc"].append(S[a, b])
            else:
                groups["diff_pde_diff_bc"].append(S[a, b])
    return {k: float(np.mean(v)) for k, v in groups.items()}


def bootstrap(statistic, rs, B=2000, seed=20261119) -> dict:
    """95% percentile interval of statistic(samples) when the samples are resampled."""
    rng = np.random.default_rng(seed)
    values = np.array([statistic(rng.choice(rs, size=len(rs), replace=True)) for _ in range(B)])
    values = values[np.isfinite(values)]
    return {"lo": float(np.quantile(values, 0.025)), "hi": float(np.quantile(values, 0.975)),
            "replicates": int(len(values))}


# ---------------------------------------------------------------------------
# Similarity matrices: models and baselines
# ---------------------------------------------------------------------------
def load_embeddings(folder: Path, model: str):
    path = folder / f"{model}.npz"
    if not path.exists():
        return None
    with np.load(path) as stored:
        return {k: stored[k] for k in stored.files}


def paired_similarity(get, rs) -> np.ndarray:
    """S_ij = mean over samples r of z_i(r) . z_j(r); get(name, r) returns one embedding."""
    Z = np.stack([np.stack([get(name, r) for name in NAMES]) for r in rs])   # [R, 54, dim]
    return np.einsum("rid,rjd->ij", Z, Z) / len(rs)


def embedding_similarity(emb: dict, task: str, mode: str, rs) -> np.ndarray:
    return paired_similarity(lambda name, r: emb[f"{task}|{mode}|{name}|r{r:03d}"], rs)


def encoded_samples(emb: dict, task: str, mode: str, max_r: int = 100) -> list[int]:
    """Samples for which all 54 problems were encoded."""
    return [r for r in range(max_r) if all(f"{task}|{mode}|{name}|r{r:03d}" in emb for name in NAMES)]


def tfidf_similarity(rs) -> np.ndarray:
    """Character 3-5-gram TF-IDF cosine between descriptions of the same sample."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    texts = [deck_text(name, r) for r in rs for name in NAMES]
    X = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), sublinear_tf=True).fit_transform(texts)
    K = len(NAMES)
    S = np.zeros((K, K))
    for i in range(len(rs)):
        block = X[i * K:(i + 1) * K]
        S += (block @ block.T).toarray()
    return S / len(rs)


def pixel_similarity(rs, data: Path) -> np.ndarray:
    """Cosine between the raw RGB pixels of the sheets."""
    S = np.zeros((len(NAMES), len(NAMES)))
    for r in rs:
        P = np.stack([np.asarray(Image.open(sheet_file(data, name, r)), dtype=np.float32).ravel()
                      for name in NAMES])
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        S += P @ P.T
    return S / len(rs)


def fourier_features(arr: np.ndarray, kmax: int = 8) -> np.ndarray:
    """[..., 5, 2, 64, 64] -> magnitudes of the 2D DFT modes with |kx|, ky <= kmax, flattened."""
    F = np.abs(np.fft.rfft2(arr, axes=(-2, -1)))
    kx = np.r_[0:kmax + 1, -kmax:0]
    F = F[..., kx, :][..., :kmax + 1]
    return F.reshape(*arr.shape[:-4], -1)


def fourier_similarity(rs, data: Path) -> np.ndarray:
    """Cosine between low-frequency Fourier magnitudes of the five frames."""
    features = np.stack([fourier_features(np.load(five_frame_file(data, name))[rs]) for name in NAMES],
                        axis=1)                                             # [R, 54, dim]
    features /= np.linalg.norm(features, axis=-1, keepdims=True)
    return np.einsum("rid,rjd->ij", features, features) / len(rs)


# ---------------------------------------------------------------------------
# Sections of analysis.json
# ---------------------------------------------------------------------------
def reference_summary(ref: Reference, rs) -> dict:
    """Spread of D, agreement of D with D100, and how many triplets are resolved."""
    D, D100 = ref.D(rs), ref.D(rs, 100)
    eps, exact = ref.eps(rs)
    per_family = {}
    for family, idx in FAMILY_INDEX.items():
        pairs = pairs_within(idx)
        d = np.array([D[a, b] for a, b in pairs])
        trips = triplets_within(idx)
        resolved = resolved_mask(D, eps, trips)
        certified = resolved & np.array([exact[i] and exact[j] and exact[k] for i, j, k in trips])
        per_family[family] = {
            "D_min": float(d.min()), "D_median": float(np.median(d)), "D_max": float(d.max()),
            "spearman_D5_vs_D100": float(spearmanr(d, [D100[a, b] for a, b in pairs]).statistic),
            "resolved_fraction": float(resolved.mean()), "certified_fraction": float(certified.mean()),
            "unresolved_triplets": int((~resolved).sum()), "triplets": len(trips),
            "eps_max": float(eps[idx].max())}
    return {"per_family": per_family,
            "eps_max_linear_certified": float(eps[exact].max()),
            "eps_max_nonlinear_estimated": float(eps[~exact].max()),
            "spearman_D5_vs_D100_full54": spearman(-D100, D, pairs_within(range(len(NAMES)))),
            "out_of_range_pixels_total": int(ref.out_of_range.sum()),
            "D_matrix_five_frame": D.tolist(),
            "D_matrix_100_frame": D100.tolist()}


def baselines(ref: Reference, rs, data: Path) -> dict:
    """The four model-free similarity baselines on samples rs."""
    D = ref.D(rs)
    eps, exact = ref.eps(rs)
    similarities = {"tfidf_text": tfidf_similarity(rs),
                    "pixels_image": pixel_similarity(rs, data),
                    "fourier_image": fourier_similarity(rs, data),
                    "physical_cosine": ref.physical_cosine(rs)}
    block = {}
    for name, S in similarities.items():
        block[name] = evaluate(S, D, eps, exact)
        block[name]["factor_means"] = factor_means(S)
    # A first-frame image is the same for every PDE and BC, so all scores tie.
    block["first_frame_image"] = {"spearman": "undefined (constant scores)", "triplet_accuracy": 0.5}
    return block


def main_task_results(emb: dict, mode: str, rs, ref: Reference, D_all: np.ndarray, n_boot: int) -> dict:
    """Metrics of one model and input type on the main task."""
    S = embedding_similarity(emb, "main", mode, rs)
    D = ref.D(rs)
    eps, exact = ref.eps(rs)
    row = evaluate(S, D, eps, exact)
    row["realizations"] = rs
    within = [S[a, b] for idx in FAMILY_INDEX.values() for a, b in pairs_within(idx)]
    row["S_offdiag_range_within_families"] = [float(min(within)), float(max(within))]
    row["spearman_vs_D100_mean"] = mean_spearman(S, ref.D(rs, 100))
    row["spearman_vs_D_all100_mean"] = mean_spearman(S, D_all)
    row["spearman_vs_physical_cosine_mean"] = mean_spearman(S, -ref.physical_cosine(rs))
    row["factor_means"] = factor_means(S)
    if n_boot:
        def statistic(sample):
            return mean_spearman(embedding_similarity(emb, "main", mode, sample), ref.D(sample))
        row["mean_spearman_ci95"] = bootstrap(statistic, rs, B=n_boot)
    return row


def time_reversal_control(emb: dict, ref: Reference) -> dict | None:
    rs = encoded_samples(emb, "reversal", "image")
    if not rs:
        return None
    S_rev = embedding_similarity(emb, "reversal", "image", rs)
    S_img = embedding_similarity(emb, "main", "image", rs)
    D = ref.D(rs)
    self_similarity = [emb[f"reversal|image|{n}|r{r:03d}"] @ emb[f"main|image|{n}|r{r:03d}"]
                       for n in NAMES for r in rs]
    return {"realizations": rs,
            "spearman_reversed_vs_ref": mean_spearman(S_rev, D),
            "spearman_original_vs_ref": mean_spearman(S_img, D),
            "spearman_reversed_vs_original": mean_rank_agreement(S_rev, S_img),
            "mean_self_similarity_original_vs_reversed": float(np.mean(self_similarity))}


def notation_control(emb: dict, ref: Reference) -> dict | None:
    """Is a description closer to its own rewrite than to the other 53 problems?"""
    rs = encoded_samples(emb, "notation", "text")
    if not rs:
        return None
    S_rewrite = embedding_similarity(emb, "notation", "text", rs)
    S_text = embedding_similarity(emb, "main", "text", rs)
    D = ref.D(rs)
    top1, self_similarity, nearest_other, closer = [], [], [], []
    for r in rs:
        originals = np.stack([emb[f"main|text|{n}|r{r:03d}"] for n in NAMES])
        for i, name in enumerate(NAMES):
            rewrite = emb[f"notation|text|{name}|r{r:03d}"]
            to_rewrite = originals @ rewrite
            top1.append(int(np.argmax(to_rewrite) == i))
            self_similarity.append(float(to_rewrite[i]))
            others = originals @ originals[i]
            others[i] = -np.inf
            nearest_other.append(float(others.max()))
            # how many of the other 53 problems are closer to the original than its rewrite
            closer.append(int((others > float(originals[i] @ rewrite)).sum()))
    return {"realizations": rs,
            "rewrite_retrieval_top1_among_54": float(np.mean(top1)),
            "mean_similarity_original_vs_rewrite": float(np.mean(self_similarity)),
            "mean_similarity_original_vs_nearest_other_configuration": float(np.mean(nearest_other)),
            "median_other_problems_closer_than_rewrite": float(np.median(closer)),
            "fraction_rewrite_farther_than_all_53_others": float(np.mean(np.array(closer) == len(NAMES) - 1)),
            "spearman_rewritten_vs_ref": mean_spearman(S_rewrite, D),
            "spearman_original_vs_ref": mean_spearman(S_text, D)}


def mismatch_control(emb: dict, ref: Reference) -> dict | None:
    """Image of one PDE + description of the next: which reference does S follow?"""
    rs = encoded_samples(emb, "mismatch", "both")
    if not rs:
        return None
    S_mismatch = embedding_similarity(emb, "mismatch", "both", rs)
    S_both = embedding_similarity(emb, "main", "both", rs)
    D = ref.D(rs)
    partner = [NAMES.index(problem_name(PDES[C.mismatch_partner(PDES.index(pde))], family, bc))
               for pde, family, bc in PROBLEMS]
    D_text = D[np.ix_(partner, partner)]            # reference of the problems in the descriptions
    return {"realizations": rs,
            "spearman_vs_image_reference": mean_spearman(S_mismatch, D),
            "spearman_vs_deck_reference": mean_spearman(S_mismatch, D_text),
            "spearman_matched_both_vs_ref": mean_spearman(S_both, D)}


def first_frame_control(emb: dict) -> dict | None:
    """Encoding the same t = 0 image twice must give the same embedding."""
    keys = [k for k in emb if k.startswith("firstframe|") and k.endswith("copy0")]
    if not keys:
        return None
    diffs = [float(np.abs(emb[k] - emb[k[:-1] + "1"]).max()) for k in keys if k[:-1] + "1" in emb]
    return {"pairs_checked": len(diffs), "max_abs_difference_between_copies": max(diffs) if diffs else None,
            "spearman": "undefined (constant scores within each IC family)", "triplet_accuracy": 0.5}


def heat_control(emb: dict) -> dict | None:
    """Does similarity to nu = 0.01 fall as the heat solutions move apart?"""
    if not any(k.startswith("heat|") for k in emb):
        return None
    anchor = emb[f"heat|text|nu={C.HEAT_ANCHOR:.3f}|"]
    rows = [{"nu": nu, "rms_five_frames": C.heat_rms(C.HEAT_ANCHOR, nu),
             "similarity_to_anchor": float(anchor @ emb[f"heat|text|nu={nu:.3f}|"])}
            for nu in C.HEAT_CANDIDATES]
    sims = [row["similarity_to_anchor"] for row in rows]
    return {"rows": rows, "ordering_preserved": bool(sims[0] > sims[1] > sims[2])}


def precision_check(fp32: dict, bf16: dict, ref: Reference) -> dict:
    """EmbeddingGemma 2 in float32 vs the bfloat16 run used in the study."""
    result = {}
    for mode in MODES:
        rs = encoded_samples(fp32, "main", mode)
        if not rs:
            continue
        keys = [f"main|{mode}|{n}|r{r:03d}" for n in NAMES for r in rs]
        cosines = [float(fp32[k] @ bf16[k]) for k in keys]
        S32 = embedding_similarity(fp32, "main", mode, rs)
        S16 = embedding_similarity(bf16, "main", mode, rs)
        D = ref.D(rs)
        result[mode] = {"realizations": rs, "min_cosine_fp32_bf16": min(cosines),
                        "mean_cosine_fp32_bf16": float(np.mean(cosines)),
                        "spearman_S32_vs_S16_mean": mean_rank_agreement(S32, S16),
                        "spearman_ref_fp32": mean_spearman(S32, D),
                        "spearman_ref_bf16": mean_spearman(S16, D)}
    return result


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="folder written by run_simulations.py")
    parser.add_argument("--reference", type=Path, default=RESULTS / "reference_terms.npz")
    parser.add_argument("--emb-dir", type=Path, default=EMBEDDINGS)
    parser.add_argument("--out", type=Path, default=RESULTS / "analysis.json")
    parser.add_argument("--bootstrap", type=int, default=2000, help="bootstrap replicates (0 = skip)")
    args = parser.parse_args()

    ref = Reference(args.reference)
    all_r = list(range(ref.sq5.shape[0]))
    D_all = ref.D(all_r)
    out: dict = {"realizations_reference": len(all_r),
                 "reference": reference_summary(ref, all_r),
                 "baselines_all100": baselines(ref, all_r, args.data),
                 "models": {}}

    subset = None          # samples encoded by every model and input type
    for model in MODELS:
        emb = load_embeddings(args.emb_dir, model)
        if emb is None:
            continue
        results = {}
        for mode in MODES:
            rs = encoded_samples(emb, "main", mode)
            if not rs:
                continue
            subset = rs if subset is None else sorted(set(subset) & set(rs))
            results[mode] = main_task_results(emb, mode, rs, ref, D_all, args.bootstrap)
        controls = {"time_reversal": time_reversal_control(emb, ref),
                    "notation": notation_control(emb, ref),
                    "mismatch": mismatch_control(emb, ref),
                    "first_frame": first_frame_control(emb),
                    "heat_calibration": heat_control(emb)}
        results["controls"] = {k: v for k, v in controls.items() if v is not None}
        out["models"][model] = results

    fp32, bf16 = load_embeddings(args.emb_dir, "embeddinggemma2_fp32"), load_embeddings(args.emb_dir, "embeddinggemma2")
    if fp32 is not None and bf16 is not None:
        out["precision_diagnostic_embeddinggemma2"] = precision_check(fp32, bf16, ref)

    if subset:
        out["baselines_embedding_subset"] = baselines(ref, subset, args.data)
        out["baselines_embedding_subset"]["realizations"] = subset
        out["reference"]["subset_vs_all100_spearman_mean"] = mean_rank_agreement(ref.D(subset), D_all,
                                                                                average=np.mean)
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
