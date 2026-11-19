#!/usr/bin/env python3
"""Write the paper's LaTeX tables and number macros from the result files.

Every number in the output is read from benchmark/*.json or
benchmark/results/*; nothing is typed by hand.

Output (default: sections/generated/, next to the paper source):
  tab_validation.tex  tab_main.tex  tab_secondary.tex  tab_perfamily.tex
  tab_factors.tex  tab_controls.tex  tab_model_inputs.tex  macros.tex
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from common import BENCHMARK, EMBEDDINGS, FAMILY_INDEX, PROBLEMS, REPO

MODELS = [("embeddinggemma2", "EmbeddingGemma 2"), ("qwen3vl2b", "Qwen3-VL-Emb.-2B"), ("gme2b", "GME-Qwen2-VL-2B")]
SHORT = {"embeddinggemma2": "Gem", "qwen3vl2b": "Qw", "gme2b": "Gme"}     # macro name parts
MODES = ["text", "image", "both"]
MODE_LABEL = {"text": "text", "image": "image", "both": "joint"}
MODE_MACRO = {"text": "Text", "image": "Image", "both": "Both"}
FAMILY_LABELS = [("single_gaussian", "Bump"), ("low_frequency", "Oscill."), ("patch_mixture", "Patches")]
PDE_LABELS = {"diffusion": "Diffusion", "advection_diffusion": "Advection--diffusion",
              "linear_exchange": "Linear exchange", "fisher_kpp": "Fisher--KPP",
              "shifted_allen_cahn": "Allen--Cahn (shifted)", "gray_scott": "Gray--Scott"}
BASELINES = [("tfidf_text", "Tfidf"), ("pixels_image", "Pix"), ("fourier_image", "Four"), ("physical_cosine", "Phys")]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def load(name: str):
    path = BENCHMARK / name
    return json.loads(path.read_text()) if path.exists() else None


def f2(x, digits=2) -> str:
    """Fixed-point number with a LaTeX minus sign; '--' for missing values."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:.{digits}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def sci(x, digits=1) -> str:
    """Scientific notation for math mode, e.g. 3.2\\times10^{-5}."""
    if x == 0:
        return "0"
    exponent = int(math.floor(math.log10(abs(x))))
    mantissa = x / 10**exponent
    if round(mantissa, digits) >= 10:
        mantissa /= 10
        exponent += 1
    return f"{mantissa:.{digits}f}\\times10^{{{exponent}}}"


def macro(name: str, value: str) -> str:
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def table(environment: str, placement: str, caption: str, label: str, head: str, body: str,
          columns: str, preamble: str = "", resize: bool = False) -> str:
    """A booktabs table. `head` and `body` are complete LaTeX rows."""
    begin_tabular = f"\\begin{{tabular}}{{{columns}}}"
    end_tabular = r"\end{tabular}"
    if resize:
        begin_tabular = "\\resizebox{\\columnwidth}{!}{%\n" + begin_tabular
        end_tabular += "}"
    return (f"\\begin{{{environment}}}[{placement}]\n\\centering\n\\caption{{{caption}}}\n\\label{{{label}}}\n"
            f"\\small\n{preamble}{begin_tabular}\n\\toprule\n{head}\n\\midrule\n{body}\n\\bottomrule\n"
            f"{end_tabular}\n\\end{{{environment}}}\n")


def model_rows(an):
    """(key, label, mode, result) for every model and input type that was run."""
    for key, label in MODELS:
        results = an["models"].get(key)
        if not results:
            continue
        for mode in MODES:
            if results.get(mode):
                yield key, label, mode, results[mode]


def first_label(label: str, mode: str) -> str:
    """Show the model name only on its first (text) row."""
    return label if mode == "text" else ""


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def validation_table(val, expo) -> str:
    linear = val["linear_vs_exact_semidiscrete"]
    refined = val["time_refinement_vs_substeps_16"]
    space = val["spatial_refinement_five_frames"]
    rows = []
    for pde, label in PDE_LABELS.items():
        def worst(results, field):
            return max(v[field] for k, v in results.items() if k.startswith(pde + "/"))
        if pde in ("diffusion", "advection_diffusion", "linear_exchange"):
            t = max(v["substeps_4"]["max_abs"] for k, v in linear.items() if k.startswith(pde + "/"))
            time_cell = f"${sci(t)}$ (exact)"
        else:
            time_cell = f"${sci(worst(refined, 'substeps_4_max_abs'))}$ (refined)"
        exponax_cell = "--"
        if expo:
            values = [v["rms_fd_minus_spectral_n64_128_256"][0] for k, v in expo["rows"].items()
                      if k.startswith(pde + "/")]
            if values:
                exponax_cell = f"${sci(max(values))}$"
        rows.append(f"{label} & {time_cell} & ${sci(worst(space, 'rms_64_vs_256'))}$ & "
                    f"${sci(worst(space, 'rms_128_vs_256'))}$ & {exponax_cell} \\\\")
    caption = (r"Solver checks on pilot samples. Time error: largest absolute error with four steps per output "
               r"interval, measured against the exact grid solution (linear equations) or against a run with four "
               r"times as many steps (nonlinear equations). Space: largest five-frame RMS difference from the "
               r"$256\times256$ solution after averaging to the coarser grid. Exponax: largest five-frame RMS "
               r"difference from an independent spectral solver at $64\times64$ (periodic BC only).")
    head = r"Equation & Time error & Space, $64^2$ & Space, $128^2$ & Exponax, $64^2$ \\"
    return table("table", "h", caption, "tab:validation", head, "\n".join(rows), "lcccc")


def main_table(an) -> str:
    model_lines = []
    for key, label in MODELS:
        results = an["models"].get(key)
        if not results:
            continue
        cells = []
        for mode in MODES:
            r = results.get(mode)
            if not r:
                cells.append("--")
                continue
            cell = f2(r["mean"]["spearman"])
            ci = r.get("mean_spearman_ci95")
            if ci:
                cell += f" {{\\scriptsize[{f2(ci['lo'])}, {f2(ci['hi'])}]}}"
            cells.append(cell)
        model_lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    subset = an.get("baselines_embedding_subset", {})
    all100 = an["baselines_all100"]
    baseline_lines = []
    for label, key, column in [("Char.\\ TF--IDF (text)", "tfidf_text", 0),
                               ("Rendered pixels (image)", "pixels_image", 1),
                               ("Fourier magnitudes (image)", "fourier_image", 1)]:
        cells = ["--", "--", "--"]
        cells[column] = f2(subset.get(key, {}).get("mean", {}).get("spearman"))
        if key in all100:
            cells[column] += f" {{\\scriptsize({f2(all100[key]['mean']['spearman'])})}}"
        baseline_lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    baseline_lines.append(r"Initial frame only (image) & -- & undefined & -- \\")

    caption = (r"Main result: Spearman correlation between model similarity $S^{\mathrm{emb}}$ and $-D$, averaged "
               r"over the three IC families (153 pairs each). Samples 0--9, with the reference computed on the same "
               r"samples. Brackets: 95\% bootstrap intervals. Baseline values in parentheses use all 100 samples.")
    body = "\n".join(model_lines) + "\n\\midrule\n" + "\n".join(baseline_lines)
    return table("table", "t", caption, "tab:main", r"Model & Text & Image & Joint \\", body, "lccc",
                 preamble="\\setlength{\\tabcolsep}{3.5pt}\n", resize=True)


def secondary_table(an) -> str:
    lines = []
    for _, label, mode, r in model_rows(an):
        m = r["mean"]
        lines.append(f"{first_label(label, mode)} & {MODE_LABEL[mode]} & {f2(m['triplet_accuracy'])} & "
                     f"{f2(m['triplet_accuracy_resolved'])} & {f2(m['triplet_accuracy_certified'])} & "
                     f"{f2(m['spearman_same_bc_pairs'])} & {f2(m['spearman_changed_bc_pairs'])} & "
                     f"{f2(r['spearman_vs_D100_mean'])} & {f2(r['full_54']['spearman'])} \\\\")
    subset = an.get("baselines_embedding_subset", {})
    for label, key in [("TF--IDF", "tfidf_text"), ("Pixels", "pixels_image"), ("Fourier", "fourier_image"),
                       ("Phys.\\ cosine", "physical_cosine")]:
        r = subset.get(key)
        if not r:
            continue
        m = r["mean"]
        lines.append(f"{label} & -- & {f2(m['triplet_accuracy'])} & {f2(m['triplet_accuracy_resolved'])} & "
                     f"{f2(m['triplet_accuracy_certified'])} & {f2(m['spearman_same_bc_pairs'])} & "
                     f"{f2(m['spearman_changed_bc_pairs'])} & -- & {f2(r['full_54']['spearman'])} \\\\")
    caption = (r"Further results on samples 0--9, averaged over IC families. Triplet accuracy for all triplets, for "
               r"triplets whose order is guaranteed despite numerical error (``reliable''), and for triplets made "
               r"only of linear equations, where the guarantee is exact. Spearman correlation for pairs with the same "
               r"BC (45 per family) and with different BCs (108 per family), against the 100-frame reference, and "
               r"over all 1,431 pairs of the 54 problems.")
    head = (r"& & \multicolumn{3}{c}{Triplet accuracy} & \multicolumn{2}{c}{Spearman by BC} & "
            r"\multicolumn{2}{c}{Spearman, other references} \\" + "\n"
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}" + "\n"
            r"Model & Input & All & Reliable & Linear & Same BC & Diff.\ BC & 100 frames & All 54 \\")
    return table("table*", "t", caption, "tab:secondary", head, "\n".join(lines), "llccccccc")


def per_family_table(an) -> str:
    lines = []
    for _, label, mode, r in model_rows(an):
        cells = [f2(r["per_family"][family]["spearman"]) for family, _ in FAMILY_LABELS]
        lo, hi = r["S_offdiag_range_within_families"]
        lines.append(f"{first_label(label, mode)} & {MODE_LABEL[mode]} & " + " & ".join(cells) +
                     f" & {lo:.3f}--{hi:.3f} \\\\")
    caption = (r"Spearman correlation with $-D$ for each IC family (samples 0--9), and the range of similarities "
               r"between different problems in the same family.")
    head = "Model & Input & " + " & ".join(label for _, label in FAMILY_LABELS) + r" & $S^{\mathrm{emb}}$ range \\"
    return table("table", "h", caption, "tab:perfamily", head, "\n".join(lines), "llcccc")


def reference_factor_means(D: np.ndarray) -> dict[str, float]:
    """Mean reference distance of pairs that share the PDE, the BC, or neither."""
    groups = {"same_pde": [], "same_bc": [], "neither": []}
    for idx in FAMILY_INDEX.values():
        for i, j in itertools.combinations(idx, 2):
            if PROBLEMS[i][0] == PROBLEMS[j][0]:
                groups["same_pde"].append(D[i, j])
            elif PROBLEMS[i][2] == PROBLEMS[j][2]:
                groups["same_bc"].append(D[i, j])
            else:
                groups["neither"].append(D[i, j])
    return {k: np.mean(v) for k, v in groups.items()}


def factor_table(an) -> str:
    lines = []
    for _, label, mode, r in model_rows(an):
        fm = r["factor_means"]
        lines.append(f"{first_label(label, mode)} & {MODE_LABEL[mode]} & {fm['same_pde_diff_bc']:.4f} & "
                     f"{fm['diff_pde_same_bc']:.4f} & {fm['diff_pde_diff_bc']:.4f} \\\\")
    ref = reference_factor_means(np.array(an["reference"]["D_matrix_five_frame"]))
    lines.append(r"\midrule" + "\n" + f"Reference $D$ & -- & {ref['same_pde']:.4f} & {ref['same_bc']:.4f} & "
                 f"{ref['neither']:.4f} \\\\")
    caption = (r"Exploratory (not planned in advance): average similarity between problems in the same IC family, "
               r"grouped by what they share: the same equation with a different BC (9 pairs per family), the same BC "
               r"with a different equation (45), or neither (99). The last row gives the average reference distance "
               r"for the same groups (smaller means more similar).")
    head = r"Model & Input & Same eq. & Same BC & Neither \\"
    return table("table", "h", caption, "tab:factors", head, "\n".join(lines), "llccc")


def controls_table(an) -> str:
    lines = []
    for key, label in MODELS:
        results = an["models"].get(key)
        if not results:
            continue
        c = results.get("controls", {})
        rev, nota, mis, heat = (c.get(k) for k in ("time_reversal", "notation", "mismatch", "heat_calibration"))
        cells = [
            f2(rev["spearman_original_vs_ref"]) + "/" + f2(rev["spearman_reversed_vs_ref"]) if rev else "--",
            f2(rev["mean_self_similarity_original_vs_reversed"], 3) if rev else "--",
            f2(nota["rewrite_retrieval_top1_among_54"]) if nota else "--",
            (f2(nota["mean_similarity_original_vs_rewrite"], 3) + "/"
             + f2(nota["mean_similarity_original_vs_nearest_other_configuration"], 3)) if nota else "--",
            f2(mis["spearman_vs_image_reference"]) + "/" + f2(mis["spearman_vs_deck_reference"]) if mis else "--",
            ("yes" if heat["ordering_preserved"] else "no") if heat else "--",
        ]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    caption = (r"Controls on samples 0--2 (heat test: four short descriptions). Time order: Spearman correlation with "
               r"$-D$ for original/reversed images, and the average similarity between an image and its reversed "
               r"copy. Notation: how often a rewritten description finds its own original among the 54 descriptions "
               r"of its sample (top-1), and the average similarity of an original to its rewrite/to its most similar "
               r"other problem. Mismatch: Spearman correlation of joint embeddings (image of one problem, description "
               r"of the next equation) with the image's reference/the description's reference. Heat: whether "
               r"$S(0.01,0.011)>S(0.01,0.02)>S(0.01,0.05)$ with text input.")
    head = (r"& \multicolumn{2}{c}{Time order (image)} & \multicolumn{2}{c}{Notation (text)} & Mismatch (joint) & "
            r"Heat \\" + "\n" r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}" + "\n"
            r"Model & Orig./rev. & Self-sim. & Top-1 & Rewrite/other & Image/text ref. & order \\")
    return table("table*", "t", caption, "tab:controls", head, "\n".join(lines), "lcccccc")


def model_input_table(emb_dir: Path) -> tuple[str | None, str]:
    """Table of model inputs from the embedding metadata, and the failure-count macros."""
    image_sizes = {"embeddinggemma2": "280 tok.", "qwen3vl2b": "$704\\times288$", "gme2b": "$728\\times308$"}
    dims = {"embeddinggemma2": 768, "qwen3vl2b": 2048, "gme2b": 1536}
    rows, macros = [], ""
    for key, label in MODELS:
        path = emb_dir / f"{key}.meta.json"
        if not path.exists():
            continue
        meta = json.loads(path.read_text())
        records = meta.get("records", {})
        text_tokens = [v["tokens"] for k, v in records.items() if k.startswith("main|text|") and "tokens" in v]
        seconds = {m: [v["seconds"] for k, v in records.items() if k.startswith(f"main|{m}|")] for m in MODES}
        tokens = f"{min(text_tokens)}--{max(text_tokens)}" if text_tokens else "--"
        timing = "/".join(f"{np.mean(seconds[m]):.1f}" if seconds[m] else "--" for m in MODES)
        rows.append(f"{label} & {dims[key]:,} & {image_sizes[key]} & {tokens} & {timing} \\\\")
        macros += macro("fail" + SHORT[key], str(len(meta.get("failures", {}))))
    if not rows:
        return None, macros
    caption = (r"The three models and their inputs. Image: size after the model's own resizing (EmbeddingGemma 2 "
               r"always uses 280 image tokens). Text tokens include the model's prompt template (sample 0 of all 54 "
               r"problems). Last column: average CPU seconds per text/image/joint input (bfloat16, two cores).")
    head = r"Model & Dim. & Image & Text tokens & s/input \\"
    text = table("table", "t", caption, "tab:model-inputs", head, "\n".join(rows), "lcccc",
                 preamble="\\setlength{\\tabcolsep}{3pt}\n", resize=True)
    return text, macros


# ---------------------------------------------------------------------------
# Macros for numbers quoted in the text
# ---------------------------------------------------------------------------
def solver_macros(val, expo, pilot, sim) -> str:
    out = ""
    if val:
        orders = [o for v in val["manufactured_continuum"].values() for o in v["observed_order"]]
        out += macro("valOrderMin", f"{min(orders):.3f}") + macro("valOrderMax", f"{max(orders):.3f}")
    if expo:
        out += macro("expoRatio", f"{expo['median_ratio_64_128']:.2f}")
        out += macro("expoMax", f"\\ensuremath{{{sci(expo['max_rms_difference_n64'])}}}")
    if pilot:
        refinement = pilot["P7_spatial_refinement_of_distances"].values()
        out += macro("pilotSpearmanMin", f"{min(v['spearman_D64_vs_D128'] for v in refinement):.5f}")
        out += macro("pilotFlips", str(sum(v["triplet_order_flips"] for v in refinement)))
        out += macro("pilotTriplets", f"{sum(v['triplets'] for v in refinement):,}")
        out += macro("pilotMaxU", f"{pilot['P6_pilot_channel_maximum'][0]:.3f}")
        out += macro("pilotMaxV", f"{pilot['P6_pilot_channel_maximum'][1]:.3f}")
    if sim:
        out += macro("simSeconds", f"{sim['elapsed_seconds'] / 60:.0f}")
        out += macro("simOOR", f"{sim['contact_sheet']['out_of_range_pixels_total']:,}")
        lows = [d["min"] for d in sim["diagnostics"].values()]
        highs = [d["max"] for d in sim["diagnostics"].values()]
        out += macro("simMin", f"\\ensuremath{{{sci(min(lows))}}}") + macro("simMax", f"{max(highs):.3f}")
    return out


def reference_macros(ref) -> str:
    families = ref["per_family"].values()
    resolved = [v["resolved_fraction"] for v in families]
    certified = [v["certified_fraction"] for v in families]
    out = macro("refEpsLinear", f"\\ensuremath{{{sci(ref['eps_max_linear_certified'])}}}")
    out += macro("refEpsNonlinear", f"\\ensuremath{{{sci(ref['eps_max_nonlinear_estimated'])}}}")
    out += macro("refResolvedMin", f"{100 * min(resolved):.1f}")
    out += macro("refCertified", f"{100 * sum(certified) / len(certified):.1f}")
    out += macro("refUnresolved", str(sum(v["unresolved_triplets"] for v in families)))
    out += macro("refDFiveHundred", f"{min(v['spearman_D5_vs_D100'] for v in families):.3f}")
    if "subset_vs_all100_spearman_mean" in ref:
        out += macro("refSubsetVsAll", f"{ref['subset_vs_all100_spearman_mean']:.3f}")
    out += macro("refDmin", f"\\ensuremath{{{sci(min(v['D_min'] for v in families))}}}")
    out += macro("refDmax", f"{max(v['D_max'] for v in families):.3f}")
    return out


def model_macros(an) -> str:
    out = ""
    for key, _, mode, r in model_rows(an):                        # headline numbers first
        name = SHORT[key] + MODE_MACRO[mode]
        out += macro(f"sp{name}", f2(r["mean"]["spearman"]))
        out += macro(f"tr{name}", f2(r["mean"]["triplet_accuracy"]))
    for key, short in SHORT.items():
        results = an["models"].get(key)
        if not results:
            continue
        for mode in MODES:
            r = results.get(mode)
            if not r:
                continue
            name, m = short + MODE_MACRO[mode], r["mean"]
            out += macro(f"sp{name}Same", f2(m["spearman_same_bc_pairs"]))
            out += macro(f"sp{name}Ch", f2(m["spearman_changed_bc_pairs"]))
            out += macro(f"sp{name}Hund", f2(r["spearman_vs_D100_mean"]))
            out += macro(f"sp{name}Full", f2(r["full_54"]["spearman"]))
            out += macro(f"sp{name}Phys", f2(r["spearman_vs_physical_cosine_mean"]))
            out += macro(f"tr{name}Cert", f2(m["triplet_accuracy_certified"]))
            ci = r.get("mean_spearman_ci95")
            if ci:
                out += macro(f"ci{name}", f"[{f2(ci['lo'])}, {f2(ci['hi'])}]")
            out += macro(f"rangeLo{name}", f"{r['S_offdiag_range_within_families'][0]:.4f}")
            out += macro(f"rangeHi{name}", f"{r['S_offdiag_range_within_families'][1]:.4f}")
            fm = r["factor_means"]
            out += macro(f"fac{name}", f"{fm['same_pde_diff_bc']:.3f}/{fm['diff_pde_same_bc']:.3f}/"
                                       f"{fm['diff_pde_diff_bc']:.3f}")
        out += control_macros(results.get("controls", {}), short)
    for mode in MODES:                                            # ranges over the three models
        rows = [an["models"][k][mode]["mean"] for k in SHORT if k in an["models"] and mode in an["models"][k]]
        if rows:
            sp, tr = [m["spearman"] for m in rows], [m["triplet_accuracy"] for m in rows]
            name = MODE_MACRO[mode]
            out += macro(f"sp{name}Min", f2(min(sp))) + macro(f"sp{name}Max", f2(max(sp)))
            out += macro(f"tr{name}Min", f2(min(tr))) + macro(f"tr{name}Max", f2(max(tr)))
    return out


def control_macros(c: dict, short: str) -> str:
    out = ""
    if "time_reversal" in c:
        t = c["time_reversal"]
        out += macro(f"revOrig{short}", f2(t["spearman_original_vs_ref"]))
        out += macro(f"revRev{short}", f2(t["spearman_reversed_vs_ref"]))
        out += macro(f"revSelf{short}", f"{t['mean_self_similarity_original_vs_reversed']:.4f}")
        out += macro(f"revCorr{short}", f2(t["spearman_reversed_vs_original"]))
    if "notation" in c:
        t = c["notation"]
        out += macro(f"notTop{short}", f2(t["rewrite_retrieval_top1_among_54"]))
        out += macro(f"notSelf{short}", f"{t['mean_similarity_original_vs_rewrite']:.3f}")
        out += macro(f"notOther{short}", f"{t['mean_similarity_original_vs_nearest_other_configuration']:.3f}")
        out += macro(f"notSp{short}", f2(t["spearman_rewritten_vs_ref"]))
        out += macro(f"notCloser{short}", f"{t['median_other_problems_closer_than_rewrite']:.0f}")
        out += macro(f"notAll{short}", f"{100 * t['fraction_rewrite_farther_than_all_53_others']:.0f}\\%")
        out += macro(f"notSpOrig{short}", f2(t["spearman_original_vs_ref"]))
    if "mismatch" in c:
        t = c["mismatch"]
        out += macro(f"misImg{short}", f2(t["spearman_vs_image_reference"]))
        out += macro(f"misDeck{short}", f2(t["spearman_vs_deck_reference"]))
        out += macro(f"misBoth{short}", f2(t["spearman_matched_both_vs_ref"]))
    if "heat_calibration" in c:
        t = c["heat_calibration"]
        out += macro(f"heat{short}", "preserved" if t["ordering_preserved"] else "not preserved")
        out += macro(f"heatVals{short}", "/".join(f"{row['similarity_to_anchor']:.4f}" for row in t["rows"]))
    if "first_frame" in c:
        out += macro(f"ffDiff{short}", f"{c['first_frame']['max_abs_difference_between_copies']:.1g}")
    return out


def baseline_macros(an) -> str:
    out = ""
    subset, all100 = an.get("baselines_embedding_subset", {}), an["baselines_all100"]
    for key, short in BASELINES:
        if key in subset:
            m = subset[key]["mean"]
            out += macro(f"sp{short}", f2(m["spearman"]))
            out += macro(f"tr{short}", f2(m["triplet_accuracy"]))
            out += macro(f"sp{short}Same", f2(m["spearman_same_bc_pairs"]))
            out += macro(f"sp{short}Full", f2(subset[key]["full_54"]["spearman"]))
        if key in all100:
            out += macro(f"sp{short}All", f2(all100[key]["mean"]["spearman"]))
    return out


def precision_macros(an) -> str:
    out = ""
    for mode in MODES:
        p = an["precision_diagnostic_embeddinggemma2"].get(mode)
        if p:
            name = MODE_MACRO[mode]
            out += macro(f"prec{name}MinCos", f"{p['min_cosine_fp32_bf16']:.5f}")
            out += macro(f"prec{name}SS", f"{p['spearman_S32_vs_S16_mean']:.3f}")
            out += macro(f"prec{name}RefA", f2(p["spearman_ref_fp32"]))
            out += macro(f"prec{name}RefB", f2(p["spearman_ref_bf16"]))
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=REPO / "sections" / "generated")
    args = parser.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    val = load("simulator_validation.json")
    pilot = load("pilot_report.json")
    expo = load("exponax_crosscheck.json")
    sim = load("results/simulation_manifest.json")
    an = load("results/analysis.json")

    if val:
        (out / "tab_validation.tex").write_text(validation_table(val, expo))
    macros = solver_macros(val, expo, pilot, sim)
    if an:
        macros += reference_macros(an["reference"])
        (out / "tab_main.tex").write_text(main_table(an))
        (out / "tab_secondary.tex").write_text(secondary_table(an))
        (out / "tab_perfamily.tex").write_text(per_family_table(an))
        (out / "tab_factors.tex").write_text(factor_table(an))
        (out / "tab_controls.tex").write_text(controls_table(an))
        macros += model_macros(an) + baseline_macros(an)
    inputs_table, failure_macros = model_input_table(EMBEDDINGS)
    if inputs_table:
        (out / "tab_model_inputs.tex").write_text(inputs_table)
    macros += failure_macros
    if an and "precision_diagnostic_embeddinggemma2" in an:
        macros += precision_macros(an)
    (out / "macros.tex").write_text(macros)
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    main()
