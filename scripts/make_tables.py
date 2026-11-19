#!/usr/bin/env python3
"""Generate LaTeX tables and number macros for the manuscript from result files.

Every number in sections/generated/*.tex is read from benchmark/*.json or
benchmark/results/*.json; nothing is typed by hand.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
B = ROOT / "benchmark"
OUT = ROOT / "sections" / "generated"
MODELS = [("embeddinggemma2", "EmbeddingGemma 2"), ("qwen3vl2b", "Qwen3-VL-Emb.-2B"), ("gme2b", "GME-Qwen2-VL-2B")]
MODES = ["text", "image", "both"]
MODE_LABEL = {"text": "text", "image": "image", "both": "joint"}
FAMS = [("single_gaussian", "Bump"), ("low_frequency", "Oscill."), ("patch_mixture", "Patches")]


def load(name):
    p = B / name
    return json.loads(p.read_text()) if p.exists() else None


def f2(x, nd=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    s = f"{x:.{nd}f}"
    return s.replace("-", "$-$") if s.startswith("-") else s


def sci(x, nd=1):
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10**e
    if round(m, nd) >= 10:
        m /= 10
        e += 1
    return f"{m:.{nd}f}\\times10^{{{e}}}"


def macro(name, value):
    return f"\\newcommand{{\\{name}}}{{{value}}}\n"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    macros = ""
    val = load("simulator_validation.json")
    pilot = load("pilot_report.json")
    expo = load("exponax_crosscheck.json")
    sim = load("results/simulation_manifest.json")
    an = load("results/analysis.json")

    # ---------------- validation table ----------------
    if val:
        lin = val["linear_vs_exact_semidiscrete"]
        tref = val["time_refinement_vs_substeps_16"]
        sp = val["spatial_refinement_five_frames"]
        rows = []
        names = {"diffusion": "Diffusion", "advection_diffusion": "Advection--diffusion",
                 "linear_exchange": "Linear exchange", "fisher_kpp": "Fisher--KPP",
                 "shifted_allen_cahn": "Allen--Cahn (shifted)", "gray_scott": "Gray--Scott"}
        for pde, label in names.items():
            if pde in ("diffusion", "advection_diffusion", "linear_exchange"):
                t = max(v["substeps_4"]["max_abs"] for k, v in lin.items() if k.startswith(pde + "/"))
                tcol = f"${sci(t)}$ (exact)"
            else:
                t = max(v["substeps_4_max_abs"] for k, v in tref.items() if k.startswith(pde + "/"))
                tcol = f"${sci(t)}$ (refined)"
            s64 = max(v["rms_64_vs_256"] for k, v in sp.items() if k.startswith(pde + "/"))
            s128 = max(v["rms_128_vs_256"] for k, v in sp.items() if k.startswith(pde + "/"))
            ex = "--"
            if expo:
                vals = [v["rms_fd_minus_spectral_n64_128_256"][0] for k, v in expo["rows"].items() if k.startswith(pde + "/")]
                if vals:
                    ex = f"${sci(max(vals))}$"
            rows.append(f"{label} & {tcol} & ${sci(s64)}$ & ${sci(s128)}$ & {ex} \\\\")
        table = (r"""\begin{table}[h]
\centering
\caption{Solver checks on pilot samples. Time error: largest absolute error with four steps per output interval, measured against the exact grid solution (linear equations) or against a run with four times as many steps (nonlinear equations). Space: largest five-frame RMS difference from the $256\times256$ solution after averaging to the coarser grid. Exponax: largest five-frame RMS difference from an independent spectral solver at $64\times64$ (periodic BC only).}
\label{tab:validation}
\small
\begin{tabular}{lcccc}
\toprule
Equation & Time error & Space, $64^2$ & Space, $128^2$ & Exponax, $64^2$ \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
        (OUT / "tab_validation.tex").write_text(table)
        orders = [o for v in val["manufactured_continuum"].values() for o in v["observed_order"]]
        macros += macro("valOrderMin", f"{min(orders):.3f}") + macro("valOrderMax", f"{max(orders):.3f}")
    if expo:
        macros += macro("expoRatio", f"{expo['median_ratio_64_128']:.2f}")
        macros += macro("expoMax", f"\\ensuremath{{{sci(expo['max_rms_difference_n64'])}}}")
    if pilot:
        ref = pilot["P7_spatial_refinement_of_distances"]
        macros += macro("pilotSpearmanMin", f"{min(v['spearman_D64_vs_D128'] for v in ref.values()):.5f}")
        macros += macro("pilotFlips", str(sum(v["triplet_order_flips"] for v in ref.values())))
        macros += macro("pilotTriplets", f"{sum(v['triplets'] for v in ref.values()):,}")
        macros += macro("pilotMaxU", f"{pilot['P6_pilot_channel_maximum'][0]:.3f}")
        macros += macro("pilotMaxV", f"{pilot['P6_pilot_channel_maximum'][1]:.3f}")
    if sim:
        macros += macro("simSeconds", f"{sim['elapsed_seconds'] / 60:.0f}")
        macros += macro("simOOR", f"{sim['contact_sheet']['out_of_range_pixels_total']:,}")
        mins = [d["min"] for d in sim["diagnostics"].values()]
        maxs = [d["max"] for d in sim["diagnostics"].values()]
        macros += macro("simMin", f"\\ensuremath{{{sci(min(mins))}}}") + macro("simMax", f"{max(maxs):.3f}")

    if an:
        ref = an["reference"]
        macros += macro("refEpsLinear", f"\\ensuremath{{{sci(ref['eps_max_linear_certified'])}}}")
        macros += macro("refEpsNonlinear", f"\\ensuremath{{{sci(ref['eps_max_nonlinear_estimated'])}}}")
        res = [v["resolved_fraction"] for v in ref["per_family"].values()]
        cert = [v["certified_fraction"] for v in ref["per_family"].values()]
        macros += macro("refResolvedMin", f"{100 * min(res):.1f}")
        macros += macro("refCertified", f"{100 * sum(cert) / len(cert):.1f}")
        macros += macro("refUnresolved", str(sum(v["unresolved_triplets"] for v in ref["per_family"].values())))
        macros += macro("refDFiveHundred", f"{min(v['spearman_D5_vs_D100'] for v in ref['per_family'].values()):.3f}")
        if "subset_vs_all100_spearman_mean" in ref:
            macros += macro("refSubsetVsAll", f"{ref['subset_vs_all100_spearman_mean']:.3f}")
        dmin = min(v["D_min"] for v in ref["per_family"].values())
        dmax = max(v["D_max"] for v in ref["per_family"].values())
        macros += macro("refDmin", f"\\ensuremath{{{sci(dmin)}}}") + macro("refDmax", f"{dmax:.3f}")

        # ---------------- main table ----------------
        lines = []
        for key, label in MODELS:
            m = an["models"].get(key)
            if not m:
                continue
            cells = []
            for mode in MODES:
                r = m.get(mode)
                if not r:
                    cells.append("--")
                    continue
                ci = r.get("mean_spearman_ci95")
                cell = f2(r["mean"]["spearman"])
                if ci:
                    cell += f" {{\\scriptsize[{f2(ci['lo'])}, {f2(ci['hi'])}]}}"
                cells.append(cell)
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        bs = an.get("baselines_embedding_subset", {})
        ba = an["baselines_all100"]
        blines = [
            ("Char.\\ TF--IDF (text)", "tfidf_text", 0),
            ("Rendered pixels (image)", "pixels_image", 1),
            ("Fourier magnitudes (image)", "fourier_image", 1),
        ]
        for label, key, col in blines:
            cells = ["--", "--", "--"]
            v = bs.get(key, {}).get("mean", {}).get("spearman")
            cells[col] = f2(v) + (f" {{\\scriptsize({f2(ba[key]['mean']['spearman'])})}}" if key in ba else "")
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        lines.append(r"Initial frame only (image) & -- & undefined & -- \\")
        table = (r"""\begin{table}[t]
\centering
\caption{Main result: Spearman correlation between model similarity $S^{\mathrm{emb}}$ and $-D$, averaged over the three IC families (153 pairs each). Samples 0--9, with the reference computed on the same samples. Brackets: 95\% bootstrap intervals. Baseline values in parentheses use all 100 samples.}
\label{tab:main}
\small
\setlength{\tabcolsep}{3.5pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lccc}
\toprule
Model & Text & Image & Joint \\
\midrule
""" + "\n".join(lines[:len([1 for k, _ in MODELS if an['models'].get(k)])]) + "\n\\midrule\n" +
                 "\n".join(lines[len([1 for k, _ in MODELS if an['models'].get(k)]):]) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
        (OUT / "tab_main.tex").write_text(table)

        # ---------------- secondary table: triplets, BC breakdown ----------------
        lines = []
        for key, label in MODELS:
            m = an["models"].get(key)
            if not m:
                continue
            for mode in MODES:
                r = m.get(mode)
                if not r:
                    continue
                mm = r["mean"]
                lines.append(f"{label if mode == 'text' else ''} & {MODE_LABEL[mode]} & {f2(mm['triplet_accuracy'])} & "
                             f"{f2(mm['triplet_accuracy_resolved'])} & {f2(mm['triplet_accuracy_certified'])} & "
                             f"{f2(mm['spearman_same_bc_pairs'])} & {f2(mm['spearman_changed_bc_pairs'])} & "
                             f"{f2(r['spearman_vs_D100_mean'])} & {f2(r['full_54']['spearman'])} \\\\")
        for label, key in [("TF--IDF", "tfidf_text"), ("Pixels", "pixels_image"), ("Fourier", "fourier_image"),
                           ("Phys.\\ cosine", "physical_cosine")]:
            r = bs.get(key)
            if not r:
                continue
            mm = r["mean"]
            lines.append(f"{label} & -- & {f2(mm['triplet_accuracy'])} & {f2(mm['triplet_accuracy_resolved'])} & "
                         f"{f2(mm['triplet_accuracy_certified'])} & {f2(mm['spearman_same_bc_pairs'])} & "
                         f"{f2(mm['spearman_changed_bc_pairs'])} & -- & {f2(r['full_54']['spearman'])} \\\\")
        table = (r"""\begin{table*}[t]
\centering
\caption{Further results on samples 0--9, averaged over IC families. Triplet accuracy for all triplets, for triplets whose order is guaranteed despite numerical error (``reliable''), and for triplets made only of linear equations, where the guarantee is exact. Spearman correlation for pairs with the same BC (45 per family) and with different BCs (108 per family), against the 100-frame reference, and over all 1,431 pairs of the 54 problems.}
\label{tab:secondary}
\small
\begin{tabular}{llccccccc}
\toprule
& & \multicolumn{3}{c}{Triplet accuracy} & \multicolumn{2}{c}{Spearman by BC} & \multicolumn{2}{c}{Spearman, other references} \\
\cmidrule(lr){3-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}
Model & Input & All & Reliable & Linear & Same BC & Diff.\ BC & 100 frames & All 54 \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
        (OUT / "tab_secondary.tex").write_text(table)

        # ---------------- per-family table ----------------
        lines = []
        for key, label in MODELS:
            m = an["models"].get(key)
            if not m:
                continue
            for mode in MODES:
                r = m.get(mode)
                if not r:
                    continue
                cells = [f2(r["per_family"][f]["spearman"]) for f, _ in FAMS]
                lo, hi = r["S_offdiag_range_within_families"]
                lines.append(f"{label if mode == 'text' else ''} & {MODE_LABEL[mode]} & " + " & ".join(cells) +
                             f" & {lo:.3f}--{hi:.3f} \\\\")
        table = (r"""\begin{table}[h]
\centering
\caption{Spearman correlation with $-D$ for each IC family (samples 0--9), and the range of similarities between different problems in the same family.}
\label{tab:perfamily}
\small
\begin{tabular}{llcccc}
\toprule
Model & Input & """ + " & ".join(l for _, l in FAMS) + r""" & $S^{\mathrm{emb}}$ range \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
        (OUT / "tab_perfamily.tex").write_text(table)

        # ---------------- factor table (exploratory) ----------------
        lines = []
        for key, label in MODELS:
            m = an["models"].get(key)
            if not m:
                continue
            for mode in MODES:
                r = m.get(mode)
                if not r:
                    continue
                fm = r["factor_means"]
                lines.append(f"{label if mode == 'text' else ''} & {MODE_LABEL[mode]} & {fm['same_pde_diff_bc']:.4f} & "
                             f"{fm['diff_pde_same_bc']:.4f} & {fm['diff_pde_diff_bc']:.4f} \\\\")
        # reference: mean D by factor
        import numpy as np
        D = np.array(ref["D_matrix_five_frame"])
        names = None
        sim_names = sim["configurations"] if sim else None
        if sim_names:
            conf = [n.split("__") for n in sim_names]
            g = {"a": [], "b": [], "c": []}
            for fam in [f for f, _ in FAMS]:
                idx = [i for i, c in enumerate(conf) if c[1] == fam]
                for x in range(len(idx)):
                    for y in range(x + 1, len(idx)):
                        i, j = idx[x], idx[y]
                        k = "a" if conf[i][0] == conf[j][0] else ("b" if conf[i][2] == conf[j][2] else "c")
                        g[k].append(D[i, j])
            lines.append(r"\midrule" + "\n" + f"Reference $D$ & -- & {np.mean(g['a']):.4f} & {np.mean(g['b']):.4f} & {np.mean(g['c']):.4f} \\\\")
        table = (r"""\begin{table}[h]
\centering
\caption{Exploratory (not planned in advance): average similarity between problems in the same IC family, grouped by what they share: the same equation with a different BC (9 pairs per family), the same BC with a different equation (45), or neither (99). The last row gives the average reference distance for the same groups (smaller means more similar).}
\label{tab:factors}
\small
\begin{tabular}{llccc}
\toprule
Model & Input & Same eq. & Same BC & Neither \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
        (OUT / "tab_factors.tex").write_text(table)

        # ---------------- controls table ----------------
        lines = []
        for key, label in MODELS:
            m = an["models"].get(key)
            if not m:
                continue
            c = m.get("controls", {})
            tr, no, mi, he = c.get("time_reversal"), c.get("notation"), c.get("mismatch"), c.get("heat_calibration")
            cells = [
                f2(tr["spearman_original_vs_ref"]) + "/" + f2(tr["spearman_reversed_vs_ref"]) if tr else "--",
                f2(tr["mean_self_similarity_original_vs_reversed"], 3) if tr else "--",
                f2(no["rewrite_retrieval_top1_among_54"]) if no else "--",
                f2(no["mean_similarity_original_vs_rewrite"], 3) + "/" + f2(no["mean_similarity_original_vs_nearest_other_configuration"], 3) if no else "--",
                f2(mi["spearman_vs_image_reference"]) + "/" + f2(mi["spearman_vs_deck_reference"]) if mi else "--",
                ("yes" if he["ordering_preserved"] else "no") if he else "--",
            ]
            lines.append(f"{label} & " + " & ".join(cells) + r" \\")
        table = (r"""\begin{table*}[t]
\centering
\caption{Controls on samples 0--2 (heat test: four short descriptions). Time order: Spearman correlation with $-D$ for original/reversed images, and the average similarity between an image and its reversed copy. Notation: how often a rewritten description finds its own original among the 54 descriptions of its sample (top-1), and the average similarity of an original to its rewrite/to its most similar other problem. Mismatch: Spearman correlation of joint embeddings (image of one problem, description of the next equation) with the image's reference/the description's reference. Heat: whether $S(0.01,0.011)>S(0.01,0.02)>S(0.01,0.05)$ with text input.}
\label{tab:controls}
\small
\begin{tabular}{lcccccc}
\toprule
& \multicolumn{2}{c}{Time order (image)} & \multicolumn{2}{c}{Notation (text)} & Mismatch (joint) & Heat \\
\cmidrule(lr){2-3}\cmidrule(lr){4-5}
Model & Orig./rev. & Self-sim. & Top-1 & Rewrite/other & Image/text ref. & order \\
\midrule
""" + "\n".join(lines) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")
        (OUT / "tab_controls.tex").write_text(table)

        # numbers used inline
        for key, short in [("embeddinggemma2", "Gem"), ("qwen3vl2b", "Qw"), ("gme2b", "Gme")]:
            m = an["models"].get(key)
            if not m:
                continue
            for mode, ms in [("text", "Text"), ("image", "Image"), ("both", "Both")]:
                r = m.get(mode)
                if r:
                    macros += macro(f"sp{short}{ms}", f2(r["mean"]["spearman"]))
                    macros += macro(f"tr{short}{ms}", f2(r["mean"]["triplet_accuracy"]))
        short_of = {"embeddinggemma2": "Gem", "qwen3vl2b": "Qw", "gme2b": "Gme"}
        for key, short in short_of.items():
            m = an["models"].get(key)
            if not m:
                continue
            for mode, ms in [("text", "Text"), ("image", "Image"), ("both", "Both")]:
                r = m.get(mode)
                if not r:
                    continue
                mm = r["mean"]
                macros += macro(f"sp{short}{ms}Same", f2(mm["spearman_same_bc_pairs"]))
                macros += macro(f"sp{short}{ms}Ch", f2(mm["spearman_changed_bc_pairs"]))
                macros += macro(f"sp{short}{ms}Hund", f2(r["spearman_vs_D100_mean"]))
                macros += macro(f"sp{short}{ms}Full", f2(r["full_54"]["spearman"]))
                macros += macro(f"sp{short}{ms}Phys", f2(r["spearman_vs_physical_cosine_mean"]))
                macros += macro(f"tr{short}{ms}Cert", f2(mm["triplet_accuracy_certified"]))
                ci = r.get("mean_spearman_ci95")
                if ci:
                    macros += macro(f"ci{short}{ms}", f"[{f2(ci['lo'])}, {f2(ci['hi'])}]")
                macros += macro(f"rangeLo{short}{ms}", f"{r['S_offdiag_range_within_families'][0]:.4f}")
                macros += macro(f"rangeHi{short}{ms}", f"{r['S_offdiag_range_within_families'][1]:.4f}")
                fm = r["factor_means"]
                macros += macro(f"fac{short}{ms}", f"{fm['same_pde_diff_bc']:.3f}/{fm['diff_pde_same_bc']:.3f}/{fm['diff_pde_diff_bc']:.3f}")
            c = m.get("controls", {})
            if "time_reversal" in c:
                t = c["time_reversal"]
                macros += macro(f"revOrig{short}", f2(t["spearman_original_vs_ref"]))
                macros += macro(f"revRev{short}", f2(t["spearman_reversed_vs_ref"]))
                macros += macro(f"revSelf{short}", f"{t['mean_self_similarity_original_vs_reversed']:.4f}")
                macros += macro(f"revCorr{short}", f2(t["spearman_reversed_vs_original"]))
            if "notation" in c:
                t = c["notation"]
                macros += macro(f"notTop{short}", f2(t["rewrite_retrieval_top1_among_54"]))
                macros += macro(f"notSelf{short}", f"{t['mean_similarity_original_vs_rewrite']:.3f}")
                macros += macro(f"notOther{short}", f"{t['mean_similarity_original_vs_nearest_other_configuration']:.3f}")
                macros += macro(f"notSp{short}", f2(t["spearman_rewritten_vs_ref"]))
                macros += macro(f"notCloser{short}", f"{t['median_other_problems_closer_than_rewrite']:.0f}")
                macros += macro(f"notAll{short}", f"{100 * t['fraction_rewrite_farther_than_all_53_others']:.0f}\\%")
                macros += macro(f"notSpOrig{short}", f2(t["spearman_original_vs_ref"]))
            if "mismatch" in c:
                t = c["mismatch"]
                macros += macro(f"misImg{short}", f2(t["spearman_vs_image_reference"]))
                macros += macro(f"misDeck{short}", f2(t["spearman_vs_deck_reference"]))
                macros += macro(f"misBoth{short}", f2(t["spearman_matched_both_vs_ref"]))
            if "heat_calibration" in c:
                t = c["heat_calibration"]
                macros += macro(f"heat{short}", "preserved" if t["ordering_preserved"] else "not preserved")
                macros += macro(f"heatVals{short}", "/".join(f"{row['similarity_to_anchor']:.4f}" for row in t["rows"]))
            if "first_frame" in c:
                macros += macro(f"ffDiff{short}", f"{c['first_frame']['max_abs_difference_between_copies']:.1g}")
        for mode, ms in [("text", "Text"), ("image", "Image"), ("both", "Both")]:
            vals = [an["models"][k][mode]["mean"]["spearman"] for k in short_of if k in an["models"] and mode in an["models"][k]]
            trs = [an["models"][k][mode]["mean"]["triplet_accuracy"] for k in short_of if k in an["models"] and mode in an["models"][k]]
            if vals:
                macros += macro(f"sp{ms}Min", f2(min(vals))) + macro(f"sp{ms}Max", f2(max(vals)))
                macros += macro(f"tr{ms}Min", f2(min(trs))) + macro(f"tr{ms}Max", f2(max(trs)))
        for key, short in [("tfidf_text", "Tfidf"), ("pixels_image", "Pix"), ("fourier_image", "Four"),
                           ("physical_cosine", "Phys")]:
            if key in bs:
                macros += macro(f"sp{short}", f2(bs[key]["mean"]["spearman"]))
                macros += macro(f"tr{short}", f2(bs[key]["mean"]["triplet_accuracy"]))
                macros += macro(f"sp{short}Same", f2(bs[key]["mean"]["spearman_same_bc_pairs"]))
                macros += macro(f"sp{short}Full", f2(bs[key]["full_54"]["spearman"]))
            if key in ba:
                macros += macro(f"sp{short}All", f2(ba[key]["mean"]["spearman"]))
    # ---------------- model input table (from embedding metadata) ----------------
    rows = []
    sizes = {"embeddinggemma2": "280 tok.", "qwen3vl2b": "$704\\times288$", "gme2b": "$728\\times308$"}
    import numpy as np
    for key, label in MODELS:
        mp = B / "results" / "embeddings" / f"{key}.meta.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text())
        rec = meta.get("records", {})
        tok = {m: [v["tokens"] for k, v in rec.items() if k.startswith(f"main|{m}|") and "tokens" in v] for m in MODES}
        sec = {m: [v["seconds"] for k, v in rec.items() if k.startswith(f"main|{m}|")] for m in MODES}
        dims = {"embeddinggemma2": 768, "qwen3vl2b": 2048, "gme2b": 1536}[key]
        tt = f"{min(tok['text'])}--{max(tok['text'])}" if tok["text"] else "--"
        tb = f"{min(tok['both'])}--{max(tok['both'])}" if tok["both"] else "--"
        secs = "/".join(f"{np.mean(sec[m]):.1f}" if sec[m] else "--" for m in MODES)
        rows.append(f"{label} & {dims:,} & {sizes[key]} & {tt} & {secs} \\\\")
        macros += macro("fail" + {"embeddinggemma2": "Gem", "qwen3vl2b": "Qw", "gme2b": "Gme"}[key], str(len(meta.get("failures", {}))))
    if rows:
        table = (r"""\begin{table}[t]
\centering
\caption{The three models and their inputs. Image: size after the model's own resizing (EmbeddingGemma 2 always uses 280 image tokens). Text tokens include the model's prompt template (sample 0 of all 54 problems). Last column: average CPU seconds per text/image/joint input (bfloat16, two cores).}
\label{tab:model-inputs}
\small
\setlength{\tabcolsep}{3pt}
\resizebox{\columnwidth}{!}{%
\begin{tabular}{lcccc}
\toprule
Model & Dim. & Image & Text tokens & s/input \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}}
\end{table}
""")
        (OUT / "tab_model_inputs.tex").write_text(table)
    if an and "precision_diagnostic_embeddinggemma2" in an:
        for mode, ms in [("text", "Text"), ("image", "Image"), ("both", "Both")]:
            p_ = an["precision_diagnostic_embeddinggemma2"].get(mode)
            if p_:
                macros += macro(f"prec{ms}MinCos", f"{p_['min_cosine_fp32_bf16']:.5f}")
                macros += macro(f"prec{ms}SS", f"{p_['spearman_S32_vs_S16_mean']:.3f}")
                macros += macro(f"prec{ms}RefA", f2(p_["spearman_ref_fp32"]))
                macros += macro(f"prec{ms}RefB", f2(p_["spearman_ref_bf16"]))
    (OUT / "macros.tex").write_text(macros)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
