"""Export report assets from the experiment outputs: tables, number macros, figures.

Run after judge_check.py, evaluate.py, compare.py and make_figures.py:
    uv run python scripts/export_report_assets.py

Writes into report/<milestone>/:
    generated/numbers.tex    every number quoted in the text, as LaTeX macros
    generated/*.tex          tables (main results, dAUC, paired comparison, findings by disease, ...)
    figures/*.png            figures copied from outputs/figures
Commit these files: they are what the report compiles from (also on Overleaf).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys

import numpy as np
import pandas as pd

from c21.config import load_config, repo_root
from c21.constants import NIH_LABELS
from c21.data import find_label_csv, index_images, load_labels, read_split
from c21.evaluation import predictions_path, results_dir
from c21.models.resunet import ResUNet, count_parameters
from c21.noise import get_level
from c21.report import (
    PREFIX,
    ci,
    dataset_tex,
    findings_by_disease,
    findings_table_tex,
    integer,
    join_names,
    judge_auc_tex,
    judge_levels_tex,
    macro,
    noise_levels_tex,
    num,
    pct,
    probability_shift,
    signed,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--milestone", default="m2")
    parser.add_argument("--level", help="noise level (default: config primary, medium)")
    parser.add_argument("--config", help="config file (default configs/default.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    level = get_level(cfg, args.level)
    lv = level.name
    outputs = cfg["paths"]["outputs"]
    res = outputs / "results"
    rdir = results_dir(cfg, "test", lv)
    target = repo_root() / "report" / args.milestone
    gen, figs = target / "generated", target / "figures"

    needed = {
        "summary (evaluate.py)": rdir / "summary.csv",
        "paired comparison (compare.py)": rdir / "compare_resunet_vs_bm3d.csv",
        "noise levels (make_noisy_sets.py)": res / "noise_levels_test.csv",
        "judge tables (judge_check.py, rerun after Batch 7)": res / "judge_noise_levels.csv",
        "training log (train.py)": res / f"training_log_{lv}.csv",
        "figures (make_figures.py)": outputs / "figures" / f"psnr_vs_delta_p_{lv}.png",
    }
    missing = [f"  - {k}: {v}" for k, v in needed.items() if not v.exists()]
    if missing:
        print("Missing inputs. Run the scripts named below first:\n" + "\n".join(missing))
        return 1
    gen.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    ccfg = cfg["classifier"]
    thr = ccfg["threshold"]
    m = []  # macro lines

    # --- main results per method -------------------------------------------------
    summary = pd.read_csv(rdir / "summary.csv").set_index("method")
    for meth, p in PREFIX.items():
        if meth not in summary.index:
            continue
        r = summary.loc[meth]
        m += [macro(f"{p}PSNR", num(r["psnr_mean"])), macro(f"{p}PSNRsd", num(r["psnr_std"])),
              macro(f"{p}SSIM", num(r["ssim_mean"], 3)), macro(f"{p}Dp", num(r["delta_p_mean"], 3)),
              macro(f"{p}Agr", num(r["agreement_mean"], 3)), macro(f"{p}Lost", pct(r["lost_findings"])),
              macro(f"{p}New", pct(r["new_findings"])), macro(f"{p}DAUC", signed(r["macro_dauc"])),
              macro(f"{p}DAUCci", ci(r["macro_dauc_ci_low"], r["macro_dauc_ci_high"]))]
        if meth != "noisy":
            m.append(macro(f"{p}Gain", num(r["psnr_mean"] - summary.loc["noisy", "psnr_mean"], 1)))
    bm = json.loads((rdir / "bm3d_summary.json").read_text(encoding="utf-8"))
    ours = json.loads((rdir / "resunet_summary.json").read_text(encoding="utf-8"))
    m += [macro("BMSec", num(bm["seconds_per_image"], 1)),
          macro("BMMinutes", num(bm["seconds_per_image"] * bm["n_images"] / 60, 0)),
          macro("BMHardware", (bm.get("hardware") or f"CPU, {bm.get('workers', 1)} worker(s)").replace("(s)", "s")),
          macro("OursMs", num(ours["seconds_per_image"] * 1000, 1)),
          macro("GPUName", ours.get("hardware", "GPU")),
          macro("Speedup", integer(round(bm["seconds_per_image"] / ours["seconds_per_image"])))]

    # --- paired comparison ResU-Net vs BM3D ----------------------------------------
    cmp_df = pd.read_csv(rdir / "compare_resunet_vs_bm3d.csv").set_index("metric")
    for metric, key, d in [("PSNR (dB)", "PSNR", 2), ("SSIM", "SSIM", 3), ("delta_p", "Dp", 3),
                           ("Agreement", "Agr", 3), ("Macro AUC", "AUC", 3)]:
        r = cmp_df.loc[metric]
        m += [macro(f"Cmp{key}", signed(r["diff"], d)), macro(f"Cmp{key}ci", ci(r["ci_low"], r["ci_high"], d))]
        if not pd.isna(r["a_better_share"]):
            m.append(macro(f"Cmp{key}Share", pct(r["a_better_share"], 0)))
    m += [macro("OursAUC", num(cmp_df.loc["Macro AUC", "a"], 3)), macro("BMAUC", num(cmp_df.loc["Macro AUC", "b"], 3)),
          macro("CmpDpRel", pct(-cmp_df.loc["delta_p", "diff"] / cmp_df.loc["delta_p", "b"], 0))]

    # --- per-disease dAUC (ResU-Net, Nodule) ---------------------------------------
    auc_ours = pd.read_csv(rdir / "resunet_auc.csv").set_index("disease")
    used = [d for d in auc_ours.index if d != "macro"]
    m += [macro("MinPos", str(ccfg["min_positives_for_auc"])), macro("NumAUCDiseases", str(len(used))),
          macro("AUCDiseases", join_names(used))]
    if "Nodule" in auc_ours.index:
        r = auc_ours.loc["Nodule"]
        m += [macro("OursNoduleDAUC", signed(r["delta_auc"])), macro("OursNoduleDAUCci", ci(r["ci_low"], r["ci_high"]))]

    # --- findings analysis from saved predictions ------------------------------------
    p_clean = np.load(predictions_path(cfg, "test", None, "clean"))
    preds = {meth: np.load(predictions_path(cfg, "test", lv, meth)) for meth in ("bm3d", "resunet")}
    fb = findings_by_disease(p_clean, preds, thr)
    fb.to_csv(res / f"findings_by_disease_{lv}.csv", index=False, lineterminator="\n")
    (gen / "findings_by_disease.tex").write_text(findings_table_tex(fb), encoding="utf-8")
    fbi = fb.set_index("disease")
    for disease in ("Nodule", "Mass", "Fibrosis"):
        for meth in ("bm3d", "resunet"):
            for kind in ("lost", "new"):
                m.append(macro(f"{PREFIX[meth]}{kind.capitalize()}{disease}", pct(fbi.loc[disease, f'{meth}_{kind}'], 0)))
    for meth in ("bm3d", "resunet"):
        s = probability_shift(p_clean, preds[meth], thr)
        p = PREFIX[meth]
        m += [macro(f"{p}ShiftAll", signed(s["all"])), macro(f"{p}ShiftFindings", signed(s["findings"])),
              macro(f"{p}ShiftNonFindings", signed(s["non_findings"]))]
    yes = p_clean >= thr
    m += [macro("JudgeFindings", integer(yes.sum())), macro("JudgePairs", integer(yes.size)),
          macro("JudgeFindingsPct", pct(yes.mean()))]

    # --- judge tables (judge_check.py) ---------------------------------------------
    jl = pd.read_csv(res / "judge_noise_levels.csv")
    (gen / "judge_levels.tex").write_text(judge_levels_tex(jl), encoding="utf-8")
    for _, r in jl.iterrows():
        p = r["level"].capitalize()
        m += [macro(f"Judge{p}Dp", num(r["delta_p"], 3)), macro(f"Judge{p}Lost", pct(r["lost_findings"], 0)),
              macro(f"Judge{p}DAUC", signed(r["macro_dauc"]))]
    ja = pd.read_csv(res / "judge_clean_auc.csv")
    (gen / "judge_auc.tex").write_text(judge_auc_tex(ja), encoding="utf-8")
    m.append(macro("JudgeMedianAUC", num(ja["auc_clean"].median(), 3)))

    # --- noise levels -----------------------------------------------------------------
    nl = pd.read_csv(res / "noise_levels_test.csv")
    (gen / "noise_levels.tex").write_text(noise_levels_tex(nl), encoding="utf-8")
    for _, r in nl.iterrows():
        m.append(macro(f"Noise{r['level'].capitalize()}PSNR", num(r["psnr_mean"])))

    # --- training --------------------------------------------------------------------
    log = pd.read_csv(res / f"training_log_{lv}.csv")
    best = log.loc[log["val_psnr"].idxmax()]
    m += [macro("TrainEpochs", str(len(log))), macro("BestEpoch", str(int(best["epoch"]))),
          macro("ValPSNR", num(best["val_psnr"])), macro("TrainMinutes", num(log["seconds"].sum() / 60, 1)),
          macro("SecPerEpoch", num(log["seconds"].median(), 1)), macro("Params", integer(count_parameters(ResUNet())))]

    # --- dataset -------------------------------------------------------------------------
    split = read_split(cfg)
    (gen / "dataset.tex").write_text(dataset_tex(split, used), encoding="utf-8")
    for s in ("train", "val", "test"):
        part = split[split["split"] == s]
        m += [macro(f"N{s.capitalize()}", integer(len(part))),
              macro(f"Pat{s.capitalize()}", integer(part["patient_id"].nunique())),
              macro(f"NoFind{s.capitalize()}", pct(part["no_finding"].mean()))]
    m += [macro("NPatients", integer(split["patient_id"].nunique())), macro("Seed", str(cfg["seed"])),
          macro("MaxPerPatient", str(cfg["data"]["max_images_per_patient"]))]
    raw = cfg["paths"]["raw_data"]
    if raw.exists():
        labels = load_labels(find_label_csv(raw))
        available = set(index_images(raw)) & set(labels["image"])
        m += [macro("NAvailable", integer(len(available))), macro("NoFindFull", pct(labels["no_finding"].mean())),
              macro("NFull", integer(len(labels))), macro("PatFull", integer(labels["patient_id"].nunique()))]
    else:
        print(f"Note: {raw} not found; dataset-size macros set to 'n/a'.")
        m += [macro(k, "n/a") for k in ("NAvailable", "NoFindFull", "NFull", "PatFull")]

    # --- copy generated tables and figures -----------------------------------------------
    for name, dest in [("summary.tex", "summary.tex"), ("dauc.tex", "dauc.tex"),
                       ("compare_resunet_vs_bm3d.tex", "compare.tex")]:
        shutil.copyfile(rdir / name, gen / dest)
    fig_src = outputs / "figures"
    copied = []
    for name in ["noise_levels.png", f"training_curves_{lv}.png", f"psnr_vs_delta_p_{lv}.png",
                 f"qualitative_{lv}_1_typical.png", f"qualitative_{lv}_2_small_finding.png",
                 f"qualitative_{lv}_3_worst_case.png"]:
        if (fig_src / name).exists():
            shutil.copyfile(fig_src / name, figs / name)
            copied.append(name)

    header = ["% Generated by scripts/export_report_assets.py. Do not edit by hand: rerun the script.",
              f"% Noise level: {lv}; seed {cfg['seed']}."]
    (gen / "numbers.tex").write_text("\n".join(header + m) + "\n", encoding="utf-8")
    labels_missing = [lbl for lbl in NIH_LABELS if lbl not in set(fb["disease"])]
    assert not labels_missing

    print(f"Wrote {len(m)} number macros to {gen / 'numbers.tex'}")
    print(f"Tables: {', '.join(sorted(p.name for p in gen.glob('*.tex') if p.name != 'numbers.tex'))}")
    print(f"Figures: {', '.join(copied)}")
    print(f"\nNow compile report/{args.milestone}/main.tex (pdflatex, bibtex, pdflatex, pdflatex) or upload "
          f"report/ to Overleaf.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
