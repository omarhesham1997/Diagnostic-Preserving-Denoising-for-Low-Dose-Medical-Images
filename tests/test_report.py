"""Batch 7 tests: report helpers (number formatting, findings analysis, generated tables)."""

import numpy as np
import pandas as pd
import pytest

from c21.constants import NIH_LABELS
from c21.report import (
    ci,
    dataset_tex,
    findings_by_disease,
    findings_table_tex,
    integer,
    join_names,
    macro,
    pct,
    probability_shift,
    signed,
)


def test_number_formatting():
    assert signed(0.0166) == "$+0.017$" and signed(-0.02, 2) == "$-0.02$"
    assert ci(-0.005, 0.036) == "[$-0.005$, $+0.036$]"
    assert pct(0.2848) == "28.5\\%" and pct(0.52, 0) == "52\\%"
    assert integer(449489) == "449{,}489"
    assert join_names(["A", "B", "Pleural_Thickening"]) == "A, B and Pleural Thickening"


def test_macro_names_must_be_letters_only():
    assert macro("OursPSNR", "34.47") == "\\newcommand{\\OursPSNR}{34.47}"
    with pytest.raises(ValueError):
        macro("Noise2PSNR", "1")


def test_findings_by_disease_counts_lost_and_new():
    p_clean = np.full((4, 14), 0.2)
    p_clean[:2, 0] = 0.8  # two clean findings for disease 0
    out = p_clean.copy()
    out[0, 0] = 0.3  # one of them lost
    out[3, 1] = 0.9  # one new finding for disease 1 (out of 4 non-findings)
    df = findings_by_disease(p_clean, {"m": out}).set_index("disease")
    assert df.loc[NIH_LABELS[0], "clean_findings"] == 2
    assert df.loc[NIH_LABELS[0], "m_lost"] == pytest.approx(0.5)
    assert df.loc[NIH_LABELS[1], "m_new"] == pytest.approx(0.25)


def test_probability_shift_signs():
    p_clean = np.array([[0.8, 0.2]])
    s = probability_shift(p_clean, p_clean + np.array([[-0.1, 0.3]]))
    assert s["findings"] == pytest.approx(-0.1) and s["non_findings"] == pytest.approx(0.3)
    assert s["all"] == pytest.approx(0.1)


def test_findings_table_bolds_the_lower_value():
    df = pd.DataFrame([{"disease": "Nodule", "clean_findings": 10, "bm3d_lost": 0.04, "resunet_lost": 0.52,
                        "bm3d_new": 0.52, "resunet_new": 0.02}])
    tex = findings_table_tex(df)
    assert "\\textbf{4\\%} & 52\\%" in tex and "52\\% & \\textbf{2\\%}" in tex


def test_dataset_table_has_total_row():
    split = pd.DataFrame({"split": ["train", "train", "val", "test"], "patient_id": [1, 1, 2, 3],
                          "no_finding": [1, 0, 1, 0], "Nodule": [0, 1, 0, 1]})
    tex = dataset_tex(split, ["Nodule"])
    assert "\\textit{Total} & 4 & 3" in tex and tex.count("\\\\") == 5
