"""Opt-in checks against the shipped bundle and training tables (ECOLI_ATLAS_REAL_DATA=1, run from the
repository root with atlas_data/ and outputs/models, outputs/atlas present)."""
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from ecoli_atlas.cli import main

ROOT = Path(os.environ.get("BS_DATA_ROOT", "."))
pytestmark = pytest.mark.skipif(not os.environ.get("ECOLI_ATLAS_REAL_DATA"), reason="set ECOLI_ATLAS_REAL_DATA=1")

COUNTS = ROOT / "atlas_data/merged_count_table.txt"
META = ROOT / "atlas_data/metadata_revised.csv"
FEAT = ROOT / "atlas_data/GCF_000005845.2_ASM584v2_feature_table.txt"
MODELS = ROOT / "outputs/models"
ATLAS = ROOT / "outputs/atlas"


def test_figures_reproduce_manifest(tmp_path):
    out = tmp_path / "figs"
    rc = main(["figures", "--counts", str(COUNTS), "--metadata", str(META), "--feature-table", str(FEAT), "--models", str(MODELS),
               "--atlas", str(ATLAS), "--out", str(out), "--device", "cpu", "--no-retrain", "--strict", "--formats", "png",
               "--only", "factor_test_eval,study_leakage"])
    assert rc == 0
    side = json.loads((out / "figure_metrics.json").read_text())
    tm = side["metrics"]["test_metrics"]
    assert (tm["growth_phase"]["n"], tm["medium_type"]["n"], tm["is_lab_strain"]["n"]) == (118, 108, 124)
    assert abs(tm["growth_phase"]["balanced_acc"] - 0.920) < 1e-3 and abs(tm["medium_type"]["balanced_acc"] - 0.979) < 1e-3
    assert abs(tm["is_lab_strain"]["balanced_acc"] - 0.727) < 1e-3
    assert abs(tm["growth_phase"]["auroc"] - 0.978) < 1e-3 and tm["medium_type"]["auroc"] == 1.0 and abs(tm["is_lab_strain"]["auroc"] - 0.738) < 1e-3
    assert side["atlas_comparison"]["max_abs_diff"] < 1e-3


def test_neighbors_oxygen_studies(tmp_path):
    rc = main(["neighbors", "GSE189154", "GSE78756", "--atlas", str(ATLAS), "--models", str(MODELS), "--out", str(tmp_path / "nb")])
    assert rc == 0
    s = pd.read_csv(tmp_path / "nb" / "summary.tsv", sep="\t", index_col=0)
    assert set(s["query_split_bucket"]) == {"test"} and len(s) == 17
