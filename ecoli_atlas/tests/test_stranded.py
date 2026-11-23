"""Only strand-specific libraries are embedded or trained on (`data.check_stranded`)."""
import pandas as pd
import pytest

from ecoli_atlas.cli import main
from ecoli_atlas.data import check_stranded, load_tables


def test_check_stranded():
    qc = pd.DataFrame({"fraction_forward": [0.02, 0.50, 0.85, None], "fraction_reverse": [0.97, 0.49, 0.10, None],
                       "layout": "PE"}, index=["rev", "unstranded", "fwd", "unknown"])
    check_stranded(qc, ["rev", "fwd"])
    with pytest.raises(ValueError, match="not strand-specific.*unstranded"):
        check_stranded(qc, ["rev", "unstranded"])
    with pytest.raises(ValueError, match="without strand fractions.*unknown"):
        check_stranded(qc, ["unknown"])
    with pytest.raises(ValueError, match="no QC table"):
        check_stranded(None, ["rev"])


def _unstranded_qc(atlas_files, tmp_path):
    qc = pd.read_csv(atlas_files["qc"], sep="\t", index_col=0, keep_default_na=False)
    bad = qc.index[qc["in_model"] != ""][0]
    qc.loc[bad, ["full_fraction_forward", "full_fraction_reverse"]] = [0.51, 0.48]
    path = tmp_path / "qc_unstranded.tsv"
    qc.to_csv(path, sep="\t")
    return path, bad


def test_training_and_embedding_refuse_an_unstranded_library(trained, train_args, atlas_files, tmp_path):
    path, bad = _unstranded_qc(atlas_files, tmp_path)
    with pytest.raises(ValueError, match=f"not strand-specific.*{bad}"):
        load_tables(atlas_files["counts"], atlas_files["metadata"], path)
    args = list(train_args)
    args[args.index("--qc") + 1] = str(path)
    for cmd in (["train", *args, "--out", str(tmp_path / "bundle"), "--epochs", "1", "--device", "cpu"],
                ["embed", "--counts", str(atlas_files["counts"]), "--qc", str(path), "--models", str(trained),
                 "--out", str(tmp_path / "emb"), "--device", "cpu"]):
        try:
            assert main(cmd) != 0                                     # refused with an error code ...
        except ValueError as e:                                       # ... or with the error itself
            assert "strand-specific" in str(e)
    assert not (tmp_path / "bundle" / "model.pt").exists() and not (tmp_path / "emb" / "embedding.tsv").exists()
