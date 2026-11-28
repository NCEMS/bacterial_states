"""samples.tsv: the LB, mid-exponential, untreated wild-type lab-strain samples of the atlas (the selection of the
fig_batch_tracking panel of the atlas_lab_strain_LB_variation example, `figures.pca.lab_subsets`), with each sample's
mean log2 expression of the ribosomal proteins and of the RegulonDB regulons of common.SETS (whole regulons: activated
and repressed targets), and a few single genes. Needs the atlas tables and the module table (`ecoli_atlas download
modules`).
    python scripts/samples.py [--models DIR]"""
import argparse
import numpy as np, pandas as pd
from ecoli_atlas.cli import build_parser, _figure_context
from ecoli_atlas.figures.pca import lab_subsets
from ecoli_atlas.figures.batch_tracking import RIBOSOME
from ecoli_atlas.modules import read_module_table, whole_regulons
from common import ROOT, FEATURES, SAMPLES, SETS

GENES = ["flhD", "flhC", "fliC", "crp", "cyaA", "rpoS", "fliA"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", default=str(ROOT / "outputs/models"))
    p.add_argument("--threads", default="8")
    a = p.parse_args()
    D = ROOT / "atlas_data"
    argv = ["figures", "--counts", f"{D}/atlas_v2_counts.tsv", "--antisense", f"{D}/atlas_v2_antisense_counts.tsv",
            "--metadata", f"{D}/metadata_v2.tsv", "--qc", f"{D}/sample_qc.tsv", "--features", str(FEATURES),
            "--models", a.models, "--out", "/dev/null", "--threads", a.threads]
    ctx, _, _ = _figure_context(build_parser().parse_args(argv))
    sym = pd.Index([str(x) for x in ctx.gene_names])
    rib = np.where(sym.str.match(RIBOSOME))[0]
    mod = whole_regulons(read_module_table(f"{D}/modules/modules.tsv")).set_index("term")
    groups = {s: np.where(sym.isin(mod.loc[f"regulondb:{s}", "genes"].split(",")))[0] for s in SETS}
    rows = []
    for strain, mask in lab_subsets(ctx).items():
        for i in np.where(mask)[0]:
            rows.append({"gsm": ctx.ids[i], "gse": ctx.gse[i], "strain": strain,
                         "strain_name": ctx.meta["strain_name"].iloc[i] if "strain_name" in ctx.meta else "",
                         "ribosome": ctx.logx[i, rib].mean(),
                         **{s: ctx.logx[i, c].mean() for s, c in groups.items()},
                         **{g: ctx.logx[i, sym.get_loc(g)] for g in GENES if g in sym}})
    pd.DataFrame(rows).to_csv(SAMPLES, sep="\t", index=False, float_format="%.4f")
    print(f"{SAMPLES.name}: {len(rows)} samples")


if __name__ == "__main__":
    main()
