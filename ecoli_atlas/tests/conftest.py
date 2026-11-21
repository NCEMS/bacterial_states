"""Synthetic atlas tables: a small counting annotation, counts with a phase and a medium signal,
metadata in the metadata_v2 layout and a QC table in the sample_qc layout."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

N_CDS, N_STUDIES, PER_STUDY = 80, 12, 10


@pytest.fixture(scope="session")
def feature_table(tmp_path_factory):
    rows = [dict(Feature_ID=f"CDS_b{i:04d}_g{i:03d}", Symbol=f"g{i:03d}", Locus_Tag=f"b{i:04d}", Type="CDS", Source="NCBI",
                 Length=900 + i) for i in range(N_CDS)]
    rows += [dict(Feature_ID="pseudo_b9000_yzzZ", Symbol="yzzZ", Locus_Tag="b9000", Type="pseudo", Source="NCBI", Length=500),
             dict(Feature_ID="ncRNA_b9100_ssrA", Symbol="ssrA", Locus_Tag="b9100", Type="ncRNA", Source="NCBI", Length=363),
             dict(Feature_ID="ncRNA_RF00001", Symbol="RF00001", Locus_Tag="", Type="ncRNA", Source="Rfam", Length=120),
             dict(Feature_ID="rRNA_b9201_rrsA", Symbol="rrsA", Locus_Tag="b9201", Type="rRNA", Source="NCBI", Length=1542),
             dict(Feature_ID="rRNA_b9202_rrlA", Symbol="rrlA", Locus_Tag="b9202", Type="rRNA", Source="NCBI", Length=2904),
             dict(Feature_ID="rRNA_b9203_rrfA", Symbol="rrfA", Locus_Tag="b9203", Type="rRNA", Source="NCBI", Length=120),
             dict(Feature_ID="tRNA_b9301_ileV", Symbol="ileV", Locus_Tag="b9301", Type="tRNA", Source="NCBI", Length=77)]
    path = tmp_path_factory.mktemp("annotation") / "features.tsv"
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
    return path


@pytest.fixture(scope="session")
def atlas_files(tmp_path_factory, feature_table):
    """Paths of counts, metadata and QC tables for N_STUDIES studies. Phase raises the first ten
    genes, a defined medium the next ten; every study has its own depth."""
    rng = np.random.RandomState(0)
    feat = pd.read_csv(feature_table, sep="\t", keep_default_na=False)
    base = rng.gamma(2.0, 200.0, size=len(feat))
    counts, meta, qc = {}, [], []
    for s in range(N_STUDIES):
        for j in range(PER_STUDY):
            gsm = f"GSM{s:02d}{j:02d}"
            stationary = j >= PER_STUDY // 2 if s % 3 == 0 else s % 3 == 2
            defined = s % 2 == 1
            mu = base * (0.5 + s / N_STUDIES)
            mu[:10] *= 6.0 if stationary else 1.0
            mu[10:20] *= 5.0 if defined else 1.0
            counts[gsm] = rng.negative_binomial(5, 5 / (5 + mu))
            phase = "stationary" if stationary else "mid_exponential"
            if s == 6:
                phase = "late_stationary" if stationary else "transition"
            treated = "true" if (s == 7 and j % 2) else "false"
            meta.append(dict(GSM=gsm, GSE=f"GSE{s:02d}", growth_phase=phase, growth_phase_basis="inferred" if s == 9 else "stated", culture_mode="batch_shaken",
                             base_medium="M9" if defined else "LB", medium_class=("minimal_plus_hydrolysate" if s == 10 else "defined_minimal" if defined else "complex"),
                             amino_acids_supplied="false" if defined else "true", carbon_source="glucose", oxygen="aerobic",
                             growth_temperature="37", genotype_class="wild_type" if j % 2 == 0 else "deletion",
                             perturbed_genes="" if j % 2 == 0 else "g001", inducer_added="false",
                             treatment="heat" if treated == "true" else "none", exposure="", is_control="true" if j % 2 == 0 else "false",
                             is_wild_type="true" if j % 2 == 0 else "false",
                             is_reference_genotype="true" if j % 2 == 0 else "false", phase2="" if (treated == "true" or s == 6 and not stationary)
                             else ("stationary" if stationary else "exponential"), planktonic_batch="true",
                             medium_type=("" if s == 10 else "minimal" if defined else "rich"), treated=treated, standard_rnaseq="true",
                             stress_family="temperature" if treated == "true" else "none", strain_group=("UPEC" if s % 4 == 3 else "" if s == 8 else "K-12"), usable="true"))
            paired = s % 2 == 0
            qc.append(dict(GSM=gsm, GSE=f"GSE{s:02d}", layout="PAIRED" if paired else "SINGLE", instrument_model="Illumina NovaSeq 6000",
                           in_model="" if s == N_STUDIES - 1 and j == 0 else "atlas_test", full_layout="PE" if paired else "SE",
                           full_length_q05_r1=40, full_length_q50_r1=75, full_length_q95_r1=75, full_cycles_r1=75, full_pct_kept=95.0,
                           full_pct_adapter_trimmed=10.0 + s, full_duplication_rate=0.3, full_gc_content=0.51,
                           full_fraction_forward=0.02, full_fraction_reverse=0.97, full_length_q50_r2=75 if paired else "",
                           full_fragment_q25=150 if paired else "", full_fragment_q50=200 if paired else "",
                           full_fragment_q75=260 if paired else "", full_dna_fraction_est=0.0 if j == 1 else 0.004,
                           full_leakage_b=0.004, full_intergenic_share=0.05, full_gene_body_slope=-0.1 * (s % 4),
                           full_pct_short_features=2.0, full_pct_top_other_genus_nostable=0.5,
                           full_substitution_rate=0.003 * (1 + s % 3) if (s, j) != (5, 2) else "",
                           full_gc_slope=-0.1 * (s % 4) + 0.01 * j, full_gc_curv=-0.15 - 0.01 * (s % 3)))
    d = tmp_path_factory.mktemp("atlas")
    table = pd.DataFrame(counts, index=feat["Feature_ID"])
    table.index.name = "Feature_ID"
    table.to_csv(d / "counts.tsv", sep="\t")
    pd.DataFrame(meta).to_csv(d / "metadata.tsv", sep="\t", index=False)
    pd.DataFrame(qc).to_csv(d / "qc.tsv", sep="\t", index=False)
    return {"counts": d / "counts.tsv", "metadata": d / "metadata.tsv", "qc": d / "qc.tsv", "features": feature_table}


@pytest.fixture(scope="session")
def train_args(atlas_files):
    return ["--counts", str(atlas_files["counts"]), "--metadata", str(atlas_files["metadata"]), "--qc", str(atlas_files["qc"]),
            "--features", str(atlas_files["features"])]


@pytest.fixture(scope="session")
def trained(tmp_path_factory, train_args):
    """A bundle trained for a few epochs through the CLI."""
    from ecoli_atlas.cli import main
    out = tmp_path_factory.mktemp("bundle")
    assert main(["train", *train_args, "--out", str(out), "--epochs", "25", "--device", "cpu", "--test-frac", "0.2"]) == 0
    return out
