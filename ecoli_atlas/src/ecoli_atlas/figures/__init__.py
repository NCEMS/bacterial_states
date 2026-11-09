"""Publication figures of the atlas paper, regenerated from a saved model bundle and the training
tables (``ecoli_atlas figures``). One function per notebook figure cell; see ``run.run_figures``."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FigSpec:
    name: str
    outputs: tuple          # file stems / tables written into the output directory
    doc: str
    needs_retrain: bool = False


# Order matters: factor_test_eval validates the data path against the manifest early on.
FIGURES = {
    "corpus_composition": FigSpec("corpus_composition", ("fig_corpus_composition",), "pies of the corpus labels (cell 11)"),
    "factor_test_eval": FigSpec("factor_test_eval", ("fig_factor_test_eval", "growth_phase_misclassified.tsv"),
                                "TEST confusion matrices, balanced accuracy, AUROC (cells 29, 35)"),
    "split_balance": FigSpec("split_balance", ("fig_split_balance",), "class balance per fold and TEST (cell 14)"),
    "pca_batch": FigSpec("pca_batch", ("fig_pca_mg1655_batch", "fig_pca_bw_batch", "fig_pca_strain_all_complex", "pca_logcpm_coords.csv"),
                         "PCA of log1p(CPM) within matched subsets (cell 16)"),
    "batch_genes": FigSpec("batch_genes", ("fig_batch_genes", "batch_genes.tsv"), "most study-variable genes (cell 17)"),
    "factor_umaps": FigSpec("factor_umaps", ("fig_factor_umaps", "fig_factor_umaps_overlay", "umap_factor_coords.csv"),
                            "UMAP of each factor block (cell 27)"),
    "factor_gene_correlates": FigSpec("factor_gene_correlates",
                                      ("fig_factor_gene_correlates", "factor_gene_correlations.txt", "factor_gene_correlations.tsv"),
                                      "gene correlates of each factor axis (cells 31, 33)"),
    "study_leakage": FigSpec("study_leakage", ("fig_study_leakage",), "same-study neighbour fraction on TEST (cell 44)"),
    "threshold_curves": FigSpec("threshold_curves", ("fig_threshold_curves", "oof_scores.csv"),
                                "balanced accuracy vs threshold on out-of-fold refits (cell 25)", needs_retrain=True),
}


def list_figures() -> list:
    return list(FIGURES)


def figure_outputs(names=None, formats=("pdf", "png")) -> list:
    """Every file a run of the given figures writes (figure stems expanded per format)."""
    out = []
    for n in (names or FIGURES):
        for stem in FIGURES[n].outputs:
            if "." in stem:
                out.append(stem)
            else:
                out += [f"{stem}.{fmt}" for fmt in formats]
    return out
