"""Figures. ``DATA_FIGURES`` describe the training data and need only the metadata and the split;
``MODEL_FIGURES`` need the training tables and a trained bundle (``context.FigureContext``).
One name can stand for several files (`pca_batch` writes three figures)."""
from .corpus import fig_corpus_composition
from .split_balance import fig_split_balance

DATA_FIGURES = {"split_balance": fig_split_balance, "corpus_composition": fig_corpus_composition}
MODEL_FIGURES = ("pca_batch", "batch_genes", "head_test_eval", "head_scores", "head_gene_correlates", "study_leakage",
                 "threshold_curves", "gsea")
FIGURES = DATA_FIGURES          # the name this registry had when there were only the data figures
