"""Figures that describe the training data: `split_balance` and `corpus_composition`."""
from .corpus import fig_corpus_composition
from .split_balance import fig_split_balance

FIGURES = {"split_balance": fig_split_balance, "corpus_composition": fig_corpus_composition}
