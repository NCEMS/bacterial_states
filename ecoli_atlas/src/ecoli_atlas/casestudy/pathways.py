"""Pathway analysis of a case-study axis: the genes ranked by their correlation with the axis, tested by preranked GSEA
(``ecoli_atlas.gsea``, two-sided) against GO terms and against a curated module table (`ecoli_atlas download modules`:
iModulons, KEGG pathways and modules, RegulonDB regulons by direction); one table per kind of gene set and one
figure per database."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

MODULE_DATABASES = ("iModulon", "KEGG pathway", "KEGG module", "RegulonDB")     # one figure each, in this order
TOP = 20
def pathway_analysis(rho: pd.Series, out: Path, name: str, high: str, low: str, formats, gsea: bool, modules, gene_sets,
                     permutations: int, log, axis_label: str | None = None, sides: str = "two") -> dict:
    """Preranked GSEA of a gene ranking (``rho``, indexed by gene label) against GO terms (``gsea``) and a module table
    (``modules``): ``<name>_gsea_go.tsv`` / ``<name>_gsea_modules.tsv`` and their figures. Positive NES = towards
    ``high``. Returns the summaries under ``gsea_go`` / ``gsea_modules``."""
    from ..figures.gsea import fig_gsea_bars, fig_gsea_by_database
    from ..gsea import ranking_by_symbol, run as run_gsea
    result = {}
    axis = axis_label or name
    ranking = ranking_by_symbol(rho.to_numpy(), rho.index)
    for key, path, what, on in (("go", gene_sets, "GO terms", gsea), ("modules", modules, "pathways, regulons and iModulons", modules)):
        if not on:
            continue
        res, info = run_gsea(ranking, path, permutations=permutations, sides=sides)   # a module table is not a GO table: kept whole
        res.to_csv(out / f"{name}_gsea_{key}.tsv", sep="\t", index=False, float_format="%.5g")
        if key == "modules":                     # one figure per database: fig_<name>_gsea_<database>
            fig_gsea_by_database(res, f"sets along the {axis} axis", out, f"fig_{name}_gsea", formats, top=TOP,
                                 up_label=f"towards {high}", down_label=f"towards {low}", databases=MODULE_DATABASES)
        else:
            fig_gsea_bars(res, f"{what} along the {axis} axis", out, f"fig_{name}_gsea_{key}", formats, top=TOP, both=True,
                          up_label=f"towards {high}", down_label=f"towards {low}")
        result[f"gsea_{key}"] = info
        log(f"[{name}] {what}: {info['terms_tested']} tested, FDR < 0.05: {info['fdr_below_0.05']['up']} towards {high}, "
            f"{info['fdr_below_0.05']['down']} towards {low}")
    return result
