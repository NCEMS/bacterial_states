#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(DESeq2)
  library(ggplot2)
  library(tibble)
  library(dplyr)
  library(readr)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
counts_path <- args[1]
sample_metadata_path <- args[2]
output_dir <- args[3]
contrasts_json_string <- args[4]
deseq2_contrasts <- fromJSON(contrasts_json_string, simplifyVector = FALSE) 

counts <- read.delim(counts_path, row.names = 1, check.names = FALSE)
colnames(counts) <- basename(colnames(counts))
colnames(counts) <- sub("_sort.bam$", "", colnames(counts))
sample_table <- read.delim(sample_metadata_path)
rownames(sample_table) <- sample_table$sample_id
counts <- counts[, rownames(sample_table)]
counts <- counts[rowSums(counts) > 0, ]

sample_table$condition <- factor(sample_table$condition)
if ("condition2" %in% colnames(sample_table)) {
  sample_table$condition2 <- factor(sample_table$condition2)
}

#Some datasets dont have proper groupings to perform DESeq2 normalization
if (nlevels(sample_table$condition) < 2) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  
  output_files <- c(
    file.path(output_dir, "pca_coordinates.tsv"),
    file.path(output_dir, "pca_plot.png"),
    file.path(output_dir, "normalized_counts.tsv")
  )
  for (contrast_info in deseq2_contrasts) {
    ref_level <- contrast_info[[2]]
    test_level <- contrast_info[[3]]
    file_name <- paste0(ref_level, "_vs_", test_level, "_deseq_results.tsv")
    output_files <- c(output_files, file.path(output_dir, file_name))
  }
  
  for (f in output_files) {
    if (grepl("\\.png$", f)) {
      png(f)
      plot.new()
      text(0.5, 0.5, "Insufficient metadata", cex = 1.5)
      dev.off()
    } else {
      writeLines("Insufficient metadata", f)
    }
  }
  
  quit(save = "no", status = 0)
}


if ("condition2" %in% colnames(sample_table)) {
  dds <- DESeqDataSetFromMatrix(countData = counts,
                                colData = sample_table,
                                design = ~ condition2 + condition)
} else {
  dds <- DESeqDataSetFromMatrix(countData = counts,
                                colData = sample_table,
                                design = ~ condition)
}

dds <- DESeq(dds)

vsd <- vst(dds, blind = FALSE)
pca <- prcomp(t(assay(vsd)))
percentVar <- round(100 * (pca$sdev^2 / sum(pca$sdev^2)), 1)
pca_df <- data.frame(pca$x, sample_table)
write.table(pca_df, file = file.path(output_dir, "pca_coordinates.tsv"),
            sep = "\t", quote = FALSE, row.names = TRUE)
ggsave(
  filename = file.path(output_dir, "pca_plot.png"),
  plot = ggplot(pca_df, aes(PC1, PC2, color = condition)) +
    geom_point(size = 4) +
    xlab(paste0("PC1: ", percentVar[1], "% variance")) +
    ylab(paste0("PC2: ", percentVar[2], "% variance")) +
    theme_minimal()
)

norm_counts <- counts(dds, normalized = TRUE)
write.table(norm_counts, file = file.path(output_dir, "normalized_counts.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)
if (length(deseq2_contrasts) > 0) {
  for (contrast_info in deseq2_contrasts) {
    factor_name <- contrast_info[[1]]
    ref_level <- contrast_info[[2]]
    test_level <- contrast_info[[3]]
  
    if (!factor_name %in% colnames(colData(dds))) {
      warning(paste0("Factor '", factor_name, "' not found in sample metadata. Skipping contrast: ",
                     ref_level, " vs ", test_level, "."))
      next
    }
  
    current_factor_levels <- levels(colData(dds)[[factor_name]])
    if (!(ref_level %in% current_factor_levels)) {
      warning(paste0("Reference level '", ref_level, "' not found in levels for factor '", factor_name,
                     "'. Skipping contrast: ", ref_level, " vs ", test_level, "."))
      next
    }
    if (!(test_level %in% current_factor_levels)) {
      warning(paste0("Test level '", test_level, "' not found in levels for factor '", factor_name,
                     "'. Skipping contrast: ", ref_level, " vs ", test_level, "."))
      next
    }
  
    res <- results(dds, contrast = c(factor_name, test_level, ref_level))
  
    res_df <- as.data.frame(res) %>%
      rownames_to_column("Gene_Symbol") %>%
      mutate(Gene_Symbol = trimws(as.character(Gene_Symbol))) %>%
      arrange(padj)
  
    file_name <- paste0(ref_level, "_vs_", test_level, "_deseq_results.tsv")
    write.table(res_df,
                file = file.path(output_dir, file_name),
                sep = "\t", quote = FALSE, row.names = FALSE)
  }
}
