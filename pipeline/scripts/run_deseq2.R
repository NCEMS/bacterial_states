#!/usr/bin/env Rscript

#Required packages
suppressPackageStartupMessages({
  library(DESeq2)
  library(ggplot2)
  library(tibble)
  library(dplyr)
  library(readr)
})

#Loading in arguments
args <- commandArgs(trailingOnly = TRUE)
counts_path <- args[1]
sample_metadata_path <- args[2]
output_dir <- args[3]

#Getting featureCounts data
counts <- read.delim(counts_path, row.names = 1, check.names = FALSE)
colnames(counts) <- basename(colnames(counts))
colnames(counts) <- sub("_sort.bam$", "", colnames(counts))
sample_table <- read.delim(sample_metadata_path)
rownames(sample_table) <- sample_table$sample_id
counts <- counts[, rownames(sample_table)]
counts <- counts[rowSums(counts) > 0, ]

#Design formula
if ("condition2" %in% colnames(sample_table)) {
  dds <- DESeqDataSetFromMatrix(countData = counts,
                                colData = sample_table,
                                design = ~ condition2 + condition)
} else {
  dds <- DESeqDataSetFromMatrix(countData = counts,
                                colData = sample_table,
                                design = ~ condition)
}
dds$condition <- relevel(dds$condition, ref = levels(dds$condition)[1])
if ("condition2" %in% colnames(sample_table)) {
  dds$condition2 <- relevel(dds$condition2, ref = levels(dds$condition2)[1])
}

#Running DEseq2
dds <- DESeq(dds)

#Making PCA plot
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

#Outputting normalized counts for downstream anaylsis
norm_counts <- counts(dds, normalized = TRUE)
write.table(norm_counts, file = file.path(output_dir, "normalized_counts.tsv"),
            sep = "\t", quote = FALSE, col.names = NA)


#Outputting DESeq2 results for comparisons listed in config file
conds <- levels(dds$condition)
for (i in 2:length(conds)) {
  ref <- conds[1]
  test <- conds[i]
  res <- results(dds, contrast = c("condition", test, ref))
  res_df <- as.data.frame(res) %>%
    rownames_to_column("Gene_Symbol") %>%
    mutate(Gene_Symbol = trimws(as.character(Gene_Symbol))) %>%
    arrange(padj)

  write.table(res_df,
              file = file.path(output_dir, paste0(test, "_vs_", ref, "_deseq_results.tsv")),
              sep = "\t", quote = FALSE, row.names = FALSE)
}