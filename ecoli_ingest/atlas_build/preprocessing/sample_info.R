library(tidyverse)
library(GEOquery)
library(RSQLite)
library(DBI)

Sys.setenv(ENTREZ_KEY = "XXXX")
options(timeout = 5000)

cat("=== Downloading E. coli RNA-Seq data from ENA ===\n")

ena_url <- paste0(
  "https://www.ebi.ac.uk/ena/portal/api/search?",
  "result=read_run",
  "&query=tax_eq(562)+AND+library_strategy=%22RNA-Seq%22",
  "&fields=run_accession,sample_accession,study_accession,experiment_accession,",
  "scientific_name,library_strategy,instrument_platform,instrument_model,",
  "sample_title,center_name,first_public,fastq_ftp",
  "&format=tsv",
  "&limit=0"
)

ena_data <- read_tsv(ena_url, show_col_types = FALSE) %>%
  filter(
    library_strategy == "RNA-Seq",
    str_detect(scientific_name, regex("Escherichia coli|E\\. coli", ignore_case = TRUE)),
    !is.na(run_accession)
  )

cat("Total E. coli RNA-Seq runs from ENA:", nrow(ena_data), "\n\n")

cat("=== Downloading GEO metadata ===\n")

sqlite_file_path <- "GEOmetadb.sqlite"
if (!file.exists(sqlite_file_path)) {
  cat("Downloading GEOmetadb.sqlite...\n")
  getSQLiteFile()
} else {
  cat("GEOmetadb.sqlite already exists\n")
}

con <- dbConnect(RSQLite::SQLite(), "GEOmetadb.sqlite")

gse_query <- "
SELECT DISTINCT gse.gse, gse.title, gse.type, gse.pubmed_id, gse.submission_date
FROM gse
JOIN gse_gsm ON gse.gse = gse_gsm.gse
JOIN gsm      ON gse_gsm.gsm = gsm.gsm
WHERE gsm.organism_ch1 LIKE '%Escherichia coli%'
  AND gse.type LIKE '%expression profiling by high throughput sequencing%'
  AND gse.gse NOT IN (
    SELECT DISTINCT g2.gse
    FROM gse_gsm g2
    JOIN gsm s2 ON g2.gsm = s2.gsm
    WHERE s2.organism_ch1 IS NOT NULL
      AND s2.organism_ch1 != ''
      AND s2.organism_ch1 NOT LIKE '%Escherichia coli%'
      AND s2.organism_ch1 NOT LIKE '%E. coli%'
  )
"

gse_data <- dbGetQuery(con, gse_query)
gse_list  <- unique(gse_data$gse)
cat("Found", length(gse_list), "pure E. coli RNA-Seq studies in GEO\n")

gsm_query <- sprintf("
SELECT gsm.gsm,
       gsm.title           AS sample_title,
       gsm.organism_ch1,
       gsm.characteristics_ch1,
       gsm.treatment_protocol_ch1,
       gse_gsm.gse
FROM gsm
JOIN gse_gsm ON gsm.gsm = gse_gsm.gsm
WHERE gse_gsm.gse IN ('%s')
  AND (
    gsm.organism_ch1 LIKE '%%Escherichia coli%%'
    OR gsm.organism_ch1 LIKE '%%E. coli%%'
  )
", paste(gse_list, collapse = "','"))

gsm_data <- dbGetQuery(con, gsm_query)
dbDisconnect(con)

cat("Found", nrow(gsm_data), "E. coli samples in GEO\n\n")

cat("=== Mapping GSM to SRX ===\n")

gsm_data$SRX <- sapply(seq_along(gsm_data$gsm), function(i) {
  if (i %% 100 == 0) cat(i, " ")
  gsm_id <- as.character(gsm_data$gsm[i])

  gsm_obj <- suppressMessages(suppressWarnings(
    tryCatch(getGEO(gsm_id, GSEMatrix = FALSE), error = function(e) NULL)
  ))
  if (is.null(gsm_obj)) return(NA_character_)

  rel      <- Meta(gsm_obj)$relation
  sra_line <- rel[grep("SRA:", rel)]
  if (length(sra_line) == 0) return(NA_character_)

  sub(".*term=(SRX[0-9]+).*", "\\1", sra_line[1])
})

cat("\nGSMs with SRX mappings:", sum(!is.na(gsm_data$SRX)), "\n\n")

gsm_data_clean <- gsm_data %>%
  filter(!is.na(SRX)) %>%
  select(gsm, SRX, gse, characteristics_ch1, treatment_protocol_ch1) %>%
  rename(GSM = gsm, GSE = gse)

cat("=== Merging ENA and GEO data ===\n")

final_data <- ena_data %>%
  left_join(gsm_data_clean, by = c("experiment_accession" = "SRX")) %>%
  mutate(
    sample_id  = ifelse(!is.na(GSM), GSM, experiment_accession),
    project_id = ifelse(!is.na(GSE), GSE, study_accession)
  ) %>%
  rename(data_type = library_strategy) %>%
  distinct(run_accession, .keep_all = TRUE) %>%
  filter(!is.na(GSM), !is.na(GSE))   #require both. drop anything not in GEO

char_cols <- sapply(final_data, is.character)
for (col_name in names(final_data)[char_cols]) {
  final_data[[col_name]] <- gsub("\t", " ", final_data[[col_name]])
}

final_data_selected <- final_data %>%
  select(
    run_accession, study_accession, sample_title,
    GSM, GSE, project_id,
    characteristics_ch1, treatment_protocol_ch1,
    instrument_platform, instrument_model,
    first_public, fastq_ftp
  )

write.table(
  final_data_selected,
  file      = "ecoli_rnaseq_metadata.txt",
  sep       = "\t",
  row.names = FALSE,
  quote     = FALSE
)

cat("Saved to: ecoli_rnaseq_metadata.txt\n")
