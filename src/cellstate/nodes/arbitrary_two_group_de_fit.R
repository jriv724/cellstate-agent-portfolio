#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(DESeq2))

args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 10) stop("expected ten arguments")

counts_path <- args[[1]]
metadata_path <- args[[2]]
output_path <- args[[3]]
runtime_path <- args[[4]]
replicate_col <- args[[5]]
group_col <- args[[6]]
dataset_col <- args[[7]]
group_a <- args[[8]]
group_b <- args[[9]]
design_mode <- args[[10]]

counts <- read.delim(
  counts_path, row.names=1, check.names=FALSE, stringsAsFactors=FALSE
)
meta <- read.delim(
  metadata_path, check.names=FALSE, stringsAsFactors=FALSE
)
if (anyDuplicated(meta[[replicate_col]])) stop("independent replicates are duplicated")
if (!setequal(colnames(counts), meta[[replicate_col]])) {
  stop("count/metadata replicates are misaligned")
}
meta <- meta[match(colnames(counts), meta[[replicate_col]]),,drop=FALSE]
meta[[group_col]] <- factor(meta[[group_col]], levels=c(group_b, group_a))

if (design_mode == "unadjusted_group") {
  design_formula <- reformulate(group_col)
} else if (design_mode != "standard") {
  stop("unsupported design mode")
} else if (length(unique(meta[[dataset_col]])) == 1) {
  design_formula <- reformulate(group_col)
} else {
  meta[[dataset_col]] <- factor(meta[[dataset_col]])
  design_formula <- reformulate(c(dataset_col, group_col))
}
design_matrix <- model.matrix(design_formula, data=meta)
if (qr(design_matrix)$rank < ncol(design_matrix)) stop("design is rank deficient")

dds <- DESeqDataSetFromMatrix(
  countData=round(as.matrix(counts)),
  colData=meta,
  design=design_formula
)
dds <- DESeq(dds, test="Wald")
result <- results(
  dds,
  contrast=c(group_col, group_a, group_b),
  alpha=0.05,
  independentFiltering=TRUE
)
table <- as.data.frame(result)
table$gene <- rownames(table)
table$signed_statistic <- table$stat
table <- table[,c(
  "gene", "baseMean", "log2FoldChange", "lfcSE", "stat",
  "pvalue", "padj", "signed_statistic"
)]
write.table(
  table, output_path, sep="\t", row.names=FALSE, quote=FALSE, na="NA"
)
runtime <- data.frame(
  key=c("R", "DESeq2", "design_formula", "coefficient_extracted"),
  value=c(
    as.character(getRversion()),
    as.character(packageVersion("DESeq2")),
    paste(deparse(design_formula), collapse=""),
    paste(group_a, "minus", group_b)
  )
)
write.table(
  runtime, runtime_path, sep="\t", row.names=FALSE, quote=FALSE, na="NA"
)
