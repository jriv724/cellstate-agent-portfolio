#!/usr/bin/env Rscript
suppressPackageStartupMessages(library(DESeq2))
args <- commandArgs(trailingOnly=TRUE)
if (length(args) != 8) stop("expected eight arguments")
counts_path<-args[[1]]; metadata_path<-args[[2]]; output_dir<-args[[3]]
sample_col<-args[[4]]; dataset_col<-args[[5]]; stage_col<-args[[6]]; state_col<-args[[7]]; target_state<-args[[8]]
dir.create(output_dir,recursive=TRUE,showWarnings=FALSE)
counts<-read.csv(counts_path,row.names=1,check.names=FALSE)
meta<-read.csv(metadata_path,check.names=FALSE,stringsAsFactors=FALSE)
meta<-meta[meta[[state_col]]==target_state,,drop=FALSE]
if(anyDuplicated(meta[[sample_col]])) stop("biological samples are duplicated")
if(!setequal(colnames(counts),meta[[sample_col]])) stop("count/metadata samples are misaligned")
meta<-meta[match(colnames(counts),meta[[sample_col]]),,drop=FALSE]
if(!all(meta[[stage_col]] %in% c("NBM","SMM","NDMM"))) stop("invalid stage")
meta[[stage_col]]<-relevel(factor(meta[[stage_col]],levels=c("NBM","SMM","NDMM")),ref="NBM")
meta[[dataset_col]]<-factor(meta[[dataset_col]])
formula<-reformulate(c(dataset_col,stage_col)); design_matrix<-model.matrix(formula,data=meta)
if(qr(design_matrix)$rank<ncol(design_matrix)) stop("design is rank deficient")
keep<-rowSums(counts>=10)>=3; counts<-counts[keep,,drop=FALSE]
if(!nrow(counts)) stop("no genes remain after approved prefilter")
dds<-DESeqDataSetFromMatrix(round(as.matrix(counts)),meta,formula); dds<-DESeq(dds)
write.csv(data.frame(metric=c("samples","genes_after_filter","design_rank","design_columns"),value=c(ncol(counts),nrow(counts),qr(design_matrix)$rank,ncol(design_matrix))),file.path(output_dir,"model_qc.csv"),row.names=FALSE)
for(comparison in c("SMM","NDMM")){
 result<-results(dds,contrast=c(stage_col,comparison,"NBM"),alpha=0.05,independentFiltering=TRUE)
 table<-as.data.frame(result); table$gene<-rownames(table); table$cooks_or_independent_filtered<-is.na(table$padj)
 write.csv(table,file.path(output_dir,paste0("unshrunk_",comparison,"_vs_NBM.csv")),row.names=FALSE)
 if(requireNamespace("apeglm",quietly=TRUE)){
  coefficient<-paste0(stage_col,"_",comparison,"_vs_NBM")
  if(coefficient %in% resultsNames(dds)){
   shrunk<-as.data.frame(lfcShrink(dds,coef=coefficient,type="apeglm")); shrunk$gene<-rownames(shrunk)
   write.csv(shrunk,file.path(output_dir,paste0("apeglm_",comparison,"_vs_NBM.csv")),row.names=FALSE)
  }
 }
}
