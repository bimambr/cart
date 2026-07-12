suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(ordinal)
  library(emmeans)
  library(moments)
})

df <- read.csv("translations_long.csv")

df$accuracy <- factor(
  df$accuracy,
  ordered = TRUE, levels = c(1, 2, 3)
)
df$acceptability <- factor(
  df$acceptability,
  ordered = TRUE, levels = c(1, 2, 3)
)
df$readability <- factor(
  df$readability,
  ordered = TRUE, levels = c(1, 2, 3)
)

df$rag_status <- as.factor(df$rag_status)
df$refine_status <- as.factor(df$refine_status)
df$idiom_id <- as.factor(df$idiom_id)

# Set baselines
df$rag_status <- relevel(df$rag_status, ref = "RAG-")
df$refine_status <- relevel(df$refine_status, ref = "Refine-")

cat("\n======================================================\n")
cat("   1. CLMM: ACCURACY\n")
cat("======================================================\n")
model_accu <- clmm(
  accuracy ~ rag_status * refine_status + (1 | idiom_id),
  data = df,
  nAGQ = 10
)
print(summary(model_accu))

cat("\n======================================================\n")
cat("   2. CLMM: ACCEPTABILITY\n")
cat("======================================================\n")
model_acce <- clmm(
  acceptability ~ rag_status * refine_status + (1 | idiom_id),
  data = df,
  nAGQ = 10
)
print(summary(model_acce))

cat("\n======================================================\n")
cat("   3. CLMM: READABILITY\n")
cat("======================================================\n")
model_read <- clmm(
  readability ~ rag_status * refine_status + (1 | idiom_id),
  data = df,
  nAGQ = 10
)
print(summary(model_read))

cat("\n======================================================\n")
cat("   4. LMM: WEIGHTED TQA COMPOSITE (ANOVA)\n")
cat("======================================================\n")
model_tqa <- lmer(
  weighted_tqa ~ rag_status * refine_status + (1 | idiom_id),
  data = df
)

res <- resid(model_tqa)
qqnorm(resid(model_tqa))
qqline(resid(model_tqa), col = "red")
skew_val <- skewness(res)
cat("\nResidual Skewness:", round(skew_val, 3), "\n")

print(anova(model_tqa))

cat("\n======================================================\n")
cat("   5. POST-HOC ANALYSIS (COMPOSITE)\n")
cat("======================================================\n")
interaction_analysis <- emmeans(
  model_tqa, pairwise ~ rag_status * refine_status,
  lmer.df = "satterthwaite"
)
print(summary(interaction_analysis))
