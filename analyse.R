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

maybe_run_clmm_posthoc <- function(model) {
  ct <- coef(summary(model))

  interaction_row <- grep(":", rownames(ct))

  if (length(interaction_row) == 0) {
    cat("No interaction term found.\n")
    return(invisible(NULL))
  }

  p <- ct[interaction_row, "Pr(>|z|)"]

  if (p < 0.05) {
    cat(
      sprintf(
        "Interaction significant (p = %.4f).",
        p
      ),
      "Running pairwise comparisons...\n"
    )

    print(emmeans(model, pairwise ~ rag_status * refine_status))
  } else {
    cat(
      sprintf(
        "Interaction not significant (p = %.4f).",
        p
      ),
      "Pairwise comparisons skipped."
    )
  }
}

section <- local({
  i <- 0
  function(title) {
    i <<- i + 1
    cat(
      "======================================================\n",
      sprintf("%d. %s\n", i, title),
      "======================================================\n\n",
      sep = ""
    )
  }
})

spacer <- function(n = 1) {
  cat(strrep("\n", n))
}

section("CLMM: ACCURACY")
print(nominal_test(clm(
  accuracy ~ rag_status * refine_status,
  data = df
)))
spacer()
model_accu <- clmm(
  accuracy ~ rag_status * refine_status + (1 | idiom_id),
  data = df,
  nAGQ = 10
)
print(summary(model_accu))
spacer()
maybe_run_clmm_posthoc(model_accu)

spacer(2)
section("CLMM: ACCEPTABILITY")
print(nominal_test(clm(
  acceptability ~ rag_status * refine_status,
  data = df
)))
spacer()
model_acce <- clmm(
  acceptability ~ rag_status * refine_status + (1 | idiom_id),
  data = df,
  nAGQ = 10
)
print(summary(model_acce))
spacer()
maybe_run_clmm_posthoc(model_acce)

spacer(2)
section("CLMM: READABILITY")
print(nominal_test(clm(
  readability ~ rag_status * refine_status,
  data = df
)))
spacer()
model_read <- clmm(
  readability ~ rag_status * refine_status + (1 | idiom_id),
  data = df,
  nAGQ = 10
)
print(summary(model_read))
spacer()
maybe_run_clmm_posthoc(model_read)

spacer(2)
section("LMM: WEIGHTED TQA COMPOSITE (ANOVA)")
model_tqa <- lmer(
  weighted_tqa ~ rag_status * refine_status + (1 | idiom_id),
  data = df
)

res <- resid(model_tqa)
qqnorm(resid(model_tqa))
qqline(resid(model_tqa), col = "red")
skew_val <- skewness(res)
cat("Residual Skewness:", round(skew_val, 3), "\n")
spacer()
plot(model_tqa)

anova_results <- anova(model_tqa)
print(anova_results)
spacer()

interaction_p <- anova_results["rag_status:refine_status", "Pr(>F)"]

if (!is.na(interaction_p) && interaction_p < 0.05) {
  cat(
    sprintf(
      "Interaction significant (p = %.4f).",
      interaction_p
    ),
    "Running pairwise comparisons...\n"
  )

  interaction_analysis <- emmeans(
    model_tqa,
    pairwise ~ rag_status * refine_status,
    lmer.df = "satterthwaite"
  )

  print(summary(interaction_analysis))
} else {
  cat(
    sprintf(
      "Interaction not significant (p = %.4f).",
      interaction_p
    ),
    "Pairwise comparisons skipped."
  )
}
