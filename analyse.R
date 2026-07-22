library(brms)
library(emmeans)
library(bayestestR)
library(tidybayes)
library(performance)
library(ggplot2)
library(loo)

df <- read.csv("translations_long.csv")

tqa_table <- aggregate(
  with(df, (3 * accuracy + 2 * acceptability + readability) / 6),
  by = list(treatment = df$treatment),
  FUN = mean
)
cat("TQA (Weighted Average) Summary:\n")
print(tqa_table)

for (col in c("accuracy", "acceptability", "readability")) {
  df[[col]] <- factor(df[[col]], ordered = TRUE, levels = c(1, 2, 3))
}

df$rag_status <- as.factor(df$rag_status)
df$refine_status <- as.factor(df$refine_status)
df$idiom_id <- as.factor(df$idiom_id)

df$rag_status <- relevel(df$rag_status, ref = "RAG-")
df$refine_status <- relevel(df$refine_status, ref = "Refine-")

evaluate_bayes_clmm <- function(response_var, data) {
  cat("\n======================================================\n")
  cat(sprintf(
    "ANALYSING: %s (Bayesian Cumulative Logit)\n",
    toupper(response_var)
  ))
  cat("======================================================\n")

  model_specs <- list(
    PO = list(
      formula = paste(
        response_var,
        "~ rag_status * refine_status + (1 | idiom_id)"
      )
    ),
    NPO = list(
      formula = paste(
        response_var,
        "~ cs(rag_status * refine_status) + (1 | idiom_id)"
      )
    )
  )

  models <- list()

  for (model_name in names(model_specs)) {
    cat("\n---------------------------------\n")
    cat("Fitting", model_name, "model\n")
    cat("---------------------------------\n")

    models[[model_name]] <- brm(
      formula = as.formula(model_specs[[model_name]]$formula),
      data = data,
      family = cumulative(link = "logit"),
      prior = set_prior("normal(0, 1)", class = "b"),
      cores = 4,
      chains = 4,
      iter = 4000,
      warmup = 2000,
      seed = 123,
      control = list(adapt_delta = 0.95),
      save_pars = save_pars(all = TRUE)
    )

    print(
      pp_check(models[[model_name]],
        type = "bars",
        ndraws = 100
      ) +
        ggtitle(sprintf(
          "%s model: %s",
          model_name,
          toupper(response_var)
        ))
    )
    print(describe_posterior(
      models[[model_name]],
      ci = 0.89,
      ci_method = "hdi",
      rope_ci = 1.0
    ))

    cat("\n--- Posterior Contrasts (", model_name, ") ---\n", sep = "")
    em <- emmeans(
      models[[model_name]],
      ~ rag_status * refine_status,
      mode = "latent"
    )
    contrasts <- pairs(em)
    print(describe_posterior(
      contrasts,
      rope_range = rope_range(models[[model_name]]),
      ci = 0.89,
      ci_method = "hdi",
      rope_ci = 1.0
    ))
  }

  cat("\n=====================================\n")
  cat("LOO MODEL COMPARISON\n")
  cat("=====================================\n")

  loo_list <- lapply(models, loo, cores = 4)

  for (nm in names(loo_list)) {
    cat("\nLOO summary:", nm, "\n")
    print(loo_list[[nm]])
  }
  cat("\n")

  pareto_tables <- lapply(loo_list, pareto_k_table)
  comparisons <- loo_compare(loo_list)
  print(comparisons)

  invisible(list(
    models = models,
    loo = loo_list,
    comparison = comparisons,
    pareto_k = pareto_tables
  ))
}

acc_res <- evaluate_bayes_clmm("accuracy", df)
accpt_res <- evaluate_bayes_clmm("acceptability", df)
read_res <- evaluate_bayes_clmm("readability", df)
