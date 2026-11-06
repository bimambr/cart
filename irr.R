library(irrCAC)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Please provide at least two file paths as arguments.\nUsage: Rscript tqa_analysis.R <file1.csv> <file2.csv> [file3.csv ...]")
}

valid_files <- c()
for (f in args) {
  if (file.exists(f)) {
    valid_files <- c(valid_files, f)
  } else {
    warning(sprintf("File not found (skipping): %s", f))
  }
}

num_raters <- length(valid_files)
if (num_raters < 2) {
  stop("At least 2 valid data files are required to compute Inter-Rater Reliability.")
}

metrics <- c("accuracy", "acceptability", "readability")
df_list <- list()
raw_dfs <- list()

for (i in seq_along(valid_files)) {
  temp <- read.csv(valid_files[i])
  raw_dfs[[i]] <- temp

  for (m in metrics) {
    if (m %in% colnames(temp)) {
      colnames(temp)[colnames(temp) == m] <- paste0(m, "_R", i)
    }
  }

  keep_cols <- c("idiom_id", "treatment", paste0(metrics, "_R", i))
  df_list[[i]] <- temp[, intersect(keep_cols, colnames(temp))]
}

merged_df <- Reduce(
  function(x, y) merge(x, y, by = c("idiom_id", "treatment")),
  df_list
)

if (nrow(merged_df) == 0) {
  stop("No matching (idiom_id, treatment) instances found across all provided files.")
}

cat(sprintf(
  "\nSuccessfully merged %d raters across %d matched translation instances.\n",
  num_raters, nrow(merged_df)
))

cat("\n========================================\n")
cat(sprintf(" INTER-RATER RELIABILITY (GWET'S AC2: %d RATERS)\n", num_raters))
cat("========================================\n")
cat("Using quadratic weights for 1-3 ordinal scale.\n")
cat("Applying Cluster Bootstrap (B=1000) by idiom_id to correct SE for repeated measures.\n\n")

B <- 1000
unique_ids <- unique(merged_df$idiom_id)
set.seed(42)

for (metric in metrics) {
  cat(sprintf("--- %s ---\n", toupper(metric)))

  rater_cols <- paste0(metric, "_R", 1:num_raters)
  rating_data <- merged_df[, rater_cols, drop = FALSE]
  orig_ac2_res <- gwet.ac1.raw(rating_data,
    weights = "quadratic",
    categ.labels = 1:3
  )
  orig_ac2 <- orig_ac2_res$est$coeff.val

  boot_ac2 <- numeric(B)

  for (b in 1:B) {
    boot_ids <- sample(unique_ids, replace = TRUE)
    boot_list <- lapply(boot_ids, function(id) {
      merged_df[merged_df$idiom_id == id, rater_cols, drop = FALSE]
    })
    boot_df <- do.call(rbind, boot_list)
    boot_val <- tryCatch(
      {
        gwet.ac1.raw(boot_df,
          weights = "quadratic",
          categ.labels = 1:3
        )$est$coeff.val
      },
      error = function(e) NA
    )
    boot_ac2[b] <- boot_val
  }

  boot_ac2_clean <- na.omit(boot_ac2)
  boot_se <- sd(boot_ac2_clean)
  boot_ci <- quantile(boot_ac2_clean, probs = c(0.025, 0.975))

  z_stat <- orig_ac2 / boot_se
  p_val <- 2 * (1 - pnorm(abs(z_stat)))

  cat(sprintf("Original AC2:    %.3f\n", orig_ac2))
  cat(sprintf("Bootstrap SE:    %.3f\n", boot_se))
  cat(sprintf("95%% CI:          [%.3f, %.3f]\n", boot_ci[1], boot_ci[2]))
  cat(sprintf("Boot p-value:    %.3f\n\n", p_val))
}

cat("\n========================================\n")
cat(" OVERALL TQA SUMMARY & FREQUENCIES (Pooled Raters)\n")
cat("========================================\n")

pooled_df <- do.call(rbind, raw_dfs)

tqa_table <- aggregate(
  cbind(accuracy, acceptability, readability) ~ treatment,
  data = pooled_df,
  FUN = function(x) round(mean(as.numeric(x), na.rm = TRUE), 2)
)

tqa_table$Weighted_Average <- round((
  3 * tqa_table$accuracy +
    2 * tqa_table$acceptability +
    1 * tqa_table$readability
) / 6, 2)

colnames(tqa_table) <- c(
  "Treatment",
  "Accuracy Mean",
  "Acceptability Mean",
  "Readability Mean",
  "Weighted Average"
)

cat("TQA (Weighted Average) Summary:\n")
print(tqa_table)

for (col in metrics) {
  pooled_df[[col]] <- factor(pooled_df[[col]],
    ordered = TRUE, levels = c(1, 2, 3)
  )

  cat(sprintf("\n--- %s Frequencies ---\n", toupper(col)))
  freq_table <- table(pooled_df$treatment, pooled_df[[col]])
  freq_df <- as.data.frame.matrix(freq_table)

  for (score in as.character(1:3)) {
    if (!score %in% colnames(freq_df)) {
      freq_df[[score]] <- 0
    }
  }
  freq_df <- freq_df[, as.character(1:3)]
  colnames(freq_df) <- paste("Score", colnames(freq_df))

  print(freq_df)
}
