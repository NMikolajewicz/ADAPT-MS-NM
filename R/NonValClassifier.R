# =============================================================================
# NonValClassifier.R
# XGBoost classifier without external validation (repeated train/test splits)
# R equivalent of utils/NonValClassifier.py
# =============================================================================

library(xgboost)
library(randomForest)
library(caret)
library(pROC)
library(VIM)

if (!exists(".adaptms_impute_dataset", mode = "function")) {
  util_candidates <- character(0)
  if (requireNamespace("here", quietly = TRUE)) {
    util_candidates <- c(util_candidates, file.path(here::here(), "R", "ImputationUtils.R"))
  }

  source_file <- tryCatch({
    of <- sys.frame(1)$ofile
    if (is.null(of)) "" else normalizePath(of, winslash = "/", mustWork = FALSE)
  }, error = function(e) "")
  if (nzchar(source_file)) {
    util_candidates <- c(util_candidates, file.path(dirname(source_file), "ImputationUtils.R"))
  }

  util_candidates <- c(util_candidates, file.path("R", "ImputationUtils.R"), "ImputationUtils.R")
  util_candidates <- unique(util_candidates[nzchar(util_candidates)])
  util_path <- util_candidates[file.exists(util_candidates)][1]
  if (is.na(util_path)) {
    stop("Missing required utility file: R/ImputationUtils.R")
  }
  source(util_path)
}

.interp_tpr_on_grid <- function(fpr, tpr, mean_fpr) {
  ord <- order(fpr, tpr)
  fpr_sorted <- fpr[ord]
  tpr_sorted <- tpr[ord]
  keep <- !duplicated(fpr_sorted)
  if (sum(keep) < 2) {
    return(NULL)
  }
  interp_tpr <- approx(fpr_sorted[keep], tpr_sorted[keep], xout = mean_fpr, rule = 2)$y
  interp_tpr[1] <- 0
  interp_tpr
}

.get_env_int_nonval <- function(var_name, default_value, min_value = 1L) {
  raw <- Sys.getenv(var_name, unset = "")
  if (!nzchar(raw)) {
    return(as.integer(default_value))
  }
  parsed <- suppressWarnings(as.integer(raw))
  if (is.na(parsed) || parsed < min_value) {
    return(as.integer(default_value))
  }
  parsed
}

NonValClassifier <- setRefClass(
  "NonValClassifier",
  fields = list(
    prot_df    = "data.frame",
    cat_df     = "data.frame",
    gene_dict  = "ANY",
    between    = "character",
    figures    = "list"
  ),
  methods = list(
    initialize = function(prot_df, cat_df, gene_dict = NULL, between = NULL) {
      prot_df <<- prot_df
      cat_df <<- cat_df
      gene_dict <<- gene_dict
      between <<- if (is.null(between)) "" else between
      figures <<- list()
    },

    classify_and_plot_multiple = function(category1, category2, num_repeats = 10) {
      if (between == "") {
        stop("The 'between' parameter must be set to specify the categories for classification.")
      }

      rf_trees <- .get_env_int_nonval("ADAPTMS_RF_TREES", 100L, min_value = 1L)
      xgb_nrounds <- .get_env_int_nonval("ADAPTMS_XGB_NROUNDS", 100L, min_value = 1L)

      # Fast KNN-style imputation (configurable via ADAPTMS_IMPUTE_METHOD).
      prot_imp <- .adaptms_impute_dataset(.self$prot_df, k = 5)

      # Merge
      shared_ids <- intersect(rownames(prot_imp), rownames(.self$cat_df))
      d_ML <- cbind(prot_imp[shared_ids, , drop = FALSE],
                    .self$cat_df[shared_ids, , drop = FALSE])

      feat_cols <- setdiff(colnames(d_ML), between)
      X <- as.matrix(d_ML[, feat_cols, drop = FALSE])
      y <- d_ML[[between]]

      keep <- y %in% c(category1, category2)
      X_filtered <- X[keep, , drop = FALSE]
      y_filtered <- ifelse(y[keep] == category1, 0, 1)
      class_counts <- table(factor(y[keep], levels = c(category1, category2)))

      if (length(y_filtered) < 2) {
        stop(sprintf(
          "Need at least 2 samples after filtering '%s' for '%s' vs '%s' (found %d).",
          between, category1, category2, length(y_filtered)
        ))
      }
      if (any(class_counts == 0)) {
        stop(sprintf(
          "Both classes must be present after filtering '%s' for '%s' vs '%s'. Counts: %s",
          between, category1, category2,
          paste(sprintf("%s=%d", names(class_counts), as.integer(class_counts)), collapse = ", ")
        ))
      }

      test_aucs <- numeric(0)
      all_test_fprs <- list()
      all_test_tprs <- list()

      for (repeat_i in seq_len(num_repeats)) {
        message(sprintf("Running iteration %d/%d", repeat_i, num_repeats))

        random_state <- 42 + repeat_i - 1
        set.seed(random_state)
        train_idx <- createDataPartition(factor(y_filtered, levels = c(0, 1)),
                                         p = 0.8, list = FALSE)[, 1]

        X_train <- X_filtered[train_idx, , drop = FALSE]
        X_test  <- X_filtered[-train_idx, , drop = FALSE]
        y_train <- y_filtered[train_idx]
        y_test  <- y_filtered[-train_idx]

        # Feature selection with Random Forest
        rf_model <- randomForest(
          x = X_train, y = factor(y_train),
          ntree = rf_trees, importance = TRUE
        )
        importance_vals <- importance(rf_model, type = 1)[, 1]
        selected_feats <- names(importance_vals[importance_vals > mean(importance_vals)])
        if (length(selected_feats) < 2) {
          selected_feats <- names(sort(importance_vals, decreasing = TRUE))[1:min(10, length(importance_vals))]
        }

        X_train_sel <- X_train[, selected_feats, drop = FALSE]
        X_test_sel  <- X_test[, selected_feats, drop = FALSE]

        # Train XGBoost
        dtrain <- xgb.DMatrix(data = X_train_sel, label = y_train)
        dtest  <- xgb.DMatrix(data = X_test_sel)

        params <- list(
          objective = "binary:logistic",
          eval_metric = "logloss",
          max_depth = 6,
          eta = 0.1
        )
        model <- xgb.train(params = params, data = dtrain, nrounds = xgb_nrounds, verbose = 0)

        y_test_pred <- predict(model, dtest)

        if (length(unique(y_test)) > 1) {
          roc_obj <- roc(y_test, y_test_pred, quiet = TRUE)
          auc_val <- as.numeric(auc(roc_obj))
          test_aucs <- c(test_aucs, auc_val)
          all_test_fprs[[length(all_test_fprs) + 1]] <- 1 - roc_obj$specificities
          all_test_tprs[[length(all_test_tprs) + 1]] <- roc_obj$sensitivities
        }
      }

      if (length(test_aucs) == 0) {
        stop("No valid ROC/AUC values were produced. Check that both classes are present in each split.")
      }

      # Calculate mean AUC and 95% CI
      mean_auc <- mean(test_aucs, na.rm = TRUE)
      if (length(test_aucs) > 1) {
        se_auc   <- sd(test_aucs, na.rm = TRUE) / sqrt(length(test_aucs))
        ci_lower <- mean_auc - qt(0.975, df = length(test_aucs) - 1) * se_auc
        ci_upper <- mean_auc + qt(0.975, df = length(test_aucs) - 1) * se_auc
      } else {
        ci_lower <- mean_auc
        ci_upper <- mean_auc
      }

      # Interpolate all ROC curves to common FPR axis
      mean_fpr <- seq(0, 1, length.out = 100)
      interp_tprs <- matrix(nrow = 0, ncol = 100)

      for (k in seq_along(all_test_fprs)) {
        interp_tpr <- .interp_tpr_on_grid(all_test_fprs[[k]], all_test_tprs[[k]], mean_fpr)
        if (!is.null(interp_tpr)) {
          interp_tprs <- rbind(interp_tprs, interp_tpr)
        }
      }

      if (nrow(interp_tprs) == 0) {
        stop("Unable to build aggregated ROC curve because interpolation failed for all runs.")
      }

      mean_tpr <- colMeans(interp_tprs)
      mean_tpr[100] <- 1.0
      std_tpr <- apply(interp_tprs, 2, sd)
      n_curves <- nrow(interp_tprs)
      tprs_upper <- pmin(mean_tpr + 1.96 * std_tpr / sqrt(n_curves), 1)
      tprs_lower <- pmax(mean_tpr - 1.96 * std_tpr / sqrt(n_curves), 0)

      # Build plot data
      # Individual curves
      individual_dfs <- lapply(seq_along(all_test_fprs), function(k) {
        data.frame(fpr = all_test_fprs[[k]], tpr = all_test_tprs[[k]], run = k)
      })
      ind_df <- do.call(rbind, individual_dfs)

      p <- ggplot2::ggplot() +
        ggplot2::geom_line(data = ind_df, ggplot2::aes(x = fpr, y = tpr, group = run),
                           color = "grey", alpha = 0.3, linewidth = 0.5) +
        ggplot2::geom_ribbon(ggplot2::aes(x = mean_fpr, ymin = tprs_lower, ymax = tprs_upper),
                             fill = "blue", alpha = 0.2) +
        ggplot2::geom_line(ggplot2::aes(x = mean_fpr, y = mean_tpr),
                           color = "blue", linewidth = 1) +
        ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
        ggplot2::labs(
          x = "False Positive Rate",
          y = "True Positive Rate",
          title = sprintf("Aggregated ROC Curves with 95%% CI (n=%d)", num_repeats),
          subtitle = sprintf("Mean AUC = %.2f, 95%% CI: %.2f - %.2f",
                             mean_auc, ci_lower, ci_upper)
        ) +
        ggplot2::xlim(0, 1) + ggplot2::ylim(0, 1.05) +
        ggplot2::theme_minimal()

      figures[[length(figures) + 1]] <<- p
      print(p)

      return(list(mean_auc = mean_auc, ci = c(ci_lower, ci_upper)))
    },

    save_plots_to_pdf = function(pdf_path) {
      if (length(figures) == 0) {
        message("No figures available.")
        return(invisible(NULL))
      }
      pdf(pdf_path, width = 10, height = 8)
      for (fig in figures) {
        if (inherits(fig, "ggplot")) print(fig)
      }
      dev.off()
      message(sprintf("Plots saved to %s", pdf_path))
    }
  )
)
