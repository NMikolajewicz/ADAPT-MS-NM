# =============================================================================
# XGBfillClassifier.R
# XGBoost with RF feature selection, zero-fill validation strategy
# R equivalent of utils/XGBfillClassifier.py (both DF and Folder variants)
# =============================================================================

library(xgboost)
library(randomForest)
library(caret)
library(pROC)
library(VIM)

.prepare_binary_eval_xgb <- function(predictions, category1, category2, context) {
  true_labels <- sapply(predictions, function(x) as.character(x$true_label))
  pred_probs <- as.numeric(sapply(predictions, function(x) x$prob))

  keep <- true_labels %in% c(category1, category2) & !is.na(pred_probs)
  if (!any(keep)) {
    stop(sprintf("No usable %s predictions found for categories '%s' and '%s'.",
                 context, category1, category2))
  }

  true_labels <- true_labels[keep]
  pred_probs <- pred_probs[keep]
  true_numeric <- ifelse(true_labels == category1, 0L, 1L)

  if (length(unique(true_numeric)) < 2) {
    stop(sprintf("Need predictions from both '%s' and '%s' to compute ROC.",
                 category1, category2))
  }

  list(true_numeric = true_numeric, pred_probs = pred_probs)
}

.sanitize_row_ids_xgb <- function(x, prefix = "row") {
  ids <- trimws(as.character(x))
  missing <- is.na(ids) | ids == ""
  if (any(missing)) {
    ids[missing] <- paste0(prefix, "_", seq_len(sum(missing)))
  }
  make.unique(ids)
}

.get_env_int_xgb <- function(var_name, default_value, min_value = 1L) {
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

.resolve_n_jobs_xgb <- function(n_jobs) {
  nj <- suppressWarnings(as.integer(n_jobs))
  if (is.na(nj)) nj <- 1L
  if (nj <= 0L) {
    detected <- suppressWarnings(parallel::detectCores(logical = FALSE))
    if (is.na(detected) || detected < 1L) detected <- 1L
    nj <- detected
  }
  nj
}

.parallel_lapply_xgb <- function(X, FUN, n_jobs = 1L) {
  nj <- .resolve_n_jobs_xgb(n_jobs)
  if (length(X) <= 1L || nj <= 1L || .Platform$OS.type == "windows") {
    return(lapply(X, FUN))
  }
  parallel::mclapply(X, FUN, mc.cores = min(length(X), nj))
}

.run_xgb_rf_iteration <- function(seed, X_filtered, X_imputed, y_filtered,
                                  n_estimators_rf, xgb_nrounds, cv_folds,
                                  xgb_nthread = 1L) {
  set.seed(seed - 1L)
  train_idx <- createDataPartition(factor(y_filtered, levels = c(0, 1)),
                                   p = 0.8, list = FALSE)[, 1]

  X_train_imp <- as.matrix(X_imputed[train_idx, , drop = FALSE])
  X_train_raw <- X_filtered[train_idx, , drop = FALSE]
  X_test_raw <- X_filtered[-train_idx, , drop = FALSE]
  y_train <- y_filtered[train_idx]
  y_test <- y_filtered[-train_idx]

  rf_model <- randomForest(
    x = X_train_imp, y = factor(y_train),
    ntree = n_estimators_rf, importance = TRUE
  )
  imp_obj <- importance(rf_model, type = 1)
  imp_vals <- if (is.null(dim(imp_obj))) {
    vals <- as.numeric(imp_obj)
    names(vals) <- colnames(X_train_imp)
    vals
  } else {
    imp_obj[, 1]
  }
  sel_feats <- names(imp_vals[imp_vals > mean(imp_vals, na.rm = TRUE)])
  if (length(sel_feats) < 2) {
    sel_feats <- names(sort(imp_vals, decreasing = TRUE))[1:min(10, length(imp_vals))]
  }
  sel_feats <- unique(sel_feats[!is.na(sel_feats)])
  if (length(sel_feats) == 0) return(NULL)

  X_train_sel <- as.matrix(X_train_raw[, sel_feats, drop = FALSE])
  X_test_sel <- as.matrix(X_test_raw[, sel_feats, drop = FALSE])

  params <- list(
    objective = "binary:logistic",
    eval_metric = "logloss",
    max_depth = 6,
    eta = 0.1,
    nthread = xgb_nthread
  )

  dtrain <- xgb.DMatrix(data = X_train_sel, label = y_train)
  xgb_model <- xgb.train(params = params, data = dtrain, nrounds = xgb_nrounds, verbose = 0)

  min_class <- min(sum(y_train == 0), sum(y_train == 1))
  if (min_class < 2L) {
    folds <- list()
  } else {
    k_folds <- min(cv_folds, min_class)
    folds <- createFolds(y_train, k = k_folds, list = TRUE, returnTrain = TRUE)
  }
  mean_fpr <- seq(0, 1, length.out = 100)
  cv_tprs <- matrix(nrow = 0, ncol = 100)

  for (fold_train_idx in folds) {
    fold_val_idx <- setdiff(seq_along(y_train), fold_train_idx)
    dcv_train <- xgb.DMatrix(data = X_train_sel[fold_train_idx, , drop = FALSE],
                             label = y_train[fold_train_idx])
    dcv_val <- xgb.DMatrix(data = X_train_sel[fold_val_idx, , drop = FALSE])
    y_cv_val <- y_train[fold_val_idx]

    cv_model <- xgb.train(params = params, data = dcv_train, nrounds = xgb_nrounds, verbose = 0)
    probas <- predict(cv_model, dcv_val)

    if (length(unique(y_cv_val)) > 1) {
      roc_obj <- roc(y_cv_val, probas, quiet = TRUE)
      interp_tpr <- approx(1 - roc_obj$specificities, roc_obj$sensitivities,
                           xout = mean_fpr, rule = 2)$y
      interp_tpr[1] <- 0
      cv_tprs <- rbind(cv_tprs, interp_tpr)
    }
  }

  dtest <- xgb.DMatrix(data = X_test_sel)
  test_probs <- predict(xgb_model, dtest)
  auc_val <- if (length(unique(y_test)) > 1) {
    as.numeric(auc(roc(y_test, test_probs, quiet = TRUE)))
  } else {
    NA_real_
  }

  mean_tpr <- if (nrow(cv_tprs) > 0) {
    out <- colMeans(cv_tprs)
    out[100] <- 1.0
    out
  } else {
    NULL
  }

  list(model = xgb_model, sel_feats = sel_feats, mean_tpr = mean_tpr, auc = auc_val)
}

# -----------------------------------------------------------------------------
# XGBRFClassifierDF
# DataFrame-based variant
# -----------------------------------------------------------------------------

XGBRFClassifierDF <- setRefClass(
  "XGBRFClassifierDF",
  fields = list(
    prot_df           = "data.frame",
    cat_df            = "data.frame",
    gene_dict         = "ANY",
    between           = "character",
    figures           = "list",
    models            = "list",
    selectors         = "list",
    predictions       = "list",
    feature_names     = "character",
    training_data     = "ANY",
    training_targets  = "ANY",
    final_model       = "ANY",
    final_selector    = "ANY",
    selected_features = "ANY",
    fill_na           = "ANY"
  ),
  methods = list(
    initialize = function(prot_df, cat_df, gene_dict = NULL, between = NULL) {
      prot_df <<- prot_df[, !duplicated(colnames(prot_df)), drop = FALSE]
      cat_df <<- cat_df
      gene_dict <<- gene_dict
      between <<- if (is.null(between)) "" else between
      figures <<- list()
      models <<- list()
      selectors <<- list()
      predictions <<- list()
      feature_names <<- colnames(prot_df)
      training_data <<- NULL
      training_targets <<- NULL
      final_model <<- NULL
      final_selector <<- NULL
      selected_features <<- NULL
      fill_na <<- NULL
    },

    classify_and_plot = function(category1, category2, n_runs = 10, n_estimators_rf = 200, n_jobs = 1) {
      if (between == "") {
        stop("No between variable given. Please provide a variable for classification.")
      }

      n_estimators_rf <- .get_env_int_xgb("ADAPTMS_RF_TREES", n_estimators_rf, min_value = 1L)
      xgb_nrounds <- .get_env_int_xgb("ADAPTMS_XGB_NROUNDS", 100L, min_value = 1L)
      cv_folds <- .get_env_int_xgb("ADAPTMS_CV_FOLDS", 5L, min_value = 2L)
      run_n_jobs <- .resolve_n_jobs_xgb(n_jobs)
      xgb_nthread_default <- if (run_n_jobs > 1L) 1L else run_n_jobs
      xgb_nthread <- .get_env_int_xgb("ADAPTMS_XGB_NTHREADS", xgb_nthread_default, min_value = 1L)

      # Drop all-NA columns
      all_na_cols <- sapply(.self$prot_df, function(x) all(is.na(x)))
      prot_clean <- .self$prot_df[, !all_na_cols, drop = FALSE]

      shared_ids <- intersect(rownames(prot_clean), rownames(.self$cat_df))
      d_ML <- cbind(prot_clean[shared_ids, , drop = FALSE],
                    .self$cat_df[shared_ids, between, drop = FALSE])

      X_raw <- d_ML[, colnames(d_ML) != between, drop = FALSE]
      y <- d_ML[[between]]

      keep <- y %in% c(category1, category2)
      X_filtered <- X_raw[keep, , drop = FALSE]
      y_filtered <- ifelse(y[keep] == category1, 0, 1)
      names(y_filtered) <- rownames(X_filtered)
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

      training_data <<- X_filtered
      training_targets <<- y_filtered

      # Impute for feature selection only
      X_imputed <- as.data.frame(kNN(X_filtered, k = 5, imp_var = FALSE))
      rownames(X_imputed) <- rownames(X_filtered)

      mean_fpr <- seq(0, 1, length.out = 100)
      all_tprs <- matrix(nrow = 0, ncol = 100)
      aucs <- numeric(0)

      run_results <- .parallel_lapply_xgb(
        seq_len(n_runs),
        function(i) .run_xgb_rf_iteration(
          seed = i,
          X_filtered = X_filtered,
          X_imputed = X_imputed,
          y_filtered = y_filtered,
          n_estimators_rf = n_estimators_rf,
          xgb_nrounds = xgb_nrounds,
          cv_folds = cv_folds,
          xgb_nthread = xgb_nthread
        ),
        n_jobs = run_n_jobs
      )

      for (res in run_results) {
        if (is.null(res)) next
        selectors[[length(selectors) + 1]] <<- res$sel_feats
        models[[length(models) + 1]] <<- res$model
        aucs <- c(aucs, res$auc)
        if (!is.null(res$mean_tpr)) {
          all_tprs <- rbind(all_tprs, res$mean_tpr)
        }
      }

      # Train final selector on all imputed data
      set.seed(42)
      rf_final <- randomForest(
        x = as.matrix(X_imputed), y = factor(y_filtered),
        ntree = n_estimators_rf, importance = TRUE
      )
      imp_final <- importance(rf_final, type = 1)[, 1]
      final_sel_feats <- sort(names(imp_final[imp_final > mean(imp_final)]))
      if (length(final_sel_feats) < 2) {
        final_sel_feats <- sort(names(sort(imp_final, decreasing = TRUE))[1:min(10, length(imp_final))])
      }
      selected_features <<- final_sel_feats

      # Train final XGBoost on raw data with selected features
      X_final <- as.matrix(.self$training_data[, selected_features, drop = FALSE])
      dfinal <- xgb.DMatrix(data = X_final, label = y_filtered)
      params_final <- list(
        objective = "binary:logistic",
        eval_metric = "logloss",
        max_depth = 6,
        eta = 0.1,
        nthread = xgb_nthread
      )
      final_model <<- xgb.train(params = params_final, data = dfinal, nrounds = xgb_nrounds, verbose = 0)

      # Plot
      if (nrow(all_tprs) > 0) {
        median_tpr <- apply(all_tprs, 2, median)
        lower_tpr  <- apply(all_tprs, 2, function(x) quantile(x, 0.025))
        upper_tpr  <- apply(all_tprs, 2, function(x) quantile(x, 0.975))
        mean_auc   <- mean(aucs, na.rm = TRUE)

        p <- ggplot2::ggplot() +
          ggplot2::geom_ribbon(ggplot2::aes(x = mean_fpr, ymin = lower_tpr, ymax = upper_tpr),
                               fill = "grey", alpha = 0.3) +
          ggplot2::geom_line(ggplot2::aes(x = mean_fpr, y = median_tpr),
                             color = "blue", linewidth = 1) +
          ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                               color = "navy", alpha = 0.8) +
          ggplot2::labs(x = "False Positive Rate", y = "True Positive Rate",
                        title = "Aggregated ROC Curve (XGBoost with RF Feature Selection)",
                        subtitle = sprintf("Median AUC = %.2f", mean_auc)) +
          ggplot2::xlim(0, 1) + ggplot2::ylim(0, 1.05) +
          ggplot2::theme_minimal()

        figures[[length(figures) + 1]] <<- p
        print(p)
      }
    },

    classify_dataframe = function(validation_df, validation_cat_df, fill_na_strategy = "zero") {
      if (is.null(final_model) || is.null(selected_features)) {
        stop("No trained model available. Run 'classify_and_plot' first.")
      }

      fill_na <<- fill_na_strategy
      shared_ids <- intersect(rownames(validation_df), rownames(validation_cat_df))
      if (length(shared_ids) == 0) return(invisible(NULL))

      val_mat <- matrix(NA_real_, nrow = length(shared_ids), ncol = length(selected_features),
                        dimnames = list(shared_ids, selected_features))
      shared_features <- intersect(selected_features, colnames(validation_df))
      if (length(shared_features) > 0) {
        val_mat[, shared_features] <- as.matrix(validation_df[shared_ids, shared_features, drop = FALSE])
      }

      if (fill_na_strategy == "zero") {
        val_mat[is.na(val_mat)] <- 0
      }

      dmat <- xgb.DMatrix(data = val_mat)
      probs <- as.numeric(predict(final_model, dmat))
      true_labels <- validation_cat_df[shared_ids, between, drop = TRUE]

      new_preds <- lapply(seq_along(shared_ids), function(i) {
        list(
          sample = shared_ids[i],
          true_label = true_labels[i],
          prob = probs[i]
        )
      })
      predictions <<- c(predictions, new_preds)
      message(sprintf("Classified %d samples...", length(.self$predictions)))
    },

    plot_validation_roc = function(category1, category2) {
      if (length(predictions) == 0) {
        stop("No predictions available. Run 'classify_dataframe' first.")
      }

      eval_data <- tryCatch(
        .prepare_binary_eval_xgb(predictions, category1, category2, "validation"),
        error = function(e) {
          message(conditionMessage(e))
          NULL
        }
      )
      if (is.null(eval_data)) return(invisible(NULL))
      roc_obj <- roc(eval_data$true_numeric, eval_data$pred_probs, quiet = TRUE)
      auc_val <- as.numeric(auc(roc_obj))

      plot_df <- data.frame(fpr = 1 - roc_obj$specificities,
                            tpr = roc_obj$sensitivities)

      p <- ggplot2::ggplot(plot_df, ggplot2::aes(x = fpr, y = tpr)) +
        ggplot2::geom_line(color = "red", linewidth = 1) +
        ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                             color = "navy", alpha = 0.8) +
        ggplot2::labs(x = "False Positive Rate", y = "True Positive Rate",
                      title = "Validation ROC Curve (XGBoost with RF Feature Selection)",
                      subtitle = sprintf("AUC = %.2f", auc_val)) +
        ggplot2::xlim(0, 1) + ggplot2::ylim(0, 1.05) +
        ggplot2::theme_minimal()

      figures[[length(figures) + 1]] <<- p
      print(p)
    },

    save_plots_to_pdf = function(file_name) {
      if (length(figures) == 0) {
        message("No figures available. Run classification first.")
        return(invisible(NULL))
      }
      pdf(file_name, width = 8, height = 6)
      for (fig in figures) {
        if (inherits(fig, "ggplot")) print(fig)
      }
      dev.off()
      message(sprintf("Plots saved to %s.", file_name))
    },

    get_feature_importance = function() {
      if (is.null(final_model)) {
        stop("No trained model available. Run 'classify_and_plot' first.")
      }
      imp <- xgb.importance(model = final_model, feature_names = selected_features)
      return(imp)
    }
  )
)


# -----------------------------------------------------------------------------
# XGBRFClassifierFolder
# Folder-based variant for individual sample files
# -----------------------------------------------------------------------------

XGBRFClassifierFolder <- setRefClass(
  "XGBRFClassifierFolder",
  fields = list(
    prot_df           = "data.frame",
    cat_df            = "data.frame",
    gene_dict         = "ANY",
    between           = "character",
    cohort            = "ANY",
    figures           = "list",
    models            = "list",
    selectors         = "list",
    predictions       = "list",
    feature_names     = "character",
    training_data     = "ANY",
    training_targets  = "ANY",
    final_model       = "ANY",
    final_selector    = "ANY",
    selected_features = "ANY",
    fill_na           = "ANY"
  ),
  methods = list(
    initialize = function(prot_df, cat_df, gene_dict = NULL, between = NULL, cohort = NULL) {
      prot_df <<- prot_df[, !duplicated(colnames(prot_df)), drop = FALSE]
      cat_df <<- cat_df
      gene_dict <<- gene_dict
      between <<- if (is.null(between)) "" else between
      cohort <<- cohort
      figures <<- list()
      models <<- list()
      selectors <<- list()
      predictions <<- list()
      feature_names <<- colnames(prot_df)
      training_data <<- NULL
      training_targets <<- NULL
      final_model <<- NULL
      final_selector <<- NULL
      selected_features <<- NULL
      fill_na <<- NULL
    },

    classify_and_plot = function(category1, category2, n_runs = 10, n_estimators_rf = 200, n_jobs = 1) {
      if (between == "") {
        stop("No between variable given.")
      }

      n_estimators_rf <- .get_env_int_xgb("ADAPTMS_RF_TREES", n_estimators_rf, min_value = 1L)
      xgb_nrounds <- .get_env_int_xgb("ADAPTMS_XGB_NROUNDS", 100L, min_value = 1L)
      cv_folds <- .get_env_int_xgb("ADAPTMS_CV_FOLDS", 5L, min_value = 2L)
      run_n_jobs <- .resolve_n_jobs_xgb(n_jobs)
      xgb_nthread_default <- if (run_n_jobs > 1L) 1L else run_n_jobs
      xgb_nthread <- .get_env_int_xgb("ADAPTMS_XGB_NTHREADS", xgb_nthread_default, min_value = 1L)

      shared_ids <- intersect(rownames(.self$prot_df), rownames(.self$cat_df))
      d_ML <- cbind(.self$prot_df[shared_ids, , drop = FALSE],
                    .self$cat_df[shared_ids, between, drop = FALSE])

      X_raw <- d_ML[, colnames(d_ML) != between, drop = FALSE]
      X_raw[] <- lapply(X_raw, function(x) as.numeric(as.character(x)))
      y <- d_ML[[between]]

      keep <- y %in% c(category1, category2)
      X_filtered <- X_raw[keep, , drop = FALSE]
      y_filtered <- ifelse(y[keep] == category1, 0, 1)
      names(y_filtered) <- rownames(X_filtered)
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

      training_data <<- X_filtered
      training_targets <<- y_filtered

      X_imputed <- as.data.frame(kNN(X_filtered, k = 5, imp_var = FALSE))
      rownames(X_imputed) <- rownames(X_filtered)

      mean_fpr <- seq(0, 1, length.out = 100)
      all_tprs <- matrix(nrow = 0, ncol = 100)
      aucs <- numeric(0)

      run_results <- .parallel_lapply_xgb(
        seq_len(n_runs),
        function(i) .run_xgb_rf_iteration(
          seed = i,
          X_filtered = X_filtered,
          X_imputed = X_imputed,
          y_filtered = y_filtered,
          n_estimators_rf = n_estimators_rf,
          xgb_nrounds = xgb_nrounds,
          cv_folds = cv_folds,
          xgb_nthread = xgb_nthread
        ),
        n_jobs = run_n_jobs
      )

      for (res in run_results) {
        if (is.null(res)) next
        selectors[[length(selectors) + 1]] <<- res$sel_feats
        models[[length(models) + 1]] <<- res$model
        aucs <- c(aucs, res$auc)
        if (!is.null(res$mean_tpr)) {
          all_tprs <- rbind(all_tprs, res$mean_tpr)
        }
      }

      # Final model
      set.seed(42)
      rf_final <- randomForest(
        x = as.matrix(X_imputed), y = factor(y_filtered),
        ntree = n_estimators_rf, importance = TRUE
      )
      imp_final <- importance(rf_final, type = 1)[, 1]
      final_sel_feats <- sort(names(imp_final[imp_final > mean(imp_final)]))
      if (length(final_sel_feats) < 2) {
        final_sel_feats <- sort(names(sort(imp_final, decreasing = TRUE))[1:min(10, length(imp_final))])
      }
      selected_features <<- final_sel_feats

      X_final <- as.matrix(.self$training_data[, selected_features, drop = FALSE])
      dfinal <- xgb.DMatrix(data = X_final, label = y_filtered)
      params_final <- list(
        objective = "binary:logistic",
        eval_metric = "logloss",
        max_depth = 6,
        eta = 0.1,
        nthread = xgb_nthread
      )
      final_model <<- xgb.train(params = params_final, data = dfinal, nrounds = xgb_nrounds, verbose = 0)

      if (nrow(all_tprs) > 0) {
        median_tpr <- apply(all_tprs, 2, median)
        lower_tpr  <- apply(all_tprs, 2, function(x) quantile(x, 0.025))
        upper_tpr  <- apply(all_tprs, 2, function(x) quantile(x, 0.975))
        mean_auc   <- mean(aucs, na.rm = TRUE)

        p <- ggplot2::ggplot() +
          ggplot2::geom_ribbon(ggplot2::aes(x = mean_fpr, ymin = lower_tpr, ymax = upper_tpr),
                               fill = "grey", alpha = 0.3) +
          ggplot2::geom_line(ggplot2::aes(x = mean_fpr, y = median_tpr),
                             color = "blue", linewidth = 1) +
          ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                               color = "navy", alpha = 0.8) +
          ggplot2::labs(x = "False Positive Rate", y = "True Positive Rate",
                        title = "Aggregated ROC Curve (XGBoost with RF Feature Selection)",
                        subtitle = sprintf("Median AUC = %.2f", mean_auc)) +
          ggplot2::xlim(0, 1) + ggplot2::ylim(0, 1.05) +
          ggplot2::theme_minimal()

        figures[[length(figures) + 1]] <<- p
        print(p)
      }
    },

    classify_sample = function(d_sample, true_label) {
      if (is.null(final_model) || is.null(selected_features)) {
        stop("No trained model available. Run 'classify_and_plot' first.")
      }

      if (length(selected_features) == 0) return(invisible(NULL))

      d_sample_aligned <- matrix(NA_real_, nrow = 1, ncol = length(selected_features),
                                 dimnames = list(rownames(d_sample)[1], selected_features))
      shared_features <- intersect(selected_features, colnames(d_sample))
      if (length(shared_features) > 0) {
        d_sample_aligned[1, shared_features] <- as.numeric(d_sample[1, shared_features, drop = TRUE])
      }

      if (!is.null(fill_na) && fill_na == "zero") {
        d_sample_aligned[is.na(d_sample_aligned)] <- 0
      }

      dmat <- xgb.DMatrix(data = d_sample_aligned)
      prob <- predict(final_model, dmat)
      predictions[[length(predictions) + 1]] <<- list(
        sample = rownames(d_sample)[1], true_label = true_label, prob = prob
      )
    },

    classify_directory = function(directory, cat_validation_pool_SF, category1, category2,
                                   fill_na_strategy = "zero") {
      fill_na <<- fill_na_strategy

      files <- list.files(directory, pattern = "mzML\\.pg_matrix\\.tsv$", full.names = TRUE)

      for (fpath in files) {
        fname <- basename(fpath)
        if (!is.null(.self$cohort) && !grepl(.self$cohort, fname)) next

        d_sample <- read.delim(fpath, sep = "\t", check.names = FALSE)
        d_sample <- d_sample[, c(1, 6), drop = FALSE]
        sample_col_name <- basename(sub(".*/", "", colnames(d_sample)[2]))
        colnames(d_sample) <- c("Protein.Group", sample_col_name)
        rownames(d_sample) <- .sanitize_row_ids_xgb(d_sample$Protein.Group, prefix = "protein_group")
        d_sample$Protein.Group <- NULL

        d_sample[] <- lapply(d_sample, function(x) {
          x <- suppressWarnings(as.numeric(x))
          x[x <= 0] <- NA_real_
          log10(x)
        })
        d_sample <- as.data.frame(t(d_sample))
        sample_name <- rownames(d_sample)[1]

        if (sample_name %in% rownames(cat_validation_pool_SF)) {
          true_label <- cat_validation_pool_SF[sample_name, between]
          if (true_label %in% c(category1, category2)) {
            classify_sample(d_sample, true_label)
          }
        } else {
          message(sprintf("Warning: Sample %s not found in validation metadata", sample_name))
        }
      }
    },

    plot_accumulated_roc = function(category1, category2) {
      if (length(predictions) == 0) {
        stop("No predictions available. Run 'classify_directory' first.")
      }

      eval_data <- tryCatch(
        .prepare_binary_eval_xgb(predictions, category1, category2, "directory"),
        error = function(e) {
          message(conditionMessage(e))
          NULL
        }
      )
      if (is.null(eval_data)) return(invisible(NULL))
      roc_obj <- roc(eval_data$true_numeric, eval_data$pred_probs, quiet = TRUE)
      auc_val <- as.numeric(auc(roc_obj))

      plot_df <- data.frame(fpr = 1 - roc_obj$specificities,
                            tpr = roc_obj$sensitivities)

      p <- ggplot2::ggplot(plot_df, ggplot2::aes(x = fpr, y = tpr)) +
        ggplot2::geom_line(color = "red", linewidth = 1) +
        ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                             color = "navy", alpha = 0.8) +
        ggplot2::labs(x = "False Positive Rate", y = "True Positive Rate",
                      title = "Validation ROC Curve (XGBoost with RF Feature Selection)",
                      subtitle = sprintf("AUC = %.2f", auc_val)) +
        ggplot2::xlim(0, 1) + ggplot2::ylim(0, 1.05) +
        ggplot2::theme_minimal()

      figures[[length(figures) + 1]] <<- p
      print(p)

      message(sprintf("Total samples classified: %d", length(predictions)))
      message(sprintf("AUC: %.3f", auc_val))
      return(auc_val)
    },

    save_plots_to_pdf = function(file_name) {
      if (length(figures) == 0) {
        message("No figures available. Run classification first.")
        return(invisible(NULL))
      }
      pdf(file_name, width = 8, height = 6)
      for (fig in figures) {
        if (inherits(fig, "ggplot")) print(fig)
      }
      dev.off()
      message(sprintf("Plots saved to %s.", file_name))
    },

    get_feature_importance = function() {
      if (is.null(final_model)) {
        stop("No trained model available. Run 'classify_and_plot' first.")
      }
      return(xgb.importance(model = final_model, feature_names = selected_features))
    },

    get_predictions_df = function() {
      if (length(predictions) == 0) {
        stop("No predictions available. Run 'classify_directory' first.")
      }
      data.frame(
        sample = sapply(predictions, function(x) x$sample),
        true_label = sapply(predictions, function(x) x$true_label),
        predicted_probability = sapply(predictions, function(x) x$prob),
        stringsAsFactors = FALSE
      )
    }
  )
)
