# =============================================================================
# BaselineClassifier.R
# Baseline RF feature selection + XGBoost classifier
# R equivalent of utils/BaselineClassifier.py
# =============================================================================

library(xgboost)
library(randomForest)
library(caret)
library(pROC)
library(VIM)

.impute_with_reference <- function(reference_df, target_df, k = 5) {
  if (nrow(target_df) == 0) {
    return(target_df)
  }

  ref <- reference_df
  tgt <- target_df
  ref_ids <- paste0("ref__", seq_len(nrow(ref)))
  tgt_ids <- paste0("target__", seq_len(nrow(tgt)))
  rownames(ref) <- ref_ids
  rownames(tgt) <- tgt_ids

  combined <- rbind(ref, tgt)
  combined_imp <- as.data.frame(kNN(combined, k = k, imp_var = FALSE))
  tgt_imp <- combined_imp[tgt_ids, , drop = FALSE]
  rownames(tgt_imp) <- rownames(target_df)
  tgt_imp
}

.get_env_int_baseline <- function(var_name, default_value, min_value = 1L) {
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

BaselineClassifier <- setRefClass(
  "BaselineClassifier",
  fields = list(
    prot_df         = "data.frame",
    cat_df          = "data.frame",
    gene_dict       = "ANY",
    between         = "character",
    figures         = "list",
    models          = "list",
    selectors       = "list",
    imputers        = "list",
    feature_columns = "ANY"
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
      imputers <<- list()
      feature_columns <<- NULL
    },

    classify_and_plot = function(category1, category2, n_runs = 10, n_estimators = 200) {
      if (between == "") {
        stop("The 'between' attribute must be set to a valid column name.")
      }

      rf_trees <- .get_env_int_baseline("ADAPTMS_RF_TREES", n_estimators, min_value = 1L)
      xgb_nrounds <- .get_env_int_baseline("ADAPTMS_XGB_NROUNDS", 100L, min_value = 1L)
      cv_folds <- .get_env_int_baseline("ADAPTMS_CV_FOLDS", 5L, min_value = 2L)

      # Merge on row names
      shared_ids <- intersect(rownames(.self$prot_df), rownames(.self$cat_df))
      d_ML <- cbind(.self$prot_df[shared_ids, , drop = FALSE],
                    .self$cat_df[shared_ids, , drop = FALSE])

      feature_columns <<- setdiff(colnames(d_ML), between)

      X <- d_ML[, feature_columns, drop = FALSE]
      y <- d_ML[[between]]

      # Filter to two categories
      keep <- y %in% c(category1, category2)
      X_filtered <- X[keep, , drop = FALSE]
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

      mean_fpr <- seq(0, 1, length.out = 100)
      all_tprs <- matrix(nrow = 0, ncol = 100)
      aucs <- numeric(0)

      for (i in seq_len(n_runs)) {
        set.seed(i - 1)
        train_idx <- createDataPartition(factor(y_filtered, levels = c(0, 1)),
                                         p = 0.8, list = FALSE)[, 1]
        X_train <- X_filtered[train_idx, , drop = FALSE]
        X_test  <- X_filtered[-train_idx, , drop = FALSE]
        y_train <- y_filtered[train_idx]
        y_test  <- y_filtered[-train_idx]

        # VIM::kNN errors when columns are entirely NA.
        # Drop such columns for this run and align test columns accordingly.
        non_all_na <- colSums(!is.na(X_train)) > 0
        if (!any(non_all_na)) {
          warning(sprintf("Run %d skipped: all training features are completely NA.", i))
          next
        }
        X_train <- X_train[, non_all_na, drop = FALSE]
        X_test  <- X_test[, colnames(X_train), drop = FALSE]

        # KNN impute train and test
        X_train_imp <- as.data.frame(kNN(X_train, k = 5, imp_var = FALSE))
        rownames(X_train_imp) <- rownames(X_train)

        # For test: impute with train rows as context while preserving exact row alignment.
        X_test_imp <- .impute_with_reference(X_train, X_test, k = 5)

        # Store imputer reference data for validation
        imputers[[length(imputers) + 1]] <<- X_train_imp

        # Feature selection using Random Forest
        rf_model <- randomForest(
          x = as.matrix(X_train_imp), y = factor(y_train),
          ntree = rf_trees, importance = TRUE
        )
        importance_vals <- importance(rf_model, type = 1)[, 1]  # Mean decrease accuracy
        # Select features above mean importance (mimics SelectFromModel)
        selected_feats <- names(importance_vals[importance_vals > mean(importance_vals)])
        if (length(selected_feats) < 2) {
          selected_feats <- names(sort(importance_vals, decreasing = TRUE))[1:min(10, length(importance_vals))]
        }
        selectors[[length(selectors) + 1]] <<- selected_feats

        X_train_sel <- X_train_imp[, selected_feats, drop = FALSE]
        X_test_sel  <- X_test_imp[, selected_feats, drop = FALSE]

        # Train XGBoost
        dtrain <- xgb.DMatrix(data = as.matrix(X_train_sel), label = y_train)
        dtest  <- xgb.DMatrix(data = as.matrix(X_test_sel), label = y_test)

        params <- list(
          objective = "binary:logistic",
          eval_metric = "logloss",
          max_depth = 6,
          eta = 0.1
        )
        xgb_model <- xgb.train(params = params, data = dtrain, nrounds = xgb_nrounds,
                                verbose = 0)
        models[[length(models) + 1]] <<- xgb_model

        # 5-fold CV for ROC
        k_folds <- max(2L, min(cv_folds, length(y_train)))
        folds <- createFolds(y_train, k = k_folds, list = TRUE, returnTrain = TRUE)
        cv_tprs <- matrix(nrow = 0, ncol = 100)

        for (fold_train_idx in folds) {
          fold_val_idx <- setdiff(seq_along(y_train), fold_train_idx)

          dcv_train <- xgb.DMatrix(data = as.matrix(X_train_sel[fold_train_idx, , drop = FALSE]),
                                    label = y_train[fold_train_idx])
          dcv_val <- xgb.DMatrix(data = as.matrix(X_train_sel[fold_val_idx, , drop = FALSE]))
          y_cv_val <- y_train[fold_val_idx]

          cv_model <- xgb.train(params = params, data = dcv_train, nrounds = xgb_nrounds,
                                 verbose = 0)
          probas <- predict(cv_model, dcv_val)

          if (length(unique(y_cv_val)) > 1) {
            roc_obj <- roc(y_cv_val, probas, quiet = TRUE)
            interp_tpr <- approx(1 - roc_obj$specificities, roc_obj$sensitivities,
                                  xout = mean_fpr, rule = 2)$y
            interp_tpr[1] <- 0
            cv_tprs <- rbind(cv_tprs, interp_tpr)
          }
        }

        # Test AUC
        test_probs <- predict(xgb_model, dtest)
        if (length(unique(y_test)) > 1) {
          test_roc <- roc(y_test, test_probs, quiet = TRUE)
          auc_val <- as.numeric(auc(test_roc))
        } else {
          auc_val <- NA
        }
        aucs <- c(aucs, auc_val)

        if (nrow(cv_tprs) > 0) {
          mean_tpr_fold <- colMeans(cv_tprs)
          mean_tpr_fold[100] <- 1.0
          all_tprs <- rbind(all_tprs, mean_tpr_fold)
        }
      }

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
                        title = "Aggregated ROC Curve",
                        subtitle = sprintf("Median AUC = %.2f", mean_auc)) +
          ggplot2::xlim(0, 1) + ggplot2::ylim(0, 1.05) +
          ggplot2::theme_minimal()

        figures[[length(figures) + 1]] <<- p
        print(p)
      }
    },

    validate_and_plot = function(validation_prot_df, validation_cat_df, category1, category2) {
      if (length(models) == 0 || length(selectors) == 0) {
        stop("No trained models available. Run 'classify_and_plot' first.")
      }
      if (is.null(feature_columns)) {
        stop("Feature columns not set. Run 'classify_and_plot' first.")
      }

      validation_prot_df <- validation_prot_df[, !duplicated(colnames(validation_prot_df)), drop = FALSE]

      shared_ids <- intersect(rownames(validation_prot_df), rownames(validation_cat_df))
      val_data <- cbind(validation_prot_df[shared_ids, , drop = FALSE],
                        validation_cat_df[shared_ids, , drop = FALSE])

      # Reindex to match training features
      missing_cols <- setdiff(feature_columns, colnames(val_data))
      for (mc in missing_cols) val_data[[mc]] <- NA

      X_val <- val_data[, feature_columns, drop = FALSE]
      y_val <- val_data[[between]]

      keep <- y_val %in% c(category1, category2)
      X_val <- X_val[keep, , drop = FALSE]
      y_val <- ifelse(y_val[keep] == category1, 0, 1)

      mean_fpr <- seq(0, 1, length.out = 100)
      all_tprs <- matrix(nrow = 0, ncol = 100)
      aucs <- numeric(0)

      for (k in seq_along(models)) {
        xgb_model <- models[[k]]
        selected_feats <- selectors[[k]]
        imputer_ref <- imputers[[k]]

        # Keep selected features in training order; add missing validation columns as NA.
        selected_feats <- unique(selected_feats)
        if (length(selected_feats) == 0) next
        missing_val_cols <- setdiff(selected_feats, colnames(X_val))
        for (mc in missing_val_cols) X_val[[mc]] <- NA

        # Impute only model-selected features to avoid broad alignment issues.
        ref_feats <- intersect(selected_feats, colnames(imputer_ref))
        if (length(ref_feats) == 0) next
        X_val_imp <- .impute_with_reference(
          imputer_ref[, ref_feats, drop = FALSE],
          X_val[, ref_feats, drop = FALSE],
          k = 5
        )

        # Require exactly the trained feature set for this model.
        if (!all(selected_feats %in% colnames(X_val_imp))) next

        X_val_sel <- X_val_imp[, selected_feats, drop = FALSE]

        dval <- xgb.DMatrix(data = as.matrix(X_val_sel))
        y_pred_prob <- predict(xgb_model, dval)

        if (length(unique(y_val)) > 1) {
          roc_obj <- roc(y_val, y_pred_prob, quiet = TRUE)
          auc_val <- as.numeric(auc(roc_obj))
          interp_tpr <- approx(1 - roc_obj$specificities, roc_obj$sensitivities,
                                xout = mean_fpr, rule = 2)$y
          all_tprs <- rbind(all_tprs, interp_tpr)
          aucs <- c(aucs, auc_val)
        }
      }

      if (nrow(all_tprs) > 0) {
        median_tpr <- apply(all_tprs, 2, median)
        lower_tpr  <- apply(all_tprs, 2, function(x) quantile(x, 0.025))
        upper_tpr  <- apply(all_tprs, 2, function(x) quantile(x, 0.975))
        mean_auc   <- mean(aucs, na.rm = TRUE)

        p <- ggplot2::ggplot() +
          ggplot2::geom_ribbon(ggplot2::aes(x = mean_fpr, ymin = lower_tpr, ymax = upper_tpr),
                               fill = "grey", alpha = 0.3) +
          ggplot2::geom_line(ggplot2::aes(x = mean_fpr, y = median_tpr),
                             color = "green", linewidth = 1) +
          ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed",
                               color = "navy", alpha = 0.8) +
          ggplot2::labs(x = "False Positive Rate", y = "True Positive Rate",
                        title = "Validation ROC Curve",
                        subtitle = sprintf("Median Validation AUC = %.2f", mean_auc)) +
          ggplot2::xlim(0, 1) + ggplot2::ylim(0, 1.05) +
          ggplot2::theme_minimal()

        figures[[length(figures) + 1]] <<- p
        print(p)
      }
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
    }
  )
)
