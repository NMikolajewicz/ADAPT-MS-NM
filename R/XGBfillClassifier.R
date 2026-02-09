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

    classify_and_plot = function(category1, category2, n_runs = 10, n_estimators_rf = 200) {
      if (between == "") {
        stop("No between variable given. Please provide a variable for classification.")
      }

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

      training_data <<- X_filtered
      training_targets <<- y_filtered

      # Impute for feature selection only
      X_imputed <- as.data.frame(kNN(X_filtered, k = 5, imp_var = FALSE))
      rownames(X_imputed) <- rownames(X_filtered)

      mean_fpr <- seq(0, 1, length.out = 100)
      all_tprs <- matrix(nrow = 0, ncol = 100)
      aucs <- numeric(0)

      for (i in seq_len(n_runs)) {
        set.seed(i - 1)
        train_idx <- createDataPartition(y_filtered, p = 0.8, list = FALSE)[, 1]

        X_train_imp <- as.matrix(X_imputed[train_idx, , drop = FALSE])
        X_test_imp  <- as.matrix(X_imputed[-train_idx, , drop = FALSE])
        X_train_raw <- X_filtered[train_idx, , drop = FALSE]
        X_test_raw  <- X_filtered[-train_idx, , drop = FALSE]
        y_train <- y_filtered[train_idx]
        y_test  <- y_filtered[-train_idx]

        # RF feature selection on imputed data
        rf_model <- randomForest(
          x = X_train_imp, y = factor(y_train),
          ntree = n_estimators_rf, importance = TRUE
        )
        importance_vals <- importance(rf_model, type = 1)[, 1]
        sel_feats <- names(importance_vals[importance_vals > mean(importance_vals)])
        if (length(sel_feats) < 2) {
          sel_feats <- names(sort(importance_vals, decreasing = TRUE))[1:min(10, length(importance_vals))]
        }
        selectors[[length(selectors) + 1]] <<- sel_feats

        # Train XGBoost on raw data (XGBoost handles NaN natively)
        X_train_sel <- as.matrix(X_train_raw[, sel_feats, drop = FALSE])
        X_test_sel  <- as.matrix(X_test_raw[, sel_feats, drop = FALSE])

        dtrain <- xgb.DMatrix(data = X_train_sel, label = y_train)

        params <- list(
          objective = "binary:logistic",
          eval_metric = "logloss",
          max_depth = 6,
          eta = 0.1
        )

        xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100, verbose = 0)
        models[[length(models) + 1]] <<- xgb_model

        # 5-fold CV
        folds <- createFolds(y_train, k = 5, list = TRUE, returnTrain = TRUE)
        cv_tprs <- matrix(nrow = 0, ncol = 100)

        for (fold_train_idx in folds) {
          fold_val_idx <- setdiff(seq_along(y_train), fold_train_idx)
          dcv_train <- xgb.DMatrix(data = X_train_sel[fold_train_idx, , drop = FALSE],
                                    label = y_train[fold_train_idx])
          dcv_val <- xgb.DMatrix(data = X_train_sel[fold_val_idx, , drop = FALSE])
          y_cv_val <- y_train[fold_val_idx]

          cv_model <- xgb.train(params = params, data = dcv_train, nrounds = 100, verbose = 0)
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
        dtest <- xgb.DMatrix(data = X_test_sel)
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
      final_model <<- xgb.train(params = params, data = dfinal, nrounds = 100, verbose = 0)

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

      for (idx in shared_ids) {
        row_data <- validation_df[idx, , drop = FALSE]

        # Create aligned sample with all selected features
        d_sample <- data.frame(matrix(NA_real_, nrow = 1, ncol = length(selected_features)))
        colnames(d_sample) <- selected_features
        rownames(d_sample) <- idx

        for (feat in selected_features) {
          if (feat %in% colnames(row_data) && !is.na(row_data[1, feat])) {
            d_sample[1, feat] <- as.numeric(row_data[1, feat])
          }
        }

        # Handle NaN strategy
        if (fill_na_strategy == "zero") {
          d_sample[is.na(d_sample)] <- 0
        }
        # If "keep", XGBoost handles NAs natively

        true_label <- validation_cat_df[idx, between]

        dmat <- xgb.DMatrix(data = as.matrix(d_sample))
        prob <- predict(final_model, dmat)
        predictions[[length(predictions) + 1]] <<- list(
          sample = idx, true_label = true_label, prob = prob
        )

        if (length(.self$predictions) %% 10 == 0) {
          message(sprintf("Classified %d samples...", length(.self$predictions)))
        }
      }
    },

    plot_validation_roc = function(category1, category2) {
      if (length(predictions) == 0) {
        stop("No predictions available. Run 'classify_dataframe' first.")
      }

      true_labels <- sapply(predictions, function(x) x$true_label)
      pred_probs  <- sapply(predictions, function(x) x$prob)
      label_map <- c(setNames(0, category1), setNames(1, category2))
      true_numeric <- label_map[true_labels]

      roc_obj <- roc(true_numeric, pred_probs, quiet = TRUE)
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

    classify_and_plot = function(category1, category2, n_runs = 10, n_estimators_rf = 200) {
      if (between == "") {
        stop("No between variable given.")
      }

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

      training_data <<- X_filtered
      training_targets <<- y_filtered

      X_imputed <- as.data.frame(kNN(X_filtered, k = 5, imp_var = FALSE))
      rownames(X_imputed) <- rownames(X_filtered)

      mean_fpr <- seq(0, 1, length.out = 100)
      all_tprs <- matrix(nrow = 0, ncol = 100)
      aucs <- numeric(0)

      params <- list(
        objective = "binary:logistic",
        eval_metric = "logloss",
        max_depth = 6,
        eta = 0.1
      )

      for (i in seq_len(n_runs)) {
        set.seed(i - 1)
        train_idx <- createDataPartition(y_filtered, p = 0.8, list = FALSE)[, 1]

        X_train_imp <- as.matrix(X_imputed[train_idx, , drop = FALSE])
        X_train_raw <- X_filtered[train_idx, , drop = FALSE]
        X_test_raw  <- X_filtered[-train_idx, , drop = FALSE]
        y_train <- y_filtered[train_idx]
        y_test  <- y_filtered[-train_idx]

        rf_model <- randomForest(
          x = X_train_imp, y = factor(y_train),
          ntree = n_estimators_rf, importance = TRUE
        )
        imp_vals <- importance(rf_model, type = 1)[, 1]
        sel_feats <- names(imp_vals[imp_vals > mean(imp_vals)])
        if (length(sel_feats) < 2) {
          sel_feats <- names(sort(imp_vals, decreasing = TRUE))[1:min(10, length(imp_vals))]
        }
        selectors[[length(selectors) + 1]] <<- sel_feats

        X_train_sel <- as.matrix(X_train_raw[, sel_feats, drop = FALSE])
        X_test_sel  <- as.matrix(X_test_raw[, sel_feats, drop = FALSE])

        dtrain <- xgb.DMatrix(data = X_train_sel, label = y_train)
        xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100, verbose = 0)
        models[[length(models) + 1]] <<- xgb_model

        # 5-fold CV
        folds <- createFolds(y_train, k = 5, list = TRUE, returnTrain = TRUE)
        cv_tprs <- matrix(nrow = 0, ncol = 100)

        for (fold_train_idx in folds) {
          fold_val_idx <- setdiff(seq_along(y_train), fold_train_idx)
          dcv_train <- xgb.DMatrix(data = X_train_sel[fold_train_idx, , drop = FALSE],
                                    label = y_train[fold_train_idx])
          dcv_val <- xgb.DMatrix(data = X_train_sel[fold_val_idx, , drop = FALSE])
          y_cv_val <- y_train[fold_val_idx]

          cv_model <- xgb.train(params = params, data = dcv_train, nrounds = 100, verbose = 0)
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
      final_model <<- xgb.train(params = params, data = dfinal, nrounds = 100, verbose = 0)

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

      d_sample_aligned <- data.frame(matrix(NA_real_, nrow = 1, ncol = length(selected_features)))
      colnames(d_sample_aligned) <- selected_features
      rownames(d_sample_aligned) <- rownames(d_sample)[1]

      for (feat in selected_features) {
        if (feat %in% colnames(d_sample) && !is.na(d_sample[1, feat])) {
          d_sample_aligned[1, feat] <- as.numeric(d_sample[1, feat])
        }
      }

      if (!is.null(fill_na) && fill_na == "zero") {
        d_sample_aligned[is.na(d_sample_aligned)] <- 0
      }

      dmat <- xgb.DMatrix(data = as.matrix(d_sample_aligned))
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
        rownames(d_sample) <- d_sample$Protein.Group
        d_sample$Protein.Group <- NULL

        d_sample[] <- log10(d_sample)
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

      true_labels <- sapply(predictions, function(x) x$true_label)
      pred_probs  <- sapply(predictions, function(x) x$prob)
      label_map <- c(setNames(0, category1), setNames(1, category2))
      true_numeric <- label_map[true_labels]

      roc_obj <- roc(true_numeric, pred_probs, quiet = TRUE)
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
