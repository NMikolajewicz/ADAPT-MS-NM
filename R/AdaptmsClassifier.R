# =============================================================================
# AdaptmsClassifier.R
# Binary ADAPT-MS classifiers: DataFrame-based and Folder-based
# R equivalent of utils/AdaptmsClassifier.py
# =============================================================================

library(glmnet)
library(caret)
library(pROC)
library(VIM)

.prepare_binary_eval <- function(predictions, category1, category2, context) {
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

.normalize_confusion_matrix <- function(cm) {
  rs <- rowSums(cm)
  cm_norm <- matrix(0, nrow = nrow(cm), ncol = ncol(cm),
                    dimnames = dimnames(cm))
  valid_rows <- rs > 0
  if (any(valid_rows)) {
    cm_norm[valid_rows, ] <- cm[valid_rows, , drop = FALSE] / rs[valid_rows]
  }
  cm_norm
}

.sanitize_row_ids <- function(x, prefix = "row") {
  ids <- trimws(as.character(x))
  missing <- is.na(ids) | ids == ""
  if (any(missing)) {
    ids[missing] <- paste0(prefix, "_", seq_len(sum(missing)))
  }
  make.unique(ids)
}

# -----------------------------------------------------------------------------
# AdaptmsClassifierDF
# For validating against a data.frame of samples
# -----------------------------------------------------------------------------

AdaptmsClassifierDF <- setRefClass(
  "AdaptmsClassifierDF",
  fields = list(
    prot_df        = "data.frame",
    cat_df         = "data.frame",
    gene_dict      = "ANY",
    between        = "character",
    figures        = "list",
    models         = "list",
    selected_features = "character",
    predictions    = "list",
    feature_names  = "character",
    training_data  = "ANY",
    training_targets = "ANY"
  ),
  methods = list(
    initialize = function(prot_df, cat_df, gene_dict = NULL, between = NULL) {
      # Remove duplicate columns
      prot_df <<- prot_df[, !duplicated(colnames(prot_df)), drop = FALSE]
      cat_df <<- cat_df
      gene_dict <<- gene_dict
      between <<- if (is.null(between)) "" else between
      figures <<- list()
      models <<- list()
      selected_features <<- character(0)
      predictions <<- list()
      feature_names <<- colnames(prot_df)
      training_data <<- NULL
      training_targets <<- NULL
    },

    classify_and_plot = function(category1, category2, n_runs = 10, topn_features = 50) {
      if (between == "") {
        stop("No between variable given. Please provide a variable for classification.")
      }

      # Drop columns that are all NA
      all_na_cols <- sapply(.self$prot_df, function(x) all(is.na(x)))
      prot_clean <- .self$prot_df[, !all_na_cols, drop = FALSE]

      # Merge protein and category data on row names
      shared_ids <- intersect(rownames(prot_clean), rownames(.self$cat_df))
      d_ML <- cbind(prot_clean[shared_ids, , drop = FALSE],
                    .self$cat_df[shared_ids, between, drop = FALSE])

      # Separate features and target
      X_raw <- d_ML[, colnames(d_ML) != between, drop = FALSE]
      y <- d_ML[[between]]

      # Filter to only the two categories
      keep <- y %in% c(category1, category2)
      X_filtered <- X_raw[keep, , drop = FALSE]
      y_filtered <- ifelse(y[keep] == category1, 0, 1)
      names(y_filtered) <- rownames(X_filtered)

      # KNN impute the filtered data
      X_imputed <- as.data.frame(kNN(X_filtered, k = 5, imp_var = FALSE))
      rownames(X_imputed) <- rownames(X_filtered)

      # Store training data
      training_data <<- X_imputed
      training_targets <<- y_filtered

      mean_fpr <- seq(0, 1, length.out = 100)
      all_tprs <- matrix(nrow = 0, ncol = 100)
      aucs <- numeric(0)

      for (i in seq_len(n_runs)) {
        set.seed(i - 1)
        train_idx <- createDataPartition(y_filtered, p = 0.8, list = FALSE)[, 1]
        X_train_raw <- X_filtered[train_idx, , drop = FALSE]
        X_test_raw  <- X_filtered[-train_idx, , drop = FALSE]
        y_train <- y_filtered[train_idx]
        y_test  <- y_filtered[-train_idx]

        # Feature selection: t-tests on raw data (with NAs)
        p_values <- sapply(colnames(X_train_raw), function(col) {
          g1 <- X_train_raw[y_train == 0, col]
          g2 <- X_train_raw[y_train == 1, col]
          g1 <- g1[!is.na(g1)]
          g2 <- g2[!is.na(g2)]
          if (length(g1) > 0 && length(g2) > 0) {
            tryCatch(t.test(g1, g2)$p.value, error = function(e) 1.0)
          } else {
            1.0
          }
        })

        # FDR correction
        corrected_p <- p.adjust(p_values, method = "BH")

        # Select top features
        top_idx <- order(corrected_p)[seq_len(min(topn_features, length(corrected_p)))]
        top_feats <- colnames(X_train_raw)[top_idx]
        selected_features <<- unique(c(.self$selected_features, top_feats))

        # Get imputed data for selected features
        X_train_sel <- X_imputed[rownames(X_train_raw), top_feats, drop = FALSE]
        X_test_sel  <- X_imputed[rownames(X_test_raw), top_feats, drop = FALSE]

        # Train logistic regression
        train_df <- data.frame(X_train_sel, y = factor(y_train))
        model <- glm(y ~ ., data = train_df, family = binomial(link = "logit"),
                      control = list(maxit = 1000))
        models[[length(models) + 1]] <<- model

        # 5-fold CV for ROC
        folds <- createFolds(y_train, k = 5, list = TRUE, returnTrain = TRUE)
        cv_tprs <- matrix(nrow = 0, ncol = 100)

        for (fold_train_idx in folds) {
          fold_val_idx <- setdiff(seq_along(y_train), fold_train_idx)
          cv_train_df <- data.frame(X_train_sel[fold_train_idx, , drop = FALSE],
                                     y = factor(y_train[fold_train_idx]))
          cv_val_df   <- X_train_sel[fold_val_idx, , drop = FALSE]
          y_cv_val    <- y_train[fold_val_idx]

          cv_model <- glm(y ~ ., data = cv_train_df, family = binomial(link = "logit"),
                          control = list(maxit = 1000))
          probas <- predict(cv_model, newdata = cv_val_df, type = "response")

          if (length(unique(y_cv_val)) > 1) {
            roc_obj <- roc(y_cv_val, probas, quiet = TRUE)
            interp_tpr <- approx(1 - roc_obj$specificities, roc_obj$sensitivities,
                                  xout = mean_fpr, rule = 2)$y
            interp_tpr[1] <- 0
            cv_tprs <- rbind(cv_tprs, interp_tpr)
          }
        }

        # Predict on test set
        test_probs <- predict(model, newdata = X_test_sel, type = "response")
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

      # Aggregate and plot
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

    classify_dataframe = function(validation_df, validation_cat_df) {
      shared_ids <- intersect(rownames(validation_df), rownames(validation_cat_df))

      for (idx in shared_ids) {
        row_data <- validation_df[idx, , drop = FALSE]
        non_na_cols <- colnames(row_data)[!is.na(row_data[1, ])]
        available_features <- intersect(.self$selected_features, non_na_cols)
        if (length(available_features) < 2) next
        d_sample <- row_data[, available_features, drop = FALSE]

        true_label <- validation_cat_df[idx, between]

        # Re-train model using stored training data with available features
        X_train_filt <- .self$training_data[, available_features, drop = FALSE]
        y_train_filt <- .self$training_targets

        train_df <- data.frame(X_train_filt, y = factor(y_train_filt))
        model <- tryCatch(
          glm(y ~ ., data = train_df, family = binomial(link = "logit"),
              control = list(maxit = 2000)),
          error = function(e) NULL
        )

        if (!is.null(model)) {
          prob <- predict(model, newdata = d_sample, type = "response")
          predictions[[length(predictions) + 1]] <<- list(
            sample = idx, true_label = true_label, prob = as.numeric(prob)
          )
        }

        if (length(.self$predictions) %% 10 == 0) {
          message(sprintf("Classified %d samples...", length(.self$predictions)))
        }
      }
    },

    plot_validation_roc = function(category1, category2) {
      if (length(predictions) == 0) {
        stop("No predictions available. Run 'classify_dataframe' first.")
      }

      eval_data <- tryCatch(
        .prepare_binary_eval(predictions, category1, category2, "validation"),
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
                      title = "Validation ROC Curve",
                      subtitle = sprintf("AUC = %.2f", auc_val)) +
        ggplot2::xlim(0, 1) + ggplot2::ylim(0, 1.05) +
        ggplot2::theme_minimal()

      figures[[length(figures) + 1]] <<- p
      print(p)
    },

    plot_confusion_matrix = function(category1, category2, normalize = TRUE) {
      if (length(predictions) == 0) {
        stop("No predictions available. Run 'classify_dataframe' first.")
      }

      eval_data <- tryCatch(
        .prepare_binary_eval(predictions, category1, category2, "validation"),
        error = function(e) {
          message(conditionMessage(e))
          NULL
        }
      )
      if (is.null(eval_data)) return(invisible(NULL))
      true_numeric <- eval_data$true_numeric
      pred_labels <- ifelse(eval_data$pred_probs > 0.5, 1, 0)

      cm <- table(True = factor(true_numeric, levels = c(0, 1)),
                  Predicted = factor(pred_labels, levels = c(0, 1)))

      if (normalize) {
        cm_norm <- .normalize_confusion_matrix(cm)
        cm_df <- as.data.frame(as.table(cm_norm))
        fmt_label <- "Proportion"
        title_txt <- "Normalized Confusion Matrix"
      } else {
        cm_df <- as.data.frame(as.table(cm))
        fmt_label <- "Count"
        title_txt <- "Confusion Matrix"
      }
      colnames(cm_df) <- c("True", "Predicted", "Value")

      # Map 0/1 back to category names
      cm_df$True <- factor(cm_df$True, levels = c("0", "1"),
                           labels = c(category1, category2))
      cm_df$Predicted <- factor(cm_df$Predicted, levels = c("0", "1"),
                                labels = c(category1, category2))

      p <- ggplot2::ggplot(cm_df, ggplot2::aes(x = Predicted, y = True, fill = Value)) +
        ggplot2::geom_tile() +
        ggplot2::geom_text(ggplot2::aes(label = round(Value, 2)), size = 5) +
        ggplot2::scale_fill_gradient(low = "white", high = "steelblue") +
        ggplot2::labs(x = "Predicted Label", y = "True Label", title = title_txt) +
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
        if (inherits(fig, "ggplot")) {
          print(fig)
        }
      }
      dev.off()
      message(sprintf("Plots saved to %s.", file_name))
    }
  )
)


# -----------------------------------------------------------------------------
# AdaptmsClassifierFolder
# For validating against individual sample files in a directory
# -----------------------------------------------------------------------------

AdaptmsClassifierFolder <- setRefClass(
  "AdaptmsClassifierFolder",
  fields = list(
    prot_df        = "data.frame",
    cat_df         = "data.frame",
    gene_dict      = "ANY",
    between        = "character",
    cohort         = "ANY",
    figures        = "list",
    models         = "list",
    selected_features = "character",
    predictions    = "list",
    feature_names  = "character",
    training_data  = "ANY",
    training_targets = "ANY"
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
      selected_features <<- character(0)
      predictions <<- list()
      feature_names <<- colnames(prot_df)
      training_data <<- NULL
      training_targets <<- NULL
    },

    classify_and_plot = function(category1, category2, n_runs = 10, topn_features = 50) {
      if (between == "") {
        stop("No between variable given. Please provide a variable for classification.")
      }

      # Merge on row names
      shared_ids <- intersect(rownames(.self$prot_df), rownames(.self$cat_df))
      d_ML <- cbind(.self$prot_df[shared_ids, , drop = FALSE],
                    .self$cat_df[shared_ids, between, drop = FALSE])

      X_unimputed <- d_ML[, colnames(d_ML) != between, drop = FALSE]
      # Coerce to numeric
      X_unimputed[] <- lapply(X_unimputed, function(x) as.numeric(as.character(x)))
      y <- d_ML[[between]]

      # KNN impute
      X_imputed <- as.data.frame(kNN(X_unimputed, k = 5, imp_var = FALSE))
      rownames(X_imputed) <- rownames(X_unimputed)

      # Filter to categories
      keep <- y %in% c(category1, category2)
      X_filt_unimp <- X_unimputed[keep, , drop = FALSE]
      X_filt_imp   <- X_imputed[keep, , drop = FALSE]
      y_filt <- ifelse(y[keep] == category1, 0, 1)
      names(y_filt) <- rownames(X_filt_unimp)

      training_data <<- X_filt_imp
      training_targets <<- y_filt

      mean_fpr <- seq(0, 1, length.out = 100)
      all_tprs <- matrix(nrow = 0, ncol = 100)
      aucs <- numeric(0)

      for (i in seq_len(n_runs)) {
        set.seed(i - 1)
        train_idx <- createDataPartition(y_filt, p = 0.8, list = FALSE)[, 1]

        X_train_unimp <- X_filt_unimp[train_idx, , drop = FALSE]
        X_train_imp   <- X_filt_imp[train_idx, , drop = FALSE]
        X_test_imp    <- X_filt_imp[-train_idx, , drop = FALSE]
        y_train <- y_filt[train_idx]
        y_test  <- y_filt[-train_idx]

        # t-test feature selection on unimputed data
        p_values <- sapply(colnames(X_train_unimp), function(col) {
          g1 <- X_train_unimp[y_train == 0, col]
          g2 <- X_train_unimp[y_train == 1, col]
          g1 <- g1[!is.na(g1)]
          g2 <- g2[!is.na(g2)]
          if (length(g1) > 1 && length(g2) > 1) {
            tryCatch(t.test(g1, g2)$p.value, error = function(e) 1.0)
          } else {
            1.0
          }
        })

        corrected_p <- p.adjust(p_values, method = "BH")
        top_idx <- order(corrected_p)[seq_len(min(topn_features, length(corrected_p)))]
        top_feats <- colnames(X_train_unimp)[top_idx]
        selected_features <<- unique(c(.self$selected_features, top_feats))

        X_train_sel <- X_train_imp[, top_feats, drop = FALSE]
        X_test_sel  <- X_test_imp[, top_feats, drop = FALSE]

        train_df <- data.frame(X_train_sel, y = factor(y_train))
        model <- glm(y ~ ., data = train_df, family = binomial(link = "logit"),
                      control = list(maxit = 1000))
        models[[length(models) + 1]] <<- model

        # 5-fold CV
        folds <- createFolds(y_train, k = 5, list = TRUE, returnTrain = TRUE)
        cv_tprs <- matrix(nrow = 0, ncol = 100)

        for (fold_train_idx in folds) {
          fold_val_idx <- setdiff(seq_along(y_train), fold_train_idx)
          cv_train_df <- data.frame(X_train_sel[fold_train_idx, , drop = FALSE],
                                     y = factor(y_train[fold_train_idx]))
          cv_val_df   <- X_train_sel[fold_val_idx, , drop = FALSE]
          y_cv_val    <- y_train[fold_val_idx]

          cv_model <- tryCatch(
            glm(y ~ ., data = cv_train_df, family = binomial(link = "logit"),
                control = list(maxit = 1000)),
            error = function(e) NULL
          )
          if (!is.null(cv_model)) {
            probas <- predict(cv_model, newdata = cv_val_df, type = "response")
            if (length(unique(y_cv_val)) > 1) {
              roc_obj <- roc(y_cv_val, probas, quiet = TRUE)
              interp_tpr <- approx(1 - roc_obj$specificities, roc_obj$sensitivities,
                                    xout = mean_fpr, rule = 2)$y
              interp_tpr[1] <- 0
              cv_tprs <- rbind(cv_tprs, interp_tpr)
            }
          }
        }

        test_probs <- predict(model, newdata = X_test_sel, type = "response")
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

    classify_sample = function(d_sample, true_label) {
      if (length(models) == 0 || length(selected_features) == 0) {
        stop("No trained models available. Run 'classify_and_plot' first.")
      }

      available_features <- intersect(.self$selected_features, colnames(d_sample))
      if (length(available_features) < 2) return(invisible(NULL))
      d_sample_filt <- d_sample[, available_features, drop = FALSE]
      d_sample_filt[is.na(d_sample_filt)] <- 0

      X_train_filt <- .self$training_data[, available_features, drop = FALSE]
      y_train_filt <- .self$training_targets

      train_df <- data.frame(X_train_filt, y = factor(y_train_filt))
      model <- tryCatch(
        glm(y ~ ., data = train_df, family = binomial(link = "logit"),
            control = list(maxit = 1000)),
        error = function(e) NULL
      )

      if (!is.null(model)) {
        prob <- predict(model, newdata = d_sample_filt, type = "response")
        predictions[[length(predictions) + 1]] <<- list(
          sample = rownames(d_sample)[1], true_label = true_label, prob = as.numeric(prob)
        )
      }
    },

    classify_directory = function(directory, cat_validation_pool_SF, category1, category2) {
      files <- list.files(directory, pattern = "mzML\\.pg_matrix\\.tsv$", full.names = TRUE)

      for (fpath in files) {
        fname <- basename(fpath)
        if (!is.null(.self$cohort) && !grepl(.self$cohort, fname)) next

        d_sample <- read.delim(fpath, sep = "\t", check.names = FALSE)
        # Select columns 1 and 6 (Protein.Group and sample value)
        d_sample <- d_sample[, c(1, 6), drop = FALSE]
        sample_col_name <- basename(sub(".*/", "", colnames(d_sample)[2]))
        colnames(d_sample) <- c("Protein.Group", sample_col_name)
        rownames(d_sample) <- .sanitize_row_ids(d_sample$Protein.Group, prefix = "protein_group")
        d_sample$Protein.Group <- NULL

        # Log10 transform with robust handling of non-positive values.
        d_sample[] <- lapply(d_sample, function(x) {
          x <- suppressWarnings(as.numeric(x))
          x[x <= 0] <- NA_real_
          log10(x)
        })

        # Transpose: 1 row = 1 sample
        d_sample <- as.data.frame(t(d_sample))
        sample_name <- rownames(d_sample)[1]

        if (sample_name %in% rownames(cat_validation_pool_SF)) {
          true_label <- cat_validation_pool_SF[sample_name, between]
          if (true_label %in% c(category1, category2)) {
            classify_sample(d_sample, true_label)
          }
        }
      }
    },

    plot_accumulated_roc = function(category1, category2) {
      if (length(predictions) == 0) {
        stop("No predictions available. Run 'classify_directory' first.")
      }

      eval_data <- tryCatch(
        .prepare_binary_eval(predictions, category1, category2, "directory"),
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
                      title = "Aggregated Validation ROC Curve",
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
    }
  )
)
