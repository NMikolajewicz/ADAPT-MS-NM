# =============================================================================
# NonRefitClassifier.R
# ADAPT-MS ablation: t-test + logistic regression WITHOUT per-sample refitting
# R equivalent of utils/NonRefitClassifier.py
# =============================================================================

library(caret)
library(pROC)
library(VIM)

NonRefitClassifier <- setRefClass(
  "NonRefitClassifier",
  fields = list(
    prot_df           = "data.frame",
    cat_df            = "data.frame",
    gene_dict         = "ANY",
    between           = "character",
    figures           = "list",
    models            = "list",
    selected_features = "list"
  ),
  methods = list(
    initialize = function(prot_df, cat_df, gene_dict = NULL, between = NULL) {
      prot_df <<- prot_df[, !duplicated(colnames(prot_df)), drop = FALSE]
      cat_df <<- cat_df
      gene_dict <<- gene_dict
      between <<- if (is.null(between)) "" else between
      figures <<- list()
      models <<- list()
      selected_features <<- list()
    },

    classify_and_plot = function(category1, category2, n_runs = 10) {
      if (between == "") {
        stop("The 'between' attribute must be set to a valid column name.")
      }

      # Merge on row names
      shared_ids <- intersect(rownames(.self$prot_df), rownames(.self$cat_df))
      d_ML <- cbind(.self$prot_df[shared_ids, , drop = FALSE],
                    .self$cat_df[shared_ids, , drop = FALSE])

      # KNN impute
      feat_cols <- setdiff(colnames(d_ML), between)
      X_raw <- d_ML[, feat_cols, drop = FALSE]
      X_imputed <- as.data.frame(kNN(X_raw, k = 5, imp_var = FALSE))
      rownames(X_imputed) <- rownames(X_raw)

      y <- d_ML[[between]]

      # Filter to two categories
      keep <- y %in% c(category1, category2)
      X_filtered <- X_imputed[keep, , drop = FALSE]
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

        # t-test feature selection on imputed data
        p_values <- sapply(colnames(X_train), function(col) {
          g1 <- X_train[y_train == 0, col]
          g2 <- X_train[y_train == 1, col]
          tryCatch(t.test(g1, g2)$p.value, error = function(e) 1.0)
        })

        corrected_p <- p.adjust(p_values, method = "BH")
        top_idx <- order(corrected_p)[seq_len(min(50, length(corrected_p)))]
        top_feats <- colnames(X_train)[top_idx]
        selected_features[[length(selected_features) + 1]] <<- top_feats

        X_train_sel <- X_train[, top_feats, drop = FALSE]
        X_test_sel  <- X_test[, top_feats, drop = FALSE]

        # Train logistic regression
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

    validate_and_plot = function(validation_prot_df, validation_cat_df, category1, category2) {
      if (length(models) == 0 || length(selected_features) == 0) {
        stop("No trained models available. Run 'classify_and_plot' first.")
      }

      # Merge and impute validation data
      shared_ids <- intersect(rownames(validation_prot_df), rownames(validation_cat_df))
      val_data <- cbind(validation_prot_df[shared_ids, , drop = FALSE],
                        validation_cat_df[shared_ids, , drop = FALSE])

      feat_cols <- setdiff(colnames(val_data), between)
      X_val_raw <- val_data[, feat_cols, drop = FALSE]
      X_val_imp <- as.data.frame(kNN(X_val_raw, k = 5, imp_var = FALSE))
      rownames(X_val_imp) <- rownames(X_val_raw)

      y_val <- val_data[[between]]
      keep <- y_val %in% c(category1, category2)
      X_val_imp <- X_val_imp[keep, , drop = FALSE]
      y_val <- ifelse(y_val[keep] == category1, 0, 1)

      mean_fpr <- seq(0, 1, length.out = 100)
      all_tprs <- matrix(nrow = 0, ncol = 100)
      aucs <- numeric(0)

      for (k in seq_along(models)) {
        model <- models[[k]]
        top_feats <- selected_features[[k]]
        avail_feats <- intersect(top_feats, colnames(X_val_imp))
        if (length(avail_feats) < 2) next

        X_val_sel <- X_val_imp[, avail_feats, drop = FALSE]
        y_pred_prob <- predict(model, newdata = X_val_sel, type = "response")

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
