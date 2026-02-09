# =============================================================================
# AdaptmsMultiClassifier.R
# Multiclass ADAPT-MS classifier
# R equivalent of utils/AdaptmsMultiClassifier.py
# =============================================================================

library(glmnet)
library(caret)
library(pROC)
library(VIM)
library(nnet)

# -----------------------------------------------------------------------------
# AdaptmsMulticlassClassifier
# Multiclass extension using one-vs-rest logistic regression with per-sample
# refitting for the ADAPT-MS approach.
# -----------------------------------------------------------------------------

AdaptmsMulticlassClassifier <- setRefClass(
  "AdaptmsMulticlassClassifier",
  fields = list(
    prot_df           = "data.frame",
    cat_df            = "data.frame",
    gene_dict         = "ANY",
    between_column    = "character",
    figures           = "list",
    models            = "list",
    selected_features = "character",
    predictions       = "list",
    feature_names     = "character",
    training_data     = "ANY",
    training_targets  = "ANY",
    label_levels      = "character"
  ),
  methods = list(
    initialize = function(prot_df, cat_df, gene_dict = NULL, between_column) {
      prot_df <<- prot_df[, !duplicated(colnames(prot_df)), drop = FALSE]
      cat_df <<- cat_df
      gene_dict <<- gene_dict
      between_column <<- between_column
      figures <<- list()
      models <<- list()
      selected_features <<- character(0)
      predictions <<- list()
      feature_names <<- colnames(prot_df)
      training_data <<- NULL
      training_targets <<- NULL
      label_levels <<- character(0)
    },

    # --- Private: pairwise t-test feature selection on unimputed data ---
    perform_feature_selection = function(X_df, y_vec, n_features_per_pair = 200) {
      unique_labels <- sort(unique(y_vec))
      sel_feats <- character(0)

      for (i in seq_along(unique_labels)) {
        for (j in seq_along(unique_labels)) {
          if (j <= i) next
          lab1 <- unique_labels[i]
          lab2 <- unique_labels[j]

          group1 <- X_df[y_vec == lab1, , drop = FALSE]
          group2 <- X_df[y_vec == lab2, , drop = FALSE]

          p_values <- sapply(colnames(X_df), function(feat) {
            f1 <- group1[[feat]]
            f2 <- group2[[feat]]
            f1 <- f1[!is.na(f1)]
            f2 <- f2[!is.na(f2)]
            if (length(f1) > 5 && length(f2) > 5) {
              tryCatch(t.test(f1, f2)$p.value, error = function(e) 1.0)
            } else {
              1.0
            }
          })

          corrected_p <- p.adjust(p_values, method = "BH")
          top_idx <- order(corrected_p)[seq_len(min(n_features_per_pair, length(corrected_p)))]
          sel_feats <- unique(c(sel_feats, colnames(X_df)[top_idx]))
        }
      }

      selected_features <<- unique(c(.self$selected_features, sel_feats))
      return(sel_feats)
    },

    classify_and_plot = function(categories = NULL, n_runs = 3, n_features = 100) {
      # Merge on row names
      shared_ids <- intersect(rownames(.self$prot_df), rownames(.self$cat_df))
      d_ML <- cbind(.self$prot_df[shared_ids, , drop = FALSE],
                    .self$cat_df[shared_ids, between_column, drop = FALSE])

      y_original <- d_ML[[between_column]]

      if (!is.null(categories)) {
        keep <- y_original %in% categories
        d_ML <- d_ML[keep, , drop = FALSE]
        y_original <- y_original[keep]
      }

      training_targets <<- y_original
      label_levels <<- sort(unique(y_original))

      # KNN impute
      X_raw <- d_ML[, colnames(d_ML) != between_column, drop = FALSE]
      X_imputed <- as.data.frame(kNN(X_raw, k = 5, imp_var = FALSE))
      rownames(X_imputed) <- rownames(X_raw)
      training_data <<- X_imputed

      mean_fpr <- seq(0, 1, length.out = 100)

      # Storage for per-class ROC curves
      class_curves <- lapply(label_levels, function(l) {
        list(tprs = list(), aucs = numeric(0))
      })
      names(class_curves) <- label_levels

      for (i in seq_len(n_runs)) {
        set.seed(i - 1)
        # Encode as factor for stratified split
        y_factor <- factor(y_original, levels = label_levels)
        train_idx <- createDataPartition(y_factor, p = 0.8, list = FALSE)[, 1]

        X_orig_train <- X_raw[train_idx, , drop = FALSE]
        X_imp_train  <- X_imputed[train_idx, , drop = FALSE]
        y_train      <- y_original[train_idx]
        y_test       <- y_original[-train_idx]

        # Feature selection on unimputed training data
        sel_feats <- perform_feature_selection(X_orig_train, y_train,
                                               n_features_per_pair = n_features)

        X_imp_train_sel <- X_imp_train[, sel_feats, drop = FALSE]

        # Multinomial logistic regression
        train_df <- data.frame(X_imp_train_sel, y = factor(y_train, levels = label_levels))
        model <- tryCatch(
          nnet::multinom(y ~ ., data = train_df, maxit = 10000, trace = FALSE),
          error = function(e) NULL
        )
        if (is.null(model)) next
        models[[length(models) + 1]] <<- model

        # 5-fold CV
        folds <- createFolds(factor(y_train, levels = label_levels),
                             k = 5, list = TRUE, returnTrain = TRUE)

        for (fold_train_idx in folds) {
          fold_val_idx <- setdiff(seq_along(y_train), fold_train_idx)
          cv_train_df <- data.frame(X_imp_train_sel[fold_train_idx, , drop = FALSE],
                                     y = factor(y_train[fold_train_idx], levels = label_levels))
          cv_val_df   <- X_imp_train_sel[fold_val_idx, , drop = FALSE]
          y_cv_val    <- y_train[fold_val_idx]

          cv_model <- tryCatch(
            nnet::multinom(y ~ ., data = cv_train_df, maxit = 10000, trace = FALSE),
            error = function(e) NULL
          )
          if (is.null(cv_model)) next

          probas <- predict(cv_model, newdata = cv_val_df, type = "probs")
          if (is.null(dim(probas))) {
            # Only 2 classes: probas is a vector for second class
            probas <- cbind(1 - probas, probas)
            colnames(probas) <- label_levels
          }

          for (lbl in label_levels) {
            true_bin <- as.integer(y_cv_val == lbl)
            if (length(unique(true_bin)) < 2) next
            if (lbl %in% colnames(probas)) {
              prob_class <- probas[, lbl]
            } else {
              next
            }

            roc_obj <- tryCatch(roc(true_bin, prob_class, quiet = TRUE),
                                error = function(e) NULL)
            if (!is.null(roc_obj)) {
              interp_tpr <- approx(1 - roc_obj$specificities, roc_obj$sensitivities,
                                    xout = mean_fpr, rule = 2)$y
              interp_tpr[1] <- 0
              class_curves[[lbl]]$tprs[[length(class_curves[[lbl]]$tprs) + 1]] <- interp_tpr
              class_curves[[lbl]]$aucs <- c(class_curves[[lbl]]$aucs,
                                             as.numeric(auc(roc_obj)))
            }
          }
        }
      }

      # Plot one-vs-rest ROC for each class
      colors <- grDevices::rainbow(length(label_levels))

      p <- ggplot2::ggplot()

      for (k in seq_along(label_levels)) {
        lbl <- label_levels[k]
        tpr_mat <- do.call(rbind, class_curves[[lbl]]$tprs)
        if (is.null(tpr_mat) || nrow(tpr_mat) == 0) next

        mean_tpr <- colMeans(tpr_mat)
        mean_tpr[100] <- 1.0
        mean_auc <- mean(class_curves[[lbl]]$aucs, na.rm = TRUE)
        sd_auc   <- sd(class_curves[[lbl]]$aucs, na.rm = TRUE)
        std_tpr  <- apply(tpr_mat, 2, sd)
        upper    <- pmin(mean_tpr + std_tpr, 1)
        lower    <- pmax(mean_tpr - std_tpr, 0)

        curve_df <- data.frame(fpr = mean_fpr, tpr = mean_tpr,
                                lower = lower, upper = upper)

        p <- p +
          ggplot2::geom_ribbon(data = curve_df,
                               ggplot2::aes(x = fpr, ymin = lower, ymax = upper),
                               fill = colors[k], alpha = 0.2) +
          ggplot2::geom_line(data = curve_df,
                             ggplot2::aes(x = fpr, y = tpr),
                             color = colors[k], linewidth = 1)
      }

      p <- p +
        ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "black") +
        ggplot2::labs(x = "False Positive Rate", y = "True Positive Rate",
                      title = "Cross-validation ROC Curves (One-vs-Rest)") +
        ggplot2::xlim(0, 1) + ggplot2::ylim(0, 1.05) +
        ggplot2::theme_minimal()

      figures[[length(figures) + 1]] <<- p
      print(p)

      # Print summary
      for (lbl in label_levels) {
        if (length(class_curves[[lbl]]$aucs) > 0) {
          message(sprintf("  %s: mean AUC = %.2f +/- %.2f",
                          lbl, mean(class_curves[[lbl]]$aucs, na.rm = TRUE),
                          sd(class_curves[[lbl]]$aucs, na.rm = TRUE)))
        }
      }
    },

    classify_dataframe = function(validation_df, validation_cat_df) {
      if (length(selected_features) == 0) {
        stop("No features selected. Run 'classify_and_plot' first.")
      }

      shared_ids <- intersect(rownames(validation_df), rownames(validation_cat_df))

      for (idx in shared_ids) {
        row_data <- validation_df[idx, , drop = FALSE]
        non_na_cols <- colnames(row_data)[!is.na(row_data[1, ])]
        available_features <- intersect(.self$selected_features, non_na_cols)

        if (length(available_features) < 5) {
          message(sprintf("Sample %s has too few available features. Skipping.", idx))
          next
        }

        d_sample <- row_data[, available_features, drop = FALSE]
        true_label <- if (idx %in% rownames(validation_cat_df)) {
          validation_cat_df[idx, between_column]
        } else {
          "Unknown"
        }

        # Re-train model on stored training data with available features
        X_train_filt <- .self$training_data[, available_features, drop = FALSE]
        y_train_filt <- .self$training_targets

        train_df <- data.frame(X_train_filt, y = factor(y_train_filt, levels = label_levels))
        model <- tryCatch(
          nnet::multinom(y ~ ., data = train_df, maxit = 10000, trace = FALSE),
          error = function(e) NULL
        )

        if (!is.null(model)) {
          probs <- predict(model, newdata = d_sample, type = "probs")
          if (is.null(dim(probs))) {
            probs <- c(1 - probs, probs)
            names(probs) <- label_levels
          }
          predictions[[length(predictions) + 1]] <<- list(
            sample = idx, true_label = true_label, probs = probs
          )
        }

        if (length(.self$predictions) %% 10 == 0) {
          message(sprintf("Classified %d samples...", length(.self$predictions)))
        }
      }
    },

    plot_validation_roc = function() {
      if (length(predictions) == 0) {
        stop("No predictions available. Run 'classify_dataframe' first.")
      }

      true_labels <- sapply(predictions, function(x) x$true_label)
      known_idx <- which(true_labels != "Unknown")
      if (length(known_idx) == 0) stop("No samples with known labels.")

      true_labels <- true_labels[known_idx]
      pred_probs_list <- lapply(known_idx, function(i) predictions[[i]]$probs)

      unique_labels <- sort(unique(true_labels))
      colors <- grDevices::rainbow(length(unique_labels))

      p <- ggplot2::ggplot()

      for (k in seq_along(unique_labels)) {
        lbl <- unique_labels[k]
        true_bin <- as.integer(true_labels == lbl)

        prob_class <- sapply(pred_probs_list, function(pr) {
          if (lbl %in% names(pr)) pr[lbl] else 0
        })

        if (length(unique(true_bin)) < 2) next

        roc_obj <- tryCatch(roc(true_bin, prob_class, quiet = TRUE),
                            error = function(e) NULL)
        if (is.null(roc_obj)) next

        auc_val <- as.numeric(auc(roc_obj))
        plot_df <- data.frame(fpr = 1 - roc_obj$specificities,
                              tpr = roc_obj$sensitivities)

        p <- p +
          ggplot2::geom_line(data = plot_df, ggplot2::aes(x = fpr, y = tpr),
                             color = colors[k], linewidth = 1)
      }

      p <- p +
        ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
        ggplot2::labs(x = "False Positive Rate", y = "True Positive Rate",
                      title = "Validation ROC Curves (One-vs-Rest)") +
        ggplot2::xlim(0, 1) + ggplot2::ylim(0, 1.05) +
        ggplot2::theme_minimal()

      figures[[length(figures) + 1]] <<- p
      print(p)
    },

    save_figures = function(output_directory, prefix = "") {
      if (!dir.exists(output_directory)) dir.create(output_directory, recursive = TRUE)

      for (i in seq_along(figures)) {
        fname <- file.path(output_directory, sprintf("%sfigure_%d.pdf", prefix, i))
        pdf(fname, width = 10, height = 8)
        if (inherits(figures[[i]], "ggplot")) print(figures[[i]])
        dev.off()
        message(sprintf("Saved figure to %s", fname))
      }
    }
  )
)
