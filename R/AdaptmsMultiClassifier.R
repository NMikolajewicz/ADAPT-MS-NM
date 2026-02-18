# =============================================================================
# AdaptmsMultiClassifier.R
# Multiclass ADAPT-MS classifier
# R equivalent of utils/AdaptmsMultiClassifier.py
# =============================================================================

.resolve_n_jobs_multi <- function(n_jobs) {
  nj <- suppressWarnings(as.integer(n_jobs))
  if (is.na(nj)) nj <- 1L
  if (nj <= 0L) {
    detected <- suppressWarnings(parallel::detectCores(logical = FALSE))
    if (is.na(detected) || detected < 1L) detected <- 1L
    nj <- detected
  }
  nj
}

.parallel_lapply_multi <- function(X, FUN, n_jobs = 1L) {
  nj <- .resolve_n_jobs_multi(n_jobs)
  if (length(X) <= 1L || nj <= 1L || .Platform$OS.type == "windows") {
    return(lapply(X, FUN))
  }
  parallel::mclapply(X, FUN, mc.cores = min(length(X), nj))
}

.numeric_matrix_multi <- function(df) {
  as.matrix(data.frame(
    lapply(df, function(x) suppressWarnings(as.numeric(as.character(x)))),
    check.names = FALSE
  ))
}

.welch_pvalues_pair <- function(group1_mat, group2_mat, min_non_na = 6L) {
  n1 <- colSums(!is.na(group1_mat))
  n2 <- colSums(!is.na(group2_mat))
  valid <- n1 >= min_non_na & n2 >= min_non_na

  m1 <- colMeans(group1_mat, na.rm = TRUE)
  m2 <- colMeans(group2_mat, na.rm = TRUE)

  centered1 <- sweep(group1_mat, 2, m1, FUN = "-")
  centered2 <- sweep(group2_mat, 2, m2, FUN = "-")
  centered1[is.na(centered1)] <- 0
  centered2[is.na(centered2)] <- 0

  v1 <- colSums(centered1 ^ 2) / pmax(n1 - 1, 1)
  v2 <- colSums(centered2 ^ 2) / pmax(n2 - 1, 1)

  se <- sqrt((v1 / pmax(n1, 1)) + (v2 / pmax(n2, 1)))
  t_stat <- (m1 - m2) / se

  df_num <- (v1 / pmax(n1, 1) + v2 / pmax(n2, 1)) ^ 2
  df_den <- (v1 ^ 2) / (pmax(n1, 1) ^ 2 * pmax(n1 - 1, 1)) +
    (v2 ^ 2) / (pmax(n2, 1) ^ 2 * pmax(n2 - 1, 1))
  df <- df_num / df_den

  p_values <- 2 * stats::pt(-abs(t_stat), df = df)
  p_values[!is.finite(p_values)] <- 1.0
  p_values[!valid] <- 1.0
  as.numeric(p_values)
}

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
    label_levels      = "character",
    sample_model_cache = "ANY"
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
      sample_model_cache <<- new.env(parent = emptyenv())
    },

    # --- Private: pairwise t-test feature selection on unimputed data ---
    perform_feature_selection = function(X_df, y_vec, n_features_per_pair = 200, n_jobs = 1) {
      unique_labels <- sort(unique(y_vec))
      sel_feats <- character(0)
      X_mat <- .numeric_matrix_multi(X_df)
      pairs <- utils::combn(unique_labels, 2, simplify = FALSE)
      col_names <- colnames(X_df)

      pair_features <- .parallel_lapply_multi(
        pairs,
        function(pair_lbl) {
          lab1 <- pair_lbl[1]
          lab2 <- pair_lbl[2]
          group1 <- X_mat[y_vec == lab1, , drop = FALSE]
          group2 <- X_mat[y_vec == lab2, , drop = FALSE]
          p_values <- .welch_pvalues_pair(group1, group2, min_non_na = 6L)
          corrected_p <- p.adjust(p_values, method = "BH")
          top_idx <- order(corrected_p)[seq_len(min(n_features_per_pair, length(corrected_p)))]
          col_names[top_idx]
        },
        n_jobs = n_jobs
      )

      for (pair_top in pair_features) {
        sel_feats <- unique(c(sel_feats, pair_top))
      }

      selected_features <<- unique(c(.self$selected_features, sel_feats))
      return(sel_feats)
    },

    classify_and_plot = function(categories = NULL, n_runs = 3, n_features = 100, n_jobs = 1) {
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
      class_counts <- table(y_original)

      if (length(y_original) < 2) {
        stop(sprintf(
          "Need at least 2 samples after filtering '%s' (found %d).",
          between_column, length(y_original)
        ))
      }
      if (length(label_levels) < 2) {
        stop(sprintf(
          "Need at least 2 classes in '%s' for stratified splitting. Counts: %s",
          between_column,
          paste(sprintf("%s=%d", names(class_counts), as.integer(class_counts)), collapse = ", ")
        ))
      }

      # Fast KNN-style imputation (configurable via ADAPTMS_IMPUTE_METHOD).
      X_raw <- d_ML[, colnames(d_ML) != between_column, drop = FALSE]
      X_imputed <- .adaptms_impute_dataset(X_raw, k = 5)
      training_data <<- X_imputed
      sample_model_cache <<- new.env(parent = emptyenv())

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
                                               n_features_per_pair = n_features,
                                               n_jobs = n_jobs)

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
      selected_feature_order <- unique(.self$selected_features)
      cache_env <- .self$sample_model_cache
      if (!is.environment(cache_env)) {
        cache_env <- new.env(parent = emptyenv())
        sample_model_cache <<- cache_env
      }

      for (idx in shared_ids) {
        row_data <- validation_df[idx, , drop = FALSE]
        aligned_row <- rep(NA_real_, length(selected_feature_order))
        names(aligned_row) <- selected_feature_order
        common_cols <- intersect(selected_feature_order, colnames(row_data))
        if (length(common_cols) > 0) {
          aligned_row[common_cols] <- suppressWarnings(as.numeric(row_data[1, common_cols, drop = TRUE]))
        }
        available_features <- selected_feature_order[!is.na(aligned_row)]

        if (length(available_features) < 5) {
          message(sprintf("Sample %s has too few available features. Skipping.", idx))
          next
        }

        d_sample <- as.data.frame(t(aligned_row[available_features]), stringsAsFactors = FALSE)
        colnames(d_sample) <- available_features
        true_label <- if (idx %in% rownames(validation_cat_df)) {
          validation_cat_df[idx, between_column, drop = TRUE]
        } else {
          "Unknown"
        }

        feature_key <- paste(available_features, collapse = "\r")
        if (exists(feature_key, envir = cache_env, inherits = FALSE)) {
          model <- get(feature_key, envir = cache_env, inherits = FALSE)
        } else {
          X_train_filt <- .self$training_data[, available_features, drop = FALSE]
          y_train_filt <- .self$training_targets
          train_df <- data.frame(X_train_filt, y = factor(y_train_filt, levels = label_levels))
          model <- tryCatch(
            nnet::multinom(y ~ ., data = train_df, maxit = 10000, trace = FALSE),
            error = function(e) NULL
          )
          assign(feature_key, model, envir = cache_env)
        }

        if (!is.null(model)) {
          probs <- predict(model, newdata = d_sample, type = "probs")
          probs_vec <- if (is.null(dim(probs))) {
            vals <- as.numeric(probs)
            nm <- names(probs)
            if (is.null(nm)) {
              nm <- if (length(vals) == length(label_levels)) label_levels else tail(label_levels, length(vals))
            }
            names(vals) <- nm
            vals
          } else {
            vals <- as.numeric(probs[1, ])
            names(vals) <- colnames(probs)
            vals
          }
          full_probs <- setNames(rep(0, length(label_levels)), label_levels)
          full_probs[names(probs_vec)] <- probs_vec
          predictions[[length(predictions) + 1]] <<- list(
            sample = idx, true_label = true_label, probs = full_probs
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
      prob_mat <- do.call(rbind, lapply(pred_probs_list, function(pr) {
        out <- setNames(rep(0, length(label_levels)), label_levels)
        out[names(pr)] <- as.numeric(pr)
        out
      }))

      unique_labels <- sort(unique(true_labels))
      colors <- grDevices::rainbow(length(unique_labels))

      p <- ggplot2::ggplot()

      for (k in seq_along(unique_labels)) {
        lbl <- unique_labels[k]
        true_bin <- as.integer(true_labels == lbl)

        prob_class <- if (lbl %in% colnames(prob_mat)) {
          as.numeric(prob_mat[, lbl])
        } else {
          rep(0, nrow(prob_mat))
        }

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
