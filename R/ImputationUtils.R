# =============================================================================
# ImputationUtils.R
# Fast numeric imputation helpers with configurable KNN backends.
# =============================================================================

.adaptms_get_env_int <- function(var_name, default_value, min_value = 0L) {
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

.adaptms_get_env_num <- function(var_name, default_value, min_value = 0) {
  raw <- Sys.getenv(var_name, unset = "")
  if (!nzchar(raw)) {
    return(as.numeric(default_value))
  }
  parsed <- suppressWarnings(as.numeric(raw))
  if (is.na(parsed) || parsed < min_value) {
    return(as.numeric(default_value))
  }
  parsed
}

.adaptms_normalize_impute_method <- function(method) {
  out <- tolower(as.character(method)[1])
  if (!out %in% c("fast_knn", "vim_knn", "median")) {
    out <- "fast_knn"
  }
  out
}

.adaptms_impute_config <- function(k = 5, method = NULL) {
  default_method <- Sys.getenv("ADAPTMS_IMPUTE_METHOD", unset = "fast_knn")
  resolved_method <- if (is.null(method)) default_method else method
  nn_searchtype <- tolower(Sys.getenv("ADAPTMS_NN_SEARCHTYPE", unset = "priority"))
  if (!nn_searchtype %in% c("priority", "standard", "radius")) {
    nn_searchtype <- "priority"
  }

  list(
    method = .adaptms_normalize_impute_method(resolved_method),
    k = .adaptms_get_env_int("ADAPTMS_IMPUTE_K", k, min_value = 1L),
    pca_dims = .adaptms_get_env_int("ADAPTMS_NN_PCA_DIMS", 30L, min_value = 0L),
    nn_eps = .adaptms_get_env_num("ADAPTMS_NN_EPS", 0.1, min_value = 0),
    nn_searchtype = nn_searchtype
  )
}

.adaptms_numeric_df <- function(df) {
  out <- as.data.frame(
    lapply(df, function(x) suppressWarnings(as.numeric(as.character(x)))),
    check.names = FALSE
  )
  rownames(out) <- rownames(df)
  out
}

.adaptms_col_medians <- function(df) {
  if (ncol(df) == 0) return(numeric(0))
  meds <- vapply(df, function(x) {
    m <- suppressWarnings(stats::median(x, na.rm = TRUE))
    if (!is.finite(m)) 0 else m
  }, numeric(1))
  meds
}

.adaptms_fill_na_with <- function(df, fill_vals) {
  out <- df
  if (nrow(out) == 0 || ncol(out) == 0) return(out)
  for (j in seq_len(ncol(out))) {
    miss <- is.na(out[[j]])
    if (any(miss)) out[[j]][miss] <- fill_vals[[j]]
  }
  out
}

.adaptms_align_to_columns <- function(df, target_cols) {
  out <- as.data.frame(
    matrix(NA_real_, nrow = nrow(df), ncol = length(target_cols),
           dimnames = list(rownames(df), target_cols)),
    stringsAsFactors = FALSE
  )
  if (length(target_cols) == 0) return(out)
  numeric_df <- .adaptms_numeric_df(df)
  shared <- intersect(target_cols, colnames(numeric_df))
  if (length(shared) > 0) {
    out[, shared] <- numeric_df[, shared, drop = FALSE]
  }
  out
}

.adaptms_project_query <- function(query_filled, prep) {
  if (nrow(query_filled) == 0) {
    return(matrix(numeric(0), nrow = 0, ncol = ncol(prep$ref_embed)))
  }
  if (is.null(prep$pca)) {
    return(as.matrix(query_filled))
  }
  projected <- tryCatch(
    predict(prep$pca, newdata = query_filled),
    error = function(e) NULL
  )
  if (is.null(projected)) {
    return(as.matrix(query_filled))
  }
  as.matrix(projected)
}

.adaptms_knn_indices <- function(prep, query_embed, exclude_self = FALSE, query_ids = NULL) {
  n_ref <- nrow(prep$ref_embed)
  n_query <- nrow(query_embed)
  k_req <- prep$cfg$k

  if (n_ref == 0 || n_query == 0 || k_req <= 0) {
    return(matrix(integer(0), nrow = n_query, ncol = 0))
  }

  k_search <- min(n_ref, k_req + if (exclude_self) 1L else 0L)
  nn <- RANN::nn2(
    data = prep$ref_embed,
    query = query_embed,
    k = k_search,
    eps = prep$cfg$nn_eps,
    searchtype = prep$cfg$nn_searchtype
  )
  idx <- nn$nn.idx
  if (!is.matrix(idx)) idx <- matrix(idx, ncol = k_search)

  same_reference <- exclude_self &&
    n_ref == n_query &&
    !is.null(query_ids) &&
    !is.null(prep$ref_ids) &&
    identical(as.character(query_ids), as.character(prep$ref_ids))

  if (!same_reference) {
    if (ncol(idx) < k_req) {
      idx <- cbind(idx, matrix(idx[, 1], nrow = nrow(idx), ncol = k_req - ncol(idx)))
    }
    return(idx[, seq_len(min(k_req, ncol(idx))), drop = FALSE])
  }

  out <- matrix(1L, nrow = n_query, ncol = k_req)
  all_ref <- seq_len(n_ref)
  for (i in seq_len(n_query)) {
    cand <- idx[i, ]
    cand <- cand[cand != i]
    if (length(cand) < k_req) {
      extra <- setdiff(all_ref, c(i, cand))
      cand <- c(cand, head(extra, k_req - length(cand)))
    }
    if (length(cand) == 0) cand <- i
    if (length(cand) < k_req) cand <- c(cand, rep.int(cand[1], k_req - length(cand)))
    out[i, ] <- cand[seq_len(k_req)]
  }
  out
}

.adaptms_apply_neighbor_imputation <- function(query_aligned, prep, idx) {
  out <- as.matrix(query_aligned)
  ref_raw <- as.matrix(prep$ref_raw)
  med <- prep$col_medians

  if (nrow(out) == 0 || ncol(out) == 0) {
    out_df <- as.data.frame(out, stringsAsFactors = FALSE)
    rownames(out_df) <- rownames(query_aligned)
    colnames(out_df) <- colnames(query_aligned)
    return(out_df)
  }

  for (j in seq_len(ncol(out))) {
    miss_rows <- which(is.na(out[, j]))
    if (length(miss_rows) == 0) next

    if (ncol(idx) == 0) {
      out[miss_rows, j] <- med[[j]]
      next
    }

    idx_block <- idx[miss_rows, , drop = FALSE]
    if (ncol(idx_block) == 1L) {
      donor_vals <- matrix(ref_raw[idx_block[, 1], j], ncol = 1L)
    } else {
      donor_vals <- sapply(seq_len(ncol(idx_block)), function(col_i) {
        ref_raw[idx_block[, col_i], j]
      })
      if (!is.matrix(donor_vals)) {
        donor_vals <- matrix(donor_vals, nrow = length(miss_rows), ncol = ncol(idx_block))
      }
    }
    fill_vals <- apply(donor_vals, 1, function(v) {
      vv <- v[!is.na(v)]
      if (length(vv) == 0) med[[j]] else stats::median(vv)
    })
    out[miss_rows, j] <- fill_vals
  }

  out_df <- as.data.frame(out, stringsAsFactors = FALSE)
  rownames(out_df) <- rownames(query_aligned)
  colnames(out_df) <- colnames(query_aligned)
  out_df
}

.adaptms_prepare_fast_knn <- function(reference_df, cfg) {
  ref <- .adaptms_numeric_df(reference_df)
  if (ncol(ref) == 0) {
    return(list(
      method = "fast_knn",
      cfg = cfg,
      ref_raw = ref,
      ref_embed = matrix(numeric(0), nrow = nrow(ref), ncol = 0),
      col_medians = numeric(0),
      pca = NULL,
      ref_ids = rownames(ref)
    ))
  }

  non_all_na <- colSums(!is.na(ref)) > 0
  ref <- ref[, non_all_na, drop = FALSE]
  med <- .adaptms_col_medians(ref)
  ref_filled <- .adaptms_fill_na_with(ref, med)
  ref_embed <- as.matrix(ref_filled)
  pca_obj <- NULL

  if (cfg$pca_dims > 0L && ncol(ref_embed) > 1L && nrow(ref_embed) > 2L) {
    rank_k <- min(cfg$pca_dims, ncol(ref_embed), nrow(ref_embed) - 1L)
    if (rank_k >= 2L) {
      pca_obj <- tryCatch(
        stats::prcomp(ref_embed, center = TRUE, scale. = TRUE, rank. = rank_k),
        error = function(e) NULL
      )
      if (!is.null(pca_obj)) {
        ref_embed <- pca_obj$x
      }
    }
  }

  list(
    method = "fast_knn",
    cfg = cfg,
    ref_raw = ref,
    ref_embed = ref_embed,
    col_medians = med,
    pca = pca_obj,
    ref_ids = rownames(ref)
  )
}

.adaptms_prepare_imputer <- function(reference_df, k = 5, method = NULL) {
  cfg <- .adaptms_impute_config(k = k, method = method)
  ref_numeric <- .adaptms_numeric_df(reference_df)

  if (identical(cfg$method, "fast_knn") && !requireNamespace("RANN", quietly = TRUE)) {
    warning("RANN is not installed; falling back to median imputation.")
    cfg$method <- "median"
  }
  if (identical(cfg$method, "vim_knn") && !requireNamespace("VIM", quietly = TRUE)) {
    warning("VIM is not installed; falling back to median imputation.")
    cfg$method <- "median"
  }

  if (cfg$method == "median") {
    non_all_na <- if (ncol(ref_numeric) > 0) colSums(!is.na(ref_numeric)) > 0 else logical(0)
    ref_numeric <- ref_numeric[, non_all_na, drop = FALSE]
    med <- .adaptms_col_medians(ref_numeric)
    return(list(
      method = "median",
      cfg = cfg,
      ref_cols = colnames(ref_numeric),
      col_medians = med
    ))
  }

  if (cfg$method == "vim_knn") {
    non_all_na <- if (ncol(ref_numeric) > 0) colSums(!is.na(ref_numeric)) > 0 else logical(0)
    ref_numeric <- ref_numeric[, non_all_na, drop = FALSE]
    return(list(
      method = "vim_knn",
      cfg = cfg,
      reference = ref_numeric
    ))
  }

  .adaptms_prepare_fast_knn(reference_df = ref_numeric, cfg = cfg)
}

.adaptms_imputer_columns <- function(imputer) {
  if (is.null(imputer$method)) return(character(0))
  if (identical(imputer$method, "fast_knn")) return(colnames(imputer$ref_raw))
  if (identical(imputer$method, "median")) return(imputer$ref_cols)
  if (identical(imputer$method, "vim_knn")) return(colnames(imputer$reference))
  character(0)
}

.adaptms_impute_with_imputer <- function(imputer, target_df, exclude_self = FALSE) {
  if (is.null(imputer$method)) {
    stop("Invalid imputer object: missing method.")
  }

  query_ids <- rownames(target_df)
  if (is.null(query_ids)) {
    query_ids <- as.character(seq_len(nrow(target_df)))
  }

  if (identical(imputer$method, "median")) {
    aligned <- .adaptms_align_to_columns(target_df, imputer$ref_cols)
    out <- .adaptms_fill_na_with(aligned, imputer$col_medians)
    rownames(out) <- query_ids
    return(out)
  }

  if (identical(imputer$method, "vim_knn")) {
    ref <- imputer$reference
    aligned <- .adaptms_align_to_columns(target_df, colnames(ref))
    if (ncol(aligned) == 0 || nrow(aligned) == 0) return(aligned)
    ref_medians <- .adaptms_col_medians(ref)

    ref_ids <- paste0("ref__", seq_len(nrow(ref)))
    tgt_ids <- paste0("target__", seq_len(nrow(aligned)))
    rownames(ref) <- ref_ids
    rownames(aligned) <- tgt_ids

    combined <- rbind(ref, aligned)
    combined_imp <- as.data.frame(VIM::kNN(combined, k = imputer$cfg$k, imp_var = FALSE))
    out <- combined_imp[tgt_ids, , drop = FALSE]
    # Guardrail: if VIM still leaves NA, fill from reference medians.
    out <- .adaptms_fill_na_with(out, ref_medians)
    rownames(out) <- query_ids
    return(out)
  }

  # fast_knn
  ref_cols <- colnames(imputer$ref_raw)
  aligned <- .adaptms_align_to_columns(target_df, ref_cols)
  if (ncol(aligned) == 0 || nrow(aligned) == 0) return(aligned)

  query_filled <- .adaptms_fill_na_with(aligned, imputer$col_medians)
  query_embed <- .adaptms_project_query(query_filled, imputer)
  idx <- .adaptms_knn_indices(
    prep = imputer,
    query_embed = query_embed,
    exclude_self = exclude_self,
    query_ids = query_ids
  )
  out <- .adaptms_apply_neighbor_imputation(aligned, imputer, idx)
  rownames(out) <- query_ids
  out
}

.adaptms_impute_dataset <- function(df, k = 5, method = NULL) {
  imputer <- .adaptms_prepare_imputer(df, k = k, method = method)
  .adaptms_impute_with_imputer(imputer, df, exclude_self = TRUE)
}
