# =============================================================================
# install_dependencies.R
# Convenience helpers for installing packages used by ADAPT-MS workflows.
# =============================================================================

.adaptms_runtime_packages <- function() {
  c(
    "caret",
    "randomForest",
    "xgboost",
    "nnet",
    "pROC",
    "VIM",
    "RANN",
    "ggplot2"
  )
}

.adaptms_optional_packages <- function() {
  c("readxl", "readr", "rmarkdown", "knitr")
}

install_if_missing <- function(pkg, repos = "https://cloud.r-project.org") {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Installing %s ...", pkg))
    install.packages(pkg, repos = repos)
  } else {
    message(sprintf("%s is already installed.", pkg))
  }
  invisible(pkg)
}

install_adaptms_dependencies <- function(include_optional = TRUE,
                                         repos = "https://cloud.r-project.org") {
  pkgs <- .adaptms_runtime_packages()
  if (isTRUE(include_optional)) {
    pkgs <- c(pkgs, .adaptms_optional_packages())
  }

  message("Installing ADAPT-MS dependencies")
  invisible(lapply(pkgs, install_if_missing, repos = repos))
}
