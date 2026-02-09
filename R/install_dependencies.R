# =============================================================================
# install_dependencies.R
# Install all R packages required by the ADAPT-MS R pipeline
# Run this script once before using the classifiers:
#   source("R/install_dependencies.R")
# =============================================================================

install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message(sprintf("Installing %s...", pkg))
    install.packages(pkg, repos = "https://cloud.r-project.org")
  } else {
    message(sprintf("  %s already installed.", pkg))
  }
}

message("=== ADAPT-MS R: Installing dependencies ===\n")

# Core data manipulation
install_if_missing("readxl")       # Read Excel files (.xlsx)
install_if_missing("readr")        # Fast CSV/TSV reading

# Machine learning & statistics
install_if_missing("glmnet")       # Logistic regression (regularized)
install_if_missing("caret")        # ML utilities (stratified splits, CV folds)
install_if_missing("randomForest") # Random Forest for feature selection
install_if_missing("xgboost")      # XGBoost classifier
install_if_missing("nnet")         # Multinomial logistic regression
install_if_missing("pROC")         # ROC curves and AUC computation
install_if_missing("VIM")          # KNN imputation (kNN function)

# Plotting
install_if_missing("ggplot2")      # Publication-quality plots

# Notebooks
install_if_missing("rmarkdown")    # R Markdown rendering
install_if_missing("knitr")        # Notebook chunk execution

message("\n=== All dependencies installed successfully. ===")
message("You can now source the classifier files, e.g.:")
message('  source("R/AdaptmsClassifier.R")')
