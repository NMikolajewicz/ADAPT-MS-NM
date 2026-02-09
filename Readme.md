# ADAPT-MS basic code and notebooks

<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/othneildrew/Best-README-Template">
    <img src="data/Graphical_abstract.jpg" alt="Logo" width="800" height="560">
  </a>
  </p>
</div>

## Description
- Notebooks exemplifying ADAPT-MS on two proteomics datasets
- ML function wrappers for baseline classifiers and ADAPT-MS architecture
- Example data to reproduce paper figures and test ML pipelines
- **Available in both Python and R implementations**

---

## R Implementation

The `R/` directory contains a full R port of the ADAPT-MS pipeline. All six classifier architectures are faithfully translated, preserving the same statistical methods, feature selection, imputation, and per-sample refitting logic.

### Quick Start (R)

```r
# 1. Install dependencies (run once)
source("R/install_dependencies.R")

# 2. Source a classifier
source("R/AdaptmsClassifier.R")

# 3. Load your data
prot_df <- read.delim("data/your_proteomics.tsv", sep = "\t", check.names = FALSE)
# ... preprocessing (see notebooks_R/ for full examples)

# 4. Classify
classifier <- AdaptmsClassifierDF$new(prot_df, cat_df, gene_dict, between = "target_column")
classifier$classify_and_plot("class_A", "class_B", n_runs = 10)
classifier$classify_dataframe(validation_df, validation_cat_df)
classifier$plot_validation_roc("class_A", "class_B")
classifier$save_plots_to_pdf("results.pdf")
```

### R Dependencies

Install all at once with `source("R/install_dependencies.R")`, or manually:

| R Package | Purpose | Python Equivalent |
|-----------|---------|-------------------|
| `glmnet` | Logistic regression | `sklearn.linear_model.LogisticRegression` |
| `caret` | Stratified splits, CV folds | `sklearn.model_selection` |
| `randomForest` | RF feature selection | `sklearn.ensemble.RandomForestClassifier` |
| `xgboost` | XGBoost classifier | `xgboost.XGBClassifier` |
| `nnet` | Multinomial logistic regression | `sklearn.linear_model.LogisticRegression` (multiclass) |
| `pROC` | ROC curves and AUC | `sklearn.metrics.roc_curve, roc_auc_score` |
| `VIM` | KNN imputation | `sklearn.impute.KNNImputer` |
| `ggplot2` | Plotting | `matplotlib` |
| `readxl` | Excel file reading | `pandas.read_excel` |

### R Classifier Files

| File | Python Equivalent | Description |
|------|-------------------|-------------|
| `R/AdaptmsClassifier.R` | `utils/AdaptmsClassifier.py` | Binary ADAPT-MS with per-sample refitting (DataFrame and Folder variants) |
| `R/AdaptmsMultiClassifier.R` | `utils/AdaptmsMultiClassifier.py` | Multiclass ADAPT-MS (one-vs-rest with per-sample refitting) |
| `R/BaselineClassifier.R` | `utils/BaselineClassifier.py` | RF feature selection + XGBoost (fixed model baseline) |
| `R/NonRefitClassifier.R` | `utils/NonRefitClassifier.py` | t-test + logistic regression without refitting (ablation study) |
| `R/NonValClassifier.R` | `utils/NonValClassifier.py` | XGBoost with repeated splits (no external validation) |
| `R/XGBfillClassifier.R` | `utils/XGBfillClassifier.py` | XGBoost with zero-fill validation (DataFrame and Folder variants) |
| `R/install_dependencies.R` | `requirements.txt` | One-command dependency installer |

### R Notebook Examples

The `notebooks_R/` directory contains R Markdown notebooks mirroring the Python Jupyter notebooks:

| R Notebook | Python Notebook | Dataset |
|------------|-----------------|---------|
| `ADAPT-MS_Xue_study.Rmd` | `20250310_ADAPT-MS_Xue.ipynb` | Published Xue dataset |
| `ADAPT-MS_AD_CSF_full_cohort.Rmd` | `20250316_ADAPT-MS_AD_CSF_full_cohort.ipynb` | Full AD CSF study |
| `ADAPT-MS_sepsis_study.Rmd` | `20250310_ADAPT-MS_sepsis_study.ipynb` | Sepsis dataset |
| `ADAPT-MS_AD_CSF_cross_cohort.Rmd` | `20250316_ADAPT-MS_AD_CSF_train_on_Sweden+Kiel_apply_to_Berlin.ipynb` | Cross-cohort validation |

Run notebooks in RStudio or from the command line:

```r
rmarkdown::render("notebooks_R/ADAPT-MS_Xue_study.Rmd")
```

### API Reference (R)

All classifiers follow the same interface pattern:

```r
# Constructor
classifier <- AdaptmsClassifierDF$new(prot_df, cat_df, gene_dict, between)

# Discovery (training with cross-validation ROC)
classifier$classify_and_plot(category1, category2, n_runs = 10, topn_features = 50)

# Validation (per-sample refitting for ADAPT-MS; fixed model for baselines)
classifier$classify_dataframe(validation_df, validation_cat_df)
# OR for folder-based:
classifier$classify_directory(directory, cat_pool, category1, category2)

# Plot validation results
classifier$plot_validation_roc(category1, category2)

# Save all figures
classifier$save_plots_to_pdf("output.pdf")
```

**Data format requirements** (same as Python):
- `prot_df`: data.frame with samples as rows, proteins as columns, row names = sample IDs
- `cat_df`: data.frame with sample metadata, row names = sample IDs
- `between`: character string naming the target column in `cat_df`

### Key Implementation Notes

- **KNN imputation**: Uses `VIM::kNN()` (equivalent to sklearn's `KNNImputer`)
- **Logistic regression**: Uses base R `glm()` with `family = binomial` (binary) or `nnet::multinom()` (multiclass)
- **Feature selection**: Uses `stats::p.adjust(method = "BH")` for FDR correction (equivalent to `statsmodels.multipletests`)
- **XGBoost**: Uses the `xgboost` R package with `xgb.DMatrix` for native NaN handling
- **ROC/AUC**: Uses `pROC::roc()` and `pROC::auc()`
- **Plots**: All plots use `ggplot2` and can be saved to PDF

---

## Python Implementation (Original)

### Quick Start (Python)

```bash
pip install -r requirements.txt
pip install seaborn openpyxl
pip install -e .
jupyter notebook notebooks/
```

### Python Dependencies

See `requirements.txt` for version-pinned dependencies.
