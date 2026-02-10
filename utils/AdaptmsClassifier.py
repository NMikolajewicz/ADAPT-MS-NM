import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from joblib import Parallel, delayed


def _extract_label(value):
    if isinstance(value, pd.Series):
        return value.iloc[0]
    if isinstance(value, np.ndarray):
        return value[0]
    return value


def _vectorized_ttest_pvalues(X_df, y_series, min_non_na=1):
    X_arr = X_df.to_numpy(dtype=np.float64, copy=False)
    y_arr = y_series.to_numpy()

    group0 = X_arr[y_arr == 0]
    group1 = X_arr[y_arr == 1]

    with np.errstate(invalid='ignore'):
        p_values = ttest_ind(group0, group1, axis=0, nan_policy='omit').pvalue

    p_values = np.asarray(p_values, dtype=np.float64)
    count0 = np.sum(~np.isnan(group0), axis=0)
    count1 = np.sum(~np.isnan(group1), axis=0)
    valid_mask = (count0 >= min_non_na) & (count1 >= min_non_na)
    p_values[~np.isfinite(p_values)] = 1.0
    p_values[~valid_mask] = 1.0
    return p_values


def _run_logreg_iteration(seed, X_unimputed, X_imputed, y_filtered, topn_features, min_non_na, max_iter):
    X_train_unimputed, X_test_unimputed, y_train, y_test = train_test_split(
        X_unimputed, y_filtered, test_size=0.2, stratify=y_filtered, random_state=seed
    )

    p_values = _vectorized_ttest_pvalues(X_train_unimputed, y_train, min_non_na=min_non_na)
    _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')

    top_feature_indices = np.argsort(corrected_p_values)[:topn_features]
    top_features = list(X_train_unimputed.columns[top_feature_indices])

    X_train_selected = X_imputed.loc[X_train_unimputed.index, top_features].to_numpy()
    X_test_selected = X_imputed.loc[X_test_unimputed.index, top_features].to_numpy()

    y_train_arr = y_train.to_numpy()
    y_test_arr = y_test.to_numpy()

    mean_fpr = np.linspace(0, 1, 100)
    cv = StratifiedKFold(n_splits=5)
    tprs = []

    for train_idx, val_idx in cv.split(X_train_selected, y_train_arr):
        model_cv = LogisticRegression(max_iter=max_iter, random_state=seed)
        model_cv.fit(X_train_selected[train_idx], y_train_arr[train_idx])
        probas_cv = model_cv.predict_proba(X_train_selected[val_idx])[:, 1]
        fpr, tpr, _ = roc_curve(y_train_arr[val_idx], probas_cv)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

    model = LogisticRegression(max_iter=max_iter, random_state=seed)
    model.fit(X_train_selected, y_train_arr)
    auc = roc_auc_score(y_test_arr, model.predict_proba(X_test_selected)[:, 1])

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    return model, top_features, mean_tpr, auc

class AdaptmsClassifierDF:
    def __init__(self, prot_df, cat_df, gene_dict=None, between=None):
        self.prot_df = prot_df.loc[:, ~prot_df.columns.duplicated()]
        self.cat_df = cat_df
        self.gene_dict = gene_dict
        self.figures = []  # Store figures for saving to a PDF
        self.models = []  # Store trained models
        self.selected_features = set()  # Store selected feature names
        self.predictions = []  # Store sample-by-sample predictions
        self.feature_names = list(prot_df.columns)  # Store original feature set
        self.training_data = None  # Store training data without NaNs
        self.training_targets = None  # Store training targets
        self.between = between

    def classify_and_plot(self, category1, category2, n_runs=10, topn_features=50, n_jobs=1):
        if self.between == None:
            raise ValueError("No between variable given. Please provide a variable for classification.")
        # Merge the dataframes on the index
        self.prot_df = self.prot_df.dropna(how='all', axis=1)

        d_ML = self.prot_df.join(self.cat_df[[self.between]], how='inner')

        # Separate features and target
        X_raw = d_ML.drop(columns=[self.between])
        y = d_ML[self.between]

        # Filter data to include only the specified categories
        filtered_indices = y.isin([category1, category2])
        X_filtered = X_raw[filtered_indices]
        y_filtered = y[filtered_indices]

        # Encode the target variable
        y_filtered = y_filtered.map({category1: 0, category2: 1})

        # Store training targets
        self.training_targets = y_filtered

        # Handle missing values using KNN Imputer for the entire dataset
        # (We'll still keep this for compatibility with other methods)
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = pd.DataFrame(
            imputer.fit_transform(X_filtered),
            index=X_filtered.index,
            columns=X_filtered.columns
        )

        # Store clean training data
        self.training_data = X_imputed

        mean_fpr = np.linspace(0, 1, 100)
        all_tprs = []
        aucs = []

        run_indices = list(range(n_runs))
        if n_jobs == 1:
            run_results = [
                _run_logreg_iteration(
                    i, X_filtered, X_imputed, y_filtered, topn_features, min_non_na=1, max_iter=1000
                )
                for i in run_indices
            ]
        else:
            run_results = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(_run_logreg_iteration)(
                    i, X_filtered, X_imputed, y_filtered, topn_features, min_non_na=1, max_iter=1000
                )
                for i in run_indices
            )

        for model, top_features, mean_tpr, auc in run_results:
            self.models.append(model)
            self.selected_features.update(top_features)
            all_tprs.append(mean_tpr)
            aucs.append(auc)

        # Aggregate TPRs for plotting
        all_tprs = np.array(all_tprs)
        median_tpr = np.median(all_tprs, axis=0)
        lower_tpr = np.percentile(all_tprs, 2.5, axis=0)
        upper_tpr = np.percentile(all_tprs, 97.5, axis=0)

        # Plot aggregated ROC curve
        plt.figure()
        plt.plot(mean_fpr, median_tpr, color='blue', label=f'Median ROC (AUC = {np.mean(aucs):.2f})', lw=2)
        plt.fill_between(mean_fpr, lower_tpr, upper_tpr, color='grey', alpha=0.3, label='95% CI')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Aggregated ROC Curve')
        plt.legend(loc='lower right')
        self.figures.append(plt.gcf())
        plt.show()

    def classify_dataframe(self, validation_df, validation_cat_df):
        selected_feature_list = sorted(self.selected_features)
        if not selected_feature_list:
            raise ValueError("No selected features available. Run 'classify_and_plot' first.")

        validation_aligned = validation_df.reindex(columns=selected_feature_list).apply(pd.to_numeric, errors='coerce')
        y_train_filtered = self.training_targets.dropna()
        X_train_base = self.training_data.loc[y_train_filtered.index]
        model_cache = {}
        feature_array = np.asarray(selected_feature_list, dtype=object)

        for idx, row in tqdm(validation_aligned.iterrows()):
            row_values = row.to_numpy(dtype=np.float64, copy=False)
            available_mask = ~np.isnan(row_values)
            if not np.any(available_mask):
                continue

            available_features = tuple(feature_array[available_mask].tolist())
            model = model_cache.get(available_features)
            if model is None:
                X_train_filtered = X_train_base.loc[:, list(available_features)]
                model = LogisticRegression(max_iter=2000).fit(X_train_filtered, y_train_filtered)
                model_cache[available_features] = model

            sample_values = row_values[available_mask].reshape(1, -1)
            prob = model.predict_proba(sample_values)[:, 1][0]
            true_label = _extract_label(validation_cat_df.loc[idx, self.between])
            self.predictions.append((idx, true_label, prob))

    def plot_validation_roc(self, category1, category2):
        if not self.predictions:
            raise ValueError("No predictions available. Run 'classify_dataframe' first.")

        sample_names, true_labels, pred_probs = zip(*self.predictions)
        label_mapping = {category1: 0, category2: 1}
        true_labels = np.array([label_mapping[label] for label in true_labels])
        pred_probs = np.array(pred_probs)
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        auc = roc_auc_score(true_labels, pred_probs)
        plt.figure()
        plt.plot(fpr, tpr, color='red', label=f'Validation ROC (AUC = {auc:.2f})', lw=2)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', alpha=0.8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        self.figures.append(plt.gcf())
        plt.show()
    def plot_confusion_matrix(self, category1, category2, normalize=True):
        """
        Plot confusion matrix for the validation predictions.
        
        Parameters:
        -----------
        category1 : label for class 0
        category2 : label for class 1
        normalize : bool, whether to normalize the confusion matrix
        """
        if not self.predictions:
            raise ValueError("No predictions available. Run 'classify_dataframe' first.")
        
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        # Extract predictions
        sample_names, true_labels, pred_probs = zip(*self.predictions)
        label_mapping = {category1: 0, category2: 1}
        true_labels = np.array([label_mapping[label] for label in true_labels])
        pred_probs = np.array(pred_probs)
        
        # Convert probabilities to binary predictions (threshold = 0.5)
        pred_labels = (pred_probs > 0.5).astype(int)
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        # Plot
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                    xticklabels=[category1, category2],
                    yticklabels=[category1, category2])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        self.figures.append(plt.gcf())
        plt.show()

    def save_plots_to_pdf(self, file_name):
        if not self.figures:
            print("No figures available. Run classification first.")
            return

        with PdfPages(file_name) as pdf:
            for fig in self.figures:
                pdf.savefig(fig)
            print(f"Plots saved to {file_name}.")

class AdaptmsClassifierFolder:
    def __init__(self, prot_df, cat_df, gene_dict=None, between=None, cohort=None):
        self.prot_df = prot_df.loc[:, ~prot_df.columns.duplicated()]
        self.cat_df = cat_df
        self.gene_dict = gene_dict
        self.figures = []  # Store figures for saving to a PDF
        self.models = []  # Store trained models
        self.selected_features = set()  # Store selected feature names
        self.predictions = []  # Store sample-by-sample predictions
        self.feature_names = list(prot_df.columns)  # Store original feature set
        self.training_data = None  # Store training data
        self.training_targets = None  # Store training targets
        self.between = between
        self.cohort = cohort  # select cohorts by substring of raws
        self._sample_model_cache = {}

    def classify_and_plot(self, category1, category2, n_runs=10, topn_features=50, n_jobs=1):
        if self.between is None:
            raise ValueError("No between variable given. Please provide a variable for classification.")
        # Merge the dataframes on the index
        d_ML = self.prot_df.join(self.cat_df, how='inner')

        # Store unimputed dataset for feature selection
        X_unimputed = d_ML.drop(columns=[self.between]).apply(pd.to_numeric, errors='coerce')
        y = d_ML[self.between]

        # Impute missing values using KNN Imputer for model training
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = pd.DataFrame(imputer.fit_transform(X_unimputed), 
                                 index=X_unimputed.index, 
                                 columns=X_unimputed.columns)

        # Filter data to include only the specified categories
        filtered_indices = y.isin([category1, category2])
        X_filtered_unimputed = X_unimputed.loc[filtered_indices]
        X_filtered_imputed = X_imputed.loc[filtered_indices]
        y_filtered = y.loc[filtered_indices]

        # Encode the target variable
        y_filtered = y_filtered.map({category1: 0, category2: 1})

        # Store training data and targets
        self.training_data = X_filtered_imputed
        self.training_targets = y_filtered

        mean_fpr = np.linspace(0, 1, 100)
        all_tprs = []
        aucs = []

        run_indices = list(range(n_runs))
        if n_jobs == 1:
            run_results = [
                _run_logreg_iteration(
                    i,
                    X_filtered_unimputed,
                    X_filtered_imputed,
                    y_filtered,
                    topn_features,
                    min_non_na=2,
                    max_iter=1000
                )
                for i in run_indices
            ]
        else:
            run_results = Parallel(n_jobs=n_jobs, prefer='threads')(
                delayed(_run_logreg_iteration)(
                    i,
                    X_filtered_unimputed,
                    X_filtered_imputed,
                    y_filtered,
                    topn_features,
                    min_non_na=2,
                    max_iter=1000
                )
                for i in run_indices
            )

        for model, top_features, mean_tpr, auc in run_results:
            self.models.append(model)
            self.selected_features.update(top_features)
            all_tprs.append(mean_tpr)
            aucs.append(auc)

        self._sample_model_cache = {}

        # Aggregate TPRs for plotting
        all_tprs = np.array(all_tprs)
        median_tpr = np.median(all_tprs, axis=0)
        lower_tpr = np.percentile(all_tprs, 2.5, axis=0)
        upper_tpr = np.percentile(all_tprs, 97.5, axis=0)

        # Plot aggregated ROC curve
        plt.figure()
        plt.plot(mean_fpr, median_tpr, color='blue', label=f'Median ROC (AUC = {np.mean(aucs):.2f})', lw=2)
        plt.fill_between(mean_fpr, lower_tpr, upper_tpr, color='grey', alpha=0.3, label='95% CI')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Aggregated ROC Curve')
        plt.legend(loc='lower right')
        self.figures.append(plt.gcf())
        plt.show()
        
    def classify_sample(self, d_sample, true_label):
        if not self.models or not self.selected_features:
            raise ValueError("No trained models available. Run 'classify_and_plot' first.")

        # Get intersection of selected features and available features in the sample
        available_features = tuple(sorted(self.selected_features.intersection(d_sample.columns)))
        if not available_features:
            return

        d_sample = d_sample.loc[:, list(available_features)].fillna(0)

        # Re-train model using only available features from stored training data
        y_train_filtered = self.training_targets.dropna()
        model = self._sample_model_cache.get(available_features)
        if model is None:
            X_train_filtered = self.training_data.loc[y_train_filtered.index, list(available_features)]
            model = LogisticRegression(max_iter=1000).fit(X_train_filtered, y_train_filtered)
            self._sample_model_cache[available_features] = model

        # Predict probability for the sample
        prob = model.predict_proba(d_sample)[:, 1][0]
        self.predictions.append((d_sample.index[0], true_label, prob))

    def classify_directory(self, directory, cat_validation_pool_SF, category1, category2):
        for file in os.listdir(directory):
            if file.endswith("mzML.pg_matrix.tsv"):
                if self.cohort is None or self.cohort in file:
                    file_path = os.path.join(directory, file)
                    d_sample = pd.read_csv(file_path, sep='\t')
                    d_sample = d_sample.iloc[:, np.r_[0, 5]]
                    d_sample.columns = ['Protein.Group', d_sample.columns[1].split('/')[-1]]
                    d_sample = d_sample.set_index('Protein.Group')
                    d_sample = np.log10(d_sample)
                    d_sample = d_sample.T
                    sample_name = d_sample.index[0]
                    true_label = cat_validation_pool_SF.loc[sample_name][self.between]
                    if true_label in [category1, category2]:
                        self.classify_sample(d_sample, true_label)
                    else:
                        pass

    def plot_accumulated_roc(self, category1, category2):
        if not self.predictions:
            raise ValueError("No predictions available. Run 'classify_directory' first.")
        sample_names, true_labels, pred_probs = zip(*self.predictions)
        label_mapping = {category1: 0, category2: 1}
        true_labels = np.array([label_mapping[label] for label in true_labels])
        pred_probs = np.array(pred_probs)
        fpr, tpr, _ = roc_curve(true_labels, pred_probs)
        auc = roc_auc_score(true_labels, pred_probs)
        plt.figure()
        plt.plot(fpr, tpr, color='red', label=f'Aggregated ROC (AUC = {auc:.2f})', lw=2)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', alpha=0.8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        self.figures.append(plt.gcf())
        plt.show()

    def save_plots_to_pdf(self, file_name):
        with PdfPages(file_name) as pdf:
            for fig in self.figures:
                pdf.savefig(fig)
            print(f"Plots saved to {file_name}.")
