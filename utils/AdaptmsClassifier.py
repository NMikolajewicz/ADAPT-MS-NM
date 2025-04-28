import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from numpy import interp
from matplotlib.backends.backend_pdf import PdfPages

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from numpy import interp
from matplotlib.backends.backend_pdf import PdfPages


class AdaptmsClassifierDF:
    def __init__(self, prot_df, cat_df, gene_dict, between=None):
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

    def classify_and_plot(self, category1, category2, n_runs=10, topn_features=50):
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
        X = imputer.fit_transform(X_filtered)

        # Store clean training data
        self.training_data = pd.DataFrame(X, index=X_filtered.index, columns=X_filtered.columns)

        mean_fpr = np.linspace(0, 1, 100)
        all_tprs = []
        aucs = []

        for i in range(n_runs):
            # Split into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, stratify=y_filtered, random_state=i
            )

            # Perform multiple t-tests for feature selection on raw data (with NaNs)
            group1 = X_train[y_train == 0]
            group2 = X_train[y_train == 1]

            # Perform t-tests with nan-omit policy
            p_values = []
            for col in X_train.columns:
                g1_vals = group1[col].dropna()
                g2_vals = group2[col].dropna()
                if len(g1_vals) > 0 and len(g2_vals) > 0:
                    p_values.append(ttest_ind(g1_vals, g2_vals)[1])
                else:
                    p_values.append(1.0)  # If no data in either group, assign high p-value

            # Correct for multiple testing using FDR
            _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')

            # Select the top 20 features by p-value
            top_feature_indices = np.argsort(corrected_p_values)[:topn_features]
            top_features = list(X_train.columns[top_feature_indices])
            self.selected_features.update(top_features)

            # Get the imputed data for these features
            # We need to map from the raw data to the imputed data indices
            X_train_imputed_idx = [X_filtered.index.get_indexer([idx])[0] for idx in X_train.index]
            X_test_imputed_idx = [X_filtered.index.get_indexer([idx])[0] for idx in X_test.index]

            # Get the column indices for the selected features
            col_indices = [X_filtered.columns.get_indexer([col])[0] for col in top_features]

            # Extract the selected features from the imputed data
            X_train_selected = X[X_train_imputed_idx][:, col_indices]
            X_test_selected = X[X_test_imputed_idx][:, col_indices]

            # Train logistic regression model
            model = LogisticRegression(max_iter=1000, random_state=i)
            self.models.append(model)

            # Perform cross-validation
            cv = StratifiedKFold(n_splits=5)
            tprs = []

            # Convert to NumPy arrays for compatibility with scikit-learn
            X_train_arr = np.array(X_train_selected)
            y_train_arr = np.array(y_train)

            for train_idx, val_idx in cv.split(X_train_arr, y_train_arr):
                X_cv_train, X_cv_val = X_train_arr[train_idx], X_train_arr[val_idx]
                y_cv_train, y_cv_val = y_train_arr[train_idx], y_train_arr[val_idx]

                # Train and predict probabilities
                model.fit(X_cv_train, y_cv_train)
                probas_ = model.predict_proba(X_cv_val)
                fpr, tpr, _ = roc_curve(y_cv_val, probas_[:, 1])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0

            # Final train on the whole training set
            model.fit(X_train_selected, y_train)

            # Aggregate TPRs and AUCs
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            auc = roc_auc_score(y_test, model.predict_proba(X_test_selected)[:, 1])
            aucs.append(auc)
            all_tprs.append(mean_tpr)

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
        for idx, row in tqdm(validation_df.iterrows()):
            d_sample = row.dropna().to_frame().T  # Convert to DataFrame and drop NaNs
            available_features = list(self.selected_features.intersection(d_sample.columns))
            d_sample = d_sample[available_features]

            true_label = validation_cat_df.loc[idx, self.between]

            # Re-train model using stored training data
            X_train_filtered = self.training_data[available_features].copy()
            y_train_filtered = self.training_targets.dropna()
            X_train_filtered = X_train_filtered.loc[y_train_filtered.index]

            model = LogisticRegression(max_iter=1000).fit(X_train_filtered, y_train_filtered)

            # Predict probability for the sample
            prob = model.predict_proba(d_sample)[:, 1][0]
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

    def save_plots_to_pdf(self, file_name):
        if not self.figures:
            print("No figures available. Run classification first.")
            return

        with PdfPages(file_name) as pdf:
            for fig in self.figures:
                pdf.savefig(fig)
            print(f"Plots saved to {file_name}.")

class AdaptmsClassifierFolder:
    def __init__(self, prot_df, cat_df, gene_dict, between=None):
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

    def classify_and_plot(self, category1, category2, n_runs=10, topn_features=50):
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

        for i in range(n_runs):
            # Split into training and test sets
            X_train_unimputed, X_test_unimputed, y_train, y_test = train_test_split(
                X_filtered_unimputed, y_filtered, test_size=0.2, stratify=y_filtered, random_state=i
            )
            X_train_imputed, X_test_imputed = X_filtered_imputed.loc[X_train_unimputed.index], X_filtered_imputed.loc[X_test_unimputed.index]

            # Perform multiple t-tests for feature selection
            p_values = []
            for col in X_train_unimputed.columns:
                group1 = X_train_unimputed.loc[y_train == 0, col].dropna()
                group2 = X_train_unimputed.loc[y_train == 1, col].dropna()
                if len(group1) > 1 and len(group2) > 1:
                    p_values.append(ttest_ind(group1, group2, nan_policy='omit')[1])
                else:
                    p_values.append(1.0)

            # Correct for multiple testing using FDR
            _, corrected_p_values, _, _ = multipletests(p_values, method='fdr_bh')

            # Select the top 120 features by p-value
            top_features = list(X_train_unimputed.columns[np.argsort(corrected_p_values)[:topn_features]])
            self.selected_features.update(top_features)

            X_train_selected = X_train_imputed[top_features]
            X_test_selected = X_test_imputed[top_features]

            # Train logistic regression classifier
            model = LogisticRegression(max_iter=1000, random_state=i)
            self.models.append(model)

            # Perform cross-validation
            cv = StratifiedKFold(n_splits=5)
            tprs = []
            for train_idx, val_idx in cv.split(X_train_selected, y_train):
                X_cv_train, X_cv_val = X_train_selected.iloc[train_idx], X_train_selected.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Train and predict probabilities
                probas_ = model.fit(X_cv_train, y_cv_train).predict_proba(X_cv_val)
                fpr, tpr, _ = roc_curve(y_cv_val, probas_[:, 1])
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0

            # Aggregate TPRs and AUCs
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            auc = roc_auc_score(y_test, model.predict_proba(X_test_selected)[:, 1])
            aucs.append(auc)
            all_tprs.append(mean_tpr)

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
        available_features = list(self.selected_features.intersection(d_sample.columns))
        d_sample = d_sample[available_features].copy()
        d_sample = d_sample.fillna(0)

        # Re-train model using only available features from stored training data
        X_train_filtered = self.training_data[available_features].copy()
        y_train_filtered = self.training_targets.dropna()
        X_train_filtered = X_train_filtered.loc[y_train_filtered.index]

        model = LogisticRegression(max_iter=1000).fit(X_train_filtered, y_train_filtered)

        # Predict probability for the sample
        prob = model.predict_proba(d_sample)[:, 1][0]
        self.predictions.append((d_sample.index[0], true_label, prob))

    def classify_directory(self, directory, cat_validation_pool_SF, category1, category2):
        for file in os.listdir(directory):
            if file.endswith("mzML.pg_matrix.tsv"):
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