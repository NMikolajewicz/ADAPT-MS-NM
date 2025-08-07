import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
from numpy import interp
from matplotlib.backends.backend_pdf import PdfPages
import xgboost as xgb

class XGBRFClassifierDF:
    def __init__(self, prot_df, cat_df, gene_dict, between=None):
        self.prot_df = prot_df.loc[:, ~prot_df.columns.duplicated()]
        self.cat_df = cat_df
        self.gene_dict = gene_dict
        self.figures = []  # Store figures for saving to a PDF
        self.models = []  # Store trained models
        self.selectors = []  # Store RF feature selectors
        self.predictions = []  # Store sample-by-sample predictions
        self.feature_names = list(prot_df.columns)  # Store original feature set
        self.training_data = None  # Store training data
        self.training_targets = None  # Store training targets
        self.between = between
        self.final_model = None  # Store the final trained XGB model
        self.final_selector = None  # Store the final RF selector
        self.selected_features = None  # Store selected feature names
        self.fill_na = None  # Store NaN handling strategy for validation

    def classify_and_plot(self, category1, category2, n_runs=10, n_estimators_rf=200):
        """
        Train XGBoost classifier with Random Forest feature selection
        
        Parameters:
        -----------
        category1, category2 : str
            The two categories to classify between
        n_runs : int
            Number of runs for cross-validation
        n_estimators_rf : int
            Number of trees for Random Forest feature selection
        """
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

        # Store training targets and raw data
        self.training_targets = y_filtered
        self.training_data = X_filtered.copy()

        # Impute missing values for feature selection only
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = imputer.fit_transform(X_filtered)
        X_imputed_df = pd.DataFrame(X_imputed, index=X_filtered.index, columns=X_filtered.columns)

        mean_fpr = np.linspace(0, 1, 100)
        all_tprs = []
        aucs = []
        
        # Collect all selected features across runs
        all_selected_features = set()

        for i in range(n_runs):
            # Split into training and test sets
            X_train_imp, X_test_imp, y_train, y_test = train_test_split(
                X_imputed, y_filtered, test_size=0.2, stratify=y_filtered, random_state=i
            )
            
            # Also get the raw data splits for XGB training
            X_train_raw, X_test_raw, _, _ = train_test_split(
                X_filtered, y_filtered, test_size=0.2, stratify=y_filtered, random_state=i
            )

            # Feature selection using Random Forest on imputed data
            selector = SelectFromModel(
                RandomForestClassifier(n_estimators=n_estimators_rf, random_state=i)
            )
            selector.fit(X_train_imp, y_train)
            
            # Store selector
            self.selectors.append(selector)
            
            # Get selected feature names
            selected_features = X_filtered.columns[selector.get_support()]
            all_selected_features.update(selected_features)
            
            # Extract selected features from raw data (with NaNs)
            X_train_selected = X_train_raw[selected_features].copy()
            X_test_selected = X_test_raw[selected_features].copy()

            # Train XGBoost model on raw data (XGB handles NaNs natively)
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='binary:logistic',
                random_state=i,
                use_label_encoder=False,
                eval_metric='logloss',
                missing=np.nan  # Explicitly handle NaN values
            )
            self.models.append(model)

            # Perform cross-validation
            cv = StratifiedKFold(n_splits=5)
            tprs = []

            for train_idx, val_idx in cv.split(X_train_selected, y_train):
                X_cv_train, X_cv_val = X_train_selected.iloc[train_idx], X_train_selected.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

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

        # Train final selector on all imputed data
        self.final_selector = SelectFromModel(
            RandomForestClassifier(n_estimators=n_estimators_rf, random_state=42)
        )
        self.final_selector.fit(X_imputed, self.training_targets)
        
        # Get final selected features
        self.selected_features = sorted(X_filtered.columns[self.final_selector.get_support()].tolist())
        
        # Train final XGB model on raw data with selected features
        X_final = self.training_data[self.selected_features].copy()
        
        self.final_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            missing=np.nan  # Explicitly handle NaN values
        )
        self.final_model.fit(X_final, self.training_targets)

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
        plt.title('Aggregated ROC Curve (XGBoost with RF Feature Selection)')
        plt.legend(loc='lower right')
        self.figures.append(plt.gcf())
        plt.show()

    def classify_dataframe(self, validation_df, validation_cat_df, fill_na='zero'):
        """
        Apply the trained XGBoost model to validation samples one by one
        
        Parameters:
        -----------
        validation_df : DataFrame
            Validation proteomics data
        validation_cat_df : DataFrame
            Validation metadata
        fill_na : str
            How to handle NaN values: 'zero' to fill with 0s, 'keep' to keep as NaN (XGB handles them)
        """
        if self.final_model is None or self.selected_features is None:
            raise ValueError("No trained model available. Run 'classify_and_plot' first.")
        
        self.fill_na = fill_na
        
        for idx, row in tqdm(validation_df.iterrows()):
            # Create a sample with all selected features, initialized with NaN
            d_sample = pd.DataFrame(index=[idx], columns=self.selected_features, dtype=np.float64)
            
            # Fill in values for features that exist in the validation sample
            for feature in self.selected_features:
                if feature in row.index:
                    d_sample.loc[idx, feature] = float(row[feature]) if pd.notna(row[feature]) else np.nan
                else:
                    # Feature not present in validation sample, will be NaN
                    d_sample.loc[idx, feature] = np.nan
            
            # Handle NaN values according to strategy
            if self.fill_na == 'zero':
                d_sample = d_sample.fillna(0)
            # If 'keep', XGBoost will handle NaNs natively
            
            # Ensure all columns are float type
            d_sample = d_sample.astype(np.float64)
            
            # Get true label
            true_label = validation_cat_df.loc[idx, self.between]

            # Predict probability using the final trained model
            prob = self.final_model.predict_proba(d_sample)[:, 1][0]
            self.predictions.append((idx, true_label, prob))

    def plot_validation_roc(self, category1, category2):
        """
        Plot ROC curve for validation predictions
        """
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
        plt.title('Validation ROC Curve (XGBoost with RF Feature Selection)')
        plt.legend(loc='lower right')
        self.figures.append(plt.gcf())
        plt.show()

    def save_plots_to_pdf(self, file_name):
        """
        Save all generated plots to a PDF file
        """
        if not self.figures:
            print("No figures available. Run classification first.")
            return

        with PdfPages(file_name) as pdf:
            for fig in self.figures:
                pdf.savefig(fig)
            print(f"Plots saved to {file_name}.")
            
    def get_feature_importance(self):
        """
        Get feature importance from the final XGBoost model
        """
        if self.final_model is None:
            raise ValueError("No trained model available. Run 'classify_and_plot' first.")
        
        feature_importance = pd.DataFrame({
            'feature': self.selected_features,
            'importance': self.final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance


class XGBRFClassifierFolder:
    def __init__(self, prot_df, cat_df, gene_dict, between=None, cohort=None):
        self.prot_df = prot_df.loc[:, ~prot_df.columns.duplicated()]
        self.cat_df = cat_df
        self.gene_dict = gene_dict
        self.figures = []  # Store figures for saving to a PDF
        self.models = []  # Store trained models
        self.selectors = []  # Store RF feature selectors
        self.predictions = []  # Store sample-by-sample predictions
        self.feature_names = list(prot_df.columns)  # Store original feature set
        self.training_data = None  # Store training data
        self.training_targets = None  # Store training targets
        self.between = between
        self.cohort = cohort  # select cohorts by substring of raws
        self.final_model = None  # Store the final trained XGB model
        self.final_selector = None  # Store the final RF selector
        self.selected_features = None  # Store selected feature names
        self.fill_na = None  # Store NaN handling strategy for validation

    def classify_and_plot(self, category1, category2, n_runs=10, n_estimators_rf=200):
        """
        Train XGBoost classifier with Random Forest feature selection
        
        Parameters:
        -----------
        category1, category2 : str
            The two categories to classify between
        n_runs : int
            Number of runs for cross-validation
        n_estimators_rf : int
            Number of trees for Random Forest feature selection
        """
        if self.between is None:
            raise ValueError("No between variable given. Please provide a variable for classification.")
        
        # Merge the dataframes on the index
        d_ML = self.prot_df.join(self.cat_df[[self.between]], how='inner')

        # Separate features and target
        X_raw = d_ML.drop(columns=[self.between]).apply(pd.to_numeric, errors='coerce')
        y = d_ML[self.between]

        # Filter data to include only the specified categories
        filtered_indices = y.isin([category1, category2])
        X_filtered = X_raw[filtered_indices]
        y_filtered = y[filtered_indices]

        # Encode the target variable
        y_filtered = y_filtered.map({category1: 0, category2: 1})

        # Store training targets and raw data
        self.training_targets = y_filtered
        self.training_data = X_filtered.copy()

        # Impute missing values for feature selection only
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = imputer.fit_transform(X_filtered)
        X_imputed_df = pd.DataFrame(X_imputed, index=X_filtered.index, columns=X_filtered.columns)

        mean_fpr = np.linspace(0, 1, 100)
        all_tprs = []
        aucs = []

        for i in range(n_runs):
            # Split into training and test sets
            X_train_imp, X_test_imp, y_train, y_test = train_test_split(
                X_imputed, y_filtered, test_size=0.2, stratify=y_filtered, random_state=i
            )
            
            # Also get the raw data splits for XGB training
            X_train_raw, X_test_raw, _, _ = train_test_split(
                X_filtered, y_filtered, test_size=0.2, stratify=y_filtered, random_state=i
            )

            # Feature selection using Random Forest on imputed data
            selector = SelectFromModel(
                RandomForestClassifier(n_estimators=n_estimators_rf, random_state=i)
            )
            selector.fit(X_train_imp, y_train)
            
            # Store selector
            self.selectors.append(selector)
            
            # Get selected feature names
            selected_features = X_filtered.columns[selector.get_support()]
            
            # Extract selected features from raw data (with NaNs)
            X_train_selected = X_train_raw[selected_features].copy()
            X_test_selected = X_test_raw[selected_features].copy()

            # Train XGBoost model on raw data (XGB handles NaNs natively)
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='binary:logistic',
                random_state=i,
                use_label_encoder=False,
                eval_metric='logloss',
                missing=np.nan  # Explicitly handle NaN values
            )
            self.models.append(model)

            # Perform cross-validation
            cv = StratifiedKFold(n_splits=5)
            tprs = []
            for train_idx, val_idx in cv.split(X_train_selected, y_train):
                X_cv_train, X_cv_val = X_train_selected.iloc[train_idx], X_train_selected.iloc[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Train and predict probabilities
                model.fit(X_cv_train, y_cv_train)
                probas_ = model.predict_proba(X_cv_val)
                fpr, tpr, _ = roc_curve(y_cv_val, probas_[:, 1])
                tprs.append(interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0

            # Final train on the whole training set
            model.fit(X_train_selected, y_train)

            # Aggregate TPRs and AUCs
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            auc = roc_auc_score(y_test, model.predict_proba(X_test_selected)[:, 1])
            aucs.append(auc)
            all_tprs.append(mean_tpr)

        # Train final selector on all imputed data
        self.final_selector = SelectFromModel(
            RandomForestClassifier(n_estimators=n_estimators_rf, random_state=42)
        )
        self.final_selector.fit(X_imputed, self.training_targets)
        
        # Get final selected features
        self.selected_features = sorted(X_filtered.columns[self.final_selector.get_support()].tolist())
        
        # Train final XGB model on raw data with selected features
        X_final = self.training_data[self.selected_features].copy()
        
        self.final_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic',
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            missing=np.nan  # Explicitly handle NaN values
        )
        self.final_model.fit(X_final, self.training_targets)

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
        plt.title('Aggregated ROC Curve (XGBoost with RF Feature Selection)')
        plt.legend(loc='lower right')
        self.figures.append(plt.gcf())
        plt.show()
        
    def classify_sample(self, d_sample, true_label):
        """
        Classify a single sample using the trained XGBoost model
        """
        if self.final_model is None or self.selected_features is None:
            raise ValueError("No trained model available. Run 'classify_and_plot' first.")

        # Create a sample with all selected features, initialized with NaN
        d_sample_aligned = pd.DataFrame(index=d_sample.index, columns=self.selected_features, dtype=np.float64)
        
        # Fill in values for features that exist in the sample
        for feature in self.selected_features:
            if feature in d_sample.columns:
                d_sample_aligned.loc[:, feature] = float(d_sample[feature].iloc[0]) if pd.notna(d_sample[feature].iloc[0]) else np.nan
            else:
                # Feature not present in sample, will be NaN
                d_sample_aligned.loc[:, feature] = np.nan
        
        # Handle NaN values according to strategy
        if self.fill_na == 'zero':
            d_sample_aligned = d_sample_aligned.fillna(0)
        # If 'keep', XGBoost will handle NaNs natively
        
        # Ensure all columns are float type
        d_sample_aligned = d_sample_aligned.astype(np.float64)

        # Predict probability using the final trained model
        prob = self.final_model.predict_proba(d_sample_aligned)[:, 1][0]
        self.predictions.append((d_sample.index[0], true_label, prob))

    def classify_directory(self, directory, cat_validation_pool_SF, category1, category2, fill_na='zero'):
        """
        Classify all samples in a directory
        
        Parameters:
        -----------
        directory : str
            Path to directory containing sample files
        cat_validation_pool_SF : DataFrame
            Validation metadata
        category1, category2 : str
            Categories to classify
        fill_na : str
            How to handle NaN values: 'zero' to fill with 0s, 'keep' to keep as NaN (XGB handles them)
        """
        self.fill_na = fill_na
        
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
                    
                    if sample_name in cat_validation_pool_SF.index:
                        true_label = cat_validation_pool_SF.loc[sample_name][self.between]
                        if true_label in [category1, category2]:
                            self.classify_sample(d_sample, true_label)
                    else:
                        print(f"Warning: Sample {sample_name} not found in validation metadata")

    def plot_accumulated_roc(self, category1, category2):
        """
        Plot ROC curve for all accumulated predictions
        """
        if not self.predictions:
            raise ValueError("No predictions available. Run 'classify_directory' first.")
            
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
        plt.title('Validation ROC Curve (XGBoost with RF Feature Selection)')
        plt.legend(loc='lower right')
        self.figures.append(plt.gcf())
        plt.show()
        
        # Print summary statistics
        print(f"Total samples classified: {len(self.predictions)}")
        print(f"AUC: {auc:.3f}")
        
        return auc

    def save_plots_to_pdf(self, file_name):
        """
        Save all generated plots to a PDF file
        """
        if not self.figures:
            print("No figures available. Run classification first.")
            return

        with PdfPages(file_name) as pdf:
            for fig in self.figures:
                pdf.savefig(fig)
            print(f"Plots saved to {file_name}.")
            
    def get_feature_importance(self):
        """
        Get feature importance from the final XGBoost model
        """
        if self.final_model is None:
            raise ValueError("No trained model available. Run 'classify_and_plot' first.")
        
        feature_importance = pd.DataFrame({
            'feature': self.selected_features,
            'importance': self.final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return feature_importance
    
    def get_predictions_df(self):
        """
        Return predictions as a DataFrame for further analysis
        """
        if not self.predictions:
            raise ValueError("No predictions available. Run 'classify_directory' first.")
        
        sample_names, true_labels, pred_probs = zip(*self.predictions)
        return pd.DataFrame({
            'sample': sample_names,
            'true_label': true_labels,
            'predicted_probability': pred_probs
        })