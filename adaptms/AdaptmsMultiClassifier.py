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
from sklearn.preprocessing import LabelEncoder


class AdaptmsMulticlassClassifier:
    def __init__(self, prot_df, cat_df, gene_dict, between_column):
        """
        Initialize the classifier.
        
        Args:
            prot_df: DataFrame with proteomics data
            cat_df: DataFrame with categories
            gene_dict: Dictionary with gene information (optional)
        """
        self.prot_df = prot_df.loc[:, ~prot_df.columns.duplicated()]
        self.cat_df = cat_df
        self.gene_dict = gene_dict
        self.figures = []
        self.models = []
        self.selected_features = set()
        self.predictions = []
        self.feature_names = list(prot_df.columns)
        self.training_data = None
        self.training_targets = None
        self.label_encoder = LabelEncoder()
        self.between_column = between_column
        
    def _perform_feature_selection(self, X_df, y_series, n_features_per_pair=200):
        """
        Perform feature selection using pairwise t-tests between all category combinations
        on unimputed data.
        
        Args:
            X_df: Training features dataframe (unimputed)
            y_series: Training labels series
            n_features_per_pair: Number of top features to select from each pairwise comparison
            
        Returns:
            selected_feature_indices: Indices of selected features
        """
        unique_labels = np.unique(y_series)
        selected_features = set()
        
        # Perform pairwise t-tests between all category combinations
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                label1, label2 = unique_labels[i], unique_labels[j]
                
                # Get data for both categories
                mask1 = y_series == label1
                mask2 = y_series == label2
                group1 = X_df[mask1]
                group2 = X_df[mask2]
                
                # Perform t-test for each feature
                p_values = []
                for feat_name in X_df.columns:
                    # Extract non-NA values for this feature from both groups
                    feat1 = group1[feat_name].dropna()
                    feat2 = group2[feat_name].dropna()
                    
                    # Only perform t-test if we have enough data points
                    if len(feat1) > 5 and len(feat2) > 5:
                        t_stat, p_val = ttest_ind(feat1, feat2)
                        p_values.append((feat_name, p_val))
                    else:
                        p_values.append((feat_name, 1.0))  # High p-value for features with insufficient data
                
                # Correct for multiple testing using FDR
                feat_names, p_vals = zip(*p_values)
                _, corrected_p_values, _, _ = multipletests(p_vals, method='fdr_bh')
                
                # Sort features by corrected p-values
                sorted_features = [(feat, p) for feat, p in zip(feat_names, corrected_p_values)]
                sorted_features.sort(key=lambda x: x[1])
                
                # Select top features for this pair
                top_features = [feat for feat, _ in sorted_features[:n_features_per_pair]]
                selected_features.update(top_features)
        
        # Update class-level selected features
        self.selected_features.update(selected_features)
        
        # Return indices of selected features in the original dataframe
        selected_indices = [X_df.columns.get_loc(feat) for feat in selected_features if feat in X_df.columns]
        return list(selected_features), selected_indices

    def classify_and_plot(self, categories=None, n_runs=3, n_features=100):
        """
        Perform classification and plot ROC curves for each class (one-vs-rest).
        
        Args:
            categories: List of categories to classify (if None, use all categories)
            n_runs: Number of classification runs
        """
        # Merge the dataframes on the index
        d_ML_original = self.prot_df.join(self.cat_df[[self.between_column]], how='inner')
        
        # Filter categories if specified
        y_original = d_ML_original[self.between_column]
        if categories:
            filtered_indices = y_original.isin(categories)
            d_ML_original = d_ML_original.loc[filtered_indices]
            y_original = y_original.loc[filtered_indices]
        
        # Store training targets and fit label encoder
        self.training_targets = y_original
        self.label_encoder.fit(y_original)
        y_original_encoded = self.label_encoder.transform(y_original)
        
        # Prepare imputed version for ML
        imputer = KNNImputer(n_neighbors=5)
        X_imputed = imputer.fit_transform(d_ML_original.drop(columns=[self.between_column]))
        
        # Store clean training data
        self.training_data = pd.DataFrame(X_imputed, index=d_ML_original.index, 
                                        columns=d_ML_original.drop(columns=[self.between_column]).columns)
        
        mean_fpr = np.linspace(0, 1, 100)
        unique_labels = np.unique(y_original)
        
        # Store ROC curves for each class
        class_curves = {label: {'tprs': [], 'aucs': []} for label in unique_labels}
        
        for i in tqdm(range(n_runs)):
            # Split into training and test sets
            X_orig_train, X_orig_test, X_imp_train, X_imp_test, y_train, y_test = train_test_split(
                d_ML_original.drop(columns=[self.between_column]),
                self.training_data,
                y_original_encoded, 
                test_size=0.2, 
                stratify=y_original_encoded, 
                random_state=i
            )
            
            # Perform feature selection on UNIMPUTED data
            selected_feature_names, _ = self._perform_feature_selection(X_orig_train, 
                                                        self.label_encoder.inverse_transform(y_train), n_features_per_pair=n_features)
            
            # Use these selected features on the imputed data for ML
            X_imp_train_selected = X_imp_train[selected_feature_names]
            
            # Create and train the model for this run
            model = LogisticRegression(max_iter=10000, random_state=i)
            self.models.append(model)
            
            # Perform cross-validation using imputed data
            cv = StratifiedKFold(n_splits=5)
            
            for train_idx, val_idx in cv.split(X_imp_train_selected, y_train):
                X_cv_train = X_imp_train_selected.iloc[train_idx]
                X_cv_val = X_imp_train_selected.iloc[val_idx]
                y_cv_train = y_train[train_idx]
                y_cv_val = y_train[val_idx]
                
                # Train and predict probabilities
                model.fit(X_cv_train, y_cv_train)
                probas_ = model.predict_proba(X_cv_val)
                
                # Calculate ROC curve for each class (one-vs-rest)
                for idx, encoded_label in enumerate(model.classes_):
                    original_label = self.label_encoder.inverse_transform([encoded_label])[0]
                    fpr, tpr, _ = roc_curve((y_cv_val == encoded_label).astype(int), probas_[:, idx])
                    class_curves[original_label]['tprs'].append(interp(mean_fpr, fpr, tpr))
                    class_curves[original_label]['tprs'][-1][0] = 0.0
                    
                    # Calculate AUC for this fold
                    auc = roc_auc_score((y_cv_val == encoded_label).astype(int), probas_[:, idx])
                    class_curves[original_label]['aucs'].append(auc)
        
        # Plot ROC curves for all classes
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            tprs = np.array(class_curves[label]['tprs'])
            aucs = np.array(class_curves[label]['aucs'])
            
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            
            # Plot mean ROC curve
            plt.plot(mean_fpr, mean_tpr, color=color,
                    label=f'{label} (AUC = {mean_auc:.2f} ± {std_auc:.2f})',
                    lw=2, alpha=.8)
            
            # Plot confidence interval
            std_tpr = np.std(tprs, axis=0)
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper,
                           color=color, alpha=.2)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Cross-validation ROC Curves (One-vs-Rest)')
        plt.legend(loc='lower right')
        self.figures.append(plt.gcf())
        plt.show()

    def classify_dataframe(self, validation_df, validation_cat_df):
        """
        Classify samples in a validation dataset.
        
        Args:
            validation_df: DataFrame with validation proteomics data
            validation_cat_df: DataFrame with validation categories
        """
        if not self.selected_features:
            raise ValueError("No features selected. Run 'classify_and_plot' first.")
            
        for idx, row in tqdm(validation_df.iterrows()):
            # Handle the validation sample
            d_sample = row.to_frame().T
            
            # Find the intersection of:
            # 1. Features selected from discovery
            # 2. Features available in the validation sample (not NaN)
            available_features = [feat for feat in self.selected_features 
                                 if feat in d_sample.columns and not pd.isna(d_sample[feat].iloc[0])]
            
            # Skip samples with too few available features
            if len(available_features) < 5:
                print(f"Sample {idx} has too few available features. Skipping.")
                continue
                
            # Extract only non-NA values for this sample
            d_sample = d_sample[available_features]
            
            # Get true label if available
            true_label = validation_cat_df.loc[idx, self.between_column] if idx in validation_cat_df.index else "Unknown"
            
            # Re-train model using stored imputed training data but only with available features
            X_train_filtered = self.training_data[available_features].copy()
            y_train_filtered = self.training_targets.dropna()
            X_train_filtered = X_train_filtered.loc[y_train_filtered.index]
            
            # Train model on discovery data with only these features
            model = LogisticRegression(max_iter=10000)
            # Encode labels for training
            y_train_encoded = self.label_encoder.transform(y_train_filtered)
            model.fit(X_train_filtered, y_train_encoded)
            
            # Predict probabilities for all classes (no imputation of validation sample)
            probs = model.predict_proba(d_sample)[0]
            self.predictions.append((idx, true_label, probs))
            
    def plot_validation_roc(self):
        """Plot ROC curves for validation data (one-vs-rest for each class)."""
        if not self.predictions:
            raise ValueError("No predictions available. Run 'classify_dataframe' first.")
        
        sample_names, true_labels, pred_probs = zip(*self.predictions)
        
        # Filter out unknown labels for ROC calculation
        known_indices = [i for i, label in enumerate(true_labels) if label != "Unknown"]
        if not known_indices:
            raise ValueError("No samples with known labels in validation data.")
            
        filtered_labels = [true_labels[i] for i in known_indices]
        filtered_probs = [pred_probs[i] for i in known_indices]
        
        unique_labels = np.unique(filtered_labels)
        
        plt.figure(figsize=(10, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        # Calculate and plot ROC curve for each class
        for idx, (label, color) in enumerate(zip(unique_labels, colors)):
            true_bin = (np.array(filtered_labels) == label).astype(int)
            
            # Extract probabilities for this class
            class_idx = list(self.label_encoder.classes_).index(label) if label in self.label_encoder.classes_ else idx
            pred_probs_class = np.array([p[class_idx] if class_idx < len(p) else 0 for p in filtered_probs])
            
            fpr, tpr, _ = roc_curve(true_bin, pred_probs_class)
            auc = roc_auc_score(true_bin, pred_probs_class)
            
            plt.plot(fpr, tpr, color=color, 
                    label=f'{label} (AUC = {auc:.2f})',
                    lw=2, alpha=.8)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Validation ROC Curves (One-vs-Rest)')
        plt.legend(loc='lower right')
        self.figures.append(plt.gcf())
        plt.show()
        
    def save_figures(self, output_directory, prefix=""):
        """
        Save all figures to the specified directory.
        
        Args:
            output_directory: Directory to save figures
            prefix: Prefix for filenames
        """
        os.makedirs(output_directory, exist_ok=True)
        
        for i, fig in enumerate(self.figures):
            filename = os.path.join(output_directory, f"{prefix}figure_{i+1}.pdf")
            fig.savefig(filename)
            print(f"Saved figure to {filename}")