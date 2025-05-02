import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from numpy import interp
from sklearn.ensemble import RandomForestClassifier
from matplotlib.backends.backend_pdf import PdfPages
import scipy.stats as st

class NonValClassifier:
    def __init__(self, prot_df, cat_df, gene_dict, between=None):
        self.prot_df = prot_df
        self.cat_df = cat_df
        self.gene_dict = gene_dict
        self.figures = []  # To store the figures for saving to a PDF
        self.between = between
        
    def classify_and_plot_multiple(self, category1, category2, num_repeats=10):
        """
        Run the classification process multiple times and generate aggregated ROC curve with CI.
        
        Parameters:
        category1, category2 : str
            The two categories to classify between.
        num_repeats : int
            Number of times to repeat the train-test split and classification process.
        """
        if self.between == None:
            raise ValueError("The 'between' parameter must be set to specify the categories for classification.")
        
        # Handle missing values using KNN Imputer on protein data only
        imputer = KNNImputer(n_neighbors=5)
        prot_df_imputed = pd.DataFrame(
            imputer.fit_transform(self.prot_df),
            columns=self.prot_df.columns,
            index=self.prot_df.index
        )

        # Now merge with the category information
        d_ML = prot_df_imputed.join(self.cat_df, how='inner')
        
        # Separate features and target
        X = d_ML.drop(columns=[self.between]).values
        y = d_ML[self.between]

        # Filter data to include only the specified categories
        filtered_indices = y.isin([category1, category2])
        X_filtered = X[filtered_indices]
        y_filtered = y[filtered_indices]

        # Encode the target variable
        y_filtered = y_filtered.map({category1: 0, category2: 1})
        
        # Storage for test set AUCs across all repeats
        test_aucs = []
        
        # Storage for ROC curve data
        all_test_fprs = []
        all_test_tprs = []
        
        # Repeat the process num_repeats times
        for repeat in range(num_repeats):
            print(f"Running iteration {repeat+1}/{num_repeats}")
            
            # Split into training and test sets with a different random state for each repeat
            random_state = 42 + repeat
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, stratify=y_filtered, random_state=random_state
            )

            # Feature selection
            selector = SelectFromModel(RandomForestClassifier(n_estimators=100))
            selector.fit(X_train, y_train)
            
            # Get the selected features
            selected_features = selector.get_support(indices=True)
            
            # Transform data with selected features
            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)
            
            # Train model and evaluate
            model = XGBClassifier(eval_metric='logloss')
            model.fit(X_train_selected, y_train)
            
            # Evaluate on test set
            y_test_pred_prob = model.predict_proba(X_test_selected)[:, 1]
            fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_prob)
            roc_auc_test = roc_auc_score(y_test, y_test_pred_prob)
            
            # Store the test AUC and ROC curve data
            test_aucs.append(roc_auc_test)
            all_test_fprs.append(fpr_test)
            all_test_tprs.append(tpr_test)
        
        # Calculate the mean and 95% CI for AUC across all repeats
        def calc_overall_ci(arr):
            arr_clean = np.array([x for x in arr if not np.isnan(x)])
            if len(arr_clean) == 0:
                return np.nan, (np.nan, np.nan)
            mean_val = np.mean(arr_clean)
            se = st.sem(arr_clean)  # Standard error
            if np.isnan(se) or len(arr_clean) <= 1:
                return mean_val, (np.nan, np.nan)
            ci = st.t.interval(0.95, len(arr_clean) - 1, loc=mean_val, scale=se)
            return mean_val, ci
        
        # Calculate overall metrics for test set results
        test_auc_mean, test_auc_ci = calc_overall_ci(test_aucs)
        
        # Create aggregated ROC curve with confidence intervals
        plt.figure(figsize=(10, 8))
        
        # Interpolate all ROC curves to a common x-axis for proper averaging
        mean_fpr = np.linspace(0, 1, 100)
        interp_tprs = []
        
        # Plot individual ROC curves with transparency
        for i in range(len(all_test_fprs)):
            fpr = all_test_fprs[i]
            tpr = all_test_tprs[i]
            plt.plot(fpr, tpr, color='grey', alpha=0.3, lw=1)
            interp_tpr = interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            interp_tprs.append(interp_tpr)
        
        # Calculate mean and std of interpolated TPRs
        mean_tpr = np.mean(interp_tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr = np.std(interp_tprs, axis=0)
        
        # Calculate 95% confidence intervals
        tprs_upper = np.minimum(mean_tpr + 1.96 * std_tpr / np.sqrt(len(interp_tprs)), 1)
        tprs_lower = np.maximum(mean_tpr - 1.96 * std_tpr / np.sqrt(len(interp_tprs)), 0)
        
        # Plot mean ROC
        plt.plot(mean_fpr, mean_tpr, color='blue', 
                label=f'Mean ROC (AUC = {test_auc_mean:.2f}, 95% CI: {test_auc_ci[0]:.2f}-{test_auc_ci[1]:.2f})', 
                lw=2, alpha=1)
        
        # Plot confidence interval
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='blue', alpha=0.2, 
                        label=r'95% Confidence Interval')
        
        # Plot diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        
        # Set plot attributes
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'Aggregated ROC Curves with 95% CI (n={num_repeats})', fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        
        # Save the figure
        agg_roc_fig = plt.gcf()
        self.figures.append(agg_roc_fig)
        plt.show()
        
        return test_auc_mean, test_auc_ci
        
    def save_plots_to_pdf(self, pdf_path):
        """
        Save all generated figures to a single PDF file.
        
        Parameters:
        pdf_path : str
            Path to save the PDF file.
        """
        with PdfPages(pdf_path) as pdf:
            for fig in self.figures:
                pdf.savefig(fig)
        print(f"Plots saved to {pdf_path}")