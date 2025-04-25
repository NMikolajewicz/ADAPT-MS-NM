import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


class BaselineClassifier:
    def __init__(self, prot_df, cat_df, gene_dict, between=None):
        # Ensure unique columns in the proteomics DataFrame
        self.prot_df = prot_df.loc[:, ~prot_df.columns.duplicated()]
        self.cat_df = cat_df
        self.gene_dict = gene_dict
        self.figures = []  # To store the figures for saving to a PDF
        self.models = []  # To store the trained models
        self.selectors = []  # To store feature selectors
        self.between = between  

    def classify_and_plot(self, category1, category2, n_runs=10, n_estimators=200):
        if self.between is None:
            raise ValueError("The 'between' attribute must be set to a valid column name.")
        # Merge the dataframes on the index
        d_ML = self.prot_df.join(self.cat_df, how='inner')

        # Handle missing values using KNN Imputer
        imputer = KNNImputer(n_neighbors=5)
        X = imputer.fit_transform(d_ML.drop(columns=[self.between]))

        # Separate features and target
        y = d_ML[self.between]

        # Filter data to include only the specified categories
        filtered_indices = y.isin([category1, category2])
        X_filtered = X[filtered_indices]
        y_filtered = y[filtered_indices]

        # Encode the target variable
        y_filtered = y_filtered.map({category1: 0, category2: 1})

        mean_fpr = np.linspace(0, 1, 100)
        all_tprs = []
        aucs = []

        for i in range(n_runs):
            # Split into training and test sets
            X_train, X_test, y_train, y_test = train_test_split(
                X_filtered, y_filtered, test_size=0.2, stratify=y_filtered, random_state=i
            )

            # Feature selection using RandomForest for initial feature selection
            selector = SelectFromModel(RandomForestClassifier(n_estimators=n_estimators, random_state=i))
            selector.fit(X_train, y_train)
            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)

            # Initialize the XGBoost model
            model = XGBClassifier(eval_metric='logloss', random_state=i)

            # Stratified K-Fold cross-validation
            cv = StratifiedKFold(n_splits=5)
            tprs = []

            for train_idx, val_idx in cv.split(X_train_selected, y_train):
                X_cv_train, X_cv_val = X_train_selected[train_idx], X_train_selected[val_idx]
                y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

                # Train and predict
                probas_ = model.fit(X_cv_train, y_cv_train).predict_proba(X_cv_val)
                fpr, tpr, _ = roc_curve(y_cv_val, probas_[:, 1])
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0

            # Store model and selector
            self.models.append(model)
            self.selectors.append(selector)

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

    def validate_and_plot(self, validation_prot_df, validation_cat_df, category1, category2):
        if not self.models or not self.selectors:
            raise ValueError("No trained models available. Run 'classify_and_plot' first.")

        # Ensure unique columns in validation data
        validation_prot_df = validation_prot_df.loc[:, ~validation_prot_df.columns.duplicated()]

        # Align validation features to match the training features
        training_features = self.prot_df.columns
        validation_prot_df = validation_prot_df.reindex(columns=training_features, fill_value=0)

        # Merge proteomics and metadata dataframes
        validation_data = validation_prot_df.join(validation_cat_df, how='inner')

        # Handle missing values
        imputer = KNNImputer(n_neighbors=5)
        X_validation = imputer.fit_transform(validation_data.drop(columns=[self.between]))
        y_validation = validation_data[self.between]
        y_validation = y_validation.map({category1: 0, category2: 1})

        # Ensure alignment of X_validation and y_validation
        mask = ~np.isnan(y_validation)
        X_validation = X_validation[mask]
        y_validation = y_validation[mask]

        mean_fpr = np.linspace(0, 1, 100)
        all_tprs = []
        aucs = []

        for model, selector in zip(self.models, self.selectors):
            # Select features using the corresponding selector
            X_validation_selected = selector.transform(X_validation)

            # Predict probabilities
            y_pred_prob = model.predict_proba(X_validation_selected)[:, 1]

            # Compute ROC curve and AUC
            fpr, tpr, _ = roc_curve(y_validation, y_pred_prob)
            all_tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(roc_auc_score(y_validation, y_pred_prob))

        # Aggregate TPRs for plotting
        all_tprs = np.array(all_tprs)
        median_tpr = np.median(all_tprs, axis=0)
        lower_tpr = np.percentile(all_tprs, 2.5, axis=0)
        upper_tpr = np.percentile(all_tprs, 97.5, axis=0)

        # Plot aggregated ROC curve
        plt.figure()
        plt.plot(mean_fpr, median_tpr, color='green', label=f'Median Validation ROC (AUC = {np.mean(aucs):.2f})', lw=2)
        plt.fill_between(mean_fpr, lower_tpr, upper_tpr, color='grey', alpha=0.3, label='95% CI')
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='navy', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Validation ROC Curve')
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


