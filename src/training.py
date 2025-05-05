import pandas as pd
import matplotlib.pyplot as plt
import pickle
import yaml
import os
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score, f1_score,precision_score, roc_auc_score,roc_curve,precision_recall_curve
from sklearn.model_selection import train_test_split, GridSearchCV
import mlflow
from mlflow.models import infer_signature
from urllib.parse import urlparse
from dotenv import load_dotenv
import tempfile
from helpers import make_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
warnings.filterwarnings("ignore")
load_dotenv()

# Load parameters from params.yaml
params = yaml.safe_load(open("params.yaml"))['train']

# Train basic model first
def train_rf(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns = ["ProdTaken"])
    y = data['ProdTaken']

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

    # Start mlflow run
    with mlflow.start_run(run_name='BaseRandomForest'):
        X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size = 0.2, random_state=params['random_state'])
        signature = infer_signature(X_train, y_train)

        # Define model and train
        rf_model = RandomForestClassifier(random_state = 0)
        rf_model.fit(X_train, y_train)

        # Predict and evaluate the model
        y_pred = rf_model.predict(X_test)
        y_scores = rf_model.predict_proba(X_test)[:, 1]

        # Compute performace metrics
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_scores)

        # Log Metrics in mlflow
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("F1score", f1)
        mlflow.log_metric("AUC", auc)

        # Log model parameters
        parameters = rf_model.get_params()
        mlflow.log_param("n_estimators", parameters['n_estimators'])
        mlflow.log_param("max_depth", parameters['max_depth'])
        mlflow.log_param("min_samples_split", parameters['min_samples_split'])
        mlflow.log_param("min_samples_leaf", parameters['min_samples_leaf'])
        mlflow.log_param("criterion", parameters['criterion'])
        
        # Create temporary directory for plot files
        with tempfile.TemporaryDirectory() as tmpdir:
            
            # Generate your plots
            fig_cm = make_confusion_matrix(rf_model, X_test, y_test)
            fig_roc = plot_roc_curve(rf_model, X_test, y_test)
            fig_pr = plot_precision_recall_curve(rf_model, X_test, y_test)
            report = classification_report(y_test, y_pred)

            # Define file paths
            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            roc_path = os.path.join(tmpdir, "roc_curve.png")
            pr_path = os.path.join(tmpdir, "precision_recall_curve.png")
            report_path = os.path.join(tmpdir, "report.txt")

            # Save plots to files
            fig_cm.savefig(cm_path)
            fig_roc.savefig(roc_path)
            fig_pr.savefig(pr_path)

            # Save the classification report to the file
            with open(report_path, "w") as f:
                f.write(report)

            # Log them as artifacts to MLflow
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(roc_path)
            mlflow.log_artifact(pr_path)
            mlflow.log_artifact(report_path)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(rf_model, 
                                     artifact_path = "model", 
                                     signature=signature,
                                     input_example=X_train.iloc[:5], 
                                     registered_model_name='Base Random Forest')
        else:
            mlflow.sklearn.log_model(rf_model, "model", signature=signature)

        # Create a directory to save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        filename = model_path
        pickle.dump(rf_model, open(filename, 'wb'))
        print(f"Model saved to {model_path}.")


# Function to perform hyperparameter tuning
def hyperparameter_tuning(X_train, y_train, param_grid):

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv = 5, n_jobs= -1, verbose=1)
    grid_search.fit(X_train, y_train)

    return grid_search


# Train model with hyperparameter tuning
def train_tuned(data_path, model_path, random_state):
    data = pd.read_csv(data_path)
    X = data.drop(columns = ["ProdTaken"])
    y = data['ProdTaken']

    # Start mlflow run
    with mlflow.start_run(run_name='TunedRandomForest'):
        X_train, X_test, y_train,  y_test = train_test_split(X, y, test_size = 0.2, random_state=random_state)
        signature = infer_signature(X_train, y_train)

        param_grid = {
            'n_estimators' : [100, 150, 200],
            'max_depth' : [5, 8, 10],
            'min_samples_split' : [3, 5, 7],
            'min_samples_leaf' : [1, 3, 5],
            'criterion': ['gini', 'entropy', 'log_loss']
                    }

        # Perform hyperparameter tuning
        grid_search = hyperparameter_tuning(X_train, y_train, param_grid)

        # Get best model
        best_model = grid_search.best_estimator_

        # Predict and evaluate the model
        y_pred = best_model.predict(X_test)
        y_scores = best_model.predict_proba(X_test)[:, 1]

        # Compute performace metrics
        accuracy = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_scores)

        # Log Metrics in mlflow
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("F1score", f1)
        mlflow.log_metric("AUC", auc)

        # Log model parameters
        mlflow.log_param("best_n_estimators", grid_search.best_params_['n_estimators'])
        mlflow.log_param("best_max_depth", grid_search.best_params_['max_depth'])
        mlflow.log_param("best_min_samples_split", grid_search.best_params_['min_samples_split'])
        mlflow.log_param("best_min_samples_leaf", grid_search.best_params_['min_samples_leaf'])
        
        # Create temporary directory for plot files
        with tempfile.TemporaryDirectory() as tmpdir:
            
            # Generate your plots
            fig_cm = make_confusion_matrix(best_model, X_test, y_test)
            fig_roc = plot_roc_curve(best_model, X_test, y_test)
            fig_pr = plot_precision_recall_curve(best_model, X_test, y_test)
            report = classification_report(y_test, y_pred)

            # Define file paths
            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            roc_path = os.path.join(tmpdir, "roc_curve.png")
            pr_path = os.path.join(tmpdir, "precision_recall_curve.png")
            report_path = os.path.join(tmpdir, "report.txt")

            # Save plots to files
            fig_cm.savefig(cm_path)
            fig_roc.savefig(roc_path)
            fig_pr.savefig(pr_path)

            # Save the classification report to the file
            with open(report_path, "w") as f:
                f.write(report)

            # Log them as artifacts to MLflow
            mlflow.log_artifact(cm_path)
            mlflow.log_artifact(roc_path)
            mlflow.log_artifact(pr_path)
            mlflow.log_artifact(report_path)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != 'file':
            mlflow.sklearn.log_model(best_model,
                                     artifact_path = "model", 
                                     signature=signature,
                                     input_example=X_train.iloc[:5], 
                                     registered_model_name='Tuned Random Forest')
        else:
            mlflow.sklearn.log_model(best_model, "model", signature=signature)

        # Create a directory to save the model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        filename = model_path
        pickle.dump(best_model, open(filename, 'wb'))
        print(f"Model saved to {model_path}.")

if __name__ == '__main__':
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    params = yaml.safe_load(open("params.yaml"))['train']
    train_rf(params['data'], params['models']['base_model'])
    train_tuned(params['data'], params['models']['tuned_model'], params['random_state'])



