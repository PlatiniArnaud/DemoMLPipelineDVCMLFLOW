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

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns = ["ProdTaken"])
    y = data['ProdTaken']

    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))

    # Load model from the disk
    model = pickle.load(open(model_path, 'rb'))

    pred = model.predict(X)
    pred_scores = model.predict_proba(X)[:, 1]

    # Compute performace metrics
    accuracy = accuracy_score(y, pred)
    recall = recall_score(y, pred)
    precision = precision_score(y, pred)
    f1 = f1_score(y, pred)
    auc = roc_auc_score(y, pred_scores)

    # Log Metrics in mlflow
    mlflow.log_metric("Accuracy", accuracy)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("F1score", f1)
    mlflow.log_metric("AUC", auc)

if __name__=="__main__":

    # Load parameters from params.yaml
    models = yaml.safe_load(open("params.yaml"))['train']['models']
    test = yaml.safe_load(open("params.yaml"))['evaluate']

    evaluate(test['data'], models['base_model'])
    evaluate(test['data'], models['tuned_model'])