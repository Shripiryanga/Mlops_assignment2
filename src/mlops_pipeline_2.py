import numpy as np
import pandas as pd
import h2o
import logging
import os
from h2o.automl import H2OAutoML
from tensorflow.keras.datasets import fashion_mnist, mnist
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score
import optuna
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.options.data_drift import DataDriftOptions
from h2o.estimators import (
    H2OGradientBoostingEstimator,
    H2OXGBoostEstimator,
    H2ORandomForestEstimator,
    H2OGeneralizedLinearEstimator,
)

# Setup logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "pipeline.log"),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logging.getLogger().addHandler(console_handler)

def initialize_h2o():
    logging.info("==== Initializing H2O Server ====")
    h2o.init(log_dir=LOG_DIR)

def load_data():
    logging.info("==== Loading Fashion MNIST Data ====")
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, y_train = X_train[:100].reshape(100, -1), y_train[:100]
    X_test, y_test = X_test[:100].reshape(100, -1), y_test[:100]
    return X_train, y_train, X_test, y_test

def convert_to_h2o(X_train, y_train, X_test, y_test):
    logging.info("==== Converting Data to H2O Frames ====")
    train_df = pd.DataFrame(X_train)
    train_df["label"] = y_train
    test_df = pd.DataFrame(X_test)
    test_df["label"] = y_test
    train_h2o = h2o.H2OFrame(train_df)
    test_h2o = h2o.H2OFrame(test_df)
    train_h2o["label"] = train_h2o["label"].asfactor()
    test_h2o["label"] = test_h2o["label"].asfactor()
    return train_h2o, test_h2o

def train_automl(train_h2o, x, y):
    logging.info("==== AutoML Training Started ====")
    aml = H2OAutoML(max_models=10, seed=42, exclude_algos=["StackedEnsemble", "DeepLearning"], max_runtime_secs=50)
    aml.train(x=x, y=y, training_frame=train_h2o)
    best_model = aml.leader
    logging.info(f"==== Best Model Selected: {best_model.algo} ====")
    return best_model

def log_model(best_model, test_h2o, y_test):
    logging.info("==== Logging Model to MLflow ====")
    mlflow.set_experiment("FashionMNIST_H2O")
    with mlflow.start_run():
        predictions = best_model.predict(test_h2o).as_data_frame()["predict"]
        accuracy = accuracy_score(y_test, predictions)
        logging.info(f"Test Accuracy Before Hyperparameter Tuning: {accuracy:.4f}")
        mlflow.log_param("Best_Model_Type", best_model.algo)
        mlflow.log_metric("Test_Accuracy", accuracy)
        mlflow.sklearn.log_model(best_model, "best_h2o_model")
    return best_model.algo

def objective(trial, best_algo, x, y, train_h2o, test_h2o):
    model = None
    params = {}
    if best_algo == "gbm":
        params = {
            "ntrees": trial.suggest_int("ntrees", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learn_rate": trial.suggest_float("learn_rate", 0.01, 0.3),
            "sample_rate": trial.suggest_float("sample_rate", 0.6, 1.0),
            "col_sample_rate": trial.suggest_float("col_sample_rate", 0.6, 1.0),
        }
        model = H2OGradientBoostingEstimator(**params)
    elif best_algo == "xgboost":
        params = {
            "ntrees": trial.suggest_int("ntrees", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learn_rate": trial.suggest_float("learn_rate", 0.01, 0.3),
            "col_sample_rate": trial.suggest_float("col_sample_rate", 0.6, 1.0),
            "min_rows": trial.suggest_int("min_rows", 1, 20),
        }
        model = H2OXGBoostEstimator(**params)
    elif best_algo == "drf":
        params = {
            "ntrees": trial.suggest_int("ntrees", 50, 500),
            "max_depth": trial.suggest_int("max_depth", 5, 20),
            "sample_rate": trial.suggest_float("sample_rate", 0.6, 1.0),
            "mtries": trial.suggest_int("mtries", 1, min(len(x), 50)),
        }
        model = H2ORandomForestEstimator(**params)
    elif best_algo == "glm":
        params = {"alpha": trial.suggest_float("alpha", 0.0, 1.0)}
        model = H2OGeneralizedLinearEstimator(**params)
    if model:
        model.train(x=x, y=y, training_frame=train_h2o)
        loss = model.model_performance(test_h2o).logloss()
        return loss
    return 0

def optimize_hyperparameters(best_algo, x, y, train_h2o, test_h2o):
    logging.info(f"==== Hyperparameter Optimization for {best_algo} ====")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, best_algo, x, y, train_h2o, test_h2o), n_trials=3)
    logging.info(f"Best Hyperparameters: {study.best_params}")
    logging.info(f"Best Log Loss: {study.best_value}")
    return study.best_params, study.best_value

def detect_data_drift():
    logging.info("==== Running Data Drift Detection ====")
    (X_train_mnist, _), _ = mnist.load_data()
    (X_train_fashion, _), _ = fashion_mnist.load_data()
    df_mnist = pd.DataFrame(X_train_mnist[:100].reshape(100, -1) / 255.0)
    df_fashion = pd.DataFrame(X_train_fashion[:100].reshape(100, -1) / 255.0)
    df_mnist["dataset"] = "MNIST"
    df_fashion["dataset"] = "Fashion-MNIST"
    report = Report(metrics=[DataDriftPreset(drift_share=0.5)])
    report.run(reference_data=df_mnist, current_data=df_fashion)
    report.save_html(os.path.join(LOG_DIR, "data_drift_report.html"))
    logging.info("==== Data Drift Report Generated ====")

def run_pipeline():
    initialize_h2o()
    X_train, y_train, X_test, y_test = load_data()
    train_h2o, test_h2o = convert_to_h2o(X_train, y_train, X_test, y_test)
    x, y = train_h2o.columns[:-1], "label"
    best_model = train_automl(train_h2o, x, y)
    best_algo = log_model(best_model, test_h2o, y_test)
    best_params, best_logloss = optimize_hyperparameters(best_algo, x, y, train_h2o, test_h2o)
    detect_data_drift()
    logging.info("==== MLOps Pipeline Completed Successfully ====")

if __name__ == "__main__":
    run_pipeline()
