import numpy as np
import pandas as pd
import h2o
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
import os

def initialize_h2o():
    print("\n \n")
    print("============================= Initializing H2O Server ===========================")
    print("\n \n")
    h2o.init()

def load_data():
    print("======================== Loading Fashion MNIST Data ============================")
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train, y_train = X_train[:100].reshape(100, -1), y_train[:100]
    X_test, y_test = X_test[:100].reshape(100, -1), y_test[:100]
    return X_train, y_train, X_test, y_test

def convert_to_h2o(X_train, y_train, X_test, y_test):
    print("\n \n")
    print("============================== Converting Data to H2O Frames =========================")
    print("\n \n")
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
    print("\n \n")
    print("================================== AutoML Training Started ============================")
    print("\n \n")
    aml = H2OAutoML(max_models=10, seed=42, exclude_algos=["StackedEnsemble", "DeepLearning"], max_runtime_secs=50)
    aml.train(x=x, y=y, training_frame=train_h2o)
    best_model = aml.leader
    print(f"==== Best Model Selected: {best_model.algo} ====")
    return best_model

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
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, best_algo, x, y, train_h2o, test_h2o), n_trials=3)
    best_params_log = f"Best Params: {study.best_params}\nBest LogLoss: {study.best_value}"
    #save_log("optuna_study.txt", best_params_log)
    print("\n \n")
    print("================================ Logging Model to MLflow ==============================")
    print("\n \n")
    with mlflow.start_run():
        mlflow.log_params(study.best_params)
        mlflow.log_metric("Best_LogLoss", study.best_value)
    return study.best_params, study.best_value


def log_model(best_model, test_h2o, y_test):
    #print("================================ Logging Model to MLflow ==============================")
    mlflow.set_experiment("FashionMNIST_H2O")
    with mlflow.start_run():
        predictions = best_model.predict(test_h2o).as_data_frame()["predict"]
        accuracy = accuracy_score(y_test, predictions)
        print(f"Test Accuracy Before Hyperparameter Tuning: {accuracy:.4f}")
        mlflow.log_param("Best_Model_Type", best_model.algo)
        mlflow.log_metric("Test_Accuracy", accuracy)
        mlflow.sklearn.log_model(best_model, "best_h2o_model")
    return best_model.algo

def detect_data_drift():
    print("\n \n")
    print("================================ Running Data Drift Detection ================================")
    print("\n \n")
    (X_train_mnist, _), _ = mnist.load_data()
    (X_train_fashion, _), _ = fashion_mnist.load_data()
    df_mnist = pd.DataFrame(X_train_mnist[:100].reshape(100, -1) / 255.0)
    df_fashion = pd.DataFrame(X_train_fashion[:100].reshape(100, -1) / 255.0)
    df_mnist["dataset"] = "MNIST"
    df_fashion["dataset"] = "Fashion-MNIST"
    report = Report(metrics=[DataDriftPreset(drift_share=0.5)])
    report.run(reference_data=df_mnist, current_data=df_fashion)
    os.makedirs("reports", exist_ok=True)
    report.save_html("reports/data_drift_report.html")
    print("==== Data Drift Report Generated ====")

def run_pipeline():
    print("\n \n")
    print("================================ MLOps Pipeline Started Successfully ================================")
    print("\n \n")
    initialize_h2o()
    X_train, y_train, X_test, y_test = load_data()
    train_h2o, test_h2o = convert_to_h2o(X_train, y_train, X_test, y_test)
    x, y = train_h2o.columns[:-1], "label"
    best_model = train_automl(train_h2o, x, y)
    best_algo = log_model(best_model, test_h2o, y_test)
    print("Best ML model identified:", best_algo)
    best_params, best_logloss = optimize_hyperparameters(best_algo, x, y, train_h2o, test_h2o)
    print("Best Hyperparameters:", best_params)
    print("Best Log Loss:", best_logloss)
    #detect_data_drift()
    print("\n \n")
    print("================================ MLOps Pipeline Completed Successfully ================================")
    print("\n \n")

if __name__ == "__main__":
    run_pipeline()
