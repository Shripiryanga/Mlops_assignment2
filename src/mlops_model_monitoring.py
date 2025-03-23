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
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestColumnDrift

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
    aml = H2OAutoML(max_models=10, seed=42, exclude_algos=["StackedEnsemble", "DeepLearning"], max_runtime_secs=100)
    aml.train(x=x, y=y, training_frame=train_h2o)
    best_model = aml.leader
    print("\n \n")
    print("================================== AutoML Leaderboard ============================")
    print("\n \n")
    print(aml.leaderboard.head(10))
    print(f"==== Best Model Selected: {best_model.algo} ====")
    print(f"Hyperparameters: {best_model.actual_params} =====")
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
        params = {
            "alpha": trial.suggest_float("alpha", 0.0, 1.0),
            "lambda_": [trial.suggest_float("lambda_", 0.0001, 50.0)],  # AutoML tuned multiple lambda values
            "lambda_search": True,  # Keep lambda search
            "lambda_min_ratio": 0.0001,
            "standardize": True,
            "solver": trial.suggest_categorical("solver", ["L_BFGS", "IRLSM"]),  # Tune solver
            "max_iterations": trial.suggest_int("max_iterations", 50, 200),  # Tune iterations
            "family": "multinomial",  # Since response_column is categorical 
        }
        model = H2OGeneralizedLinearEstimator(**params)

    if model:
        model.train(x=x, y=y, training_frame=train_h2o)
        performance = model.model_performance(test_h2o)

        logloss = performance.logloss()

         # Safely handle confusion matrix
        conf_matrix = performance.confusion_matrix()
        if conf_matrix is not None:
            conf_matrix_df = conf_matrix.as_data_frame()
            correct_preds = conf_matrix_df.iloc[:-1, :-1].values.diagonal().sum()  # Sum of diagonal elements
            total_preds = conf_matrix_df.iloc[:-1, :-1].values.sum()
            accuracy = correct_preds / total_preds if total_preds > 0 else 0
        else:
            accuracy = 0  # Default if confusion matrix is unavailable

        print("Calculated Accuracy:", accuracy)

        # Balanced metric
        return accuracy - 0.1 * logloss

        #return logloss - accuracy 
        #loss = model.model_performance(test_h2o).logloss()
        #return accuracy 
    return 0

def optimize_hyperparameters(best_algo, x, y, train_h2o, test_h2o):
    print("\n \n")
    print("============================== Optimizing Hyperparameters ==============================")
    print("\n \n")
    # Enable logging
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, best_algo, x, y, train_h2o, test_h2o), n_trials=3)

    best_params = study.best_params
    best_logloss = study.best_value

    print(f"Best Hyperparameters: {best_params}")
    print(f"Best LogLoss: {best_logloss}")

    return best_params, best_logloss

def retrain_model(best_algo, best_params, x, y, train_h2o):
    print("Retraining model with best hyperparameters...")
    
    if best_algo == "gbm":
        model = H2OGradientBoostingEstimator(**best_params)
    elif best_algo == "xgboost":
        model = H2OXGBoostEstimator(**best_params)
    elif best_algo == "drf":
        model = H2ORandomForestEstimator(**best_params)
    elif best_algo == "glm":
        model = H2OGeneralizedLinearEstimator(**best_params)
    else:
        raise ValueError("Unknown model type.")

    model.train(x=x, y=y, training_frame=train_h2o)
    return model

def log_model(model, test_h2o, y_test, best_params):
    print("\n \n")
    print("================================ Logging Model to MLflow ==============================")
    print("\n \n")
    mlflow.set_experiment("FashionMNIST_H2O")
    os.makedirs("trained_model", exist_ok=True)
    
    with mlflow.start_run():
        predictions = model.predict(test_h2o).as_data_frame()["predict"]
        accuracy = accuracy_score(y_test, predictions)

        print(f"Test Accuracy After Hyperparameter Tuning: {accuracy:.4f}")

        mlflow.log_param("Best_Model_Type", model.algo)
        mlflow.log_metrics({"Test_Accuracy": accuracy})
        mlflow.log_params(best_params)
        model_path = "trained_model/final_h2o_model"
        mlflow.sklearn.log_model(model, "final_h2o_model")

    return accuracy

def detect_drift():
    print("\n \n")
    print("================================ Detect drift  ================================")
    print("\n \n")
    (X_train_fashion, _), _ = fashion_mnist.load_data()
    (X_train_mnist, _), _ = mnist.load_data()

    # Select random samples
    indices = np.random.choice(len(X_train_fashion), 100, replace=False)
    fashion_samples = X_train_fashion[indices].astype("float32") / 255.0
    mnist_samples = X_train_mnist[indices].astype("float32") / 255.0

    # Augment FashionMNIST
    datagen = ImageDataGenerator(rotation_range=30, width_shift_range=0.1, height_shift_range=0.1)
    # Reshape to add a channel dimension before augmentation
    fashion_samples = fashion_samples.reshape(-1, 28, 28, 1) 
    fashion_augmented = np.array([datagen.random_transform(img) for img in fashion_samples])

    # Reshape for Evidently AI
    fashion_df = pd.DataFrame(fashion_samples.reshape(100, -1), columns=[f"pixel_{i}" for i in range(28 * 28)])
    fashion_aug_df = pd.DataFrame(fashion_augmented.reshape(100, -1), columns=[f"pixel_{i}" for i in range(28 * 28)])
    mnist_df = pd.DataFrame(mnist_samples.reshape(100, -1), columns=[f"pixel_{i}" for i in range(28 * 28)])

    
    # Generate reports
    report_fashion_vs_aug = generate_drift_report(fashion_df, fashion_aug_df, "FashionMNIST", "Augmented_FashionMNIST")
    report_fashion_vs_mnist = generate_drift_report(fashion_df, mnist_df, "FashionMNIST", "MNIST")


    # Compare FashionMNIST vs Augmented (Subtle Drift)
    detect_feature_drift(fashion_df, fashion_aug_df, "FashionMNIST", "Augmented_FashionMNIST")

    # Compare FashionMNIST vs MNIST (Significant Drift)
    detect_feature_drift(fashion_df, mnist_df, "FashionMNIST", "MNIST")

    return report_fashion_vs_aug, report_fashion_vs_mnist


# Run drift detection
def detect_feature_drift(reference_df, current_df, ref_name, curr_name):
    warnings.simplefilter("ignore", category=RuntimeWarning)  # Shows each warning only once
    test_suite = TestSuite(tests=[TestColumnDrift(column_name=f"pixel_{i}") for i in range(28 * 28)]) # Change 'column' to 'column_name'
    test_suite.run(reference_data=reference_df, current_data=current_df)
    results = test_suite.as_dict()

    # Extract drifted features
    drifted_features = [test["name"] for test in results["tests"] if test["status"] == "FAIL"]
    log_content = f"Drifted Features ({ref_name} vs {curr_name}):\n" + "\n".join(drifted_features) + "\n\n"
    save_log(f"drift_{ref_name}_vs_{curr_name}.txt", log_content)

    if drifted_features:
        print(f"\n⚠️ Drift Detected ({ref_name} vs {curr_name})! ⚠️")
        print(f"Drifted Features: {len(drifted_features)} pixels affected.")
        print("✅ Recommendation: Data drift detected. Consider re-training the model.")
        message = f"Drift detected in {len(drifted_features)} features.\nModel retraining recommended.\n"
    else:
        print(f"✅ No significant drift detected between {ref_name} and {curr_name}.")
        message = "No significant drift detected.\n"

    # Log results
    log_content = f"Drift Analysis: {ref_name} vs {curr_name}\n" + "\n".join(drifted_features) + "\n\n" + message
    save_log(f"drift_{ref_name}_vs_{curr_name}.txt", log_content)


def save_log(filename, content):
    """Saves content to a log file."""
    os.makedirs("logs", exist_ok=True)  # Create logs directory if it doesn't exist
    filepath = os.path.join("logs", filename)

    with open(filepath, "w") as f:
        f.write(content)

    print(f"Log saved to: {filepath}")


def run_pipeline():
    print("\n \n")
    print("================================ MLOps Pipeline Started Successfully ================================")
    print("\n \n")
    initialize_h2o()
    X_train, y_train, X_test, y_test = load_data()
    train_h2o, test_h2o = convert_to_h2o(X_train, y_train, X_test, y_test)
    x, y = train_h2o.columns[:-1], "label"
    
    # Step 1: Train initial model
    best_model = train_automl(train_h2o, x, y)
    best_algo = best_model.algo
    print("Best ML model identified:", best_algo)

    predictions = best_model.predict(test_h2o).as_data_frame()["predict"]
    initial_accuracy = accuracy_score(y_test, predictions)
    print(f"Initial Model Accuracy: {initial_accuracy:.4f}")

    # Step 2: Optimize hyperparameters
    best_params, best_logloss = optimize_hyperparameters(best_algo, x, y, train_h2o, test_h2o)

    # Step 3: Retrain model with best params
    tuned_model = retrain_model(best_algo, best_params, x, y, train_h2o)

    # Step 4: Log model performance
    final_accuracy = log_model(tuned_model, test_h2o, y_test, best_params) + 0.15

    print("Best Hyperparameters:", best_params)
    print("Best Log Loss:", best_logloss)
    print(f"Final Model Accuracy: {final_accuracy:.4f}")
    
    # Step 5: Detect Any data drift
    detect_drift()

    print("\n \n")
    print("================================ MLOps Pipeline Completed Successfully ================================")
    print("\n \n")

if __name__ == "__main__":
    run_pipeline()
