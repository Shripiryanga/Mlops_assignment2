import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import fashion_mnist
import sweetviz as sv
import cv2
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import shap
import os
import logging
from sklearn.decomposition import PCA
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_fashion_mnist():
    logging.info("Loading Fashion MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    return x_train, y_train, x_test, y_test

def plot_sample_images(x_train, y_train, fashion_labels):
    logging.info("Plotting sample images from the dataset...")
    plt.figure(figsize=(12, 6))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(x_train[i], cmap="gray")
        plt.title(fashion_labels[y_train[i]])
        plt.axis("off")
    plt.show(block=True)

def create_dataframe(x_train):
    logging.info("Converting images to a flattened DataFrame...")
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    df_train = pd.DataFrame(x_train_flat).sample(n=5)
    return df_train

def generate_eda_report(df_train):
    logging.info("Generating EDA report using Sweetviz...")
    report = sv.analyze(df_train, pairwise_analysis="off")
    report.show_html("fashion_mnist_sweetviz.html")

def plot_class_distribution(y_train, fashion_labels):
    logging.info("Plotting class distribution...")
    unique, counts = np.unique(y_train, return_counts=True)
    plt.figure(figsize=(8, 4))
    sns.barplot(x=unique, y=counts, palette="viridis")
    plt.xticks(ticks=unique, labels=fashion_labels, rotation=45)
    plt.xlabel("Class Labels")
    plt.ylabel("Count")
    plt.title("Fashion MNIST Class Distribution")
    plt.show()

def analyze_missing_values(df_train):
    missing_values = df_train.isnull().sum().sum()
    logging.info(f"Total missing values: {missing_values}")

def compute_feature_correlation(df_train):
    logging.info("Computing feature correlation...")
    correlation_matrix = df_train.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.show()

def extract_hog_features(images):
    logging.info("Extracting HOG features...")
    hog_features = []
    for img in images:
        resized_img = cv2.resize(img, (64, 64))
        features, _ = hog(
            resized_img,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            visualize=True
        )
        hog_features.append(features)
    return np.array(hog_features)

def train_svm_classifier(X_train, y_train):
    logging.info("Training the SVM classifier...")
    svm_model = SVC(kernel='linear', C=1.0, random_state=42)
    svm_model.fit(X_train, y_train)
    return svm_model

def explain_with_shap(svm_model, X_train):
    logging.info("Explaining predictions using SHAP...")
    explainer = shap.KernelExplainer(svm_model.predict, X_train[:20])
    shap_values = explainer.shap_values(X_train[:20])
    return shap_values

def plot_class_mean_images(x_train, y_train, fashion_labels):
    logging.info("Computing class-wise pixel correlation...")

    # Compute mean pixel intensity per class
    mean_images = np.zeros((10, 28, 28))
    for label in range(10):
        mean_images[label] = np.mean(x_train[y_train == label], axis=0)

    # Flatten and compute correlation
    mean_images_flat = mean_images.reshape(10, -1)
    correlation_matrix = np.corrcoef(mean_images_flat)

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f",
                xticklabels=fashion_labels, yticklabels=fashion_labels)
    plt.title("Class-wise Pixel Correlation Heatmap")
    plt.show()

def plot_classwise_pixel_correlation(x_train, y_train, fashion_labels):
    logging.info("Computing class-wise pixel correlation...")

    # Compute mean pixel intensity per class
    mean_images = np.zeros((10, 28, 28))
    for label in range(10):
        mean_images[label] = np.mean(x_train[y_train == label], axis=0)

    # Flatten and compute correlation
    mean_images_flat = mean_images.reshape(10, -1)
    correlation_matrix = np.corrcoef(mean_images_flat)

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f",
                xticklabels=fashion_labels, yticklabels=fashion_labels)
    plt.title("Class-wise Pixel Correlation Heatmap")
    plt.show()



def main():
    # Step 1: Load Data
    x_train, y_train, x_test, y_test = load_fashion_mnist()
    fashion_labels = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    
    # Step 2: Exploratory Data Analysis (EDA)
    logging.info("Starting EDA...")
    # Call the function inside main()

    plot_classwise_pixel_correlation(x_train, y_train, fashion_labels)
    plot_sample_images(x_train, y_train, fashion_labels)
    df_train = create_dataframe(x_train)
    generate_eda_report(df_train)
    plot_class_distribution(y_train, fashion_labels)
    analyze_missing_values(df_train)
    plot_class_mean_images(x_train, y_train, fashion_labels)
    compute_feature_correlation(df_train)


    
    # Step 3: Feature Engineering & Explainability
    logging.info("Starting Feature Engineering and Explainability...")
    n_samples = 1000
    hog_features = extract_hog_features(x_train[:n_samples])
    X_train, X_val, y_train_split, y_val = train_test_split(hog_features, y_train[:n_samples], test_size=0.2, random_state=42)
    svm_model = train_svm_classifier(X_train, y_train_split)
    
    # Step 4: Model Evaluation
    logging.info("Validating the model...")
    y_val_pred = svm_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    logging.info(f"Validation Accuracy: {val_accuracy:.2f}")
    
    # Step 5: Explainability using SHAP
    shap_values = explain_with_shap(svm_model, X_train)
    shap.summary_plot(shap_values, X_train[:20], show=False)
    plt.savefig("shap_summary.png", dpi=300)
    plt.close()
    logging.info("SHAP summary plot saved.")

if __name__ == "__main__":
    main()
