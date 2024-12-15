from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import yaml
import logging
from utils import load_csv
from custom_logistic_regression import CustomLogisticRegression
from report import generate_report

logging.basicConfig(level=logging.INFO)

def preprocess_data(data):
    """Preprocess the data by scaling features and binarizing the target variable."""
    X = data.drop(columns=['rain', 'date'])  # Exclude 'date' column
    y = (data['rain'] > 0).astype(int)  # Binarize the 'rain' column
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    train_data = load_csv(config['data']['train_data'])
    test_data = load_csv(config['data']['test_data'])

    X_train, y_train = preprocess_data(train_data)
    X_test, y_test = preprocess_data(test_data)

    # Check the distribution of the target variable
    logging.info("Training data class distribution:")
    logging.info(y_train.value_counts())

    # Train the custom logistic regression model
    model = CustomLogisticRegression(learning_rate=0.01, max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logging.info(f"Accuracy: {accuracy}")
    logging.info("Classification Report:")
    logging.info(report)

    # Prepare results for the report
    results = [{
        'learning_rate': 0.01,
        'max_iter': 1000,
        'accuracy': accuracy,
        'y_true': y_test,
        'y_pred': y_pred
    }]

    # Generate a detailed report
    generate_report(results, config['report']['file_path'])

if __name__ == "__main__":
    main()