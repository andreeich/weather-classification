from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import pandas as pd
import yaml
import logging
from utils import load_csv
from report import generate_report

logging.basicConfig(level=logging.INFO)

def preprocess_data(data):
    """Preprocess the data by scaling features and binarizing the target variable."""
    X = data.drop(columns=['rain', 'date'])  # Exclude 'date' column
    y = (data['rain'] > 0).astype(int)  # Binarize the 'rain' column
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def train_model(X_train, y_train, max_iter, C, solver):
    """Train the logistic regression model with specified parameters."""
    model = LogisticRegression(max_iter=max_iter, C=C, solver=solver)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return accuracy and classification report."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)
    return accuracy, report

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

    parameters = config['model']['parameters']
    results = []

    for params in parameters:
        model = train_model(X_train, y_train, **params)
        accuracy, report = evaluate_model(model, X_test, y_test)
        results.append({
            'max_iter': params['max_iter'],
            'C': params['C'],
            'solver': params['solver'],
            'accuracy': accuracy,
            'y_true': y_test,
            'y_pred': model.predict(X_test),
            'params': params
        })

    # Generate a detailed report
    generate_report(results, config['report']['file_path'])

if __name__ == "__main__":
    main()