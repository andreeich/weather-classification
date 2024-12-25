from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import yaml
import logging
from utils import load_csv
from custom_logistic_regression import CustomLogisticRegression

logging.basicConfig(level=logging.INFO)

def preprocess_data(data):
    """Preprocess the data by scaling features and binarizing the target variable."""
    X = data.drop(columns=['rain', 'date'])  # Exclude 'date' column
    y = (data['rain'] > 0).astype(int)  # Binarize the 'rain' column
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def generate_report(results, file_path='detailed_report.csv'):
    """Generate a detailed report of model performance metrics."""
    report_data = []

    for result in results:
        y_true = result['y_true']
        y_pred = result['y_pred']
        learning_rate = result.get('learning_rate', 'N/A')
        max_iter = result.get('max_iter', 'N/A')

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        report_data.append({
            'learning_rate': learning_rate,
            'max_iter': max_iter,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

    report_df = pd.DataFrame(report_data)
    report_df.to_csv(file_path, index=False)
    logging.info(f"Detailed report generated and saved as {file_path}")

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

    results = []

    # Iterate over parameter combinations
    for params in config['model']['parameters']:
        if 'learning_rate' in params and 'max_iter' in params:
            model = CustomLogisticRegression(learning_rate=params['learning_rate'], max_iter=params['max_iter'])
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)

            logging.info(f"Accuracy: {accuracy}")
            logging.info("Classification Report:")
            logging.info(report)

            results.append({
                'learning_rate': params['learning_rate'],
                'max_iter': params['max_iter'],
                'accuracy': accuracy,
                'y_true': y_test,
                'y_pred': y_pred
            })

    # Generate a detailed report
    generate_report(results, config['report']['file_path'])

if __name__ == "__main__":
    main()