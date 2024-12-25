from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import yaml
import logging
from utils import load_csv
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense

logging.basicConfig(level=logging.INFO)

def preprocess_data(data):
    """Preprocess the data by scaling features and binarizing the target variable."""
    X = data.drop(columns=['rain', 'date'])  # Exclude 'date' column
    y = (data['rain'] > 0).astype(int)  # Binarize the 'rain' column
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y

def build_model(input_dim):
    """Build a neural network model."""
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def generate_report(results, file_path='detailed_report.csv'):
    """Generate a detailed report of model performance metrics."""
    report_data = []

    for result in results:
        y_true = result['y_true']
        y_pred = result['y_pred']
        model = result.get('model', 'N/A')

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        report_data.append({
            'Model': model,
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

    # Build and train the neural network model
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train, epochs=config['model']['parameters'][0]['epochs'], batch_size=config['model']['parameters'][0]['batch_size'], validation_split=0.2)

    # Evaluate the model
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    logging.info(f"Accuracy: {accuracy}")
    logging.info("Classification Report:")
    logging.info(report)

    # Prepare results for the report
    results = [{
        'model': 'Neural Network',
        'accuracy': accuracy,
        'y_true': y_test,
        'y_pred': y_pred
    }]

    # Generate a detailed report
    generate_report(results, config['report']['file_path'])

if __name__ == "__main__":
    main()