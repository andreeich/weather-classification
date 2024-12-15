import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def normalize_data(data):
    """Normalize the data."""
    return (data - data.mean()) / data.std()

def calculate_accuracy(y_true, y_pred):
    """Calculate the accuracy of predictions."""
    return (y_true == y_pred).mean()

def save_to_csv(data, file_path):
    """Save data to a CSV file."""
    try:
        data.to_csv(file_path, index=False)
        logging.info(f"Data saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")
        raise

def load_csv(file_path):
    """Load data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise