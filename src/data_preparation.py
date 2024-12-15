import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging
from utils import load_csv, save_to_csv  # Importing utility functions

logging.basicConfig(level=logging.INFO)

def load_data(file_path):
    """Load the cleaned weather dataset."""
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data from {file_path}: {e}")
        raise

def split_data(data):
    """Split the dataset into training, validation, and test sets."""
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    validation_data, test_data = train_test_split(temp_data, test_size=0.3333, random_state=42)  # 0.3333 * 0.3 = 0.1
    return train_data, validation_data, test_data

def save_data(train_data, validation_data, test_data, config):
    """Save the datasets into separate CSV files."""
    save_to_csv(train_data, config['data']['train_data'])
    save_to_csv(validation_data, config['data']['validation_data'])
    save_to_csv(test_data, config['data']['test_data'])
    logging.info("Data saved successfully")

def main():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    data = load_data(config['data']['cleaned_data'])
    train_data, validation_data, test_data = split_data(data)
    save_data(train_data, validation_data, test_data, config)

if __name__ == "__main__":
    main()