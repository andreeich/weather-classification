import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    """Load the cleaned weather dataset."""
    return pd.read_csv(file_path)

def split_data(data):
    """Split the dataset into training, validation, and test sets."""
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    validation_data, test_data = train_test_split(temp_data, test_size=0.3333, random_state=42)  # 0.3333 * 0.3 = 0.1

    return train_data, validation_data, test_data

def save_data(train_data, validation_data, test_data):
    """Save the datasets into separate CSV files."""
    train_data.to_csv('data/train.csv', index=False)
    validation_data.to_csv('data/validation.csv', index=False)
    test_data.to_csv('data/test.csv', index=False)

def main():
    data = load_data('data/cleaned_weather.csv')
    train_data, validation_data, test_data = split_data(data)
    save_data(train_data, validation_data, test_data)

if __name__ == "__main__":
    main()