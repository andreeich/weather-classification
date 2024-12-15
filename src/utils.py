def normalize_data(data):
    return (data - data.mean()) / data.std()

def calculate_accuracy(y_true, y_pred):
    return (y_true == y_pred).mean()

def save_to_csv(data, file_path):
    data.to_csv(file_path, index=False)

def load_csv(file_path):
    import pandas as pd
    return pd.read_csv(file_path)