import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

logging.basicConfig(level=logging.INFO)

def generate_report(results, file_path='detailed_report.csv'):
    """Generate a detailed report of model performance metrics."""
    report_data = []

    for result in results:
        y_true = result['y_true']
        y_pred = result['y_pred']
        params = result['params']

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')

        report_data.append({
            'max_iter': params['max_iter'],
            'C': params['C'],
            'solver': params['solver'],
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })

    report_df = pd.DataFrame(report_data)
    report_df.to_csv(file_path, index=False)
    logging.info(f"Detailed report generated and saved as {file_path}")