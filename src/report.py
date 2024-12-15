import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def generate_report(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred)

    report_df = pd.DataFrame({
        'Model': [model_name],
        'Accuracy': [accuracy],
        'Report': [report]
    })

    report_df.to_csv('model_report.csv', index=False)
    print("Report generated and saved as model_report.csv")