# Weather Classification Project

This project aims to classify weather conditions using a dataset of weather observations. The dataset includes various atmospheric and environmental parameters, and the project implements a linear classification or logistic regression model to predict weather conditions.

## Project Structure

```
weather-classification
├── data
│   ├── cleaned_weather.csv      # Original dataset with weather observations
│   ├── test.csv                 # Test dataset (10% of the original dataset)
│   ├── train.csv                # Training dataset (70% of the original dataset)
│   └── validation.csv           # Validation dataset (20% of the original dataset)
├── src
│   ├── data_preparation.py      # Functions for loading and splitting the dataset
│   ├── model.py                 # Implementation of the classification model
│   ├── report.py                # Generates a report of model performance metrics
│   └── utils.py                 # Utility functions for data processing
├── requirements.txt             # List of dependencies for the project
└── README.md                    # Documentation for the project
```

## Setup Instructions

1. **Clone the repository**:
   ```
   git clone <repository-url>
   cd weather-classification
   ```

2. **Install dependencies**:
   It is recommended to use a virtual environment. You can create one using `venv` or `conda`. After activating your environment, run:
   ```
   pip install -r requirements.txt
   ```

3. **Prepare the data**:
   Run the `data_preparation.py` script to load the cleaned dataset and split it into training, validation, and test sets:
   ```
   python src/data_preparation.py
   ```

4. **Train the model**:
   Use the `model.py` script to train the classification model:
   ```
   python src/model.py
   ```

5. **Generate the report**:
   After training, run the `report.py` script to generate a report of the model's performance:
   ```
   python src/report.py
   ```

## Results Interpretation

The generated report will include various metrics such as accuracy, precision, recall, and F1-score, which will help in evaluating the model's performance. 

## License

This project is licensed under the MIT License.