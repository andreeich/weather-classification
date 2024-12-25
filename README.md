# Weather Classification Project

The aim of the work is to investigate and compare the effectiveness of machine learning methods for the classification of weather conditions, specifically rainfall prediction based on meteorological data. The project implements a custom logistic regression model, the scikit-learn library for gradient boosting, and the TensorFlow/Keras framework for building a neural network.

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
│   ├── logistic_regression.py   # Implementation of the logistic regression model
│   ├── gradient_boosting.py     # Implementation of the gradient boosting model
│   ├── neural_network.py        # Implementation of the neural network model
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

   **Note**: This project requires Python 3.9 - 3.12.

3. **Prepare the data**:
   Run the `data_preparation.py` script to load the cleaned dataset and split it into training, validation, and test sets:

   ```
   python src/data_preparation.py
   ```

4. **Train the models and generate the report**:
   Use the respective scripts to train the classification models and automatically generate a report of the model's performance:

   ```
   python src/logistic_regression.py
   python src/gradient_boosting.py
   python src/neural_network.py
   ```

## Results Interpretation

The generated report will include various metrics such as accuracy, precision, recall, and F1-score, which will help in evaluating the model's performance. The logistic regression model achieved the best results with an accuracy of 98.01%.

## License

This project is licensed under the MIT License.
