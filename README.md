# Credit Card Fraud Detection

This project aims to detect fraudulent credit card transactions using machine learning techniques. The dataset used for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Table of Contents

1. [Installation and Dependencies](#installation-and-dependencies)
2. [Usage Instructions](#usage-instructions)
3. [Project Structure](#project-structure)
4. [Dataset Information](#dataset-information)
5. [Methodology](#methodology)
6. [Results and Evaluation](#results-and-evaluation)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact Information](#contact-information)

## Installation and Dependencies

To run this project, you'll need the following dependencies:

- Python 3.x
- Jupyter Notebook
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install the required packages using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## Usage Instructions

To use this project, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/credit-card-fraud-detection.git
    ```
2. Navigate to the project directory:
    ```bash
    cd credit-card-fraud-detection
    ```
3. Open the Jupyter Notebook:
    ```bash
    jupyter notebook
    ```
4. Open the `Credit_card_Fraud_detection.ipynb` notebook and run the cells sequentially.

## Project Structure

The project directory contains the following files:

- `Credit_card_Fraud_detection.ipynb`: Jupyter Notebook containing the code for the project.
- `README.md`: This README file.

## Dataset Information

The dataset used in this project contains transactions made by credit cards in September 2013 by European cardholders. It presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, with the positive class (frauds) accounting for 0.172% of all transactions.

**Features:**
- The dataset contains only numerical input variables which are the result of a PCA transformation.
- Features V1, V2, ... V28 are the principal components obtained with PCA.
- The only features which have not been transformed with PCA are 'Time' and 'Amount'.
- Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset.
- Feature 'Amount' is the transaction Amount.
- Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.

## Methodology

The project follows these steps:

1. **Data Preprocessing:**
    - Load the dataset.
    - Perform exploratory data analysis (EDA).
    - Handle missing values and data imbalances.

2. **Feature Engineering:**
    - Create new features if necessary.
    - Normalize or scale the features.

3. **Model Training:**
    - Split the data into training and testing sets.
    - Train various machine learning models (e.g., Logistic Regression, Random Forest, XGBoost).
    - Perform hyperparameter tuning.

4. **Model Evaluation:**
    - Evaluate the models using appropriate metrics (e.g., precision, recall, F1-score, ROC-AUC).
    - Compare the performance of different models.

5. **Model Deployment:**
    - Save the best model for future use.

## Results and Evaluation

The best-performing model was the **Logistic Regression model**, which achieved the following metrics:

- Test Accuracy Score: 0.9478672985781991
- Training Accuracy Score: 0.9466192170818505
  
## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
