# Customer Churn Prediction

### A deep learning project for spotting telecom customers at risk of leaving

This project explores whether a customer is likely to churn using account, service, and billing information from the IBM Telco Customer Churn dataset. It follows the full modeling workflow—from data exploration and preparation to neural network training and evaluation—and compares a small multi-layer perceptron built with NumPy against Keras models with different hidden-layer activations.

<p align="center">
  <strong>Dataset:</strong> 7,043 customer records &nbsp;·&nbsp;
  <strong>Task:</strong> Binary classification &nbsp;·&nbsp;
  <strong>Best reported ROC-AUC:</strong> 0.838
</p>

## Project files

| File | What you'll find |
| --- | --- |
| [`Untitled7 (1).ipynb`](Untitled7%20%281%29.ipynb) | Exploratory analysis, preprocessing, NumPy MLP, Keras experiments, and evaluation plots |
| [`Customer_Churn_Prediction_Project_Report.docx`](Customer_Churn_Prediction_Project_Report.docx) | Written project report |
| [`Customer_Churn_Prediction_Presentation.pptx`](Customer_Churn_Prediction_Presentation.pptx) | Project presentation and results |

## What the project does

1. Explores the customer and churn distributions, numerical features, and categorical features.
2. Removes the customer identifier and converts blank `TotalCharges` values to zero.
3. Encodes categorical features, separates the target, and creates a stratified train/test split.
4. Scales the features and applies SMOTE to the training split to address class imbalance.
5. Trains a feed-forward neural network implemented from scratch with NumPy.
6. Trains Keras MLP variants using ReLU, Tanh, and Sigmoid hidden-layer activations.
7. Compares accuracy, precision, recall, ROC-AUC, and training behavior.

## Reported results

The project presentation reports the Sigmoid Keras model as the strongest of the tested activation variants:

| Metric | Reported result |
| --- | ---: |
| Accuracy | 77.4% |
| Precision | 56.1% |
| Recall | 67.6% |
| ROC-AUC | 0.838 |

These are the values reported in the accompanying presentation; run the notebook to reproduce the experiments in your own environment. Results can vary with library versions and runtime settings.

## Getting started

### Requirements

The notebook uses Python and the following packages:

- NumPy and pandas
- Matplotlib and Seaborn
- scikit-learn
- imbalanced-learn
- TensorFlow
- XGBoost

Install the packages in a Python environment with:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn tensorflow xgboost jupyter
```

### Dataset

The notebook expects the Telco Customer Churn CSV named `WA_Fn-UseC_-Telco-Customer-Churn.csv`. The dataset is not included in this repository. Obtain it from the IBM Telco Customer Churn dataset source, then update the CSV path in the notebook's data-loading cell. The current path, `/content/WA_Fn-UseC_-Telco-Customer-Churn.csv`, is set up for Google Colab.

### Run the notebook

```bash
jupyter notebook "Untitled7 (1).ipynb"
```

In Google Colab, upload or mount the dataset at the path used by the notebook, then run the cells from top to bottom. For local use, change the `pd.read_csv(...)` path to the location of your CSV file.

## Notes on the workflow

- The test set is stratified and SMOTE is applied to the training data after the split.
- The notebook saves categorical encoders as `encoder.pkl` when that cell runs.
- The project is an educational modeling exercise. Its results are not a guarantee of performance on other customer populations or future data.

## Credits

Created as a Semester 5 B.Tech Computer Science and Engineering Deep Learning project at Sri Sri University. The project materials credit Sritam Mishra, Binayak Bhoi, Dhruv Dash, and Sarthak Kumar Patel, with guidance from Mr. Yugnash.
