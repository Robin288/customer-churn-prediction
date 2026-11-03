<div align="center">

# Customer Churn Prediction

### Finding the customers who may be ready to leave

An end-to-end deep learning study on telecom customer churn, with a neural network built from scratch in **NumPy** and activation-function experiments in **TensorFlow / Keras**.

<br />

![Python](https://img.shields.io/badge/Python-3.x-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-MLP-013243?style=flat-square&logo=numpy&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Task](https://img.shields.io/badge/Task-Binary%20classification-5B6CFF?style=flat-square)

<br />

**7,043 customers** &nbsp; · &nbsp; **19 input attributes** &nbsp; · &nbsp; **0.838 reported ROC-AUC**

</div>

---

## The project

Customer churn is a classification problem with a practical goal: identify customers who may cancel so a retention team can decide where to focus. This project uses the IBM Telco Customer Churn dataset to explore that problem and compare compact neural network models.

The notebook covers data exploration, preprocessing, class balancing, model training, and evaluation. It includes a manually implemented NumPy MLP alongside Keras models using **ReLU**, **Tanh**, and **Sigmoid** hidden-layer activations.

## Results at a glance

The project presentation reports the Sigmoid Keras model as the top activation variant:

<div align="center">

| Accuracy | Precision | Recall | ROC-AUC |
|:---:|:---:|:---:|:---:|
| **77.4%** | **56.1%** | **67.6%** | **0.838** |

</div>

These are the reported project results. Run the notebook to reproduce the experiments; metrics may vary with the runtime and package versions.

## The workflow

| Stage | What happens |
|:--|:--|
| **Explore** | Inspect churn balance, feature distributions, and relationships between numerical variables. |
| **Prepare** | Remove the customer ID, handle blank `TotalCharges`, encode categories, and create a stratified train/test split. |
| **Balance** | Scale the features and apply SMOTE to the training split. |
| **Build** | Train a NumPy MLP from scratch and Keras MLP variants with three hidden-layer activations. |
| **Compare** | Review accuracy, precision, recall, ROC-AUC, and training curves. |

## Repository

| File | Description |
|:--|:--|
| [`Model.ipynb`](Model.ipynb) | Full analysis, preprocessing, training, and evaluation notebook. |
| [`README.md`](README.md) | Project overview and instructions. |

## Run it yourself

### 1. Set up Python

```bash
python -m venv .venv
```

Activate the environment, then install the notebook dependencies:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn tensorflow xgboost jupyter
```

### 2. Add the dataset

The notebook expects a CSV named `WA_Fn-UseC_-Telco-Customer-Churn.csv`. The dataset is not included here. Place your copy where the notebook can access it, then update the `pd.read_csv(...)` path in the data-loading cell. Its current path, `/content/WA_Fn-UseC_-Telco-Customer-Churn.csv`, is configured for Google Colab.

### 3. Open the notebook

```bash
jupyter notebook Model.ipynb
```

Run the cells from top to bottom. The notebook writes `encoder.pkl` when its categorical encoding cell runs.

## A few implementation details

- The train/test split is stratified and uses a fixed random seed.
- SMOTE is applied to the training data after the split; the test set remains unresampled.
- The target is `Churn`, with `Yes` encoded as 1 and `No` as 0.
- This is an educational project. Results on this dataset do not guarantee performance on other customer populations or future data.

## Credits

Semester 5 B.Tech Computer Science and Engineering Deep Learning project at Sri Sri University.

**Project team:** Sritam Mishra · Binayak Bhoi · Dhruv Dash · Sarthak Kumar Patel

**Guided by:** Mr. Yugnash
