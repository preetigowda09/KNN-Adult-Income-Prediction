# 🧠 Adult Income Prediction using K-Nearest Neighbors

A machine learning classification project that predicts whether a person's annual income is **≤50K or >50K** using the **K-Nearest Neighbors (KNN)** algorithm.

This project focuses not only on building a KNN model, but also on understanding the complete machine learning workflow — from data preprocessing and encoding to hyperparameter tuning, model evaluation, and prediction on new data.

---

## 📌 Project Overview

The **Adult Census Income dataset** contains demographic and employment-related information about individuals.

The goal of this project is to predict the income class of an individual:

* `<=50K`
* `>50K`

The project uses **K-Nearest Neighbors (KNN)** for binary classification.

---

## 🎯 Objectives

* Understand and implement the KNN classification workflow
* Handle numerical and categorical features
* Apply appropriate feature preprocessing
* Understand why feature scaling is important for KNN
* Use `LabelEncoder` for the target variable
* Use `OneHotEncoder` for categorical input features
* Use `ColumnTransformer` to combine different preprocessing techniques
* Tune the K hyperparameter using validation data
* Evaluate the final model using multiple metrics
* Make predictions on new, unseen data

---

## 📂 Dataset

**Dataset:** Adult Census Income Dataset

The dataset contains demographic, education, employment, and financial information.

### Target

| Target  | Meaning                                    |
| ------- | ------------------------------------------ |
| `<=50K` | Annual income is less than or equal to 50K |
| `>50K`  | Annual income is greater than 50K          |

### Features

The dataset contains features such as:

* Age
* Workclass
* Final weight (`fnlwgt`)
* Education
* Education number
* Marital status
* Occupation
* Relationship
* Race
* Sex
* Capital gain
* Capital loss
* Hours per week
* Native country

---

## 🔄 Machine Learning Workflow

```text
                    Adult Dataset
                         │
                         ▼
                  Data Cleaning
                         │
                         ▼
                 Feature / Target
                    Separation
                         │
                         ▼
                 Train / Test Split
                    ┌────┴────┐
                    ▼         ▼
                 Training     Test
                    │
                    ▼
              Preprocessing
             ┌──────┴──────┐
             │             │
      Numerical Data   Categorical Data
             │             │
      StandardScaler   OneHotEncoder
             │             │
             └──────┬──────┘
                    ▼
             ColumnTransformer
                    │
                    ▼
              Validation Split
                    │
                    ▼
            Try Different K Values
              1, 3, 5, 7, 9
                    │
                    ▼
               Select K = 7
                    │
                    ▼
           Train Final KNN Model
                    │
                    ▼
             Final Test Evaluation
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
       Accuracy  Confusion  Classification
                  Matrix       Report
                    │
                    ▼
              New Data Prediction
```

---

## 🧹 Data Preprocessing

### 1. Cleaning

Column names and target values were stripped of unnecessary whitespace.

```python
df.columns = df.columns.str.strip()
df["income"] = df["income"].str.strip()
```

### 2. Train-Test Split

The dataset was divided into:

* **80% training data**
* **20% test data**

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The test set was kept separate for final evaluation.

---

## 🔢 Encoding

The dataset contains both numerical and categorical features.

### LabelEncoder

`LabelEncoder` was used for the **target variable**:

```text
<=50K → 0
>50K  → 1
```

### OneHotEncoder

`OneHotEncoder` was used for categorical input features such as:

* Workclass
* Education
* Occupation
* Sex
* Native country

This converts categories into numerical binary features that the KNN algorithm can work with.

---

## 📏 Feature Scaling

KNN is distance-based, so feature scaling is very important.

`StandardScaler` was applied to numerical features:

```python
StandardScaler()
```

It standardizes numerical features so that features with large numerical ranges do not dominate the distance calculation.

For example:

```text
Age              → relatively small range
Capital-gain     → potentially much larger values
```

Without scaling, the larger-valued feature could have an excessive influence on Euclidean distance.

---

## 🧩 ColumnTransformer

Different preprocessing techniques were required for different types of features.

Therefore, `ColumnTransformer` was used:

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", encoder, categorical_cols)
    ]
)
```

This allows:

```text
Numerical columns → StandardScaler
Categorical columns → OneHotEncoder
```

to be performed within one preprocessing pipeline.

---

## 🤖 KNN Model

The project uses:

```python
KNeighborsClassifier()
```

KNN classifies a new sample by:

1. Calculating distances to training samples
2. Finding the nearest `K` neighbours
3. Looking at their class labels
4. Selecting the class with the majority vote

For example:

```text
K = 5

Neighbour 1 → >50K
Neighbour 2 → >50K
Neighbour 3 → <=50K
Neighbour 4 → >50K
Neighbour 5 → <=50K

>50K → 3 votes
<=50K → 2 votes

Prediction → >50K
```

---

# 🎛️ Hyperparameter Tuning

Instead of assuming a value for `K`, different values were tested using a validation set.

The following values were evaluated:

```text
K = 1
K = 3
K = 5
K = 7
K = 9
```

### Validation Results

|     K | Validation Accuracy |
| ----: | ------------------: |
|     1 |              79.65% |
|     3 |              82.26% |
|     5 |              82.67% |
| **7** |          **83.36%** |
|     9 |              83.15% |

Based on validation performance:

> **K = 7 was selected as the final K value.**

The model was then retrained using the full training dataset with `K = 7`.

---

# 📊 Model Evaluation

The final model achieved:

## 🎯 Final Test Accuracy

**83.92%**

However, accuracy alone is not sufficient because the dataset contains an imbalance between the two income classes.

Therefore, the model was also evaluated using:

* Confusion Matrix
* Precision
* Recall
* F1-score
* Support

---

## 🔲 Confusion Matrix

The confusion matrix helps identify the types of predictions the model gets right and wrong.

```text
                    Predicted
                 <=50K     >50K

Actual <=50K      4484      458
Actual >50K        609      962
```

### Interpretation

* **4484** → Correctly predicted `<=50K`
* **458** → Predicted `>50K`, but actual was `<=50K`
* **609** → Predicted `<=50K`, but actual was `>50K`
* **962** → Correctly predicted `>50K`

---

## 📋 Classification Report

The final model produced:

| Class   | Precision | Recall | F1-score |
| ------- | --------: | -----: | -------: |
| `<=50K` |      0.88 |   0.91 |     0.89 |
| `>50K`  |      0.68 |   0.61 |     0.64 |

### Why not rely only on accuracy?

The dataset has more `<=50K` samples than `>50K` samples.

Because of this imbalance, a model could achieve a reasonably high accuracy while still performing poorly on the minority class.

Therefore, **precision, recall, F1-score, and the confusion matrix provide a more complete picture of model performance.**

---

# 🧪 Prediction on New Data

After training the final KNN model, it was also tested on a completely new person's information.

Example input:

```text
Age: 30
Workclass: Private
Education: Bachelors
Occupation: Tech-support
Hours per week: 40
Sex: Female
...
```

The new data goes through the **same preprocessing steps** used during training:

```text
New Person
    ↓
ColumnTransformer
    ↓
Scaling + One-Hot Encoding
    ↓
KNN Model
    ↓
Income Prediction
```

The model then predicts either:

```text
<=50K
```

or

```text
>50K
```

---

# 🧠 Key Concepts Learned

Through this project, the following concepts were practiced:

* K-Nearest Neighbors
* Euclidean distance
* K value selection
* Hyperparameter tuning
* Training / validation / test split
* Feature scaling
* Standardization
* Label Encoding
* One-Hot Encoding
* ColumnTransformer
* Data leakage prevention
* Binary classification
* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1-score
* Class imbalance
* Prediction on unseen data

---

# ⚖️ Why KNN?

KNN was chosen because it provides an intuitive approach to classification:

> **Similar data points tend to have similar labels.**

Unlike Logistic Regression, KNN does not try to learn a fixed linear decision boundary. Instead, it makes predictions based on the local neighbourhood of each new sample.

However, KNN also has limitations:

* Prediction can become expensive for large datasets
* Feature scaling is important
* Sensitive to irrelevant features
* Sensitive to the choice of `K`
* Can struggle with high-dimensional data

---

# 🚀 Future Improvements

Possible improvements include:

* Experimenting with different distance metrics
* Testing additional K values
* Trying weighted KNN
* Handling class imbalance more explicitly
* Comparing KNN with Logistic Regression, Decision Trees, and Random Forest
* Performing more systematic hyperparameter tuning
* Building a simple prediction interface

---

# 🛠️ Technologies Used

* **Python**
* **Pandas**
* **Scikit-learn**
* **NumPy**
* **K-Nearest Neighbors**
* **LabelEncoder**
* **OneHotEncoder**
* **StandardScaler**
* **ColumnTransformer**

---

# 📁 Project Structure

```text
KNN-Adult-Income-Prediction/
│
├── main.py
├── adult.csv
├── README.md
└── .gitignore
```

> **Note:** If the dataset is not included in the repository, download the Adult Census Income dataset from its original source and place it in the project directory as `adult.csv`.

---

# 📌 Conclusion

This project demonstrates a complete machine learning classification workflow using **K-Nearest Neighbors**.

The model achieved a final test accuracy of approximately **83.92%** after selecting **K = 7** through validation.

More importantly, the project demonstrates that model evaluation should not rely on accuracy alone, especially when working with imbalanced data. The confusion matrix and classification report provided additional insight into how the model performed for each income class.

The project also provided practical experience with **encoding, feature scaling, preprocessing pipelines, hyperparameter tuning, and predictions on unseen data**.

---

## 👩‍💻 Author

**Preethi A**

CSE (AI & ML) Student
Interested in Machine Learning, AI, and real-world applications.

---

⭐ If you found this project useful, feel free to explore the code and experiment with different K values and models.
#   K N N - A d u l t - I n c o m e - P r e d i c t i o n  
 