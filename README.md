# 🧠 Adult Income Prediction using K-Nearest Neighbors

> 🎓 **Mini Learning Project | Machine Learning Classification**

A hands-on machine learning project that predicts whether a person's annual income is **`<=50K` or `>50K`** using the **K-Nearest Neighbors (KNN)** algorithm.

The goal of this mini project was to understand a complete ML workflow — **data preprocessing → feature scaling → KNN → hyperparameter tuning → evaluation → new-data prediction**.

---

## 📌 Project Overview

The **Adult Census Income Dataset** contains demographic, education, employment, and financial information about individuals.

The model predicts one of two income classes:

| Income  | Meaning             |
| ------- | ------------------- |
| `<=50K` | Annual income ≤ 50K |
| `>50K`  | Annual income > 50K |

---

## 🎯 What I Practiced

* KNN classification
* Numerical and categorical preprocessing
* `LabelEncoder`
* `OneHotEncoder`
* `StandardScaler`
* `ColumnTransformer`
* Train / validation / test split
* Hyperparameter tuning
* Confusion Matrix
* Precision, Recall and F1-score
* Prediction on new data

---

## 📂 Dataset

### Features

The dataset contains:

`Age` · `Workclass` · `fnlwgt` · `Education` · `Education-num` · `Marital-status` · `Occupation` · `Relationship` · `Race` · `Sex` · `Capital-gain` · `Capital-loss` · `Hours-per-week` · `Native-country`

### Target

**`income`**

```text
<=50K
>50K
```

---

## 🔄 ML Workflow

```text
Adult Dataset
     ↓
Data Cleaning
     ↓
Feature / Target Separation
     ↓
Train / Test Split
     ↓
Preprocessing
     ├── Numerical → StandardScaler
     └── Categorical → OneHotEncoder
     ↓
ColumnTransformer
     ↓
Validation Split
     ↓
Try K = 1, 3, 5, 7, 9
     ↓
Select K = 7
     ↓
Final KNN Model
     ↓
Test Evaluation
     ↓
New Data Prediction
```

---

## 🧹 Data Preprocessing

### Data Cleaning

Extra spaces in column names and target values were removed:

```python
df.columns = df.columns.str.strip()
df["income"] = df["income"].str.strip()
```

### Train-Test Split

```python
train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

**80%** of the data was used for training and **20%** was kept for final testing.

---

## 🔢 Encoding

### Target — LabelEncoder

```text
<=50K → 0
>50K  → 1
```

### Categorical Features — OneHotEncoder

Categorical features such as **Workclass, Education, Occupation, Relationship, Race, Sex, and Native Country** were converted into numerical features using `OneHotEncoder`.

---

## 📏 Feature Scaling

KNN is a **distance-based algorithm**, so feature scaling is important.

For example, `Age` and `Capital-gain` have very different numerical ranges.

Therefore, numerical features were standardized using:

```python
StandardScaler()
```

---

## 🧩 ColumnTransformer

Different transformations were required for different types of features.

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", encoder, categorical_cols)
    ]
)
```

After preprocessing:

```text
X_train → (26048, 108)
X_test  → (6513, 108)
```

The original features were transformed into **108 features** after One-Hot Encoding and scaling.

---

## 🤖 K-Nearest Neighbors

The initial model used:

```python
KNeighborsClassifier(n_neighbors=5)
```

KNN:

1. Calculates distances
2. Finds the nearest K neighbours
3. Checks their class labels
4. Uses majority voting

For example, with `K = 5`:

```text
>50K   → 3 neighbours
<=50K  → 2 neighbours

Prediction → >50K
```

---

## 🎛️ K Hyperparameter Tuning

I tested different values of K using validation data.

|     K | Validation Accuracy |
| ----: | ------------------: |
|     1 |              79.65% |
|     3 |              82.26% |
|     5 |              82.67% |
| **7** |          **83.36%** |
|     9 |              83.15% |

### 🏆 Best K: **7**

K = 7 achieved the highest validation accuracy among the tested values.

The final model was therefore trained using **K = 7**.

---

## 📊 Model Evaluation

### ⭐ Final Test Accuracy

# **83.92%**

### Confusion Matrix

| Actual / Predicted | `<=50K` | `>50K` |
| ------------------ | ------: | -----: |
| `<=50K`            |    4484 |    458 |
| `>50K`             |     609 |    962 |

### Classification Report

| Class   | Precision | Recall | F1-Score |
| ------- | --------: | -----: | -------: |
| `<=50K` |      0.88 |   0.91 |     0.89 |
| `>50K`  |      0.68 |   0.61 |     0.64 |

Since the dataset contains more `<=50K` samples than `>50K` samples, I considered **precision, recall and F1-score** along with accuracy.

---

## 🧪 New Data Prediction

The trained model was also used to predict the income class of a manually created new person.

```text
New Person
     ↓
ColumnTransformer
     ↓
Scaling + Encoding
     ↓
KNN Model
     ↓
Predicted Income Class
```

This demonstrates prediction on **previously unseen data**.

---

## 🧠 Key Learning

This project helped me understand how a complete KNN classification pipeline works in practice.

### Concepts reinforced

* Euclidean distance
* K selection
* Feature scaling
* Encoding categorical data
* `ColumnTransformer`
* Validation-based hyperparameter tuning
* Confusion Matrix
* Precision
* Recall
* F1-score
* Class imbalance
* New-data prediction

---

## ⚖️ Why KNN?

> **Similar data points tend to have similar labels.**

KNN provides a simple distance-based approach to classification.

### Limitations

* Can be expensive for large datasets
* Requires feature scaling
* Sensitive to irrelevant features
* Performance depends on K
* Can struggle with high-dimensional data

---

## 🚀 Future Improvements

* Try different distance metrics
* Experiment with weighted KNN
* Perform more systematic hyperparameter tuning
* Compare with Logistic Regression
* Compare with Decision Tree
* Compare with Random Forest

---

## 🛠️ Technologies

| Technology      | Used For             |
| --------------- | -------------------- |
| 🐍 Python       | Programming          |
| 🐼 Pandas       | Data handling        |
| 🔢 NumPy        | Numerical operations |
| 🤖 Scikit-learn | ML & preprocessing   |
| 💻 VS Code      | Development          |
| 🐙 Git & GitHub | Version control      |

---

## 📁 Project Structure

```text
KNN-Adult-Income-Prediction/
│
├── main.py
├── adult.csv
├── README.md
└── .gitignore
```

---

## 📌 Conclusion

This **mini learning project** gave me hands-on experience with the complete KNN classification workflow.

🎯 **Final Test Accuracy: 83.92%**

🏆 **Selected K: 7**

The most important learning outcomes were **encoding, feature scaling, ColumnTransformer, validation-based K tuning, and model evaluation**.

---

## 👩‍💻 Author

**Preethi A**
*CSE (AI & ML) Student*

⭐ *A mini learning project built while learning and practicing Machine Learning.*
