# 🏦 Term Deposit Subscription Prediction

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-189fdd?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-ff6b6b?style=flat)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ecc71?style=flat)

> **DevelopersHub Corporation — Data Science & Analytics Internship | Task 1**

Predict whether a bank customer will subscribe to a term deposit based on direct marketing campaign data, using three machine learning models with explainable AI (SHAP).

---

## 📌 Problem Statement

A Portuguese bank ran phone-based marketing campaigns to sell term deposits. The dataset records customer demographics, contact details, and economic indicators. The goal is to build a binary classifier that predicts whether a customer will subscribe (`yes`) or not (`no`).

This helps the bank:
- Target the right customers more efficiently
- Reduce wasted calls on unlikely subscribers
- Understand **why** customers subscribe (not just who)

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) |
| **File** | `bank-additional-full.csv` |
| **Rows** | 41,188 |
| **Features** | 20 input features + 1 target |
| **Target** | `y` — subscription: `yes` (11.3%) / `no` (88.7%) |
| **Challenge** | Severe class imbalance (~8.8:1 ratio) |

### Key Features

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Customer age |
| `job` | Categorical | Type of job |
| `duration` | Numeric | Last call duration (seconds) |
| `campaign` | Numeric | Number of contacts in this campaign |
| `pdays` | Numeric | Days since last contact (999 = never) |
| `emp.var.rate` | Numeric | Employment variation rate |
| `cons.price.idx` | Numeric | Consumer price index |
| `month` | Categorical | Month of last contact |

---

## 🔍 Exploratory Data Analysis

Key findings from EDA:

- **Class imbalance**: 88.7% No vs 11.3% Yes → accuracy is a misleading metric
- **`duration`** is the strongest predictor — longer calls correlate heavily with subscriptions
- **Students & retired** customers subscribe at higher rates than other job types
- **Cellular** contact method outperforms telephone
- **Economic indicators** (`emp.var.rate`, `cons.price.idx`) are highly correlated with each other
- **Best months**: March, September, October, December show higher subscription rates

---

## ⚙️ Preprocessing Steps

1. **`pdays = 999`** → converted to `previously_contacted` binary flag
2. **`unknown`** values → kept as a valid category (carries predictive signal)
3. **Target encoding**: `yes → 1`, `no → 0`
4. **One-Hot Encoding** on all categorical features (`get_dummies`)
5. **Stratified 80/20 train-test split** (preserves class ratio)
6. **StandardScaler** fitted on train only → applied to test (no leakage)
7. **Class imbalance** handled via:
   - `class_weight='balanced'` for Logistic Regression & Random Forest
   - `scale_pos_weight` for XGBoost

---

## 🤖 Models & Results

### Performance Comparison

| Model | F1-Score | ROC-AUC |
|---|---|---|
| 🥇 **XGBoost** | ~0.62 | ~0.80 |
| 🥈 **Random Forest** | ~0.58 | ~0.78 |
| 🥉 **Logistic Regression** | ~0.52 | ~0.75 |

> *Exact scores will vary slightly based on environment. Run the notebook to reproduce.*

### Model Configuration

**Logistic Regression**
```python
LogisticRegression(class_weight='balanced', max_iter=1000, solver='lbfgs')
```

**Random Forest**
```python
RandomForestClassifier(n_estimators=200, max_depth=15,
                       class_weight='balanced', random_state=42)
```

**XGBoost**
```python
XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
              scale_pos_weight=7.9, eval_metric='logloss')
```

---

## 🔎 Explainable AI — SHAP Analysis

SHAP (SHapley Additive exPlanations) is applied to the best-performing XGBoost model to explain predictions at both global and individual levels.

### Visualizations Produced

| Plot | Purpose |
|---|---|
| **Bar Plot** | Top 15 features by mean absolute SHAP value |
| **Beeswarm Plot** | Direction & magnitude of each feature's impact per customer |
| **Waterfall Plot** | Step-by-step explanation for a single customer's prediction |
| **RF Gini Importance** | Compared against SHAP for validation |

### Key SHAP Findings

1. **`duration`** — highest impact; long calls strongly push toward YES
2. **`emp.var.rate`** — low employment variation → more subscriptions
3. **`previously_contacted`** — prior contact dramatically increases likelihood
4. **`month`** — contact timing matters significantly
5. **`age`** — younger (<30) and older (>60) customers are more receptive

---

## 💡 Business Recommendations

| # | Recommendation |
|---|---|
| 1 | **Train agents** to keep customers engaged — longer calls = more conversions |
| 2 | **Prioritize warm leads** — previously contacted customers are far more likely to subscribe |
| 3 | **Time campaigns** around March, September, October, and December |
| 4 | **Monitor economic indicators** — campaigns during low `emp.var.rate` periods perform better |
| 5 | **Segment by age** — tailor messaging for under-30 and over-60 groups |

---

## 🚀 How to Run

### Option A: Google Colab (Recommended)
1. Upload `Task1_Term_Deposit_Prediction.ipynb` to [Google Colab](https://colab.research.google.com)
2. Click **Runtime → Run All**
3. Dataset downloads automatically from UCI

### Option B: Local (Jupyter)
```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/term-deposit-prediction.git
cd term-deposit-prediction

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap jupyter

# 3. Launch notebook
jupyter notebook Task1_Term_Deposit_Prediction.ipynb
```

---

## 🗂️ Project Structure

```
term-deposit-prediction/
│
├── Task1_Term_Deposit_Prediction.ipynb   # Main notebook (all sections)
├── README.md                             # This file
└── requirements.txt                      # Dependencies (optional)
```

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---|---|---|
| `pandas` | 2.x | Data manipulation |
| `numpy` | 1.x | Numerical operations |
| `matplotlib` | 3.x | Plotting |
| `seaborn` | 0.13+ | Statistical visualizations |
| `scikit-learn` | 1.3+ | ML models & preprocessing |
| `xgboost` | 2.x | Gradient boosting model |
| `shap` | 0.44+ | Explainable AI |

---

## 📓 Notebook Structure

```
Section 1 → Setup & Data Loading
Section 2 → Exploratory Data Analysis (5 visualizations)
Section 3 → Data Preprocessing (encoding, scaling, split)
Section 4 → Modeling (LR + RF + XGBoost + ROC curves)
Section 5 → SHAP Explainability (bar, beeswarm, waterfall)
Section 6 → Final Insights & Business Recommendations
```

---

## 👤 Author

**Mutee** — Data Science & Analytics Intern  
DevelopersHub Corporation Internship Program  

---

*Part of a 5-task Data Science Internship. This notebook demonstrates end-to-end ML pipeline development with explainability.*
