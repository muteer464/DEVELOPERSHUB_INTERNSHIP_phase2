# 🛍️ Task 2: Customer Segmentation

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat&logo=jupyter&logoColor=white)
![Unsupervised ML](https://img.shields.io/badge/Unsupervised-ML-9b59b6?style=flat)
![Status](https://img.shields.io/badge/Status-Complete-2ecc71?style=flat)

> **DevelopersHub Corporation — Data Science & Analytics Internship | Task 2**

Group mall customers into meaningful segments based on demographics and spending behavior using unsupervised machine learning — enabling targeted marketing strategies for each group.

---

## 📌 Problem Statement

A mall wants to understand its customer base better to run more effective marketing campaigns. Instead of treating all customers the same, the goal is to identify **distinct customer groups** based on their age, annual income, and spending score — then craft tailored strategies for each group.

This is an **unsupervised learning** problem: there are no predefined labels. The model discovers natural groupings in the data.

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Source** | Mall Customers Dataset (open dataset) |
| **Rows** | 200 customers |
| **Features** | 5 columns |
| **Target** | None — unsupervised clustering |

### Features

| Column | Type | Description |
|---|---|---|
| `CustomerID` | Integer | Unique customer identifier |
| `Gender` | Categorical | Male / Female |
| `Age` | Numeric | Customer age (18–70) |
| `Annual_Income` | Numeric | Annual income in k$ |
| `Spending_Score` | Numeric | Mall-assigned score 1–100 |

---

## 🔍 Exploratory Data Analysis

Key findings from EDA:

- **No missing values** — clean dataset, minimal preprocessing needed
- **Annual Income vs Spending Score** scatter plot reveals **5 natural clusters** clearly
- **Females** slightly outnumber males (56% vs 44%) and tend to have higher spending scores
- **18–25 year olds** have the highest spending scores despite lower incomes
- Features are **largely uncorrelated** — each adds independent clustering signal

---

## ⚙️ Preprocessing Steps

1. **Gender encoding**: Female → 1, Male → 0
2. **Two feature sets defined**:
   - `Set A (2D)`: Annual Income + Spending Score → primary clustering
   - `Set B (3D)`: Age + Annual Income + Spending Score → richer segmentation
3. **StandardScaler** applied to both sets — essential for distance-based clustering
4. Scaling verified visually with before/after scatter plots

---

## 🤖 Models Used

### K-Means Clustering
- **Optimal k** found using Elbow Method + Silhouette Score
- Both methods agree on **k = 5**
- `init='k-means++'` for better centroid initialization
- Silhouette Score: **~0.55**

### Hierarchical (Agglomerative) Clustering
- **Ward linkage** — minimizes within-cluster variance
- Dendrogram cut at k=5 confirms the same grouping
- Produces near-identical cluster assignments to K-Means

### Dimensionality Reduction (Visualization)
- **PCA** — reduces 3D features to 2D, capturing ~85% of variance
- **t-SNE** — preserves local structure, confirms clean cluster separation

---

## 📊 Results — 5 Customer Segments

| Cluster | Segment Name | Avg Income | Avg Spending | Strategy |
|---|---|---|---|---|
| 1 | 🔴 Careful Spenders | Low | Low | Budget deals, loyalty rewards |
| 2 | 🔵 Young Impulsives | Low | High | Trendy products, BNPL, social ads |
| 3 | 🟢 Average Customers | Medium | Medium | Seasonal promotions |
| 4 | 🟡 Wealthy Conservatives | High | Low | VIP programs, exclusivity |
| 5 | 🟣 Premium Targets ⭐ | High | High | Retention, personalized offers |

> **Cluster 5 (Premium Targets)** is the most valuable segment — high income AND high spending. Prioritize retention strategies here.

---

## 🆕 New Customer Prediction

The notebook includes a live prediction function — input any customer's details and get their segment instantly:

```python
predict_customer_segment(
    annual_income  = 95,       # in k$
    spending_score = 82,       # 1–100
    gender         = 'Female',
    age            = 28
)
```

**Output:**
```
=======================================================
         NEW CUSTOMER SEGMENT PREDICTION
=======================================================
  Gender         : Female
  Age            : 28
  Annual Income  : $95k
  Spending Score : 82/100
-------------------------------------------------------
  Predicted Cluster : 5
  Segment           : 🟣 Premium Targets ⭐ — High income, high spending
  Distance to center: 0.312 (lower = more central)
=======================================================
```

The prediction also **plots the new customer** as a star (⭐) on the cluster map.

---

## 🚀 How to Run

### Google Colab (Recommended)
1. Upload `Task2_Customer_Segmentation.ipynb` to [Google Colab](https://colab.research.google.com)
2. Click **Runtime → Run All**
3. Dataset loads automatically from GitHub URL

### Local Setup
```bash
# Clone the repo
git clone https://github.com/muteer464/DevelopersHub-Data-Science-Internship.git
cd DevelopersHub-Data-Science-Internship/Task2_Customer_Segmentation

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter

# Launch notebook
jupyter notebook Task2_Customer_Segmentation.ipynb
```

---

## 🗂️ Project Structure

```
Task2_Customer_Segmentation/
├── Task2_Customer_Segmentation.ipynb   # Full notebook
└── README.md                           # This file
```

---

## 📓 Notebook Structure

```
Section 1 → Setup & Data Loading
Section 2 → Exploratory Data Analysis (6 visualizations)
Section 3 → Preprocessing & Scaling
Section 4 → K-Means Clustering (Elbow + Silhouette + Cluster plot)
Section 5 → Hierarchical Clustering (Dendrogram + Comparison)
Section 6 → PCA & t-SNE Visualization
Section 7 → Cluster Profiling & Business Insights
Section 8 → New Customer Prediction (interactive)
```

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualization |
| `scikit-learn` | K-Means, Agglomerative, PCA, t-SNE, Silhouette |
| `scipy` | Hierarchical dendrogram |

---

## 👤 Author

**Mutee Alpha** — Data Science & Analytics Intern
GitHub: [@muteer464](https://github.com/muteer464)
Program: DevelopersHub Corporation Internship

---



