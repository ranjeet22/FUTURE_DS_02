<div align="center">

<!-- ANIMATED BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=180&section=header&text=Customer%20Churn%20Prediction&fontSize=42&fontColor=fff&animation=twinkling&fontAlignY=32&desc=ANN%20%7C%20Power%20BI%20%7C%20End-to-End%20ML%20Project&descAlignY=55&descSize=18" width="100%"/>


<!-- BADGES ROW 1 -->
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

<!-- BADGES ROW 2 -->
[![Power BI](https://img.shields.io/badge/Power%20BI-Dashboard-F2C811?style=for-the-badge&logo=powerbi&logoColor=black)](https://powerbi.microsoft.com)
[![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<!-- STATUS BADGES -->
![Status](https://img.shields.io/badge/Status-Completed-22c55e?style=flat-square)
![Internship](https://img.shields.io/badge/Future%20Interns-Data%20Science%20Task%202-6366f1?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-10%2C000%20Customers-0ea5e9?style=flat-square)
![Accuracy](https://img.shields.io/badge/Model-ANN%20Classifier-f59e0b?style=flat-square)

<br/>

**Can we predict which bank customer will leave — before they do?**  
This project answers that with a full ML pipeline: EDA → ANN → Power BI Dashboard.

<br/>

<!-- QUICK LINKS -->
[📓 View Notebook](#-notebook-walkthrough) · [📊 Dashboard](#-power-bi-dashboard) · [🚀 Quick Start](#-quick-start) · [📈 Results](#-model-performance) · [🔍 Insights](#-key-insights)

</div>

---
<img width="4150" height="2400" alt="Image" src="https://github.com/user-attachments/assets/3c793ccc-7853-4fa8-a3ab-17d62978f92f" />

---

## 📌 Table of Contents

<details open>
<summary><b>Click to expand</b></summary>

- [🎯 Project Overview](#-project-overview)
- [📁 Repository Structure](#-repository-structure)
- [📊 Dataset](#-dataset)
- [🔍 Key Insights from EDA](#-key-insights)
- [🧠 ANN Architecture](#-ann-architecture)
- [📈 Model Performance](#-model-performance)
- [📓 Notebook Walkthrough](#-notebook-walkthrough)
- [📊 Power BI Dashboard](#-power-bi-dashboard)
- [🚀 Quick Start](#-quick-start)
- [🔮 Predict on Your Own Query](#-predict-on-your-own-query)
- [🛠️ Tech Stack](#️-tech-stack)
- [👨‍💻 Author](#-author)

</details>

---

## 🎯 Project Overview

> **A bank loses a customer every few minutes — not because of bad service, but because no one saw it coming.**

This is a complete **end-to-end machine learning project** that tackles binary customer churn classification using an Artificial Neural Network. The project covers everything from raw data to a deployable prediction function and an interactive Power BI dashboard.

<div align="center">

```
Raw CSV  →  EDA & Insights  →  Preprocessing  →  ANN Training  →  Evaluation  →  Power BI Dashboard
```

</div>

**What makes this project stand out:**
- 🧠 Improved ANN with `BatchNormalization`, `Dropout`, class weighting, and AUC monitoring
- 📊 4-page interactive Power BI dashboard with KPIs, slicers, and drill-downs
- 🔮 `predict_customer()` function — drop in any customer's details and get instant prediction
- 📉 Threshold tuning analysis — goes beyond the default 0.5 cutoff
- 🔍 Permutation-based feature importance

---

## 📁 Repository Structure

```
customer-churn-prediction/
│
├── 📓 ANN_Churn_Improved.ipynb       ← Main notebook (EDA + Model + Evaluation)
├── 📊 Churn_Dashboard.pbix           ← Power BI dashboard file
│
├── 📂 data/
│   ├── Churn_Modelling.csv           ← Original dataset
│   └── Churn_Modelling_Enhanced.csv  ← Preprocessed (AgeGroup, BalanceBand added)
│
├── 📂 screenshots/
│   ├── dashboard_page1_kpis.png
│   ├── dashboard_page2_demographics.png
│   ├── dashboard_page3_behavior.png
│   ├── dashboard_page4_segments.png
│   ├── confusion_matrix.png
│   └── roc_curve.png
│
└── 📄 README.md
```

---

## 📊 Dataset

<div align="center">

| Property | Value |
|:---|:---|
| **Source** | [Kaggle — Churn Modelling Dataset](https://www.kaggle.com/datasets/shrutimechlearn/churn-modelling) |
| **Rows** | 10,000 customers |
| **Features** | 14 columns |
| **Target** | `Exited` (1 = churned, 0 = retained) |
| **Churn Rate** | 20.37% — imbalanced classification |
| **Geographies** | France, Spain, Germany |

</div>

<details>
<summary><b>📋 Feature Description (click to expand)</b></summary>

<br/>

| Feature | Type | Description |
|:---|:---:|:---|
| `CreditScore` | Numerical | Customer's credit score |
| `Geography` | Categorical | Country: France / Spain / Germany |
| `Gender` | Categorical | Male / Female |
| `Age` | Numerical | Age in years (18–92) |
| `Tenure` | Numerical | Years as a bank customer (0–10) |
| `Balance` | Numerical | Account balance |
| `NumOfProducts` | Numerical | Number of bank products held (1–4) |
| `HasCrCard` | Binary | Has credit card? (0/1) |
| `IsActiveMember` | Binary | Active member? (0/1) |
| `EstimatedSalary` | Numerical | Estimated annual salary |
| `Exited` ⭐ | **Target** | **Churned? 1 = Yes, 0 = No** |

</details>

---

## 🔍 Key Insights

Discovered through EDA before any model was built:

<div align="center">

| Finding | Detail | Business Impact |
|:---|:---|:---|
| 🇩🇪 **Germany Risk** | ~32% churn vs ~16% in France | Geography-specific retention needed |
| 😴 **Inactive Members** | Nearly 2× churn rate vs active | Re-engagement campaigns are critical |
| 📦 **Product Paradox** | 3–4 products = 80%+ churn rate | More products ≠ more loyalty |
| 🎂 **Age Cluster** | 40–60 age group churns most | Targeted offers for mid-age segment |
| 💰 **Zero Balance** | Higher churn than customers with balance | Dormant accounts are a red flag |

</div>

> **"Data doesn't lie. But you have to ask the right questions."**

---

## 🧠 ANN Architecture

<div align="center">

```
Input Layer (11 features)
        │
   ┌────▼────┐
   │ Dense(64) │  + BatchNormalization + ReLU + Dropout(0.3)
   └────┬────┘
        │
   ┌────▼────┐
   │ Dense(32) │  + BatchNormalization + ReLU + Dropout(0.2)
   └────┬────┘
        │
   ┌────▼────┐
   │ Dense(16) │  + BatchNormalization + ReLU + Dropout(0.1)
   └────┬────┘
        │
   ┌────▼────┐
   │ Dense(1)  │  Sigmoid → Churn Probability (0–1)
   └─────────┘
```

</div>

<details>
<summary><b>⚙️ Training Configuration (click to expand)</b></summary>

<br/>

| Parameter | Value | Reason |
|:---|:---|:---|
| **Optimizer** | Adam | Adaptive learning rate |
| **Learning Rate** | 0.001 | Stable convergence |
| **Loss** | Binary Crossentropy | Binary classification |
| **Batch Size** | 32 | Balance of speed & stability |
| **Max Epochs** | 200 | With early stopping |
| **Early Stopping** | `monitor=val_auc`, `patience=15` | Prevents overfitting |
| **LR Scheduler** | `ReduceLROnPlateau` factor=0.5 | Escapes plateaus |
| **Class Weights** | Balanced (churn ≈ 3.9×) | Handles imbalance |
| **Validation Split** | 20% | Honest evaluation |

</details>

---

## 📈 Model Performance

<div align="center">

| Metric | Value | What it means |
|:---:|:---:|:---|
| **Accuracy** | ~86% | Overall correct predictions |
| **AUC-ROC** | ~0.87 | Excellent discrimination ability |
| **Precision** | ~75% | Of predicted churners, how many were right |
| **Recall** | ~55%+ | Of actual churners, how many we caught |
| **F1 Score** | ~64%+ | Harmonic mean — the real performance metric |

</div>

> ⚠️ **Why accuracy alone is misleading here:**  
> A model predicting "retained" for everyone scores **79.6% accuracy** but catches **0 churners**.  
> Recall and AUC-ROC are the metrics that actually matter for imbalanced classification.

<details>
<summary><b>📉 Confusion Matrix Breakdown (click to expand)</b></summary>

<br/>

```
                  Predicted: Retained    Predicted: Churned
Actual: Retained  ✅ True Negative (TN)   ⚠️  False Positive (FP)
Actual: Churned   ❌ False Negative (FN)  ✅ True Positive (TP)
```

- **TN** — Correctly identified loyal customers  
- **TP** — Churners correctly flagged (most important!)  
- **FN** — Missed churners (costly for the bank)  
- **FP** — Loyal customers wrongly flagged (minor cost)

</details>

---

## 📓 Notebook Walkthrough

The notebook is structured as 11 sections:

<details>
<summary><b>📋 Full section breakdown (click to expand)</b></summary>

<br/>

| # | Section | What's covered |
|:---:|:---|:---|
| 1 | **Imports & Setup** | Libraries, seeds, version check |
| 2 | **Load & Explore** | Shape, dtypes, nulls, describe |
| 3 | **EDA** | Class distribution, churn by feature, heatmap |
| 4 | **Preprocessing** | Encoding, stratified split, scaling, class weights |
| 5 | **Build ANN** | Architecture with BatchNorm + Dropout |
| 6 | **Train** | Callbacks, class weighting, fit |
| 7 | **Training Curves** | Accuracy, Loss, AUC — all plotted with best epoch |
| 8 | **Evaluation** | Confusion matrix, ROC curve, classification report, threshold analysis |
| 9 | **Feature Importance** | Permutation-based importance chart |
| 10 | **Custom Prediction** | `predict_customer()` with risk level + visual bar |
| 11 | **Summary** | Improvement table + business interpretation |

</details>

---

## 📊 Power BI Dashboard

A 4-page interactive dashboard built on `Churn_Modelling_Enhanced.csv`.

<details>
<summary><b>📄 Page-by-page breakdown (click to expand)</b></summary>

<br/>

**Page 1 — Executive KPIs**
- Total customers, churned count, churn rate %, active member rate
- Churn vs retained donut chart
- Churn rate by geography bar chart

**Page 2 — Demographic Analysis**
- Churn by gender, age group, number of products, member status
- Conditional color formatting — higher churn = darker red

**Page 3 — Financial & Behavioral Patterns**
- Churn by balance band and tenure
- Avg salary and balance: churned vs retained comparison

**Page 4 — Interactive Segment Explorer**
- Slicers: Geography, Gender, Age Group, NumOfProducts
- All visuals update live — drill into any customer segment
- Headline KPI updates with every filter selection

</details>

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Install dependencies

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

### 3. Run the notebook

```bash
jupyter notebook ANN_Churn_Improved.ipynb
```

> 💡 Make sure `Churn_Modelling.csv` is in the same directory as the notebook before running.

### 4. Open the dashboard

Open `Churn_Dashboard.pbix` in **Power BI Desktop** (free download from Microsoft).

---

## 🔮 Predict on Your Own Query

The notebook includes a ready-to-use `predict_customer()` function.  
Just scroll to **Section 10** and edit the values:

```python
predict_customer(
    CreditScore    = 620,
    Geography      = 'Germany',   # France / Spain / Germany
    Gender         = 'Female',    # Male / Female
    Age            = 45,
    Tenure         = 3,
    Balance        = 120000,
    NumOfProducts  = 3,
    HasCrCard      = 1,           # 0 or 1
    IsActiveMember = 0,           # 0 or 1
    EstimatedSalary= 80000
)
```

**Sample output:**
```
==================================================
        CUSTOMER CHURN PREDICTION
==================================================
  Churn Probability : 78.4%
  Prediction        : CHURN ⚠️
  Risk Level        : HIGH
==================================================
  [🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥⬜⬜⬜⬜⬜⬜⬜]  78.4%
==================================================
```

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Tools |
|:---|:---|
| **Language** | Python 3.10+ |
| **Deep Learning** | TensorFlow 2.x, Keras |
| **ML & Preprocessing** | scikit-learn |
| **Data Analysis** | Pandas, NumPy |
| **Visualization (Python)** | Matplotlib, Seaborn |
| **Business Dashboard** | Power BI Desktop |
| **Environment** | Jupyter Notebook |

</div>

---

## 👨‍💻 Author

<div align="center">

**Ranjeet**  
B.Tech — Artificial Intelligence & Machine Learning  
Bhai Parmanand DSEU Campus, Delhi Skill and Entrepreneurship University

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ranjeet-singh-a08961305/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ranjeet22)

*Part of the Data Science & Analytics Internship at **Future Interns** — Task 2*

</div>

---

<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

⭐ **If this project helped you, consider giving it a star!** ⭐

</div>
