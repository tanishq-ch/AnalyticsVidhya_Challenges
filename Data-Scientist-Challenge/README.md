# 🚗 VahanBima · Customer Lifetime Value Predictor

> **Data Scientist Hiring Hackathon** · Motor Vehicle Insurance · CatBoost Ensemble

---

## 📌 Project Overview

**VahanBima**, a leading motor vehicle insurance company, aims to segment its customers to offer personalized services and optimize claim settlement resources.

This project builds a high-performance machine learning pipeline to predict the **Customer Lifetime Value (CLTV)** of a policyholder based on their demographics, policy details, and past interaction history.

---

## 🏆 Model Performance

| Metric | Score |
|---|---|
| **Evaluation Metric** | R-squared (R²) |
| **Local OOF CV Score** | `~0.1606` |
| **Hackathon Baseline** | `0.15` ✅ Crossed |
| **Validation Strategy** | 5-Fold Cross-Validation |

> **Note:** The OOF CV score is computed across 5 held-out folds, making it a highly reliable estimate of real-world generalization — not an optimistic in-sample score.

---

## 🧠 Core Methodology

The jump from a baseline to a highly predictive model was achieved by treating this **strictly as an insurance-domain problem**, focusing heavily on data representation and localized risk segments.

---

### 1. 🔧 Correcting the Ordinal Data Leak *(Crucial Step)*

Initial EDA revealed that key numerical drivers — `income` and `num_policies` — were formatted as **text ranges** (e.g., `"5L-10L"`, `"More than 1"`).

Standard imputation destroyed this information. We applied **Ordinal Encoding** to map these strings into a mathematical hierarchy, restoring the natural magnitude of a customer's wealth and loyalty.

---

### 2. ⚙️ Domain-Specific Feature Engineering

Tree-based models need clear mathematical signals to make accurate splits. The following features were engineered to represent **risk and financial stress**:

| Feature | Description |
|---|---|
| `has_claimed` | Binary flag — strongest initial risk separator |
| `claim_to_income_tier` | Proxy for financial stress: claim amount ÷ income bracket |
| `policies_per_year` | Loyalty velocity: Total Policies ÷ Vintage |

---

### 3. 🔗 Cross-Features (Categorical Interaction)

Insurance risk is **deeply localized**. By concatenating strings to create cross-features like `area_policy_combo` (e.g., `"Rural_Platinum"`), we allowed the model to immediately identify **highly specific customer niches** without building deep, overfitting tree branches.

---

### 4. 🤖 Algorithm Selection: CatBoost

We selected **CatBoost** over standard Gradient Boosting or LightGBM because of the dataset's **high cardinality of nominal variables** (`area`, `qualification`, `policy`).

CatBoost's proprietary **Ordered Target Encoding** processes categorical strings natively — without requiring sparse One-Hot Encoded matrices — drastically reducing data leakage and overfitting.

---

### 5. ✅ Validation Strategy

To ensure robust performance on both the Public and Private Leaderboards, we used **5-Fold Cross-Validation**.

The final `submission.csv` is an **ensemble average of 5 distinct CatBoost models**, reducing prediction variance and increasing stability.

---

## 🛠️ How to Run

### Prerequisites

Requires **Python 3.8+**. Install dependencies:

```bash
pip install pandas numpy catboost scikit-learn
```

### Execution

1. Place `train.csv` and `test.csv` in the **same root directory** as the script.
2. Run the pipeline:

```bash
python solution2.py
```

3. The script will:
   - Print **fold-by-fold validation scores** to the console
   - Generate final predictions in **`submission_v2.csv`**

---

## 📂 Repository Structure

```
📦 analyticsvidhya_challenges/
├── 📖 README.md                        # Main repository overview
├── 🔒 .gitignore                       # Git ignore rules
└── 📁 Data-Scientist-Challenge/
    ├── 📄 train.csv                    # Training dataset (contains `cltv`) — not included yet
    ├── 📄 test.csv                     # Test dataset — not included yet
    ├── 🐍 solution2.py               # End-to-end ML pipeline (commented)
    ├── 📊 submission_v2.csv    # Final predictions for evaluation
    └── 📖 README.md                    # Project documentation
```

> ⚠️ **Note:** Dataset files (`train.csv`, `test.csv`) are not included in this repository as the competition is still active and the solution is not yet finalized.

---

## 🔄 Pipeline Summary

```
Raw Data
   │
   ▼
Ordinal Encoding (income, num_policies)
   │
   ▼
Feature Engineering (has_claimed, claim_to_income_tier, policies_per_year)
   │
   ▼
Cross-Feature Creation (area_policy_combo, ...)
   │
   ▼
5-Fold CatBoost Training
   │
   ▼
Ensemble Averaging → final_catboost_submission.csv
```

---

## 📋 Key Takeaways

- **Domain knowledge > raw modeling power** — understanding how insurance data is structured was the highest-leverage lever.
- **Ordinal encoding of text ranges** was a non-obvious but critical fix that restored true signal from corrupted features.
- **Cross-features** encoded localized risk patterns that a single variable could never represent.
- **CatBoost's native categorical handling** eliminated the need for OHE, reducing dimensionality and overfitting simultaneously.
- **OOF ensembling** produced a stable, low-variance submission that generalizes beyond the public leaderboard split.

---

*Built for the AnalyticsVidhya Data Scientist Hiring Hackathon.*