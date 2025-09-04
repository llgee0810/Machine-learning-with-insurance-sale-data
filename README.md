# Customer Churn Prediction & Profit Optimisation

## Portfolio Summary
Built an end-to-end predictive analytics pipeline to forecast customer churn and optimise retention strategy using profit maximisation and stochastic risk analysis.  
The project combines:
- Data cleaning & feature engineering
- Predictive modelling (Logistic Regression, Random Forest, XGBoost)
- Threshold optimisation for maximum profitability
- Monte Carlo simulation for risk analysis
- Business recommendations with actionable insights

---

## Business Problem
In the competitive U.S. wireless telecommunications industry, customer churn significantly impacts revenue.  
Retaining customers is more cost-effective than acquiring new ones, yet companies often lack a data-driven approach to retention.  

**Key Question:**  
> Which customers should be targeted with retention offers to maximise profit while managing risk?

---

## Dataset
- **Source:** Historical customer data with usage, demographics, product details, and churn status.
- **Size:** 10,000+ records (sample dataset used for reproducibility).
- **Target Variable:** `Churn` (1 = churned, 0 = retained).

### Schema Highlights
- **Demographics:** Age, Gender, Occupation, Credit Rating, Income
- **Usage:** Monthly Revenue, Minutes, Calls, Device tenure
- **Services:** Product types, premium plans, additional offers
- **Target:** Churn flag (binary)

---

## Data Preparation & Descriptive Analysis
### Key Steps:
1. Identifier removal (CustomerID in training set).
2. Error handling: Replaced impossible values (negative revenue).
3. Zero-inflated variable handling: Converted highly sparse service usage into binary indicators.
4. Outlier & skewness adjustment: Applied log transformation to 11 skewed variables.
5. Feature reduction: Removed highly correlated and redundant variables (from 44 → 31).

### Descriptive Insights:
<img width="291" height="193" alt="image" src="https://github.com/user-attachments/assets/3b64bc57-34aa-42a2-87e4-c126f16a9020" />
<img width="288" height="191" alt="image" src="https://github.com/user-attachments/assets/998a780e-5036-4500-80f4-f365f9b83e95" />
<img width="292" height="193" alt="image" src="https://github.com/user-attachments/assets/6260ffb1-5d00-4f9c-b7da-3de8f7449617" />
<img width="292" height="193" alt="image" src="https://github.com/user-attachments/assets/81756365-5942-4202-a3bb-4e518dbc5b8d" />

- Churn rate ≈ 40%
- Most churners leave between 10–12 months of service.
- Mid-credit score customers are more stable; low & high scores churn more.
- Students & homemakers show the highest churn.

---

## Predictive Modelling
### Models Tested:
- Logistic Regression (LASSO)
- Random Forest (with class weights, SMOTE)
- XGBoost (with Boruta feature selection)

### Evaluation Metrics:
| Model | AUC | Accuracy | Precision | Recall | F1 | Threshold |
|-------|-----|----------|-----------|--------|----|-----------|
| RF (Class Weights) | 0.645 | 0.633 | 0.398 | 0.535 | 0.457 | – |
| XGBoost | 0.643 | 0.392 | 0.313 | 0.931 | 0.468 | – |
| XGBoost – Boruta | 0.642 | 0.572 | 0.370 | 0.687 | **0.481** | 0.48 |
| RF (CW) – Boruta | 0.639 | **0.687** | 0.439 | 0.313 | 0.365 | 0.38 |
| XGBoost – Tuned | 0.637 | 0.591 | 0.371 | 0.605 | 0.460 | 0.50 |
| RF + SMOTE | 0.628 | 0.580 | 0.364 | 0.616 | 0.457 | – |
| Logistic (LASSO) | 0.612 | 0.305 | 0.291 | **0.988** | 0.450 | – |
| Logistic (Boruta) | 0.604 | 0.404 | 0.310 | 0.874 | 0.458 | 0.44 |

**Best Model:**  
XGBoost (Boruta) — Best F1 score (0.481), strong recall (0.687), moderate precision (0.370).

---

## Decision Optimisation
<img width="400" height="300" alt="image" src="https://github.com/user-attachments/assets/b2c3f302-fd43-4a90-8e3f-679c657e93eb" />

- Goal: Select churn probability threshold to maximise expected profit.
- Two scenarios:
  1. Unconstrained targeting (target all above threshold)
  2. Capped targeting (max 3,000 customers)

**Optimal Threshold:** 0.355  
<img width="301" height="199" alt="image" src="https://github.com/user-attachments/assets/a7d5b14d-3c65-4032-bbec-6e7de4e3cfd0" />

- Customers targeted: 2,828  
- Expected annual profit: $753K  
- Profit drops if threshold is lower/higher.

---

## Stochastic Risk Analysis
<img width="253" height="167" alt="image" src="https://github.com/user-attachments/assets/9c83757f-c8e5-4c5e-a477-a18aecb15ece" />

- Method: Monte Carlo Simulation (5,000 runs)
- Assumptions:
  - Contacted → churn risk ↓ 90%
  - Contacted & retained → +10% profit
  - Not contacted & retained → +15% profit
<img width="272" height="180" alt="image" src="https://github.com/user-attachments/assets/d4f6b63b-efd9-4bf2-a62c-61ae2a010b08" />

| Metric | Value |
|--------|-------|
| Mean Profit | $752,850 |
| Std Dev Profit | $4,910 |
| P5 Profit | $744,720 |
| P95 Profit | $760,826 |
| P95 Churn (Contacted) | ≤ 136 customers |

**Insight:**  
Profit is stable across scenarios — low downside risk.

---

## Findings & Recommendations
**What:**  
- Precision targeting (threshold 0.355) beats blanket offers.
- Stable profit with minimal risk.

**So What:**  
- Avoids wasting incentives on low-risk customers.
- Focuses resources where marginal return is highest.

**Now What:**  
1. Adopt threshold-based targeting (0.355).
2. Run pilot, track actual retention & profit.
3. Test alternative incentives (loyalty programs).
4. Refresh model periodically with new data.
5. Ensure ethical data usage & transparency.

---


