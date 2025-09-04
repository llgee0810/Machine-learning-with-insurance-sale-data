# NEOS Adviser Recommendation Modeling — R Machine Learning Project

## Portfolio Summary
Built end-to-end classification and segmentation pipelines in R to understand and predict when advisers recommend NEOS vs other underwriters. The project covers data cleaning, feature engineering, unsupervised clustering, supervised modeling (logistic regression, decision tree, random forest, XGBoost), threshold tuning, model evaluation (AUC, F1, precision/recall), and explainability with DALEX. Delivered insights to inform adviser engagement, pricing/packaging, and product positioning.

---

## Business Problem
When do financial advisers choose NEOS over competitors, and which factors most influence that choice across product lines (Life, TPD, Trauma, IP)? The goal is to identify actionable drivers of recommendation share and segment client/adviser contexts where NEOS can win more consistently.

---

## Dataset
- Source: Adviser recommendations with client demographics, product flags and cover amounts, underwriter/package details, premiums, commission structures, and timestamps.
- Target: `Underwriter_NEOS` (1 = NEOS Life, 0 = other).
- Size: Real dataset is large; repo includes a small sample for reproducibility. See “Data Access” below.

Schema highlights:
- Demographics: AgeNext, Gender, SmokerStatus, HomeState, SelfEmployed, AnnualIncome
- Products: Life, TPD, Trauma, IP (+ cover amounts)
- Commercial: CommissionStructure, Underwriter, Package
- Pricing: Premium, AnnualisedPremium, Inside/OutsideSuperPremium
- Time: Date (used for quarterly trends)

---

## Methods

### Data Preparation
- Deduplication, type fixes (dates, numeric IDs), categorical encoding.
- Missing values: explicit “Missing” for factors; medians for numerics.
- Skew handling: log transforms for income and premium fields.
- Target engineering: binary flag for NEOS vs other.

### Feature Engineering
- CommissionGroup from CommissionStructure (Upfront/Hybrid/Level/Trail/Bundled/Other).
- OccupationGroup via keyword rules (Healthcare, Education, Finance/Admin, Trade/Skilled, IT/Tech, Sales/Marketing, Management, Manual/Construction, Self-Employed, Other).
- PackageCluster: text processing of package names (TF-IDF + k-means) to capture product packaging semantics.

### Unsupervised Segmentation
- PCA on normalized feature set, k-means clustering (k≈3), segment profiling by age, smoking, income, premium (log), and NEOS share.

### Supervised Modeling
- Primary classifier: predict NEOS vs other using logistic regression, decision tree, random forest, and XGBoost.
- Per-product models: separate logistic models for Life, TPD, Trauma, IP.
- Threshold tuning: sweep thresholds to maximize F1 and Youden’s J; record best cut-offs.
- Metrics: AUC, accuracy, precision, recall, F1; confusion matrices.
- Explainability: DALEX feature importance and partial dependence profiles.

### Temporal Trends
- Quarterly recommendation volumes for Life, TPD, Trauma, IP among NEOS cases to spot momentum and seasonality.

---

## Key Insights (what to look for once you run it)
- Drivers of NEOS selection: Expect CommissionGroup, PackageCluster, OccupationGroup, and state/smoking to appear among top features.
- Segment asymmetry: Certain clusters show materially higher NEOS share with distinct income and premium profiles.
- Product asymmetry: NEOS win rates differ by product line; thresholds optimized per product improve F1 meaningfully.
- Quarterly trends: Identify growth/slowdown periods by product to align campaigns and adviser outreach.

---

## Recommendations
1. Adviser Incentive Design  
   Align incentives where CommissionGroup patterns correlate with higher NEOS propensity, while monitoring persistency and compliance.

2. Packaging and Collateral  
   Standardize and promote high-performing package clusters; tailor collateral to occupations and states with higher lift.

3. Pricing and Affordability  
   Use premium and income signals (log-scaled) to calibrate offers and indexation options for segments with lower conversion.

4. Targeted Outreach  
   Prioritize clusters with high latent NEOS propensity but low realized share; run A/B tests on messaging and product bundles.

5. Model-in-the-Loop Enablement  
   Surface DALEX-backed “why” features to advisers (transparent drivers) rather than black-box scores.

---

## Results (placeholders to replace with your outputs)
Add your tables/figures after running:
- Model comparison: AUC, F1, precision, recall across baseline LR, tree, RF, XGB.
- Best thresholds per product (Life, TPD, Trauma, IP).
- Top features (RF Gini; DALEX dropout loss).
- Quarterly recommendation lines for NEOS by product.
- Cluster summary table: size, avg age, smoker share, income_log, premium_log, NEOS share.

Example figure slots:
- `/fig/model_roc.png`
- `/fig/feature_importance_rf.png`
- `/fig/pdp_top_features.png`
- `/fig/quarterly_trends.png`
- `/fig/cluster_profiles.png`

---

## How to Run
1. Place the sample CSV in `/data` (same columns as the full dataset).
2. Open `analysis.Rmd` or `analysis.R` and knit/run to reproduce results.
3. Outputs (tables/plots) are saved under `/out` and `/fig`.

If the full dataset is too large or confidential, provide a small sample and a “Data Access” note (below).

---

## Data Access
- Sample: `/data/sample_A3_Dataset_2025.csv` (anonymised, small).
- Full data: stored privately due to size and confidentiality. Replace with your storage link or access instructions if applicable.

---

## Reproducibility Notes
- Set a random seed for splits and k-means.
- Align factor levels between train and validation after splitting.
- Recompute and record best thresholds per product when data changes.
- Document package versions in `sessionInfo()`.

---

## Ethics and Risk
- Guard against leakage and overfitting; validate out-of-time if possible.
- Monitor fairness across demographics; avoid recommendations that could bias outcomes without justification.
- Backtest assumptions; markets and adviser behaviour can shift.

---
