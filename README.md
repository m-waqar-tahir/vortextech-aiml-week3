# Student Employability: Regression and Clustering

Vortex Tech AI & ML Internship 2026 — Week 3

## Overview

Two independent machine learning tasks on the same cleaned dataset used in
Weeks 1 and 2:

1. **Regression** — predicting `expected_salary_pkr`, a continuous target,
   using Linear Regression and Random Forest Regressor.
2. **Clustering** — grouping students into hidden segments using K-Means on
   `cgpa` and `attendance_percentage`, with the number of clusters chosen via
   the elbow method.

## Dataset

`pk_student_employability_cleaned.csv` — 47,839 cleaned student records, 49
columns, the same cleaned file produced in the Week 1 project.

## Files

- `student_employability_regression_clustering.ipynb` — main notebook, split
  into a Regression section and a Clustering section, every code cell
  preceded by a titled markdown description
- `pk_student_employability_cleaned.csv` — cleaned dataset used by the
  notebook
- `Week3_Regression_Clustering_Full_Guide.txt` — step-by-step explanation of
  every code cell, including the reasoning behind feature and column choices
- `requirements.txt` — Python libraries required to run the notebook

## Part 1: Regression

- Target: `expected_salary_pkr`
- Features exclude `student_id` (identifier) and `likely_to_get_hired` /
  `employability_status` (leakage risks — both are outcome labels partly
  determined by salary and related factors)
- `job_readiness_score` is kept as a feature: it correlates with the target
  at only ~0.28, a real but moderate relationship, not leakage
- 80/20 train-test split, no stratification (continuous target)
- Two models compared: Linear Regression and Random Forest Regressor
- Evaluated with RMSE and R²

**Results:**
- Linear Regression: RMSE ≈ 19,847 PKR, R² ≈ 0.454
- Random Forest: RMSE ≈ 20,336 PKR, R² ≈ 0.427
- Linear Regression slightly outperforms Random Forest — a useful finding in
  itself, since more complexity doesn't automatically help, especially with
  200+ one-hot encoded columns where Random Forest has more room to fit noise

## Part 2: Clustering

- Columns: `cgpa` and `attendance_percentage`, scaled with StandardScaler
- Elbow method run for k = 1 to 10; k=4 chosen based on where the rate of
  inertia reduction drops by more than half (from ~21,564 between k=2→3 to
  ~9,515 between k=3→4)
- Final KMeans fit with k=4, visualized as a 2D scatter plot colored by
  cluster

**Cluster profiles:**
- High performer, lower attendance (CGPA ~3.35, attendance ~66%)
- At-risk: low CGPA and low attendance (CGPA ~2.50, attendance ~66%)
- High performer, engaged (CGPA ~3.29, attendance ~88%)
- Present but struggling: high attendance, lower CGPA (CGPA ~2.45,
  attendance ~88%)

## How to run

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Open the notebook:
   ```
   jupyter notebook student_employability_regression_clustering.ipynb
   ```
3. Run all cells in order.

## Key Findings

- **Regression:** Linear Regression outperformed Random Forest for predicting
  expected salary (RMSE ≈ 19,847 PKR vs ≈ 20,336 PKR, R² ≈ 0.454 vs ≈ 0.427).
  This is a meaningful result in itself — added model complexity did not
  translate into better predictions here, likely because Random Forest had
  more room to fit noise across the 200+ columns produced by one-hot
  encoding several high-cardinality categorical features.
- **Regression accuracy in context:** an R² of ~0.45 means the model
  explains under half of what drives expected salary. Academic and
  activity-based features alone are informative but incomplete — factors
  like actual work experience or standardized skill assessments, not
  present in this dataset, likely account for a meaningful share of the
  remaining variation.
- **Clustering:** four distinct student segments emerged from CGPA and
  attendance alone, split almost evenly across ~11,300–13,400 students each.
  The clearest actionable group is the at-risk segment (low CGPA, low
  attendance), while the "present but struggling" segment (high attendance,
  low CGPA) suggests attendance alone is not a reliable proxy for academic
  engagement.
- **Practical implication:** these two techniques answer different
  questions — regression estimates a specific expected value, while
  clustering reveals structure and subgroups that a single average would
  hide. Together, they suggest a university advising team could use the
  clusters to prioritize *who* to reach out to, and features like CGPA and
  internship experience to inform *what* outcomes to expect.

## Conclusion

This project applied two core machine learning techniques, regression and
clustering, to the same real-world student dataset, moving beyond the
descriptive analysis of Weeks 1 and 2 into predictive and unsupervised
methods. The regression results highlight the value of testing multiple
models rather than assuming complexity improves performance, while the
clustering results demonstrate how unsupervised learning can surface
actionable subgroups that summary statistics alone would miss. Together,
these findings reflect a practical, evidence-based approach to student
employability analysis, with clear directions identified for further
refinement.

## Repository

https://github.com/m-waqar-tahir/vortextech-aiml-week3

## Status

✅ Completed

## Author

Muhammad Waqar Tahir
Vortex Tech AI & ML Internship 2026 — Week 3
