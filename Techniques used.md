# Techniques Used in Pranayama Blood Pressure Analysis Project

## 1. Pandas DataFrame Operations

**How it works:** Pandas DataFrames are 2D labeled structures with columns of potentially different types.

**Impact:** Structured loading, storage, and manipulation of data.

**Why used:** Pandas is the standard for tabular data analysis in Python.

---

## 2. Displaying Data

**How it works:** `display()` from `IPython.display` formats output in Jupyter.

**Impact:** Easier readability of DataFrame preview.

**Why used:** Better visual clarity than `print()`.

---

## 3. Handling FileNotFoundError

**How it works:** `try...except` catches and handles file-related errors.

**Impact:** Prevents crashes and shows meaningful messages.

**Why used:** To improve robustness in file loading.

---

## 4. Filling Missing Values

**How it works:** `fillna()` fills NaNs with column means.

**Impact:** Prepares clean data for modeling.

**Why used:** Common and simple technique for numeric imputation.

---

## 5. Correlation Matrix

**How it works:** `corr()` computes linear relationships between variables.

**Impact:** Reveals dependencies and potential multicollinearity.

**Why used:** Fast and intuitive relationship analysis.

---

## 6. Box Plots

**How it works:** Show medians, IQR, and outliers.

**Impact:** Outlier detection and distribution understanding.

**Why used:** Standard visualization for numeric data spread.

---

## 7. Winsorizing

**How it works:** `clip()` caps values at lower and upper bounds.

**Impact:** Controls influence of outliers without deleting data.

**Why used:** Reduces noise while preserving data.

---

## 8. Dropping Duplicates

**How it works:** `drop_duplicates()` removes repeated rows.

**Impact:** Prevents bias and ensures integrity.

**Why used:** To clean and de-duplicate dataset.

---

## 9. Feature Engineering (BP Difference)

**How it works:** New columns = before - after.

**Impact:** Represents the effect of pranayama.

**Why used:** Directly models the variable of interest.

---

## 10. Encoding Categorical Features

**How it works:** `map()` converts categories into numbers.

**Impact:** Makes data suitable for ML models.

**Why used:** Necessary for numeric model input.

---

## 11. Defining Features and Target Variables

**How it works:** Splitting DataFrame into X (inputs) and y (outputs).

**Impact:** Structures data for ML workflows.

**Why used:** Required step before model training.

---

## 12. Splitting Data (train\_test\_split)

**How it works:** Creates training and testing sets.

**Impact:** Enables unbiased model evaluation.

**Why used:** Standard in supervised ML.

---

## 13. Scaling Numerical Features (StandardScaler)

**How it works:** Transforms features to zero mean and unit variance.

**Impact:** Prevents magnitude bias in models.

**Why used:** Improves model performance and convergence.

---

## 14. Paired t-tests

**How it works:** Tests mean differences between paired samples.

**Impact:** Evaluates significance of BP change.

**Why used:** Appropriate for pre/post intervention analysis.

---

## 15. Wilcoxon Signed-Rank Tests

**How it works:** Non-parametric paired test.

**Impact:** Confirms BP changes without normality assumptions.

**Why used:** More robust than t-test for non-normal data.

---

## 16. Effect Size (Hedges' g)

**How it works:** Quantifies magnitude of difference.

**Impact:** Adds practical significance to statistical tests.

**Why used:** Complements p-value with effect size.

---

## 17. Regression Models

**Linear Regression:** Fits straight-line model.

**SVR:** Fits data with margins using kernels.

**Random Forest Regressor:** Averages multiple trees for prediction.

**Impact:** Predicts BP changes.

**Why used:** Offers a range of model complexity for comparison.

---

## 18. Simple Imputer

**How it works:** Fills missing values using a strategy like mean.

**Impact:** Ensures model-ready data.

**Why used:** Prevents model failures due to NaNs.

---

## 19. GridSearchCV

**How it works:** Searches best hyperparameters via cross-validation.

**Impact:** Improves model accuracy.

**Why used:** Finds optimal model settings.

---

## 20. Evaluation Metrics

**MSE:** Penalizes large errors.

**R-squared:** Explains variance.

**MAE:** Penalizes average error.

**Impact:** Quantifies model performance.

**Why used:** Standard tools to compare models.

---

## 21. Data Visualization

**How it works:** Matplotlib and Seaborn create charts.

**Impact:** Aids in understanding and communicating results.

**Why used:** Essential for data storytelling and analysis.

---

## Summary

This project followed a full data science pipeline: loading → cleaning → exploring → transforming → modeling → validating. Each method was selected based on the nature of blood pressure data and the goal of measuring the impact of pranayama on health metrics.
