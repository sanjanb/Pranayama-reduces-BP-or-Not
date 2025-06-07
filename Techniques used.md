# Methodology Explanation

This document outlines the techniques used in the analysis, detailing how they work, their impact on the project, and the rationale behind their selection.

---

## 1. Pandas DataFrame

**How it works:**
Pandas DataFrames are two-dimensional labeled data structures with columns of potentially different types. They are the core object for data manipulation and analysis in Python.

**Impact:**
DataFrames allowed us to load, store, and manipulate the dataset in a structured way. All subsequent steps relied on operating on this DataFrame.

**Why used:**
Pandas is the standard library in Python for data handling and analysis. It provides powerful and efficient tools for working with tabular data.

---

## 2. Displaying Data

**How it works:**
`display(df.head())` from IPython.display is used in environments like Jupyter notebooks to render rich outputs, including DataFrames, in a more formatted and readable way than the standard print().

**Impact:**
Provides a quick and clear view of the first few rows of the DataFrame, helping to understand the data structure and contents.

**Why used:**
To present the DataFrame's head in a user-friendly format within the notebook environment.

---

## 3. Handling FileNotFoundError

**How it works:**
This is a standard Python error handling mechanism. The code inside the try block is executed, and if a FileNotFoundError occurs, the code inside the except block is executed.

**Impact:**
Prevents the notebook from crashing if the specified Excel file is not found in the current directory. It provides a helpful error message to the user.

**Why used:**
To make the data loading process more robust and provide informative feedback in case of a missing file.

---

## 4. Filling Missing Values

**How it works:**
The `.fillna()` method in pandas replaces missing values (NaN) in a DataFrame or Series. In this case, it replaces NaNs with the mean of the respective column. `inplace=True` modifies the DataFrame directly.

**Impact:**
Addresses missing data points in the 'after pranayama systolic' and 'after pranayama diastolic' columns, which is necessary for most statistical analyses and machine learning models.

**Why used:**
Imputing with the mean is a simple and common strategy for handling missing numerical data, especially when the distribution is relatively symmetric.

---

## 5. Correlation Matrix

**How it works:**
The `.corr()` method calculates the pairwise correlation of columns in a DataFrame. It measures the linear relationship between variables.

**Impact:**
Provides insights into the linear associations between the numerical features. This helps in understanding potential relationships and multicollinearity.

**Why used:**
To quickly assess the linear relationships between numerical variables before further analysis or modeling. It was specifically used on numerical columns to avoid errors with non-numerical types.

---

## 6. Box Plots

**How it works:**
Box plots visually display the distribution of a numerical dataset through quartiles. They show the median, interquartile range (IQR), and potential outliers.

**Impact:**
Used to identify the presence and distribution of outliers in the numerical features. The seaborn.boxplot() was used later to compare the distribution of blood pressure changes across different pranayama types.

**Why used:**
A common and effective way to visualize the spread and identify outliers in numerical data.

---

## 7. Winsorizing

**How it works:**
Winsorizing is a method of handling outliers by capping them at a specified percentile or a multiple of the IQR. In this case, values below the lower bound and above the upper bound (calculated using the 1.5*IQR rule) are replaced with the bounds.

**Impact:**
Reduces the influence of extreme outlier values on statistical analyses and model training without completely removing the data points.

**Why used:**
To mitigate the impact of outliers identified through box plots, providing a more robust dataset for modeling.

---

## 8. Dropping Duplicates

**How it works:**
The `.drop_duplicates()` method identifies and removes duplicate rows from a DataFrame. `inplace=True` modifies the DataFrame directly.

**Impact:**
Ensures that each observation in the dataset is unique, preventing skewed results due to repeated entries.

**Why used:**
To maintain data integrity and avoid bias introduced by duplicate records.

---

## 9. Feature Engineering

**How it works:**
Creating new features based on existing ones to capture more relevant information for the analysis or model. Here, the difference in blood pressure is calculated.

**Impact:**
These new features represent the change in blood pressure, which is the target variable for the regression models.

**Why used:**
To directly model the effect of pranayama on blood pressure change, rather than predicting the absolute blood pressure values after pranayama.

---

## 10. Encoding Categorical Features

**How it works:**
Mapping categorical values (like 'Female' and 'Male') to numerical representations (0 and 1). This is necessary for most machine learning algorithms.

**Impact:**
Converts the 'gender' feature into a format that can be used by the regression models.

**Why used:**
Many machine learning models require numerical input data. Simple mapping is suitable for binary categorical features.

---

## 11. Defining Features (X) and Target (y) Variables

**How it works:**
Separating the independent variables (features) that will be used for prediction from the dependent variable(s) (target) that are being predicted.

**Impact:**
Structures the data into the standard format required for training supervised machine learning models.

**Why used:**
To prepare the data for the train_test_split and subsequent model training.

---

## 12. Splitting Data

**How it works:**
Divides the dataset into two subsets: a training set used to train the model and a testing set used to evaluate its performance on unseen data. `test_size` specifies the proportion of data for the test set, and `random_state` ensures reproducibility. `stratify` ensures that the proportion of a given variable (here, 'pranayama') is the same in both the training and testing sets.

**Impact:**
Provides an unbiased evaluation of the model's generalization ability by testing it on data it hasn't seen during training. Stratification is important for maintaining the distribution of the 'pranayama' variable, especially if there are imbalances.

**Why used:**
Standard practice in machine learning to prevent overfitting and get a reliable estimate of model performance.

---

## 13. Scaling Numerical Features

**How it works:**
StandardScaler standardizes features by removing the mean and scaling to unit variance (z-score normalization). It transforms data such that the mean is 0 and the standard deviation is 1.

**Impact:**
Ensures that features on different scales do not disproportionately influence models that are sensitive to feature magnitudes (like SVR and Linear Regression).

**Why used:**
To improve the performance of certain machine learning algorithms by bringing all numerical features to a similar scale. Only 'age' was scaled here as the other numerical features were either target variables or already represented changes.

---

## 14. Paired t-tests

**How it works:**
A statistical test used to compare the means of two related groups (paired observations). It's suitable for comparing measurements taken from the same individuals before and after an intervention. It assumes the differences between pairs are normally distributed.

**Impact:**
Used to determine if there is a statistically significant difference in the mean blood pressure before and after pranayama.

**Why used:**
Appropriate for analyzing paired data where the same individuals are measured at two different time points.

---

## 15. Wilcoxon Signed-Rank Tests

**How it works:**
A non-parametric statistical test used to compare the medians of two related groups. It is an alternative to the paired t-test when the assumption of normality of differences is not met.

**Impact:**
Provides a non-parametric assessment of whether there is a statistically significant difference in blood pressure before and after pranayama.

**Why used:**
As a robust alternative to the paired t-test, especially when the data may not strictly follow a normal distribution.

---

## 16. Effect Size (Hedges' g)

**How it works:**
A measure of the magnitude of the difference between two groups. Hedges' g is a variant of Cohen's d that is less biased for small sample sizes.

**Impact:**
Provides a standardized measure of the practical significance of the observed difference in blood pressure, independent of sample size.

**Why used:**
To quantify the strength of the effect of pranayama on blood pressure, complementing the p-values from the statistical tests.

---

## 17. Regression Models

**How they work:**
- **Linear Regression:** Fits a linear model to predict the target variable as a linear combination of the input features.
- **Support Vector Regression (SVR):** A variant of Support Vector Machines used for regression. It finds a hyperplane that best fits the data points within a margin of tolerance.
- **Random Forest Regressor:** An ensemble learning method that builds multiple decision trees and averages their predictions to improve accuracy and reduce overfitting.

**Impact:**
These models were trained to predict the change in systolic and diastolic blood pressure based on the input features (age, pranayama type, encoded gender).

**Why used:**
Different regression models were chosen to compare their performance and identify which one best captures the relationship between the features and the change in blood pressure. They represent a range of model complexities.

---

## 18. Simple Imputer

**How it works:**
Replaces missing values in a dataset using a specified strategy (e.g., mean, median, most frequent).

**Impact:**
Handled potential NaN values that might still be present in the training data after initial cleaning, preventing errors during model training.

**Why used:**
To ensure that the input data for the machine learning models does not contain missing values, as most algorithms cannot handle them directly.

---

## 19. GridSearchCV

**How it works:**
An exhaustive search over a specified range of hyperparameter values for an estimator. It evaluates each combination of hyperparameters using cross-validation and finds the combination that results in the best performance according to a specified scoring metric.

**Impact:**
Optimizes the performance of the SVR and Random Forest models by systematically searching for the best hyperparameter settings.

**Why used:**
To improve the accuracy and generalization ability of the chosen models by finding the optimal configuration of their hyperparameters.

---

## 20. Evaluation Metrics

**How they work:**
- **Mean Squared Error (MSE):** The average of the squared differences between the actual and predicted values. Lower values indicate better performance.
- **R-squared:** Represents the proportion of the variance in the target variable that is predictable from the features. A value of 1 indicates a perfect fit.
- **Mean Absolute Error (MAE):** The average of the absolute differences between the actual and predicted values. Less sensitive to outliers than MSE.

**Impact:**
Used to quantitatively assess how well the trained and optimized models perform on the unseen test data.

**Why used:**
Standard metrics for evaluating the performance of regression models, providing different perspectives on the prediction accuracy.

---

## 21. Data Visualization

**How it works:**
Libraries for creating static, interactive, and animated visualizations in Python.

**Impact:**
Used to visually explore the data, understand relationships between variables, and present the results of the model evaluations. This makes the findings more interpretable.

**Why used:**
To provide graphical representations of the data and model performance, making it easier to identify patterns, trends, and the effectiveness of the models.

---

## Summary

The methodology follows a typical data science workflow: data understanding, cleaning, preparation, analysis, modeling, and evaluation, with visualization used throughout to aid in understanding and communication. The techniques were chosen based on the nature of the data (tabular), the research question (difference in BP before/after pranayama, predicting BP change), and standard practices in statistical analysis and machine learning.
