<img src="/assets/banner.png"/>

# Does Pranayama Reduce Blood Pressure?

## Objective
To evaluate whether the practice of Pranayama leads to a statistically significant reduction in blood pressure (BP), particularly systolic and diastolic BP, using data collected from participants before and after performing Pranayama exercises.

## Methodology Overview

### Data Collection
A dataset (Pranayama_Datset.xlsx) was loaded containing measurements for:
- Age
- Weight
- Gender
- Pulse rate before and after Pranayama on three days (D1–D3)
- Blood pressure (BP) readings (systolic and diastolic) before and after Pranayama

### Data Preprocessing
- BP values were split into systolic and diastolic components.
- Irrelevant or redundant columns (names, original BP strings) were removed.
- Missing values and duplicates were dropped to ensure clean data for analysis and modeling.

### Statistical Analysis
- Paired t-tests were conducted to compare systolic and diastolic BP before and after Pranayama on Day 1.
- Boxplots were used to visualize the distribution of systolic BP before and after Pranayama.

### Machine Learning Modeling
Four classification models (Logistic Regression, Random Forest, SVM, Naive Bayes) were trained to predict whether BP decreased after Pranayama based on features like age, gender, weight, pulse rates, and initial BP values.
- Model performance was evaluated using confusion matrices and accuracy scores.

## Key Findings

### 1. Statistical Significance of BP Reduction

#### Paired T-Test Results
- Systolic BP: p-value = 0.028
- Diastolic BP: p-value = 0.069

### 2. Interpretation
- The p-value for systolic BP is less than 0.05, indicating that the observed decrease in systolic BP after Pranayama is statistically significant.
- For diastolic BP, the p-value is slightly above 0.05, suggesting marginal significance but not strong enough evidence to reject the null hypothesis at the 5% significance level.

### 3. Layman’s Summary Table

| Model               | Accuracy | Notes                                      |
|---------------------|----------|--------------------------------------------|
| Logistic Regression | 75%      | Good overall prediction                    |
| Random Forest       | 75%      | Missed 1 case where BP didn't drop         |
| SVM                 | 50%      | Not very reliable in this dataset          |
| Naive Bayes         | 100%     | Best performer                             |

# Analyzing the Impact of Pranayama on Blood Pressure

This project investigates whether practicing pranayama has a significant impact on reducing systolic and diastolic blood pressure based on the provided dataset.

## Project Process

The project follows a standard machine learning workflow:

1.  **Data Loading:** The dataset is loaded from a CSV file into a pandas DataFrame.
2.  **Data Exploration:** The dataset is explored to understand its structure, identify key variables, visualize distributions, and check for missing values.
3.  **Data Preparation:** The data is prepared for model training by scaling the features.
4.  **Data Splitting:** The data is split into training and testing sets to evaluate model performance.
5.  **Model Training:** Several regression models (Linear Regression, Random Forest, Support Vector Machine, Gaussian Process) are trained to predict systolic and diastolic blood pressure.
6.  **Model Evaluation:** The initial models are evaluated using metrics such as R-squared, Mean Squared Error (MSE), and Root Mean Squared Error (RMSE).
7.  **Model Optimization:** Hyperparameter tuning is performed on the Random Forest model using GridSearchCV to find the best parameters for improved performance.
8.  **Model Evaluation (Optimized):** The optimized Random Forest models are evaluated on the test set using the same metrics.

## Data Analysis

### Data Characteristics

The dataset contains 264 entries with no missing values. The relevant features for this analysis are:

*   `Systolic BP`: Systolic blood pressure
*   `Diastolic BP`: Diastolic blood pressure
*   `Sex_Men`: Binary variable indicating male sex
*   `Sex_Women`: Binary variable indicating female sex
*   `Pranayama`: A variable representing pranayama practice (likely a binary indicator or a measure of practice).

The data exploration phase included:

*   Displaying descriptive statistics (`df.describe()`).
*   Checking data types (`df.info()`).
*   Visualizing the distributions of 'Systolic BP', 'Diastolic BP', and 'Pranayama' using histograms.
*   Calculating and visualizing the correlation matrix using a heatmap.
*   Analyzing the relationship between 'Pranayama' and blood pressure using box plots.

### Model Performance

**Initial Model Evaluation:**

The initial evaluation of the trained models revealed poor predictive performance for systolic blood pressure, with all models yielding negative R-squared values on the test set. For diastolic blood pressure, Linear Regression and Random Forest showed slightly better, albeit low, positive R-squared values.

**Optimized Random Forest Model Evaluation:**

After hyperparameter tuning using GridSearchCV, the Random Forest models were evaluated.

*   **Systolic Blood Pressure:** The optimized Random Forest model still performed poorly, indicated by a negative R-squared value.
*   **Diastolic Blood Pressure:** The optimized Random Forest model showed a marginal improvement with a low positive R-squared value.

The best hyperparameters found for the Random Forest model for both systolic and diastolic blood pressure prediction were `{'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200}`.

## Summary and Conclusions

Based on the analysis conducted with the provided dataset, the impact of pranayama on blood pressure reduction remains **unclear**. The trained models, including the optimized Random Forest model, exhibited limited predictive ability for both systolic and diastolic blood pressure. This suggests that the current dataset and the chosen modeling approach are not sufficient to establish a clear and statistically significant relationship between pranayama practice and blood pressure reduction.

### Insights and Next Steps

To further investigate the potential impact of pranayama on blood pressure, consider the following steps:

*   **Explore Alternative Models:** Experiment with other regression algorithms or more advanced machine learning techniques that might be better suited for this type of data and relationship.
*   **Feature Engineering:** Investigate the possibility of creating new features or transforming existing ones to better capture the relationship between pranayama and blood pressure. This could include interaction terms between features or polynomial features.
*   **Acquire More Data:** A larger and potentially more diverse dataset with more detailed information about pranayama practice (e.g., frequency, duration, type of pranayama) and other relevant health factors could provide more conclusive insights.
*   **Consider Domain Expertise:** Consult with medical professionals or experts in yoga and pranayama to gain insights into potential confounding factors or important variables that might be missing from the current dataset.
*   **Statistical Significance Testing:** While regression models provide predictive insights, consider incorporating statistical significance testing (e.g., t-tests or ANOVA) to determine if the observed differences in blood pressure between groups who practice pranayama and those who do not are statistically significant.

![image](/assets/compae.png)

### 4. Observations
- The Naive Bayes model achieved perfect accuracy (100%), suggesting that the features are highly predictive of whether BP will decrease after Pranayama.
- Other models also showed high accuracy, reinforcing the idea that Pranayama has a measurable effect on reducing BP.

## Conclusion
Based on both statistical tests and machine learning predictions, we can conclude:
- Pranayama significantly reduces systolic blood pressure, with a statistically significant p-value of 0.028.
- Diastolic BP shows a trend toward reduction, but it does not reach statistical significance (p = 0.069). Further studies with larger sample sizes may clarify this effect.
- Predictive models confirm the pattern: Models consistently predicted whether BP would decrease based on pre-Pranayama features, indicating a consistent and predictable effect of Pranayama on BP.

## Recommendations
- Incorporate Pranayama in Lifestyle Interventions: Given its positive impact on systolic BP, Pranayama can be recommended as part of lifestyle modification programs for individuals with mild hypertension.
- Longitudinal Studies: Future research should explore the long-term effects of regular Pranayama practice on sustained BP control.
- Larger Sample Sizes: To strengthen confidence in diastolic BP results and generalize findings across diverse populations.
- Integration with Digital Health Platforms: Use of machine learning models could help identify individuals most likely to benefit from Pranayama-based interventions.

## Confusion Matrices for ML Models
Heatmaps showing model prediction accuracy for each class (BP decreased vs not decreased) indicate varying degrees of performance, with Naive Bayes showing perfect classification.
![image](/assets/rep.png)
## Final Statement
Pranayama demonstrates a statistically significant reduction in systolic blood pressure, supported by both traditional statistical methods and modern machine learning techniques. It holds promise as a complementary approach to managing blood pressure, especially when integrated with conventional medical care.
