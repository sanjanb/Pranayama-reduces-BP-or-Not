<img src="/assets/banner.png" style="width: 50%; object-fit: cover;">

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
