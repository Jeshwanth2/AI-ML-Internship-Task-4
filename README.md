# AI-ML-Internship-Task-4
 Feature Encoding &amp; Scaling
# Adult Income Prediction Dataset - Preprocessing

This repository contains the preprocessed version of the Adult Income Prediction dataset (`adult.csv`), originally sourced from the UCI Machine Learning Repository. The preprocessing steps documented here aim to transform the raw data into a clean, normalized, and encoded format suitable for training various machine learning models.
## Overview
This project focuses on preparing the Adult Income dataset for machine learning tasks. The primary goal is to predict whether an individual's income exceeds $50K/year based on various demographic and employment-related features. The raw dataset requires significant cleaning and transformation, including handling missing values, encoding categorical variables, and scaling numerical features.

## Original Dataset Description
The original `adult.csv` dataset consists of 48,842 entries and 15 columns. It includes a mix of numerical and categorical features:

*   **Numerical Columns**: `age`, `fnlwgt`, `educational-num`, `capital-gain`, `capital-loss`, `hours-per-week`.
*   **Categorical Columns**: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `gender`, `native-country`, `income`.

Initial inspection revealed that some categorical columns contained '?' characters, which were treated as missing values.

## Preprocessing Steps
The following steps were applied to the raw `adult.csv` dataset to prepare it for model training:

### Data Loading
The dataset was loaded into a pandas DataFrame from the path `/content/adult.csv`.

### Feature Identification
Columns were programmatically identified as either numerical (integer or float types) or categorical (object types). This separation is crucial for applying appropriate preprocessing techniques.

### Handling Missing Values
Missing '?' values found in categorical columns (such as `workclass` and `occupation`) were first replaced with `numpy.nan` and then imputed using the mode of their respective columns. This approach was chosen to maintain the most frequent category for the missing entries.

### Label Encoding for Ordinal Features
Two categorical features, `education` and `income`, were identified as ordinal and were label encoded using custom mappings:

*   **`education`**: Mapped to numerical values from 0 (Preschool) to 15 (Doctorate), reflecting their inherent order.
    ```python
education_mapping = {
    'Preschool': 0,
    '1st-4th': 1,
    '5th-6th': 2,
    '7th-8th': 3,
    '9th': 4,
    '10th': 5,
    '11th': 6,
    '12th': 7,
    'HS-grad': 8,
    'Some-college': 9,
    'Assoc-voc': 10,
    'Assoc-acdm': 11,
    'Bachelors': 12,
    'Masters': 13,
    'Prof-school': 14,
    'Doctorate': 15
}
    ```
*   **`income`**: Mapped to 0 (`<=50K`) and 1 (`>50K`), representing the target variable.
    ```python
income_mapping = {
    '<=50K': 0,
    '>50K': 1
}
    ```

### One-Hot Encoding for Nominal Features
All other categorical features (e.g., `workclass`, `marital-status`, `occupation`, `relationship`, `race`, `gender`, `native-country`) were identified as nominal. These were transformed using One-Hot Encoding with `pd.get_dummies` and `drop_first=True` to avoid multicollinearity. This process created new binary columns for each category, significantly expanding the dataset's dimensionality.

### Numerical Feature Scaling (Min-Max Scaling)
All numerical features (`age`, `fnlwgt`, `educational-num`, `capital-gain`, `capital-loss`, `hours-per-week`) were scaled using `MinMaxScaler`. This transformation rescaled their values to a range between 0 and 1. This step is crucial for algorithms sensitive to feature magnitudes. For instance, `age` previously ranged from 17 to 90, `fnlwgt` from 12,285 to 1,490,400, and `capital-gain` from 0 to 99,999. After scaling, all minimum values became 0.0, and maximum values became 1.0.

## Impact of Scaling on Machine Learning Algorithms
Scaling numerical features is a critical preprocessing step for several reasons, particularly for algorithms sensitive to the magnitude and range of input features:

1.  **Algorithms Sensitive to Feature Scales**: Many machine learning algorithms perform poorly or converge slowly when input features have vastly different scales. This includes:
    *   **Gradient Descent-based Algorithms (e.g., Logistic Regression, Neural Networks)**: Scaling helps create a more spherical cost function landscape, allowing the optimization algorithm to converge faster and avoid oscillations.
    *   **Distance-based Algorithms (e.g., K-Nearest Neighbors, Support Vector Machines, K-Means Clustering)**: Unscaled features with larger ranges can dominate distance calculations, effectively overshadowing the contributions of other features. Scaling ensures all features contribute proportionally.
    *   **Principal Component Analysis (PCA)**: PCA is sensitive to feature variances. Unscaled features with larger variances can disproportionately influence principal components, leading to biased results.

2.  **Faster Convergence and Preventing Dominance**: Scaling brings all numerical features into a comparable range (e.g., [0, 1] with Min-Max scaling or mean 0 and standard deviation 1 with Standardization). This leads to faster convergence for gradient-based methods and prevents features with larger values from inadvertently dominating the learning process.

3.  **Interpretability and Numerical Stability**: Scaling can improve the interpretability of feature importances or model coefficients for some models. It also enhances numerical stability by preventing extremely large or small feature values from causing precision issues during computations.

## Processed Dataset
The final transformed dataset, `df_encoded`, now contains 48,842 entries and 84 columns. All features are numerical (`float64`, `int64`, or `bool`), and there are no remaining missing values. This dataset is stored as `adult_processed.csv`.

## Usage
The `adult_processed.csv` file is ready for direct use in various machine learning models for tasks such as classification (predicting income >50K or <=50K). This preprocessed dataset provides a clean and scaled foundation for model training and evaluation.
