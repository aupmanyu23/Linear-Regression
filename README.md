# Linear-Regression
LR with Medical Cost Personal Datasets + Laaso&amp;Ridge regression + Feature Engineering 

In this project, we analyzed the **Medical Cost Personal Dataset** using linear regression models to predict healthcare charges based on patient characteristics, such as age, BMI, number of children, and smoking status. The dataset consisted of 1,338 entries, with no missing values, and was predominantly distributed across four regions.

1. **Exploratory Data Analysis (EDA):**
   - The distribution of charges was **right-skewed**, indicating a majority of lower healthcare costs with a few outliers.
   - **Boxplot analysis** of smoker vs. non-smoker charges showed smokers tend to have significantly higher medical charges, confirming smoking as a major cost driver.
   - **Age vs. Charges scatter plot** revealed a positive trend, with charges increasing as age rises, especially for smokers.

2. **Data Preprocessing:**
   - We handled categorical variables (e.g., sex, region, smoker) using **one-hot encoding** via `pd.get_dummies()`.
   - Outliers were filtered using **IQR method**, reducing the dataset size from 1,338 to 1,199 entries.

3. **Model Building:**
   - We employed **Ridge** and **Lasso** regression models after creating polynomial features to improve the model's ability to capture non-linear relationships between the variables.
   - **Ridge Regression** results: MSE = 34,171,440.1, R² = 0.779.
   - **Lasso Regression** results: MSE = 34,118,150.2, R² = 0.780.
   - **Cross-validation (CV) scores** further validated the model's performance, with mean R² scores of 0.749 for both Ridge and Lasso models, confirming the model's reliability.


The analysis successfully identified key drivers of medical costs, notably smoking status and age, which were supported by strong correlations and model performance. The linear models (Ridge and Lasso) showed reasonable predictive power (R² ≈ 0.78) for predicting medical charges, with Lasso having a slight edge in terms of model fit.
