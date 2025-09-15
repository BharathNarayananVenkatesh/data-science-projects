# 📊 Regression Analysis: Predicting Glucose Levels

This project uses **Multiple Linear Regression**, **Robust Regression**, and **Box-Cox Transformation** to predict **blood glucose levels** using patient health indicators.  
It demonstrates **end-to-end regression modeling**, including **data preprocessing, feature selection, model diagnostics, and performance evaluation**.

---

## 📂 Project Structure
- **data/** – Contains dataset (or link to Kaggle dataset)
- **scripts/** – Clean R script for reproducibility
- **notebooks/** – RMarkdown with full analysis, plots, and narrative
- **results/** – Saved figures, model summaries, and performance metrics

---

## 🔧 Technologies Used
- **R**: `tidyverse`, `ggplot2`, `lmtest`, `MASS`, `car`, `Metrics`
- **Statistical Methods**: Multiple Linear Regression, Robust Regression, ANOVA, Box-Cox Transformation
- **Model Evaluation**: MAE, RMSE, Adjusted R², AIC, Residual Diagnostics

---

## 📊 Key Insights
- **Final Model** achieved:
  - MAE: **19.11**
  - RMSE: **23.51**
  - Adjusted R²: **0.2017**
  - Lowest AIC among all models → **best model fit**
- Significant predictors: **Age, BMI, Insulin, BloodPressure**
- DiabetesPedigreeFunction was found **statistically insignificant**.

---

## 📈 Visualization Samples
<p align="center">
  <img src="results/figures/histogram_glucose.png" width="45%">
  <img src="results/figures/correlation_heatmap.png" width="45%">
</p>

---

## 🚀 How to Run
```r
# Clone the repository
git clone https://github.com/<your-username>/regression-analysis-glucose.git
cd regression-analysis-glucose

# Install dependencies
install.packages(c("tidyverse","ggplot2","lmtest","MASS","car","Metrics"))

# Run the analysis
source("scripts/regression_analysis.R")
