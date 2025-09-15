# 🌲 Yellowstone Visitor Forecasting using Time Series Analysis

This project uses **Trend Models** (Linear, Quadratic, Cosine, Seasonal Dummy) and **SARIMA** to forecast monthly visitors to **Yellowstone National Park**.  
It demonstrates end-to-end **time series analysis** — from data cleaning, decomposition, model selection (AIC/BIC), to 10-month forecasting.

---

## 📂 Repository Structure
- **data/** – Raw dataset (or a link to National Park Service IRMA data portal)
- **notebooks/** – RMarkdown file with complete analysis and plots
- **scripts/** – Clean R script for reproducibility
- **results/** – Saved figures, forecasts, model selection metrics
- **BHEMAN_Report.pdf** – Full project report

---

## 🔧 Technologies Used
- **R**: `forecast`, `TSA`, `FitAR`, `ggplot2`, `tseries`, `lmtest`
- **Models**: Trend Models, Seasonal Dummy Regression, SARIMA
- **Metrics**: AIC, BIC, RMSE, MAE, Ljung-Box test, Shapiro-Wilk test

---

## 📊 Key Findings
- **Seasonal Dummy Regression** explained **93.5% of variance**, best for deterministic seasonal pattern.
- **SARIMA(1,1,2)×(1,1,1)12** chosen as final stochastic model — lowest AIC/BIC, best residual diagnostics.
- Forecast shows:
  - **Sharp summer peak** (July ~939k visitors)
  - **Very low winter months**
  - Clear seasonality and gradual year-on-year growth

---

## 📈 Visualization Samples
<p align="center">
  <img src="results/figures/seasonal_model_fit.png" width="45%">
  <img src="results/figures/sarima_forecast.png" width="45%">
</p>

---

## 🚀 How to Run
```r
# Clone repository
git clone https://github.com/<your-username>/yellowstone-time-series-forecasting.git
cd yellowstone-time-series-forecasting

# Install dependencies
install.packages(c("forecast","TSA","FitAR","ggplot2","tseries","lmtest"))

# Run script
source("scripts/sarima_forecast.R")
