# 📈 Bitcoin Price Forecasting using ARIMA Models

This project performs **time series analysis and forecasting** of the **Bitcoin Index (USD)**  
from **August 2011 to January 2025** using **ARIMA models**.  

It demonstrates a complete workflow — **descriptive analysis, transformations, stationarity tests, model fitting, selection (AIC/BIC), and accuracy evaluation** — making it a great example of applied statistical modeling.

---

## 📂 Project Structure
- **data/** – Dataset used for analysis (or link to source)
- **scripts/** – Clean R script with reproducible code
- **Time_Series_Analysis_Bitcoin_Index_Report.pdf** – Final written report

---

## 🔧 Technologies & Methods
- **R**: `forecast`, `TSA`, `tseries`, `lmtest`, `zoo`, `tidyverse`
- **Techniques**:
  - Box-Cox Transformation
  - Differencing for Stationarity
  - ARIMA model specification with ACF, PACF, EACF, BIC
  - Parameter estimation using ML, CSS, CSS-ML
  - Model selection via AIC, BIC, RMSE, MAE, MAPE, MASE

---

## 📊 Key Results
- **Best Model**: `ARIMA(2,1,2)`  
- **Why?**
  - Lowest **AIC (5477.27)** among candidate models
  - Statistically significant AR & MA terms across ML, CSS, CSS-ML
  - Best combination of **RMSE**, **MAE**, and **MAPE**

---

## 📈 Visualization Samples
<p align="center">
  <img src="results/figures/acf_plot.png" width="45%">
  <img src="results/figures/arima_forecast.png" width="45%">
</p>

---

## 🚀 How to Run
```r
# Clone the repository
git clone https://github.com/<your-username>/bitcoin-time-series-forecasting.git
cd bitcoin-time-series-forecasting

# Install dependencies
install.packages(c("forecast","TSA","tseries","zoo","lmtest","tidyverse"))

# Run the main script
source("scripts/bitcoin_arima_analysis.R")
