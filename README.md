# Synthetic House Price Prediction Pipeline

This repository demonstrates an end-to-end Machine Learning pipeline that covers synthetic data generation, rigorous data cleaning (ETL), handling missing values, and predictive modeling using Random Forest Regressor.

## 🚀 Key Features
* **Automated Data Generation:** Generates a synthetic dataset of 500 houses with randomized yet structured features (Year Built, Rooms, Square Footage with text noise, Condition, and Price).
* **Advanced Data Cleaning & Transformation:**
  * Extracts numeric values from dirty string formats (e.g., converting `"1200 sqft"` to `1200`).
  * Handles missing/null values (`NaN`) dynamically using statistical metrics (`median` for sizes, `mean` for build years).
  * Implements One-Hot Encoding with `drop_first=True` to eliminate the dummy variable trap.
* **Predictive Modeling:** Trains a Robust `RandomForestRegressor` to map complex non-linear relationships and output real-world price estimates.
* **Performance Evaluation:** Validates accuracy against unseen data utilizing the Mean Absolute Error (MAE) metric.

## 🛠️ Tech Stack
* **Python 3**
* **Pandas** & **NumPy** (Data generation, cleaning, and preprocessing)
* **Scikit-Learn** (Data splitting, Random Forest Modeling, and Metrics)

## 📁 Pipeline Flow & Implementation
The single-script execution performs the following processes:
1. **Simulation:** Programmatically builds `dirty_house_prices.csv` with artificial noise and intentional missing blocks (`NaN`).
2. **Parsing & Formatting:** Strips string metrics and coerces object columns to float/numeric variables.
3. **Imputation:** Measures and injects structural means/medians into vacant rows to retain sample size without data leakage.
4. **Encoding & Partitioning:** Splits features from the target variable and subsets data into an 80% training set and a 20% validation set.
5. **Execution:** Fits the Regressor model and computes the ultimate deviation (MAE).

## 📊 Evaluation Metrics
* **Model Used:** `RandomForestRegressor(random_state=42)`
* **Observed Mean Absolute Error (MAE):** ~\$17,213.86 (A highly optimal result considering the \$20,000 artificial variance/noise injected during generation).

## ⚙️ Quick Start
1. Clone this repository to your local environment.
2. Run the main script to trigger both data simulation and model processing:
   ```bash
   python main.py
