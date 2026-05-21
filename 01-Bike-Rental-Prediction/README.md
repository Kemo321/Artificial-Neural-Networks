# Project 01: Bike Rental Prediction (Regression)

## Task Description
The goal of this project is to predict the total number of rented vehicles ($cnt$) for a specific hour and day based on historical, seasonal, and weather data. The target variable is also provided as a breakdown of registered (`registered`) and casual (`casual`) users, allowing for multi-stage modeling strategies.

## Dataset
* **Source:** University-provided historical dataset (`data.csv`).
* **Features:** Weather conditions, temperature, humidity, wind speed, holiday/working day indicators, and hourly timestamps.
* **Target:** Sum of casual and registered rentals ($cnt$).

## Technical Approach
* **Model:** Deep Multi-Layer Perceptron (MLP) built from scratch using **PyTorch**.
* **Pre-processing:** Continuous variables were normalized using `StandardScaler`. Categorical features (hours, seasons, weather types) were encoded using One-Hot Encoding.
* **Evaluation Metric:** Root Mean Squared Logarithmic Error (RMSLE).

$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i + 1))^2}$$

* **Optimization:** Mean Squared Error (MSE) loss was minimized using the Adam optimizer, with exponential targets to align with the RMSLE metric.