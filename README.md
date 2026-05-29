# Air Passenger Demand Forecasting using Deep Learning and Attention Mechanisms

## Overview

This project presents an end-to-end time series forecasting pipeline for predicting monthly airline passenger demand using advanced deep learning techniques. The workflow combines classical time series analysis, extensive feature engineering, recurrent neural networks, and attention mechanisms to capture temporal dependencies and seasonal patterns in passenger traffic.

Unlike traditional forecasting approaches that rely solely on historical observations, this project incorporates engineered temporal features, statistical diagnostics, and custom attention layers to improve forecasting performance and model interpretability.



## Problem Statement

Accurate passenger demand forecasting plays a crucial role in airline capacity planning, route optimization, resource allocation, and revenue management.

The objective of this project is to forecast future monthly passenger counts using historical airline traffic data while evaluating the effectiveness of different deep learning architectures and attention-based models.



## Dataset

The project uses the Air Passengers dataset, a widely recognized benchmark in time series forecasting.

### Dataset Characteristics

* Monthly passenger counts
* Time Period: 1949–1960
* Strong trend component
* Pronounced seasonality
* Non-stationary behavior



## Project Workflow

### 1. Exploratory Data Analysis

Initial data exploration was performed to understand long-term trends, seasonality, and distributional characteristics.

Key analyses include:

* Time series visualization
* Trend identification
* Seasonal pattern analysis
* Distribution analysis
* Missing value inspection



### 2. Statistical Time Series Diagnostics

To understand the underlying characteristics of the series, several statistical techniques were applied:

* Rolling Mean Analysis
* Rolling Standard Deviation Analysis
* Augmented Dickey-Fuller (ADF) Test
* Seasonal Decomposition
* Autocorrelation Function (ACF)
* Partial Autocorrelation Function (PACF)

These analyses help evaluate stationarity and temporal dependencies before model development.



### 3. Feature Engineering

A comprehensive set of temporal features was created to improve forecasting performance.

#### Lag Features

* Lag 1
* Lag 3
* Lag 6
* Lag 12

#### Rolling Statistics

* Rolling Mean (3)
* Rolling Mean (6)
* Rolling Mean (12)
* Rolling Standard Deviation (12)

#### Exponential Moving Averages

* EMA 3
* EMA 6
* EMA 12

#### Cyclical Seasonal Encoding

To preserve cyclical information:

* Month Sine Encoding
* Month Cosine Encoding

This transforms the problem from a simple univariate forecasting task into a richer multivariate forecasting problem.



### 4. Data Preparation

The dataset was prepared using a time-aware preprocessing pipeline:

* Chronological train-validation-test split
* Feature scaling using StandardScaler
* Target scaling
* Sequence generation for recurrent models
* Time-series window creation

#### Lookback Window

```text
12 Months
```

#### Input Shape

```text
(samples, timesteps, features)
```



## Model Architectures

### Baseline Model

A Naive Forecast model was implemented to establish a benchmark for comparison.



### LSTM Model

Long Short-Term Memory (LSTM) networks were used to capture long-range temporal dependencies and sequential patterns within the data.

---

### GRU Model

Gated Recurrent Units (GRU) were evaluated as a computationally efficient alternative to LSTMs.



### Attention-Based Models

Two advanced architectures were developed:

* LSTM + Attention
* GRU + Attention

These models incorporate feature-level and temporal attention mechanisms to dynamically identify the most informative features and historical observations.



## Custom Attention Mechanisms

### Feature Attention

Feature Attention enables the model to learn which engineered features contribute most to the prediction.

Examples include:

* Lag Features
* Rolling Statistics
* Exponential Moving Averages
* Seasonal Encodings



### Temporal Attention

Temporal Attention allows the model to focus on the most relevant historical timesteps instead of relying solely on the final hidden state.

Benefits include:

* Improved interpretability
* Better long-term dependency modeling
* Enhanced sequence representation



## Training Pipeline

The project includes a modular deep learning training framework featuring:

* Custom Dataset Classes
* PyTorch DataLoaders
* Early Stopping
* Model Checkpointing
* Device Management (CPU/GPU)
* Reproducible Training Workflow


## Evaluation Metrics

Model performance was evaluated using:

* RMSE (Root Mean Squared Error)
* MAE (Mean Absolute Error)
* MAPE (Mean Absolute Percentage Error)



## Results

The models were evaluated on the test dataset using RMSE, MAE, MAPE, and training time.

### Model Performance Comparison

| Model            | RMSE       | MAE        | MAPE (%)  | Training Time (s) |
| ---------------- | ---------- | ---------- | --------- | ----------------- |
| **GRU**          | **0.9253** | **0.8233** | **24.45** | 0.889             |
| Naive Forecast   | 1.2815     | 1.1509     | 38.73     | 0.000             |
| GRU + Attention  | 1.5693     | 1.3156     | 36.06     | 0.617             |
| LSTM             | 1.6882     | 1.4388     | 39.94     | 0.651             |
| LSTM + Attention | 1.9036     | 1.6769     | 47.68     | 0.753             |


### Key Findings

* The GRU model achieved the lowest forecasting error across all evaluation metrics.
* GRU outperformed both LSTM and attention-enhanced architectures on this dataset.
* Attention mechanisms did not improve performance significantly for this relatively small time series dataset.
* Simpler recurrent architectures can outperform more complex models when data volume is limited.
* The Naive Forecast baseline was successfully surpassed by the deep learning models.


## Visualization & Diagnostics

Comprehensive diagnostic analysis was performed using:

* Training Loss Curves
* Validation Loss Curves
* Actual vs Predicted Forecasts
* Residual Analysis
* Residual Distribution Analysis
* Model Performance Comparison Charts

These visualizations help assess model stability, forecasting quality, and error characteristics.



## Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* Statsmodels
* PyTorch



## Project Structure

```text
air-passenger-demand-forecasting/
│
├── air-passenger-notebook.ipynb
├── requirements.txt
├── README.md
│
├── data/
│   └── AirPassengers.csv
└── outputs/
    ├── plots/
```


## Future Improvements

Potential enhancements include:

* Transformer-based forecasting models
* Temporal Fusion Transformers (TFT)
* Hyperparameter optimization using Optuna
* Probabilistic forecasting
* FastAPI deployment
* Automated retraining pipelines
* MLOps integration for production deployment



## Author

**Siddharth Jain**


