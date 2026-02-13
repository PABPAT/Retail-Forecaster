# Retail Forecaster

A machine learning solution for retail inventory optimization using historical sales data and advanced forecasting techniques.

## Overview

This project implements predictive models to optimize inventory levels for retail operations. By analyzing historical sales patterns, seasonality, and trends, the model helps reduce stockouts while minimizing excess inventory costs.

## Features

- **Automated Data Pipeline**: Download and preprocess retail datasets
- **Feature Engineering**: Extract temporal patterns, trends, and seasonal components
- **Multiple Model Support**: Leverages scikit-learn and XGBoost/LightGBM for robust predictions
- **Inventory Optimization**: Translate forecasts into actionable inventory recommendations

## Project Structure

```
Retail-Forecaster/
├── main.py                      # Main execution script
├── download_datasets.py         # Dataset acquisition utilities
├── explore_data.py             # Exploratory data analysis
├── test_setup.py               # Environment validation
├── requirements.txt            # Python dependencies
├── src/
│   ├── dataPreprocessing.py    # Data cleaning and preprocessing
│   ├── featureEngineering.py   # Feature extraction and transformation
│   └── train_model.py          # Model training and evaluation
└── data/                       # Data directory (not tracked in git)
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/PABPAT/Retail-Forecaster.git
cd Retail-Forecaster
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Verify installation:
```bash
python test_setup.py
```

## Data Sources & Preprocessing

### Data Acquisition
The model uses retail transaction data containing:
- Historical sales records
- Product information
- Timestamps and seasonal indicators
- Store/location data

Run the data download script:
```bash
python download_datasets.py
```

### Preprocessing Steps
The data preprocessing pipeline (`src/dataPreprocessing.py`) handles:
- Missing value imputation
- Outlier detection and treatment
- Data type conversions
- Date/time parsing
- Data validation and quality checks

### Feature Engineering
The feature engineering module (`src/featureEngineering.py`) creates:
- **Temporal Features**: Day of week, month, quarter, holidays
- **Lag Features**: Historical sales at various time windows
- **Rolling Statistics**: Moving averages and standard deviations
- **Trend Components**: Long-term growth patterns
- **Seasonal Decomposition**: Cyclical patterns and seasonality indices

## How to Run

### 1. Explore the Data
```bash
python explore_data.py
```
Generates visualizations and statistics about the dataset.

### 2. Train the Model
```bash
python main.py
```
This will:
- Load and preprocess the data
- Engineer features
- Train multiple models (scikit-learn, XGBoost/LightGBM)
- Evaluate performance
- Generate inventory recommendations

### 3. Custom Configuration
Modify parameters in `main.py` or create a config file for:
- Model hyperparameters
- Feature selection
- Training/validation split ratios
- Inventory optimization constraints

## Technical Details & Architecture

### Machine Learning Pipeline

1. **Data Ingestion**: Raw retail transaction data
2. **Preprocessing**: Cleaning, validation, and transformation
3. **Feature Engineering**: Time-series and domain-specific features
4. **Model Training**: Ensemble approach using:
   - scikit-learn (Random Forest, Gradient Boosting)
   - XGBoost/LightGBM (Gradient boosting variants)
5. **Model Evaluation**: Cross-validation and time-series specific metrics
6. **Inventory Optimization**: Convert predictions to stock recommendations

### Models Used

- **scikit-learn**: Baseline models and ensemble methods
- **XGBoost**: High-performance gradient boosting for structured data
- **LightGBM**: Fast, distributed gradient boosting framework

### Evaluation Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Custom inventory metrics (stockout rate, holding costs)

## Results

The model outputs:
- Sales forecasts for specified time horizons
- Recommended inventory levels by product/location
- Confidence intervals for predictions
- Feature importance rankings

## Dependencies

Key libraries (see `requirements.txt` for full list):
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib/seaborn (for visualization)

## Future Improvements

- [ ] Add real-time prediction API
- [ ] Implement deep learning models (LSTM, Transformer)
- [ ] Multi-location inventory optimization
- [ ] Incorporate external factors (weather, promotions)
- [ ] Dashboard for visualization and monitoring

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is available for educational and research purposes.

## Contact

For questions or suggestions, please open an issue on GitHub.

---

**Note**: The `data/` directory is excluded from version control due to size. Use `download_datasets.py` to acquire the necessary datasets.
## Metrics:
Version 1 dt 02/12/2025
MAE : 0.85
RMSE: 2.34
MAPE : 58.02