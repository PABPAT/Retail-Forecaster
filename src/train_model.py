"""This script is used to train the model"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

def load_and_prepare_data():
    """Load features and remove rows with missing values"""
    df = pd.read_csv('../data/m5_features_file.csv')
    print(f"Shape before: {df.shape}")

    df = df.fillna(0)

    print(f"Shape after dropping na: {df.shape}")

    return df

def split_data(df):
    """Split data into train and test"""
    exclude_cols = ['date', 'sales', 'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
                'd', 'weekday', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    feature_col = [col for col in df.columns if col not in exclude_cols]

    x = df[feature_col]
    y = df['sales']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    return x_train, x_test, y_train, y_test

def train_xgboost(x_train, y_train):
    """Train the XGBoost model"""
    print("Training XGBoost model...")

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(x_train, y_train)

    feature_importance = model.feature_importances_
    feature_names = x_train.columns

    top_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 features:")
    for name, importance in top_features:
        print(f"{name}: {importance:.4f}")
    print("Trained XGBoost model")
    return model

def evaluate_model(model, x_test, y_test):
    """Evaluate XGBoost model"""
    print("Evaluating XGBoost model...")
    y_pred = model.predict(x_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mask = y_test > 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100

    print(f"\nMAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}")

    return y_pred

def main():
    print("=" * 70)
    print("Model Training")
    print("=" * 70)

    df = load_and_prepare_data()
    x_train, x_test, y_train, y_test = split_data(df)
    model = train_xgboost(x_train, y_train)
    y_pred = evaluate_model(model, x_test, y_test)

if __name__ == "__main__":
    main()
