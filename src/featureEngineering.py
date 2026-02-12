"""
Feature Engineering for M5 Walmart Data
========================
Create machine learning features from preprocessed data.

This file will create:
1. Lag Features (Past sales)
2. Rolling Averages. (7-day, 30-day)
3. Price change features.
"""
import pandas as pd
import numpy as np


def load_preprocessed_data():
    """Load the preprocessed data and return it as a pandas dataframe"""
    processed_data = pd.read_csv("../data/m5_processed_sample.csv")
    print(f"Preprocessed data: {processed_data.shape}")

    return processed_data


def create_lag_features(df, lags=[1, 7, 14, 28, 30]):
    print("Creating lag features...")

    df = df.sort_values(['item_id', 'store_id', 'date']).reset_index(drop=True)

    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['item_id', 'store_id'])['sales'].shift(lag)

    return df


def create_rolling_features(df):
    """Create the rolling features [mean and SD] from past sales data"""
    print("Creating rolling features with mean and standard deviation...")

    grouped = df.groupby(['item_id', 'store_id'])['sales'].shift(1)
    for w in [7, 14, 28, 30]:
        df[f'sales_rolling_{w}_mean'] = (
            grouped
            .rolling(w)
            .mean()
        )
        df[f'sales_rolling_{w}_std'] = (
            grouped
            .rolling(w)
            .std()
        )
    return df


def create_price_change_features(df):
    """Create the price change features from past sales data"""
    print("Creating price change features...")
    df = df.sort_values(['item_id', 'store_id', 'date'])
    grouped = df.groupby(['item_id', 'store_id'])['sell_price'].shift(1)

    for lag in [1, 7, 14, 28, 30]:
        df[f"price_lag_{lag}"] = df.groupby(['item_id', 'store_id'])['sell_price'].shift(lag)
        df[f"price_return_{lag}"] = df['sell_price'] / df[f'price_lag_{lag}'] - 1
        df[f"price_rolling_{lag}_mean"] = grouped.rolling(lag).mean()
        df[f"price_rolling_{lag}_std"] = grouped.rolling(lag).std()
        df[f"price_vs_{lag}d_mean"] = df['sell_price'] / df[f'price_rolling_{lag}_mean']

    return df


def main():
    """Main pipeline"""
    print("=" * 70)
    print("Feature Engineering")
    print("=" * 70)

    df = load_preprocessed_data()
    df = create_lag_features(df)
    df = create_rolling_features(df)
    df = create_price_change_features(df)

    df.to_csv("../data/m5_features_file.csv", index=False)
    print("Saved to ../data/m5_features_file.csv")

    return df


if __name__ == "__main__":
    main()
