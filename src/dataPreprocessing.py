"""
M5 Walmart Data Preprocessing - IMPROVED
=========================================
Handles zeros/NaNs properly and creates forecasting features
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_data():
    """Load all M5 datasets."""
    print("Loading M5 datasets...")

    sales = pd.read_csv("../data/sales_train_evaluation.csv")
    calendar = pd.read_csv("../data/calendar.csv")
    prices = pd.read_csv("../data/sell_prices.csv")

    print(f"Sales: {sales.shape}, Calendar: {calendar.shape}, Price: {prices.shape}")

    return sales, calendar, prices


def melt_sales_data(sales: pd.DataFrame):
    """Melt sales data to long format."""
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales_cols = [col for col in sales.columns if col.startswith('d_')]

    sales_long = pd.melt(
        sales,
        id_vars=id_cols,
        value_vars=sales_cols,
        var_name='d',
        value_name='sales'
    )

    print(f"Transformed sales: {sales_long.shape}")
    return sales_long


def check_data_quality(df):
    """Check for data quality issues."""
    print("\n" + "="*70)
    print("DATA QUALITY CHECKS")
    print("="*70)

    # Sales analysis
    print(f"\nSales column:")
    print(f"  Zeros: {(df['sales'] == 0).sum():,} ({(df['sales'] == 0).mean()*100:.1f}%)")
    print(f"  NaNs: {df['sales'].isna().sum():,} ({df['sales'].isna().mean()*100:.1f}%)")
    print(f"  Negative: {(df['sales'] < 0).sum():,}")
    print(f"  Mean: {df['sales'].mean():.2f}, Median: {df['sales'].median():.2f}")

    # Price analysis
    if 'sell_price' in df.columns:
        print(f"\nPrice column:")
        print(f"  NaNs: {df['sell_price'].isna().sum():,} ({df['sell_price'].isna().mean()*100:.1f}%)")
        print(f"  Zeros: {(df['sell_price'] == 0).sum():,}")

    return df


def handle_missing_values(df):
    """
    Handle missing values appropriately for retail forecasting.

    Strategy:
    - Sales NaNs -> 0 (shouldn't happen in M5, but just in case)
    - Price NaNs -> Forward fill within item-store, then backward fill
    - Remaining price NaNs -> median price for that item
    """
    print("\n" + "="*70)
    print("HANDLING MISSING VALUES")
    print("="*70)

    # Sales: NaN -> 0 (no sales that day)
    before_sales_nan = df['sales'].isna().sum()
    df['sales'] = df['sales'].fillna(0)
    print(f"Sales NaNs filled: {before_sales_nan:,} -> 0")

    # Price: More sophisticated handling
    if 'sell_price' in df.columns:
        before_price_nan = df['sell_price'].isna().sum()

        # Forward/backward fill within each item-store combination
        df['sell_price'] = df.groupby(['item_id', 'store_id'])['sell_price'].transform(
            lambda x: x.fillna(method='ffill').fillna(method='bfill')
        )

        # Fill remaining with item median
        item_median_price = df.groupby('item_id')['sell_price'].transform('median')
        df['sell_price'] = df['sell_price'].fillna(item_median_price)

        after_price_nan = df['sell_price'].isna().sum()
        print(f"Price NaNs: {before_price_nan:,} -> {after_price_nan:,}")

    return df


def merge_calendar_data(sales_long, calendar):
    """Merge sales data with calendar data."""
    df = sales_long.merge(calendar, on='d', how='left')
    df['date'] = pd.to_datetime(df['date'])

    print(f"Merged sales and calendar data: {df.shape}")
    return df


def merge_price_data(df, prices):
    """Merge with price data."""
    print(f"Merging price data...")

    df = df.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    print(f"Merged price data: {df.shape}")

    return df


def create_time_features(df):
    """Create time features from date."""
    print(f"\nCreating time features...")

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['week_of_year'] = df['date'].dt.isocalendar().week

    # Additional useful features
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['date'].dt.is_month_end.astype(int)

    # Use SNAP and event columns from calendar
    df['snap_available'] = 0
    if 'snap_CA' in df.columns:
        df.loc[df['state_id'] == 'CA', 'snap_available'] = df['snap_CA']
    if 'snap_TX' in df.columns:
        df.loc[df['state_id'] == 'TX', 'snap_available'] = df['snap_TX']
    if 'snap_WI' in df.columns:
        df.loc[df['state_id'] == 'WI', 'snap_available'] = df['snap_WI']

    df['is_event'] = (~df['event_name_1'].isna()).astype(int)

    print("Created: year, month, day, day_of_week, quarter, week_of_year, is_weekend, events, SNAP")

    return df


def create_lag_features(df, lags=[7, 14, 21, 28]):
    """
    Create lag features for time series forecasting.
    Essential for capturing trends and seasonality.
    """
    print(f"\nCreating lag features: {lags}...")

    # Sort by item, store, and date
    df = df.sort_values(['item_id', 'store_id', 'date'])

    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['item_id', 'store_id'])['sales'].shift(lag)

    return df


def create_rolling_features(df, windows=[7, 14, 28]):
    """
    Create rolling statistics.
    Captures recent trends better than simple lags.
    """
    print(f"Creating rolling features: {windows}...")

    for window in windows:
        # Rolling mean
        df[f'sales_rolling_mean_{window}'] = df.groupby(['item_id', 'store_id'])['sales'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

        # Rolling std (volatility)
        df[f'sales_rolling_std_{window}'] = df.groupby(['item_id', 'store_id'])['sales'].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std()
        )

    return df


def create_price_features(df):
    """Create price-based features."""
    print("Creating price features...")

    # Price change
    df['price_change'] = df.groupby(['item_id', 'store_id'])['sell_price'].diff()
    df['price_change_pct'] = df.groupby(['item_id', 'store_id'])['sell_price'].pct_change()

    # Price relative to item average
    df['price_vs_avg'] = df['sell_price'] / df.groupby('item_id')['sell_price'].transform('mean')

    # Days since price change
    df['price_changed'] = (df['price_change'] != 0).astype(int)
    df['days_since_price_change'] = df.groupby(['item_id', 'store_id'])['price_changed'].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount()
    )

    return df


def sample_data_smart(df, n_items=500):
    """
    Smart sampling: Keep complete time series for selected items.
    Better than random sampling for forecasting.
    """
    print(f"\nSmart sampling: {n_items} random item-store combinations...")

    # Get all unique item-store combinations
    unique_combos = df[['item_id', 'store_id']].drop_duplicates()

    # Sample combinations
    sampled_combos = unique_combos.sample(n=min(n_items, len(unique_combos)), random_state=42)

    # Filter data to keep only sampled combinations
    df_sample = df.merge(sampled_combos, on=['item_id', 'store_id'], how='inner')
    df_sample = df_sample.sort_values(['item_id', 'store_id', 'date']).reset_index(drop=True)

    print(f"Sample size: {df_sample.shape}")
    print(f"Unique items: {df_sample['item_id'].nunique()}")
    print(f"Unique stores: {df_sample['store_id'].nunique()}")
    print(f"Date range: {df_sample['date'].min()} to {df_sample['date'].max()}")

    return df_sample


def reduce_memory_usage(df):
    """Optimize memory usage by downcasting numeric types."""
    print("\nOptimizing memory usage...")

    start_mem = df.memory_usage().sum() / 1024**2

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')

    return df


def main():
    """Main data preparation pipeline."""
    print("=" * 70)
    print("M5 WALMART DATA PREPARATION - IMPROVED")
    print("=" * 70)

    # Load data
    sales, calendar, prices = load_data()

    # Transform
    sales_long = melt_sales_data(sales)
    df = merge_calendar_data(sales_long, calendar)
    df = merge_price_data(df, prices)

    # Data quality
    df = check_data_quality(df)
    df = handle_missing_values(df)

    # Features
    df = create_time_features(df)

    # Sample BEFORE creating lags (to avoid memory issues)
    df_sample = sample_data_smart(df, n_items=500)

    # Create lag and rolling features
    df_sample = create_lag_features(df_sample, lags=[7, 14, 21, 28])
    df_sample = create_rolling_features(df_sample, windows=[7, 14, 28])
    df_sample = create_price_features(df_sample)

    # Final data quality check
    df_sample = check_data_quality(df_sample)

    # Optimize memory
    df_sample = reduce_memory_usage(df_sample)

    # Save
    print(f"\nSaving processed data...")
    output_path = Path("../data/m5_processed_features.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(output_path, index=False)
    print(f"Saved to: {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total rows: {len(df_sample):,}")
    print(f"Total columns: {len(df_sample.columns)}")
    print(f"Date range: {df_sample['date'].min()} to {df_sample['date'].max()}")
    print(f"Unique items: {df_sample['item_id'].nunique()}")
    print(f"Unique stores: {df_sample['store_id'].nunique()}")
    print(f"\nFeature columns:")
    print(df_sample.columns.tolist())

    print("\nData preparation complete!")
    print("Ready for modeling!")


if __name__ == "__main__":
    main()