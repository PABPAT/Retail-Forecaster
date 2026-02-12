"""
Retail Sales Data Exploration
==============================
Explore and visualize the Store Item Demand dataset.

This script helps you understand:
- Data structure and types
- Sales patterns over time
- Distribution across stores and items
- Seasonal trends

Author: Poorna Abhijith Patel
Date: January 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_data(filepath='data/train.csv'):
    """
    Load the retail sales dataset.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
    """
    print(f"Loading data from {filepath}...")
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df


def basic_info(df):
    """Display basic information about the dataset."""
    print("\n" + "="*70)
    print("DATASET OVERVIEW")
    print("="*70)
    
    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    print("Column Information:")
    print(df.info())
    
    print("Statistical Summary:")
    print(df.describe())
    
    print("First few rows:")
    print(df.head(10))
    
    print(" Date Range:")
    print(f"   Start: {df['date'].min()}")
    print(f"   End: {df['date'].max()}")
    print(f"   Duration: {(df['date'].max() - df['date'].min()).days} days")
    
    print("Unique Values:")
    print(f"   Stores: {df['store'].nunique()}")
    print(f"   Items: {df['item'].nunique()}")
    print(f"   Total combinations: {df['store'].nunique() * df['item'].nunique()}")


def plot_sales_over_time(df):
    """Plot total sales over time."""
    print("Plotting sales over time...")
    
    # Aggregate sales by date
    sales_by_date = df.groupby('date')['sales'].sum().reset_index()
    
    plt.figure(figsize=(14, 6))
    plt.plot(sales_by_date['date'], sales_by_date['sales'], linewidth=0.8)
    plt.title('Total Sales Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Total Sales', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Plot displayed")


def plot_sales_by_store(df):
    """Compare sales across stores."""
    print("Plotting sales by store...")
    
    store_sales = df.groupby('store')['sales'].sum().sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    store_sales.plot(kind='bar', color='steelblue')
    plt.title('Total Sales by Store', fontsize=16, fontweight='bold')
    plt.xlabel('Store', fontsize=12)
    plt.ylabel('Total Sales', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()
    
    print("Plot displayed")


def plot_sales_by_item(df, top_n=10):
    """Show top selling items."""
    print(f"Plotting top {top_n} items by sales...")
    
    item_sales = df.groupby('item')['sales'].sum().sort_values(ascending=False).head(top_n)
    
    plt.figure(figsize=(12, 6))
    item_sales.plot(kind='barh', color='coral')
    plt.title(f'Top {top_n} Items by Total Sales', fontsize=16, fontweight='bold')
    plt.xlabel('Total Sales', fontsize=12)
    plt.ylabel('Item', fontsize=12)
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    print("Plot displayed")


def plot_monthly_trends(df):
    """Analyze monthly sales patterns."""
    print("Plotting monthly trends...")
    
    df['year_month'] = df['date'].dt.to_period('M')
    monthly_sales = df.groupby('year_month')['sales'].sum()
    
    plt.figure(figsize=(14, 6))
    monthly_sales.plot(linewidth=2, marker='o', markersize=4)
    plt.title('Monthly Sales Trend', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Sales', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Plot displayed")


def plot_seasonal_patterns(df):
    """Analyze seasonal patterns in sales."""
    print("Plotting seasonal patterns...")
    
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Monthly pattern
    monthly_avg = df.groupby('month')['sales'].mean().reset_index()
    axes[0].bar(monthly_avg['month'], monthly_avg['sales'], color='skyblue')
    axes[0].set_title('Average Sales by Month', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Month', fontsize=11)
    axes[0].set_ylabel('Average Sales', fontsize=11)
    axes[0].set_xticks(range(1, 13))
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Day of week pattern
    dow_avg = df.groupby('day_of_week')['sales'].mean().reset_index()
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1].bar(range(7), dow_avg['sales'], color='lightcoral')
    axes[1].set_title('Average Sales by Day of Week', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Day of Week', fontsize=11)
    axes[1].set_ylabel('Average Sales', fontsize=11)
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(days)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print("Plots displayed")


def analyze_sales_distribution(df):
    """Analyze the distribution of sales values."""
    print("Analyzing sales distribution...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(df['sales'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_title('Sales Distribution', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Sales', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Box plot
    axes[1].boxplot(df['sales'], vert=True)
    axes[1].set_title('Sales Box Plot', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Sales', fontsize=11)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.show()
    
    print(f"Sales Statistics:")
    print(f"   Mean: {df['sales'].mean():.2f}")
    print(f"   Median: {df['sales'].median():.2f}")
    print(f"   Std Dev: {df['sales'].std():.2f}")
    print(f"   Min: {df['sales'].min():.2f}")
    print(f"   Max: {df['sales'].max():.2f}")


def main():
    """
    Main exploration workflow.
    
    Run this to get a complete overview of your dataset!
    """
    print("\n" + "="*70)
    print("RETAIL SALES DATA EXPLORATION")
    print("="*70)
    
    # Load data
    df = load_data('data/train.csv')
    
    # Display basic info
    basic_info(df)
    
    # Visualizations
    plot_sales_over_time(df)
    plot_sales_by_store(df)
    plot_sales_by_item(df, top_n=15)
    plot_monthly_trends(df)
    plot_seasonal_patterns(df)
    analyze_sales_distribution(df)
    
    print("\n" + "="*70)
    print("EXPLORATION COMPLETE!")
    print("="*70)
    print("Next steps:")
    print("   1. Feature engineering - create lag features, rolling averages")
    print("   2. Train baseline model - simple moving average")
    print("   3. Build ML models - XGBoost, LightGBM")
    print("   4. Evaluate and compare models")
    print("   5. Make predictions for next 3 months!")


if __name__ == "__main__":
    main()
