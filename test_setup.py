#!/usr/bin/env python3
"""
PyCharm Environment Setup Test
================================
Run this script to verify your PyCharm Python environment is configured correctly.

How to run in PyCharm:
1. Right-click this file in Project view
2. Select "Run 'test_setup'"
3. Check the Run window for results

Author: Retail Forecasting Project
Date: January 2026
"""

import sys


def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def print_success(text):
    """Print success message."""
    print(f"{text}")


def print_error(text):
    """Print error message."""
    print(f"{text}")


def print_warning(text):
    """Print warning message."""
    print(f"{text}")


def print_info(text):
    """Print info message."""
    print(f"{text}")


def test_python_version():
    """Check if Python version is compatible."""
    print_header("PYTHON VERSION CHECK")
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    print(f"Location: {sys.executable}")
    
    if version.major >= 3 and version.minor >= 7:
        print_success("Python version is compatible (3.7+)")
        return True
    else:
        print_error("Python 3.7 or higher required. Please upgrade.")
        return False


def test_required_packages():
    """Test if required packages are installed."""
    print_header("REQUIRED PACKAGES CHECK")
    
    required_packages = {
        'pandas': 'Data manipulation and analysis',
        'numpy': 'Numerical computing',
        'sklearn': 'Machine learning (scikit-learn)',
        'matplotlib': 'Data visualization',
    }
    
    all_installed = True
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print_success(f"{package:15s} - {description}")
        except ImportError:
            print_error(f"{package:15s} - NOT INSTALLED - {description}")
            all_installed = False
    
    return all_installed


def test_ml_packages():
    """Test if machine learning packages are installed."""
    print_header("MACHINE LEARNING PACKAGES CHECK")
    
    ml_packages = {
        'xgboost': 'XGBoost gradient boosting',
        'lightgbm': 'LightGBM gradient boosting',
        'seaborn': 'Statistical data visualization',
    }
    
    installed = []
    missing = []
    
    for package, description in ml_packages.items():
        try:
            __import__(package)
            print_success(f"{package:15s} - {description}")
            installed.append(package)
        except ImportError:
            print_warning(f"{package:15s} - Not installed - {description}")
            missing.append(package)
    
    return installed, missing


def test_data_packages():
    """Test if data download packages are installed."""
    print_header("DATA DOWNLOAD PACKAGES CHECK")
    
    data_packages = {
        'kaggle': 'Kaggle API for dataset download',
        'ucimlrepo': 'UCI ML Repository access',
    }
    
    installed = []
    missing = []
    
    for package, description in data_packages.items():
        try:
            __import__(package)
            print_success(f"{package:15s} - {description}")
            installed.append(package)
        except ImportError:
            print_warning(f"{package:15s} - Not installed - {description}")
            missing.append(package)
    
    return installed, missing


def test_pandas_functionality():
    """Test basic pandas functionality."""
    print_header("PANDAS FUNCTIONALITY TEST")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create sample dataset
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = {
            'date': dates,
            'store': np.random.choice(['Store_1', 'Store_2', 'Store_3'], 100),
            'item': np.random.choice(['Item_A', 'Item_B', 'Item_C'], 100),
            'sales': np.random.randint(10, 100, 100)
        }
        
        df = pd.DataFrame(data)
        
        print_success(f"Created sample DataFrame: {df.shape[0]} rows × {df.shape[1]} columns")
        print_info(f"Columns: {', '.join(df.columns)}")
        print_info(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        
        # Test aggregation
        avg_sales = df.groupby('item')['sales'].mean()
        print_info("Average sales by item:")
        for item, sales in avg_sales.items():
            print(f"         {item}: {sales:.2f}")
        
        print_success("Pandas is working correctly!")
        return True
        
    except Exception as e:
        print_error(f"Pandas test failed: {e}")
        return False


def check_pycharm_environment():
    """Check if running in PyCharm."""
    print_header("PYCHARM ENVIRONMENT CHECK")
    
    # Check if PyCharm-specific environment variables exist
    pycharm_indicators = [
        'PYCHARM_HOSTED',
        'JETBRAINS_REMOTE_RUN',
        'PYCHARM_MATPLOTLIB_INTERACTIVE'
    ]
    
    import os
    running_in_pycharm = any(os.getenv(var) for var in pycharm_indicators)
    
    if running_in_pycharm:
        print_success("Running in PyCharm IDE")
    else:
        print_info("Not detected as PyCharm (might be running in terminal)")
    
    print_info(f"Working directory: {os.getcwd()}")
    return True


def print_installation_help(missing_required, missing_ml, missing_data):
    """Print help for installing missing packages."""
    print_header("INSTALLATION INSTRUCTIONS")
    
    all_missing = list(set(missing_required + missing_ml + missing_data))
    
    if not all_missing:
        print_success("All packages installed! You're ready to go!")
        return
    
    print_warning("Some packages are missing. Install them using one of these methods:")
    
    print("Method 1: Install all at once (Recommended)")
    print("   In PyCharm Terminal:")
    print("   pip install -r requirements.txt")
    
    print("Method 2: Install individually")
    print("   In PyCharm Terminal:")
    print(f"   pip install {' '.join(all_missing)}")
    
    print("Method 3: Using PyCharm GUI")
    print("   1. File → Settings → Project → Python Interpreter")
    print("   2. Click the '+' button")
    print("   3. Search for each package and click 'Install Package'")
    
    if 'kaggle' in all_missing or 'ucimlrepo' in all_missing:
        print(" Note: kaggle and ucimlrepo are needed for downloading datasets")


def print_next_steps(all_working):
    """Print next steps based on test results."""
    print_header("NEXT STEPS")
    
    if all_working:
        print_success("Environment setup complete! 🎉")
        print(" You're ready to:")
        print("   1. Set up Kaggle API credentials (see PYCHARM_SETUP.md)")
        print("   2. Run download_datasets.py to get data")
        print("   3. Run explore_data.py to visualize the data")
        print("   4. Start building your forecasting model!")
    else:
        print_warning("Setup incomplete - install missing packages first")
        print(" Resources:")
        print("   • PYCHARM_SETUP.md - Complete setup guide")
        print("   • README.md - Project overview")
        print("   • requirements.txt - All required packages")


def main():
    """Run all tests and display results."""
    print("\n" + "=" * 70)
    print(" PYCHARM ENVIRONMENT SETUP TEST")
    print("=" * 70)
    
    # Run all tests
    python_ok = test_python_version()
    required_ok = test_required_packages()
    ml_installed, ml_missing = test_ml_packages()
    data_installed, data_missing = test_data_packages()
    pycharm_ok = check_pycharm_environment()
    
    # Test pandas if installed
    if required_ok:
        pandas_ok = test_pandas_functionality()
    else:
        pandas_ok = False
        print_header("PANDAS FUNCTIONALITY TEST")
        print_warning("Skipping - pandas not installed")
    
    # Determine overall status
    all_required_installed = python_ok and required_ok
    all_working = all_required_installed and pandas_ok
    
    # Print installation help if needed
    missing_required = []
    if not required_ok:
        # Get list of missing required packages
        for package in ['pandas', 'numpy', 'sklearn', 'matplotlib']:
            try:
                __import__(package)
            except ImportError:
                missing_required.append(package)
    
    print_installation_help(missing_required, ml_missing, data_missing)
    
    # Print next steps
    print_next_steps(all_working)
    
    # Summary
    print_header("TEST SUMMARY")
    
    print("\n" + "=" * 70)
    if all_working:
        print("ALL TESTS PASSED - READY FOR DEVELOPMENT!")
    elif all_required_installed:
        print("BASIC SETUP COMPLETE - INSTALL OPTIONAL PACKAGES")
    else:
        print("SETUP INCOMPLETE - INSTALL REQUIRED PACKAGES")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
