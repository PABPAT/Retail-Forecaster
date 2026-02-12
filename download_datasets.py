#!/usr/bin/env python3
"""
Retail Forecasting Dataset Downloader
=====================================
This script downloads all recommended datasets for retail sales forecasting.

Requirements:
- Python 3.7+
- Internet connection
- Kaggle API credentials (for Kaggle datasets)

Author: Claude
Date: January 2026
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


class Colors:
    """Terminal colors for pretty output"""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 70}")
    print(f"{text:^70}")
    print(f"{'=' * 70}{Colors.ENDC}\n")


def print_success(text):
    """Print success message"""
    print(f"{Colors.OKGREEN} {text}{Colors.ENDC}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.FAIL} {text}{Colors.ENDC}")


def print_info(text):
    """Print info message"""
    print(f"{Colors.OKCYAN}  {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.WARNING}  {text}{Colors.ENDC}")


def install_package(package):
    """Install a Python package using pip"""
    print_info(f"Installing {package}...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])
        print_success(f"{package} installed successfully")
        return True
    except subprocess.CalledProcessError:
        print_error(f"Failed to install {package}")
        return False


def check_kaggle_credentials():
    """Check if Kaggle API credentials are set up"""
    kaggle_json = Path.home() / '.kaggle' / 'kaggle.json'
    return kaggle_json.exists()


def setup_kaggle_instructions():
    """Print instructions for setting up Kaggle API"""
    print_header("KAGGLE API SETUP INSTRUCTIONS")
    print("To download datasets from Kaggle, you need to set up API credentials:\n")
    print("1. Go to: https://www.kaggle.com/account")
    print("2. Scroll to the 'API' section")
    print("3. Click 'Create New API Token'")
    print("4. This downloads 'kaggle.json'")
    print("5. Move it to the correct location:")
    print(f"   • Linux/Mac: {Path.home() / '.kaggle' / 'kaggle.json'}")
    print(f"   • Windows: {Path.home() / '.kaggle' / 'kaggle.json'}")
    print("\n6. Set file permissions (Linux/Mac only):")
    print("   chmod 600 ~/.kaggle/kaggle.json\n")


def download_store_item_demand(output_dir):
    """Download Store Item Demand Forecasting dataset"""
    print_header("1. Store Item Demand Forecasting Dataset")
    print_info("This dataset is perfect for 3-month sales forecasting!")
    print_info("5 years of data, 50 items, 10 stores\n")
    
    try:
        os.chdir(output_dir)
        subprocess.run(['kaggle', 'competitions', 'download', '-c', 
                       'demand-forecasting-kernels-only'], check=True, 
                       capture_output=True)
        
        print_info("Extracting files...")
        subprocess.run(['unzip', '-o', '-q', 'demand-forecasting-kernels-only.zip'], 
                      check=True)
        
        print_success("Store Item Demand dataset downloaded successfully!")
        print_info(f"Location: {output_dir / 'train.csv'}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to download: {e}")
        return False
    except FileNotFoundError:
        print_error("'kaggle' or 'unzip' command not found. Install kaggle and ensure unzip is available.")
        return False


def download_m5_walmart(output_dir):
    """Download M5 Walmart Forecasting dataset"""
    print_header("2. M5 Walmart Forecasting Dataset")
    print_info("Industry-standard dataset with 5.4 years of comprehensive data")
    print_info("3,049 products across 3 categories, 10 stores\n")
    
    try:
        os.chdir(output_dir)
        subprocess.run(['kaggle', 'competitions', 'download', '-c', 
                       'm5-forecasting-accuracy'], check=True,
                       capture_output=True)
        
        print_info("Extracting files... (this may take a while)")
        subprocess.run(['unzip', '-o', '-q', 'm5-forecasting-accuracy.zip'], 
                      check=True)
        
        print_success("M5 Walmart dataset downloaded successfully!")
        print_info(f"Location: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to download: {e}")
        return False
    except FileNotFoundError:
        print_error("'kaggle' or 'unzip' command not found.")
        return False


def download_uci_online_retail(output_dir):
    """Download UCI Online Retail dataset"""
    print_header("3. UCI Online Retail Dataset")
    print_info("Real-world e-commerce transactions")
    print_info("541,909 transactions from Dec 2010 to Dec 2011\n")
    
    try:
        from ucimlrepo import fetch_ucirepo
        
        print_info("Fetching dataset from UCI repository...")
        online_retail = fetch_ucirepo(id=352)
        
        # Save to CSV
        output_file = output_dir / 'uci_online_retail.csv'
        online_retail.data.features.to_csv(output_file, index=False)
        
        print_success("UCI Online Retail dataset downloaded successfully!")
        print_info(f"Location: {output_file}")
        print_info(f"Rows: {len(online_retail.data.features):,}")
        return True
    except ImportError:
        print_error("ucimlrepo package not installed. Installing...")
        if install_package('ucimlrepo'):
            return download_uci_online_retail(output_dir)
        return False
    except Exception as e:
        print_error(f"Failed to download: {e}")
        return False


def download_online_retail_ii(output_dir):
    """Download Online Retail II dataset"""
    print_header("4. Online Retail II Dataset")
    print_info("Extended version with more recent data (2009-2011)\n")
    
    try:
        os.chdir(output_dir)
        subprocess.run(['kaggle', 'datasets', 'download', '-d', 
                       'mashlyn/online-retail-ii-uci'], check=True,
                       capture_output=True)
        
        print_info("Extracting files...")
        subprocess.run(['unzip', '-o', '-q', 'online-retail-ii-uci.zip'], 
                      check=True)
        
        print_success("Online Retail II dataset downloaded successfully!")
        print_info(f"Location: {output_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"Failed to download: {e}")
        return False
    except FileNotFoundError:
        print_error("'kaggle' or 'unzip' command not found.")
        return False


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Download retail forecasting datasets',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_datasets.py                    # Download all datasets
  python download_datasets.py --output ./data    # Specify output directory
  python download_datasets.py --dataset demand   # Download only Store Item Demand
  
Datasets:
  demand  - Store Item Demand Forecasting (Recommended for beginners)
  m5      - M5 Walmart Forecasting (Most comprehensive)
  uci     - UCI Online Retail (E-commerce focus)
  uci2    - UCI Online Retail II (Extended version)
  all     - Download all datasets (default)
        """
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='retail_datasets',
        help='Output directory for datasets (default: retail_datasets)'
    )
    
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        choices=['demand', 'm5', 'uci', 'uci2', 'all'],
        default='all',
        help='Which dataset to download (default: all)'
    )
    
    args = parser.parse_args()
    
    # Print welcome message
    print_header("🚀 RETAIL FORECASTING DATASET DOWNLOADER 🚀")
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)
    print_info(f"Output directory: {output_dir.absolute()}")
    
    # Check and install required packages
    print_info("Checking required packages...")
    required_packages = ['kaggle', 'pandas', 'ucimlrepo']
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            install_package(package)
    
    # Check Kaggle credentials
    if args.dataset in ['demand', 'm5', 'uci2', 'all']:
        if not check_kaggle_credentials():
            print_warning("Kaggle API credentials not found!")
            setup_kaggle_instructions()
            
            response = input("\nHave you set up Kaggle API credentials? (y/n): ")
            if response.lower() != 'y':
                print_error("Please set up Kaggle credentials and try again.")
                sys.exit(1)
    
    # Download datasets
    results = {}
    
    if args.dataset in ['demand', 'all']:
        results['Store Item Demand'] = download_store_item_demand(output_dir)
    
    if args.dataset in ['m5', 'all']:
        results['M5 Walmart'] = download_m5_walmart(output_dir)
    
    if args.dataset in ['uci', 'all']:
        results['UCI Online Retail'] = download_uci_online_retail(output_dir)
    
    if args.dataset in ['uci2', 'all']:
        results['Online Retail II'] = download_online_retail_ii(output_dir)
    
    # Print summary
    print_header("DOWNLOAD SUMMARY")
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for dataset, success in results.items():
        status = "SUCCESS" if success else " FAILED"
        print(f"{dataset:.<40} {status}")
    
    print(f"\n{success_count}/{total_count} datasets downloaded successfully")
    
    if success_count > 0:
        print_success(f"\nDatasets saved to: {output_dir.absolute()}")
        print_info("\nNext steps:")
        print("  1. Load the data with pandas")
        print("  2. Explore the dataset structure")
        print("  3. Start building your forecasting model!")
        print("\nCheck retail_data_download_guide.md for detailed instructions.")
    
    return success_count == total_count


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print_warning("\n\nDownload interrupted by user")
        sys.exit(1)
    except Exception as e:
        print_error(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
