# Retail Sales Forecasting - Dataset Download Package

## 🎯 Quick Start

This package helps you download datasets for building a predictive model to forecast retail product sales for the next 3 months.

### Option 1: Automated Download (Recommended)

```bash
# Make the script executable
chmod +x download_datasets.py

# Run the download script
python download_datasets.py

# Or specify output directory
python download_datasets.py --output ./my_data

# Or download specific dataset
python download_datasets.py --dataset demand
```

### Option 2: Manual Download

Follow the detailed instructions in `retail_data_download_guide.md`

---

## 📦 Available Datasets

| Dataset | Size | Best For | Difficulty |
|---------|------|----------|------------|
| **Store Item Demand** | ~5MB | 3-month forecasting | ⭐ Beginner |
| **M5 Walmart** | ~100MB | Production models | ⭐⭐⭐ Advanced |
| **UCI Online Retail** | ~23MB | E-commerce analysis | ⭐⭐ Intermediate |
| **Online Retail II** | ~45MB | Extended e-commerce | ⭐⭐ Intermediate |

---

## 🔧 Prerequisites

### 1. Install Python packages
```bash
pip install kaggle pandas ucimlrepo
```

### 2. Set up Kaggle API (for Kaggle datasets)

1. Go to: https://www.kaggle.com/account
2. Click "Create New API Token"
3. Move `kaggle.json` to:
   - **Linux/Mac:** `~/.kaggle/kaggle.json`
   - **Windows:** `C:\Users\<username>\.kaggle\kaggle.json`
4. Set permissions (Linux/Mac only):
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

---

## 📚 What's Included

1. **download_datasets.py** - Automated download script
2. **retail_data_download_guide.md** - Comprehensive guide with details
3. **README.md** - This file

---

## 💡 Recommended Workflow

### For Beginners:
```bash
# Download the simplest dataset
python download_datasets.py --dataset demand

# Start exploring
python
>>> import pandas as pd
>>> df = pd.read_csv('retail_datasets/train.csv')
>>> df.head()
```

### For Advanced Users:
```bash
# Download all datasets
python download_datasets.py

# Compare models on different datasets
# Build ensemble models
# Deploy to production
```

---

## 🚀 Next Steps After Download

1. **Load the data:**
   ```python
   import pandas as pd
   df = pd.read_csv('retail_datasets/train.csv')
   ```

2. **Explore the structure:**
   ```python
   print(df.head())
   print(df.info())
   print(df.describe())
   ```

3. **Start modeling:**
   - Build baseline model (moving average)
   - Feature engineering (lags, rolling stats)
   - Train ML models (XGBoost, LightGBM)
   - Evaluate performance (RMSE, MAE)

---

## 📊 Dataset Details

### Store Item Demand Forecasting
- **Purpose:** Predict 3 months of sales
- **Columns:** date, store, item, sales
- **Time Range:** 5 years
- **Stores:** 10
- **Items:** 50

### M5 Walmart Forecasting
- **Purpose:** Hierarchical sales forecasting
- **Products:** 3,049
- **Categories:** Hobbies, Foods, Household
- **Features:** Sales, prices, calendar, events
- **Time Range:** 2011-2016 (5.4 years)

### UCI Online Retail
- **Purpose:** E-commerce transaction analysis
- **Transactions:** 541,909
- **Columns:** InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country
- **Time Range:** Dec 2010 - Dec 2011

---

## ❓ Troubleshooting

### "Kaggle credentials not found"
→ Set up Kaggle API credentials (see Prerequisites)

### "Permission denied"
→ Run: `chmod 600 ~/.kaggle/kaggle.json`

### "Command not found: unzip"
→ Install unzip:
- **Ubuntu/Debian:** `sudo apt-get install unzip`
- **Mac:** `brew install unzip`
- **Windows:** Install 7-Zip or use PowerShell

### "Module not found"
→ Install required packages:
```bash
pip install kaggle pandas ucimlrepo
```

---

## 🎓 Learning Resources

- **Kaggle Learn:** https://www.kaggle.com/learn/time-series
- **Dataset Documentation:**
  - Store Item Demand: https://www.kaggle.com/c/demand-forecasting-kernels-only
  - M5 Walmart: https://www.kaggle.com/c/m5-forecasting-accuracy
  - UCI Online Retail: https://archive.ics.uci.edu/dataset/352/online+retail

---

## 📧 Need Help?

If you encounter issues:
1. Check the detailed guide: `retail_data_download_guide.md`
2. Verify all prerequisites are met
3. Check internet connection
4. Ensure sufficient disk space

---

## ✅ Quick Verification

After download, verify files exist:
```bash
ls -lh retail_datasets/
```

You should see:
- `train.csv` (Store Item Demand)
- `sales_train_evaluation.csv` (M5 Walmart)
- `uci_online_retail.csv` (UCI)
- And more...

---

**Happy Forecasting! 🚀**
