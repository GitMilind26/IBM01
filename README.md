# E-Commerce Shipment Delivery Prediction and Delay Analysis

**Author:** Milind Kalura  
**Dataset:** E-Commerce Delivery & Shipping Data 2026  
**Records:** 50,000 | **Columns:** 30  
**Target:** `late_delivery` (Binary — Yes: 56.3% / No: 43.7%)

---

## Project Overview

This end-to-end Data Science project builds a late-delivery prediction system using e-commerce shipment data from 2026. It covers the complete ML pipeline — from raw data loading and cleaning, through exploratory analysis and feature engineering, to training and evaluating three classification models.

**Business Question:** Given operational, logistical, and environmental features of an order, can we predict whether it will arrive late?

---

## Project Structure

```
.
├── MilindKalura_EcommerceDeliveryPrediction.ipynb  ← Main notebook (all code)
├── MilindKalura_ProjectReport.docx                 ← Full project report
├── requirements.txt                                ← Python dependencies
├── README.md                                       ← This file
└── E-commerce_Delivery_Shipping_Data_2026.csv      ← Dataset (50,000 rows)
```

---

## Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| `order_id` | string | Unique order identifier |
| `order_date` | date | Date order was placed (2026) |
| `customer_id` | string | Customer identifier |
| `customer_segment` | categorical | Consumer, Small Business, Enterprise, Premium |
| `customer_city` | string | Customer's city |
| `customer_country` | string | Customer's country |
| `warehouse_id` | string | Fulfillment warehouse ID |
| `warehouse_city` | string | Warehouse location city |
| `product_category` | categorical | 12 categories (Electronics, Fashion, etc.) |
| `product_weight_kg` | float | Weight of the product (mean: 2.26 kg) |
| `order_value_usd` | float | Order total in USD (mean: $177.92) |
| `shipping_method` | categorical | Standard, Express, Economy, International, Same Day |
| `carrier` | categorical | 8 carriers (EagleCourier, BlueRoute, etc.) |
| `distance_km` | float | Shipping distance (mean: 2,496 km) |
| `promised_delivery_days` | int | Committed delivery window (mean: 7.2 days) |
| `actual_delivery_days` | int | Actual days taken (mean: 8.35 days) |
| `delivery_status` | categorical | Delivered, In Transit, Delayed, Failed Delivery |
| `late_delivery` | **TARGET** | Yes / No — whether delivery was late |
| `delivery_delay_days` | int | Days delayed beyond promise (mean: 1.28) |
| `shipping_cost_usd` | float | Shipping fee paid (mean: $120.52) |
| `package_size` | categorical | Small, Medium, Large, Oversized |
| `payment_method` | categorical | Credit Card, Debit Card, PayPal, etc. |
| `order_priority` | categorical | Normal, High, Low, Urgent |
| `weather_condition` | categorical | Clear, Cloudy, Rain, Snow, Storm, Extreme Heat |
| `customer_rating` | int | 1–5 star rating (mean: 3.37) |
| `return_requested` | categorical | Yes / No |
| `return_reason` | string | Reason for return (if applicable) |
| `delivery_attempts` | int | Number of delivery attempts (mean: 1.17) |
| `warehouse_processing_hours` | float | Hours from order to dispatch (mean: 14.41) |
| `tracking_status` | categorical | Tracking label (mirrors delivery_status) |

---

## Setup & Installation

### 1. Clone / download the project

Place all files in the same folder.

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch the notebook

```bash
jupyter notebook MilindKalura_EcommerceDeliveryPrediction.ipynb
```

or with JupyterLab:

```bash
jupyter lab MilindKalura_EcommerceDeliveryPrediction.ipynb
```

### 5. Run all cells

Use **Kernel → Restart & Run All** to execute the full notebook from scratch.

---

## Notebook Sections

| Section | Description |
|---------|-------------|
| 1. Import Libraries | pandas, numpy, matplotlib, seaborn, scikit-learn |
| 2. Data Loading | Load CSV, inspect shape and dtypes |
| 3. Data Cleaning | Missing values, duplicates, datetime conversion |
| 4. EDA | 10 charts covering target, shipping method, weather, categories, carriers, trends |
| 5. Feature Engineering | 7 new features (delay_gap, cost_per_km, is_high_value, etc.) |
| 6. Preprocessing | Column drops, LabelEncoding, train/test split, StandardScaler |
| 7. Model Training | Logistic Regression, Decision Tree, Random Forest |
| 8. Evaluation | Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix, ROC Curve |
| 9. Feature Importance | Random Forest and Decision Tree importance plots |
| 10. Cross-Validation | 5-fold stratified CV on all three models |
| 11. Business Insights | Data-driven operational recommendations |
| 12. Conclusion | Summary, limitations, future scope |

---

## Key Findings

- **56.3%** of orders (28,162 out of 50,000) are delivered late
- **Economy and International shipping** have the highest late delivery rates
- **Storm and Snow weather** significantly elevate delay probability
- **Warehouse processing time** is among the strongest predictors of delay
- **Random Forest** outperforms Logistic Regression and Decision Tree
- **Late deliveries** drive approximately **2.4× higher return rates**
- **High-value orders (>$500)** show a noticeably different risk profile

---

## Generated Output Files (from notebook)

After running the notebook, the following plot files are saved:

| File | Description |
|------|-------------|
| `plot_01_target_distribution.png` | Late delivery count and % distribution |
| `plot_02_late_by_shipping_method.png` | Late rate by shipping method |
| `plot_03_late_by_category.png` | Late rate by product category |
| `plot_04_late_by_weather.png` | Late rate by weather condition |
| `plot_05_numerical_distributions.png` | Histograms of all numerical features |
| `plot_06_correlation_heatmap.png` | Correlation matrix of numerical features |
| `plot_07_delay_days_boxplot.png` | Delivery delay days boxplot by target |
| `plot_08_late_by_priority.png` | Late rate by order priority |
| `plot_09_monthly_trend.png` | Monthly order volume + late delivery trend |
| `plot_10_late_by_carrier.png` | Late rate by carrier |
| `plot_11_confusion_matrices.png` | Confusion matrices for all 3 models |
| `plot_12_roc_curves.png` | ROC curves comparison |
| `plot_13_model_comparison.png` | Metric bar chart comparison |
| `plot_14_feature_importance.png` | Random Forest feature importances |
| `plot_15_dt_feature_importance.png` | Decision Tree feature importances |

---

## Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.0.0
```

---

## License

This project is for educational and portfolio purposes.
