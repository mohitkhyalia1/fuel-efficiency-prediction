# 🚗 ML-Based Vehicle Fuel Efficiency Prediction

Predict vehicle fuel efficiency (MPG) using machine learning — from data exploration to model comparison and feature importance analysis.

---

## 📌 Project Overview

This project applies supervised machine learning techniques to predict the **miles per gallon (MPG)** of vehicles based on their mechanical attributes. The analysis covers the full ML pipeline: data cleaning, exploratory analysis, feature engineering, model training, evaluation, and interpretation.

**Key Question:** Which mechanical parameters most strongly influence a vehicle's fuel efficiency, and how accurately can we predict it?

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Source** | [Seaborn MPG Dataset](https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv) (originally from [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/auto+mpg)) |
| **Rows** | 398 vehicles |
| **Period** | Model years 1970–1982 |
| **Target** | `mpg` (Miles Per Gallon) |

### Features

| Feature | Type | Description |
|---|---|---|
| `cylinders` | Integer | Number of engine cylinders |
| `displacement` | Float | Engine displacement (cubic inches) |
| `horsepower` | Float | Engine horsepower |
| `weight` | Integer | Vehicle weight (lbs) |
| `acceleration` | Float | 0–60 mph time (seconds) |
| `model_year` | Integer | Model year (70–82) |
| `origin` | Category | Manufacturing region (usa / europe / japan) |

---

## 🔧 Methodology

1. **Data Loading & Overview** — Load dataset, inspect shape, dtypes, and missing values
2. **Data Cleaning** — Handle missing `horsepower` values (mean imputation)
3. **Exploratory Data Analysis (EDA)** — Correlation heatmap, distribution plots, pairplots
4. **Feature Engineering** — Create `power_to_weight` ratio (horsepower / weight)
5. **Preprocessing** — One-hot encoding for `origin`, train/test split (80/20)
6. **Model Training** — Train and compare five regression models
7. **Evaluation** — Compare R², RMSE, and MAE across models
8. **Feature Importance** — Analyze which features drive predictions (Random Forest)

---

## 📊 Model Comparison

| Model | R² Score | RMSE | MAE |
|---|---|---|---|
| Linear Regression | ~0.82 | ~3.3 | ~2.5 |
| Ridge Regression | ~0.82 | ~3.3 | ~2.5 |
| Decision Tree | ~0.83 | ~3.2 | ~2.3 |
| **Random Forest** | **~0.88** | **~2.7** | **~2.0** |
| Gradient Boosting | ~0.87 | ~2.8 | ~2.1 |

> ✅ **Random Forest** achieved the best performance overall.

---

## 🔑 Key Findings

- **Weight**, **horsepower**, and **engine displacement** are the strongest predictors of fuel efficiency
- The engineered `power_to_weight` feature improved model interpretability
- Non-linear models (Random Forest, Gradient Boosting) significantly outperformed linear models
- Japanese vehicles (origin = japan) tend to have higher fuel efficiency in this dataset

---

## 🛠 Tech Stack

- **Python 3.8+**
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — ML models, preprocessing, and evaluation

---

## ⚙️ Setup & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/mohitkhyalia1/fuel-efficiency-prediction.git
cd fuel-efficiency-prediction
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Notebook
```bash
jupyter notebook fuel_efficiency_prediction.ipynb
```

> The dataset is loaded directly from a public URL — no manual download needed.

---

## 📁 Project Structure

```
fuel-efficiency-prediction/
├── fuel_efficiency_prediction.ipynb   # Main notebook
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Files to exclude from Git
└── README.md                           # Project documentation
```

---

## 📜 License

This project is open source and available under the [MIT License](LICENSE).
