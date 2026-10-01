# House Prices Prediction

Predicting residential house sale prices with machine learning, built for the Kaggle competition
[House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

The goal is to predict `SalePrice` for houses in Ames, Iowa from 79 features describing the house
(size, quality, location, year built, garage, basement, etc.).

## Approach

1. **Exploration**: inspect missing values and the target distribution. `SalePrice` is right-skewed.
2. **Target transform**: models are trained on `log1p(SalePrice)` and predictions are converted back with `expm1`. This matches the competition metric (RMSE on log prices).
3. **Outliers**: only the few well-known extreme points (very large `GrLivArea` with a low price) are removed.
4. **Preprocessing** (inside a scikit-learn `Pipeline`, so there is no data leakage and train/test columns always match):
   - numeric features: median imputation (plus scaling for the linear model)
   - categorical features: missing values filled with `"None"` (a missing value usually means the feature does not exist, e.g. no alley or no garage), then one-hot encoding with `handle_unknown="ignore"`
   - `MSSubClass` is treated as a categorical feature
5. **Models**: Ridge regression, Random Forest, XGBoost, compared with a validation split and 5-fold cross-validation.
6. **Tuning**: `RandomizedSearchCV` (30 iterations, 5-fold CV) on the XGBoost pipeline.
7. **Final model**: the best pipeline is refit on all training data and used to create `submission.csv`.

## Results

Validation set (20% hold-out), lower RMSLE is better:

| Model | RMSLE | MAE ($) | RMSE ($) | R2 |
|---|---|---|---|
| Ridge | 0.1227 | 14305.3832 | 20007.4396 | 0.9107 |
| Random Forest | 0.1440 | 16519.1826 | 24207.4587 | 0.8770 |
| XGBoost (default) | 0.1425 | 17147.4789 | 24951.2442 | 0.8795 |
| XGBoost (tuned) | 0.1223 | 14608.8130 | 19814.0009 | 0.9113 |

## Project structure

```
house-prices-prediction/
├── house_price.ipynb    # full analysis, modeling and submission code
├── requirements.txt     # Python dependencies
├── README.md
├── .gitignore
└── data/                # not included in the repo (see below)
    ├── train.csv
    └── test.csv
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/USERNAME/house-prices-prediction.git
cd house-prices-prediction
```

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

### 4. Download the data

The dataset is not included in this repository. Download `train.csv` and `test.csv` from the
[competition data page](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data)
and place them in a folder named `data/` in the project root.

### 5. Run the notebook

```bash
jupyter notebook house_price.ipynb
```

Run all cells from top to bottom. The last cell writes `submission.csv`, which can be uploaded to Kaggle.

## Tech stack

- Python 3.10+
- pandas, NumPy
- scikit-learn
- XGBoost
- Matplotlib

## Possible improvements

- Feature engineering (total square footage, house age, total bathrooms)
- Ordinal encoding for quality columns (`ExterQual`, `KitchenQual`, `BsmtQual`, ...)
- Log-transform skewed numeric features
- Ensembling (for example XGBoost + Ridge + LightGBM)
- Feature selection based on feature importance

## License

This project is for educational purposes. The dataset belongs to Kaggle and is subject to the
competition's rules.
