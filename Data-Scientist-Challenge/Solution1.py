# =============================================================================
# VahanBima - Customer Lifetime Value (CLTV) Prediction
# Hackathon Solution | Evaluation Metric: R2 Score
# =============================================================================
# APPROACH SUMMARY:
#   1. Exploratory Data Analysis (EDA) & data quality checks
#   2. Preprocessing: missing value handling, encoding, outlier treatment
#   3. Feature Engineering: interaction terms, domain-specific features
#   4. Model Training: LightGBM, XGBoost, CatBoost, RandomForest
#   5. Hyperparameter Tuning: Optuna (Bayesian optimisation)
#   6. Ensembling: Weighted average of best models
#   7. Submission file generation
# =============================================================================

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("[INFO] Optuna not found. Install with: pip install optuna")
    print("[INFO] Falling back to default hyperparameters.")

# =============================================================================
# CONFIGURATION
# =============================================================================

TRAIN_PATH        = "train.csv"       # <- Update path if needed
TEST_PATH         = "test.csv"
SUBMISSION_PATH   = "submission.csv"
RANDOM_STATE      = 42
N_FOLDS           = 5
OPTUNA_TRIALS     = 50                # Increase for better tuning (e.g., 100)

np.random.seed(RANDOM_STATE)


# =============================================================================
# 1. DATA LOADING
# =============================================================================

print("=" * 60)
print("STEP 1: Loading Data")
print("=" * 60)

train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print(f"Train shape : {train.shape}")
print(f"Test shape  : {test.shape}")

# --- Force-convert columns that SHOULD be numeric but may have been read as strings ---
# 'income' and 'num_policies' often come in as object dtype (e.g. stored as "50000" strings)
# pd.to_numeric with errors='coerce' converts cleanly; any non-numeric becomes NaN (handled later)
FORCE_NUMERIC_COLS = ['income', 'num_policies', 'claim_amount', 'vintage', 'marital_status']
for col in FORCE_NUMERIC_COLS:
    for df in [train, test]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"\nTarget (cltv) stats:\n{train['cltv'].describe()}")
print(f"\nMissing values in train:\n{train.isnull().sum()[train.isnull().sum() > 0]}")
print(f"\nMissing values in test:\n{test.isnull().sum()[test.isnull().sum() > 0]}")
print(f"\nColumn dtypes after numeric coercion:\n{train.dtypes}")


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS (optional plots — comment out if running headless)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 2: EDA")
print("=" * 60)

# Distribution of target variable
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(train['cltv'], bins=50, color='steelblue', edgecolor='white')
axes[0].set_title("CLTV Distribution (Original)")
axes[0].set_xlabel("CLTV")

axes[1].hist(np.log1p(train['cltv']), bins=50, color='coral', edgecolor='white')
axes[1].set_title("CLTV Distribution (Log1p)")
axes[1].set_xlabel("log1p(CLTV)")

plt.tight_layout()
plt.savefig("cltv_distribution.png", dpi=100)
plt.close()
print("  -> Saved: cltv_distribution.png")

# Correlation heatmap — only use columns that are already numeric at this point
num_cols_eda = train.select_dtypes(include=[np.number]).columns.tolist()
if len(num_cols_eda) >= 2:
    plt.figure(figsize=(10, 8))
    sns.heatmap(train[num_cols_eda].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png", dpi=100)
    plt.close()
    print("  -> Saved: correlation_heatmap.png")
else:
    print("  [SKIP] Not enough numeric columns for heatmap at this stage.")


# =============================================================================
# 3. PREPROCESSING
# =============================================================================

print("\n" + "=" * 60)
print("STEP 3: Preprocessing")
print("=" * 60)

# --- 3a. Separate target and drop ID ---
TARGET = 'cltv'
ID_COL = 'id'

y_train_raw = train[TARGET].copy()
train_ids   = train[ID_COL].copy()
test_ids    = test[ID_COL].copy()

# Drop ID and target from feature sets
train.drop(columns=[ID_COL, TARGET], inplace=True)
test.drop(columns=[ID_COL], inplace=True)

# --- 3b. Log-transform target (CLTV is typically right-skewed) ---
# Using log1p to handle potential zero values; predictions will be expm1'd back
y_train = np.log1p(y_train_raw)
print(f"  Target skewness (raw)    : {y_train_raw.skew():.4f}")
print(f"  Target skewness (log1p)  : {y_train.skew():.4f}")

# --- 3c. Combine train + test for consistent preprocessing ---
n_train = len(train)
full_df = pd.concat([train, test], axis=0, ignore_index=True)

print(f"  Combined shape: {full_df.shape}")
print(f"  Dtypes:\n{full_df.dtypes.value_counts()}")

# --- 3d. Identify column types ---
cat_cols = full_df.select_dtypes(include=['object', 'category']).columns.tolist()
num_cols = full_df.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n  Categorical columns ({len(cat_cols)}): {cat_cols}")
print(f"  Numerical columns  ({len(num_cols)}): {num_cols}")

# --- 3e. Missing value imputation ---
# Numerical: fill with median (robust to outliers)
for col in num_cols:
    median_val = full_df[col].median()
    full_df[col].fillna(median_val, inplace=True)

# Categorical: fill with mode
for col in cat_cols:
    mode_val = full_df[col].mode()[0]
    full_df[col].fillna(mode_val, inplace=True)

print("\n  Missing values after imputation:", full_df.isnull().sum().sum())

# --- 3f. Outlier treatment — cap numeric columns at 1st/99th percentile (train stats only) ---
# Re-identify numeric columns AFTER force-conversion above
COLS_TO_CAP = [c for c in ['income', 'claim_amount'] if c in full_df.select_dtypes(include=[np.number]).columns]
for col in COLS_TO_CAP:
    q01 = full_df.iloc[:n_train][col].quantile(0.01)
    q99 = full_df.iloc[:n_train][col].quantile(0.99)
    full_df[col] = full_df[col].clip(lower=q01, upper=q99)
    print(f"  Capped '{col}' to [{q01:.0f}, {q99:.0f}]")


# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================

print("\n" + "=" * 60)
print("STEP 4: Feature Engineering")
print("=" * 60)

# --- 4a. Domain-specific interaction features ---
# NOTE: income and num_policies were force-converted to numeric in Step 1,
# so all arithmetic below is safe.

# Average claim per policy (high claim + few policies = high-value risk indicator)
full_df['claim_per_policy'] = full_df['claim_amount'] / (full_df['num_policies'] + 1)

# Income per year of engagement (loyalty-adjusted income)
full_df['income_per_vintage'] = full_df['income'] / (full_df['vintage'] + 1)

# Claim ratio relative to income (financial stress indicator)
full_df['claim_to_income_ratio'] = full_df['claim_amount'] / (full_df['income'] + 1)

# Policy density per vintage year
full_df['policies_per_vintage'] = full_df['num_policies'] / (full_df['vintage'] + 1)

# High-income flag (top 25% earners)
income_75th = full_df.iloc[:n_train]['income'].quantile(0.75)
full_df['is_high_income'] = (full_df['income'] >= income_75th).astype(int)

# Long-tenure customer flag
vintage_75th = full_df.iloc[:n_train]['vintage'].quantile(0.75)
full_df['is_long_tenure'] = (full_df['vintage'] >= vintage_75th).astype(int)

# Multi-policy customer flag (num_policies is now safely numeric)
full_df['is_multi_policy'] = (full_df['num_policies'] > 1).astype(int)

# Income buckets (5 quantile-based tiers) — only works on numeric income
try:
    full_df['income_bucket'] = pd.qcut(
        full_df['income'], q=5, labels=False, duplicates='drop'
    )
except Exception:
    # Fallback: simple rank-based bucket if qcut fails (e.g. too many duplicate values)
    full_df['income_bucket'] = pd.cut(
        full_df['income'],
        bins=5, labels=False
    )

# Vintage buckets (tenure tiers: new / mid / loyal)
full_df['vintage_bucket'] = pd.cut(
    full_df['vintage'],
    bins=[0, 2, 5, 10, 100],
    labels=[0, 1, 2, 3],
    include_lowest=True
).astype(int)

print("  Created 9 new engineered features.")

# --- 4b. Encode categorical variables ---
# Using Label Encoding (compatible with tree-based models)
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    full_df[col] = le.fit_transform(full_df[col].astype(str))
    label_encoders[col] = le

print(f"  Label-encoded {len(cat_cols)} categorical columns.")

# --- 4c. Split back into train and test ---
X_train = full_df.iloc[:n_train].copy()
X_test  = full_df.iloc[n_train:].copy()

print(f"\n  X_train shape: {X_train.shape}")
print(f"  X_test shape : {X_test.shape}")
print(f"  Features     : {list(X_train.columns)}")


# =============================================================================
# 5. MODEL TRAINING WITH CROSS-VALIDATION
# =============================================================================

print("\n" + "=" * 60)
print("STEP 5: Model Training & Cross-Validation")
print("=" * 60)

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)


# ── Helper: evaluate model with K-Fold CV ────────────────────────────────────
def cross_validate_model(model, X, y, model_name="Model"):
    oof_preds = np.zeros(len(X))
    scores = []
    for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y)):
        X_trn, X_val = X.iloc[trn_idx], X.iloc[val_idx]
        y_trn, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        model.fit(X_trn, y_trn)
        preds = model.predict(X_val)

        fold_r2 = r2_score(y_val, preds)
        scores.append(fold_r2)
        oof_preds[val_idx] = preds

    mean_r2 = np.mean(scores)
    std_r2  = np.std(scores)
    print(f"  {model_name:30s} | CV R2: {mean_r2:.5f} ± {std_r2:.5f}")
    return oof_preds, mean_r2


# ── Helper: train on full data and predict test ───────────────────────────────
def train_and_predict(model, X_tr, y_tr, X_te):
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


# =============================================================================
# 5a. HYPERPARAMETER TUNING WITH OPTUNA (LightGBM)
# =============================================================================

if OPTUNA_AVAILABLE:
    print("\n  [Optuna] Tuning LightGBM hyperparameters...")

    def lgb_objective(trial):
        params = {
            "n_estimators"      : trial.suggest_int("n_estimators", 300, 2000),
            "learning_rate"     : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves"        : trial.suggest_int("num_leaves", 20, 300),
            "max_depth"         : trial.suggest_int("max_depth", 3, 12),
            "min_child_samples" : trial.suggest_int("min_child_samples", 10, 100),
            "subsample"         : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree"  : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha"         : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda"        : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state"      : RANDOM_STATE,
            "n_jobs"            : -1,
            "verbosity"         : -1,
        }
        model = lgb.LGBMRegressor(**params)
        scores = []
        for trn_idx, val_idx in kf.split(X_train, y_train):
            model.fit(X_train.iloc[trn_idx], y_train.iloc[trn_idx])
            preds = model.predict(X_train.iloc[val_idx])
            scores.append(r2_score(y_train.iloc[val_idx], preds))
        return np.mean(scores)

    study_lgb = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_lgb.optimize(lgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best_lgb_params = study_lgb.best_params
    best_lgb_params.update({"random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1})
    print(f"  Best LightGBM R2 (Optuna): {study_lgb.best_value:.5f}")

    # ── Tune XGBoost ──────────────────────────────────────────────────────────
    print("  [Optuna] Tuning XGBoost hyperparameters...")

    def xgb_objective(trial):
        params = {
            "n_estimators"     : trial.suggest_int("n_estimators", 300, 2000),
            "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth"        : trial.suggest_int("max_depth", 3, 10),
            "min_child_weight" : trial.suggest_int("min_child_weight", 1, 20),
            "subsample"        : trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma"            : trial.suggest_float("gamma", 0, 5),
            "reg_alpha"        : trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda"       : trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state"     : RANDOM_STATE,
            "n_jobs"           : -1,
            "tree_method"      : "hist",
        }
        model = xgb.XGBRegressor(**params)
        scores = []
        for trn_idx, val_idx in kf.split(X_train, y_train):
            model.fit(X_train.iloc[trn_idx], y_train.iloc[trn_idx],
                      eval_set=[(X_train.iloc[val_idx], y_train.iloc[val_idx])],
                      verbose=False)
            preds = model.predict(X_train.iloc[val_idx])
            scores.append(r2_score(y_train.iloc[val_idx], preds))
        return np.mean(scores)

    study_xgb = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study_xgb.optimize(xgb_objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)
    best_xgb_params = study_xgb.best_params
    best_xgb_params.update({"random_state": RANDOM_STATE, "n_jobs": -1, "tree_method": "hist"})
    print(f"  Best XGBoost R2 (Optuna): {study_xgb.best_value:.5f}")

else:
    # Sensible defaults when Optuna is unavailable
    best_lgb_params = {
        "n_estimators": 1000, "learning_rate": 0.05, "num_leaves": 127,
        "max_depth": -1, "min_child_samples": 20, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": RANDOM_STATE, "n_jobs": -1, "verbosity": -1,
    }
    best_xgb_params = {
        "n_estimators": 1000, "learning_rate": 0.05, "max_depth": 6,
        "min_child_weight": 5, "subsample": 0.8, "colsample_bytree": 0.8,
        "gamma": 0.1, "reg_alpha": 0.1, "reg_lambda": 1.0,
        "random_state": RANDOM_STATE, "n_jobs": -1, "tree_method": "hist",
    }


# =============================================================================
# 5b. DEFINE MODELS
# =============================================================================

models = {
    "LightGBM (tuned)"  : lgb.LGBMRegressor(**best_lgb_params),
    "XGBoost (tuned)"   : xgb.XGBRegressor(**best_xgb_params),
    "CatBoost"          : cb.CatBoostRegressor(
                            iterations=1000, learning_rate=0.05, depth=7,
                            l2_leaf_reg=3, subsample=0.8, colsample_bylevel=0.8,
                            random_seed=RANDOM_STATE, verbose=0
                          ),
    "RandomForest"      : RandomForestRegressor(
                            n_estimators=500, max_depth=12, min_samples_leaf=5,
                            max_features=0.7, n_jobs=-1, random_state=RANDOM_STATE
                          ),
}


# =============================================================================
# 5c. CROSS-VALIDATE ALL MODELS & COLLECT OOF PREDICTIONS
# =============================================================================

print("\n  Cross-Validation Results (on log1p target):")
oof_store  = {}
cv_scores  = {}

for name, model in models.items():
    oof, score = cross_validate_model(model, X_train, y_train, name)
    oof_store[name]  = oof
    cv_scores[name]  = score

# Rank models by CV score
ranked = sorted(cv_scores.items(), key=lambda x: x[1], reverse=True)
print("\n  Model Ranking:")
for rank, (name, score) in enumerate(ranked, 1):
    print(f"    {rank}. {name:30s} R2 = {score:.5f}")


# =============================================================================
# 6. ENSEMBLE: WEIGHTED AVERAGE (weights ∝ CV R2 scores)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 6: Ensemble & Final Predictions")
print("=" * 60)

# Only use models with positive R2 for ensembling
positive_models = {k: v for k, v in cv_scores.items() if v > 0}
total_score     = sum(positive_models.values())
weights         = {k: v / total_score for k, v in positive_models.items()}

print("  Ensemble weights:")
for name, w in weights.items():
    print(f"    {name:30s}: {w:.4f}")

# ── Generate test predictions (train on full train set) ────────────────────
test_preds_store = {}

for name, model in models.items():
    if name in positive_models:
        test_pred = train_and_predict(model, X_train, y_train, X_test)
        test_preds_store[name] = test_pred

# Weighted average
ensemble_test_pred = np.zeros(len(X_test))
for name, pred in test_preds_store.items():
    ensemble_test_pred += weights[name] * pred

# Inverse log1p transform → original CLTV scale
final_predictions = np.expm1(ensemble_test_pred)

# Clip to avoid negative predictions (CLTV must be ≥ 0)
final_predictions = np.clip(final_predictions, 0, None)

print(f"\n  Prediction stats:")
print(f"    Min  : {final_predictions.min():.2f}")
print(f"    Max  : {final_predictions.max():.2f}")
print(f"    Mean : {final_predictions.mean():.2f}")
print(f"    Std  : {final_predictions.std():.2f}")


# =============================================================================
# 7. FEATURE IMPORTANCE (from best model)
# =============================================================================

print("\n" + "=" * 60)
print("STEP 7: Feature Importance")
print("=" * 60)

best_model_name = ranked[0][0]
best_model      = models[best_model_name]
best_model.fit(X_train, y_train)

if hasattr(best_model, "feature_importances_"):
    feat_imp = pd.DataFrame({
        "feature"    : X_train.columns,
        "importance" : best_model.feature_importances_
    }).sort_values("importance", ascending=False)

    print(f"\n  Top 15 features ({best_model_name}):")
    print(feat_imp.head(15).to_string(index=False))

    # Save feature importance plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feat_imp.head(20), x="importance", y="feature", palette="viridis")
    plt.title(f"Top 20 Feature Importances — {best_model_name}")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=100)
    plt.close()
    print("\n  -> Saved: feature_importance.png")


# =============================================================================
# 8. SUBMISSION FILE
# =============================================================================

print("\n" + "=" * 60)
print("STEP 8: Generating Submission File")
print("=" * 60)

submission = pd.DataFrame({
    "id"  : test_ids,
    "cltv": final_predictions
})

submission.to_csv(SUBMISSION_PATH, index=False)
print(f"  Saved submission to: {SUBMISSION_PATH}")
print(f"  Rows: {len(submission)}")
print(f"\n{submission.head(10)}")

print("\n" + "=" * 60)
print("DONE! Best estimated CV R2:", round(ranked[0][1], 5))
print("=" * 60)