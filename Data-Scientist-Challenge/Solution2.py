import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import warnings

warnings.filterwarnings('ignore')

def load_data():
    print("Loading data...")
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test

def clean_and_engineer(df):
    print("Engineering features and mapping ordinals...")
    df_eng = df.copy()
    
    # 1. FIX THE MASSIVE DATA LEAK: Map String Categories to Ordinal Numbers
    # This turns strings into math-friendly tiers
    income_map = {'<=2L': 1, '2L-5L': 2, '5L-10L': 3, 'More than 10L': 4}
    df_eng['income_level'] = df_eng['income'].map(income_map)
    
    policy_map = {'1': 1, 'More than 1': 2}
    df_eng['num_policies_numeric'] = df_eng['num_policies'].map(policy_map)
    
    # 2. Mathematical features (Now using the fixed numeric mappings!)
    df_eng['has_claimed'] = (df_eng['claim_amount'] > 0).astype(int)
    df_eng['claim_to_income_ratio'] = df_eng['claim_amount'] / df_eng['income_level']
    df_eng['policies_per_year'] = df_eng['num_policies_numeric'] / (df_eng['vintage'] + 1)
    
    # 3. Cross-Features
    df_eng['area_policy_combo'] = df_eng['area'].astype(str) + "_" + df_eng['policy'].astype(str)

    # 4. Strictly define what is categorical for CatBoost
    cat_cols = ['gender', 'area', 'qualification', 'income', 'num_policies', 'policy', 'type_of_policy', 'area_policy_combo']
    
    for col in cat_cols:
        if col in df_eng.columns:
            df_eng[col] = df_eng[col].fillna('Unknown').astype(str)

    return df_eng

def train_and_evaluate(train, test):
    print("Preparing CatBoost model training...")
    
    X = train.drop(columns=['id', 'cltv'])
    y = train['cltv'] 
    X_test = test.drop(columns=['id'])
    
    # Explicitly tell CatBoost which columns are text/categories
    cat_features = ['gender', 'area', 'qualification', 'income', 'num_policies', 'policy', 'type_of_policy', 'area_policy_combo']
    
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    
    # CatBoost parameters optimized for R2
    cb_params = {
        'iterations': 1500,
        'learning_rate': 0.05,
        'depth': 6,
        'loss_function': 'RMSE',
        'eval_metric': 'R2',          
        'l2_leaf_reg': 3.0,           
        'random_seed': 42,
        'verbose': 0                  
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        train_pool = Pool(X_train, y_train, cat_features=cat_features)
        val_pool = Pool(X_val, y_val, cat_features=cat_features)
        test_pool = Pool(X_test, cat_features=cat_features)
        
        model = CatBoostRegressor(**cb_params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=150,
            use_best_model=True
        )
        
        oof_preds[val_idx] = model.predict(val_pool)
        test_preds += model.predict(test_pool) / n_splits
        
        fold_r2 = r2_score(y_val, oof_preds[val_idx])
        print(f"Fold {fold+1} R2: {fold_r2:.4f}")
        
    overall_r2 = r2_score(y, oof_preds)
    print("-" * 30)
    print(f"Overall Local OOF R2 Score: {overall_r2:.4f}")
    print("-" * 30)
    
    return test_preds

def main():
    train, test = load_data()
    
    train_eng = clean_and_engineer(train)
    test_eng = clean_and_engineer(test)
    
    final_predictions = train_and_evaluate(train_eng, test_eng)
    
    print("Creating submission file...")
    submission = pd.DataFrame({
        'id': test['id'],
        'cltv': final_predictions
    })
    
    submission.to_csv('submission_v6.csv', index=False)
    print("Success! 'submission_v6.csv' is ready for upload.")

if __name__ == "__main__":
    main()