"""
train.py — Vehicle Mileage MLOps Project
Author: Suresh D R | AI Product Developer & Technology Mentor
DV Analytics

Trains the best model on encoded data from S3.
Evaluates against existing model — saves to S3 if better.
Logs all experiments to MLflow.
Called by: SageMaker Pipeline (automated), local training
"""

import pandas as pd
import numpy as np
import boto3
import io
import os
import json
import joblib
import warnings
from dotenv import load_dotenv
load_dotenv()
import argparse
from datetime import datetime
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except:
    XGB_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except:
    LGB_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CB_AVAILABLE = True
except:
    CB_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except:
    MLFLOW_AVAILABLE = False

# ── AWS Configuration ──────────────────────────────────────────────────────
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', 'YOUR_AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'YOUR_AWS_SECRET_KEY')
BUCKET         = os.getenv('S3_BUCKET', 'vehicle-mileage-project')
REGION         = os.getenv('AWS_REGION', 'ap-south-1')
MLFLOW_URI     = os.getenv('MLFLOW_TRACKING_URI',
                            f's3://{BUCKET}/mlflow')

def get_s3_client():
    return boto3.client(
        's3',
        region_name=REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

def read_csv_s3(key):
    s3  = get_s3_client()
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj['Body'].read()))

def save_model_s3(model, key):
    s3  = get_s3_client()
    buf = io.BytesIO()
    joblib.dump(model, buf)
    buf.seek(0)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    print(f"Model saved to S3: s3://{BUCKET}/{key}")

def load_model_s3(key):
    s3  = get_s3_client()
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return joblib.load(io.BytesIO(obj['Body'].read()))

def save_json_s3(data, key):
    s3 = get_s3_client()
    s3.put_object(
        Bucket=BUCKET, Key=key,
        Body=json.dumps(data, indent=2)
    )

# ── Evaluation Metrics ─────────────────────────────────────────────────────
def evaluate_model(name, y_true, y_pred):
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    mae    = mean_absolute_error(y_true, y_pred)
    r2     = r2_score(y_true, y_pred)
    mape   = np.mean(np.abs((y_true.values - y_pred) / y_true.values)) * 100
    n, p   = len(y_true), 20
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

    metrics = {
        'model'  : name,
        'rmse'   : round(rmse, 2),
        'mae'    : round(mae, 2),
        'r2'     : round(r2, 4),
        'adj_r2' : round(adj_r2, 4),
        'mape'   : round(mape, 2)
    }

    print(f"  {name}")
    print(f"    RMSE    : {rmse:>10,.0f} km")
    print(f"    MAE     : {mae:>10,.0f} km")
    print(f"    R2      : {r2:>10.4f}")
    print(f"    Adj R2  : {adj_r2:>10.4f}")
    print(f"    MAPE    : {mape:>10.2f}%")
    return metrics

# ── Train All Models ───────────────────────────────────────────────────────
def train_models(X_train, X_test, y_train, y_test, cat_features=None):
    results  = []
    models   = {}

    kf = KFold(n_splits=2, shuffle=True, random_state=42)

    # ── XGBoost ────────────────────────────────────────────────────────────
    if XGB_AVAILABLE:
        print("\nTraining XGBoost...")
        xgb = XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=42,
            verbosity=0,
            tree_method='hist'
        )
        xgb.fit(X_train, y_train)
        metrics = evaluate_model("XGBoost", y_test, xgb.predict(X_test))

        cv = cross_val_score(xgb, X_train, y_train, cv=kf,
                              scoring='neg_mean_absolute_error', n_jobs=-1)
        metrics['cv_mae_mean'] = round(-cv.mean(), 2)
        metrics['cv_mae_std']  = round(cv.std(), 2)
        print(f"    CV MAE  : {-cv.mean():>10,.0f} km (std={cv.std():,.0f})")

        results.append(metrics)
        models['XGBoost'] = xgb

    # ── LightGBM ───────────────────────────────────────────────────────────
    if LGB_AVAILABLE:
        print("\nTraining LightGBM...")
        lgbm = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.08,
            num_leaves=50,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            random_state=42,
            verbose=-1
        )
        lgbm.fit(X_train, y_train)
        metrics = evaluate_model("LightGBM", y_test, lgbm.predict(X_test))

        cv = cross_val_score(lgbm, X_train, y_train, cv=kf,
                              scoring='neg_mean_absolute_error', n_jobs=-1)
        metrics['cv_mae_mean'] = round(-cv.mean(), 2)
        metrics['cv_mae_std']  = round(cv.std(), 2)
        print(f"    CV MAE  : {-cv.mean():>10,.0f} km (std={cv.std():,.0f})")

        results.append(metrics)
        models['LightGBM'] = lgbm

    # ── CatBoost ───────────────────────────────────────────────────────────
    if CB_AVAILABLE:
        print("\nTraining CatBoost...")
        cb = CatBoostRegressor(
            iterations=300,
            depth=6,
            learning_rate=0.08,
            random_seed=42,
            verbose=0,
            early_stopping_rounds=30
        )
        cb.fit(X_train, y_train, eval_set=(X_test, y_test))
        metrics = evaluate_model("CatBoost", y_test, cb.predict(X_test))
        metrics['cv_mae_mean'] = metrics['mae']
        metrics['cv_mae_std']  = 0
        results.append(metrics)
        models['CatBoost'] = cb

    return results, models

# ── Compare with Existing Model ────────────────────────────────────────────
def compare_with_existing_model(new_mape, model_key='models/best_model.pkl'):
    """Returns True if new model is better than existing."""
    try:
        s3  = get_s3_client()
        obj = s3.get_object(Bucket=BUCKET, Key='models/model_metrics.json')
        existing_metrics = json.loads(obj['Body'].read())
        existing_mape    = existing_metrics.get('mape', 999)
        print(f"\nExisting model MAPE : {existing_mape:.2f}%")
        print(f"New model MAPE      : {new_mape:.2f}%")
        if new_mape < existing_mape:
            print("New model is BETTER — will deploy!")
            return True
        else:
            print("Existing model is better — keeping current model.")
            return False
    except Exception:
        print("No existing model found — saving new model as first version.")
        return True

# ── MLflow Logging ─────────────────────────────────────────────────────────
def log_to_mlflow(model_name, model_obj, metrics, X_train):
    if not MLFLOW_AVAILABLE:
        return
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment('vehicle-mileage-prediction')
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_param('model_name',  model_name)
            mlflow.log_param('n_features',  X_train.shape[1])
            mlflow.log_param('n_train',     len(X_train))
            mlflow.log_metric('mape',       metrics['mape'])
            mlflow.log_metric('mae',        metrics['mae'])
            mlflow.log_metric('rmse',       metrics['rmse'])
            mlflow.log_metric('r2',         metrics['r2'])
            mlflow.log_metric('adj_r2',     metrics['adj_r2'])
            mlflow.log_metric('cv_mae',     metrics.get('cv_mae_mean', 0))
            mlflow.sklearn.log_model(model_obj, 'model')
        print(f"MLflow logged: {model_name}")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")

# ── Main Training Function ─────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("VEHICLE MILEAGE — MODEL TRAINING PIPELINE")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # ── Load Data from S3 ──────────────────────────────────────────────────
    print("\nLoading encoded data from S3...")
    df = read_csv_s3('data/06_encoded_tree.csv')
    print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # Load feature names
    try:
        feat_df      = read_csv_s3('data/06_feature_names.csv')
        cat_features = feat_df[feat_df['is_cat_tree'] == True]['feature'].tolist()
    except:
        cat_features = []

    # ── Prepare Features ───────────────────────────────────────────────────
    X = df.drop(columns=['annual_kms'])
    y = df['annual_kms']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")
    print(f"Target mean: {y.mean():,.0f} km | std: {y.std():,.0f} km")

    # ── Train Models ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)

    results, models = train_models(X_train, X_test, y_train, y_test, cat_features)

    # ── Pick Best Model ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    results_df = pd.DataFrame(results).sort_values('mape')
    for _, row in results_df.iterrows():
        marker = " ← BEST" if _ == 0 else ""
        print(f"  {row['model']:<15} MAPE={row['mape']:.2f}%  MAE={row['mae']:,.0f}  R2={row['r2']:.4f}{marker}")

    best_result = results_df.iloc[0]
    best_name   = best_result['model']
    best_model  = models[best_name]
    best_mape   = best_result['mape']

    print(f"\nBest model: {best_name} (MAPE={best_mape:.2f}%)")

    # ── Log to MLflow ──────────────────────────────────────────────────────
    log_to_mlflow(best_name, best_model, best_result.to_dict(), X_train)

    # ── Compare and Save ───────────────────────────────────────────────────
    is_better = compare_with_existing_model(best_mape)

    if is_better:
        print("\nSaving new best model to S3...")

        # Save model
        save_model_s3(best_model, 'models/best_model.pkl')

        # Save feature names
        s3  = get_s3_client()
        buf = io.BytesIO()
        joblib.dump(X.columns.tolist(), buf)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key='models/feature_names.pkl', Body=buf.getvalue())
        print("Feature names saved to S3")

        # Save metrics
        metrics_to_save = {
            'model'       : best_name,
            'mape'        : best_mape,
            'mae'         : float(best_result['mae']),
            'rmse'        : float(best_result['rmse']),
            'r2'          : float(best_result['r2']),
            'adj_r2'      : float(best_result['adj_r2']),
            'cv_mae'      : float(best_result.get('cv_mae_mean', 0)),
            'trained_on'  : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'n_train'     : len(X_train),
            'n_features'  : X_train.shape[1],
            'all_results' : results
        }
        save_json_s3(metrics_to_save, 'models/model_metrics.json')
        print("Model metrics saved to S3")

        print(f"\nNew best model deployed: {best_name}")
        print(f"MAPE: {best_mape:.2f}%")
        print(f"Saved: s3://{BUCKET}/models/best_model.pkl")

    else:
        print("\nKeeping existing model — no deployment needed.")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return is_better, best_mape


if __name__ == '__main__':
    main()
