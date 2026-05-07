"""
preprocess.py — Vehicle Mileage MLOps Project
Author: Suresh D R | AI Product Developer & Technology Mentor
DV Analytics

Handles all data cleaning, imputation, encoding and feature selection.
Used by: train.py, predict.py, SageMaker Pipeline
Data source: S3 bucket — vehicle-mileage-project
"""

import pandas as pd
import numpy as np
import boto3
import io
import joblib
import os
import warnings
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore')

# ── AWS Configuration ──────────────────────────────────────────────────────
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID', 'YOUR_AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'YOUR_AWS_SECRET_KEY')
BUCKET         = os.getenv('S3_BUCKET', 'vehicle-mileage-project')
REGION         = os.getenv('AWS_REGION', 'ap-south-1')

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
    df  = pd.read_csv(io.BytesIO(obj['Body'].read()))
    print(f"Loaded from S3: {key} — {df.shape[0]:,} rows x {df.shape[1]} cols")
    return df

def save_csv_s3(df, key):
    s3  = get_s3_client()
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buf.getvalue())
    print(f"Saved to S3: s3://{BUCKET}/{key}")

# ── Ordinal and Binary Encoding Maps ──────────────────────────────────────
BINARY_MAPS = {
    'transmission'  : {'Manual': 0, 'Automatic': 1},
    'marital_status': {'Single': 0, 'Married': 1},
    'gender'        : {'Female': 0, 'Male': 1, 'Other': 1},
}

ORDINAL_MAPS = {
    'vehicle_condition': {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3},
    'road_quality'     : {'Poor': 0, 'Average': 1, 'Good': 2},
    'region_type'      : {'Rural': 0, 'Semi-urban': 1, 'Urban': 2},
    'education'        : {'School': 0, 'Graduate': 1, 'PG': 2},
    'traffic_index'    : {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3},
}

ONE_HOT_COLS = [
    'fuel_type', 'brand_tier', 'vehicle_segment', 'city_tier',
    'weekend_trips', 'work_from_home', 'parking_type', 'insurance_type'
]

TARGET_ENCODE_COLS = ['state', 'vehicle_brand', 'vehicle_model', 'occupation', 'color']

# ── Step 1: Clean Raw Data ────────────────────────────────────────────────
def clean_data(df):
    """Remove duplicates, fix types, remove impossible values, standardise text."""
    df = df.copy()

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"  Duplicates removed: {before - len(df)}")

    # Fix data types
    df['owner_age']       = pd.to_numeric(df['owner_age'], errors='coerce')
    df['manufacture_year']= pd.to_numeric(df['manufacture_year'], errors='coerce')
    df['vehicle_age']     = 2024 - df['manufacture_year']
    df['vehicle_age']     = df['vehicle_age'].clip(lower=1, upper=20)

    # Remove impossible values
    df = df[~(df['owner_age'].notna() & ((df['owner_age'] < 18) | (df['owner_age'] > 90)))]
    df = df[~(df['annual_income_lakh'].notna() & (df['annual_income_lakh'] < 0))]
    df = df[~(df['manufacture_year'].notna() & (df['manufacture_year'] > 2024))]
    df = df[~((df['annual_kms'] == 999999) | (df['annual_kms'] == 0))]
    df = df[df['annual_kms'] <= 145000]

    # Standardise text categories
    for col in ['fuel_type', 'city_tier', 'gender', 'transmission',
                'occupation', 'vehicle_segment', 'brand_tier']:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

    # Fix CNG capitalisation
    if 'fuel_type' in df.columns:
        df['fuel_type'] = df['fuel_type'].replace({'Cng': 'CNG', 'cng': 'CNG'})

    print(f"  Clean dataset: {len(df):,} rows")
    return df

# ── Step 2: Impute Missing Values ─────────────────────────────────────────
def impute_missing(df):
    """Group-wise median imputation for skewed features."""
    df = df.copy()

    # home_to_office_km — group by city_tier
    col = 'home_to_office_km'
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = df.groupby('city_tier')[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    # annual_income_lakh — group by brand_tier
    col = 'annual_income_lakh'
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = df.groupby('brand_tier')[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    # engine_cc — group by brand_tier
    col = 'engine_cc'
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = df.groupby('brand_tier')[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    # owner_age — group by occupation
    col = 'owner_age'
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = df.groupby('occupation')[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    # occupation — mode
    col = 'occupation'
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

    # num_children — logical rule
    col = 'num_children'
    if col in df.columns and df[col].isnull().sum() > 0:
        df.loc[(df[col].isnull()) & (df['has_children'] == 0), col] = 0
        df[col] = df[col].fillna(df[col].median())

    # monthly_fuel_spend — group by fuel_type
    col = 'monthly_fuel_spend'
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = df.groupby('fuel_type')[col].transform(lambda x: x.fillna(x.median()))
        df[col] = df[col].fillna(df[col].median())

    # toll_spend_monthly — logical rule
    col = 'toll_spend_monthly'
    if col in df.columns and df[col].isnull().sum() > 0:
        df.loc[(df[col].isnull()) & (df['uses_for_business'] == 0), col] = 0
        df[col] = df[col].fillna(df[col].median())

    # color — mode
    col = 'color'
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].mode()[0])

    remaining = df.isnull().sum().sum()
    print(f"  Missing values remaining: {remaining}")
    return df

# ── Step 3: Cap Outliers ───────────────────────────────────────────────────
def cap_outliers(df):
    """Cap outliers using IQR method for selected features."""
    df = df.copy()

    cap_cols = ['home_to_office_km', 'selling_price',
                'monthly_fuel_spend', 'toll_spend_monthly']

    for col in cap_cols:
        if col in df.columns:
            Q1    = df[col].quantile(0.25)
            Q3    = df[col].quantile(0.75)
            IQR   = Q3 - Q1
            upper = Q3 + 1.5 * IQR
            lower = max(Q1 - 1.5 * IQR, 0)
            df[col] = df[col].clip(lower=lower, upper=upper)

    print(f"  Outliers capped for: {cap_cols}")
    return df

# ── Step 4: Encode Features ───────────────────────────────────────────────
def encode_features(df, target_means=None, fit=True):
    """
    Encode categorical features for tree models.
    fit=True  → calculate target means (training time)
    fit=False → use saved target means (prediction time)
    """
    df = df.copy()

    # Binary encoding
    for col, mapping in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    # Ordinal encoding
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    # Target encoding for high cardinality
    if fit:
        target_means = {}
        for col in TARGET_ENCODE_COLS:
            if col in df.columns and 'annual_kms' in df.columns:
                mean_map = df.groupby(col)['annual_kms'].mean().to_dict()
                target_means[col] = mean_map
                df[col] = df[col].map(mean_map).fillna(df['annual_kms'].mean())
    else:
        # Use saved means for prediction
        for col in TARGET_ENCODE_COLS:
            if col in df.columns and target_means and col in target_means:
                global_mean = np.mean(list(target_means[col].values()))
                df[col] = df[col].map(target_means[col]).fillna(global_mean)

    # Remaining object columns — integer codes
    for col in df.select_dtypes(include='object').columns:
        if col != 'annual_kms':
            df[col] = pd.Categorical(df[col]).codes

    df = df.fillna(0)
    df = df.select_dtypes(include=[np.number])

    print(f"  Encoded dataset: {df.shape[1]} features")
    return df, target_means

# ── Main Pipeline Function ────────────────────────────────────────────────
def run_preprocessing_pipeline(source='s3', save_to_s3=True):
    """
    Full preprocessing pipeline from raw data to model-ready data.
    Called by train.py and SageMaker Pipeline.
    """
    print("=" * 60)
    print("PREPROCESSING PIPELINE")
    print("=" * 60)

    # Load data
    print("\nStep 1: Loading data...")
    if source == 's3':
        df = read_csv_s3('raw/vehicle_mileage_raw.csv')
    else:
        df = pd.read_csv(source)

    # Clean
    print("\nStep 2: Cleaning data...")
    df = clean_data(df)

    # Impute
    print("\nStep 3: Imputing missing values...")
    df = impute_missing(df)

    # Cap outliers
    print("\nStep 4: Capping outliers...")
    df = cap_outliers(df)

    # Encode
    print("\nStep 5: Encoding features...")
    df_encoded, target_means = encode_features(df, fit=True)

    # Save
    if save_to_s3:
        print("\nStep 6: Saving to S3...")
        save_csv_s3(df_encoded, 'data/06_encoded_tree.csv')

        # Save target means for prediction time
        s3  = get_s3_client()
        buf = io.BytesIO()
        joblib.dump(target_means, buf)
        buf.seek(0)
        s3.put_object(Bucket=BUCKET, Key='models/target_means.pkl', Body=buf.getvalue())
        print("  Target means saved to S3")

    print("\nPreprocessing complete!")
    print(f"Final shape: {df_encoded.shape}")
    return df_encoded, target_means


# ── Single Record Preprocessing (for prediction) ─────────────────────────
def preprocess_single_record(record_dict, target_means=None):
    """
    Preprocess a single vehicle record for prediction.
    record_dict: dictionary with vehicle features
    Returns: DataFrame ready for model.predict()
    """
    df = pd.DataFrame([record_dict])

    # Apply same cleaning
    for col in ['fuel_type', 'city_tier', 'gender', 'transmission',
                'occupation', 'vehicle_segment', 'brand_tier']:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

    if 'manufacture_year' in df.columns:
        df['manufacture_year'] = pd.to_numeric(df['manufacture_year'], errors='coerce')
        df['vehicle_age'] = 2024 - df['manufacture_year']
        df['vehicle_age'] = df['vehicle_age'].clip(lower=1, upper=20)

    # Load target means from S3 if not provided
    if target_means is None:
        try:
            s3  = get_s3_client()
            obj = s3.get_object(Bucket=BUCKET, Key='models/target_means.pkl')
            target_means = joblib.load(io.BytesIO(obj['Body'].read()))
        except Exception as e:
            print(f"Warning: Could not load target means: {e}")
            target_means = {}

    # Binary encoding
    for col, mapping in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    # Ordinal encoding
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    # Target encoding
    for col in TARGET_ENCODE_COLS:
        if col in df.columns and col in target_means:
            global_mean = np.mean(list(target_means[col].values()))
            df[col] = df[col].map(target_means[col]).fillna(global_mean)

    # Remaining object columns
    for col in df.select_dtypes(include='object').columns:
        df[col] = 0  # unknown category → 0

    df = df.fillna(0)

    # Load feature names to align columns
    try:
        s3       = get_s3_client()
        obj      = s3.get_object(Bucket=BUCKET, Key='models/feature_names.pkl')
        feat_names = joblib.load(io.BytesIO(obj['Body'].read()))
        # Add missing columns as 0, remove extra columns
        for f in feat_names:
            if f not in df.columns:
                df[f] = 0
        df = df[feat_names]
    except Exception as e:
        print(f"Warning: Feature alignment issue: {e}")
        df = df.select_dtypes(include=[np.number])

    return df


if __name__ == '__main__':
    print("Running preprocessing pipeline...")
    df_encoded, target_means = run_preprocessing_pipeline()
    print(f"\nFinal dataset: {df_encoded.shape}")
    print("Preprocessing complete!")
