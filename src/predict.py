"""
predict.py — Vehicle Mileage MLOps Project
Author: Suresh D R | AI Product Developer & Technology Mentor
DV Analytics

Core prediction module. Loads best model from S3 and returns predictions.
Used by: app.py, api.py

FIX: Removed target_means (never saved in notebooks).
     Encoding now matches NB06 exactly — integer codes for all categoricals.
"""

import numpy as np
import pandas as pd
import boto3
import io
import os
import json
import joblib
import warnings
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore')

# ── AWS Configuration ──────────────────────────────────────────────────────
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID',     'YOUR_AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'YOUR_AWS_SECRET_KEY')
BUCKET         = os.getenv('S3_BUCKET',             'vehicle-mileage-project')
REGION         = os.getenv('AWS_REGION',            'ap-south-1')

# ── Model Cache (loaded once at startup) ──────────────────────────────────
_model         = None
_feature_names = None
_model_metrics = None

def get_s3_client():
    return boto3.client(
        's3',
        region_name=REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

def load_model():
    """Load model and feature names from S3 — cached after first load."""
    global _model, _feature_names, _model_metrics

    if _model is not None:
        return _model, _feature_names, _model_metrics

    s3 = get_s3_client()
    print("Loading model from S3...")

    # Load model
    obj    = s3.get_object(Bucket=BUCKET, Key='models/best_model.pkl')
    _model = joblib.load(io.BytesIO(obj['Body'].read()))
    print("Model loaded!")

    # Load feature names (saved as X_tree.columns.tolist() in NB07)
    try:
        obj            = s3.get_object(Bucket=BUCKET, Key='models/feature_names.pkl')
        _feature_names = joblib.load(io.BytesIO(obj['Body'].read()))
        print(f"Feature names loaded: {len(_feature_names)} features")
    except Exception as e:
        print(f"Warning: Could not load feature names: {e}")
        _feature_names = None

    # Load model metrics (optional — won't crash if missing)
    try:
        obj            = s3.get_object(Bucket=BUCKET, Key='models/model_metrics.json')
        _model_metrics = json.loads(obj['Body'].read())
        print(f"Model metrics loaded")
    except Exception:
        _model_metrics = {}

    return _model, _feature_names, _model_metrics


# ── Risk Category ──────────────────────────────────────────────────────────
def get_risk_category(predicted_kms):
    if predicted_kms < 10000:
        return "Low",       0.7,  "Basic Third Party"
    elif predicted_kms < 25000:
        return "Medium",    1.0,  "Standard Comprehensive"
    elif predicted_kms < 40000:
        return "High",      1.4,  "Comprehensive Plus"
    elif predicted_kms < 70000:
        return "Very High", 1.8,  "Commercial Vehicle Plan"
    else:
        return "Extreme",   2.5,  "Rideshare Commercial Plan"

def get_premium_estimate(predicted_kms, base_premium=12000):
    risk_cat, multiplier, product = get_risk_category(predicted_kms)
    return {
        'risk_category'      : risk_cat,
        'premium_multiplier' : multiplier,
        'estimated_premium'  : round(base_premium * multiplier),
        'recommended_product': product
    }


# ── Encoding Maps — must match NB06 exactly ───────────────────────────────

# Binary columns (NB06 Step 1)
BINARY_MAPS = {
    'transmission'  : {'Manual': 0, 'Automatic': 1},
    'marital_status': {'Single': 0, 'Married': 1},
    'gender'        : {'Female': 0, 'Male': 1, 'Other': 1},
}

# Ordinal columns (NB06 Step 2)
ORDINAL_MAPS = {
    'vehicle_condition': {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3},
    'road_quality'     : {'Poor': 0, 'Average': 1, 'Good': 2},
    'region_type'      : {'Rural': 0, 'Semi-urban': 1, 'Urban': 2},
    'education'        : {'School': 0, 'Graduate': 1, 'PG': 2},
    'traffic_index'    : {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 3},
}

# NB06 Step 3: One-hot encoded columns (drop_first=True, dtype=int)
# These become 0/1 columns in X_tree — we simulate by integer coding
ONE_HOT_COLS = [
    'fuel_type', 'brand_tier', 'vehicle_segment', 'city_tier',
    'weekend_trips', 'work_from_home', 'parking_type', 'insurance_type',
    'color', 'occupation'
]

# NB06 Step 4: Remaining object cols → pd.Categorical().codes (integer)
# These include: state, vehicle_brand, vehicle_model etc.


def preprocess_input(record_dict, feature_names):
    """
    Encode a single record to match NB06 X_tree encoding exactly.
    NB06 used:
      - Binary maps for 2-value cols
      - Ordinal maps for ordered cols
      - pd.Categorical().codes for all remaining object cols (incl. one-hot candidates)
      - No target encoding, no StandardScaler
    """
    df = pd.DataFrame([record_dict])

    # Standardise text (same as NB06)
    for col in ['fuel_type', 'city_tier', 'gender', 'transmission',
                'occupation', 'vehicle_segment', 'brand_tier',
                'vehicle_brand', 'vehicle_model', 'state', 'color',
                'weekend_trips', 'work_from_home', 'parking_type',
                'insurance_type', 'road_quality', 'region_type',
                'vehicle_condition', 'education', 'traffic_index',
                'marital_status']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Compute vehicle_age from manufacture_year
    if 'manufacture_year' in df.columns:
        df['manufacture_year'] = pd.to_numeric(df['manufacture_year'], errors='coerce')
        df['vehicle_age']      = 2024 - df['manufacture_year']
        df['vehicle_age']      = df['vehicle_age'].clip(lower=1, upper=20)

    # Step 1: Binary encoding
    for col, mapping in BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    # Step 2: Ordinal encoding
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    # Steps 3+4: All remaining object columns → integer codes
    # (NB06 used pd.Categorical().codes for everything not already encoded)
    for col in df.select_dtypes(include='object').columns:
        df[col] = pd.Categorical(df[col]).codes

    df = df.fillna(0)

    # Align columns to exactly what model was trained on
    if feature_names:
        for f in feature_names:
            if f not in df.columns:
                df[f] = 0          # missing feature → 0
        df = df[feature_names]     # exact order as training

    return df


# ── Main Predict Function ─────────────────────────────────────────────────
def predict(record_dict):
    """
    Main prediction function.

    Args:
        record_dict: dict with vehicle and owner features

    Returns:
        dict with predicted_kms, risk_category, premium, shap, model_info
    """
    model, feature_names, model_metrics = load_model()

    # Preprocess
    X = preprocess_input(record_dict, feature_names)

    # Predict
    predicted_kms = float(model.predict(X)[0])
    predicted_kms = max(0, round(predicted_kms))

    # Risk and premium
    premium_info = get_premium_estimate(predicted_kms)

    # SHAP explanation (optional — skips silently if unavailable)
    shap_explanation = {}
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X)
        feat_list = feature_names if feature_names else list(X.columns)
        shap_df   = pd.DataFrame({
            'feature'   : feat_list,
            'shap_value': shap_vals[0]
        }).sort_values('shap_value', key=abs, ascending=False).head(8)
        shap_explanation = dict(zip(
            shap_df['feature'].tolist(),
            [round(v, 0) for v in shap_df['shap_value'].tolist()]
        ))
    except Exception:
        pass

    return {
        'predicted_kms'      : predicted_kms,
        'predicted_kms_range': {
            'low' : max(0, round(predicted_kms * 0.9)),
            'high': round(predicted_kms * 1.1)
        },
        'risk_category'      : premium_info['risk_category'],
        'premium_multiplier' : premium_info['premium_multiplier'],
        'estimated_premium'  : premium_info['estimated_premium'],
        'recommended_product': premium_info['recommended_product'],
        'shap_explanation'   : shap_explanation,
        'model_info'         : {
            'model_name': model_metrics.get('model', 'Best Model'),
            'mape'      : model_metrics.get('mape', 'N/A'),
            'trained_on': model_metrics.get('trained_on', 'N/A'),
            'version'   : model_metrics.get('version', 'v1')
        }
    }


def get_model_info():
    """Returns current model metadata."""
    _, _, model_metrics = load_model()
    return model_metrics


if __name__ == '__main__':
    sample = {
        'vehicle_brand'     : 'Maruti',
        'vehicle_model'     : 'Swift',
        'vehicle_segment'   : 'Hatchback',
        'brand_tier'        : 'Budget',
        'manufacture_year'  : 2019,
        'engine_cc'         : 1200,
        'fuel_type'         : 'Petrol',
        'transmission'      : 'Manual',
        'num_owners'        : 1,
        'vehicle_condition' : 'Good',
        'is_premium_brand'  : 0,
        'selling_price'     : 600000,
        'insurance_type'    : 'Comprehensive',
        'num_accidents'     : 0,
        'occupation'        : 'Salaried',
        'owner_age'         : 32,
        'annual_income_lakh': 8.5,
        'gender'            : 'Male',
        'num_drivers'       : 1,
        'has_children'      : 1,
        'num_children'      : 1,
        'driving_exp_years' : 8,
        'education'         : 'Graduate',
        'marital_status'    : 'Married',
        'household_size'    : 3,
        'num_vehicles_owned': 1,
        'city_tier'         : 'Metro',
        'state'             : 'Karnataka',
        'home_to_office_km' : 18,
        'road_quality'      : 'Good',
        'traffic_index'     : 'High',
        'has_metro_rail'    : 1,
        'region_type'       : 'Urban',
        'highway_access'    : 0,
        'parking_type'      : 'Society',
        'weekend_trips'     : 'Monthly',
        'uses_for_business' : 0,
        'uses_for_tourism'  : 0,
        'is_rideshare'      : 0,
        'night_driving'     : 0,
        'work_from_home'    : 'Sometimes',
        'monthly_fuel_spend': 3500,
        'toll_spend_monthly': 200,
        'daily_trips'       : 4,
    }

    result = predict(sample)
    print("\nPrediction Result:")
    print(f"  Predicted KMs    : {result['predicted_kms']:,} km/year")
    print(f"  Range            : {result['predicted_kms_range']['low']:,} - {result['predicted_kms_range']['high']:,} km")
    print(f"  Risk Category    : {result['risk_category']}")
    print(f"  Premium Estimate : Rs {result['estimated_premium']:,}")
    print(f"  Recommended Plan : {result['recommended_product']}")
    print(f"  Model            : {result['model_info']['model_name']}")
