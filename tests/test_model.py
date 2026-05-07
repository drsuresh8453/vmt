"""
test_model.py — Vehicle Mileage MLOps Project
Author: Suresh D R | AI Product Developer & Technology Mentor
DV Analytics

8 automated tests that run on every git push via GitHub Actions.
All tests must pass before Docker image is built and deployed.
"""

import pytest
import sys
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# ── Sample Vehicle Record for Testing ─────────────────────────────────────
SAMPLE_RECORD = {
    'vehicle_brand'      : 'Maruti',
    'vehicle_model'      : 'Swift',
    'vehicle_segment'    : 'Hatchback',
    'brand_tier'         : 'Budget',
    'manufacture_year'   : 2019,
    'vehicle_age'        : 5,
    'engine_cc'          : 1200,
    'fuel_type'          : 'Petrol',
    'transmission'       : 'Manual',
    'num_owners'         : 1,
    'vehicle_condition'  : 'Good',
    'is_premium_brand'   : 0,
    'seating_capacity'   : 5,
    'color'              : 'White',
    'selling_price'      : 600000,
    'insurance_type'     : 'Comprehensive',
    'num_accidents'      : 0,
    'occupation'         : 'Salaried',
    'owner_age'          : 32,
    'annual_income_lakh' : 8.5,
    'gender'             : 'Male',
    'num_drivers'        : 1,
    'has_children'       : 1,
    'num_children'       : 1,
    'driving_exp_years'  : 8,
    'education'          : 'Graduate',
    'marital_status'     : 'Married',
    'household_size'     : 3,
    'num_vehicles_owned' : 1,
    'city_tier'          : 'Metro',
    'state'              : 'Karnataka',
    'home_to_office_km'  : 18,
    'road_quality'       : 'Good',
    'traffic_index'      : 'High',
    'has_metro_rail'     : 1,
    'region_type'        : 'Urban',
    'highway_access'     : 0,
    'parking_type'       : 'Society',
    'weekend_trips'      : 'Monthly',
    'uses_for_business'  : 0,
    'uses_for_tourism'   : 0,
    'is_rideshare'       : 0,
    'night_driving'      : 0,
    'work_from_home'     : 'Sometimes',
    'monthly_fuel_spend' : 3500,
    'toll_spend_monthly' : 200,
    'daily_trips'        : 4,
}


# ── Test 1: Model File Exists in S3 ───────────────────────────────────────
def test_model_exists_in_s3():
    """Test that the trained model exists in S3."""
    import boto3
    AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID',     'YOUR_AWS_ACCESS_KEY')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'YOUR_AWS_SECRET_KEY')
    BUCKET         = os.getenv('S3_BUCKET',             'vehicle-mileage-project')
    REGION         = os.getenv('AWS_REGION',            'ap-south-1')

    s3 = boto3.client('s3', region_name=REGION,
                       aws_access_key_id=AWS_ACCESS_KEY,
                       aws_secret_access_key=AWS_SECRET_KEY)
    try:
        s3.head_object(Bucket=BUCKET, Key='models/best_model.pkl')
        print("\nTest 1 PASSED: Model exists in S3")
    except Exception as e:
        pytest.fail(f"Model not found in S3: {e}")


# ── Test 2: Model Loads Without Error ─────────────────────────────────────
def test_model_loads():
    """Test that the model loads from S3 without any errors."""
    from predict import load_model
    try:
        model, feature_names, metrics = load_model()
        assert model is not None, "Model is None"
        print("\nTest 2 PASSED: Model loads successfully")
    except Exception as e:
        pytest.fail(f"Model failed to load: {e}")


# ── Test 3: Prediction Returns a Number ───────────────────────────────────
def test_prediction_returns_number():
    """Test that prediction returns a numeric value."""
    from predict import predict
    result = predict(SAMPLE_RECORD)
    assert isinstance(result['predicted_kms'], (int, float)), \
        "Prediction is not a number"
    assert not np.isnan(result['predicted_kms']), \
        "Prediction is NaN"
    print(f"\nTest 3 PASSED: Prediction = {result['predicted_kms']:,} km")


# ── Test 4: Prediction is in Valid Range ──────────────────────────────────
def test_prediction_valid_range():
    """Test that prediction is in valid business range (0 to 200,000 km)."""
    from predict import predict
    result = predict(SAMPLE_RECORD)
    assert 0 < result['predicted_kms'] < 200000, \
        f"Prediction out of range: {result['predicted_kms']}"
    print(f"\nTest 4 PASSED: Prediction in valid range: {result['predicted_kms']:,} km")


# ── Test 5: Risk Category is Valid ────────────────────────────────────────
def test_risk_category_valid():
    """Test that risk category is one of the expected values."""
    from predict import predict
    result     = predict(SAMPLE_RECORD)
    valid_cats = ['Low', 'Medium', 'High', 'Very High', 'Extreme']
    assert result['risk_category'] in valid_cats, \
        f"Invalid risk category: {result['risk_category']}"
    print(f"\nTest 5 PASSED: Risk category = {result['risk_category']}")


# ── Test 6: Premium Estimate is Positive ──────────────────────────────────
def test_premium_estimate_positive():
    """Test that estimated premium is a positive number."""
    from predict import predict
    result = predict(SAMPLE_RECORD)
    assert result['estimated_premium'] > 0, \
        f"Premium estimate is not positive: {result['estimated_premium']}"
    print(f"\nTest 6 PASSED: Premium estimate = Rs {result['estimated_premium']:,}")


# ── Test 7: Rideshare Prediction Higher Than Personal ─────────────────────
def test_rideshare_prediction_higher():
    """Test that rideshare vehicles predict higher mileage than personal."""
    from predict import predict

    personal_record            = SAMPLE_RECORD.copy()
    personal_record['is_rideshare'] = 0

    rideshare_record            = SAMPLE_RECORD.copy()
    rideshare_record['is_rideshare'] = 1

    personal_result  = predict(personal_record)
    rideshare_result = predict(rideshare_record)

    assert rideshare_result['predicted_kms'] > personal_result['predicted_kms'], \
        (f"Rideshare ({rideshare_result['predicted_kms']:,}) should be higher "
         f"than personal ({personal_result['predicted_kms']:,})")

    print(f"\nTest 7 PASSED: "
          f"Rideshare={rideshare_result['predicted_kms']:,} km > "
          f"Personal={personal_result['predicted_kms']:,} km")


# ── Test 8: Model MAPE is Below Threshold ─────────────────────────────────
def test_model_mape_below_threshold():
    """Test that model MAPE is below 20% (production threshold)."""
    from predict import get_model_info
    info = get_model_info()
    mape = info.get('mape', 0)
    assert mape < 20, \
        f"Model MAPE {mape}% exceeds 20% threshold — not production ready"
    print(f"\nTest 8 PASSED: Model MAPE = {mape}% (below 20% threshold)")


# ── Run All Tests ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 60)
    print("RUNNING ALL 8 TESTS")
    print("=" * 60)
    pytest.main([__file__, '-v', '--tb=short'])
