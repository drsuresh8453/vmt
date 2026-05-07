"""
detect_drift.py — Vehicle Mileage MLOps Project
Author: Suresh D R | AI Product Developer & Technology Mentor
DV Analytics

Uses Evidently AI to detect data drift between reference and current data.
Generates HTML drift report.
Sends report via email using SNS/SES.
Triggers SageMaker retraining pipeline if drift is detected.
"""

import pandas as pd
import numpy as np
import boto3
import io
import os
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from scipy import stats
import warnings
from dotenv import load_dotenv
load_dotenv()
warnings.filterwarnings('ignore')

# ── Evidently AI ───────────────────────────────────────────────────────────
try:
    from evidently.report import Report
    from evidently.metric_suite import MetricSuite
    from evidently.metrics import (
        DatasetDriftMetric,
        DataDriftTable,
        ColumnDriftMetric,
        DatasetMissingValuesSummary,
        DatasetSummaryMetric,
    )
    from evidently.test_suite import TestSuite
    from evidently.tests import (
        TestNumberOfColumnsWithMissingValues,
        TestNumberOfRowsWithMissingValues,
        TestShareOfDriftedColumns,
    )
    EVIDENTLY_AVAILABLE = True
    print("Evidently AI loaded successfully")
except ImportError:
    EVIDENTLY_AVAILABLE = False
    print("Warning: Evidently not installed — using scipy KS test for drift")

# ── AWS Configuration ──────────────────────────────────────────────────────
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY_ID',     'YOUR_AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_ACCESS_KEY', 'YOUR_AWS_SECRET_KEY')
BUCKET         = os.getenv('S3_BUCKET',             'vehicle-mileage-project')
REGION         = os.getenv('AWS_REGION',            'ap-south-1')
PIPELINE_NAME  = os.getenv('SAGEMAKER_PIPELINE',    'vehicle-mileage-pipeline')
SNS_TOPIC_ARN  = os.getenv('SNS_TOPIC_ARN',         '')
ALERT_EMAIL    = os.getenv('ALERT_EMAIL',           'suresh@dvanalytics.com')

# Features to monitor for drift
NUMERICAL_FEATURES = [
    'annual_kms', 'home_to_office_km', 'selling_price',
    'annual_commute_km', 'family_trip_km', 'daily_trips',
    'driving_exp_years', 'num_vehicles_owned', 'num_children'
]

CATEGORICAL_FEATURES = [
    'fuel_type', 'brand_tier', 'occupation', 'is_rideshare',
    'uses_for_business', 'is_high_mileage', 'is_premium_brand',
    'highway_access', 'night_driving', 'has_metro_rail'
]

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

def upload_to_s3(local_path, s3_key):
    s3 = get_s3_client()
    s3.upload_file(local_path, BUCKET, s3_key)
    print(f"Uploaded to S3: s3://{BUCKET}/{s3_key}")

# ── Evidently Drift Report ─────────────────────────────────────────────────
def run_evidently_drift_report(reference_df, current_df, report_path):
    """Generate full Evidently HTML drift report."""
    print("\nGenerating Evidently AI Drift Report...")

    common_cols = [c for c in NUMERICAL_FEATURES + CATEGORICAL_FEATURES
                   if c in reference_df.columns and c in current_df.columns]

    ref = reference_df[common_cols].copy()
    cur = current_df[common_cols].copy()

    report = Report(metrics=[
        DatasetDriftMetric(),
        DataDriftTable(),
        DatasetSummaryMetric(),
        DatasetMissingValuesSummary(),
    ])

    report.run(reference_data=ref, current_data=cur)
    report.save_html(report_path)
    print(f"Evidently report saved: {report_path}")

    report_dict   = report.as_dict()
    drift_metrics = report_dict['metrics'][0]['result']
    dataset_drift = drift_metrics.get('dataset_drift', False)
    drifted_cols  = drift_metrics.get('number_of_drifted_columns', 0)
    total_cols    = drift_metrics.get('number_of_columns', len(common_cols))
    drift_share   = drift_metrics.get('share_of_drifted_columns', 0)

    print(f"Dataset Drift    : {dataset_drift}")
    print(f"Drifted Columns  : {drifted_cols} / {total_cols}")
    print(f"Drift Share      : {drift_share:.2%}")

    return {
        'dataset_drift'  : dataset_drift,
        'drifted_columns': drifted_cols,
        'total_columns'  : total_cols,
        'drift_share'    : round(drift_share, 4),
        'report_path'    : report_path,
    }

# ── Fallback KS Test Drift Detection ──────────────────────────────────────
def run_ks_drift_detection(reference_df, current_df):
    """Scipy KS test drift detection — fallback when Evidently not available."""
    print("\nRunning KS Test Drift Detection (scipy)...")

    drifted = []
    results = {}

    for feature in NUMERICAL_FEATURES:
        if feature in reference_df.columns and feature in current_df.columns:
            ref_col = pd.to_numeric(reference_df[feature], errors='coerce').dropna()
            cur_col = pd.to_numeric(current_df[feature],   errors='coerce').dropna()
            stat, p_value = stats.ks_2samp(ref_col, cur_col)
            is_drifted = p_value < 0.05
            results[feature] = {
                'p_value': round(p_value, 6),
                'drifted': is_drifted,
                'ks_stat': round(stat, 4)
            }
            if is_drifted:
                drifted.append(feature)
                print(f"  DRIFT: {feature:<30} p={p_value:.6f}  KS={stat:.4f}")
            else:
                print(f"  OK   : {feature:<30} p={p_value:.6f}  KS={stat:.4f}")

    for feature in CATEGORICAL_FEATURES:
        if feature in reference_df.columns and feature in current_df.columns:
            try:
                ref_counts = reference_df[feature].value_counts()
                cur_counts = current_df[feature].value_counts()
                all_cats   = set(ref_counts.index) | set(cur_counts.index)
                ref_arr    = np.array([ref_counts.get(c, 0) for c in all_cats])
                cur_arr    = np.array([cur_counts.get(c, 0) for c in all_cats])

                if ref_arr.sum() > 0 and cur_arr.sum() > 0:
                    ref_arr = ref_arr / ref_arr.sum()
                    cur_arr = cur_arr / cur_arr.sum()
                    stat, p_value = stats.chisquare(cur_arr + 1e-10, ref_arr + 1e-10)
                    is_drifted = p_value < 0.05
                    results[feature] = {
                        'p_value': round(p_value, 6),
                        'drifted': is_drifted,
                        'test'   : 'chi_square'
                    }
                    if is_drifted:
                        drifted.append(feature)
                        print(f"  DRIFT: {feature:<30} p={p_value:.6f}")
                    else:
                        print(f"  OK   : {feature:<30} p={p_value:.6f}")
            except Exception as e:
                print(f"  SKIP : {feature} ({e})")

    total_features = len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)
    drift_ratio    = len(drifted) / max(total_features, 1)
    dataset_drift  = drift_ratio > 0.25   # FIX: lowered from 0.3 to 0.25

    print(f"\nDrifted features : {len(drifted)} / {total_features}")
    print(f"Drift ratio      : {drift_ratio:.2%}")
    print(f"Dataset drift    : {dataset_drift}")

    return {
        'dataset_drift'   : dataset_drift,
        'drifted_features': drifted,
        'drift_ratio'     : round(drift_ratio, 4),
        'feature_results' : results,
    }

# ── Generate HTML Drift Report (without Evidently) ────────────────────────
def generate_html_report(reference_df, current_df, drift_results, report_path):
    """Generate HTML report when Evidently not available."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    drifted   = drift_results.get('drifted_features', [])
    total     = len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Vehicle Mileage — Data Drift Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .header {{ background: #1F4E79; color: white; padding: 20px; border-radius: 8px; }}
            .summary {{ background: white; padding: 20px; margin: 10px 0; border-radius: 8px;
                        border-left: 5px solid {'#C00000' if drift_results['dataset_drift'] else '#28a745'}; }}
            table {{ width: 100%; border-collapse: collapse; background: white; }}
            th {{ background: #1F4E79; color: white; padding: 10px; }}
            td {{ padding: 8px; border: 1px solid #ddd; }}
            .drift {{ color: #C00000; font-weight: bold; }}
            .ok    {{ color: #28a745; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Vehicle Mileage Prediction — Data Drift Report</h1>
            <p>Author: Suresh D R | DV Analytics | Generated: {timestamp}</p>
        </div>

        <div class="summary">
            <h2>{'DRIFT DETECTED — Retraining Triggered' if drift_results['dataset_drift'] else 'No Significant Drift — Model Stable'}</h2>
            <p><strong>Drifted Features:</strong> {len(drifted)} / {total}</p>
            <p><strong>Drift Ratio:</strong> {drift_results.get('drift_ratio', 0):.2%}</p>
            <p><strong>Reference Data:</strong> {len(reference_df):,} records (training data — 06_encoded_tree.csv)</p>
            <p><strong>Current Data:</strong> {len(current_df):,} records (new policies — data/new_policies.csv)</p>
        </div>

        <h2>Feature Drift Details</h2>
        <table>
            <tr>
                <th>Feature</th>
                <th>P-Value</th>
                <th>Status</th>
            </tr>
    """

    for feat, res in drift_results.get('feature_results', {}).items():
        status_class = 'drift' if res['drifted'] else 'ok'
        status_text  = 'DRIFTED' if res['drifted'] else 'OK'
        html += f"""
            <tr>
                <td>{feat}</td>
                <td>{res['p_value']}</td>
                <td class="{status_class}">{status_text}</td>
            </tr>
        """

    html += """
        </table>

        <h2>Statistical Summary — Reference vs Current</h2>
        <table>
            <tr><th>Feature</th><th>Reference Mean</th><th>Current Mean</th><th>Change %</th></tr>
    """

    for col in NUMERICAL_FEATURES:
        if col in reference_df.columns and col in current_df.columns:
            ref_col  = pd.to_numeric(reference_df[col], errors='coerce')
            cur_col  = pd.to_numeric(current_df[col],   errors='coerce')
            ref_mean = ref_col.mean()
            cur_mean = cur_col.mean()
            pct      = ((cur_mean - ref_mean) / ref_mean * 100) if ref_mean != 0 else 0
            color    = '#C00000' if abs(pct) > 20 else '#28a745'
            html += f"""
            <tr>
                <td>{col}</td>
                <td>{ref_mean:,.2f}</td>
                <td>{cur_mean:,.2f}</td>
                <td style="color:{color}; font-weight:bold">{pct:+.1f}%</td>
            </tr>
            """

    html += """
        </table>
        <p><em>Report generated by Vehicle Mileage MLOps Pipeline | DV Analytics</em></p>
    </body>
    </html>
    """

    with open(report_path, 'w') as f:
        f.write(html)
    print(f"HTML report saved: {report_path}")
    return report_path

# ── Send Email with Report ─────────────────────────────────────────────────
def send_email_with_report(drift_results, report_path, recipient_email):
    """Send drift report via SNS and upload to S3."""
    print(f"\nSending drift report to {recipient_email}...")

    subject = (
        "DRIFT DETECTED — Vehicle Mileage Model Retraining Triggered"
        if drift_results['dataset_drift']
        else "No Drift — Vehicle Mileage Model Stable"
    )

    body = f"""
Vehicle Mileage Prediction — Daily Drift Report
================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Author: Suresh D R | DV Analytics

DRIFT STATUS: {'DRIFT DETECTED' if drift_results['dataset_drift'] else 'NO DRIFT'}

Summary:
  Drifted Features : {len(drift_results.get('drifted_features', []))}
  Total Features   : {len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)}
  Drift Ratio      : {drift_results.get('drift_ratio', 0):.2%}

{'ACTION TAKEN: SageMaker Pipeline triggered for retraining.' if drift_results['dataset_drift'] else 'NO ACTION NEEDED: Model is still accurate.'}

---
Vehicle Mileage MLOps Pipeline | DV Analytics
    """

    if SNS_TOPIC_ARN:
        try:
            sns = boto3.client('sns', region_name=REGION,
                               aws_access_key_id=AWS_ACCESS_KEY,
                               aws_secret_access_key=AWS_SECRET_KEY)
            sns.publish(TopicArn=SNS_TOPIC_ARN, Subject=subject, Message=body)
            print(f"SNS alert sent!")
        except Exception as e:
            print(f"SNS failed: {e}")

    try:
        s3_report_key = f"reports/drift_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"
        upload_to_s3(report_path, s3_report_key)
        print(f"Report uploaded: s3://{BUCKET}/{s3_report_key}")
    except Exception as e:
        print(f"S3 upload failed: {e}")

    print("Notification complete!")

# ── Trigger SageMaker Pipeline ─────────────────────────────────────────────
def trigger_sagemaker_pipeline():
    """Trigger SageMaker retraining pipeline."""
    print("\nTriggering SageMaker retraining pipeline...")
    try:
        sm = boto3.client('sagemaker', region_name=REGION,
                          aws_access_key_id=AWS_ACCESS_KEY,
                          aws_secret_access_key=AWS_SECRET_KEY)
        response = sm.start_pipeline_execution(
            PipelineName=PIPELINE_NAME,
            PipelineExecutionDisplayName=f"auto-retrain-drift-{datetime.now().strftime('%Y%m%d-%H%M')}"
        )
        arn = response['PipelineExecutionArn']
        print(f"Pipeline triggered: {arn}")
        return arn
    except Exception as e:
        print(f"SageMaker trigger failed: {e}")
        return None

# ── Main Drift Detection Function ──────────────────────────────────────────
def main(recipient_email=None):
    print("=" * 60)
    print("VEHICLE MILEAGE — DAILY DRIFT DETECTION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    if recipient_email is None:
        recipient_email = ALERT_EMAIL

    # ── Load Reference Data (exact training data) ─────────────────────────
    print("\nLoading reference data from S3...")
    try:
        reference_df = read_csv_s3('reference/training_reference.csv')
        print(f"Reference loaded from reference/training_reference.csv: {len(reference_df):,} rows")
    except Exception:
        print("reference/training_reference.csv not found — using data/06_encoded_tree.csv...")
        reference_df = read_csv_s3('data/06_encoded_tree.csv')
        print(f"Reference loaded from data/06_encoded_tree.csv: {len(reference_df):,} rows")

    # ── Load Current Data (new incoming policies) ─────────────────────────
    print("\nLoading current data from S3...")
    try:
        current_df = read_csv_s3('current/new_policies.csv')
        print(f"Current loaded from current/new_policies.csv: {len(current_df):,} rows")
    except Exception:
        try:
            current_df = read_csv_s3('data/new_policies.csv')
            print(f"Current loaded from data/new_policies.csv: {len(current_df):,} rows")
        except Exception as e2:
            print(f"No current data found: {e2}")
            print("Using reference data as current (no drift expected)...")
            current_df = reference_df.copy()

    print(f"\nReference : {len(reference_df):,} rows x {reference_df.shape[1]} cols")
    print(f"Current   : {len(current_df):,} rows x {current_df.shape[1]} cols")

    # ── Run Drift Detection ────────────────────────────────────────────────
    # FIX: use current directory instead of /tmp/ (Windows compatible)
    report_path = f"drift_report_{datetime.now().strftime('%Y%m%d_%H%M')}.html"

    if EVIDENTLY_AVAILABLE:
        print("\nUsing Evidently AI for drift detection...")
        drift_results = run_evidently_drift_report(reference_df, current_df, report_path)
    else:
        print("\nUsing scipy KS test for drift detection...")
        drift_results = run_ks_drift_detection(reference_df, current_df)
        generate_html_report(reference_df, current_df, drift_results, report_path)

    # ── Save Drift Log to S3 ───────────────────────────────────────────────
    s3 = get_s3_client()
    drift_log = {
        'timestamp'       : datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_drift'   : drift_results['dataset_drift'],
        'drift_ratio'     : drift_results.get('drift_ratio', 0),
        'drifted_features': drift_results.get('drifted_features', []),
        'reference_rows'  : len(reference_df),
        'current_rows'    : len(current_df),
    }
    s3.put_object(
        Bucket=BUCKET,
        Key=f"reports/drift_log_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        Body=json.dumps(drift_log, indent=2)
    )
    print(f"Drift log saved to S3")

    # ── Send Notification ──────────────────────────────────────────────────
    send_email_with_report(drift_results, report_path, recipient_email)

    # ── Trigger Retraining if Drift Detected ──────────────────────────────
    pipeline_arn = None
    if drift_results['dataset_drift']:
        print("\n" + "=" * 60)
        print("DRIFT DETECTED — TRIGGERING RETRAINING")
        print("=" * 60)
        pipeline_arn = trigger_sagemaker_pipeline()
    else:
        print("\nNo significant drift detected — model is stable.")
        print("No retraining needed.")

    print("\n" + "=" * 60)
    print("DRIFT CHECK COMPLETE")
    print(f"Dataset Drift    : {drift_results['dataset_drift']}")
    print(f"Drifted Features : {drift_results.get('drifted_features', [])}")
    print(f"Pipeline ARN     : {pipeline_arn or 'Not triggered'}")
    print(f"Finished         : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    return drift_results


# ── Lambda Handler (for AWS Lambda) ───────────────────────────────────────
def lambda_handler(event, context):
    """AWS Lambda entry point — called by EventBridge daily."""
    recipient = event.get('recipient_email', ALERT_EMAIL)
    results   = main(recipient_email=recipient)
    return {
        'statusCode': 200,
        'body': json.dumps({
            'dataset_drift'   : results['dataset_drift'],
            'drift_ratio'     : results.get('drift_ratio', 0),
            'drifted_features': results.get('drifted_features', []),
        })
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--email', default=ALERT_EMAIL,
                        help='Email to send drift report to')
    args = parser.parse_args()
    main(recipient_email=args.email)
