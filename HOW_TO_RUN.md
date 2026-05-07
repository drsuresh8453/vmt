# Vehicle Mileage MLOps — Complete How to Run Guide
## Author: Suresh D R | AI Product Developer & Technology Mentor
## DV Analytics

---

## The Golden Rule — You Do Nothing After Setup

```
CODE CHANGE → git push → GitHub Actions → tests → Docker → ECR → EKS → live ✅
You do: git push
System does: everything else

DATA DRIFT → Lambda runs daily → Evidently detects drift → email sent →
SageMaker Pipeline → pulls code from GitHub → trains on SageMaker →
new model to S3 → Docker → ECR → EKS → live ✅
You do: nothing
System does: everything
```

---

## Project Structure

```
vehicle-mileage-mlops/
├── src/
│   ├── preprocess.py         ← All cleaning + encoding (from NB2+NB3+NB6)
│   ├── train.py              ← Model training (from NB7)
│   ├── predict.py            ← Prediction function — loads model from S3
│   ├── app.py                ← Streamlit web app for insurance agents
│   └── api.py                ← FastAPI REST endpoint
├── monitoring/
│   ├── detect_drift.py       ← Evidently AI drift detection + email report
│   └── track_experiment.py  ← MLflow experiment tracking
├── tests/
│   └── test_model.py         ← 8 automated tests
├── k8s/
│   ├── deployment.yml        ← Production (2 replicas)
│   └── deployment-uat.yml   ← UAT (1 replica)
├── .github/workflows/
│   └── mlops_pipeline.yml   ← CI/CD pipeline
├── pipeline/
│   └── sagemaker_pipeline.py← SageMaker automated retraining
├── Dockerfile
├── requirements.txt
├── .gitignore
└── HOW_TO_RUN.md             ← This file
```

---

## What Is Already Done (from Google Colab)

```
✅ 7 notebooks ran in Google Colab
✅ Training data processed and saved to S3
✅ Best model trained and saved to S3

S3 bucket: vehicle-mileage-project
  raw/vehicle_mileage_raw.csv              ← original raw data
  data/06_encoded_tree.csv                 ← for tree models
  data/06_encoded_linear.csv              ← for linear models
  data/06_feature_names.csv               ← feature list
  models/best_model.pkl                   ← trained best model
  models/feature_names.pkl                ← feature names list
  models/model_metrics.json              ← MAPE, MAE, R2 etc
```

---

## PHASE 1 — Local Setup and Test

### Step 1 — Download and Open in VS Code

```bash
# Download and unzip vehicle-mileage-mlops.zip
# Right click → Extract All → Desktop
# Open VS Code → File → Open Folder → select vehicle-mileage-mlops
```

### Step 2 — Create Virtual Environment

```bash
# In VS Code Terminal (Git Bash on Windows):
python -m venv venv
source venv/Scripts/activate      # Windows
# source venv/bin/activate         # Mac/Linux

pip install -r requirements.txt
```

### Step 3 — Test the App Locally

```bash
# Test Streamlit app (loads model from S3)
streamlit run src/app.py
# Open http://localhost:8501 → fill form → click Predict → Ctrl+C

# Test FastAPI
python src/api.py
# Open http://localhost:8000/docs → test /predict endpoint → Ctrl+C
```

### Step 4 — Run All 8 Tests

```bash
pytest tests/test_model.py -v

# Expected output:
# test_model_exists_in_s3           PASSED
# test_model_loads                  PASSED
# test_prediction_returns_number    PASSED
# test_prediction_valid_range       PASSED
# test_risk_category_valid          PASSED
# test_premium_estimate_positive    PASSED
# test_rideshare_prediction_higher  PASSED
# test_model_mape_below_threshold   PASSED
# 8 passed in X seconds ✅
```

### Step 5 — Test Drift Detection Locally

```bash
python monitoring/detect_drift.py --email your@email.com

# Expected output:
# Running Evidently AI Drift Detection...
# Report saved: /tmp/drift_report_YYYYMMDD.html
# Email sent ✅
# No drift detected (first run — same data) ✅
```

---

## PHASE 2 — GitHub Repository and CI/CD

### Step 6 — Create GitHub Repository

```
1. Go to https://github.com
2. Click + → New repository
3. Name: vehicle-mileage-mlops
4. Public → Create repository
```

### Step 7 — Push Code to GitHub

```bash
git init
git config --global user.name "Your Name"
git config --global user.email "your@email.com"
git add .
git commit -m "Initial commit — Vehicle Mileage MLOps pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/vehicle-mileage-mlops.git
git push -u origin main
```

### Step 8 — Add GitHub Secrets

```
GitHub repo → Settings → Secrets and variables → Actions → New secret

Add these secrets:
  Name                   Value
  ──────────────────────────────────────────────────────
  AWS_ACCESS_KEY_ID      your AWS access key
  AWS_SECRET_ACCESS_KEY  your AWS secret key
  AWS_REGION             ap-south-1
  ECR_REGISTRY           YOUR_ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com
  ECR_REPOSITORY         vehicle-mileage
  EKS_CLUSTER_NAME       vehicle-mileage-cluster
```

### Step 9 — Verify CI/CD Pipeline Runs

```
GitHub → Actions tab
You will see: Vehicle Mileage MLOps Pipeline running
  ✅ Run Tests (8 tests)
  ✅ Build and Push Docker
  ✅ Deploy to EKS

Every future git push triggers this automatically.
```

---

## PHASE 3 — Docker and ECR

### Step 10 — Create ECR Repository

```bash
aws ecr create-repository \
  --repository-name vehicle-mileage \
  --region ap-south-1

# Note your ECR URL:
# YOUR_ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com/vehicle-mileage
```

### Step 11 — Build and Push Docker Image

⚠️ Switch to mobile hotspot for Docker commands

```bash
# Build image
docker build -t vehicle-mileage:v1.0 .

# Login to ECR
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin \
  YOUR_ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com

# Tag and push
docker tag vehicle-mileage:v1.0 \
  YOUR_ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com/vehicle-mileage:latest
docker push \
  YOUR_ACCOUNT_ID.dkr.ecr.ap-south-1.amazonaws.com/vehicle-mileage:latest

# Pushed ✅
```

---

## PHASE 4 — EKS Cluster and First Deployment

### Step 12 — Create EKS Cluster

⚠️ Costs Rs 600-800/day. Delete when done.

```bash
eksctl create cluster \
  --name vehicle-mileage-cluster \
  --region ap-south-1 \
  --nodegroup-name workers \
  --node-type t3.medium \
  --nodes 2 \
  --nodes-min 1 \
  --nodes-max 3 \
  --managed

# Wait 15-20 minutes...
kubectl get nodes
# 2 nodes Ready ✅
```

### Step 13 — Update k8s YAML and Deploy

```bash
# Replace ACCOUNT_ID in deployment.yml with your actual AWS account ID
# Open k8s/deployment.yml → find ACCOUNT_ID → replace → save

# Configure kubectl
aws eks update-kubeconfig \
  --region ap-south-1 \
  --name vehicle-mileage-cluster

# Create AWS secrets in Kubernetes
kubectl create secret generic aws-secrets \
  --from-literal=aws-access-key-id=YOUR_ACCESS_KEY \
  --from-literal=aws-secret-access-key=YOUR_SECRET_KEY

# Deploy
kubectl apply -f k8s/deployment.yml
kubectl get pods
kubectl get service vehicle-mileage-service

# Get external URL
# Open browser → http://EXTERNAL-IP:8501
# App is LIVE on AWS! 🎉
```

---

## PHASE 5 — S3 Data Setup for Drift Detection

### Step 14 — Upload Reference Baseline

```bash
# Upload training data as reference baseline
# Evidently will compare new data against this daily
aws s3 cp data/reference/training_reference.csv \
  s3://vehicle-mileage-project/reference/training_reference.csv

# If you don't have this file locally, copy from existing S3 data:
aws s3 cp \
  s3://vehicle-mileage-project/raw/vehicle_mileage_raw.csv \
  s3://vehicle-mileage-project/reference/training_reference.csv

echo "Reference baseline uploaded ✅"
```

### Step 15 — Verify S3 Structure

```bash
aws s3 ls s3://vehicle-mileage-project/ --recursive

# Should show:
# reference/training_reference.csv   ← baseline for drift
# current/new_policies.csv           ← new data (upload daily)
# data/06_encoded_tree.csv           ← model training data
# models/best_model.pkl              ← trained model
# models/feature_names.pkl           ← feature list
# models/model_metrics.json         ← MAPE, MAE etc
```

---

## PHASE 6 — SageMaker Pipeline (Automated Retraining)

### Step 16 — Create SageMaker Notebook

```
1. AWS Console → SageMaker AI → Notebooks
2. Create notebook instance:
   Name          → vehicle-mileage-notebook
   Instance type → ml.t3.medium
   IAM role      → Create new role → Any S3 → Create role
3. Click Create notebook instance
4. Wait 3-5 minutes → InService ✅
5. Click Open JupyterLab → File → New → Terminal
```

### Step 17 — Clone Code and Setup Pipeline

```bash
# Inside SageMaker terminal:
git clone https://github.com/YOUR_USERNAME/vehicle-mileage-mlops.git
cd vehicle-mileage-mlops
pip install -r requirements.txt

# Update GITHUB_REPO in pipeline/sagemaker_pipeline.py
# Replace YOUR_USERNAME with your actual GitHub username

# Create the pipeline (run once only)
python pipeline/sagemaker_pipeline.py

# Pipeline created: vehicle-mileage-pipeline ✅
# It will run automatically when triggered by Lambda/EventBridge
```

### Step 18 — Stop SageMaker Notebook

⚠️ ml.t3.medium costs Rs 5/hour. Always stop after use.

```
AWS Console → SageMaker → Notebooks → select → Stop
```

---

## PHASE 7 — Lambda + EventBridge (Daily Drift Check)

### Step 19 — Create Lambda Function

```
1. AWS Console → Lambda → Create function
2. Settings:
   Name    → vehicle-mileage-drift-checker
   Runtime → Python 3.10
   Role    → Create new role with basic Lambda permissions
3. Click Create function
```

Paste this code in the Lambda editor:

```python
import boto3
import json
import os
import sys

def lambda_handler(event, context):
    """
    Runs daily via EventBridge.
    Checks for data drift using Evidently AI.
    Triggers SageMaker Pipeline if drift found.
    Sends drift report via email.
    """
    # Lambda has limited disk space — use /tmp
    sys.path.append('/tmp')

    s3     = boto3.client('s3', region_name='ap-south-1')
    sm     = boto3.client('sagemaker', region_name='ap-south-1')
    sns    = boto3.client('sns', region_name='ap-south-1')
    BUCKET = 'vehicle-mileage-project'

    # Download data
    s3.download_file(BUCKET, 'reference/training_reference.csv', '/tmp/reference.csv')
    try:
        s3.download_file(BUCKET, 'current/new_policies.csv', '/tmp/current.csv')
    except:
        print("No new data — using reference as current")
        s3.download_file(BUCKET, 'reference/training_reference.csv', '/tmp/current.csv')

    # KS test drift detection
    import pandas as pd
    from scipy import stats
    import numpy as np

    reference = pd.read_csv('/tmp/reference.csv')
    current   = pd.read_csv('/tmp/current.csv')

    FEATURES = ['home_to_office_km', 'annual_kms', 'monthly_fuel_spend',
                'annual_income_lakh', 'engine_cc', 'owner_age', 'daily_trips']

    drifted = []
    for feat in FEATURES:
        if feat in reference.columns and feat in current.columns:
            stat, p = stats.ks_2samp(
                reference[feat].dropna(),
                current[feat].dropna()
            )
            if p < 0.01:
                drifted.append(feat)
                print(f"DRIFT: {feat} (p={p:.6f})")
            else:
                print(f"OK:    {feat} (p={p:.6f})")

    drift_ratio   = len(drifted) / len(FEATURES)
    dataset_drift = drift_ratio > 0.3

    print(f"Drifted: {len(drifted)}/{len(FEATURES)} ({drift_ratio:.2%})")
    print(f"Dataset drift: {dataset_drift}")

    if dataset_drift:
        # Trigger SageMaker retraining
        response = sm.start_pipeline_execution(
            PipelineName='vehicle-mileage-pipeline',
            PipelineExecutionDisplayName='auto-retrain-drift-detected'
        )
        print(f"Pipeline triggered: {response['PipelineExecutionArn']}")

        # Send SNS alert
        sns.publish(
            TopicArn=os.environ.get('SNS_TOPIC_ARN', ''),
            Subject='DRIFT DETECTED — Vehicle Mileage Model Retraining Triggered',
            Message=f'Drifted features: {drifted}\nDrift ratio: {drift_ratio:.2%}\nPipeline started automatically.'
        )
    else:
        print("No drift — model still good ✅")
        sns.publish(
            TopicArn=os.environ.get('SNS_TOPIC_ARN', ''),
            Subject='✅ No Drift — Vehicle Mileage Model Stable',
            Message=f'Daily drift check: No significant drift detected.\nDrift ratio: {drift_ratio:.2%}\nModel is stable.'
        )

    return {
        'statusCode': 200,
        'body': json.dumps({
            'dataset_drift': dataset_drift,
            'drifted_features': drifted,
            'drift_ratio': drift_ratio
        })
    }
```

Click **Deploy** ✅

### Step 20 — Add Lambda Permissions

```
Lambda → vehicle-mileage-drift-checker
→ Configuration → Permissions → click role name
→ Add permissions → Attach policies:

  AmazonS3FullAccess       ✅
  AmazonSageMakerFullAccess ✅
  AmazonSNSFullAccess      ✅
```

### Step 21 — Add Environment Variables to Lambda

```
Lambda → Configuration → Environment variables:
  SNS_TOPIC_ARN  →  arn:aws:sns:ap-south-1:YOUR_ACCOUNT_ID:mlops-alerts
```

### Step 22 — Create EventBridge Schedule

```
AWS Console → EventBridge → Schedules → Create schedule:
  Name        → daily-vehicle-drift-check
  Schedule    → Recurring schedule
  Cron        → 0 2 * * ? *   (daily at 2am UTC = 7:30am IST)
  Target      → Lambda function
  Select      → vehicle-mileage-drift-checker

Click Create schedule ✅
```

Now every day at 7:30am IST — drift check runs automatically.
If drift found → SageMaker Pipeline retrains → new model deployed.
You do nothing. ✅

---

## PHASE 8 — SNS Email Alerts

### Step 23 — Create SNS Topic and Subscribe Email

```bash
# Create SNS topic
aws sns create-topic \
  --name mlops-alerts \
  --region ap-south-1

# Subscribe your email
aws sns subscribe \
  --topic-arn arn:aws:sns:ap-south-1:YOUR_ACCOUNT_ID:mlops-alerts \
  --protocol email \
  --notification-endpoint your@email.com

# Check email and click CONFIRM SUBSCRIPTION link ✅
```

---

## PHASE 9 — Full Loop Test

### Step 24 — Test Trigger 1 (Code Change → Auto Deploy)

```bash
# Make any small change
# Open src/train.py → change a comment → save

git add .
git commit -m "Test CI/CD pipeline trigger"
git push

# Watch GitHub Actions:
# ✅ Tests run (8 tests)
# ✅ Docker image built
# ✅ Pushed to ECR
# ✅ EKS rolling update
# ✅ New code live
# You did only: git push ✅
```

### Step 25 — Test Trigger 2 (Data Drift → Auto Retrain)

```bash
# Upload drifted data to simulate real-world data change
# (modify some distributions to trigger drift)
aws s3 cp data/demo_drift/drifted_current.csv \
  s3://vehicle-mileage-project/current/new_policies.csv

# Trigger Lambda manually for demo:
# AWS Console → Lambda → vehicle-mileage-drift-checker → Test
# Create test event → {} → Test

# You will see:
# DRIFT: home_to_office_km (p=0.000001)
# DRIFT: annual_kms (p=0.000001)
# Dataset drift: True
# Pipeline triggered: arn:aws:sagemaker:... ✅
# Email sent ✅

# SageMaker Pipeline then:
# → Pulls train.py from GitHub ✅
# → Pulls data from S3 ✅
# → Trains new model on ml.m5.large ✅
# → Saves model to S3 ✅
# → Triggers GitHub Actions ✅
# → Docker build → ECR → EKS ✅
# → New model live ✅
```

---

## PHASE 10 — Monitor Everything

```bash
# EKS Pods
kubectl get pods -l app=vehicle-mileage
kubectl logs <pod-name>

# Get app URL
kubectl get service vehicle-mileage-service

# SageMaker Pipelines
# AWS Console → SageMaker → Pipelines → vehicle-mileage-pipeline

# CloudWatch Logs
# AWS Console → CloudWatch → Log groups
# /aws/lambda/vehicle-mileage-drift-checker

# MLflow Experiments
export MLFLOW_TRACKING_URI=s3://vehicle-mileage-project/mlflow
mlflow ui
# Open http://localhost:5000

# Drift Reports
aws s3 ls s3://vehicle-mileage-project/reports/
```

---

## PHASE 11 — Cleanup (Save Cost)

```bash
# Delete EKS cluster (Rs 600-800/day)
eksctl delete cluster \
  --name vehicle-mileage-cluster \
  --region ap-south-1

# Delete ECR repository
aws ecr delete-repository \
  --repository-name vehicle-mileage \
  --region ap-south-1 \
  --force

# Stop SageMaker notebook
# AWS Console → SageMaker → Notebooks → Stop

# Delete Lambda and EventBridge
# AWS Console → Lambda → Delete
# AWS Console → EventBridge → Delete

# Keep S3 (cheap, stores model and data)
# Keep GitHub (free)
```

---

## All Commands Quick Reference

```bash
# Local
python -m venv venv && source venv/Scripts/activate
pip install -r requirements.txt
streamlit run src/app.py
python src/api.py
pytest tests/test_model.py -v
python monitoring/detect_drift.py --email your@email.com

# Git
git add . && git commit -m "message" && git push

# S3
aws s3 ls s3://vehicle-mileage-project/ --recursive
aws s3 cp your_file.csv s3://vehicle-mileage-project/current/new_policies.csv

# ECR
aws ecr create-repository --repository-name vehicle-mileage --region ap-south-1
aws ecr get-login-password --region ap-south-1 | \
  docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.ap-south-1.amazonaws.com
docker build -t vehicle-mileage:v1.0 .
docker tag vehicle-mileage:v1.0 YOUR_ACCOUNT.dkr.ecr.ap-south-1.amazonaws.com/vehicle-mileage:latest
docker push YOUR_ACCOUNT.dkr.ecr.ap-south-1.amazonaws.com/vehicle-mileage:latest

# EKS
eksctl create cluster --name vehicle-mileage-cluster --region ap-south-1 \
  --nodegroup-name workers --node-type t3.medium --nodes 2 --managed
aws eks update-kubeconfig --region ap-south-1 --name vehicle-mileage-cluster
kubectl apply -f k8s/deployment.yml
kubectl get pods
kubectl get service vehicle-mileage-service
kubectl rollout status deployment/vehicle-mileage-app

# Cleanup
eksctl delete cluster --name vehicle-mileage-cluster --region ap-south-1
```

---

## The Complete Automated Flow Summary

```
ONE TIME SETUP:
  GitHub repo          ✅
  ECR repository       ✅
  EKS cluster          ✅
  S3 baseline data     ✅
  SageMaker Pipeline   ✅
  Lambda + EventBridge ✅
  SNS email alerts     ✅

AFTER SETUP — FULLY AUTOMATED:

YOU: git push
SYSTEM:
  → Tests run (8 automated tests)
  → Docker image built with new code
  → Pushed to ECR
  → EKS rolling update (zero downtime)
  → New code live ✅

YOU: upload new data to S3/current/
SYSTEM:
  → Lambda checks drift next day at 7:30am IST
  → Evidently AI generates drift report
  → Report sent to your email ✅
  → If drift: SageMaker Pipeline triggered
  → Pulls latest code from GitHub
  → Trains on new data in SageMaker
  → Evaluates new vs old model
  → If better: saves to S3
  → Triggers GitHub Actions
  → Docker build → ECR → EKS
  → New better model live ✅

YOU: nothing (happens automatically)
SYSTEM:
  → Lambda runs daily at 7:30am IST
  → Evidently drift check
  → Email report sent regardless of drift
  → If drift found: full retrain pipeline
```

---

**Author: Suresh D R | AI Product Developer & Technology Mentor**
**DV Analytics — Industry MLOps Projects Series**
