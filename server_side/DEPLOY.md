# Google Cloud Deployment Guide

This guide explains how to deploy the People Counting server to Google Cloud Run.

## Prerequisites

- [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed
- Docker installed locally
- A Google Cloud project with billing enabled

## Quick Start

### 1. Authenticate with Google Cloud

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
```

### 2. Enable Required APIs

```bash
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable cloudbuild.googleapis.com
```

### 3. Deploy Backend

```bash
cd server_side

# Build and push to Container Registry
gcloud builds submit --tag gcr.io/xenon-diorama-477010-g1/people-counting-api .

# Deploy to Cloud Run
gcloud run deploy people-counting-api \
    --image gcr.io/xenon-diorama-477010-g1/people-counting-api \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 2Gi \
    --cpu 2 \
    --port 8000
```

### 4. Deploy Frontend

```bash
cd server_side/frontend

# Set the backend URL (use the URL from step 3)
export BACKEND_URL=https://people-counting-api-xxxxx.run.app

# Build with backend URL
gcloud builds submit --tag gcr.io/xenon-diorama-477010-g1/people-counting-frontend \
    --build-arg NEXT_PUBLIC_API_URL=$BACKEND_URL

# Deploy to Cloud Run
gcloud run deploy people-counting-frontend \
    --image gcr.io/xenon-diorama-477010-g1/people-counting-frontend \
    --platform managed \
    --region us-central1 \
    --allow-unauthenticated \
    --memory 512Mi \
    --port 3000 \
    --set-env-vars NEXT_PUBLIC_API_URL=$BACKEND_URL
```

## Local Testing with Docker

```bash
cd server_side

# Build and run both services
docker-compose up --build

# Access:
# - Frontend: http://localhost:3000
# - Backend API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

## Environment Variables

### Backend

| Variable | Description | Default |
| -------- | ----------- | ------- |
| `PORT`   | Server port | `8000`  |

### Frontend

| Variable              | Description     | Default                 |
| --------------------- | --------------- | ----------------------- |
| `NEXT_PUBLIC_API_URL` | Backend API URL | `http://localhost:8000` |
| `PORT`                | Server port     | `3000`                  |

## Costs Estimation

Google Cloud Run pricing (as of 2024):

- **CPU**: $0.00002400 per vCPU-second
- **Memory**: $0.00000250 per GiB-second
- **Free tier**: 2 million requests/month

For a typical deployment:

- Idle: ~$0-5/month (scales to zero)
- Active: ~$10-30/month depending on usage

## Troubleshooting

### CORS Issues

Ensure the backend allows your frontend domain in CORS settings.

### Memory Errors

Increase memory allocation:

```bash
gcloud run services update people-counting-api --memory 4Gi
```

### Cold Start Latency

Set minimum instances to avoid cold starts:

```bash
gcloud run services update people-counting-api --min-instances 1
```
