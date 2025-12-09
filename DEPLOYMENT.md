# Deployment Guide - RAG Finance System

> Complete guide for deploying the RAG Finance System to production

## Table of Contents

- [Quick Start](#quick-start)
- [Platform Comparison](#platform-comparison)
- [Railway Deployment](#railway-deployment)
- [Render Deployment](#render-deployment)
- [GitHub Actions CI/CD](#github-actions-cicd)
- [Environment Configuration](#environment-configuration)
- [Monitoring Setup](#monitoring-setup)
- [Security Best Practices](#security-best-practices)
- [Scaling Guidelines](#scaling-guidelines)
- [Troubleshooting](#troubleshooting)

---

## Quick Start

### Prerequisites Checklist

- [ ] OpenAI API key ([Get here](https://platform.openai.com/api-keys))
- [ ] GitHub repository with the code
- [ ] Git installed locally
- [ ] (Optional) Docker installed for local testing
- [ ] (Optional) Pinecone account for production vector store

### 5-Minute Railway Deployment

```bash
# 1. Install Railway CLI
npm i -g @railway/cli

# 2. Login and initialize
railway login
railway init

# 3. Set required environment variables
railway variables set OPENAI_API_KEY=sk-your-key-here
railway variables set VECTOR_STORE_MODE=chroma
railway variables set MAX_CORRECTIONS=2

# 4. Deploy
railway up

# 5. Check deployment
railway open
```

### 5-Minute Render Deployment

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Blueprint"
3. Connect your GitHub repository
4. Render will detect `render.yaml` automatically
5. Set `OPENAI_API_KEY` in environment variables
6. Click "Apply" to deploy

---

## Platform Comparison

| Feature | Railway | Render | Recommendation |
|---------|---------|--------|----------------|
| **Pricing** | $5/month + usage | $7-25/month | Railway for prototypes, Render for production |
| **Free Tier** | $5 credit (trial) | 750 hrs/month | Render for free testing |
| **Auto-scaling** | ❌ Manual | ✅ Yes (Standard+) | Render for high traffic |
| **Persistent Storage** | ✅ Volumes | ✅ Disks | Both excellent |
| **Docker Support** | ✅ Native | ✅ Native | Both excellent |
| **CLI Tool** | ✅ Excellent | ⚠️ Basic | Railway for CLI lovers |
| **Blueprint/IaC** | ✅ railway.json | ✅ render.yaml | Both excellent |
| **Custom Domains** | ✅ Free | ✅ Free | Both excellent |
| **SSL/HTTPS** | ✅ Automatic | ✅ Automatic | Both excellent |
| **Deployment Speed** | ⚡ Very fast | ⚡ Fast | Railway slightly faster |
| **Logs & Metrics** | ✅ Good | ✅ Good | Render has better UI |
| **Database Options** | ✅ Many | ✅ Many | Both excellent |

**Verdict:**
- **Choose Railway if**: You want faster deployments, better CLI, or prototyping
- **Choose Render if**: You need auto-scaling, better UI, or generous free tier

---

## Railway Deployment

### Method 1: Railway CLI (Recommended)

#### Step 1: Install CLI

```bash
# Using npm
npm install -g @railway/cli

# Using Homebrew (macOS)
brew install railway

# Using Scoop (Windows)
scoop install railway
```

#### Step 2: Authentication

```bash
railway login
```

This opens your browser for authentication.

#### Step 3: Create Project

```bash
# Option A: New project
railway init

# Option B: Link existing project
railway link [project-id]
```

#### Step 4: Configure Environment Variables

```bash
# Required variables
railway variables set OPENAI_API_KEY="sk-your-key-here"
railway variables set VECTOR_STORE_MODE="chroma"
railway variables set MAX_CORRECTIONS="2"
railway variables set LOG_LEVEL="INFO"

# Optional: Pinecone (for production)
railway variables set PINECONE_API_KEY="your-pinecone-key"
railway variables set PINECONE_INDEX_NAME="financial-docs"

# Optional: Observability
railway variables set ENABLE_TRACING="false"
railway variables set ENABLE_METRICS="true"

# View all variables
railway variables
```

#### Step 5: Deploy

```bash
# Deploy current branch
railway up

# Deploy specific branch
railway up --branch main

# Deploy with detached mode (CI/CD)
railway up --detach
```

#### Step 6: Monitor Deployment

```bash
# View logs
railway logs

# Open service in browser
railway open

# Get service URL
railway status
```

### Method 2: Railway Dashboard

1. Go to [Railway Dashboard](https://railway.app/new)
2. Click "New Project" → "Deploy from GitHub repo"
3. Select your repository
4. Railway auto-detects `railway.json`
5. Add environment variables in settings:
   - `OPENAI_API_KEY`
   - `VECTOR_STORE_MODE`
   - `MAX_CORRECTIONS`
6. Click "Deploy"

### Railway Configuration Details

The `railway.json` file configures:

```json
{
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "numReplicas": 1,
    "startCommand": "uvicorn src.api.main:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 300
  }
}
```

**Key Features:**
- ✅ Automatic health checks every 30 seconds
- ✅ Auto-restart on failure (max 3 retries)
- ✅ Environment-specific configs (production/staging)
- ✅ Dockerfile-based builds

### Adding Persistent Storage (Railway)

```bash
# Create volume
railway volume create chroma-data

# Mount volume
railway volume mount chroma-data /app/data
```

Or via dashboard: Settings → Volumes → Add Volume

### Custom Domain (Railway)

```bash
# Add custom domain
railway domain add yourdomain.com

# View domains
railway domain list
```

---

## Render Deployment

### Method 1: Blueprint Deployment (Recommended)

#### Step 1: Connect Repository

1. Fork or push code to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com/)
3. Click "New +" → "Blueprint"

#### Step 2: Configure Blueprint

Render automatically detects `render.yaml`:

```yaml
services:
  - type: web
    name: rag-finance-api
    env: docker
    dockerfilePath: ./Dockerfile
    plan: starter
    region: oregon
```

#### Step 3: Set Environment Variables

In the Render dashboard, add:

| Variable | Value |
|----------|-------|
| `OPENAI_API_KEY` | `sk-your-key-here` |
| `VECTOR_STORE_MODE` | `chroma` |
| `MAX_CORRECTIONS` | `2` |
| `LOG_LEVEL` | `INFO` |

#### Step 4: Deploy

Click "Apply" to deploy. Render will:
1. Clone your repository
2. Build Docker image
3. Deploy to production
4. Assign HTTPS URL

### Method 2: Manual Web Service

1. Dashboard → "New +" → "Web Service"
2. Connect GitHub repository
3. Configure:
   - **Name**: `rag-finance-api`
   - **Environment**: `Docker`
   - **Dockerfile Path**: `./Dockerfile`
   - **Region**: Choose closest to your users
4. Set environment variables (see above)
5. Click "Create Web Service"

### Adding Persistent Disk (Render)

Required for ChromaDB data persistence:

1. Go to your service dashboard
2. "Storage" → "Add Disk"
3. Configure:
   - **Name**: `chroma-data`
   - **Mount Path**: `/app/data`
   - **Size**: 1 GB (scale as needed)
4. Save and redeploy

**Cost**: $0.25/GB/month

### Auto-scaling (Render)

Available on Standard plan and above:

```yaml
scaling:
  minInstances: 1
  maxInstances: 3
  targetMemoryPercent: 80
  targetCPUPercent: 80
```

### Custom Domain (Render)

1. Service Dashboard → "Settings"
2. "Custom Domains" → "Add Custom Domain"
3. Enter your domain (e.g., `api.yourdomain.com`)
4. Add CNAME record to your DNS:
   ```
   CNAME api.yourdomain.com -> your-service.onrender.com
   ```
5. Render automatically provisions SSL certificate

---

## GitHub Actions CI/CD

### Setup Instructions

#### Step 1: Fork or Clone Repository

```bash
git clone https://github.com/yourusername/rag-finance-system.git
cd rag-finance-system
```

#### Step 2: Add GitHub Secrets

Go to: Repository → Settings → Secrets and variables → Actions

**Required Secrets:**

| Secret | Description | How to Get |
|--------|-------------|------------|
| `OPENAI_API_KEY` | OpenAI API key | [Platform Dashboard](https://platform.openai.com/api-keys) |
| `RAILWAY_TOKEN` | Railway API token | Railway Dashboard → Account → Tokens |
| `RAILWAY_PROJECT_ID` | Project ID | Railway project settings |
| `RAILWAY_URL` | Deployment URL | `https://your-app.railway.app` |
| `RENDER_API_KEY` | Render API key | Render Dashboard → Account → API Keys |
| `RENDER_SERVICE_ID` | Service ID | From service URL or API |
| `RENDER_URL` | Deployment URL | `https://your-app.onrender.com` |

**Optional Secrets:**

| Secret | Description |
|--------|-------------|
| `DOCKER_USERNAME` | Docker Hub username (for image registry) |
| `DOCKER_PASSWORD` | Docker Hub password/token |

#### Step 3: Configure Workflow

The workflow (`.github/workflows/deploy.yml`) automatically runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`

**Workflow stages:**

1. **Test**: Run linting and pytest with coverage
2. **Build**: Build and test Docker image
3. **Deploy**: Deploy to Railway and/or Render
4. **Health Check**: Verify deployment success
5. **Notify**: Alert on failures

#### Step 4: Trigger Deployment

```bash
# Make a change
git add .
git commit -m "Deploy to production"
git push origin main
```

GitHub Actions automatically:
- ✅ Runs all tests
- ✅ Builds Docker image
- ✅ Deploys to Railway
- ✅ Deploys to Render
- ✅ Runs health checks
- ✅ Reports status

### Customizing the Workflow

#### Deploy Only to Railway

Comment out the `deploy-render` job:

```yaml
# deploy-render:
#   name: Deploy to Render
#   ...
```

#### Deploy Only on Main Branch

Already configured:

```yaml
if: github.event_name == 'push' && github.ref == 'refs/heads/main'
```

#### Add Slack Notifications

Add to the `notify` job:

```yaml
- name: Slack Notification
  uses: 8398a7/action-slack@v3
  with:
    status: ${{ job.status }}
    webhook_url: ${{ secrets.SLACK_WEBHOOK }}
```

---

## Environment Configuration

### Complete Environment Variables Reference

#### Core Configuration

```bash
# OpenAI API (REQUIRED)
OPENAI_API_KEY=sk-your-key-here

# Vector Store Mode (REQUIRED)
VECTOR_STORE_MODE=chroma  # Options: chroma, pinecone

# API Server (REQUIRED)
API_HOST=0.0.0.0
API_PORT=8000  # Railway uses $PORT automatically
```

#### RAG System Configuration

```bash
# Self-correction attempts
MAX_CORRECTIONS=2

# Retrieval settings
RETRIEVAL_TOP_K=5
RELEVANCE_THRESHOLD=0.7

# LLM settings
GENERATION_TEMPERATURE=0.2
MAX_TOKENS_PER_REQUEST=4000
```

#### Pinecone Configuration (Optional)

```bash
PINECONE_API_KEY=your-pinecone-key
PINECONE_INDEX_NAME=financial-docs
PINECONE_ENVIRONMENT=us-east-1
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
```

#### Observability Configuration

```bash
# Logging
LOG_LEVEL=INFO  # Options: DEBUG, INFO, WARNING, ERROR

# OpenTelemetry
OTEL_SERVICE_NAME=rag-finance-system
ENABLE_TRACING=false  # true if using Jaeger
ENABLE_METRICS=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318

# Prometheus
PROMETHEUS_PORT=8001
```

#### Development Settings

```bash
# Development mode
DEBUG=true
RELOAD=true  # Auto-reload on code changes
```

#### Security Settings (Production)

```bash
# API Authentication (optional)
API_SECRET_KEY=your-secret-key-here

# CORS Origins (comma-separated)
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Rate limiting
MAX_COST_PER_QUERY=0.50
RATE_LIMIT_PER_MINUTE=60
```

### Environment-Specific Configurations

#### Local Development

```bash
VECTOR_STORE_MODE=chroma
LOG_LEVEL=DEBUG
ENABLE_TRACING=true
ENABLE_METRICS=true
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318
```

#### Production (Railway/Render)

```bash
VECTOR_STORE_MODE=chroma  # or pinecone
LOG_LEVEL=INFO
ENABLE_TRACING=false  # Unless using managed Jaeger
ENABLE_METRICS=true
MAX_CORRECTIONS=2
```

---

## Monitoring Setup

### Local Monitoring (Docker Compose)

Start full observability stack:

```bash
docker-compose up -d
```

Access dashboards:
- **API Docs**: http://localhost:8000/docs
- **Jaeger**: http://localhost:16686
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000 (admin/admin)

### Production Monitoring

#### Option 1: Grafana Cloud (Recommended)

1. Sign up at [Grafana Cloud](https://grafana.com/products/cloud/)
2. Get your API key and endpoint
3. Set environment variables:
   ```bash
   ENABLE_TRACING=true
   OTEL_EXPORTER_OTLP_ENDPOINT=https://your-endpoint:4318
   ```

**Cost**: Free tier (10k series, 50GB traces/month)

#### Option 2: Jaeger Cloud

1. Sign up for managed Jaeger service
2. Configure endpoint:
   ```bash
   ENABLE_TRACING=true
   OTEL_EXPORTER_OTLP_ENDPOINT=https://your-jaeger-endpoint:4318
   ```

#### Option 3: Self-Hosted on Railway/Render

Deploy monitoring stack separately:

```bash
# Deploy Grafana
railway add --service grafana/grafana

# Deploy Prometheus
railway add --service prom/prometheus
```

### Key Metrics to Monitor

| Metric | Alert Threshold | Action |
|--------|----------------|--------|
| Response time (p95) | > 5 seconds | Scale up, optimize queries |
| Error rate | > 5% | Check logs, API key validity |
| Token usage | Unexpected spike | Check for abuse, add rate limiting |
| Memory usage | > 80% | Scale up plan |
| Disk usage | > 80% | Increase disk size |

### Setting Up Alerts

**Railway:**
```bash
railway notifications add --type email --event deploy-failed
```

**Render:**
- Dashboard → Service → Notifications
- Add email or Slack webhook

---

## Security Best Practices

### 1. API Key Management

✅ **Do:**
- Store API keys in environment variables
- Use platform secret management
- Rotate keys regularly
- Use separate keys for dev/prod

❌ **Don't:**
- Commit keys to git
- Share keys in chat/email
- Use same key across environments

### 2. Environment Variables

```bash
# Good: Using platform secrets
railway variables set OPENAI_API_KEY=$OPENAI_KEY --secret

# Bad: Hardcoding in Dockerfile
ENV OPENAI_API_KEY=sk-...  # NEVER DO THIS
```

### 3. Network Security

- Enable HTTPS (automatic on Railway/Render)
- Use custom domains with SSL
- Configure CORS properly:
  ```python
  CORS_ORIGINS=https://yourdomain.com
  ```

### 4. Rate Limiting

Implement in production:

```python
from fastapi import Request
from slowapi import Limiter

limiter = Limiter(key_func=lambda request: request.client.host)

@app.post("/query")
@limiter.limit("10/minute")
async def query_endpoint(...):
    ...
```

### 5. Input Validation

Already implemented with Pydantic:

```python
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    max_retries: int = Field(2, ge=0, le=5)
```

### 6. Dependency Security

Regular updates:

```bash
# Check for vulnerabilities
pip-audit

# Update dependencies
pip install --upgrade -r requirements.txt
```

---

## Scaling Guidelines

### Vertical Scaling (Upgrade Plan)

**When to scale:**
- Response time > 5 seconds
- Memory usage > 80%
- CPU usage consistently > 70%

**Railway Plans:**
- **Starter**: $5/month + usage → Basic apps
- **Developer**: $20/month → Small production
- **Team**: $50/month → Multiple services

**Render Plans:**
- **Starter**: $7/month → Testing
- **Standard**: $25/month → Production (recommended)
- **Pro**: $85/month → High traffic

### Horizontal Scaling

**Render Auto-scaling:**

```yaml
scaling:
  minInstances: 2
  maxInstances: 10
  targetCPUPercent: 70
```

**Railway:** Manual replica management

### Database Scaling

**ChromaDB** (included):
- Good for: < 1M vectors
- Storage: Persistent disk
- Cost: Included in hosting

**Pinecone** (recommended for scale):
- Good for: > 1M vectors
- Storage: Managed serverless
- Cost: $0.096/GB/month
- Setup:
  ```bash
  railway variables set VECTOR_STORE_MODE=pinecone
  railway variables set PINECONE_API_KEY=your-key
  ```

### Caching Layer

Implement Redis for frequently asked questions:

```bash
# Add Redis service
railway add redis

# Update code to use caching
@cache.cached(timeout=3600)
def query_rag(question: str):
    ...
```

---

## Troubleshooting

### Build Failures

**Issue**: Docker build fails

**Solutions:**
```bash
# Test build locally
docker build -t rag-finance-system .

# Check Dockerfile syntax
docker build --no-cache -t rag-finance-system .

# Verify all files committed
git status
git add .
git commit -m "Fix deployment"
```

### Health Check Failures

**Issue**: `/health` endpoint returns 500

**Solutions:**
1. Check logs for errors
2. Verify environment variables set
3. Test locally:
   ```bash
   docker run -p 8000:8000 \
     -e OPENAI_API_KEY=$OPENAI_KEY \
     -e VECTOR_STORE_MODE=chroma \
     rag-finance-system
   ```

### High Costs

**Issue**: Unexpected OpenAI charges

**Solutions:**
1. Check token usage in logs
2. Add cost limits:
   ```bash
   MAX_COST_PER_QUERY=0.50
   ```
3. Use gpt-4o-mini for all agents:
   ```python
   model_name="gpt-4o-mini"  # 60% cheaper
   ```
4. Implement caching
5. Add rate limiting

### Slow Response Times

**Issue**: Queries take > 10 seconds

**Solutions:**
1. Switch to Pinecone (faster than ChromaDB)
2. Reduce retrieval count:
   ```bash
   RETRIEVAL_TOP_K=3  # Down from 5
   ```
3. Lower relevance threshold:
   ```bash
   RELEVANCE_THRESHOLD=0.6  # Down from 0.7
   ```
4. Optimize chunking strategy
5. Upgrade hosting plan

### Disk Space Issues

**Issue**: ChromaDB disk full

**Solutions:**
1. Increase disk size (Railway/Render settings)
2. Clean old data:
   ```bash
   railway run python scripts/cleanup_chroma.py
   ```
3. Switch to Pinecone (no disk needed)

### Memory Issues

**Issue**: Application crashes (OOM)

**Solutions:**
1. Upgrade to larger plan
2. Reduce batch sizes in processing
3. Optimize vector store queries
4. Add memory limits:
   ```python
   MAX_DOCUMENTS_PER_QUERY=5
   ```

---

## Support & Resources

### Documentation
- [Railway Docs](https://docs.railway.app/)
- [Render Docs](https://render.com/docs)
- [FastAPI Docs](https://fastapi.tiangolo.com/)
- [LangChain Docs](https://python.langchain.com/)

### Community
- [Railway Discord](https://discord.gg/railway)
- [Render Community](https://community.render.com/)
- [GitHub Issues](your-repo/issues)

### Cost Calculators
- [Railway Pricing](https://railway.app/pricing)
- [Render Pricing Calculator](https://render.com/pricing)
- [OpenAI Pricing](https://openai.com/pricing)

---

**Last Updated**: 2025-12-09

