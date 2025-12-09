# Step-by-Step Deployment Guide

> Complete walkthrough from API keys to production deployment

**Time to complete:** 20-30 minutes  
**Difficulty:** Beginner-friendly  
**Cost:** ~$10-30/month after free trials

---

## 📋 Table of Contents

1. [Getting Your OpenAI API Key](#step-1-getting-your-openai-api-key) (Required - 5 mins)
2. [Setting Up Your Environment File](#step-2-setting-up-your-environment-file) (2 mins)
3. [Testing Locally](#step-3-testing-locally) (5 mins)
4. [Choose Your Deployment Platform](#step-4-choose-your-deployment-platform)
5. [Deploy to Railway](#option-a-deploy-to-railway) (10 mins)
6. [Deploy to Render](#option-b-deploy-to-render) (10 mins)
7. [Set Up CI/CD (Optional)](#step-6-set-up-cicd-optional) (10 mins)
8. [Verify Deployment](#step-7-verify-deployment) (2 mins)

---

## Step 1: Getting Your OpenAI API Key

**⚠️ This is the ONLY required API key to run the system**

### 1.1 Create OpenAI Account

1. Go to: **https://platform.openai.com/**
2. Click **"Sign Up"** (or "Log In" if you have an account)
3. Sign up with:
   - Email address, OR
   - Google account, OR
   - Microsoft account
4. Verify your email address
5. Complete phone number verification

### 1.2 Add Payment Method (Required)

**Why?** OpenAI requires a payment method for API access (even with free credits)

1. Go to: **https://platform.openai.com/account/billing/overview**
2. Click **"Add payment method"**
3. Enter your credit/debit card information
4. **Set a spending limit** (recommended: $50/month to start)
   - Go to: https://platform.openai.com/account/billing/limits
   - Set "Hard limit" to control maximum spend

**Free Trial:**
- New accounts get $5 in free credits
- Credits expire after 3 months
- After that, pay-as-you-go pricing applies

### 1.3 Create Your API Key

1. Go to: **https://platform.openai.com/api-keys**
2. Click **"Create new secret key"**
3. Give it a name: `RAG Finance System`
4. Click **"Create secret key"**
5. **IMPORTANT:** Copy the key immediately!
   - It looks like: `sk-proj-AbCdEf123456...`
   - You won't be able to see it again!
   - Store it safely (we'll add it to `.env` next)

**Security Tips:**
- ✅ Never share your API key
- ✅ Don't commit it to GitHub
- ✅ Don't share in chat/email
- ✅ You can always create a new key if needed

---

## Step 2: Setting Up Your Environment File

### 2.1 Locate Your .env File

The `.env` file has been created in your project root:
```
C:\Users\ngyat\OneDrive\Documents\projects\rag-finance-system\.env
```

### 2.2 Edit the .env File

1. Open `.env` in your editor (VS Code, Notepad++, etc.)
2. Find this line:
   ```bash
   OPENAI_API_KEY=sk-your-openai-api-key-here
   ```
3. Replace `sk-your-openai-api-key-here` with your actual API key:
   ```bash
   OPENAI_API_KEY=sk-proj-AbCdEf123456YourActualKeyHere
   ```
4. **Save the file** (Ctrl+S)

### 2.3 Verify Configuration

Your `.env` should now look like this:

```bash
# Required - replace with your actual key
OPENAI_API_KEY=sk-proj-your-actual-key-here

# These can stay as-is
VECTOR_STORE_MODE=chroma
API_HOST=0.0.0.0
API_PORT=8000
MAX_CORRECTIONS=2
LOG_LEVEL=INFO
```

**That's it!** You only need the OpenAI key to get started.

---

## Step 3: Testing Locally

Before deploying, let's make sure everything works locally.

### 3.1 Install Dependencies (if not done)

```powershell
# Activate your virtual environment
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3.2 Start the API Server

```powershell
# Start the server
uvicorn src.api.main:app --reload
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### 3.3 Test the Health Endpoint

**Open a new PowerShell window** and run:

```powershell
curl http://localhost:8000/health
```

**Expected response:**
```json
{"status":"healthy","version":"1.0.0"}
```

### 3.4 Test the API Documentation

Open your browser and go to:
```
http://localhost:8000/docs
```

You should see the **Swagger UI** with interactive API documentation.

### 3.5 Test a Query (Optional)

```powershell
# Test query
curl -X POST http://localhost:8000/query `
  -H "Content-Type: application/json" `
  -d '{"query": "What is this system?"}'
```

**If everything works locally, you're ready to deploy! 🚀**

Press `Ctrl+C` in the server window to stop it.

---

## Step 4: Choose Your Deployment Platform

| Feature | Railway | Render |
|---------|---------|--------|
| **Best for** | Quick deploys, prototypes | Production apps |
| **Free Trial** | $5 credit | 750 hrs/month |
| **Starting Cost** | ~$10/month | ~$8/month |
| **CLI Tool** | ⭐ Excellent | Basic |
| **Dashboard** | Good | ⭐ Excellent |
| **Deploy Speed** | ⚡ Very fast | Fast |

**Recommendation:**
- **Railway** if: You like CLI tools, want fastest deploys
- **Render** if: You prefer dashboard UI, want better free tier

---

## Option A: Deploy to Railway

### A.1 Install Railway CLI

```powershell
# Install Railway CLI
npm install -g @railway/cli
```

**Verify installation:**
```powershell
railway --version
```

### A.2 Login to Railway

```powershell
# Login (opens browser)
railway login
```

This opens your browser. Choose:
- **Sign up with GitHub** (recommended), OR
- Sign up with email

### A.3 Initialize Railway Project

```powershell
# Create new Railway project
railway init
```

**Follow the prompts:**
1. "Create a new project" → Enter
2. Project name: `rag-finance-system` (or your choice)
3. Wait for project creation

### A.4 Set Environment Variables

```powershell
# Set your OpenAI API key (replace with your actual key)
railway variables set OPENAI_API_KEY="sk-proj-your-actual-key-here"

# Set other required variables
railway variables set VECTOR_STORE_MODE="chroma"
railway variables set MAX_CORRECTIONS="2"
railway variables set LOG_LEVEL="INFO"
```

**Verify variables:**
```powershell
railway variables
```

### A.5 Deploy!

```powershell
# Deploy your application
railway up
```

**What happens:**
1. Code is uploaded to Railway
2. Docker image is built (takes 2-5 minutes)
3. Service is deployed
4. You get a URL like: `https://your-app.railway.app`

### A.6 Get Your App URL

```powershell
# Open your app in browser
railway open
```

Or get the URL:
```powershell
railway status
```

### A.7 Add Custom Domain (Optional)

```powershell
# Add your domain
railway domain add api.yourdomain.com
```

Then add a CNAME record in your DNS:
```
CNAME api.yourdomain.com -> your-app.railway.app
```

**✅ You're deployed on Railway!**

---

## Option B: Deploy to Render

### B.1 Create Render Account

1. Go to: **https://render.com/**
2. Click **"Get Started"**
3. Sign up with GitHub (recommended) or email

### B.2 Connect Your GitHub Repository

1. Make sure your code is pushed to GitHub:
   ```powershell
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. In Render dashboard, click **"New +"** → **"Blueprint"**

3. Click **"Connect account"** → Authorize GitHub

4. Select your repository: `rag-finance-system`

### B.3 Render Detects Configuration

Render automatically finds your `render.yaml` file.

You'll see:
- ✅ 1 web service: `rag-finance-api`
- ✅ Docker configuration detected
- ✅ Auto-deploy enabled

### B.4 Set Environment Variables

**Before clicking "Apply":**

1. Click on **"rag-finance-api"** service
2. Find **"Environment Variables"** section
3. Click **"Add Environment Variable"**

Add these variables:

| Key | Value |
|-----|-------|
| `OPENAI_API_KEY` | `sk-proj-your-actual-key-here` |
| `VECTOR_STORE_MODE` | `chroma` |
| `MAX_CORRECTIONS` | `2` |
| `LOG_LEVEL` | `INFO` |

4. Click **"Apply"** to deploy

### B.5 Add Persistent Disk (Important for ChromaDB)

1. After service is created, go to service dashboard
2. Left sidebar → **"Disks"**
3. Click **"Add Disk"**
4. Configure:
   - **Name:** `chroma-data`
   - **Mount Path:** `/app/data`
   - **Size:** `1 GB` (can increase later)
5. Click **"Create"**

### B.6 Wait for Deployment

Render will:
1. Clone your repository
2. Build Docker image (3-5 minutes)
3. Deploy to production
4. Run health checks

**Watch the logs:**
- Service dashboard → "Logs" tab
- Look for: `Application startup complete`

### B.7 Get Your App URL

Your service URL is shown at the top:
```
https://your-app-name.onrender.com
```

### B.8 Add Custom Domain (Optional)

1. Service dashboard → **"Settings"**
2. Scroll to **"Custom Domains"**
3. Click **"Add Custom Domain"**
4. Enter: `api.yourdomain.com`
5. Add CNAME record in your DNS:
   ```
   CNAME api.yourdomain.com -> your-app-name.onrender.com
   ```

**✅ You're deployed on Render!**

---

## Step 6: Set Up CI/CD (Optional)

This enables automatic deployments when you push code to GitHub.

### 6.1 Required GitHub Secrets

Go to your GitHub repository:
1. **Settings** → **Secrets and variables** → **Actions**
2. Click **"New repository secret"**

Add these secrets:

#### For All Deployments:
| Secret Name | Value | Where to Get |
|-------------|-------|--------------|
| `OPENAI_API_KEY` | Your OpenAI key | From Step 1 |

#### For Railway:
| Secret Name | Value | Where to Get |
|-------------|-------|--------------|
| `RAILWAY_TOKEN` | Railway API token | Railway → Account Settings → Tokens |
| `RAILWAY_PROJECT_ID` | Your project ID | Railway project URL |
| `RAILWAY_URL` | Full app URL | `https://your-app.railway.app` |

#### For Render:
| Secret Name | Value | Where to Get |
|-------------|-------|--------------|
| `RENDER_API_KEY` | Render API key | Render → Account Settings → API Keys |
| `RENDER_SERVICE_ID` | Service ID | Service settings URL |
| `RENDER_URL` | Full app URL | `https://your-app.onrender.com` |

### 6.2 How to Get Railway Token

```powershell
# Using Railway CLI
railway whoami --token
```

Or via dashboard:
1. Go to: https://railway.app/account/tokens
2. Click **"Create Token"**
3. Name: `GitHub Actions`
4. Copy the token

### 6.3 How to Get Render API Key

1. Go to: https://dashboard.render.com/
2. Click your profile (top right) → **"Account Settings"**
3. Left sidebar → **"API Keys"**
4. Click **"Create API Key"**
5. Copy the key

### 6.4 How to Get Service IDs

**Railway Project ID:**
- Check your Railway dashboard URL
- Format: `https://railway.app/project/abc123def456`
- Copy: `abc123def456`

**Render Service ID:**
- Go to your service dashboard
- Check the URL
- Format: `https://dashboard.render.com/web/srv-abc123def456`
- Copy: `srv-abc123def456`

### 6.5 Test CI/CD

```powershell
# Make a small change
echo "# Test deployment" >> README.md

# Commit and push
git add .
git commit -m "Test CI/CD deployment"
git push origin main
```

**Watch the deployment:**
1. Go to GitHub → Your repository → **Actions** tab
2. You'll see the workflow running
3. Stages: Test → Build → Deploy → Health Check

**Expected result:**
- ✅ All tests pass
- ✅ Docker image builds
- ✅ Deploys to Railway/Render
- ✅ Health check succeeds

---

## Step 7: Verify Deployment

### 7.1 Check Health Endpoint

Replace `YOUR_APP_URL` with your actual URL:

```powershell
# Railway
curl https://your-app.railway.app/health

# Render
curl https://your-app.onrender.com/health
```

**Expected response:**
```json
{"status":"healthy","version":"1.0.0"}
```

### 7.2 Check API Documentation

Open in browser:
```
https://YOUR_APP_URL/docs
```

You should see the Swagger UI interface.

### 7.3 Test a Query

```powershell
# Test query (replace YOUR_APP_URL)
curl -X POST https://YOUR_APP_URL/query `
  -H "Content-Type: application/json" `
  -d '{"query": "What is this system?"}'
```

**Expected:** A JSON response with an answer.

### 7.4 Check Logs

**Railway:**
```powershell
railway logs
```

**Render:**
- Dashboard → Your service → "Logs" tab

Look for:
- ✅ "Application startup complete"
- ✅ No error messages
- ✅ Health checks passing

---

## 🎉 Congratulations! You're Deployed!

### What You've Accomplished:

- ✅ Got your OpenAI API key
- ✅ Configured your environment
- ✅ Tested locally
- ✅ Deployed to production (Railway or Render)
- ✅ (Optional) Set up automated deployments

### Next Steps:

1. **Add sample data** (SEC filings)
   ```powershell
   python scripts/ingest_sec_data.py --year 2024 --quarter 3
   python scripts/process_documents.py
   ```

2. **Monitor costs**
   - OpenAI: https://platform.openai.com/usage
   - Railway: https://railway.app/account/usage
   - Render: Dashboard → Billing

3. **Set up monitoring** (optional)
   - Use Docker Compose locally for Grafana/Jaeger
   - Or integrate with Grafana Cloud

4. **Add custom domain** (optional)
   - Improves branding
   - Free SSL included

5. **Implement rate limiting** (recommended)
   - Prevents API abuse
   - Controls costs

---

## 📊 Cost Breakdown

Based on 1000-5000 queries/month:

| Component | Monthly Cost |
|-----------|-------------|
| **OpenAI API** | $60-180 |
| **Railway/Render** | $10-30 |
| **Total** | **$70-210** |

**Ways to reduce costs:**
1. Use `gpt-4o-mini` for all agents (60% cheaper)
2. Implement caching for repeated queries
3. Start with free tiers and scale up
4. Set cost limits in OpenAI dashboard

---

## 🆘 Common Issues

### Issue: "Invalid API key"

**Solution:**
- Verify key starts with `sk-`
- Check no extra spaces in `.env`
- Ensure payment method added to OpenAI
- Try creating a new key

### Issue: Build fails on Railway/Render

**Solution:**
- Check all files are committed to git
- Verify Dockerfile syntax
- Review build logs for specific errors
- Test Docker build locally first

### Issue: Health check fails

**Solution:**
- Check logs for errors
- Verify `OPENAI_API_KEY` is set in platform
- Ensure port configuration is correct
- Check OpenAI API status

### Issue: High costs

**Solution:**
- Set spending limits in OpenAI dashboard
- Add `MAX_COST_PER_QUERY=0.50` to environment
- Use `gpt-4o-mini` instead of `gpt-4-turbo`
- Implement caching layer

---

## 📚 Documentation Reference

- **API Keys Guide**: `API_KEYS_GUIDE.md`
- **Quick Deploy**: `QUICK_DEPLOY.md`
- **Full Deployment**: `DEPLOYMENT.md`
- **Main README**: `README.md`

---

## 💬 Getting Help

- **Railway**: [Discord](https://discord.gg/railway)
- **Render**: [Community Forum](https://community.render.com/)
- **OpenAI**: [Help Center](https://help.openai.com/)
- **Project Issues**: [GitHub Issues](your-repo-url/issues)

---

**You did it! 🚀 Your RAG Finance System is now running in production!**

**Last Updated**: 2025-12-09

