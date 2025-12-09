# API Keys Setup Guide

> Complete guide to getting and configuring all API keys for RAG Finance System

## 📋 Quick Summary

| API Key | Required? | Purpose | Cost |
|---------|-----------|---------|------|
| **OpenAI** | ✅ YES | LLM & embeddings | Pay-as-you-go |
| **Pinecone** | ⭐ Optional | Production vector store | Free tier available |
| **Railway** | ⭐ For deployment | Railway hosting | $5 trial credit |
| **Render** | ⭐ For deployment | Render hosting | Free tier available |

---

## 🔑 Required API Key

### 1. OpenAI API Key (REQUIRED)

**What it's for:** Powers the AI system (GPT-4, embeddings)

**Cost:** Pay-as-you-go pricing
- Embeddings: ~$0.02 per 1M tokens
- GPT-4o-mini: ~$0.15-0.60 per 1M tokens
- GPT-4-turbo: ~$10-30 per 1M tokens
- Estimated: $60-180/month for 1000-5000 queries

#### Step-by-Step Instructions:

1. **Go to OpenAI Platform**
   - Visit: https://platform.openai.com/
   - Click "Sign Up" (or "Log In" if you have an account)

2. **Create an Account**
   - Use your email or Google/Microsoft account
   - Verify your email address
   - Complete phone verification

3. **Add Payment Method**
   - Go to: https://platform.openai.com/account/billing/overview
   - Click "Add payment method"
   - Add credit card (required for API access)
   - ⚠️ Set a spending limit to avoid surprise charges!
     - Go to: https://platform.openai.com/account/billing/limits
     - Set monthly limit (e.g., $50-100)

4. **Create API Key**
   - Go to: https://platform.openai.com/api-keys
   - Click "Create new secret key"
   - Give it a name (e.g., "RAG Finance System")
   - **IMPORTANT:** Copy the key immediately (starts with `sk-`)
   - You won't be able to see it again!

5. **Add to .env file**
   ```bash
   # Open .env file and replace this line:
   OPENAI_API_KEY=sk-your-actual-key-here
   ```

#### ⚠️ Important Notes:
- The key starts with `sk-`
- Keep it secret! Never share or commit to git
- Free trial: $5 credit (expires after 3 months)
- Pay-as-you-go after trial
- Set spending limits to control costs

---

## ⭐ Optional API Keys

### 2. Pinecone API Key (Optional - For Production Scale)

**What it's for:** Production vector database (faster than ChromaDB for large datasets)

**When to use:** 
- ✅ Production deployment with >1M vectors
- ✅ Need faster search performance
- ❌ Local development (use ChromaDB instead)
- ❌ Small-scale projects

**Cost:**
- Free tier: 1 index, 100K vectors
- Serverless: ~$0.096/GB/month
- Estimated: $0-70/month depending on data size

#### Step-by-Step Instructions:

1. **Create Account**
   - Visit: https://www.pinecone.io/
   - Click "Sign Up Free"
   - Use email or GitHub account

2. **Get API Key**
   - After login, you'll see the dashboard
   - Left sidebar → "API Keys"
   - Copy your API key (or create a new one)

3. **Create Index (Required for Pinecone)**
   - Dashboard → "Indexes" → "Create Index"
   - Name: `financial-docs`
   - Dimensions: `1536` (for OpenAI embeddings)
   - Metric: `cosine`
   - Cloud: `aws`
   - Region: `us-east-1` (or closest to you)

4. **Add to .env file**
   ```bash
   # Uncomment and fill these lines in .env:
   VECTOR_STORE_MODE=pinecone
   PINECONE_API_KEY=your-pinecone-key-here
   PINECONE_ENVIRONMENT=us-east-1
   PINECONE_INDEX_NAME=financial-docs
   ```

#### When to Switch to Pinecone:
- You have >100K documents
- ChromaDB searches are slow (>2 seconds)
- You need production-grade performance

---

## 🚀 Deployment Platform API Keys

### 3. Railway API Token (For Railway Deployment)

**What it's for:** Deploy your app to Railway via CLI or CI/CD

**Cost:** 
- Trial: $5 credit
- Starter: $5/month + usage
- Estimated: $10-30/month

#### Step-by-Step Instructions:

1. **Create Railway Account**
   - Visit: https://railway.app/
   - Click "Login" → Sign up with GitHub (recommended)

2. **Get API Token**
   - Go to: https://railway.app/account/tokens
   - Click "Create Token"
   - Name it (e.g., "RAG Finance CI/CD")
   - Copy the token

3. **Add to GitHub Secrets** (for CI/CD)
   - Go to your GitHub repository
   - Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `RAILWAY_TOKEN`
   - Value: Paste your token

4. **Get Project ID** (if using CI/CD)
   - Create a Railway project first:
     ```bash
     railway login
     railway init
     ```
   - Project ID is shown in Railway dashboard URL:
     `https://railway.app/project/YOUR_PROJECT_ID`
   - Add to GitHub Secrets as `RAILWAY_PROJECT_ID`

---

### 4. Render API Key (For Render Deployment)

**What it's for:** Deploy your app to Render via API or CI/CD

**Cost:**
- Free tier: 750 hours/month (limited resources)
- Starter: $7/month
- Standard: $25/month (recommended for production)

#### Step-by-Step Instructions:

1. **Create Render Account**
   - Visit: https://render.com/
   - Click "Get Started" → Sign up with GitHub (recommended)

2. **Get API Key**
   - Go to: https://dashboard.render.com/
   - Click your profile (top right) → "Account Settings"
   - Left sidebar → "API Keys"
   - Click "Create API Key"
   - Copy the key

3. **Add to GitHub Secrets** (for CI/CD)
   - Go to your GitHub repository
   - Settings → Secrets and variables → Actions
   - Add these secrets:
     - `RENDER_API_KEY`: Your API key
     - `RENDER_SERVICE_ID`: (get after creating service)
     - `RENDER_URL`: Your service URL

---

## 🔐 Where to Put API Keys

### Local Development (.env file)

```bash
# File: .env (in project root)
OPENAI_API_KEY=sk-your-key-here
VECTOR_STORE_MODE=chroma
```

**Check it works:**
```bash
# Test that .env is loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('✅ API key loaded!' if os.getenv('OPENAI_API_KEY') else '❌ No API key')"
```

---

### Railway Deployment

**Option 1: Railway CLI**
```bash
railway variables set OPENAI_API_KEY="sk-your-key-here"
railway variables set VECTOR_STORE_MODE="chroma"
```

**Option 2: Railway Dashboard**
1. Go to your Railway project
2. Click on your service
3. "Variables" tab
4. Add variables:
   - `OPENAI_API_KEY`
   - `VECTOR_STORE_MODE`
   - `MAX_CORRECTIONS`

---

### Render Deployment

**Render Dashboard:**
1. Go to your web service
2. "Environment" tab (left sidebar)
3. Click "Add Environment Variable"
4. Add each variable:
   - Key: `OPENAI_API_KEY`
   - Value: `sk-your-key-here`
5. Save changes (triggers redeploy)

---

### GitHub Actions (CI/CD)

**Repository Secrets:**
1. GitHub repo → Settings → Secrets and variables → Actions
2. Click "New repository secret"
3. Add each secret:

| Secret Name | Description | Example |
|-------------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI key | `sk-proj-...` |
| `RAILWAY_TOKEN` | Railway API token | `eyJ...` |
| `RAILWAY_PROJECT_ID` | Railway project ID | `a1b2c3d4...` |
| `RAILWAY_URL` | Your Railway URL | `https://your-app.railway.app` |
| `RENDER_API_KEY` | Render API key | `rnd_...` |
| `RENDER_SERVICE_ID` | Render service ID | `srv-...` |
| `RENDER_URL` | Your Render URL | `https://your-app.onrender.com` |

---

## 🔒 Security Best Practices

### ✅ DO:
- ✅ Keep API keys in `.env` file (gitignored)
- ✅ Use environment variables in production
- ✅ Set spending limits on OpenAI
- ✅ Rotate keys periodically
- ✅ Use separate keys for dev/prod

### ❌ DON'T:
- ❌ Commit `.env` to git
- ❌ Share keys in chat/email
- ❌ Hardcode keys in code
- ❌ Use production keys for testing
- ❌ Give keys overly broad permissions

---

## 🧪 Testing Your API Keys

### Test OpenAI Key:

```bash
# Option 1: Using curl
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer YOUR_API_KEY"

# Option 2: Using Python
python -c "
from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
print('✅ OpenAI API key is valid!')
print(f'Available models: {len(client.models.list().data)} models')
"
```

### Test Pinecone Key:

```bash
python -c "
from pinecone import Pinecone
import os
from dotenv import load_dotenv
load_dotenv()

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
print('✅ Pinecone API key is valid!')
print(f'Indexes: {pc.list_indexes()}')
"
```

### Test Full System:

```bash
# Start the API server
uvicorn src.api.main:app --reload

# In another terminal, test the health endpoint
curl http://localhost:8000/health

# Test a query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this system?"}'
```

---

## 💰 Cost Tracking

### Monitor OpenAI Usage:
- Dashboard: https://platform.openai.com/usage
- View daily/monthly costs
- Set up email alerts for spending thresholds

### Monitor Pinecone Usage:
- Dashboard: https://app.pinecone.io/
- View vector operations and storage
- Serverless billing is automatic

### Monitor Hosting Costs:
- **Railway**: https://railway.app/account/usage
- **Render**: Dashboard → Billing

---

## 🆘 Troubleshooting

### "Invalid API key" Error

**OpenAI:**
```
Error: Incorrect API key provided
```
**Solution:**
- Check key starts with `sk-`
- Verify no extra spaces in `.env`
- Check you copied the entire key
- Ensure payment method is added
- Try creating a new key

**Pinecone:**
```
Error: Invalid API key
```
**Solution:**
- Check key is from correct project
- Verify environment matches (`us-east-1`, etc.)
- Ensure index exists

### "Rate limit exceeded" Error

**Solution:**
- You've hit OpenAI's rate limit
- Wait a few seconds and retry
- Upgrade to higher tier at OpenAI
- Implement rate limiting in your app

### "Insufficient quota" Error

**Solution:**
- Add payment method to OpenAI account
- Free trial credit exhausted
- Add funds or set up automatic billing

---

## 📋 Quick Checklist

Before deploying, ensure:

- [ ] OpenAI API key obtained and tested
- [ ] `.env` file created with OPENAI_API_KEY
- [ ] Payment method added to OpenAI (if needed)
- [ ] Spending limits set on OpenAI account
- [ ] (Optional) Pinecone account created
- [ ] (Optional) Railway/Render account created
- [ ] GitHub secrets configured (if using CI/CD)
- [ ] API keys never committed to git
- [ ] Tested locally with `curl http://localhost:8000/health`

---

## 📚 Additional Resources

- **OpenAI Docs**: https://platform.openai.com/docs
- **Pinecone Docs**: https://docs.pinecone.io/
- **Railway Docs**: https://docs.railway.app/
- **Render Docs**: https://render.com/docs
- **Environment Variables Guide**: https://12factor.net/config

---

**Last Updated**: 2025-12-09

