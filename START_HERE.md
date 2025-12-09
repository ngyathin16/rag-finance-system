# 🚀 START HERE - Quick Setup Guide

> Your RAG Finance System is ready to deploy! Follow these steps.

---

## ✅ What's Already Done

- ✅ `.env` file created with configuration template
- ✅ All deployment files ready (Railway, Render, CI/CD)
- ✅ Docker configuration set up
- ✅ Comprehensive documentation created

---

## 🔑 STEP 1: Get Your OpenAI API Key (5 minutes)

**This is the ONLY required API key!**

### Quick Steps:

1. **Go to:** https://platform.openai.com/api-keys
2. **Sign up/Login** (use GitHub or email)
3. **Add payment method** (required for API access)
   - Go to: https://platform.openai.com/account/billing/overview
   - **Set spending limit** to $50/month (recommended)
4. **Create API key:**
   - Click "Create new secret key"
   - Name it: "RAG Finance System"
   - Copy the key (starts with `sk-`)

**💰 Cost:** $5 free credit, then pay-as-you-go (~$60-180/month for 1000-5000 queries)

---

## 📝 STEP 2: Add API Key to .env File (1 minute)

### Your .env file is located here:
```
C:\Users\ngyat\OneDrive\Documents\projects\rag-finance-system\.env
```

### What to do:

1. **Open `.env`** in your editor (VS Code, Notepad++, etc.)

2. **Find this line:**
   ```bash
   OPENAI_API_KEY=sk-your-openai-api-key-here
   ```

3. **Replace with your actual key:**
   ```bash
   OPENAI_API_KEY=sk-proj-AbCdEf123YourActualKeyHere
   ```

4. **Save the file** (Ctrl+S)

**That's it!** Leave everything else as default.

---

## 🧪 STEP 3: Test Locally (5 minutes)

### Start the server:

```powershell
# Activate virtual environment
.\venv\Scripts\activate

# Start the API
uvicorn src.api.main:app --reload
```

### Test it works:

**Open a new PowerShell window:**
```powershell
# Health check
curl http://localhost:8000/health

# Expected: {"status":"healthy","version":"1.0.0"}
```

**Or open in browser:**
```
http://localhost:8000/docs
```

**✅ If you see the Swagger UI, you're ready to deploy!**

Press `Ctrl+C` to stop the server.

---

## 🚀 STEP 4: Deploy (Choose One)

### Option A: Railway (Recommended for Quick Deploy)

**5 commands to deploy:**

```powershell
# 1. Install CLI
npm install -g @railway/cli

# 2. Login (opens browser)
railway login

# 3. Create project
railway init

# 4. Set API key (use YOUR actual key)
railway variables set OPENAI_API_KEY="sk-your-actual-key-here"
railway variables set VECTOR_STORE_MODE="chroma"

# 5. Deploy!
railway up
```

**Your app URL:**
```powershell
railway open
```

**💰 Cost:** ~$10-30/month (after $5 trial credit)

---

### Option B: Render (Recommended for Production)

**Steps:**

1. **Push code to GitHub:**
   ```powershell
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Go to:** https://dashboard.render.com/
   - Sign up with GitHub
   - Click "New +" → "Blueprint"
   - Connect your repository

3. **Set environment variable:**
   - Find "Environment Variables" section
   - Add: `OPENAI_API_KEY` = `sk-your-actual-key-here`
   - Add: `VECTOR_STORE_MODE` = `chroma`

4. **Click "Apply"** and wait 3-5 minutes

5. **Your app URL is shown at the top!**

**💰 Cost:** ~$8-30/month (free tier available)

---

## ✅ STEP 5: Verify Deployment (2 minutes)

Replace `YOUR_APP_URL` with your actual URL:

```powershell
# Test health
curl https://YOUR_APP_URL/health

# View API docs in browser
https://YOUR_APP_URL/docs
```

**Expected:** Health check returns `{"status":"healthy"}`

---

## 📚 Documentation You Have

| File | When to Use |
|------|-------------|
| **START_HERE.md** | 👈 You are here! Quick start |
| **API_KEYS_GUIDE.md** | Detailed guide for getting API keys |
| **STEP_BY_STEP_DEPLOYMENT.md** | Complete walkthrough with screenshots |
| **QUICK_DEPLOY.md** | Fast reference for deployment |
| **DEPLOYMENT.md** | Comprehensive deployment guide |
| **README.md** | Full project documentation |

---

## 🔑 API Keys Summary

### Required (You MUST have):
- ✅ **OpenAI API Key** - For LLM and embeddings
  - Get: https://platform.openai.com/api-keys
  - Cost: $60-180/month for typical usage
  - Where: Add to `.env` file

### Optional (Nice to have):
- ⭐ **Pinecone** - For production vector store (>1M documents)
  - Get: https://www.pinecone.io/
  - Cost: Free tier or ~$70/month
  - Not needed for most use cases (ChromaDB is included)

### For Deployment (if using CI/CD):
- ⭐ **Railway Token** - For automated Railway deployments
  - Get: railway.app/account/tokens
  - Add to GitHub Secrets
  
- ⭐ **Render API Key** - For automated Render deployments
  - Get: dashboard.render.com → Account Settings → API Keys
  - Add to GitHub Secrets

---

## 💰 Total Cost Estimate

| Component | Monthly Cost | Required? |
|-----------|-------------|-----------|
| OpenAI API | $60-180 | ✅ Yes |
| Railway/Render Hosting | $10-30 | ✅ Yes |
| Pinecone (optional) | $0-70 | ❌ No |
| **Total** | **$70-210** | |

**Free options:**
- Railway: $5 trial credit
- Render: 750 hours/month free tier
- Pinecone: Free tier (100K vectors)

---

## 🆘 Common Questions

### Q: Where is my .env file?
**A:** `C:\Users\ngyat\OneDrive\Documents\projects\rag-finance-system\.env`

### Q: Do I need Pinecone?
**A:** No! ChromaDB (included) works great for most use cases. Only use Pinecone for >1M documents.

### Q: How much will this cost?
**A:** $70-210/month depending on usage. Start with free tiers to test.

### Q: Can I test without deploying?
**A:** Yes! Run `uvicorn src.api.main:app --reload` and test at `http://localhost:8000`

### Q: Where do I put API keys for deployment?
**A:** 
- **Railway**: `railway variables set OPENAI_API_KEY="sk-..."`
- **Render**: Add in dashboard under Environment Variables
- **NOT in .env** (that's only for local testing)

### Q: Is my API key safe?
**A:** Yes! The `.env` file is gitignored and never committed to GitHub.

---

## 🎯 Quick Checklist

Before deploying:

- [ ] OpenAI account created
- [ ] Payment method added to OpenAI
- [ ] API key obtained (starts with `sk-`)
- [ ] API key added to `.env` file
- [ ] Tested locally (`curl http://localhost:8000/health`)
- [ ] Choose platform (Railway or Render)
- [ ] Deploy following steps above
- [ ] Verify deployment works

---

## 📖 Next Steps After Deployment

1. **Add financial data:**
   ```powershell
   python scripts/ingest_sec_data.py --year 2024 --quarter 3
   python scripts/process_documents.py
   ```

2. **Set up monitoring:**
   - Check OpenAI usage: https://platform.openai.com/usage
   - Monitor hosting: Railway/Render dashboard

3. **Add custom domain** (optional):
   - Railway: `railway domain add api.yourdomain.com`
   - Render: Dashboard → Settings → Custom Domains

4. **Set up CI/CD** (optional):
   - See: `STEP_BY_STEP_DEPLOYMENT.md` → Step 6

---

## 🆘 Need Help?

- **Detailed deployment guide:** Read `STEP_BY_STEP_DEPLOYMENT.md`
- **API key help:** Read `API_KEYS_GUIDE.md`
- **Railway issues:** [Discord](https://discord.gg/railway)
- **Render issues:** [Forum](https://community.render.com/)
- **OpenAI issues:** [Help Center](https://help.openai.com/)

---

## 🎉 You're Ready!

**Everything is set up! Just follow these 5 steps:**

1. Get OpenAI API key → **5 minutes**
2. Add to `.env` file → **1 minute**
3. Test locally → **5 minutes**
4. Deploy (Railway or Render) → **10 minutes**
5. Verify it works → **2 minutes**

**Total time: ~20-30 minutes from start to production! 🚀**

---

**Last Updated**: 2025-12-09

