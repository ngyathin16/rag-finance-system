# Quick Deploy Guide

> Fast track to deploying RAG Finance System in 5 minutes

## Prerequisites

```bash
# Required
✅ OpenAI API key (https://platform.openai.com/api-keys)
✅ GitHub account
✅ Git installed

# Optional
⭐ Pinecone account (for production scale)
⭐ Docker (for local testing)
```

---

## Railway (Fastest)

### 1-Click Deploy

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template)

### CLI Deploy (5 commands)

```bash
# Install & login
npm i -g @railway/cli && railway login

# Initialize project
railway init

# Set environment
railway variables set OPENAI_API_KEY="sk-your-key"
railway variables set VECTOR_STORE_MODE="chroma"

# Deploy
railway up
```

**Cost**: ~$10-30/month

---

## Render

### 1-Click Deploy

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

### Manual Deploy (4 steps)

1. Go to [dashboard.render.com](https://dashboard.render.com/)
2. New → Blueprint → Connect GitHub repo
3. Set `OPENAI_API_KEY` in environment variables
4. Click "Apply"

**Cost**: ~$8-30/month

---

## Environment Variables (Required)

```bash
OPENAI_API_KEY=sk-your-key-here
VECTOR_STORE_MODE=chroma
MAX_CORRECTIONS=2
```

**Optional but Recommended:**
```bash
LOG_LEVEL=INFO
ENABLE_METRICS=true
```

---

## GitHub Actions CI/CD

### Setup Secrets (2 minutes)

Go to: **Settings** → **Secrets and variables** → **Actions**

**For Railway:**
```
OPENAI_API_KEY=sk-...
RAILWAY_TOKEN=...
RAILWAY_PROJECT_ID=...
RAILWAY_URL=https://your-app.railway.app
```

**For Render:**
```
OPENAI_API_KEY=sk-...
RENDER_API_KEY=...
RENDER_SERVICE_ID=...
RENDER_URL=https://your-app.onrender.com
```

### Trigger Deploy

```bash
git add .
git commit -m "Deploy to production"
git push origin main
```

✅ Auto-deploys on every push to `main`!

---

## Post-Deploy Verification

```bash
# Health check
curl https://your-app-url.com/health

# Test query
curl -X POST https://your-app-url.com/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is this system?"}'

# View API docs
open https://your-app-url.com/docs
```

---

## Monitoring URLs

### Local (Docker Compose)
```
http://localhost:8000/docs    - API Docs
http://localhost:16686         - Jaeger Tracing
http://localhost:9090          - Prometheus
http://localhost:3000          - Grafana (admin/admin)
```

### Production
```
https://your-app-url.com/docs  - API Docs
Railway/Render Dashboard       - Logs & Metrics
```

---

## Cost Breakdown

| Component | Railway | Render |
|-----------|---------|--------|
| **Hosting** | $10-30 | $8-30 |
| **OpenAI** | $60-180 | $60-180 |
| **Storage** | Included | +$0.25/GB |
| **Total/mo** | **$70-210** | **$68-210** |

*Based on 1000-5000 queries/month*

---

## Common Issues

### Build Fails
```bash
# Test locally first
docker build -t rag-finance-system .
docker run -p 8000:8000 -e OPENAI_API_KEY=$OPENAI_KEY rag-finance-system
```

### Health Check Fails
- ✅ Check `OPENAI_API_KEY` is set
- ✅ Verify logs in dashboard
- ✅ Ensure port matches platform

### High Costs
```bash
# Add cost limits
MAX_COST_PER_QUERY=0.50
MAX_TOKENS_PER_REQUEST=4000

# Switch to cheaper model
# In code: model_name="gpt-4o-mini"
```

---

## Next Steps

1. ✅ Deploy (you're done!)
2. ⭐ Set up custom domain
3. ⭐ Configure monitoring alerts
4. ⭐ Implement rate limiting
5. ⭐ Add caching layer

---

## Full Documentation

- [DEPLOYMENT.md](DEPLOYMENT.md) - Complete deployment guide
- [README.md](README.md) - Full system documentation
- [DOCKER_SETUP.md](DOCKER_SETUP.md) - Docker setup guide

---

## Support

- Railway: [discord.gg/railway](https://discord.gg/railway)
- Render: [community.render.com](https://community.render.com/)
- Issues: [GitHub Issues](your-repo/issues)

---

**Happy Deploying! 🚀**

