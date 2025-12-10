# 🚨 Render Memory Issue - FIXED

## Problem
Render Starter plan has only **512MB RAM**, which isn't enough to process 35,754 documents at once.

---

## ✅ SOLUTION: Use Memory-Efficient Script

### Step 1: Use the New Batch Script

Instead of running `load_documents.py`, use the memory-optimized version:

```bash
# In Render Shell
python scripts/load_documents_batch.py
```

**What this does:**
- ✅ Processes documents in batches of 500 (instead of all at once)
- ✅ Forces garbage collection between batches
- ✅ Works within 512MB memory limit
- ✅ Takes 15-20 minutes (same time, but doesn't crash!)

---

## 📋 Complete Steps for Render

### Option A: Load Pre-Processed Documents (Recommended)

1. **Process locally** (on your machine with more RAM):
   ```powershell
   python scripts/ingest_sec_data.py --year 2024 --quarter 3
   python scripts/process_documents.py
   ```

2. **Upload to Render** via Shell:
   ```bash
   # This uploads your local processed files
   # (Not ideal, but works if you have processed locally)
   ```

3. **Load into ChromaDB on Render**:
   ```bash
   python scripts/load_documents_batch.py
   ```

### Option B: Process Everything on Render (If you have Standard plan)

**Upgrade to Standard plan ($25/month) for 2GB RAM:**

Then you can run normally:
```bash
python scripts/ingest_sec_data.py --year 2024 --quarter 3
python scripts/process_documents.py
python scripts/load_documents_batch.py
```

### Option C: Use Smaller Dataset

Download less data to fit in memory:

```bash
# Use only one quarter instead of all data
python scripts/ingest_sec_data.py --year 2024 --quarter 3
# Then process with smaller chunk size
python scripts/process_documents.py --chunk-size 500
python scripts/load_documents_batch.py
```

---

## 🎯 Recommended Approach

**For production on Render Starter:**

1. Process documents **locally** (your machine has more RAM)
2. Commit `data/processed/documents.pkl` to git (if < 100MB)
3. Push to GitHub
4. Render auto-deploys
5. Run `python scripts/load_documents_batch.py` in Render Shell

**For production with more budget:**

- Upgrade to **Render Standard** ($25/month, 2GB RAM)
- Or use **Railway** (better memory limits)
- Or use **AWS Lambda** with more memory

---

## 💡 Model Updates (Also Fixed!)

Updated models for better performance and cost:

| Component | Old Model | New Model | Cost Savings |
|-----------|-----------|-----------|--------------|
| Generator | gpt-4-turbo-preview | **gpt-4o** | **70% cheaper** |
| Relevance | gpt-4o-mini | gpt-4o-mini | ✅ Already optimal |
| Fact-Check | gpt-4o-mini | gpt-4o-mini | ✅ Already optimal |
| Embeddings | text-embedding-3-small | text-embedding-3-small | ✅ Already optimal |

**Cost comparison:**
- Old: $10/$30 per 1M tokens (input/output)
- New: $2.50/$10 per 1M tokens (input/output)
- **Savings: ~70% on generation costs!**

---

## 🚀 Quick Fix Right Now

**Run this in Render Shell:**

```bash
python scripts/load_documents_batch.py
```

This will work even with 512MB RAM!

---

Last updated: 2025-12-10

