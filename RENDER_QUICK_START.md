# 🚀 Render Deployment Checklist - Quick Start

## ✅ Pre-Deployment (Run Locally First)

- [ ] Test app locally: `uvicorn app:app --reload`
- [ ] Verify health endpoint: `curl http://localhost:8000/health`
- [ ] Test a question endpoint: Works without errors?
- [ ] Verify `university_guide.md` exists in root directory
- [ ] Verify `GOOGLE_API_KEY` is set in `.env`

## ✅ Git Preparation

```bash
# 1. Add new deployment files
git add render.yaml .gitignore RENDER_DEPLOYMENT.md RENDER_QUICK_START.md

# 2. Commit
git commit -m "Add Render deployment configuration for free tier"

# 3. Push to GitHub (make sure main branch is default)
git push origin main
```

## ✅ Render Dashboard Setup (5 minutes)

### Step 1: Create Web Service
- [ ] Go to https://render.com (login/signup)
- [ ] Click **"New +"** in top right
- [ ] Select **"Web Service"**
- [ ] Click **"Connect GitHub Account"** (if first time)
- [ ] Find and select your repository

### Step 2: Configure Service
Fill in these fields:
- [ ] **Name**: `university-chatbot` (lowercase, no spaces)
- [ ] **Runtime**: `Python 3` (auto-detected)
- [ ] **Build Command**: Clear if auto-detected, or use:
  ```
  pip install -r requirements.txt && python preload_model.py
  ```
- [ ] **Start Command**: Clear if auto-detected, or use:
  ```
  uvicorn app:app --host 0.0.0.0 --port 8000
  ```
- [ ] **Instance Type**: `Free`

### Step 3: Add Environment Variables
- [ ] Click **"Advanced"** (bottom of form)
- [ ] Scroll to **"Environment"**
- [ ] Click **"Add Environment Variable"**

Add **two** variables:

| Key | Value | Notes |
|-----|-------|-------|
| `GOOGLE_API_KEY` | `your-api-key-here` | Get from [Google AI Studio](https://aistudio.google.com/apikey) |
| `HF_HUB_OFFLINE` | `1` | For offline mode |

### Step 4: Deploy
- [ ] Click **"Create Web Service"** (blue button at bottom)
- [ ] Wait 5-10 minutes for first build

---

## ✅ Verify Deployment

**After deployment completes:**

1. [ ] Check status shows **"Live"** (green)
2. [ ] Copy the URL (format: `https://university-chatbot-xxxxx.onrender.com`)
3. [ ] Test in browser:
   ```
   https://your-url.onrender.com/health
   ```
   - Should show: `{"status":"ok","rag_initialized":true,"service":"university-chatbot"}`

4. [ ] Test main app:
   ```
   https://your-url.onrender.com/
   ```
   - Should load chat interface

5. [ ] Check logs for any warnings:
   - Dashboard → **Logs** tab
   - Look for "✅ Application ready!"

---

## ✅ If Build Fails

**Check these in order:**

1. **No matching files error?**
   - Make sure you're pushing from the main branch
   - Verify `render.yaml` is in repo root

2. **GOOGLE_API_KEY not found?**
   - Go back to Render dashboard
   - Click service name
   - Environment tab
   - Verify `GOOGLE_API_KEY` is there and not empty

3. **Build timeout (>15 min)?**
   - Model download is slow
   - This is normal first time
   - Just wait or check logs

4. **Dependencies not installing?**
   - Check `requirements.txt` is in repo root
   - Verify no syntax errors

---

## 🎯 Important Notes for Free Tier

| Behavior | Why | What to Do |
|----------|-----|-----------|
| **First request slow** (10-30 sec) | ChromaDB rebuilding | Normal, just wait |
| **Service spins down** after 15 min idle | Free tier limitation | Next request auto-wakes it |
| **Memory warnings** in logs | 512MB is tight | If crashes, upgrade to Paid |
| **Logs say "Cannot find university_guide.md"** | File not in repo | Add to `.gitignore` only `.cache/` not the guide |

---

## 🔗 Useful Links

- Service URL: `https://your-service-name.onrender.com`
- Render Dashboard: https://dashboard.render.com
- This Repo: Check GitHub Actions for deploy history
- Logs: Click service → **Logs** tab
- Environment Vars: Click service → **Environment** tab

---

## 📞 If You Need Help

1. **Check Render logs first**: Most errors are there
2. **Verify environment variables**: Common cause of failures
3. **Test locally first**: Run `uvicorn app:app` locally and verify /health endpoint
4. **Check GitHub integration**: Make sure you authorized Render to access GitHub

---

## ⏰ Expected Timeline

| Step | Time |
|------|------|
| Push to GitHub | 1 min |
| Render detects push | 1-2 min |
| Build starts | Instant |
| Download deps | 2-3 min |
| Cache model | 2-5 min |
| Build complete | 5-10 min **total** |
| First request | Instant (if all OK) |

**Total time to live deployment: ~10-15 minutes**

---

## ✨ Success Indicators

You've successfully deployed when:
- ✅ Dashboard shows service status as **"Live"** (green)
- ✅ `/health` endpoint returns `200 OK` with JSON
- ✅ Visit main URL loads the chat interface
- ✅ Can send a question and get a response
- ✅ Logs show "✅ Application ready!"

🎉 **You're live on Render!**
