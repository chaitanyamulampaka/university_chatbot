# Render Free Tier Deployment Guide

## Prerequisites
- GitHub account with repository pushed
- Render account (free tier)
- Google Generative AI API key

---

## Step 1: Prepare Your Repository

```bash
# Add changes
git add render.yaml .gitignore
git commit -m "Add Render deployment configuration"
git push origin main
```

---

## Step 2: Deploy on Render

### 2.1 Create Web Service
1. Go to https://render.com
2. Click **"New +"** → **"Web Service"**
3. Connect your GitHub repository
4. Select the repository

### 2.2 Configure Service
- **Name**: `university-chatbot`
- **Environment**: `Python 3`
- **Build Command**: (Leave default or auto-detect from render.yaml)
- **Start Command**: (Leave default or auto-detect from render.yaml)
- **Instance Type**: `Free` (512 MB RAM, ephemeral disk)

### 2.3 Add Environment Variables
Click **"Advanced"** → **"Add Environment Variable"**

Add these variables:
| Key | Value |
|-----|-------|
| `GOOGLE_API_KEY` | Your Google API Key |
| `HF_HUB_OFFLINE` | `1` |

### 2.4 Deploy
Click **"Create Web Service"**

Render will:
1. Pull your repository
2. Install dependencies (`pip install -r requirements.txt`)
3. Pre-download & cache the sentence-transformer model (`python preload_model.py`)
4. Start the FastAPI server

---

## Step 3: Verify Deployment

Once deployed (takes 5-10 minutes):

1. **Get Your URL**: Appears as `https://your-service-name.onrender.com`

2. **Test endpoints**:
   ```bash
   # Health check
   curl https://your-service-name.onrender.com/health
   
   # Get root (serves HTML dashboard)
   curl https://your-service-name.onrender.com/
   
   # Ask a question
   curl -X POST https://your-service-name.onrender.com/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "What courses are offered?"}'
   ```

3. **View logs**: 
   - On Render dashboard: **Logs** tab
   - Real-time monitoring of deployment

---

## Understanding Free Tier Behavior

### ✅ What Works:
- **ChromaDB auto-rebuilds** from `university_guide.md` on startup
- **Model caching** happens during build (saved in container)
- **Local embeddings** run fast with no API calls
- **Streaming responses** work correctly

### ⚠️ Free Tier Limitations:
- **Ephemeral Storage**: Render deletes disk after 15 minutes of inactivity
  - ChromaDB directories (`admissions_chroma_db/`) are rebuilt on next request
  - This adds 5-10 seconds to the first request after sleep
- **Memory**: 512 MB (tight, but manageable)
- **Spin-down**: Service spins down after 15 min inactivity
- **Build Time**: 5-10 minutes first deployment

### Cold Start Timeline:
1. Service spins up (instant)
2. Dependencies loaded (2-3 sec)
3. ChromaDB rebuilt from knowledge base (3-5 sec)
4. First API call responds (~10 sec total)

---

## Troubleshooting

### " Build failed"
- Check logs for missing dependencies
- Ensure `requirements.txt` has all packages
- Verify Python version is 3.10

### "GOOGLE_API_KEY not found"
- Ensure you added `GOOGLE_API_KEY` to Environment Variables in Render dashboard
- Verify the key is valid (test locally first)

### "ChromaDB errors"
- Normal on cold starts (rebuilds automatically)
- Check that `university_guide.md` exists in repository

### Out of Memory (OOM) Kill
- Free tier has 512 MB
- If issue persists, consider:
  - Upgrade to Paid tier ($12/month)
  - Reduce knowledge base size
  - Use external vector DB (Pinecone free tier)

---

## Optimization Tips

### To Reduce Cold Start Time:
1. Compress `university_guide.md` if possible
2. Move non-essential endpoints to separate service
3. Consider upgrading to paid tier for persistent disk

### To Save Bandwidth:
1. Most requests will hit cache (model already loaded)
2. Set up Redis caching (paid tier only)

### To Monitor Costs:
- Free tier: $0/month
- If upgraded: $12-15/month typical
- Track usage in Render dashboard

---

## Next Steps

After deployment:
1. Test all endpoints thoroughly
2. Monitor logs for errors
3. Set up auto-deployment (already enabled in render.yaml)
4. Consider upgrade if performance is unsatisfactory

---

## Support
- Render Docs: https://render.com/docs
- GitHub Integration: https://render.com/docs/deploy-from-github
