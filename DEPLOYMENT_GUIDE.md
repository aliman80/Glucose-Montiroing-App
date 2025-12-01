# Deployment Guide - Enhanced Glucose Monitoring System

## 🚀 Quick Deployment Steps

### Step 1: Test Enhanced Backend Locally

```bash
cd /Users/muhammadali/Documents/Antigravity/kids-quiz/glucose-monitor-demo/backend

# Start enhanced API server
python -m uvicorn api_server_v2:app --reload --host 0.0.0.0 --port 8001
```

**Test at**: http://localhost:8001/docs

### Step 2: Deploy Backend to Railway

```bash
cd /Users/muhammadali/Documents/Antigravity/kids-quiz/glucose-monitor-demo/backend

# Install Railway CLI (if not already installed)
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Deploy
railway up
```

**Railway will provide a URL like**: `https://glucose-monitor-demo-production.up.railway.app`

### Step 3: Update Frontend Environment Variable

```bash
cd /Users/muhammadali/Documents/Antigravity/kids-quiz/glucose-monitor-demo/frontend

# Add environment variable to Vercel
vercel env add NEXT_PUBLIC_API_BASE_URL production

# When prompted, enter your Railway backend URL
# Example: https://glucose-monitor-demo-production.up.railway.app

# Redeploy frontend
vercel --prod
```

### Step 4: Push to GitHub

```bash
cd /Users/muhammadali/Documents/Antigravity/kids-quiz/glucose-monitor-demo

# Add all new enhanced files
git add backend/data_loading_v2.py
git add backend/database.py
git add backend/model_training_v2.py
git add backend/api_server_v2.py
git add backend/README_V2.md
git add backend/models_v2/

# Commit
git commit -m "Add enhanced backend v2: 25 features, 82.5% accuracy, patient database

- Enhanced dataset generator with 25 comprehensive features
- SQLite patient database with history tracking
- Improved ML model (MAE: 11.84 mg/dL, 82.5% accuracy)
- Comprehensive API with patient management
- Feature importance analysis
- Complete documentation"

# Push to GitHub (if remote already exists)
git push origin main

# OR if you need to add remote first:
# git remote add origin https://github.com/YOUR_USERNAME/glucose-monitor-demo.git
# git branch -M main
# git push -u origin main
```

---

## 📦 What Gets Deployed

### Backend (Railway)
- Enhanced API server (api_server_v2.py)
- Trained models (models_v2/)
- SQLite database (glucose_monitor.db)
- All dependencies (requirements.txt)

### Frontend (Vercel)
- Original simple frontend (already deployed)
- Connected to enhanced backend via env variable

---

## 🔗 Your Deployed URLs

**Frontend**: https://frontend-6zgv6ad97-alis-projects-e4ae3535.vercel.app

**Backend**: (Will be provided by Railway after deployment)

**GitHub**: https://github.com/YOUR_USERNAME/glucose-monitor-demo

---

## ✅ Verification Steps

After deployment:

1. **Test Backend**:
   - Open Railway backend URL + `/docs`
   - Try creating a patient
   - Make a prediction
   - View patient history

2. **Test Frontend**:
   - Open Vercel frontend URL
   - Try making a prediction
   - Verify it connects to enhanced backend

3. **Check GitHub**:
   - Verify all files are pushed
   - Check README displays correctly
   - Confirm medical disclaimers are visible

---

## 🎯 Post-Deployment

### Share Your App

**For Friends**:
- Frontend URL: https://frontend-6zgv6ad97-alis-projects-e4ae3535.vercel.app
- GitHub Repo: https://github.com/YOUR_USERNAME/glucose-monitor-demo

**For Portfolio**:
- Highlight: 82.5% accuracy with 25 features
- Emphasize: Full-stack ML system
- Note: Research/educational demo only

### Monitor Usage

**Railway Dashboard**:
- View API requests
- Monitor database size
- Check error logs

**Vercel Dashboard**:
- View frontend traffic
- Monitor build status
- Check deployment logs

---

## 🛠️ Troubleshooting

**Backend won't start on Railway?**
- Check requirements.txt includes all dependencies
- Verify models_v2/ directory is included
- Check Railway logs for errors

**Frontend can't connect to backend?**
- Verify NEXT_PUBLIC_API_BASE_URL is set correctly
- Check CORS settings in api_server_v2.py
- Test backend URL directly in browser

**Database errors?**
- SQLite database is file-based (no external DB needed)
- Database will be created automatically on first run
- Check Railway has write permissions

---

## 📝 Environment Variables

### Backend (Railway)
No environment variables needed! SQLite is file-based.

### Frontend (Vercel)
- `NEXT_PUBLIC_API_BASE_URL` = Your Railway backend URL

---

## 🔄 Future Updates

To update after deployment:

**Backend**:
```bash
cd backend
# Make changes
railway up
```

**Frontend**:
```bash
cd frontend
# Make changes
vercel --prod
```

**GitHub**:
```bash
git add .
git commit -m "Update description"
git push
```

---

## ⚠️ Important Reminders

- This is a **research demo** - NOT for medical use
- Uses **synthetic data** only
- **NOT clinically validated**
- Always include medical disclaimers
- Monitor for appropriate use

---

**Ready to deploy!** Follow the steps above to get your enhanced system live! 🚀
