# Quick Start Guide - Glucose Monitor Demo

## ✅ Fixes Applied

I've fixed two critical errors:

1. **Frontend JSX Error**: Fixed malformed `<key={i}>` tag in `PredictionForm.tsx`
2. **Backend Python Error**: Moved `GlucoseCNN` class inside try/except block to handle missing PyTorch

## 🚀 How to Run the System

### Backend (Terminal 1)

```bash
cd /Users/muhammadali/Documents/Antigravity/kids-quiz/glucose-monitor-demo/backend

# Install dependencies
pip install -r requirements.txt

# Train models (generates synthetic data)
python model_training.py

# Start API server
python -m uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
```

The API will run at `http://localhost:8000`

### Frontend (Terminal 2)

The frontend is already running at `http://localhost:3002`

If you need to restart it:

```bash
cd /Users/muhammadali/Documents/Antigravity/kids-quiz/glucose-monitor-demo/frontend

# Create environment file
echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000" > .env.local

# Start dev server
npm run dev
```

## 🧪 Testing the System

1. **Start Backend**: Run the backend commands above
2. **Open Frontend**: Navigate to `http://localhost:3002`
3. **See Disclaimer**: Red warning banner should be visible
4. **Try Demo Mode**: Select a sample ID (0-99) and click "Get AI Estimate"
5. **Try Manual Mode**: Enter heart rate, HRV, and time since meal

## 📝 Expected Results

- **Glucose Estimate**: A number between 70-200 mg/dL
- **Range**: Color-coded badge (Low/Normal/High)
- **Confidence**: Progress bar showing model confidence
- **Disclaimer**: Warning text with every result

## ⚠️ Important Notes

- Backend must be running for frontend to work
- First run: `python model_training.py` takes 2-3 minutes
- Models are saved to `backend/models/` directory
- This is a **research demo only** - not for medical use

## 🐛 Troubleshooting

**Frontend won't compile?**
- Make sure you're in the `frontend` directory
- Run `npm install` again
- Check that `PredictionForm.tsx` has the fixed code

**Backend errors?**
- PyTorch is optional - Random Forest will still work
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check that you're in the `backend` directory

**API connection errors?**
- Verify backend is running on port 8000
- Check `.env.local` has correct API URL
- Try restarting both servers

## 📦 Ready for GitHub

The system is ready to upload to GitHub. See `GITHUB_SETUP.md` for instructions.
