# Non-Invasive Glucose Monitoring System
## Research Demonstration Only

⚠️ **CRITICAL MEDICAL DISCLAIMER** ⚠️

**This is an EXPERIMENTAL RESEARCH DEMONSTRATION ONLY.**

- This is **NOT** a medical device
- This is **NOT** clinically validated
- This is **NOT** FDA approved  
- This uses **SYNTHETIC** data for educational purposes only
- **DO NOT** use this to make health decisions
- **DO NOT** use this for diagnosis or treatment
- **ALWAYS** consult a qualified healthcare professional

---

## Overview

This is a complete end-to-end research demo system for non-invasive glucose estimation, built for educational purposes. It demonstrates how machine learning could theoretically be applied to estimate glucose levels from non-invasive signals, but uses entirely synthetic data.

### System Architecture

```
┌─────────────────┐
│  Next.js Web    │  ← User Interface (Vercel)
│  Frontend       │
└────────┬────────┘
         │ HTTP/JSON
         ▼
┌─────────────────┐
│  FastAPI        │  ← REST API Server
│  Backend        │
└────────┬────────┘
         │ Loads
         ▼
┌─────────────────┐
│  ML Models      │  ← Random Forest + 1D CNN
│  (Python)       │
└─────────────────┘
```

## Quick Start

### 1. Backend Setup

```bash
cd backend

# Install dependencies
pip install -r requirements.txt

# Train models (generates synthetic data)
python model_training.py

# Start API server
uvicorn api_server:app --reload
```

API will be available at `http://localhost:8000`

### 2. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Create .env.local file
echo "NEXT_PUBLIC_API_BASE_URL=http://localhost:8000" > .env.local

# Start development server
npm run dev
```

Web app will be available at `http://localhost:3000`

## Project Structure

```
glucose-monitor-demo/
├── backend/
│   ├── data_loading.py          # Synthetic dataset generator
│   ├── model_training.py        # Train ML models
│   ├── evaluation.py            # Model evaluation
│   ├── api_server.py            # FastAPI server
│   ├── requirements.txt
│   ├── models/                  # Saved models
│   └── README.md
├── frontend/
│   ├── app/
│   │   ├── page.tsx            # Main prediction page
│   │   └── layout.tsx
│   ├── components/
│   │   ├── DisclaimerBanner.tsx
│   │   ├── PredictionForm.tsx
│   │   └── ResultsDisplay.tsx
│   ├── lib/
│   │   └── api.ts              # API client
│   ├── package.json
│   └── README.md
└── README.md                    # This file
```

## Features

### Backend (Python + FastAPI)
- Synthetic dataset generation with realistic correlations
- Random Forest baseline model
- 1D CNN advanced model (PyTorch)
- REST API with `/predict`, `/health`, `/info` endpoints
- CORS enabled for frontend
- Medical disclaimers in all responses

### Frontend (Next.js + TypeScript)
- Prominent medical disclaimer banner
- Demo mode: Select from 100 pre-loaded samples
- Manual mode: Enter basic features
- Color-coded results (low/normal/high glucose)
- Responsive design
- Ready for Vercel deployment

## Deployment

### Backend
Deploy to Railway, Render, or AWS:
```bash
# Example with Railway
railway up
```

### Frontend
Deploy to Vercel:
```bash
# Install Vercel CLI
npm install -g vercel

# Deploy
cd frontend
vercel

# Set environment variable in Vercel dashboard:
# NEXT_PUBLIC_API_BASE_URL=https://your-backend-url.com
```

## Technical Details

### Synthetic Data
The system generates 5000 synthetic samples with:
- 10 features (HR, HRV, PPG signals, etc.)
- Realistic correlations between features and glucose
- Glucose range: 70-200 mg/dL
- Train/val/test split: 70/15/15

### Models
- **Random Forest**: MAE ~15-20 mg/dL on synthetic data
- **1D CNN**: Similar performance, demonstrates deep learning approach

### API
- FastAPI with automatic OpenAPI docs
- Pydantic validation
- CORS middleware
- Health checks

## Development

### Run Tests
```bash
cd backend
python evaluation.py
```

### View API Docs
Navigate to `http://localhost:8000/docs` when server is running

### Modify Features
Edit `data_loading.py` to change the synthetic data generation logic

## Safety & Legal

This system includes multiple layers of medical disclaimers:
1. **Banner** on web app (top of page)
2. **API responses** include disclaimer text
3. **Documentation** in all README files
4. **Results display** shows warning with every prediction

**Remember**: This is a research tool for learning about ML in healthcare. It should NEVER be used for actual medical decisions.

## License

This is a research demonstration. Not for commercial use.

## Contributing

This is an educational demo. Feel free to fork and modify for learning purposes.

## Support

For questions about the code or architecture, please refer to the individual README files in `backend/` and `frontend/` directories.

---

**Final Reminder**: This system uses synthetic data and is NOT validated for medical use. Always consult qualified healthcare professionals for medical advice and glucose monitoring.
