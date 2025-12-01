# Enhanced Glucose Monitoring System - Backend v2

⚠️ **MEDICAL DISCLAIMER**: Research demonstration ONLY. NOT for medical use.

## What's New in V2

### Enhanced Features (25 total)
- **Demographics**: Age, Gender, Weight, Height, BMI
- **Vital Signs**: HR, HRV, BP (systolic/diastolic), Respiratory Rate, Temperature, SpO2
- **Lifestyle**: Time since meal, Meal type, Activity level, Sleep hours, Stress level, Hydration
- **Medical History**: Diabetic status, Medications, Family history
- **Symptoms**: Fatigue, Thirst, Frequent urination, Blurred vision

### New Capabilities
- **Patient Database**: SQLite storage for patient records
- **Patient Management**: Create, retrieve, and list patients
- **Prediction History**: Track all predictions per patient
- **Feature Importance**: See which factors contribute most
- **Better Accuracy**: MAE 11.84 mg/dL (vs 15-20 in v1)

## Quick Start

### 1. Train Enhanced Model

```bash
cd backend
python model_training_v2.py
```

This will:
- Generate 10,000 synthetic samples with 25 features
- Train Random Forest model (150 trees)
- Save to `models_v2/` directory
- Takes ~30 seconds

### 2. Start Enhanced API Server

```bash
python -m uvicorn api_server_v2:app --reload --host 0.0.0.0 --port 8001
```

API runs at `http://localhost:8001`

### 3. Test the API

Open `http://localhost:8001/docs` for interactive API documentation.

## API Endpoints

### Patient Management

**Create Patient**
```bash
POST /patients
{
  "name": "John Doe",
  "email": "john@example.com"
}
```

**Get All Patients**
```bash
GET /patients
```

**Get Patient History**
```bash
GET /patients/{id}/history
```

### Predictions

**Make Prediction (Demo Sample)**
```bash
POST /predict
{
  "sample_id": 5
}
```

**Make Prediction (Full Features)**
```bash
POST /predict
{
  "patient_id": 1,
  "age": 45,
  "gender": 1,
  "weight": 80,
  "height": 175,
  "heart_rate": 75,
  "hrv": 45,
  "bp_systolic": 120,
  "bp_diastolic": 80,
  "respiratory_rate": 16,
  "temperature": 36.6,
  "spo2": 98,
  "time_since_meal": 2,
  "meal_type": 3,
  "activity_level": 1,
  "sleep_hours": 7,
  "stress_level": 5,
  "hydration": 1,
  "diabetic_status": 0,
  "on_medications": 0,
  "family_history": 0,
  "fatigue_level": 3,
  "thirst_level": 3,
  "frequent_urination": 0,
  "blurred_vision": 0
}
```

**Response**
```json
{
  "prediction_id": 1,
  "estimated_glucose": 105.5,
  "glucose_range": "normal",
  "confidence": 0.82,
  "model_used": "Enhanced Random Forest (25 features)",
  "top_features": [
    {"feature": "thirst_level", "importance": 0.1524},
    {"feature": "time_since_meal", "importance": 0.1490},
    {"feature": "fatigue_level", "importance": 0.1203}
  ],
  "notes": "⚠️ MEDICAL DISCLAIMER..."
}
```

## Model Performance

**Test Set Results**:
- MAE: 11.84 mg/dL
- RMSE: 14.91 mg/dL
- R²: 0.7918
- Range Accuracy: 82.5%

**Top 5 Most Important Features**:
1. Thirst Level (15.24%)
2. Time Since Meal (14.90%)
3. Fatigue Level (12.03%)
4. Diabetic Status (9.37%)
5. BP Systolic (8.90%)

## Database

The system uses SQLite (`glucose_monitor.db`) to store:
- Patient records
- Prediction history with all features
- Timestamps and results

**View Database**:
```bash
sqlite3 glucose_monitor.db
.tables
SELECT * FROM patients;
SELECT * FROM predictions LIMIT 10;
```

## Files

```
backend/
├── data_loading_v2.py          # Enhanced dataset (25 features)
├── database.py                 # SQLite patient database
├── model_training_v2.py        # Train enhanced model
├── api_server_v2.py            # Enhanced API server
├── models_v2/                  # Trained models
│   ├── rf_model_enhanced.pkl
│   ├── scaler.pkl
│   ├── test_samples.pkl
│   └── metrics.pkl
└── glucose_monitor.db          # Patient database
```

## Feature Encoding

**Gender**: 0=Female, 1=Male  
**Meal Type**: 0=Fasting, 1=Carb-heavy, 2=Protein, 3=Balanced  
**Activity Level**: 0=Sedentary, 1=Light, 2=Moderate, 3=Intense  
**Hydration**: 0=Low, 1=Normal, 2=High  
**Diabetic Status**: 0=No diabetes, 1=Pre-diabetic, 2=Type 2, 3=Type 1  
**Boolean Fields**: 0=No/False, 1=Yes/True

## Deployment

### Railway (Backend)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Deploy
railway login
railway init
railway up
```

### Environment Variables
No environment variables needed - SQLite database is file-based.

## Important Notes

- ⚠️ **NOT for medical use** - research demo only
- Uses **synthetic data** - not real patient data
- **NOT clinically validated**
- **NOT FDA approved**
- Always consult healthcare professionals

## Next Steps

1. Build enhanced frontend with multi-step form
2. Create patient dashboard
3. Add visualization charts
4. Deploy to production
