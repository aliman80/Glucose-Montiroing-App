"""
Enhanced API Server with Patient Management

⚠️ CRITICAL DISCLAIMER: Research/Educational Demo ONLY
- NOT a medical device
- NOT clinically validated
- NOT FDA approved
- Uses SYNTHETIC data
- DO NOT use for medical decisions

This enhanced API provides:
- Patient registration and management
- Comprehensive glucose predictions (25 features)
- Prediction history tracking
- Feature importance analysis
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
import pickle
import numpy as np
from datetime import datetime

from database import GlucoseDatabase
from data_loading_v2 import glucose_to_range

# Medical disclaimer
MEDICAL_DISCLAIMER = """
⚠️ CRITICAL MEDICAL DISCLAIMER ⚠️

This is an EXPERIMENTAL RESEARCH DEMONSTRATION ONLY.

- This is NOT a medical device
- This is NOT clinically validated  
- This is NOT FDA approved
- This uses SYNTHETIC data for educational purposes only
- DO NOT use this to make health decisions
- DO NOT use this for diagnosis or treatment
- ALWAYS consult a qualified healthcare professional

This system is for educational and research purposes ONLY.
"""

# Initialize FastAPI app
app = FastAPI(
    title="Enhanced Glucose Monitoring API",
    description="Research demo API with patient management (NOT for medical use)",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models
rf_model = None
scaler = None
test_samples = None
feature_importance = None
db = None

# Pydantic Models

class PatientCreate(BaseModel):
    """Schema for creating a new patient."""
    name: str = Field(..., min_length=1, max_length=100)
    email: Optional[str] = Field(None, max_length=100)

class PatientResponse(BaseModel):
    """Schema for patient response."""
    id: int
    name: str
    email: Optional[str]
    created_at: str

class EnhancedPredictionInput(BaseModel):
    """Input schema for enhanced prediction with 25 features."""
    patient_id: Optional[int] = None
    sample_id: Optional[int] = None
    
    # Demographics
    age: Optional[int] = Field(None, ge=18, le=100)
    gender: Optional[int] = Field(None, ge=0, le=1)  # 0=Female, 1=Male
    weight: Optional[float] = Field(None, ge=40, le=200)
    height: Optional[float] = Field(None, ge=140, le=220)
    
    # Vital Signs
    heart_rate: Optional[int] = Field(None, ge=40, le=150)
    hrv: Optional[float] = Field(None, ge=10, le=100)
    bp_systolic: Optional[int] = Field(None, ge=80, le=200)
    bp_diastolic: Optional[int] = Field(None, ge=50, le=120)
    respiratory_rate: Optional[int] = Field(None, ge=10, le=30)
    temperature: Optional[float] = Field(None, ge=35, le=40)
    spo2: Optional[int] = Field(None, ge=85, le=100)
    
    # Lifestyle
    time_since_meal: Optional[float] = Field(None, ge=0, le=12)
    meal_type: Optional[int] = Field(None, ge=0, le=3)  # 0=Fasting, 1=Carb, 2=Protein, 3=Balanced
    activity_level: Optional[int] = Field(None, ge=0, le=3)  # 0=Sedentary, 1=Light, 2=Moderate, 3=Intense
    sleep_hours: Optional[float] = Field(None, ge=0, le=12)
    stress_level: Optional[int] = Field(None, ge=1, le=10)
    hydration: Optional[int] = Field(None, ge=0, le=2)  # 0=Low, 1=Normal, 2=High
    
    # Medical History
    diabetic_status: Optional[int] = Field(None, ge=0, le=3)  # 0=No, 1=Pre, 2=Type2, 3=Type1
    on_medications: Optional[int] = Field(None, ge=0, le=1)
    family_history: Optional[int] = Field(None, ge=0, le=1)
    
    # Symptoms
    fatigue_level: Optional[int] = Field(None, ge=1, le=10)
    thirst_level: Optional[int] = Field(None, ge=1, le=10)
    frequent_urination: Optional[int] = Field(None, ge=0, le=1)
    blurred_vision: Optional[int] = Field(None, ge=0, le=1)

class PredictionOutput(BaseModel):
    """Output schema for prediction."""
    prediction_id: Optional[int] = None
    estimated_glucose: float
    glucose_range: str
    confidence: float
    model_used: str
    top_features: List[Dict[str, float]]
    notes: str

# Startup event
@app.on_event("startup")
async def load_models_and_db():
    """Load models, scaler, and initialize database on startup."""
    global rf_model, scaler, test_samples, feature_importance, db
    
    print("Loading enhanced models and initializing database...")
    
    # Load Random Forest model
    try:
        with open('models_v2/rf_model_enhanced.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        print("✓ Enhanced Random Forest model loaded")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        raise
    
    # Load scaler
    try:
        with open('models_v2/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Scaler loaded")
    except Exception as e:
        print(f"✗ Failed to load scaler: {e}")
        raise
    
    # Load test samples
    try:
        with open('models_v2/test_samples.pkl', 'rb') as f:
            test_samples = pickle.load(f)
        print(f"✓ Test samples loaded ({len(test_samples['X'])} samples)")
    except Exception as e:
        print(f"✗ Failed to load test samples: {e}")
        raise
    
    # Load metrics for feature importance
    try:
        with open('models_v2/metrics.pkl', 'rb') as f:
            metrics = pickle.load(f)
            feature_importance = metrics['train']['feature_importance']
        print("✓ Feature importance loaded")
    except Exception as e:
        print(f"✗ Failed to load metrics: {e}")
    
    # Initialize database
    try:
        db = GlucoseDatabase('glucose_monitor.db')
        print("✓ Database initialized")
    except Exception as e:
        print(f"✗ Failed to initialize database: {e}")
        raise
    
    print("All systems loaded successfully!")

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Enhanced Glucose Monitoring API",
        "version": "2.0.0",
        "status": "running",
        "disclaimer": MEDICAL_DISCLAIMER,
        "endpoints": {
            "patients": "/patients",
            "predict": "/predict",
            "history": "/patients/{id}/history",
            "info": "/info",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": rf_model is not None,
        "database_connected": db is not None
    }

@app.get("/info")
async def get_info():
    """Get system information and disclaimer."""
    stats = db.get_statistics() if db else {}
    return {
        "system": "Enhanced Glucose Monitoring Demo",
        "version": "2.0.0",
        "features": 25,
        "model": "Random Forest (150 trees)",
        "disclaimer": MEDICAL_DISCLAIMER,
        "statistics": stats
    }

# Patient Management Endpoints

@app.post("/patients", response_model=PatientResponse)
async def create_patient(patient: PatientCreate):
    """Create a new patient."""
    try:
        patient_id = db.create_patient(patient.name, patient.email)
        patient_data = db.get_patient(patient_id)
        return PatientResponse(**patient_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients", response_model=List[PatientResponse])
async def get_patients(limit: int = 100):
    """Get all patients."""
    try:
        patients = db.get_all_patients(limit)
        return [PatientResponse(**p) for p in patients]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/patients/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: int):
    """Get patient by ID."""
    patient = db.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return PatientResponse(**patient)

@app.get("/patients/{patient_id}/history")
async def get_patient_history(patient_id: int, limit: int = 50):
    """Get prediction history for a patient."""
    try:
        history = db.get_patient_history(patient_id, limit)
        return {
            "patient_id": patient_id,
            "total_predictions": len(history),
            "history": history
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Prediction Endpoint

@app.post("/predict", response_model=PredictionOutput)
async def predict_glucose(input_data: EnhancedPredictionInput):
    """
    Make glucose prediction with enhanced features.
    """
    # Get features
    if input_data.sample_id is not None:
        # Use demo sample
        if input_data.sample_id < 0 or input_data.sample_id >= len(test_samples['X']):
            raise HTTPException(
                status_code=400,
                detail=f"sample_id must be between 0 and {len(test_samples['X'])-1}"
            )
        features = test_samples['X'][input_data.sample_id].reshape(1, -1)
        features_dict = None
    else:
        # Use provided features
        # Calculate BMI if height and weight provided
        bmi = None
        if input_data.weight and input_data.height:
            bmi = input_data.weight / ((input_data.height / 100) ** 2)
        
        # Build feature array (25 features)
        features_array = [
            input_data.age or 50,
            input_data.gender or 0,
            input_data.weight or 75,
            input_data.height or 170,
            bmi or 25,
            input_data.heart_rate or 75,
            input_data.hrv or 45,
            input_data.bp_systolic or 120,
            input_data.bp_diastolic or 80,
            input_data.respiratory_rate or 16,
            input_data.temperature or 36.6,
            input_data.spo2 or 97,
            input_data.time_since_meal if input_data.time_since_meal is not None else 2,
            input_data.meal_type or 3,
            input_data.activity_level or 1,
            input_data.sleep_hours or 7,
            input_data.stress_level or 5,
            input_data.hydration or 1,
            input_data.diabetic_status or 0,
            input_data.on_medications or 0,
            input_data.family_history or 0,
            input_data.fatigue_level or 3,
            input_data.thirst_level or 3,
            input_data.frequent_urination or 0,
            input_data.blurred_vision or 0
        ]
        
        features = np.array(features_array).reshape(1, -1)
        features = scaler.transform(features)
        
        # Store features for database
        feature_names = [
            'age', 'gender', 'weight', 'height', 'bmi',
            'heart_rate', 'hrv', 'bp_systolic', 'bp_diastolic',
            'respiratory_rate', 'temperature', 'spo2',
            'time_since_meal', 'meal_type', 'activity_level',
            'sleep_hours', 'stress_level', 'hydration',
            'diabetic_status', 'on_medications', 'family_history',
            'fatigue_level', 'thirst_level', 'frequent_urination', 'blurred_vision'
        ]
        features_dict = dict(zip(feature_names, features_array))
    
    # Make prediction
    prediction = rf_model.predict(features)[0]
    
    # Calculate confidence (simplified - based on ensemble variance)
    tree_predictions = [tree.predict(features)[0] for tree in rf_model.estimators_[:50]]
    confidence = 1.0 - (np.std(tree_predictions) / 50.0)
    confidence = np.clip(confidence, 0.5, 0.95)
    
    # Convert to range
    glucose_range = glucose_to_range(prediction)
    
    # Get top contributing features
    top_features = []
    if feature_importance:
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        top_features = [{"feature": name, "importance": float(imp)} for name, imp in sorted_features]
    
    # Save to database if patient_id provided
    prediction_id = None
    if input_data.patient_id and features_dict:
        result = {
            'estimated_glucose': float(prediction),
            'glucose_range': glucose_range,
            'confidence': float(confidence),
            'model_used': 'Enhanced Random Forest'
        }
        prediction_id = db.save_prediction(input_data.patient_id, features_dict, result)
    
    # Prepare response
    response = PredictionOutput(
        prediction_id=prediction_id,
        estimated_glucose=round(float(prediction), 1),
        glucose_range=glucose_range,
        confidence=round(float(confidence), 2),
        model_used="Enhanced Random Forest (25 features)",
        top_features=top_features,
        notes=MEDICAL_DISCLAIMER
    )
    
    return response

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
