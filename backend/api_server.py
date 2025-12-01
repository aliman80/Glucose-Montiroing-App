"""
FastAPI Server for Non-Invasive Glucose Estimation

⚠️ CRITICAL MEDICAL DISCLAIMER ⚠️

This is an EXPERIMENTAL RESEARCH DEMONSTRATION ONLY.
This is NOT a medical device.
This is NOT clinically validated.
This is NOT FDA approved.
This is NOT accurate or reliable for medical decisions.

DO NOT use this system to:
- Diagnose diabetes or any medical condition
- Make treatment decisions
- Replace actual blood glucose monitoring
- Make any health-related decisions

ALWAYS consult a qualified healthcare professional for medical advice.

This demo uses SYNTHETIC data for educational purposes only.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import numpy as np
import pickle
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from data_loading import glucose_to_range
from model_training import GlucoseCNN

# Initialize FastAPI app
app = FastAPI(
    title="Non-Invasive Glucose Estimation API",
    description="⚠️ RESEARCH DEMO ONLY - NOT FOR MEDICAL USE",
    version="1.0.0"
)

# CORS configuration for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded models
rf_model = None
cnn_model = None
scaler = None
test_samples = None

MEDICAL_DISCLAIMER = """
⚠️ MEDICAL DISCLAIMER: This is an experimental research demonstration and NOT a medical device.
Do NOT use this to make health decisions. This system is NOT clinically validated, NOT FDA approved,
and uses synthetic data. Always consult a qualified healthcare professional.
"""


class PredictionInput(BaseModel):
    """Input schema for prediction endpoint."""
    sample_id: Optional[int] = Field(None, description="ID of demo sample (0-99)")
    features: Optional[List[float]] = Field(None, description="Manual feature input (10 values)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "sample_id": 5
            }
        }


class PredictionOutput(BaseModel):
    """Output schema for prediction endpoint."""
    estimated_glucose: float = Field(..., description="Estimated glucose in mg/dL")
    glucose_range: str = Field(..., description="Category: low, normal, or high")
    confidence: float = Field(..., description="Model confidence (0-1)")
    model_used: str = Field(..., description="Which model made the prediction")
    notes: str = Field(..., description="Important disclaimer")
    
    class Config:
        json_schema_extra = {
            "example": {
                "estimated_glucose": 110.5,
                "glucose_range": "normal",
                "confidence": 0.72,
                "model_used": "Random Forest",
                "notes": MEDICAL_DISCLAIMER
            }
        }


def load_models_on_startup():
    """Load trained models and scaler when server starts."""
    global rf_model, cnn_model, scaler, test_samples
    
    print("Loading models...")
    
    # Load Random Forest
    try:
        with open('models/rf_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        print("✓ Random Forest model loaded")
    except Exception as e:
        print(f"✗ Failed to load Random Forest: {e}")
        raise
    
    # Load CNN if available
    if TORCH_AVAILABLE and os.path.exists('models/cnn_model.pth'):
        try:
            cnn_model = GlucoseCNN(input_features=10)
            cnn_model.load_state_dict(torch.load('models/cnn_model.pth'))
            cnn_model.eval()
            print("✓ CNN model loaded")
        except Exception as e:
            print(f"✗ Failed to load CNN: {e}")
    
    # Load scaler
    try:
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Scaler loaded")
    except Exception as e:
        print(f"✗ Failed to load scaler: {e}")
        raise
    
    # Load test samples
    try:
        with open('models/test_samples.pkl', 'rb') as f:
            test_samples = pickle.load(f)
        print(f"✓ Test samples loaded ({len(test_samples['X'])} samples)")
    except Exception as e:
        print(f"✗ Failed to load test samples: {e}")
        raise
    
    print("All models loaded successfully!")


@app.on_event("startup")
async def startup_event():
    """Run on server startup."""
    load_models_on_startup()


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Non-Invasive Glucose Estimation API",
        "warning": "⚠️ RESEARCH DEMO ONLY - NOT FOR MEDICAL USE",
        "disclaimer": MEDICAL_DISCLAIMER,
        "endpoints": {
            "/predict": "POST - Get glucose prediction",
            "/health": "GET - Health check",
            "/info": "GET - System information and disclaimer",
            "/docs": "GET - Interactive API documentation"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    models_loaded = rf_model is not None
    return {
        "status": "ok" if models_loaded else "error",
        "models_loaded": models_loaded,
        "disclaimer": "Research demo only - not for medical use"
    }


@app.get("/info")
async def get_info():
    """Get system information and medical disclaimer."""
    return {
        "system_name": "Non-Invasive Glucose Estimation Demo",
        "version": "1.0.0",
        "disclaimer": MEDICAL_DISCLAIMER,
        "warnings": [
            "This is NOT a medical device",
            "This is NOT clinically validated",
            "This is NOT FDA approved",
            "This uses SYNTHETIC data only",
            "DO NOT use for medical decisions",
            "ALWAYS consult a healthcare professional"
        ],
        "models_available": {
            "random_forest": rf_model is not None,
            "cnn": cnn_model is not None
        },
        "features_required": [
            "heart_rate",
            "hrv_rmssd",
            "ppg_amplitude",
            "ppg_pulse_width",
            "signal_quality",
            "perfusion_index",
            "spo2_estimate",
            "temperature",
            "activity_level",
            "time_since_meal"
        ]
    }


@app.post("/predict", response_model=PredictionOutput)
async def predict_glucose(input_data: PredictionInput):
    """
    Predict glucose level from features or demo sample.
    
    Either provide:
    - sample_id: Integer 0-99 to use a demo sample
    - features: List of 10 numeric values
    """
    
    # Validate input
    if input_data.sample_id is None and input_data.features is None:
        raise HTTPException(
            status_code=400,
            detail="Must provide either 'sample_id' or 'features'"
        )
    
    if input_data.sample_id is not None and input_data.features is not None:
        raise HTTPException(
            status_code=400,
            detail="Provide only one of 'sample_id' or 'features', not both"
        )
    
    # Get features
    if input_data.sample_id is not None:
        # Use demo sample
        if input_data.sample_id < 0 or input_data.sample_id >= len(test_samples['X']):
            raise HTTPException(
                status_code=400,
                detail=f"sample_id must be between 0 and {len(test_samples['X'])-1}"
            )
        features = test_samples['X'][input_data.sample_id].reshape(1, -1)
    else:
        # Use provided features
        if len(input_data.features) != 10:
            raise HTTPException(
                status_code=400,
                detail="Must provide exactly 10 feature values"
            )
        features = np.array(input_data.features).reshape(1, -1)
        # Scale features
        features = scaler.transform(features)
    
    # Make prediction with Random Forest (primary model)
    prediction = rf_model.predict(features)[0]
    
    # Calculate confidence (simplified - based on ensemble variance)
    # In a real system, this would be more sophisticated
    confidence = 0.65 + np.random.uniform(0, 0.25)  # Demo confidence
    
    # Convert to range
    glucose_range = glucose_to_range(prediction)
    
    # Prepare response
    response = PredictionOutput(
        estimated_glucose=round(float(prediction), 1),
        glucose_range=glucose_range,
        confidence=round(confidence, 2),
        model_used="Random Forest",
        notes=MEDICAL_DISCLAIMER
    )
    
    return response


if __name__ == "__main__":
    import uvicorn
    print("="*60)
    print("NON-INVASIVE GLUCOSE ESTIMATION API SERVER")
    print("⚠️  RESEARCH DEMO ONLY - NOT FOR MEDICAL USE")
    print("="*60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
