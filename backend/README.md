# Non-Invasive Glucose Monitor - Backend

⚠️ **MEDICAL DISCLAIMER**: This is a research demonstration ONLY. NOT a medical device. NOT for clinical use.

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the models:**
   ```bash
   python model_training.py
   ```
   
   This will:
   - Generate synthetic dataset (5000 samples)
   - Train Random Forest and 1D CNN models
   - Save models to `models/` directory
   - Save scaler and test samples

3. **(Optional) Evaluate models:**
   ```bash
   python evaluation.py
   ```
   
   This generates performance metrics and plots in `plots/`

4. **Start the API server:**
   ```bash
   uvicorn api_server:app --reload --host 0.0.0.0 --port 8000
   ```

5. **Test the API:**
   - Open `http://localhost:8000/docs` for interactive API documentation
   - Try the `/predict` endpoint with a sample:
     ```json
     {
       "sample_id": 5
     }
     ```

## API Endpoints

### `POST /predict`
Get glucose prediction from features or demo sample.

**Request:**
```json
{
  "sample_id": 5
}
```
OR
```json
{
  "features": [75, 45, 1.0, 0.3, 0.85, 2.5, 97, 36.5, 5, 2]
}
```

**Response:**
```json
{
  "estimated_glucose": 110.5,
  "glucose_range": "normal",
  "confidence": 0.72,
  "model_used": "Random Forest",
  "notes": "⚠️ MEDICAL DISCLAIMER..."
}
```

### `GET /health`
Health check endpoint.

### `GET /info`
System information and medical disclaimer.

## Project Structure

```
backend/
├── data_loading.py         # Synthetic dataset generation
├── model_training.py       # Train RF and CNN models
├── evaluation.py           # Evaluate on test set
├── api_server.py           # FastAPI server
├── requirements.txt        # Python dependencies
├── models/                 # Saved model artifacts
│   ├── rf_model.pkl
│   ├── cnn_model.pth
│   ├── scaler.pkl
│   └── test_samples.pkl
└── plots/                  # Evaluation plots
```

## Features

The synthetic dataset includes 10 features:
1. Heart Rate (bpm)
2. Heart Rate Variability (ms)
3. PPG Amplitude
4. PPG Pulse Width
5. Signal Quality Index
6. Perfusion Index
7. SpO2 Estimate
8. Temperature (°C)
9. Activity Level
10. Time Since Last Meal (hours)

## Models

- **Random Forest**: Baseline model, fast and interpretable
- **1D CNN**: Advanced model using PyTorch (optional)

## Important Notes

- Uses **SYNTHETIC** data only
- **NOT** for medical use
- **NOT** clinically validated
- **NOT** FDA approved
- Always consult healthcare professionals
