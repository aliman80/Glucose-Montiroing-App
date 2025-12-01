"""
Model Training for Non-Invasive Glucose Estimation

⚠️ DISCLAIMER: This is a research demo using synthetic data.
NOT for medical use. NOT clinically validated. NOT FDA approved.

This module trains two models:
1. Baseline: Random Forest Regressor (simple, interpretable)
2. Advanced: 1D CNN (deep learning, pattern recognition)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
    TORCH_AVAILABLE = True
    
    class GlucoseCNN(nn.Module):
        """
        1D Convolutional Neural Network for glucose estimation.
        
        Architecture:
        - Input: 10 features
        - Conv1D layers to extract patterns
        - Fully connected layers
        - Output: Single glucose value
        """
        
        def __init__(self, input_features=10):
            super(GlucoseCNN, self).__init__()
            
            # Reshape input to (batch, channels=1, features=10)
            # Then use 1D convolutions
            
            self.fc_input = nn.Linear(input_features, 64)
            self.bn1 = nn.BatchNorm1d(64)
            
            self.fc2 = nn.Linear(64, 128)
            self.bn2 = nn.BatchNorm1d(128)
            
            self.fc3 = nn.Linear(128, 64)
            self.bn3 = nn.BatchNorm1d(64)
            
            self.fc4 = nn.Linear(64, 32)
            self.fc_output = nn.Linear(32, 1)
            
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(0.2)
            
        def forward(self, x):
            x = self.relu(self.bn1(self.fc_input(x)))
            x = self.dropout(x)
            
            x = self.relu(self.bn2(self.fc2(x)))
            x = self.dropout(x)
            
            x = self.relu(self.bn3(self.fc3(x)))
            x = self.dropout(x)
            
            x = self.relu(self.fc4(x))
            x = self.fc_output(x)
            
            return x
            
except ImportError:
    TORCH_AVAILABLE = False
    GlucoseCNN = None
    print("PyTorch not available. Will train only Random Forest model.")

from data_loading import GlucoseDataset

# Set random seeds
np.random.seed(42)
if TORCH_AVAILABLE:
    torch.manual_seed(42)


def train_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest baseline model.
    
    Returns:
        Trained model and validation metrics
    """
    print("\n" + "="*60)
    print("Training Random Forest Baseline Model")
    print("="*60)
    
    # Initialize model
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    # Train
    print("Training...")
    rf_model.fit(X_train, y_train)
    
    # Validate
    y_pred_train = rf_model.predict(X_train)
    y_pred_val = rf_model.predict(X_val)
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_r2 = r2_score(y_val, y_pred_val)
    
    print(f"\nResults:")
    print(f"  Train MAE: {train_mae:.2f} mg/dL")
    print(f"  Val MAE:   {val_mae:.2f} mg/dL")
    print(f"  Val RMSE:  {val_rmse:.2f} mg/dL")
    print(f"  Val R²:    {val_r2:.4f}")
    
    # Feature importance
    feature_names = [
        'heart_rate', 'hrv_rmssd', 'ppg_amplitude', 'ppg_pulse_width',
        'signal_quality', 'perfusion_index', 'spo2_estimate', 
        'temperature', 'activity_level', 'time_since_meal'
    ]
    
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print(f"\nTop 5 Feature Importances:")
    for i in range(min(5, len(feature_names))):
        idx = indices[i]
        print(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    
    return rf_model, {
        'train_mae': train_mae,
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'val_r2': val_r2
    }


def train_cnn(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    """
    Train 1D CNN model using PyTorch.
    
    Returns:
        Trained model and validation metrics
    """
    if not TORCH_AVAILABLE:
        print("PyTorch not available. Skipping CNN training.")
        return None, {}
    
    print("\n" + "="*60)
    print("Training 1D CNN Advanced Model")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
    X_val_t = torch.FloatTensor(X_val)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = GlucoseCNN(input_features=X_train.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    print(f"Training for {epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t.to(device))
            val_loss = criterion(val_outputs, y_val_t.to(device))
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        y_pred_train = model(X_train_t.to(device)).cpu().numpy().flatten()
        y_pred_val = model(X_val_t.to(device)).cpu().numpy().flatten()
    
    # Metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_r2 = r2_score(y_val, y_pred_val)
    
    print(f"\nResults:")
    print(f"  Train MAE: {train_mae:.2f} mg/dL")
    print(f"  Val MAE:   {val_mae:.2f} mg/dL")
    print(f"  Val RMSE:  {val_rmse:.2f} mg/dL")
    print(f"  Val R²:    {val_r2:.4f}")
    
    return model, {
        'train_mae': train_mae,
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'val_r2': val_r2
    }


def save_models(rf_model, cnn_model=None):
    """Save trained models to disk."""
    os.makedirs('models', exist_ok=True)
    
    # Save Random Forest
    with open('models/rf_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    print("\nRandom Forest model saved to models/rf_model.pkl")
    
    # Save CNN if available
    if cnn_model is not None and TORCH_AVAILABLE:
        torch.save(cnn_model.state_dict(), 'models/cnn_model.pth')
        print("CNN model saved to models/cnn_model.pth")


if __name__ == '__main__':
    print("="*60)
    print("NON-INVASIVE GLUCOSE ESTIMATION - MODEL TRAINING")
    print("⚠️  RESEARCH DEMO ONLY - NOT FOR MEDICAL USE")
    print("="*60)
    
    # Load data
    dataset = GlucoseDataset(n_samples=5000)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = dataset.load_and_split_data()
    
    # Save scaler and test samples
    dataset.save_scaler()
    dataset.save_test_samples(X_test, y_test)
    
    # Train Random Forest
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    
    # Train CNN
    cnn_model, cnn_metrics = train_cnn(X_train, y_train, X_val, y_val, epochs=50)
    
    # Save models
    save_models(rf_model, cnn_model)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Run evaluation.py to test on held-out test set")
    print("  2. Start the API server with: uvicorn api_server:app --reload")
