"""
Data Loading and Preprocessing for Non-Invasive Glucose Estimation

⚠️ DISCLAIMER: This uses SYNTHETIC data for research/educational purposes only.
This is NOT real medical data and should NOT be used for clinical decisions.

This module generates synthetic PPG-like features and glucose values with
realistic correlations to simulate a non-invasive glucose monitoring dataset.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Set random seed for reproducibility
np.random.seed(42)

class GlucoseDataset:
    """
    Synthetic dataset generator for non-invasive glucose estimation.
    
    Features simulate:
    - Heart Rate (HR)
    - Heart Rate Variability (HRV)
    - PPG Signal Amplitude
    - PPG Pulse Width
    - Signal Quality Index
    - And other derived metrics
    
    Target: Blood glucose levels (mg/dL)
    """
    
    def __init__(self, n_samples=5000):
        self.n_samples = n_samples
        self.feature_names = [
            'heart_rate',
            'hrv_rmssd',
            'ppg_amplitude',
            'ppg_pulse_width',
            'signal_quality',
            'perfusion_index',
            'spo2_estimate',
            'temperature',
            'activity_level',
            'time_since_meal'
        ]
        self.scaler = StandardScaler()
        
    def generate_synthetic_data(self):
        """
        Generate synthetic data with realistic correlations.
        
        Simulated relationships:
        - Higher glucose → slightly elevated heart rate
        - Higher glucose → reduced HRV
        - Post-meal time → affects glucose levels
        - Activity → affects both HR and glucose
        """
        
        # Base glucose levels (normal distribution around 100 mg/dL)
        glucose_base = np.random.normal(100, 20, self.n_samples)
        
        # Clip to realistic range (70-200 mg/dL)
        glucose = np.clip(glucose_base, 70, 200)
        
        # Generate features with correlations to glucose
        
        # Heart rate: slightly correlated with glucose
        heart_rate = 70 + (glucose - 100) * 0.15 + np.random.normal(0, 8, self.n_samples)
        heart_rate = np.clip(heart_rate, 50, 120)
        
        # HRV: inversely correlated with glucose (high glucose → low HRV)
        hrv_rmssd = 50 - (glucose - 100) * 0.1 + np.random.normal(0, 10, self.n_samples)
        hrv_rmssd = np.clip(hrv_rmssd, 20, 80)
        
        # PPG amplitude: weak correlation
        ppg_amplitude = 1.0 + (glucose - 100) * 0.001 + np.random.normal(0, 0.2, self.n_samples)
        ppg_amplitude = np.clip(ppg_amplitude, 0.5, 2.0)
        
        # PPG pulse width: weak inverse correlation
        ppg_pulse_width = 0.3 - (glucose - 100) * 0.0005 + np.random.normal(0, 0.05, self.n_samples)
        ppg_pulse_width = np.clip(ppg_pulse_width, 0.2, 0.5)
        
        # Signal quality: random with slight glucose dependency
        signal_quality = 0.85 - (glucose - 100) * 0.0003 + np.random.normal(0, 0.1, self.n_samples)
        signal_quality = np.clip(signal_quality, 0.5, 1.0)
        
        # Perfusion index
        perfusion_index = 2.5 + np.random.normal(0, 0.8, self.n_samples)
        perfusion_index = np.clip(perfusion_index, 1.0, 5.0)
        
        # SpO2 estimate (mostly independent)
        spo2_estimate = 97 + np.random.normal(0, 1.5, self.n_samples)
        spo2_estimate = np.clip(spo2_estimate, 94, 100)
        
        # Temperature (mostly independent)
        temperature = 36.5 + np.random.normal(0, 0.3, self.n_samples)
        temperature = np.clip(temperature, 35.5, 37.5)
        
        # Activity level
        activity_level = np.random.uniform(0, 10, self.n_samples)
        
        # Time since last meal (hours) - affects glucose
        time_since_meal = np.random.exponential(2, self.n_samples)
        time_since_meal = np.clip(time_since_meal, 0, 8)
        
        # Adjust glucose based on time since meal
        glucose = glucose + (2 - time_since_meal) * 5
        glucose = np.clip(glucose, 70, 200)
        
        # Create DataFrame
        data = pd.DataFrame({
            'heart_rate': heart_rate,
            'hrv_rmssd': hrv_rmssd,
            'ppg_amplitude': ppg_amplitude,
            'ppg_pulse_width': ppg_pulse_width,
            'signal_quality': signal_quality,
            'perfusion_index': perfusion_index,
            'spo2_estimate': spo2_estimate,
            'temperature': temperature,
            'activity_level': activity_level,
            'time_since_meal': time_since_meal,
            'glucose': glucose
        })
        
        return data
    
    def load_and_split_data(self, test_size=0.15, val_size=0.15):
        """
        Generate data and split into train/val/test sets.
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test, scaler
        """
        print("Generating synthetic dataset...")
        data = self.generate_synthetic_data()
        
        # Separate features and target
        X = data[self.feature_names].values
        y = data['glucose'].values
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42
        )
        
        # Fit scaler on training data only
        self.scaler.fit(X_train)
        
        # Transform all sets
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Dataset generated:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Val:   {X_val.shape[0]} samples")
        print(f"  Test:  {X_test.shape[0]} samples")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Glucose range: {y.min():.1f} - {y.max():.1f} mg/dL")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, self.scaler
    
    def save_scaler(self, filepath='models/scaler.pkl'):
        """Save the fitted scaler for later use in API."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {filepath}")
    
    def save_test_samples(self, X_test, y_test, filepath='models/test_samples.pkl'):
        """Save some test samples for demo purposes in the API."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save first 100 test samples
        test_data = {
            'X': X_test[:100],
            'y': y_test[:100]
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(test_data, f)
        print(f"Test samples saved to {filepath}")


def glucose_to_range(glucose_value):
    """
    Convert continuous glucose value to categorical range.
    
    Args:
        glucose_value: Blood glucose in mg/dL
        
    Returns:
        String: 'low', 'normal', or 'high'
    """
    if glucose_value < 70:
        return 'low'
    elif glucose_value <= 140:
        return 'normal'
    else:
        return 'high'


if __name__ == '__main__':
    # Example usage
    dataset = GlucoseDataset(n_samples=5000)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = dataset.load_and_split_data()
    
    # Save scaler and test samples
    dataset.save_scaler()
    dataset.save_test_samples(X_test, y_test)
    
    print("\nData loading complete!")
    print("Ready for model training.")
