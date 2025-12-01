"""
Enhanced Data Loading for Comprehensive Glucose Estimation

⚠️ DISCLAIMER: This uses SYNTHETIC data for research/educational purposes only.
This is NOT real medical data and should NOT be used for clinical decisions.

This module generates synthetic data with 25+ features including:
- Demographics (age, gender, weight, height, BMI)
- Vital signs (HR, BP, temp, SpO2, respiratory rate)
- Lifestyle factors (sleep, activity, stress, meals)
- Medical history (diabetes status, medications, family history)
- Current symptoms (fatigue, thirst, vision, urination)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Set random seed for reproducibility
np.random.seed(42)

class EnhancedGlucoseDataset:
    """
    Enhanced synthetic dataset generator with 25+ features.
    """
    
    def __init__(self, n_samples=10000):
        self.n_samples = n_samples
        self.feature_names = [
            # Demographics
            'age', 'gender', 'weight', 'height', 'bmi',
            # Vital Signs
            'heart_rate', 'hrv_rmssd', 'bp_systolic', 'bp_diastolic',
            'respiratory_rate', 'temperature', 'spo2',
            # Lifestyle
            'time_since_meal', 'meal_type', 'activity_level',
            'sleep_hours', 'stress_level', 'hydration',
            # Medical History
            'diabetic_status', 'on_medications', 'family_history',
            # Symptoms
            'fatigue_level', 'thirst_level', 'frequent_urination', 'blurred_vision'
        ]
        self.scaler = StandardScaler()
        
    def generate_synthetic_data(self):
        """
        Generate comprehensive synthetic data with realistic correlations.
        """
        
        # Demographics
        age = np.random.normal(50, 15, self.n_samples)
        age = np.clip(age, 18, 85).astype(int)
        
        gender = np.random.choice([0, 1], self.n_samples)  # 0=Female, 1=Male
        
        weight = np.random.normal(75, 15, self.n_samples)
        weight = np.clip(weight, 45, 150)
        
        height = np.random.normal(170, 10, self.n_samples)
        height = np.clip(height, 150, 200)
        
        bmi = weight / ((height / 100) ** 2)
        
        # Medical History (affects baseline glucose)
        diabetic_status = np.random.choice([0, 1, 2, 3], self.n_samples, p=[0.7, 0.15, 0.1, 0.05])
        # 0=No diabetes, 1=Pre-diabetic, 2=Type 2, 3=Type 1
        
        on_medications = (diabetic_status >= 2).astype(int)
        family_history = np.random.choice([0, 1], self.n_samples, p=[0.6, 0.4])
        
        # Base glucose levels (influenced by diabetes status)
        glucose_base = np.random.normal(100, 15, self.n_samples)
        glucose_base += diabetic_status * 20  # Diabetics have higher baseline
        glucose_base += family_history * 5    # Family history adds risk
        glucose_base += (bmi - 25) * 0.5      # Higher BMI → higher glucose
        glucose_base += (age - 50) * 0.3      # Older age → higher glucose
        
        # Lifestyle factors
        time_since_meal = np.random.exponential(2.5, self.n_samples)
        time_since_meal = np.clip(time_since_meal, 0, 12)
        
        meal_type = np.random.choice([0, 1, 2, 3], self.n_samples)
        # 0=Fasting, 1=Carb-heavy, 2=Protein, 3=Balanced
        
        activity_level = np.random.choice([0, 1, 2, 3], self.n_samples)
        # 0=Sedentary, 1=Light, 2=Moderate, 3=Intense
        
        sleep_hours = np.random.normal(7, 1.5, self.n_samples)
        sleep_hours = np.clip(sleep_hours, 4, 10)
        
        stress_level = np.random.randint(1, 11, self.n_samples)
        
        hydration = np.random.choice([0, 1, 2], self.n_samples, p=[0.2, 0.6, 0.2])
        # 0=Low, 1=Normal, 2=High
        
        # Adjust glucose based on lifestyle
        glucose_base += (3 - time_since_meal) * 8  # Recent meal → higher glucose
        glucose_base += (meal_type == 1) * 15      # Carb-heavy meal → spike
        glucose_base -= (activity_level) * 5       # Activity → lower glucose
        glucose_base += (7 - sleep_hours) * 2      # Poor sleep → higher glucose
        glucose_base += stress_level * 1.5          # Stress → higher glucose
        glucose_base -= (hydration - 1) * 3        # Dehydration → higher glucose
        
        # Vital Signs (influenced by glucose levels)
        heart_rate = 70 + (glucose_base - 100) * 0.12 + np.random.normal(0, 8, self.n_samples)
        heart_rate += activity_level * 10
        heart_rate += stress_level * 1.5
        heart_rate = np.clip(heart_rate, 50, 130)
        
        hrv_rmssd = 50 - (glucose_base - 100) * 0.08 + np.random.normal(0, 10, self.n_samples)
        hrv_rmssd -= stress_level * 2
        hrv_rmssd = np.clip(hrv_rmssd, 20, 80)
        
        bp_systolic = 120 + (age - 50) * 0.5 + (bmi - 25) * 0.8
        bp_systolic += (glucose_base - 100) * 0.15
        bp_systolic += stress_level * 1.2
        bp_systolic = np.clip(bp_systolic, 90, 180).astype(int)
        
        bp_diastolic = 80 + (age - 50) * 0.3 + (bmi - 25) * 0.5
        bp_diastolic += (glucose_base - 100) * 0.08
        bp_diastolic = np.clip(bp_diastolic, 60, 110).astype(int)
        
        respiratory_rate = 16 + (glucose_base - 100) * 0.02 + np.random.normal(0, 2, self.n_samples)
        respiratory_rate += activity_level * 2
        respiratory_rate = np.clip(respiratory_rate, 12, 25).astype(int)
        
        temperature = 36.5 + np.random.normal(0, 0.3, self.n_samples)
        temperature += (glucose_base > 150) * 0.2  # High glucose → slight fever
        temperature = np.clip(temperature, 35.5, 37.8)
        
        spo2 = 97 + np.random.normal(0, 1.5, self.n_samples)
        spo2 -= (glucose_base - 100) * 0.01  # High glucose → slightly lower SpO2
        spo2 = np.clip(spo2, 92, 100).astype(int)
        
        # Symptoms (more likely with high glucose)
        fatigue_level = np.clip(
            3 + (glucose_base - 100) * 0.05 + (10 - sleep_hours) * 0.5 + np.random.randint(-2, 3, self.n_samples),
            1, 10
        ).astype(int)
        
        thirst_level = np.clip(
            3 + (glucose_base - 100) * 0.06 + (1 - hydration) * 2 + np.random.randint(-2, 3, self.n_samples),
            1, 10
        ).astype(int)
        
        frequent_urination = ((glucose_base > 140) & (np.random.random(self.n_samples) > 0.5)).astype(int)
        
        blurred_vision = ((glucose_base > 160) & (np.random.random(self.n_samples) > 0.7)).astype(int)
        
        # Final glucose calculation
        glucose = np.clip(glucose_base + np.random.normal(0, 10, self.n_samples), 60, 250)
        
        # Create DataFrame
        data = pd.DataFrame({
            # Demographics
            'age': age,
            'gender': gender,
            'weight': weight,
            'height': height,
            'bmi': bmi,
            # Vital Signs
            'heart_rate': heart_rate,
            'hrv_rmssd': hrv_rmssd,
            'bp_systolic': bp_systolic,
            'bp_diastolic': bp_diastolic,
            'respiratory_rate': respiratory_rate,
            'temperature': temperature,
            'spo2': spo2,
            # Lifestyle
            'time_since_meal': time_since_meal,
            'meal_type': meal_type,
            'activity_level': activity_level,
            'sleep_hours': sleep_hours,
            'stress_level': stress_level,
            'hydration': hydration,
            # Medical History
            'diabetic_status': diabetic_status,
            'on_medications': on_medications,
            'family_history': family_history,
            # Symptoms
            'fatigue_level': fatigue_level,
            'thirst_level': thirst_level,
            'frequent_urination': frequent_urination,
            'blurred_vision': blurred_vision,
            # Target
            'glucose': glucose
        })
        
        return data
    
    def load_and_split_data(self, test_size=0.15, val_size=0.15):
        """
        Generate data and split into train/val/test sets.
        """
        print("Generating enhanced synthetic dataset...")
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
        
        print(f"Enhanced dataset generated:")
        print(f"  Train: {X_train.shape[0]} samples")
        print(f"  Val:   {X_val.shape[0]} samples")
        print(f"  Test:  {X_test.shape[0]} samples")
        print(f"  Features: {X_train.shape[1]}")
        print(f"  Glucose range: {y.min():.1f} - {y.max():.1f} mg/dL")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, self.scaler
    
    def save_scaler(self, filepath='models_v2/scaler.pkl'):
        """Save the fitted scaler."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {filepath}")
    
    def save_test_samples(self, X_test, y_test, filepath='models_v2/test_samples.pkl'):
        """Save test samples for demo."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        test_data = {
            'X': X_test[:100],
            'y': y_test[:100]
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(test_data, f)
        print(f"Test samples saved to {filepath}")


def glucose_to_range(glucose_value):
    """Convert continuous glucose value to categorical range."""
    if glucose_value < 70:
        return 'low'
    elif glucose_value <= 140:
        return 'normal'
    else:
        return 'high'


if __name__ == '__main__':
    # Example usage
    dataset = EnhancedGlucoseDataset(n_samples=10000)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = dataset.load_and_split_data()
    
    # Save scaler and test samples
    dataset.save_scaler()
    dataset.save_test_samples(X_test, y_test)
    
    print("\nEnhanced data loading complete!")
    print("Ready for model training with 25 features.")
