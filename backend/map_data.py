"""
Map Pima Indians Diabetes Dataset to Required Features

This script maps the Pima Indians Diabetes dataset columns
to the 25 features required by the glucose monitoring model.

Dataset columns:
- Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
  BMI, DiabetesPedigreeFunction, Age, Outcome
"""

import pandas as pd
import numpy as np

print("="*60)
print("MAPPING PIMA INDIANS DIABETES DATASET")
print("="*60)

# Load the original data
data = pd.read_csv('/Users/muhammadali/Documents/Antigravity/Diabetes/diabetes.csv')

print(f"\n📊 Original data: {len(data)} rows, {len(data.columns)} columns")

# Create mapped dataframe with all 25 required features
mapped_data = pd.DataFrame({
    # Demographics
    'age': data['Age'],
    'gender': 0,  # All female in Pima Indians dataset
    'weight': data['BMI'] * 25,  # Estimate weight from BMI (assuming avg height 160cm)
    'height': 160,  # Average height for this population
    'bmi': data['BMI'],
    
    # Vital Signs
    'heart_rate': 70 + (data['Glucose'] - 100) * 0.1,  # Estimate based on glucose
    'hrv_rmssd': 50 - (data['Age'] - 30) * 0.3,  # Estimate: decreases with age
    'bp_systolic': data['BloodPressure'],  # Direct mapping
    'bp_diastolic': data['BloodPressure'] * 0.67,  # Estimate (typical ratio)
    'respiratory_rate': 16,  # Default (not in dataset)
    'temperature': 36.6,  # Default (not in dataset)
    'spo2': 97,  # Default (not in dataset)
    
    # Lifestyle
    'time_since_meal': np.random.uniform(1, 4, len(data)),  # Random estimate
    'meal_type': 3,  # Default: balanced
    'activity_level': 1,  # Default: light activity
    'sleep_hours': 7,  # Default
    'stress_level': 5,  # Default
    'hydration': 1,  # Default: normal
    
    # Medical History
    'diabetic_status': data['Outcome'],  # 0=No diabetes, 1=Has diabetes
    'on_medications': data['Insulin'].apply(lambda x: 1 if x > 0 else 0),  # Has insulin = on meds
    'family_history': data['DiabetesPedigreeFunction'].apply(lambda x: 1 if x > 0.5 else 0),  # High pedigree = family history
    
    # Symptoms (estimate based on glucose and diabetes status)
    'fatigue_level': np.clip(3 + (data['Glucose'] - 100) * 0.03 + data['Outcome'] * 2, 1, 10).astype(int),
    'thirst_level': np.clip(3 + (data['Glucose'] - 100) * 0.04 + data['Outcome'] * 2, 1, 10).astype(int),
    'frequent_urination': ((data['Glucose'] > 140) & (data['Outcome'] == 1)).astype(int),
    'blurred_vision': ((data['Glucose'] > 160) & (data['Outcome'] == 1)).astype(int),
    
    # Target
    'glucose': data['Glucose']
})

# Clean up any invalid values
mapped_data = mapped_data.replace([np.inf, -np.inf], np.nan)
mapped_data = mapped_data.fillna(mapped_data.median())

# Remove rows with glucose = 0 (invalid data)
original_len = len(mapped_data)
mapped_data = mapped_data[mapped_data['glucose'] > 0]
removed = original_len - len(mapped_data)

if removed > 0:
    print(f"\n⚠️  Removed {removed} rows with invalid glucose values (glucose = 0)")

# Save mapped data
output_path = 'mapped_diabetes_data.csv'
mapped_data.to_csv(output_path, index=False)

print(f"\n✓ Mapped data saved to '{output_path}'")
print(f"  Final rows: {len(mapped_data)}")
print(f"  Columns: {len(mapped_data.columns)}")

print("\n📊 Glucose Statistics:")
print(f"  Min: {mapped_data['glucose'].min():.1f} mg/dL")
print(f"  Max: {mapped_data['glucose'].max():.1f} mg/dL")
print(f"  Mean: {mapped_data['glucose'].mean():.1f} mg/dL")
print(f"  Median: {mapped_data['glucose'].median():.1f} mg/dL")

print("\n📈 Diabetes Distribution:")
print(f"  No Diabetes: {(mapped_data['diabetic_status'] == 0).sum()} ({(mapped_data['diabetic_status'] == 0).mean()*100:.1f}%)")
print(f"  Has Diabetes: {(mapped_data['diabetic_status'] == 1).sum()} ({(mapped_data['diabetic_status'] == 1).mean()*100:.1f}%)")

print("\n" + "="*60)
print("✓ MAPPING COMPLETE!")
print("="*60)
print("\nNext step:")
print("  python -c \"from test_real_data import RealDataTester; tester = RealDataTester(); tester.test_on_real_data('mapped_diabetes_data.csv')\"")
