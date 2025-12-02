"""
CSV Data Preparation Helper

This script helps you:
1. Examine your downloaded CSV file
2. Map columns to required features
3. Prepare data for testing

Usage:
    python prepare_csv.py path/to/your/data.csv
"""

import pandas as pd
import sys

def examine_csv(csv_path):
    """Examine CSV file structure."""
    print("="*60)
    print("CSV FILE EXAMINATION")
    print("="*60)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    
    print(f"\n📊 File: {csv_path}")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {len(df.columns)}")
    
    print("\n📋 Column Names:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2d}. {col}")
    
    print("\n🔍 First 3 Rows:")
    print(df.head(3).to_string())
    
    print("\n📈 Data Types:")
    print(df.dtypes.to_string())
    
    print("\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("   No missing values!")
    else:
        print(missing[missing > 0].to_string())
    
    return df

def suggest_mapping(df):
    """Suggest column mappings."""
    print("\n" + "="*60)
    print("SUGGESTED COLUMN MAPPING")
    print("="*60)
    
    required_features = [
        'age', 'gender', 'weight', 'height', 'bmi',
        'heart_rate', 'hrv_rmssd', 'bp_systolic', 'bp_diastolic',
        'respiratory_rate', 'temperature', 'spo2',
        'time_since_meal', 'meal_type', 'activity_level',
        'sleep_hours', 'stress_level', 'hydration',
        'diabetic_status', 'on_medications', 'family_history',
        'fatigue_level', 'thirst_level', 'frequent_urination', 'blurred_vision'
    ]
    
    columns_lower = {col.lower(): col for col in df.columns}
    
    print("\n✅ Columns Found:")
    found = []
    for feature in required_features:
        if feature in columns_lower:
            found.append(feature)
            print(f"   ✓ {feature} → {columns_lower[feature]}")
    
    print(f"\n❌ Columns Missing ({len(required_features) - len(found)}):")
    for feature in required_features:
        if feature not in columns_lower:
            print(f"   ✗ {feature}")
    
    print("\n💡 Tip: Missing columns will be filled with default values")
    
    return found

def create_mapping_template(csv_path):
    """Create a mapping template script."""
    df = pd.read_csv(csv_path)
    
    template = f"""
# Data Mapping Template
# Edit this file to map your CSV columns to required features

import pandas as pd

# Load your data
data = pd.read_csv('{csv_path}')

# Create mapped dataframe
mapped_data = pd.DataFrame({{
    # Demographics
    'age': data['Age'] if 'Age' in data.columns else 50,  # EDIT THIS
    'gender': data['Gender'] if 'Gender' in data.columns else 0,  # 0=Female, 1=Male
    'weight': data['Weight'] if 'Weight' in data.columns else 75,  # kg
    'height': data['Height'] if 'Height' in data.columns else 170,  # cm
    'bmi': data['BMI'] if 'BMI' in data.columns else 25,
    
    # Vital Signs
    'heart_rate': data['HeartRate'] if 'HeartRate' in data.columns else 75,
    'hrv_rmssd': data['HRV'] if 'HRV' in data.columns else 45,
    'bp_systolic': data['BloodPressure'] if 'BloodPressure' in data.columns else 120,
    'bp_diastolic': 80,  # Often not in datasets
    'respiratory_rate': 16,  # Default
    'temperature': 36.6,  # Default
    'spo2': 97,  # Default
    
    # Lifestyle
    'time_since_meal': 2,  # Default (hours)
    'meal_type': 3,  # Default (0=Fasting, 1=Carb, 2=Protein, 3=Balanced)
    'activity_level': 1,  # Default (0=Sedentary, 1=Light, 2=Moderate, 3=Intense)
    'sleep_hours': 7,  # Default
    'stress_level': 5,  # Default (1-10)
    'hydration': 1,  # Default (0=Low, 1=Normal, 2=High)
    
    # Medical History
    'diabetic_status': data['Outcome'] if 'Outcome' in data.columns else 0,  # EDIT THIS
    'on_medications': 0,  # Default
    'family_history': 0,  # Default
    
    # Symptoms
    'fatigue_level': 5,  # Default (1-10)
    'thirst_level': 5,  # Default (1-10)
    'frequent_urination': 0,  # Default (0 or 1)
    'blurred_vision': 0,  # Default (0 or 1)
    
    # Target (if available)
    'glucose': data['Glucose'] if 'Glucose' in data.columns else None  # EDIT THIS
}})

# Save mapped data
mapped_data.to_csv('mapped_data.csv', index=False)
print("✓ Mapped data saved to 'mapped_data.csv'")
print(f"  Rows: {{len(mapped_data)}}")
print(f"  Columns: {{len(mapped_data.columns)}}")
"""
    
    with open('map_data.py', 'w') as f:
        f.write(template)
    
    print("\n" + "="*60)
    print("MAPPING TEMPLATE CREATED")
    print("="*60)
    print("\n✓ Created 'map_data.py'")
    print("\nNext steps:")
    print("1. Edit 'map_data.py' to match your CSV columns")
    print("2. Run: python map_data.py")
    print("3. Test: python test_real_data.py")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python prepare_csv.py path/to/your/data.csv")
        print("\nExample:")
        print("  python prepare_csv.py ~/Downloads/diabetes.csv")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    # Examine CSV
    df = examine_csv(csv_path)
    
    # Suggest mapping
    suggest_mapping(df)
    
    # Create template
    create_mapping_template(csv_path)
    
    print("\n" + "="*60)
    print("✓ EXAMINATION COMPLETE")
    print("="*60)
