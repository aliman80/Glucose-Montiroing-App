"""
Real Data Testing Script for Educational Purposes

⚠️ DISCLAIMER: For educational/research use ONLY
- Do NOT use for medical decisions
- Do NOT use on patients without proper approvals
- Only use de-identified, publicly available datasets

This script allows you to:
1. Import real glucose monitoring data from CSV
2. Test your model's performance on real data
3. Compare predictions vs actual values
4. Generate educational reports
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

class RealDataTester:
    """Test the glucose monitoring system on real data."""
    
    def __init__(self, model_path='models_v2/rf_model_enhanced.pkl', 
                 scaler_path='models_v2/scaler.pkl'):
        """Load trained model and scaler."""
        print("Loading model for educational testing...")
        print("⚠️  REMINDER: Educational use only - NOT for medical decisions\n")
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        
        self.feature_names = [
            'age', 'gender', 'weight', 'height', 'bmi',
            'heart_rate', 'hrv_rmssd', 'bp_systolic', 'bp_diastolic',
            'respiratory_rate', 'temperature', 'spo2',
            'time_since_meal', 'meal_type', 'activity_level',
            'sleep_hours', 'stress_level', 'hydration',
            'diabetic_status', 'on_medications', 'family_history',
            'fatigue_level', 'thirst_level', 'frequent_urination', 'blurred_vision'
        ]
    
    def load_csv_data(self, csv_path, glucose_column='glucose'):
        """
        Load real data from CSV file.
        
        CSV should have columns matching feature_names + glucose_column
        
        Example CSV format:
        age,gender,weight,height,bmi,heart_rate,...,glucose
        45,1,80,175,26.1,75,...,105.5
        """
        print(f"Loading data from {csv_path}...")
        data = pd.read_csv(csv_path)
        
        # Check for required columns
        missing_features = [f for f in self.feature_names if f not in data.columns]
        if missing_features:
            print(f"⚠️  Warning: Missing features: {missing_features}")
            print("   These will be filled with default values")
        
        # Extract features
        X = data[self.feature_names].fillna(0).values
        
        # Extract actual glucose values if available
        y_actual = data[glucose_column].values if glucose_column in data.columns else None
        
        print(f"✓ Loaded {len(X)} samples")
        return X, y_actual, data
    
    def test_on_real_data(self, csv_path, glucose_column='glucose'):
        """
        Test model on real data and generate report.
        """
        # Load data
        X, y_actual, original_data = self.load_csv_data(csv_path, glucose_column)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        print("\nMaking predictions...")
        y_pred = self.model.predict(X_scaled)
        
        # Create results dataframe
        results = pd.DataFrame({
            'predicted_glucose': y_pred,
            'predicted_range': [self._glucose_to_range(g) for g in y_pred]
        })
        
        if y_actual is not None:
            results['actual_glucose'] = y_actual
            results['actual_range'] = [self._glucose_to_range(g) for g in y_actual]
            results['error'] = y_pred - y_actual
            results['abs_error'] = np.abs(results['error'])
            
            # Calculate metrics
            mae = mean_absolute_error(y_actual, y_pred)
            rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
            r2 = r2_score(y_actual, y_pred)
            
            # Range accuracy
            range_accuracy = (results['predicted_range'] == results['actual_range']).mean()
            
            print("\n" + "="*60)
            print("EDUCATIONAL TESTING RESULTS")
            print("="*60)
            print(f"Samples tested: {len(y_pred)}")
            print(f"MAE: {mae:.2f} mg/dL")
            print(f"RMSE: {rmse:.2f} mg/dL")
            print(f"R²: {r2:.4f}")
            print(f"Range Accuracy: {range_accuracy*100:.1f}%")
            print("="*60)
            
            # Generate plots
            self._plot_results(y_actual, y_pred, results)
        
        return results
    
    def _glucose_to_range(self, glucose):
        """Convert glucose value to range."""
        if glucose < 70:
            return 'low'
        elif glucose <= 140:
            return 'normal'
        else:
            return 'high'
    
    def _plot_results(self, y_actual, y_pred, results):
        """Generate educational plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Predicted vs Actual
        axes[0, 0].scatter(y_actual, y_pred, alpha=0.5)
        axes[0, 0].plot([y_actual.min(), y_actual.max()], 
                        [y_actual.min(), y_actual.max()], 
                        'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Glucose (mg/dL)')
        axes[0, 0].set_ylabel('Predicted Glucose (mg/dL)')
        axes[0, 0].set_title('Predicted vs Actual Glucose')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Error distribution
        axes[0, 1].hist(results['error'], bins=30, edgecolor='black')
        axes[0, 1].axvline(0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Prediction Error (mg/dL)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Error Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Bland-Altman plot
        mean_glucose = (y_actual + y_pred) / 2
        diff = y_pred - y_actual
        axes[1, 0].scatter(mean_glucose, diff, alpha=0.5)
        axes[1, 0].axhline(diff.mean(), color='r', linestyle='--', lw=2, label='Mean')
        axes[1, 0].axhline(diff.mean() + 1.96*diff.std(), color='g', linestyle='--', lw=2, label='±1.96 SD')
        axes[1, 0].axhline(diff.mean() - 1.96*diff.std(), color='g', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Mean Glucose (mg/dL)')
        axes[1, 0].set_ylabel('Difference (Predicted - Actual)')
        axes[1, 0].set_title('Bland-Altman Plot')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Range confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(results['actual_range'], results['predicted_range'],
                             labels=['low', 'normal', 'high'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                   xticklabels=['Low', 'Normal', 'High'],
                   yticklabels=['Low', 'Normal', 'High'])
        axes[1, 1].set_xlabel('Predicted Range')
        axes[1, 1].set_ylabel('Actual Range')
        axes[1, 1].set_title('Range Classification Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig('real_data_test_results.png', dpi=300, bbox_inches='tight')
        print("\n✓ Plots saved to 'real_data_test_results.png'")
        plt.show()
    
    def test_single_sample(self, features_dict):
        """
        Test a single sample.
        
        Example:
        features = {
            'age': 45, 'gender': 1, 'weight': 80, 'height': 175,
            'heart_rate': 75, 'bp_systolic': 120, ...
        }
        """
        # Build feature array
        features_array = [features_dict.get(f, 0) for f in self.feature_names]
        X = np.array(features_array).reshape(1, -1)
        
        # Scale and predict
        X_scaled = self.scaler.transform(X)
        prediction = self.model.predict(X_scaled)[0]
        
        result = {
            'predicted_glucose': round(prediction, 1),
            'glucose_range': self._glucose_to_range(prediction),
            'note': '⚠️ Educational testing only - NOT for medical use'
        }
        
        return result


# Example usage
if __name__ == '__main__':
    print("="*60)
    print("GLUCOSE MONITORING SYSTEM - REAL DATA TESTING")
    print("="*60)
    print("\n⚠️  CRITICAL DISCLAIMER:")
    print("   - For EDUCATIONAL/RESEARCH purposes ONLY")
    print("   - NOT for medical decisions")
    print("   - NOT for patient care")
    print("   - Use only de-identified public datasets\n")
    
    # Initialize tester
    tester = RealDataTester()
    
    # Example 1: Test on CSV file
    print("\nExample 1: Testing on CSV file")
    print("-" * 60)
    print("To test your own data:")
    print("1. Prepare CSV with required columns")
    print("2. Run: results = tester.test_on_real_data('your_data.csv')")
    print("")
    
    # Example 2: Test single sample
    print("\nExample 2: Testing single sample")
    print("-" * 60)
    sample = {
        'age': 45,
        'gender': 1,
        'weight': 80,
        'height': 175,
        'bmi': 26.1,
        'heart_rate': 75,
        'hrv_rmssd': 45,
        'bp_systolic': 120,
        'bp_diastolic': 80,
        'respiratory_rate': 16,
        'temperature': 36.6,
        'spo2': 98,
        'time_since_meal': 2,
        'meal_type': 3,
        'activity_level': 1,
        'sleep_hours': 7,
        'stress_level': 5,
        'hydration': 1,
        'diabetic_status': 0,
        'on_medications': 0,
        'family_history': 0,
        'fatigue_level': 3,
        'thirst_level': 3,
        'frequent_urination': 0,
        'blurred_vision': 0
    }
    
    result = tester.test_single_sample(sample)
    print(f"Predicted Glucose: {result['predicted_glucose']} mg/dL")
    print(f"Range: {result['glucose_range']}")
    print(f"Note: {result['note']}")
