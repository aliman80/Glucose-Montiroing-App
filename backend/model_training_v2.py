"""
Enhanced Model Training with 25+ Features

⚠️ DISCLAIMER: Research/educational purposes only.
NOT for clinical use. Uses synthetic data.

Trains Random Forest model on comprehensive patient features.
"""

import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from data_loading_v2 import EnhancedGlucoseDataset, glucose_to_range

# Set random seed
np.random.seed(42)

def train_enhanced_random_forest(X_train, y_train, X_val, y_val):
    """
    Train Random Forest with 25 features.
    """
    print("\n" + "="*60)
    print("Training Enhanced Random Forest Model")
    print("="*60)
    
    # Initialize model with optimized hyperparameters
    rf_model = RandomForestRegressor(
        n_estimators=150,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    # Train
    print("\nTraining on {} samples with {} features...".format(X_train.shape[0], X_train.shape[1]))
    rf_model.fit(X_train, y_train)
    
    # Validate
    y_pred_train = rf_model.predict(X_train)
    y_pred_val = rf_model.predict(X_val)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_r2 = r2_score(y_val, y_pred_val)
    
    # Classification accuracy (glucose ranges)
    y_val_ranges = [glucose_to_range(y) for y in y_val]
    y_pred_ranges = [glucose_to_range(y) for y in y_pred_val]
    range_accuracy = sum([1 for true, pred in zip(y_val_ranges, y_pred_ranges) if true == pred]) / len(y_val_ranges)
    
    print("\n" + "-"*60)
    print("Training Results:")
    print("-"*60)
    print(f"  Train MAE:  {train_mae:.2f} mg/dL")
    print(f"  Val MAE:    {val_mae:.2f} mg/dL")
    print(f"  Val RMSE:   {val_rmse:.2f} mg/dL")
    print(f"  Val R²:     {val_r2:.4f}")
    print(f"  Range Acc:  {range_accuracy*100:.1f}%")
    print("-"*60)
    
    # Feature importance
    feature_names = [
        'age', 'gender', 'weight', 'height', 'bmi',
        'heart_rate', 'hrv_rmssd', 'bp_systolic', 'bp_diastolic',
        'respiratory_rate', 'temperature', 'spo2',
        'time_since_meal', 'meal_type', 'activity_level',
        'sleep_hours', 'stress_level', 'hydration',
        'diabetic_status', 'on_medications', 'family_history',
        'fatigue_level', 'thirst_level', 'frequent_urination', 'blurred_vision'
    ]
    
    importances = rf_model.feature_importances_
    feature_importance = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Most Important Features:")
    print("-"*60)
    for i, (name, importance) in enumerate(feature_importance[:10], 1):
        print(f"  {i:2d}. {name:20s}: {importance:.4f}")
    print("-"*60)
    
    return rf_model, {
        'train_mae': train_mae,
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'range_accuracy': range_accuracy,
        'feature_importance': dict(feature_importance)
    }


def evaluate_on_test(model, X_test, y_test):
    """
    Evaluate model on test set.
    """
    print("\n" + "="*60)
    print("Test Set Evaluation")
    print("="*60)
    
    y_pred = model.predict(X_test)
    
    test_mae = mean_absolute_error(y_test, y_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    test_r2 = r2_score(y_test, y_pred)
    
    # Range accuracy
    y_test_ranges = [glucose_to_range(y) for y in y_test]
    y_pred_ranges = [glucose_to_range(y) for y in y_pred]
    range_accuracy = sum([1 for true, pred in zip(y_test_ranges, y_pred_ranges) if true == pred]) / len(y_test_ranges)
    
    print(f"  Test MAE:   {test_mae:.2f} mg/dL")
    print(f"  Test RMSE:  {test_rmse:.2f} mg/dL")
    print(f"  Test R²:    {test_r2:.4f}")
    print(f"  Range Acc:  {range_accuracy*100:.1f}%")
    print("="*60)
    
    return {
        'test_mae': test_mae,
        'test_rmse': test_rmse,
        'test_r2': test_r2,
        'range_accuracy': range_accuracy
    }


def save_model(model, filepath='models_v2/rf_model_enhanced.pkl'):
    """Save trained model."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {filepath}")


def save_metrics(train_metrics, test_metrics, filepath='models_v2/metrics.pkl'):
    """Save training metrics."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    metrics = {
        'train': train_metrics,
        'test': test_metrics
    }
    with open(filepath, 'wb') as f:
        pickle.dump(metrics, f)
    print(f"Metrics saved to {filepath}")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("ENHANCED GLUCOSE MONITORING MODEL TRAINING")
    print("="*60)
    print("\n⚠️  DISCLAIMER: Research/Educational Use Only")
    print("   NOT for clinical decisions or real patient use\n")
    
    # Load enhanced dataset
    dataset = EnhancedGlucoseDataset(n_samples=10000)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = dataset.load_and_split_data()
    
    # Save scaler and test samples
    dataset.save_scaler('models_v2/scaler.pkl')
    dataset.save_test_samples(X_test, y_test, 'models_v2/test_samples.pkl')
    
    # Train enhanced Random Forest
    rf_model, train_metrics = train_enhanced_random_forest(X_train, y_train, X_val, y_val)
    
    # Evaluate on test set
    test_metrics = evaluate_on_test(rf_model, X_test, y_test)
    
    # Save model and metrics
    save_model(rf_model, 'models_v2/rf_model_enhanced.pkl')
    save_metrics(train_metrics, test_metrics, 'models_v2/metrics.pkl')
    
    print("\n" + "="*60)
    print("✓ Enhanced Model Training Complete!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Start enhanced API server")
    print("  2. Build enhanced frontend")
    print("  3. Deploy to production")
    print("\n")
