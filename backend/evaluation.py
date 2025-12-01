"""
Model Evaluation for Non-Invasive Glucose Estimation

⚠️ DISCLAIMER: This is a research demo using synthetic data.
NOT for medical use. NOT clinically validated. NOT FDA approved.

This module evaluates trained models on the test set and generates
visualizations of model performance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, confusion_matrix
import pickle
import os

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from data_loading import GlucoseDataset, glucose_to_range
from model_training import GlucoseCNN

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)


def load_models():
    """Load trained models from disk."""
    # Load Random Forest
    with open('models/rf_model.pkl', 'rb') as f:
        rf_model = pickle.dump(f)
    
    # Load CNN if available
    cnn_model = None
    if TORCH_AVAILABLE and os.path.exists('models/cnn_model.pth'):
        cnn_model = GlucoseCNN(input_features=10)
        cnn_model.load_state_dict(torch.load('models/cnn_model.pth'))
        cnn_model.eval()
    
    return rf_model, cnn_model


def evaluate_model(model, X_test, y_test, model_name='Model', is_cnn=False):
    """
    Evaluate a model on test set.
    
    Returns:
        predictions, metrics dict
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print('='*60)
    
    # Get predictions
    if is_cnn and TORCH_AVAILABLE:
        with torch.no_grad():
            X_test_t = torch.FloatTensor(X_test)
            y_pred = model(X_test_t).numpy().flatten()
    else:
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    # Calculate percentage within clinical accuracy zones
    # Clarke Error Grid zones (simplified)
    errors = np.abs(y_test - y_pred)
    within_5_percent = np.mean(errors / y_test <= 0.05) * 100
    within_10_percent = np.mean(errors / y_test <= 0.10) * 100
    within_15_percent = np.mean(errors / y_test <= 0.15) * 100
    
    print(f"\nRegression Metrics:")
    print(f"  MAE:  {mae:.2f} mg/dL")
    print(f"  RMSE: {rmse:.2f} mg/dL")
    print(f"  R²:   {r2:.4f}")
    
    print(f"\nClinical Accuracy:")
    print(f"  Within  5%: {within_5_percent:.1f}%")
    print(f"  Within 10%: {within_10_percent:.1f}%")
    print(f"  Within 15%: {within_15_percent:.1f}%")
    
    # Classification metrics (low/normal/high)
    y_test_cat = [glucose_to_range(g) for g in y_test]
    y_pred_cat = [glucose_to_range(g) for g in y_pred]
    
    # Confusion matrix
    categories = ['low', 'normal', 'high']
    cm = confusion_matrix(y_test_cat, y_pred_cat, labels=categories)
    
    print(f"\nClassification Performance (Low/Normal/High):")
    print(f"  Confusion Matrix:")
    print(f"           Pred Low  Pred Normal  Pred High")
    for i, cat in enumerate(categories):
        print(f"  True {cat:7s}  {cm[i,0]:4d}      {cm[i,1]:4d}        {cm[i,2]:4d}")
    
    # Overall accuracy
    correct = np.sum(np.array(y_test_cat) == np.array(y_pred_cat))
    accuracy = correct / len(y_test_cat) * 100
    print(f"\n  Overall Accuracy: {accuracy:.1f}%")
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'within_5pct': within_5_percent,
        'within_10pct': within_10_percent,
        'within_15pct': within_15_percent,
        'classification_accuracy': accuracy
    }
    
    return y_pred, metrics


def plot_results(y_test, y_pred_rf, y_pred_cnn=None):
    """Generate evaluation plots."""
    os.makedirs('plots', exist_ok=True)
    
    # Plot 1: Predicted vs True (Random Forest)
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred_rf, alpha=0.5, s=20)
    plt.plot([70, 200], [70, 200], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('True Glucose (mg/dL)', fontsize=12)
    plt.ylabel('Predicted Glucose (mg/dL)', fontsize=12)
    plt.title('Random Forest: Predicted vs True Glucose', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/rf_predicted_vs_true.png', dpi=150)
    print("\nSaved: plots/rf_predicted_vs_true.png")
    
    # Plot 2: Error Distribution (Random Forest)
    plt.figure(figsize=(10, 6))
    errors = y_test - y_pred_rf
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.axvline(0, color='r', linestyle='--', lw=2, label='Zero Error')
    plt.xlabel('Prediction Error (mg/dL)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Random Forest: Error Distribution', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/rf_error_distribution.png', dpi=150)
    print("Saved: plots/rf_error_distribution.png")
    
    # Plot 3: Comparison (if CNN available)
    if y_pred_cnn is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # RF
        axes[0].scatter(y_test, y_pred_rf, alpha=0.5, s=20)
        axes[0].plot([70, 200], [70, 200], 'r--', lw=2)
        axes[0].set_xlabel('True Glucose (mg/dL)', fontsize=12)
        axes[0].set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
        axes[0].set_title('Random Forest', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # CNN
        axes[1].scatter(y_test, y_pred_cnn, alpha=0.5, s=20, color='green')
        axes[1].plot([70, 200], [70, 200], 'r--', lw=2)
        axes[1].set_xlabel('True Glucose (mg/dL)', fontsize=12)
        axes[1].set_ylabel('Predicted Glucose (mg/dL)', fontsize=12)
        axes[1].set_title('1D CNN', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png', dpi=150)
        print("Saved: plots/model_comparison.png")
    
    plt.close('all')


if __name__ == '__main__':
    print("="*60)
    print("NON-INVASIVE GLUCOSE ESTIMATION - MODEL EVALUATION")
    print("⚠️  RESEARCH DEMO ONLY - NOT FOR MEDICAL USE")
    print("="*60)
    
    # Load data
    dataset = GlucoseDataset(n_samples=5000)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = dataset.load_and_split_data()
    
    # Load models
    print("\nLoading trained models...")
    rf_model, cnn_model = load_models()
    
    # Evaluate Random Forest
    y_pred_rf, rf_metrics = evaluate_model(rf_model, X_test, y_test, 'Random Forest')
    
    # Evaluate CNN
    y_pred_cnn = None
    if cnn_model is not None:
        y_pred_cnn, cnn_metrics = evaluate_model(cnn_model, X_test, y_test, '1D CNN', is_cnn=True)
    
    # Generate plots
    print("\nGenerating plots...")
    plot_results(y_test, y_pred_rf, y_pred_cnn)
    
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print("\nNext step: Start the API server")
    print("  uvicorn api_server:app --reload")
