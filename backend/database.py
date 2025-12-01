"""
Database Module for Patient Management

⚠️ DISCLAIMER: This is a research demo database.
NOT for storing real patient data. NOT HIPAA compliant.
For educational purposes only.

This module provides SQLite database operations for:
- Patient registration
- Prediction history storage
- Patient data retrieval
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import os

class GlucoseDatabase:
    """SQLite database for patient and prediction management."""
    
    def __init__(self, db_path='glucose_monitor.db'):
        self.db_path = db_path
        self.init_database()
    
    def get_connection(self):
        """Get database connection."""
        return sqlite3.connect(self.db_path)
    
    def init_database(self):
        """Initialize database schema."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Patients table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT UNIQUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                
                -- Demographics
                age INTEGER,
                gender INTEGER,
                weight REAL,
                height REAL,
                bmi REAL,
                
                -- Vital Signs
                heart_rate INTEGER,
                hrv REAL,
                bp_systolic INTEGER,
                bp_diastolic INTEGER,
                respiratory_rate INTEGER,
                temperature REAL,
                spo2 INTEGER,
                
                -- Lifestyle
                time_since_meal REAL,
                meal_type INTEGER,
                activity_level INTEGER,
                sleep_hours REAL,
                stress_level INTEGER,
                hydration INTEGER,
                
                -- Medical
                diabetic_status INTEGER,
                on_medications INTEGER,
                family_history INTEGER,
                
                -- Symptoms
                fatigue_level INTEGER,
                thirst_level INTEGER,
                frequent_urination INTEGER,
                blurred_vision INTEGER,
                
                -- Results
                predicted_glucose REAL,
                glucose_range TEXT,
                confidence REAL,
                model_used TEXT,
                
                FOREIGN KEY (patient_id) REFERENCES patients(id)
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"Database initialized at {self.db_path}")
    
    # Patient Operations
    
    def create_patient(self, name: str, email: Optional[str] = None) -> int:
        """Create a new patient record."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(
                'INSERT INTO patients (name, email) VALUES (?, ?)',
                (name, email)
            )
            conn.commit()
            patient_id = cursor.lastrowid
            return patient_id
        except sqlite3.IntegrityError:
            # Email already exists
            cursor.execute('SELECT id FROM patients WHERE email = ?', (email,))
            return cursor.fetchone()[0]
        finally:
            conn.close()
    
    def get_patient(self, patient_id: int) -> Optional[Dict]:
        """Get patient by ID."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT id, name, email, created_at FROM patients WHERE id = ?',
            (patient_id,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'name': row[1],
                'email': row[2],
                'created_at': row[3]
            }
        return None
    
    def get_all_patients(self, limit: int = 100) -> List[Dict]:
        """Get all patients."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT id, name, email, created_at FROM patients ORDER BY created_at DESC LIMIT ?',
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'name': row[1],
                'email': row[2],
                'created_at': row[3]
            }
            for row in rows
        ]
    
    # Prediction Operations
    
    def save_prediction(self, patient_id: int, features: Dict, result: Dict) -> int:
        """Save a prediction record."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (
                patient_id,
                age, gender, weight, height, bmi,
                heart_rate, hrv, bp_systolic, bp_diastolic,
                respiratory_rate, temperature, spo2,
                time_since_meal, meal_type, activity_level,
                sleep_hours, stress_level, hydration,
                diabetic_status, on_medications, family_history,
                fatigue_level, thirst_level, frequent_urination, blurred_vision,
                predicted_glucose, glucose_range, confidence, model_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            patient_id,
            features.get('age'), features.get('gender'), features.get('weight'),
            features.get('height'), features.get('bmi'),
            features.get('heart_rate'), features.get('hrv'), features.get('bp_systolic'),
            features.get('bp_diastolic'), features.get('respiratory_rate'),
            features.get('temperature'), features.get('spo2'),
            features.get('time_since_meal'), features.get('meal_type'),
            features.get('activity_level'), features.get('sleep_hours'),
            features.get('stress_level'), features.get('hydration'),
            features.get('diabetic_status'), features.get('on_medications'),
            features.get('family_history'),
            features.get('fatigue_level'), features.get('thirst_level'),
            features.get('frequent_urination'), features.get('blurred_vision'),
            result.get('estimated_glucose'), result.get('glucose_range'),
            result.get('confidence'), result.get('model_used')
        ))
        
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        
        return prediction_id
    
    def get_patient_history(self, patient_id: int, limit: int = 50) -> List[Dict]:
        """Get prediction history for a patient."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                id, timestamp, predicted_glucose, glucose_range,
                confidence, model_used,
                heart_rate, bp_systolic, bp_diastolic
            FROM predictions
            WHERE patient_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (patient_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': row[0],
                'timestamp': row[1],
                'predicted_glucose': row[2],
                'glucose_range': row[3],
                'confidence': row[4],
                'model_used': row[5],
                'heart_rate': row[6],
                'bp_systolic': row[7],
                'bp_diastolic': row[8]
            }
            for row in rows
        ]
    
    def get_prediction_details(self, prediction_id: int) -> Optional[Dict]:
        """Get full details of a prediction."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM predictions WHERE id = ?', (prediction_id,))
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        # Map all columns
        columns = [
            'id', 'patient_id', 'timestamp',
            'age', 'gender', 'weight', 'height', 'bmi',
            'heart_rate', 'hrv', 'bp_systolic', 'bp_diastolic',
            'respiratory_rate', 'temperature', 'spo2',
            'time_since_meal', 'meal_type', 'activity_level',
            'sleep_hours', 'stress_level', 'hydration',
            'diabetic_status', 'on_medications', 'family_history',
            'fatigue_level', 'thirst_level', 'frequent_urination', 'blurred_vision',
            'predicted_glucose', 'glucose_range', 'confidence', 'model_used'
        ]
        
        return dict(zip(columns, row))
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM patients')
        total_patients = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        
        cursor.execute('''
            SELECT AVG(predicted_glucose), MIN(predicted_glucose), MAX(predicted_glucose)
            FROM predictions
        ''')
        glucose_stats = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_patients': total_patients,
            'total_predictions': total_predictions,
            'avg_glucose': glucose_stats[0] if glucose_stats[0] else 0,
            'min_glucose': glucose_stats[1] if glucose_stats[1] else 0,
            'max_glucose': glucose_stats[2] if glucose_stats[2] else 0
        }


if __name__ == '__main__':
    # Test database
    db = GlucoseDatabase('test_glucose.db')
    
    # Create test patient
    patient_id = db.create_patient('John Doe', 'john@example.com')
    print(f"Created patient ID: {patient_id}")
    
    # Get patient
    patient = db.get_patient(patient_id)
    print(f"Patient: {patient}")
    
    # Save test prediction
    test_features = {
        'age': 45, 'gender': 1, 'weight': 80, 'height': 175, 'bmi': 26.1,
        'heart_rate': 75, 'hrv': 45, 'bp_systolic': 120, 'bp_diastolic': 80,
        'respiratory_rate': 16, 'temperature': 36.6, 'spo2': 98,
        'time_since_meal': 2, 'meal_type': 3, 'activity_level': 1,
        'sleep_hours': 7, 'stress_level': 5, 'hydration': 1,
        'diabetic_status': 0, 'on_medications': 0, 'family_history': 0,
        'fatigue_level': 3, 'thirst_level': 3, 'frequent_urination': 0, 'blurred_vision': 0
    }
    
    test_result = {
        'estimated_glucose': 105.5,
        'glucose_range': 'normal',
        'confidence': 0.75,
        'model_used': 'Random Forest'
    }
    
    pred_id = db.save_prediction(patient_id, test_features, test_result)
    print(f"Saved prediction ID: {pred_id}")
    
    # Get history
    history = db.get_patient_history(patient_id)
    print(f"Patient history: {history}")
    
    # Get stats
    stats = db.get_statistics()
    print(f"Database stats: {stats}")
    
    print("\nDatabase test complete!")
