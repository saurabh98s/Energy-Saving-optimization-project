import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os
from sklearn import __version__ as sklearn_version

def prepare_data(df):
    """Prepare data for anomaly detection."""
    # Create features
    df_features = pd.DataFrame()
    df_features['Daily Usage (kWh)'] = df['Daily Usage (kWh)'] 
    df_features['Hour'] = df.index.hour
    df_features['DayOfWeek'] = df.index.dayofweek
    df_features['IsWeekend'] = df_features['DayOfWeek'] >= 5
    return df_features

def train_anomaly_detection_model(df_features):
    """Train the anomaly detection model."""
    # Using a more robust configuration
    model = IsolationForest(
        contamination=0.01,
        random_state=42,
        n_estimators=100,
        max_samples='auto',
        bootstrap=True
    )
    # Fit the model with feature names
    feature_names = df_features.columns.tolist()
    model.fit(df_features)
    return model

def detect_anomalies(model, df_features):
    """Detect anomalies in the data."""
    predictions = model.predict(df_features)
    df_features['Anomaly'] = predictions
    anomalies = df_features[df_features['Anomaly'] == -1]
    return anomalies

def save_model(model, model_path):
    """Save the trained model with version information."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model metadata along with the model
    model_data = {
        'model': model,
        'sklearn_version': sklearn_version,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    try:
        joblib.dump(model_data, model_path, compress=3)
        print(f"Model saved successfully with scikit-learn version {sklearn_version}")
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model_stored(model_path):
    """Load the model with version checking."""
    try:
        model_data = joblib.load(model_path)
        saved_version = model_data['sklearn_version']
        current_version = sklearn_version
        
        if saved_version != current_version:
            print(f"Warning: Model was saved with scikit-learn {saved_version}, "
                  f"but current version is {current_version}")
        
        return model_data['model']
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    # Load processed data
    data_path = 'data/processed/daily_usage.csv'
    model_path = 'models/anomaly_detection_model.pkl'
    
    try:
        df_usage = pd.read_csv(data_path, parse_dates=['Timestamp'], index_col='Timestamp')
        
        # Prepare data
        df_features = prepare_data(df_usage)
        print("Training features:", df_features.columns.tolist())
        # Train model
        print("Training new model...")
        model = train_anomaly_detection_model(df_features)
        
        # Test the model before saving
        print("Testing model...")
        test_anomalies = detect_anomalies(model, df_features)
        print(f"Found {len(test_anomalies)} anomalies in training data")
        
        # Save model
        print("Saving model...")
        save_model(model, model_path)
        
        # Verify loading
        print("Verifying model can be loaded...")
        loaded_model = load_model_stored(model_path)
        if loaded_model is not None:
            print("Model successfully saved and loaded!")
        
    except Exception as e:
        print(f"Error in training pipeline: {e}")