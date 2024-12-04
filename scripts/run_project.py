import os
import subprocess

def run_data_preprocessing():
    """Run data preprocessing for all households."""
    print("Running data preprocessing for all households...")
    subprocess.run(['python', 'scripts/data_preprocessing.py'])
    print("Data preprocessing completed.\n")

def run_anomaly_detection():
    """Run anomaly detection model training for all households."""
    print("Running anomaly detection training for all households...")
    subprocess.run(['python', 'scripts/anomaly_detection.py'])
    print("Anomaly detection model training completed.\n")

def run_dashboard():
    """Start the dashboard."""
    print("Starting the dashboard...")
    subprocess.run(['streamlit', 'run', 'dashboard/app.py'])

def test_data_preprocessing():
    """Test data preprocessing."""
    print("Testing data preprocessing output...")
    if os.path.exists('data/processed/daily_usage_1.csv') and os.path.exists('data/processed/daily_usage_2.csv'):
        print("Preprocessed data files for all households found.\n")
    else:
        print("Preprocessed data files missing. Please check the preprocessing script.\n")

def test_anomaly_detection():
    """Test anomaly detection models."""
    print("Testing anomaly detection models...")
    if os.path.exists('models/anomaly_detection_model_1.pkl') and os.path.exists('models/anomaly_detection_model_2.pkl'):
        print("Anomaly detection models for all households found.\n")
    else:
        print("Anomaly detection models missing. Please check the anomaly detection script.\n")

if __name__ == "__main__":
    run_data_preprocessing()
    test_data_preprocessing()
    
    run_anomaly_detection()
    test_anomaly_detection()
    
    print("All scripts executed successfully.")
    print("To view the dashboard, run the following command:")
    print("streamlit run dashboard/app.py")
