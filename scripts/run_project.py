# scripts/run_project.py

import os
import subprocess

def run_data_preprocessing():
    print("Running data preprocessing...")
    subprocess.run(['python', 'scripts/data_preprocessing.py'])
    print("Data preprocessing completed.\n")

def run_anomaly_detection():
    print("Running anomaly detection training...")
    subprocess.run(['python', 'scripts/anomaly_detection.py'])
    print("Anomaly detection model training completed.\n")

def run_dashboard():
    print("Starting the dashboard...")
    subprocess.run(['streamlit', 'run', 'dashboard/app.py'])

if __name__ == "__main__":
    run_data_preprocessing()
    run_anomaly_detection()
    print("All scripts executed successfully.")
    print("To view the dashboard, run the following command:")
    print("streamlit run dashboard/app.py")
