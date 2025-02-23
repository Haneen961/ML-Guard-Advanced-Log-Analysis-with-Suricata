import json
import time
import csv
import joblib
import numpy as np
from collections import defaultdict

# Load AI Models
models_info = [
    {"model": joblib.load("/home/haneen/GP-latest/Models/SVM_ddos_model.pkl"), "scaler": joblib.load("/home/haneen/GP-latest/Models/SVM_ddos_scaler.pkl"),"vectorizer": joblib.load("/home/haneen/GP-latest/Models/SVM_ddos_vectorizer.pkl")} ,
    {"model": joblib.load("/home/haneen/GP-latest/Models/LogisticRegression_ddos_model.pkl"), "scaler": joblib.load("/home/haneen/GP-latest/Models/LogisticRegression_ddos_scaler.pkl"),"vectorizer": joblib.load("/home/haneen/GP-latest/Models/LogisticRegression_ddos_vectorizer.pkl")} ,
    ]

# Log file paths
SURICATA_LOG_FILE = "/var/log/suricata/eve.json"
DDOS_ALERT_FILE = "/home/haneen/GP-latest/ddos_results.txt"

def monitor_suricata_logs():
    """Monitor Suricata logs and process them."""
    with open(SURICATA_LOG_FILE, "r") as file:
        file.seek(0, 2)  # Move to end of file

        while True:
            line = file.readline()
            if not line:
                time.sleep(0.1)
                continue

            try:
                log = json.loads(line)
                features = extract_features(log)
                if features:
                    count = classify_traffic(features)
                    store_ddos_results(features, count)
            except json.JSONDecodeError as e:
                print(f"[ERROR] JSON Decode Error: {e}")


# Important features to store in ddos_results.txt
IMPORTANT_FEATURES = ["pktrate", "tot_kbps", "src", "dst", "flows", "pktcount", "bytecount", "Protocol", "tx_kbps", "rx_kbps"]

def extract_features(log):
    """Extract features from Suricata log."""
    if "flow" not in log:
        return None

    return {
        "dt": int(time.time()),  # Timestamp
        "switch": 1,  # Placeholder (Adjust if actual switch info is available)
        "src": log.get("src_ip", ""),
        "dst": log.get("dest_ip", ""),
        "pktcount": log["flow"].get("pkts_toserver", 0) + log["flow"].get("pkts_toclient", 0),
        "bytecount": log["flow"].get("bytes_toserver", 0) + log["flow"].get("bytes_toclient", 0),
        "dur": 10,  # Fixed 10-second window
        "dur_nsec": 10 * 1e9,
        "tot_dur": 10 * 1e9,
        "flows": 1,
        "packetins": log["flow"].get("pkts_toserver", 0) + log["flow"].get("pkts_toclient", 0),
        "pktperflow": log["flow"].get("pkts_toserver", 0) + log["flow"].get("pkts_toclient", 0),
        "byteperflow": log["flow"].get("bytes_toserver", 0) + log["flow"].get("bytes_toclient", 0),
        "pktrate": (log["flow"].get("pkts_toserver", 0) + log["flow"].get("pkts_toclient", 0)) / 10,
        "Pairflow": 0,  # Placeholder
        "Protocol": log.get("proto", ""),
        "port_no": log.get("dest_port", 0),
        "tx_bytes": log["flow"].get("bytes_toserver", 0),
        "rx_bytes": log["flow"].get("bytes_toclient", 0),
        "tx_kbps": (log["flow"].get("bytes_toserver", 0) * 8) / 1024,
        "rx_kbps": (log["flow"].get("bytes_toclient", 0) * 8) / 1024,
        "tot_kbps": ((log["flow"].get("bytes_toserver", 0) + log["flow"].get("bytes_toclient", 0)) * 8) / 1024
            }


def classify_traffic(features):
    """Evaluate traffic using multiple AI models."""
    count_positive_predictions = 0
    numeric_features = ["dt", "switch", "pktcount", "bytecount", "dur", "dur_nsec", "tot_dur",
        "flows", "packetins", "pktperflow", "byteperflow", "pktrate", "Pairflow",
        "port_no", "tx_bytes", "rx_bytes", "tx_kbps", "rx_kbps", "tot_kbps"]
    text_features = ["src", "dst", "Protocol"]

    X_numeric = np.array([features[key] for key in numeric_features]).reshape(1, -1)
    X_text = np.array([features[key] for key in text_features]).reshape(1, -1)

    for model_data in models_info:
        model = model_data['model']
        scaler = model_data['scaler']
        vectorizer = model_data.get('vectorizer', None)

        # Standardize numeric features
        X_numeric_scaled = scaler.transform(X_numeric)

        # Transform text data if vectorizer exists
        if vectorizer:
            X_text_transformed = vectorizer.transform(X_text)
            # Check if X_text_transformed is a sparse matrix
            if hasattr(X_text_transformed, "toarray"):
                X_text_transformed = X_text_transformed.toarray()
            X_final = np.hstack((X_numeric_scaled, X_text_transformed))
        else:
            X_final = X_numeric_scaled

        # Get prediction
        prediction = model.predict(X_final)[0]
        if prediction == 1:
            count_positive_predictions += 1

    return count_positive_predictions


def store_ddos_results(features, count):
    """Store the important features and DDoS detection count."""
    important_values = [str(features[key]) for key in IMPORTANT_FEATURES]
    result_line = "\t".join(important_values) + f"\t{count}\n"

    with open(DDOS_ALERT_FILE, "a") as file:
        file.write(result_line)


if __name__ == "__main__":
    print("[INFO] Monitoring Suricata logs...")
    monitor_suricata_logs()
