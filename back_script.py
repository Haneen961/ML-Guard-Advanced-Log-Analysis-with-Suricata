import json
import urllib.parse
import re
import time
import joblib
import os
import numpy as np

def calculate_sqli_features(query):
    features = {}
    query = query.lower()  # Convert to lowercase

    # Feature calculations (same as before)
    features['query_len'] = len(query)  # Length of the query
    features['num_words_query'] = len(query.split())  # Number of words
    features['no_single_qts'] = len(re.findall(r"'", query))  # Single quotes
    features['no_double_qts'] = len(re.findall(r'"', query))  # Double quotes
    features['no_punct'] = len(re.findall(r"[!\"#$%&\'()*+,-.\/:;<=>?@[\\]^_`{|}~]", query))  # Punctuation
    features['no_single_cmnt'] = len(re.findall(r'(--)', query))  # Single-line comments
    features['no_mult_cmnt'] = len(re.findall(r'(\/\*)', query))  # Multi-line comments
    features['no_space'] = len(re.findall(r'\s+', query))  # Spaces
    features['no_perc'] = len(re.findall(r'%', query))  # Percentage symbols
    features['no_log_opt'] = len(re.findall(r'\snot\s|\sand\s|\sor\s|\sxor\s|&&|\|\||!', query))  # Logical operators
    features['no_arith'] = len(re.findall(r'\+|-|[^\/]\*|\/[^\*]', query))  # Arithmetic operators
    features['no_null'] = len(re.findall(r'null', query))  # Null values
    features['no_hexa'] = len(re.findall(r'0[xX][0-9a-fA-F]+\s', query))  # Hexadecimal values
    features['no_alpha'] = len(re.findall(r'[a-zA-Z]', query))  # Alphabets
    features['no_digit'] = len(re.findall(r'[0-9]', query))  # Digits
    features['len_of_chr_char_null'] = len(re.findall(r'null', query)) + \
                                       len(re.findall(r'chr', query)) + \
                                       len(re.findall(r'char', query))  # Length of chr/char/null keywords
    genuine_keys = ['select', 'top', 'order', 'fetch', 'join', 'avg', 'count', 'sum', 'rows']
    features['genuine_keywords'] = sum(1 for word in query.split() if word in genuine_keys)  # Genuine keywords

    return features

def calculate_xss_features(payload):
    features = {}
    payload = payload.lower()
    
    features['Length'] = len(payload)
    features['Tag_Count'] = len(re.findall(r"<.*?>", payload))
    features['Special_Char_Count'] = len(re.findall(r'[<>"/]', payload))
    features['JS_Keyword_Count'] = sum(1 for word in ["script", "alert", "onload", "onmouseover"] if word in payload)
    
    return features

def test_sqli_models(payload, models_info):
    """
    Test a given payload against multiple AI models and count how many predict 1.

    :param payload: The input query to test.
    :param models_info: A list of dictionaries, each containing model components (model, scaler, vectorizer if applicable).
    :return: Number of models that predict 1.
    """
    # Calculate features
    features = calculate_sqli_features(payload)  # Ensure you have this function implemented

    # Prepare features for prediction
    X_text_new = [payload]  # For models using a vectorizer
    X_numeric_new = np.array([list(features.values())])  # For all models

    count_positive_predictions = 0

    for model_data in models_info:
        model = model_data['model']
        scaler = model_data['scaler']
        vectorizer = model_data.get('vectorizer', None)  # Some models may not have a vectorizer

        # Standardize numeric features
        X_numeric_scaled = scaler.transform(X_numeric_new)

        # If the model has a vectorizer, transform text data
        if vectorizer:
            X_text_transformed = vectorizer.transform(X_text_new).toarray()
            X_final = np.hstack((X_numeric_scaled, X_text_transformed))
        else:
            X_final = X_numeric_scaled

        # Get prediction
        prediction = model.predict(X_final)  # Assuming model.predict returns an array

        # Count models that predict 1
        if np.any(prediction == 1):
            count_positive_predictions += 1

    return count_positive_predictions

def test_xss_models(payload, models_info):
    # Calculate features
    features = calculate_xss_features(payload)  # Ensure you have this function implemented

    # Prepare features for prediction
    X_text_new = [payload]  # For models using a vectorizer
    X_numeric_new = np.array([list(features.values())])  # For all models

    count_positive_predictions = 0

    for model_data in models_info:
        model = model_data['model']
        scaler = model_data['scaler']
        vectorizer = model_data.get('vectorizer', None)  # Some models may not have a vectorizer

        # Standardize numeric features
        X_numeric_scaled = scaler.transform(X_numeric_new)

        # If the model has a vectorizer, transform text data
        if vectorizer:
            X_text_transformed = vectorizer.transform(X_text_new).toarray()
            X_final = np.hstack((X_numeric_scaled, X_text_transformed))
        else:
            X_final = X_numeric_scaled

        # Get prediction
        prediction = model.predict(X_final)  # Assuming model.predict returns an array

        # Count models that predict 1
        if np.any(prediction == 1):
            count_positive_predictions += 1

    return count_positive_predictions
    

def check_suricata_log(log):
    event_type = log.get("event_type", "").lower()
    if "alert" in log:
        return "Alert"
    elif event_type == "http" or "http" in log and event_type != "alert":
        return "Normal HTTP Traffic"
    else:
        return "Unknown"

def process_log(log, suricata_result):
    url = log.get('http', {}).get('url', '')
    parsed_url = urllib.parse.urlparse(url)
    query_params = urllib.parse.parse_qs(parsed_url.query)
    payload = query_params.get('id', query_params.get('name', ['']))[0]
    #payload = query_params.get('id', [''])[0]
    payload = urllib.parse.unquote(payload)
    
    sqli_status = test_sqli_models(payload, sqli_models_info)
    xss_status = test_xss_models(payload, xss_models_info)
    
    timestamp, src_ip, src_port, protocol = log.get('timestamp', ''), log.get('src_ip', ''), log.get('src_port', ''), log.get('protocol', '')
    result = f"{timestamp}\t{src_ip}\t{src_port}\t{protocol}\t{payload}\t{suricata_result}\t{sqli_status}\t{xss_status}\n"
    
    with open("/home/haneen/GP-latest/results.txt", "a") as result_file:
        result_file.write(result)
    
    print(result)

def monitor_logs(eve_file):
    with open(eve_file, 'r') as file:
        file.seek(0, 2)
        while True:
            line = file.readline()
            if not line:
                time.sleep(0.1)
                continue
            
            try:
                log = json.loads(line)
                filename = log.get('fileinfo', {}).get('filename', '')
                dest_ip = log.get('dest_ip', '')

                # Check conditions before calling check_suricata_log
                if "/DVWA-master/vulnerabilities/" in filename and dest_ip == "192.168.243.130":
                    suricata_result = check_suricata_log(log)
                    process_log(log, suricata_result)
            except json.JSONDecodeError:
                continue

sqli_models_info = [
    {"model": joblib.load("/home/haneen/GP-latest/Models/CNN_model.pkl"), "scaler": joblib.load("/home/haneen/GP-latest/Models/CNN_scaler.pkl"), "vectorizer": joblib.load("/home/haneen/GP-latest/Models/CNN_vectorizer.pkl")},
    {"model": joblib.load("/home/haneen/GP-latest/Models/logistic_regression_model.pkl"), "scaler": joblib.load("/home/haneen/GP-latest/Models/logistic_regression_scaler.pkl"), "vectorizer": joblib.load("/home/haneen/GP-latest/Models/logistic_regression_vectorizer.pkl")},
    {"model": joblib.load("/home/haneen/GP-latest/Models/LightGBM_model.pkl"), "scaler": joblib.load("/home/haneen/GP-latest/Models/LightGBM_scaler.pkl")},  # No vectorizer
    {"model": joblib.load("/home/haneen/GP-latest/Models/SVM_model.pkl"), "scaler": joblib.load("/home/haneen/GP-latest/Models/SVM_scaler.pkl")},  # No vectorizer
    {"model": joblib.load("/home/haneen/GP-latest/Models/xgboost_model.pkl"), "scaler": joblib.load("/home/haneen/GP-latest/Models/xgboost_scaler.pkl")},
    {"model": joblib.load("/home/haneen/GP-latest/Models/random_forest_model.pkl"), "scaler": joblib.load("/home/haneen/GP-latest/Models/random_forest_scaler.pkl"), "vectorizer": joblib.load("/home/haneen/GP-latest/Models/random_forest_vectorizer.pkl")},
]

xss_models_info = [
    {"model": joblib.load("/home/haneen/GP-latest/Models/LightGBM_xss_model.pkl"), "scaler": joblib.load("/home/haneen/GP-latest/Models/LightGBM_xss_scaler.pkl")},  # No vectorizer
    {"model": joblib.load("/home/haneen/GP-latest/Models/xgboost_xss_model.pkl"), "scaler": joblib.load("/home/haneen/GP-latest/Models/xgboost_xss_scaler.pkl")},  # No vectorizer
    {"model": joblib.load("/home/haneen/GP-latest/Models/K_Neighbors_xss_model.pkl"), "scaler": joblib.load("/home/haneen/GP-latest/Models/K_Neighbors_xss_scaler.pkl")},
    {"model": joblib.load("/home/haneen/GP-latest/Models/random_forest_xss_model.pkl"), "scaler": joblib.load("/home/haneen/GP-latest/Models/random_forest_xss_scaler.pkl")},
    {"model": joblib.load("/home/haneen/GP-latest/Models/DecisionTree_xss_model.pkl"), "scaler": joblib.load("/home/haneen/GP-latest/Models/DecisionTree_xss_scaler.pkl"), "vectorizer": joblib.load("/home/haneen/GP-latest/Models/DecisionTree_xss_vectorizer.pkl")},
    {"model": joblib.load("/home/haneen/GP-latest/Models/Deep_Learning_xss_model.pkl"), "scaler": joblib.load("/home/haneen/GP-latest/Models/Deep_Learning_xss_scaler.pkl"), "vectorizer": joblib.load("/home/haneen/GP-latest/Models/Deep_Learning_xss_vectorizer.pkl")},
    {"model": joblib.load("/home/haneen/GP-latest/Models/SVM_xss_model.pkl"), "scaler": joblib.load("/home/haneen/GP-latest/Models/SVM_xss_scaler.pkl")},  # No vectorizer

]

if __name__ == "__main__":
    monitor_logs("/var/log/suricata/eve.json")
