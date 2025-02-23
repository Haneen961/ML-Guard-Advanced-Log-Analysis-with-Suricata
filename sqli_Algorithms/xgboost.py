import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
file_path = '/home/haneen/GP-latest/Dataset/feature_extracted_1.csv'  # Update this path
df = pd.read_csv(file_path, encoding='latin1')

# Define features and target
features = [
    'query_len', 'num_words_query', 'no_single_qts', 'no_double_qts', 'no_punct',
    'no_single_cmnt', 'no_mult_cmnt', 'no_space', 'no_perc', 'no_log_opt',
    'no_arith', 'no_null', 'no_hexa', 'no_alpha', 'no_digit',
    'len_of_chr_char_null', 'genuine_keywords'
]
target = 'Label'  # Change this if your dataset has a different label column name

X = df[features]
y = df[target]

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the XGBoost model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',  # Binary classification (SQLi vs. normal)
    eval_metric='logloss',  # Logarithmic loss for binary classification
    use_label_encoder=False,  # Avoids warning in newer XGBoost versions
    n_estimators=100,  # Number of boosting rounds
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

xgb_model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print("\n📌 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Negatives (TN): {tn}")

import joblib

joblib.dump(xgb_model, '/home/haneen/GP-latest/Models/xgboost_model.pkl')
joblib.dump(scaler, '/home/haneen/GP-latest/Models/xgboost_scaler.pkl')
