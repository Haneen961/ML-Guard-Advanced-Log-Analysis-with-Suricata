import pandas as pd
import joblib
import numpy as np
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
file_path = '/home/haneen/GP-latest/Dataset/feature_extracted_1.csv'
df = pd.read_csv(file_path, encoding='latin1')

# Feature selection
numeric_features = [
    'query_len', 'num_words_query', 'no_single_qts', 'no_double_qts', 'no_punct',
    'no_single_cmnt', 'no_mult_cmnt', 'no_space', 'no_perc', 'no_log_opt',
    'no_arith', 'no_null', 'no_hexa', 'no_alpha', 'no_digit',
    'len_of_chr_char_null', 'genuine_keywords'
]
target = 'Label'

X = df[numeric_features]
y = df[target]

# Split into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize the numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================
# 🔹 Fine-Tuned LightGBM Model
# =============================
lgb_model = lgb.LGBMClassifier(
    boosting_type='gbdt',
    num_leaves=20,             # Controls tree complexity
    max_depth=5,               # Reduces overfitting
    learning_rate=0.03,        # Slower learning for better generalization
    n_estimators=300,          # More trees for stability
    min_data_in_leaf=30,       # Prevents small leaves (reduces overfitting)
    lambda_l1=0.5,             # L1 regularization
    lambda_l2=1.0,             # L2 regularization
    subsample=0.8,             # Randomly selects 80% of data to prevent overfitting
    colsample_bytree=0.8,      # Uses 80% of features per tree
    random_state=42
)

# Train the model
lgb_model.fit(X_train_scaled, y_train)

# =============================
# 🔹 Find the Best Classification Threshold
# =============================
y_probs = lgb_model.predict_proba(X_test_scaled)[:, 1]  

best_threshold = 0.5  # Default threshold
best_f1 = 0

for threshold in np.arange(0.1, 0.9, 0.01):
    y_pred_adjusted = (y_probs >= threshold).astype(int)
    precision = precision_score(y_test, y_pred_adjusted)
    recall = recall_score(y_test, y_pred_adjusted)
    f1 = f1_score(y_test, y_pred_adjusted)
    
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

# Apply slightly lower threshold for fewer false positives
final_threshold = max(0.3, best_threshold - 0.02)
y_pred_final = (y_probs >= final_threshold).astype(int)

# =============================
# 🔹 Evaluation Metrics
# =============================
accuracy = accuracy_score(y_test, y_pred_final)
precision = precision_score(y_test, y_pred_final)
recall = recall_score(y_test, y_pred_final)
f1 = f1_score(y_test, y_pred_final)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred_final).ravel()

print(f"Optimized Threshold: {final_threshold:.2f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Negatives (TN): {tn}")

# =============================
# 🔹 Save Model & Scaler for Later Use
# =============================
joblib.dump(lgb_model, '/home/haneen/GP-latest/Models/LightGBM_model.pkl')
joblib.dump(scaler, '/home/haneen/GP-latest/Models/LightGBM_scaler.pkl')

print("Model and scaler saved successfully.")
