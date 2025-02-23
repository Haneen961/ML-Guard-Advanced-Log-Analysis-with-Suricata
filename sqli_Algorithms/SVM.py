import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# =============================
# 🔹 Load Dataset
# =============================
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
# 🔹 Train Support Vector Machine (SVM) Model
# =============================
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)

# Train the model
svm_model.fit(X_train_scaled, y_train)

# =============================
# 🔹 Make Predictions and Evaluate Metrics
# =============================
y_pred = svm_model.predict(X_test_scaled)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

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
joblib.dump(svm_model, '/home/haneen/GP-latest/Models/SVM_model.pkl')
joblib.dump(scaler, '/home/haneen/GP-latest/Models/SVM_scaler.pkl')

print("SVM model and scaler saved successfully.")
