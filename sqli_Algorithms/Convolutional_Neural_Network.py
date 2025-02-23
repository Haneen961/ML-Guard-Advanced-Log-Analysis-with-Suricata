import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib

# Load dataset
df = pd.read_csv('/home/haneen/GP-latest/Dataset/feature_extracted_1.csv')

# Define features and target
numeric_features = [
    'query_len', 'num_words_query', 'no_single_qts', 'no_double_qts', 'no_punct',
    'no_single_cmnt', 'no_mult_cmnt', 'no_space', 'no_perc', 'no_log_opt',
    'no_arith', 'no_null', 'no_hexa', 'no_alpha', 'no_digit',
    'len_of_chr_char_null', 'genuine_keywords'
]
text_feature = 'Query'  # Column containing SQL queries

target = 'Label'

# Separate features and target
X_numeric = df[numeric_features].values
X_text = df[text_feature].astype(str)  # Ensure text data is string
y = df[target].values

# Encode target labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
y = to_categorical(y)  # One-hot encoding

# **Step 1: Process Text Data with TfidfVectorizer**
vectorizer = TfidfVectorizer(max_features=5000)  # Limit vocabulary size
X_text_tfidf = vectorizer.fit_transform(X_text).toarray()  # Convert sparse matrix to dense array

# Save vectorizer for future use
joblib.dump(vectorizer, "/home/haneen/GP-latest/Models/CNN_vectorizer.pkl")

# **Step 2: Standardize Numeric Data**
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(X_numeric)
joblib.dump(scaler, "/home/haneen/GP-latest/Models/CNN_scaler.pkl")  # Save scaler for later use

# **Step 3: Combine Text and Numeric Data**
X_combined = np.concatenate([X_text_tfidf, X_numeric_scaled], axis=1)

# Reshape for CNN input (samples, timesteps, features)
X_combined = X_combined.reshape(X_combined.shape[0], X_combined.shape[1], 1)

# **Step 4: Split Data**
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.3, random_state=42)

# **Step 5: Build CNN Model**
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")

# Save trained model
joblib.dump(model, "/home/haneen/GP-latest/Models/CNN_model.pkl")

# Predict on test data
y_pred = model.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))
y_test_labels = label_encoder.inverse_transform(np.argmax(y_test, axis=1))

# Display sample predictions
for i in range(5):
    print(f"True: {y_test_labels[i]}, Predicted: {y_pred_labels[i]}")
