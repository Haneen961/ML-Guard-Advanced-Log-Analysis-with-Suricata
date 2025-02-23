import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten


# Step 1: Load the Dataset
data = pd.read_csv('/home/haneen/GP-latest/Dataset/feature_extracted.csv', encoding='latin1')
print(data.head())

# Step 2: Preprocess the Data
X_text = data['Query'].astype(str)  # Ensure text is string format
y = data['Label']  # Binary labels (0 = benign, 1 = malicious)

# Step 3: Process text data with TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=10000)  # Limit vocabulary size
X_text_tfidf = vectorizer.fit_transform(X_text)  # Transform queries into TF-IDF vectors

# Step 4: Process numerical features
numeric_features = data.drop(columns=['Query', 'Label'])  # Remove text & target columns
scaler = StandardScaler()
X_numeric_scaled = scaler.fit_transform(numeric_features)

# Step 5: Combine text (TF-IDF) and numerical features
X_combined = np.hstack((X_text_tfidf.toarray(), X_numeric_scaled))

# Step 6: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

# Step 7: Reshape input for CNN (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Step 8: Build the Deep Learning Model (CNN with Dense Layers)
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Dropout(0.5),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Step 9: Compile the Model
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

# Step 10: Train the Model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Step 11: Evaluate the Model
y_pred = (model.predict(X_test) > 0.5).astype("int32")  # Convert probabilities to binary

# Print Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 12: Test on New Inputs (SQLi Queries)
new_queries = [
    "' AND 1=CONVERT(int, (SELECT @@version)) --",
    "' OR IF(1=1, SLEEP(5), 0) --",
    "SELECT * FROM users;",
    "' AND 1=1 --"
]

# Transform new queries using the trained TF-IDF vectorizer
X_text_new_tfidf = vectorizer.transform(new_queries)

# Assuming no additional numeric features for new queries, use zeros
X_numeric_new_scaled = np.zeros((len(new_queries), X_numeric_scaled.shape[1]))

# Combine features
X_new_combined = np.hstack((X_text_new_tfidf.toarray(), X_numeric_new_scaled))
X_new_combined = X_new_combined.reshape(X_new_combined.shape[0], X_new_combined.shape[1], 1)

# Predict SQLi vs Safe on new queries
new_predictions = model.predict(X_new_combined)

# Display predictions
for query, pred in zip(new_queries, new_predictions):
    print(f"Query: {query}")
    print("SQLi Detected!" if pred > 0.5 else "Query is Safe.")
    print("-" * 50)

import joblib

joblib.dump(model, '/home/haneen/GP-latest/Models/Deep_Learning_model.pkl')
joblib.dump(vectorizer, '/home/haneen/GP-latest/Models/Deep_Learning_vectorizer.pkl')
joblib.dump(scaler, '/home/haneen/GP-latest/Models/Deep_Learning_scaler.pkl')
