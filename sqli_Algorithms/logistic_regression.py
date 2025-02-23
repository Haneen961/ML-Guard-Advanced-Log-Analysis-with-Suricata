import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

# Load dataset
data = pd.read_csv('/home/haneen/GP-latest/Dataset/feature_extracted_1.csv', encoding='latin1')

# Assuming the dataset has 'Query', 'Length', 'Contains_SQL_Keyword', and 'Label' columns
X_text = data['Query']  # Text feature
X_numeric = data[['query_len', 'num_words_query','no_single_qts','no_double_qts','no_punct','no_single_cmnt','no_mult_cmnt','no_space','no_perc','no_log_opt','no_arith','no_null','no_hexa','no_alpha','no_digit','len_of_chr_char_null','genuine_keywords']]  # Additional numeric features
y = data['Label']  # Target variable

# Split the data into training and testing sets
X_text_train, X_text_test, X_numeric_train, X_numeric_test, y_train, y_test = train_test_split(
    X_text, X_numeric, y, test_size=0.2, random_state=42
)

# Step 1: Process text data with TfidfVectorizer
vectorizer = TfidfVectorizer()
X_text_train_tfidf = vectorizer.fit_transform(X_text_train)
X_text_test_tfidf = vectorizer.transform(X_text_test)

# Step 2: Scale numeric features
scaler = StandardScaler()
X_numeric_train_scaled = scaler.fit_transform(X_numeric_train)
X_numeric_test_scaled = scaler.transform(X_numeric_test)

# Step 3: Combine text and numeric features
X_train_combined = hstack([X_text_train_tfidf, X_numeric_train_scaled])
X_test_combined = hstack([X_text_test_tfidf, X_numeric_test_scaled])

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_combined, y_train)

# Evaluate the model
y_pred = model.predict(X_test_combined)
print(classification_report(y_test, y_pred))


import joblib

joblib.dump(model, '/home/haneen/GP-latest/Models/logistic_regression_model.pkl')
joblib.dump(vectorizer, '/home/haneen/GP-latest/Models/logistic_regression_vectorizer.pkl')
joblib.dump(scaler, '/home/haneen/GP-latest/Models/logistic_regression_scaler.pkl')

