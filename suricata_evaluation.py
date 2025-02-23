import pandas as pd
import csv
from sklearn.metrics import classification_report, confusion_matrix

# Load the original dataset with labels
original_dataset_path = './Dataset/subset_test_data.csv'  # Replace with your original dataset path
original_df = pd.read_csv(original_dataset_path, quoting=csv.QUOTE_ALL)

# Load the new labeled dataset
payloads_dataset_path = './Dataset/extracted_id_values.csv'  # Replace with your new dataset path
payloads_df = pd.read_csv(payloads_dataset_path, quoting=csv.QUOTE_ALL , delimiter=';')

output_file_path = "./Dataset/comparison_results.csv"

# Ensure column names for payloads are consistent
original_payloads = original_df['Query'].astype(str)
other_payloads = payloads_df['payload'].astype(str)

# Check if payloads exist in the other dataset
original_df['com_Label'] = original_payloads.isin(other_payloads).astype(int)

# Save the results to a new CSV file
original_df.to_csv(output_file_path, index=False)
print(f"Comparison results saved to: {output_file_path}")

# Evaluate the results
# Compare labels in the original dataset with the new computed labels
y_true = original_df['Label']  # Assuming original dataset has a column 'original_label'
y_pred = original_df['com_Label']

# Calculate precision, recall, F1-score, etc.
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Extract individual values from confusion matrix
tn, fp, fn, tp = conf_matrix.ravel()

# Calculate additional metrics
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
accuracy = (tp + tn) / (tp + tn + fp + fn)
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

# Print additional statistics
print("\nConfusion Matrix:")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")

print("\nStatistics:")
print(f"Precision: {precision:.4f}")
print(f"Recall (Sensitivity): {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"False Positive Rate (FPR): {fpr:.4f}")
print(f"True Negative Rate (TNR): {tnr:.4f}")
