import csv
from urllib.parse import urlparse, parse_qs

def extract_id_value(url):
    query = urlparse(url).query
    params = parse_qs(query)
    # Extract only the value for the 'id' parameter
    return params.get('id', [])  # Returns an empty list if 'id' is not in the URL

# File paths
input_file_path = '/home/haneen/GP/alerts2.txt'
output_file_path = '/home/haneen/GP/extracted_id_values.csv'

with open(input_file_path, 'r') as file, open(output_file_path, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['Encoded payload'])  # Write header row

    urls = file.readlines()
    # Process each URL
    for url in urls:
        url = url.strip()  # Remove leading/trailing whitespace
        id_values = extract_id_value(url)
        for value in id_values:
            csv_writer.writerow([value])  # Write each 'id' value in a new row

print(f"Extracted 'id' values have been saved to {output_file_path}")
