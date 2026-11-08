import zipfile
import os

zip_file = "phishing-and-legitimate-urls.zip"
extract_folder = "dataset"

os.makedirs(extract_folder, exist_ok=True)

with zipfile.ZipFile(zip_file, "r") as zip_ref:
    zip_ref.extractall(extract_folder)

print("Dataset extracted successfully!")
