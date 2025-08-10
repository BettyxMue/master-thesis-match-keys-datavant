import pandas as pd
import re

# Load a sample of the voter dataset
file_path = r"known_records_1000.csv"
df = pd.read_csv(file_path)

# Select relevant columns
df_new = df[[
    "first_name",
    "last_name",
    "dob",
    "year_of_birth",
    "gender",
    "zip",
    "email",
    "phone",
    "address"
]].copy()

############################
# Helper Functions
############################

# Normalize common German abbreviations in addresses
address_replacements = {
    "str.": "straße",
    "nr.": "nummer"
}

def normalize_address(address):
    for abbr, full in address_replacements.items():
        address = address.replace(abbr, full)
    return address

# Remove leading zeros from numbers in addresses
def remove_leading_zeros_from_address(address):
    return re.sub(r'\b0+(\d+)', r'\1', address) if isinstance(address, str) else address

############################
# Data Cleaning
############################

# Adjust first names
df_new["first_name"] = df_new["first_name"].str.strip().str.lower()

# Adjust last names
df_new["last_name"] = df_new["last_name"].str.replace("Beng", "", regex=False)
df_new["last_name"] = df_new["last_name"].str.strip().str.lower()

# Adjust addresses
df_new["address"] = df_new["address"].apply(normalize_address)
df_new["address"] = df_new["address"].str.replace("/", "", regex=False)
df_new["address"] = df_new["address"].apply(remove_leading_zeros_from_address)
df_new["address"] = df_new["address"].str.strip().str.lower()

# Adjust phone number
df_new["phone"] = df_new["phone"].str.replace("-", "", regex=False)
df_new["phone"] = df_new["phone"].str.replace("(0)", "", regex=False)
df_new["phone"] = df_new["phone"].str.replace("+49", "0", regex=False)
df_new["phone"] = df_new["phone"].str.replace("(", "", regex=False)
df_new["phone"] = df_new["phone"].str.replace(")", "", regex=False)
df_new["phone"] = df_new["phone"].str.replace(" ", "", regex=False)
df_new["phone"] = df_new["phone"].str.strip().str.lower()

# Adjust email addresses
df_new["email"] = df_new["email"].str.strip().str.lower()

# Extract the birth year from the date of birth
if df_new["year_of_birth"].empty and df_new["dob"].notna():
    df_new["year_of_birth"] = df_new["dob"].str[:4]

############################
# Save to File
############################

# Save the cleaned version
output_path = r"known_data_clean.csv"
df_new.to_csv(output_path, index=False, encoding="utf-8")