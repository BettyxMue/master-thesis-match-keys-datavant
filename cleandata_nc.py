import pandas as pd
import re
from gender_guesser.detector import Detector

# Load a sample of the voter dataset
file_path = r"nc_voter_clean_new_dob.csv"
df = pd.read_csv(file_path)

# Select relevant columns
df_new = df[[
    "first_name",
    "last_name",
    "dob",
    "year_of_birth",
    "zip",
    "address",
    "phone",
    "gender"
]].copy()

############################
# Helper Functions
############################

# Replace German special characters with ASCII equivalents
"""def replace_german_chars(s):
    if isinstance(s, str):
        s = s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
        s = s.replace("Ä", "Ae").replace("Ö", "Oe").replace("Ü", "Ue")
        s = s.replace("ß", "ss")
    return s"""

# Normalize common German abbreviations in addresses
address_replacements = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "ct": "court",
    "pl": "place",
    "hwy": "highway",
    "pkwy": "parkway",
    "trl": "trail"
}

def normalize_address(address):
    if not isinstance(address, str):
        address = str(address) if not pd.isna(address) else ""
    for abbr, full in address_replacements.items():
        address = address.replace(abbr, full)
    return address

############################
# Data Cleaning
############################

"""for col in ["first_name", "last_name", "address", "email"]:
    if col in df_new.columns:
        df_new[col] = df_new[col].apply(replace_german_chars)"""

# Remove entries with missing values
df_new = df_new.dropna()

# Remove duplicates
df_new = df_new.drop_duplicates(keep="first", ignore_index=True)

# Adjust first names
df_new["first_name"] = df_new["first_name"].str.strip().str.lower()

# Adjust last names
df_new["last_name"] = df_new["last_name"].str.strip().str.lower()

# Adjust addresses
df_new["address"] = df_new["address"].apply(normalize_address)
df_new["address"] = df_new["address"].str.strip().str.lower()

# Adjust zip
df_new["zip"] = pd.to_numeric(df_new["zip"], errors="coerce").astype('Int64').astype(str).str.strip().str.lower()

# Adjust phone number
"""df_new["phone"] = df_new["phone"].str.replace("-", "", regex=False)
df_new["phone"] = df_new["phone"].str.replace("(0)", "", regex=False)
df_new["phone"] = df_new["phone"].str.replace("+49", "0", regex=False)
df_new["phone"] = df_new["phone"].str.replace("(", "", regex=False)
df_new["phone"] = df_new["phone"].str.replace(")", "", regex=False)
df_new["phone"] = df_new["phone"].str.replace(" ", "", regex=False)"""
df_new["phone"] = pd.to_numeric(df_new["phone"], errors="coerce").astype('Int64').astype(str).str.strip().str.lower()

# Adjust birthday
df_new["dob"] = pd.to_datetime(df_new["dob"], errors="coerce")
df_new["dob"] = df_new["dob"].dt.strftime('%Y%m%d')

# Extract the birth year from the date of birth
if df_new["year_of_birth"].empty and df_new["dob"].notna():
    df_new["year_of_birth"] = df_new["dob"].str[:4]

############################
# Save to File
############################

# Save the cleaned version
output_path = r"nc_cleaned.csv"
df_new.to_csv(output_path, index=False, encoding="utf-8")