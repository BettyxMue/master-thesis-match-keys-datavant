import pandas as pd
import random
from faker import Faker
import re
# from genderComputer import GenderComputer
from gender_guesser.detector import Detector

# Reinitialize Faker after kernel reset
fake = Faker('de_DE')
#Faker.seed(1234)
#random.seed(1234)

Faker.seed(4321)
random.seed(4321)

# Helper to normalize strings
def normalize(s):
    return re.sub(r'\W+', '', s.lower().strip()) if s else ""

# Function to clean and extract first and last names robustly
def extract_clean_names_robust(profile_name):
    # Define a list of common German and academic prefixes/titles
    known_titles = {
        "dr", "prof", "bsc", "msc", "ba", "ma", "mba", "b.eng", "m.eng", "dipl", "dipl-ing", "diplom",
        "mag", "med", "herr", "frau", "ing", "univprof"
    }

    # Clean punctuation, lowercase, and split
    parts = re.sub(r'[^\wäöüÄÖÜß\- ]+', '', profile_name.lower()).split()

    # Filter out titles and honorifics
    name_parts = [part for part in parts if part not in known_titles]

    # Assign names based on what's left
    if len(name_parts) == 0:
        return "", ""
    elif len(name_parts) == 1:
        return name_parts[0].capitalize(), ""
    else:
        # Heuristic: first name is the first part, last name is the rest
        first_name = name_parts[0].capitalize()
        last_name = " ".join([p.capitalize() for p in name_parts[1:]])
        return first_name, last_name

# Updated function to generate a single record with aligned gender
def generate_german_record():
    gender = random.choice(['M', 'F'])
    profile = fake.simple_profile(sex='M' if gender == 'M' else 'F')

    # Extract clean first and last name
    first_name, last_name = extract_clean_names_robust(profile['name'])

    # Adjust gender based on first name
    detector = Detector()
    # Check if the first name includes whitespace or '-'
    if '-' in first_name:
        first_name_part = first_name.split('-')[0]
    elif ' ' in last_name:
        first_name_part = first_name.split(' ')[0]
    else:
        first_name_part = first_name

    # Use the first part of the name for gender detection
    guess = detector.get_gender(first_name_part)
    if guess == 'male' or guess == "mostly male":
        gender = 'M'
    elif guess == 'female' or guess == "mostly female":
        gender = 'F'
    else:
        gender = 'U' # unknown

    dob = profile['birthdate'].strftime('%Y%m%d')
    year_of_birth = profile['birthdate'].strftime('%Y')

    """ # Generate email aligned with first and last name but with randomization
    email_domains = ["example.com", "mail.com", "test.org"]
    random_domain = random.choice(email_domains)
    email = f"{normalize(first_name)}.{normalize(last_name)}{random.randint(1, 99)}@{random_domain}" """

    # email = fake.email()
    email = profile['mail']
    postcode = fake.postcode()
    phone = fake.phone_number()
    address = fake.street_address()

    return {
        "first_name": first_name,
        "last_name": last_name,
        "dob": dob,
        "year_of_birth": year_of_birth,
        "gender": gender,
        "zip": postcode,
        "email": email,
        "phone": phone,
        "address": address
    }

# Generate records
records = [generate_german_record() for _ in range(1000)]
df = pd.DataFrame(records)

# Save to CSV
csv_path = "known_records_1000.csv"
df.to_csv(csv_path, index=False)