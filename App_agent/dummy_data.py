import pandas as pd
from faker import Faker
import random

# Initialize Faker
fake = Faker()

# Generate synthetic data
data = []
genders = ['Male', 'Female', 'Other']

for i in range(100):
    user = {
        'user_id': i + 1,
        'name' : fake.name(),
        'email': fake.email(),
        'address': fake.address().replace('\n', ', '),
        'phone_number': fake.phone_number(),
        'age': random.randint(18, 70),
        'gender': random.choice(genders),
        'salary': round(random.uniform(30000, 150000), 2)
    }
    data.append(user)

# Create DataFrame
df = pd.DataFrame(data)

# Save to Parquet
df.to_parquet('synthetic_users.parquet', index=False)

print("Parquet file 'synthetic_users.parquet' has been created.")
