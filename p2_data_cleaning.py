#--------------------------------------------
# Project 2, Step 1: Data Cleaning
#--------------------------------------------

# Step 1: Import pandas
import pandas as pd

# Step 2: Loading CSV file
csv_path = r"C:\Users\MAHA ZEHRA\Desktop\fintech_itv.csv" 
df = pd.read_csv(csv_path)

# Step 3: Check initial data
print("What do we have here then:", df.shape)
print("A lil' preview:")
print(df.head())

# Step 4: Standardize column names
df.columns = [
    col.strip().lower().replace(" ", "_").replace("-", "_")
    for col in df.columns
]

# Step 5: Dropping duplicate rows (if any)
df = df.drop_duplicates()

# Step 6: Separating categorical vs numeric
categorical = [
    'customer_id', 'location', 'income_level',
    'app_usage_frequency', 'preferred_payment_method'
]
numeric = [col for col in df.columns if col not in categorical]

# Step 7: Converting all numeric columns (errors='coerce' will turn junk into NaN)
df[numeric] = df[numeric].apply(pd.to_numeric, errors='coerce')

# Step 8: Check for missing values
print("\n🧼 Missing values before dropping:")
print(df[numeric].isnull().sum())

# Step 9: Drop rows with missing numeric data
df = df.dropna(subset=numeric).reset_index(drop=True)

# Step 10: Final check
print("Think it worked:", df.shape)
print(df.head())

# Step 11: Save cleaned file
df.to_csv(r"C:\Users\MAHA ZEHRA\Desktop\fintech_itv_cleaned.csv", index=False)
print("You're good! Move on.")
