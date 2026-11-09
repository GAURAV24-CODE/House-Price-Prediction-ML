from src.data_loader import load_data
from src.preprocessing import clean_data

df = load_data()

print("Before Cleaning")
print(df.info())

print("\n")

df = clean_data(df)

print("After Cleaning")
print(df.info())

print("\nMissing Values")

print(df.isnull().sum())