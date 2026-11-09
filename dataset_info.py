from src.data_loader import load_data

df = load_data()

print("=" * 60)
print("COLUMN NAMES")
print("=" * 60)

for i, col in enumerate(df.columns, start=1):
    print(f"{i}. {col}")

print("\n")

print("=" * 60)
print("MISSING VALUES")
print("=" * 60)

print(df.isnull().sum())

print("\n")

print("=" * 60)
print("DUPLICATES")
print("=" * 60)

print(df.duplicated().sum())