import joblib

columns = joblib.load("model_columns.pkl")

print("Total Columns:", len(columns))
print("\nColumns:")
for col in columns:
    print(col)