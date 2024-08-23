import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('Housing.csv')
print("\nDataFrame without One-Hot Encoding:")
print(df)

# Identify non-numeric columns
non_numeric_columns = df.select_dtypes(exclude=['number']).columns

# Perform One-Hot Encoding on non-numeric columns
df = pd.get_dummies(df, columns=non_numeric_columns)

print("\nDataFrame with One-Hot Encoding:")
print(df)
