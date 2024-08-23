import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset into a pandas DataFrame
df = pd.read_csv('Housing.csv')
print("\nDataFrame without Label Encoding:")
print(df)

# Identify non-numeric columns
non_numeric_columns = df.select_dtypes(exclude=['number']).columns

# OR Convert non-numeric columns using label encoding
le = LabelEncoder()
for col in non_numeric_columns:
    df[col] = le.fit_transform(df[col])

print("\nDataFrame with Label Encoding:")
print(df)
