import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('people.csv')

# Impute missing values (e.g., with mean or median)
df['FirstName'].fillna(df['FirstName'].mean(), inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

print(df)
