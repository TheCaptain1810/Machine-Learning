import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('Housing.csv')

# Impute missing values (e.g., with mean or median)
df['price'].fillna(df['price'].mean(), inplace=True)

# Drop rows with missing values
df.dropna(inplace=True)

print(df.iloc[5])
