import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('people.csv')
print("File without removing duplicates:")
print(df)

# Remove duplicate rows
df.drop_duplicates(inplace=True)
print("File after removing duplicates:")
print(df)
