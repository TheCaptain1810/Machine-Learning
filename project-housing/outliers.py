import pandas as pd
from scipy import stats

# Load the dataset into a pandas DataFrame
df = pd.read_csv('Housing_processed.csv')

# Identify and handle outliers (e.g., using z-scores or IQR)
z_scores = stats.zscore(df['price'])
df = df[(z_scores < 3)]  # Remove rows with z-scores > 3 (or a different threshold)

print(df)
