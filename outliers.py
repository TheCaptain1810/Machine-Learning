
import pandas as pd

# Load the dataset into a pandas DataFrame
df = pd.read_csv('outlier.csv')

# Identify and handle outliers (e.g., using z-scores or IQR)
from scipy import stats

z_scores = stats.zscore(df['area'])
df = df[(z_scores < 3)]  # Remove rows with z-scores > 3 (or a different threshold)

print(df.iloc[0])
