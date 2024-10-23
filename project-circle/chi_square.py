import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

# Sample data: A contingency table
data = {
    'Category A': [30, 10, 20],
    'Category B': [20, 25, 15]
}

# Create a DataFrame
df = pd.DataFrame(data, index=['Group 1', 'Group 2', 'Group 3'])

# Display the contingency table
print("Contingency Table:")
print(df)

# Perform Chi-Square test
chi2_stat, p_value, dof, expected = chi2_contingency(df)

# Output results
print(f"\nChi-Square Statistic: {chi2_stat}")
print(f"P-value: {p_value}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(expected)

# Decision
alpha = 0.05
if p_value < alpha:
    print("\nReject the null hypothesis: There is a significant association between the categories.")
else:
    print("\nFail to reject the null hypothesis: No significant association between the categories.")

