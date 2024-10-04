import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
data = pd.read_csv('./circle-dataset/circles.txt')

# Print the first few rows and data info to understand the structure
print(data.head())
print(data.info())

# Select 'x' and 'y' columns
X = data['x'].values.reshape(-1, 1)  # Reshape to 2D array
y = data['y'].values

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate the coefficients and intercept of the linear regression equation
slope = model.coef_[0]
intercept = model.intercept_

# Calculate the mean squared error and R-squared score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the results
print(f"Linear Regression Equation: Y = {slope:.2f}X + {intercept:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared Score: {r2:.2f}")

# Plot the regression line
plt.scatter(X_test, y_test, label='Test Data', color='blue')
plt.plot(X_test, y_pred, label='Regression Line', color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('X vs Y')
plt.show()
