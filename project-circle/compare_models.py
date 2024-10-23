import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('circles.txt')

# Prepare the features (X) and target variable (y)
X = data[['x', 'y']]
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create polynomial features for polynomial regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Multiple Linear Regression': LinearRegression(),
    'Polynomial Regression': LinearRegression(),
    'Decision Tree Regression': DecisionTreeRegressor(random_state=42),
    'SVM Regression': SVR(kernel='rbf')
}

# Train and evaluate models
results = []

for name, model in models.items():
    if name == 'Polynomial Regression':
        model.fit(X_train_poly, y_train)
        y_pred = model.predict(X_test_poly)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    accuracy = accuracy_score(y_test, np.round(y_pred))
    
    results.append({
        'Model': name,
        'R²': r2,
        'MSE': mse,
        'RMSE': rmse,
        'Accuracy': accuracy
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plot comparison
metrics = ['R²', 'MSE', 'RMSE', 'Accuracy']
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Model Comparison', fontsize=16)

for i, metric in enumerate(metrics):
    ax = axes[i // 2, i % 2]
    ax.bar(results_df['Model'], results_df[metric])
    ax.set_title(metric)
    ax.set_xticklabels(results_df['Model'], rotation=45, ha='right')

plt.tight_layout()
plt.show()

# Print results table
print(results_df.to_string(index=False))
