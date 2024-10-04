import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Load the dataset
data = pd.read_csv('circle-dataset/circles.txt')

# Split features (X) and target (y)
X = data[['x', 'y']]
y = data['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(clf, feature_names=['x', 'y'], class_names=clf.classes_.astype(str), filled=True, rounded=True)
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.close()

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': clf.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Scatter plot of the data points with decision boundaries
def plot_decision_boundaries(clf, X, y):
    h = 0.02  # step size in the mesh
    x_min, x_max = X['x'].min() - 1, X['x'].max() + 1
    y_min, y_max = X['y'].min() - 1, X['y'].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X['x'], X['y'], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Decision Tree Classification')
    plt.colorbar(scatter)
    plt.savefig('decision_boundaries.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_decision_boundaries(clf, X, y)
