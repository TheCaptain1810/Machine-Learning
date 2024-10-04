# Step 1: Import Necessary Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Load the Data from CSV
print("Step 2: Loading the dataset...")
column_names = ['x', 'y', 'class']
df = pd.read_csv('circle-dataset/circles.txt', header=0, names=column_names)

# Display the first few rows of the dataframe
print("First few rows of the dataset:")
print(df.head())

# Assuming the last column is the target and the rest are features
X = df.iloc[:, :-1].values  # Features
y = df.iloc[:, -1].values    # Target

# Step 3: Preprocess the Data
print("\nStep 3: Preprocessing the data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split the Data into Training and Testing Sets
print("\nStep 4: Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 5: Evaluate KNN Model for K = 3, 4, 5
for k in [3, 4, 5]:
    print(f"\nStep 5: Training the KNN model with K={k}...")
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    print("Model training complete.")
    
    # Step 6: Make Predictions
    print("\nStep 6: Making predictions on the test set...")
    y_pred = knn.predict(X_test)
    
    # Step 7: Evaluate the Model
    print(f"\nResults for K={k}:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)

    class_report = classification_report(y_test, y_pred, output_dict=True)

    # Extract overall accuracy
    accuracy = class_report['accuracy']

    # Calculate average precision, recall, and F1-score across all classes
    classes = [c for c in class_report.keys() if c != 'accuracy']
    precision = np.mean([class_report[c]['precision'] for c in classes])
    recall = np.mean([class_report[c]['recall'] for c in classes])
    f1 = np.mean([class_report[c]['f1-score'] for c in classes])

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Average Precision: {precision:.2f}")
    print(f"Average Recall: {recall:.2f}")
    print(f"Average F1 Score: {f1:.2f}")