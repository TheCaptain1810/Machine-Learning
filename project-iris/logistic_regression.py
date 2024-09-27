#Data Pre-procesing Step  
# importing libraries  
import numpy as nm  
import matplotlib.pyplot as mtp  
import pandas as pd  
      
#importing datasets  
data_set= pd.read_csv('./iris dataset/iris.data')  
data_set

#Extracting Independent and dependent Variable  
x= data_set.iloc[:, [2,3]].values  
y= data_set.iloc[:, 4].values  # This should contain string labels like 'setosa', 'versicolor', 'virginica'
x
y

# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.25, random_state=0)

#feature Scaling  
from sklearn.preprocessing import StandardScaler    
st_x= StandardScaler()    
x_train= st_x.fit_transform(x_train)    
x_test= st_x.transform(x_test)  

#Fitting Logistic Regression to the training set  
from sklearn.linear_model import LogisticRegression  
classifier= LogisticRegression(random_state=0)  
classifier.fit(x_train, y_train)  

#Predicting the test set result  
y_pred= classifier.predict(x_test)  

#Creating the Confusion matrix  
from sklearn.metrics import confusion_matrix, classification_report
cm = confusion_matrix(y_test, y_pred)  
print("Confusion Matrix:")
print(cm)

#plot confusion matrix
from sklearn import metrics 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
import matplotlib.pyplot as plt 
cm_display.plot()
plt.show() 

# Calculate and print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Calculate individual metrics with appropriate averaging
precision = metrics.precision_score(y_test, y_pred, average='weighted')
recall = metrics.recall_score(y_test, y_pred, average='weighted')
f1 = metrics.f1_score(y_test, y_pred, average='weighted')
accuracy = metrics.accuracy_score(y_test, y_pred)

# Print all measuring parameters
print("\nPerformance Metrics:")
print({
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1
})

# ROC curve and AUC score are typically used for binary classification
# For multiclass, we can use ROC AUC OVR (One-vs-Rest)
y_prob = classifier.predict_proba(x_test)
auc = metrics.roc_auc_score(y_test, y_prob, multi_class='ovr', average='weighted')

print(f"\nAUC Score (weighted average): {auc}")

# We can't easily plot ROC curve for multiclass, so we'll skip that part
