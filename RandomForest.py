import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Load the dataset
file_path = "Mobile Phone Pricing.csv"  # Update path if needed
data = pd.read_csv(file_path)

# Define features and target variable
X = data.drop(columns=['price_range'])  # Features
y = data['price_range']  # Target

# Split the dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Classifier
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

# Make predictions
y_pred = rf_clf.predict(X_test)
y_pred_prob = rf_clf.predict_proba(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# For multi-class, calculate precision, recall, and specificity (macro-averaged)
precision = precision_score(y_test, y_pred, average='macro')
sensitivity = recall_score(y_test, y_pred, average='macro')

# Specificity calculation
specificity = []
for i in range(conf_matrix.shape[0]):
    TN = conf_matrix.sum() - (conf_matrix[i, :].sum() + conf_matrix[:, i].sum() - conf_matrix[i, i])
    FP = conf_matrix[:, i].sum() - conf_matrix[i, i]
    specificity.append(TN / (TN + FP) if (TN + FP) > 0 else 0)
specificity = np.mean(specificity)

# Standard error
standard_error = np.sqrt((accuracy * (1 - accuracy)) / len(y_test))

# ROC curve and AUC for multi-class
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
n_classes = y_test_bin.shape[1]
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure()
colors = ['darkorange', 'blue', 'green', 'red']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve class {i} (area = {roc_auc[i]:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest (Multi-Class)')
plt.legend(loc="lower right")
plt.show()

# Print results
print("Predictions:", y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Standard Error:", standard_error)
print("Precision (Macro):", precision)
print("Sensitivity (Recall, Macro):", sensitivity)
print("Specificity (Macro):", specificity)
print("AUC per class:", [f"{roc_auc[i]:0.2f}" for i in range(n_classes)])
