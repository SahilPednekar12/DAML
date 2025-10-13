import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# -------------------------------
# Load dataset
# -------------------------------
data = pd.read_csv("train.csv")

# -------------------------------
# Preprocessing
# -------------------------------
# Drop columns that are not useful for prediction
drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
data = data.drop(columns=drop_columns)

# Handle missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Convert categorical columns to dummy variables
X = pd.get_dummies(data.drop(columns=['Survived']), drop_first=True)
y = data['Survived']

# -------------------------------
# Split data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Train Random Forest model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=10, 
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------
# Predict and Evaluate
# -------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
spec = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # Specificity
se = np.sqrt((acc * (1 - acc)) / len(y_test))  # Standard Error
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# -------------------------------
# Print Results
# -------------------------------
print("Random Forest Evaluation on Titanic Dataset:")
print("---------------------------------------------")
print("Accuracy:", round(acc, 3))
print("Precision:", round(prec, 3))
print("Recall:", round(rec, 3))
print("Specificity:", round(spec, 3))
print("Standard Error:", round(se, 4))
print("AUC:", round(roc_auc, 3))

# -------------------------------
# Plot ROC Curve
# -------------------------------
plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})', color='blue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest (Titanic Dataset)')
plt.legend()
plt.grid(True)
plt.show()
