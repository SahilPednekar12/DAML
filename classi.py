import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error

# Load and preprocess data
data = pd.read_csv("train.csv")
data = data.dropna(subset=['Age', 'Embarked'])

# Prepare features for models
X_full = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
X_full = pd.get_dummies(X_full, drop_first=True)
X_lr = data[['Age', 'Fare', 'Pclass']]  # Features for Linear Regression
y = data['Survived']

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_full)
X_lr_scaled = scaler.fit_transform(X_lr)

# Apply PCA
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X_scaled)

# Split data
X_train_full, X_test_full, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
X_train_pca, X_test_pca, _, _ = train_test_split(X_pca, y, test_size=0.2, random_state=42)
X_train_lr, X_test_lr, _, _ = train_test_split(X_lr_scaled, y, test_size=0.2, random_state=42)

# Decision Tree Classifier
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_clf.fit(X_train_full, y_train)
dt_pred = dt_clf.predict(X_test_full)
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))
print("Decision Tree Confusion Matrix:\n", confusion_matrix(y_test, dt_pred))

# Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train_full, y_train)
nb_pred = nb_model.predict(X_test_full)
print("\n✅ Naive Bayes Accuracy:", accuracy_score(y_test, nb_pred))
print("Naive Bayes Confusion Matrix:\n", confusion_matrix(y_test, nb_pred))

# PCA Analysis
print("\nPCA Shape:", X_pca.shape)
print("Explained Variance Ratio:", pca.explained_variance_ratio_)
print("Total Variance Captured:", round(sum(pca.explained_variance_ratio_), 2))

# Linear Regression
lr = LinearRegression()
lr.fit(X_train_lr, y_train)
lr_pred = lr.predict(X_test_lr)
print("\nLinear Regression MSE:", mean_squared_error(y_test, lr_pred))
print("Sample Linear Regression Predictions:\n", lr_pred[:10])
