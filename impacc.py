import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import accuracy_score

# -------------------------------
# Load and prepare Titanic dataset
# -------------------------------
data = pd.read_csv("train.csv")

# Drop unnecessary columns
drop_columns = ['PassengerId', 'Name', 'Ticket', 'Cabin']
data = data.drop(columns=drop_columns)

# Handle missing values
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Convert categorical columns into numerical (dummy variables)
X = pd.get_dummies(data.drop(columns=['Survived']), drop_first=True)
y = data['Survived']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# -------------------------------
# 1️⃣ Cross-Validation (Decision Tree)
# -------------------------------
cv_model = DecisionTreeClassifier(max_depth=5, random_state=42)
cv_pred = cross_val_predict(cv_model, X_scaled, y, cv=5)
cv_acc = accuracy_score(y, cv_pred)
print("Cross-Validation Accuracy:", round(cv_acc, 4))

# -------------------------------
# 2️⃣ Bootstrap (Decision Tree)
# -------------------------------
X_boot, y_boot = resample(X_train, y_train, replace=True, random_state=42)
boot_model = DecisionTreeClassifier(max_depth=5, random_state=42)
boot_model.fit(X_boot, y_boot)
y_pred_boot = boot_model.predict(X_test)
boot_acc = accuracy_score(y_test, y_pred_boot)
print("Bootstrap Accuracy:", round(boot_acc, 4))


# -------------------------------
# 4️⃣ Boosting (AdaBoost)
# -------------------------------
boost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)
boost_model.fit(X_train, y_train)
y_pred_boost = boost_model.predict(X_test)
boost_acc = accuracy_score(y_test, y_pred_boost)
print("\nBoosting Accuracy:", round(boost_acc, 4))
