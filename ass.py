import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc

# Load data
df = pd.read_csv('train.csv')

# --- Data Cleaning ---
# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Fare'] = df['Fare'].fillna(df['Fare'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Convert categorical to numerical for clustering and ROC
df_clean = df.copy()
df_clean['Sex'] = df_clean['Sex'].map({'male': 0, 'female': 1})
df_clean['Embarked'] = df_clean['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# --- Apriori and FP-Growth Preparation ---
# Bin Age and Fare for association rules
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 12, 18, 60, 100], labels=['Child', 'Teen', 'Adult', 'Senior'])
df['Fare_Group'] = pd.cut(df['Fare'], bins=[-1, 10, 50, 600], labels=['Low_Fare', 'Mid_Fare', 'High_Fare'])
df['Sex'] = df['Sex'].str.capitalize()
df['Pclass'] = df['Pclass'].apply(lambda x: f'Class_{x}')
df['Survived'] = df['Survived'].replace({0: 'Did_Not_Survive', 1: 'Survived'})

df_rules = df[['Sex', 'Pclass', 'Age_Group', 'Fare_Group', 'Survived']].dropna()
transactions = df_rules.values.tolist()
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_onehot = pd.DataFrame(te_ary, columns=te.columns_)

# --- Apriori Association Rules ---
frequent_itemsets_apriori = apriori(df_onehot, min_support=0.02, use_colnames=True)
rules_apriori = association_rules(frequent_itemsets_apriori, metric="lift", min_threshold=1)
interesting_rules = rules_apriori[(rules_apriori['lift'] > 1.5) & (rules_apriori['confidence'] > 0.7)]
print("--- Top Hidden Rules Found with Apriori ---")
print(interesting_rules.sort_values(by=['confidence', 'lift'], ascending=False))

# --- FP-Growth Association Rules ---
frequent_itemsets_fp = fpgrowth(df_onehot, min_support=0.02, use_colnames=True)
rules_fp = association_rules(frequent_itemsets_fp, metric="lift", min_threshold=1.5)
print("\n--- Top Hidden Rules Found with FP-Growth ---")
print(rules_fp.sort_values(by='confidence', ascending=False).head())

# --- Outlier Detection ---
print(f"\nOriginal dataset loaded with {df.shape[0]} passengers.")
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
upper_bound = max(Q3 + 1.5 * IQR, 0)
print(f"\nIdentifying fares greater than ${upper_bound:.2f} as outliers.")

outliers_df = df[df['Fare'] > upper_bound]
no_outliers_df = df[df['Fare'] <= upper_bound]

# Visualize outliers
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
sns.boxplot(y=df['Fare'])
plt.title('Before Outlier Removal')
plt.ylabel('Fare')
plt.ylim(top=df['Fare'].max() + 20)

plt.subplot(1, 2, 2)
sns.boxplot(y=no_outliers_df['Fare'])
plt.title('After Outlier Removal')
plt.ylabel('Fare')
plt.ylim(top=df['Fare'].max() + 20)

plt.suptitle('Comparison of Fare Distribution', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# Save outlier files
no_outliers_df.to_csv('titanic_no_fare_outliers.csv', index=False)
print(f"Saved {no_outliers_df.shape[0]} passengers to 'titanic_no_fare_outliers.csv'")
outliers_df.to_csv('titanic_fare_outliers.csv', index=False)
print(f"Moved {outliers_df.shape[0]} outlier passengers to 'titanic_fare_outliers.csv'")

# --- ROC Curve Comparison ---
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'
X = df_clean[features]
y = df_clean[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train models
lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
print("\nAll models trained successfully.")

# Get prediction probabilities
lr_probs = lr_model.predict_proba(X_test)[:, 1]
tree_probs = tree_model.predict_proba(X_test)[:, 1]
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# Calculate ROC curve data
lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_probs)
lr_auc = auc(lr_fpr, lr_tpr)
tree_fpr, tree_tpr, _ = roc_curve(y_test, tree_probs)
tree_auc = auc(tree_fpr, tree_tpr)
rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
rf_auc = auc(rf_fpr, rf_tpr)

# Plot ROC curves
plt.figure(figsize=(10, 8))
plt.plot(lr_fpr, lr_tpr, label=f'Logistic Regression (AUC = {lr_auc:.2f})')
plt.plot(tree_fpr, tree_tpr, label=f'Decision Tree (AUC = {tree_auc:.2f})')
plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison for Titanic Models')
plt.legend()
plt.grid()
plt.show()
