import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from mpl_toolkits.mplot3d import Axes3D

# Assuming the CSV is available as 'train.csv'; if not, replace with StringIO for pasted data
df = pd.read_csv('train.csv')

# Data Cleaning
# Check for missing values and data info
print("Data Info Before Cleaning:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing values
# Fill Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin due to too many missing values
df.drop('Cabin', axis=1, inplace=True)

# Drop irrelevant columns for EDA (PassengerId, Name, Ticket)
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# Convert categorical to numeric if needed (e.g., Sex, Embarked)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

# Check after cleaning
print("\nData Info After Cleaning:")
print(df.info())
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# 1. Mean, Median, Mode - Comment on symmetry
numerical_cols = ['Age', 'Fare', 'SibSp', 'Parch']
for col in numerical_cols:
    mean_val = df[col].mean()
    median_val = df[col].median()
    mode_val = df[col].mode()[0] if not df[col].mode().empty else np.nan
    print(f"\n{col}:")
    print(f"Mean: {mean_val:.2f}")
    print(f"Median: {median_val:.2f}")
    print(f"Mode: {mode_val}")
    
    # Comment on symmetry/skewness
    if abs(mean_val - median_val) < 0.1 * mean_val:
        print("The data appears approximately symmetric (mean ≈ median).")
    elif mean_val > median_val:
        print("The data is right-skewed (mean > median).")
    else:
        print("The data is left-skewed (mean < median).")

# Set up plotting style
sns.set(style="whitegrid")

# 2. Bar plot (Survived count) and Stacked bar plot (Survived by Pclass)
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
df['Survived'].value_counts().plot(kind='bar', color=['skyblue', 'salmon'])
plt.title('Bar Plot: Count of Survived')
plt.xlabel('Survived')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
pd.crosstab(df['Pclass'], df['Survived']).plot(kind='bar', stacked=True, color=['skyblue', 'salmon'], ax=plt.gca())
plt.title('Stacked Bar Plot: Survived by Pclass')
plt.xlabel('Pclass')
plt.ylabel('Count')

plt.tight_layout()
plt.show()

# 3. Pie chart (Distribution of Sex)
plt.figure(figsize=(6, 6))
df['Sex'].value_counts().plot(kind='pie', autopct='%1.1f%%', labels=['Male (0)', 'Female (1)'], colors=['lightblue', 'lightpink'])
plt.title('Pie Chart: Distribution of Sex')
plt.ylabel('')
plt.show()

# 4. Histogram (Age distribution)
plt.figure(figsize=(8, 5))
df['Age'].hist(bins=20, color='purple', edgecolor='black')
plt.title('Histogram: Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 5. Scatter plot (Age vs Fare)
plt.figure(figsize=(8, 5))
plt.scatter(df['Age'], df['Fare'], alpha=0.5, color='green')
plt.title('Scatter Plot: Age vs Fare')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

# 6. Box plot (Fare by Pclass)
plt.figure(figsize=(8, 5))
sns.boxplot(x='Pclass', y='Fare', data=df, palette='Set3')
plt.title('Box Plot: Fare by Pclass')
plt.show()

# 7. Line chart (Cumulative Fare by sorted Age)
plt.figure(figsize=(8, 5))
df_sorted = df.sort_values('Age')
plt.plot(df_sorted['Age'], df_sorted['Fare'].cumsum(), color='orange')
plt.title('Line Chart: Cumulative Fare by Age (Sorted)')
plt.xlabel('Age')
plt.ylabel('Cumulative Fare')
plt.show()

# 8. Violin plot (Age by Survived)
plt.figure(figsize=(8, 5))
sns.violinplot(x='Survived', y='Age', data=df, palette='muted')
plt.title('Violin Plot: Age by Survived')
plt.show()

# 9. Swarm plot (Age by Sex)
plt.figure(figsize=(8, 5))
sns.swarmplot(x='Sex', y='Age', data=df, palette='deep')
plt.title('Swarm Plot: Age by Sex')
plt.xlabel('Sex (0: Male, 1: Female)')
plt.show()

# 10. Donut chart (Distribution of Pclass)
plt.figure(figsize=(6, 6))
values = df['Pclass'].value_counts()
plt.pie(values, labels=values.index, autopct='%1.1f%%', colors=['gold', 'lightgreen', 'lightcoral'])
# Add white circle for donut
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
plt.title('Donut Chart: Distribution of Pclass')
plt.show()

# 11. 3D Scatter chart (Age, Fare, SibSp)
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Age'], df['Fare'], df['SibSp'], c=df['Survived'], cmap='viridis', alpha=0.6)
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
ax.set_zlabel('SibSp')
ax.set_title('3D Scatter Plot: Age, Fare, SibSp (Colored by Survived)')
plt.show()
