import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("noisy_data_example.csv")

# Convert to numeric
df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
df['Salary'] = pd.to_numeric(df['Salary'], errors='coerce')

# Detect invalid entries
invalid_age = df[df['Age'].isna()]
invalid_salary = df[df['Salary'].isna()]
invalid_dates = df[pd.to_datetime(df['Join_Date'], errors='coerce').isna()]

print("Invalid Age:\n", invalid_age)
print("\nInvalid Salary:\n", invalid_salary)
print("\nInvalid Join Dates:\n", invalid_dates[['ID', 'Join_Date']])

# Detect outliers
def outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return s[(s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)]

print("\nAge Outliers:\n", outliers(df['Age']))
print("\nSalary Outliers:\n", outliers(df['Salary']))

# Plot boxplots
df[['Age', 'Salary']].plot(kind='box', subplots=True, layout=(1, 2), figsize=(10, 4))
plt.show()

# Remove outliers and invalid data
def filter_outliers(s):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return s.between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

df = df[filter_outliers(df['Age']) & filter_outliers(df['Salary'])].dropna(subset=['Age', 'Salary'])
df = df[pd.to_datetime(df['Join_Date'], errors='coerce').notna()]

# Save cleaned data
df.to_csv("cleaned_noisy_data.csv", index=False)
print("\nCleaned data saved to cleaned_noisy_data.csv")
