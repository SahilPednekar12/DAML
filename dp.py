import pandas as pd
from scipy.stats import chi2_contingency
import seaborn as sns
import matplotlib.pyplot as plt

# Chi-square Test
df_cat = pd.read_csv("categorical_chisquare_dataset.csv")
contingency = pd.crosstab(df_cat['Education_Level'], df_cat['Employment_Status'])
print("Contingency Table:\n", contingency)

chi2, p, dof, expected = chi2_contingency(contingency)
print(f"\nChi-square Statistic: {chi2:.2f}")
print(f"Degrees of Freedom: {dof}")
print(f"p-value: {p:.4f}")
print("\nExpected Frequencies:\n", pd.DataFrame(expected, index=contingency.index, columns=contingency.columns))

if p < 0.05:
    print("\nConclusion: Likely association between Education Level and Employment Status (reject H0).")
else:
    print("\nConclusion: No strong evidence of association (fail to reject H0).")

# Correlation Heatmap
df_corr = pd.read_csv("student_performance.csv")
correlation_matrix = df_corr.corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap of Correlation Matrix")
plt.tight_layout()
plt.show()
