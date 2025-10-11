import pandas as pd
import pyfpgrowth

# Load the dataset
data = pd.read_csv("products.csv")

# Preprocess the Products column into a list of transactions
transactions = data['Products'].str.split(',').apply(lambda x: [item.strip() for item in x]).tolist()

# Remove any empty strings from transactions
for i in range(len(transactions)):
    transactions[i] = [x for x in transactions[i] if x]

# Print first 10 transactions
print("Some records:", transactions[:10])

# Find frequent patterns with minimum support count of 2
patterns = pyfpgrowth.find_frequent_patterns(transactions, 2)
print("Frequent Patterns:", patterns)

# Generate association rules with minimum confidence of 0.7
rules = pyfpgrowth.generate_association_rules(patterns, 0.7)
print("Association Rules:", rules)
