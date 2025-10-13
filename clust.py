import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

# Load data
df = pd.read_csv("train.csv")

# --- Data Cleaning ---
# Handle missing values in 'Age' by filling with median
df['Age'] = df['Age'].fillna(df['Age'].median())

# Handle missing values in 'Fare' (if any) by filling with median
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# Select features for clustering
features = df[['Age', 'Fare']].dropna()

# --- K-Means Clustering (k=3) ---
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['KMeans_Cluster'] = kmeans.fit_predict(features)

# Plot K-Means clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Age', y='Fare', hue='KMeans_Cluster', palette='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', s=200, marker='X', label='Centroids')
plt.title('K-Means Clustering (k=3)')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend()
plt.show()

# --- Hierarchical Clustering (k=3) ---
linked = linkage(features, method='ward')
plt.figure(figsize=(8, 6))
dendrogram(linked)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Passenger Index')
plt.ylabel('Distance')
plt.show()

# Apply Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
df['Hierarchical_Cluster'] = hierarchical.fit_predict(features)

# Plot Hierarchical clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Age', y='Fare', hue='Hierarchical_Cluster', palette='plasma')
plt.title('Hierarchical Clustering (k=3)')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.legend()
plt.show()
