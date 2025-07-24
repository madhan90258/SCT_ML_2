# Import necessary libraries
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Step 2: Select features for clustering
features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Step 3: Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 4: Use Elbow Method to find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    wcss.append(kmeans.inertia_)

# Step 5: Plot Elbow Curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.grid(True)
plt.show()

# Step 6: Apply KMeans with k=5
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Step 7: Save clustered data to CSV
df.to_csv("Mall_Customers_Clustered.csv", index=False)
print("✅ Clustered data saved to 'Mall_Customers_Clustered.csv'")

# ---------------------------------------
# (1) Open the Mall_Customers_Clustered.csv file
# ---------------------------------------
df_clustered = pd.read_csv("Mall_Customers_Clustered.csv")
print("\n📄 Preview of Clustered Data:")
print(df_clustered.head())

# ---------------------------------------
# (2) 3D Cluster Visualization
# ---------------------------------------
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['Age'], df['Annual Income (k$)'], df['Spending Score (1-100)'],
                     c=df['Cluster'], cmap='viridis', s=60)
ax.set_title("3D Cluster Plot: Age vs Income vs Spending")
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score (1-100)")
plt.colorbar(scatter, label='Cluster')
plt.show()
