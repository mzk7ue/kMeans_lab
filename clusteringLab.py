# %%
# Imports - Libraries needed for KMeans Clustering Lab
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# %% [markdown]
## Loading the datasets 

# %%
# Load salary dataset 

# header = 1: starts read from second row of the dataset (index 1) - contains column names
# encoding = 'latin-1': accounts for special characters
salary_data = pd.read_csv('2025_salaries.csv', header = 1, encoding = 'latin-1')

# View dataset
salary_data.head()

# %%
# Load in stats (of players) dataset 

# sep = ",": individual entries in dataset are separated by commas 
stats = pd.read_csv('nba_2025.txt', sep = ",", encoding = 'latin-1')

# View dataset
stats.head()   

# %%
# Merging the two datasets by the 'Player' column

# Using the inner join method, we only keep the players that exist in both datasets 
merged_data = pd.merge(salary_data, stats, on = 'Player', how = 'inner')
merged_data.head()

# %%
# Stores the duplicated rows from merged_data DataFrame based on the 'Player' column
duplicates = merged_data[merged_data.duplicated(subset = 'Player', keep = False)]
duplicates.head()

# %%
# Check column names
merged_data.columns

# %%
# Sort players based on the number of games played (G) in descending order
merged_data = merged_data.sort_values(by = 'G', ascending = False)
merged_data.head()

# %%
# Drop duplicated and keep the first occurrence of the player (which would be ones with the highest number of games played)
merged_data = merged_data.drop_duplicates(subset = 'Player', keep = 'first')

# %%
# Keep only the columns that are relevant to the model, and drop all others

# List of column names to keep
columns_to_keep = ['Player', '2025-26', 'Rk', 'Age', 'G', '3P%', '2P%', 'eFG%', 'FT%', 'TRB', 'PTS', 'Trp-Dbl']

# Creates a new DataFrame that only contains the contents of the merged_data columns specified in columns_to_keep
cleaned_data = merged_data[columns_to_keep]
cleaned_data.head()

# %%
# Drop missing values 
cleaned_data = cleaned_data.dropna()

# %%
# Remove the $ sign and comma from 2025-26 salary column, and convert from a string to float
cleaned_data['2025-26'] = cleaned_data['2025-26'].astype(str).str.replace("[$,]", '', regex = True).astype(float)
cleaned_data.head()

# %%
# Standardizing the columns: this needs to be done because the columns are on different scales 
# KMeans will use distances between points to perform clustering, so if the features are not standardized, 
# variables with larger values/scales will dominate and create bias towards those features

# Select all the data that are floats (all columns except for Player)
x = cleaned_data.select_dtypes(include = 'float64')

# Standardizing 
scaler = StandardScaler()

# .fit_transform(): determines the mean and standard deviation, and applies the scaling formula to each column (stored in X_scaled)
X_scaled = scaler.fit_transform(x)

# Create a new dataframe with the standardized values with the original column names and index values from x
df_standardized = pd.DataFrame(X_scaled, columns = x.columns, index = x.index)

# Add the 'Player' column to the new standardized DataFrame 
df_standardized['Player'] = cleaned_data['Player']
df_standardized.head()

# %%
# Performing KMeans clustering 

# Set up algorithm for clustering with 5 clusters
kmeans = KMeans(n_clusters = 5, random_state = 42, verbose = 1)

# Run the clustering algorithm on selected rows (the numeric columns - salary)
kmeans.fit(df_standardized[['Rk', 'Age', 'G', '3P%', '2P%', 'eFG%', 'FT%', 'TRB', 'PTS', 'Trp-Dbl']])

# %%
# Outputs of the KMeans clustering algorithm

# .cluster_centers_: the coordinates of each cluster's centroid
print(kmeans.cluster_centers_)

# .inertia_: the total squared distance of every point to its cluster centroid
print(kmeans.inertia_)

# %%
# Create a new column that contains the cluster assignment (.labels_) of each player
df_standardized['Cluster'] = kmeans.labels_
df_standardized[['Player', 'Cluster']].head()

# %%
# Visualization of the result with 2 variables (2D) 
# Points are colored by cluster assignments
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    df_standardized['eFG%'],
    df_standardized['PTS'],              
    c=df_standardized['Cluster'], 
    cmap='viridis',                    
)

plt.xlabel("eFG%")
plt.ylabel("Points (PTS)")
plt.show()

# %%
# Visualization of the result with 3 variables (3D)
# cluster assignments are represented by the symbols of the points, 
# and points are colored based on their salaries
fig = px.scatter_3d(
    df_standardized,
    x = 'eFG%',
    y = 'PTS',
    z = 'G',
    color = '2025-26',
    symbol = kmeans.labels_,
    hover_data = ['Player'],
    title = 'G vs. eFG% vs. PTS for Players'
)
fig.show(renderer = 'browser')

# %%
# Evaluate the quality of the clustering using total variance explained 

# The numeric columns that are used as features for the clustering model
features = ['Rk', 'Age', 'G', '3P%', '2P%', 'eFG%', 'FT%', 'TRB', 'PTS', 'Trp-Dbl']

# Calculate total sum of squares (TSS): total variance in the data 
# Adds up all squared deviations across all features and rows
total = (df_standardized[features] - df_standardized[features].mean()) ** 2
TSS = np.sum(total)
print(f"Total Sum of Squares (TSS): {TSS}")

# Calculate bbetween cluster sum of squares (BSS = TSS - WSS): measures the variance between clusters (how separated they are)
# WSS: inertia, measures variance within clusters (how tight they are)
between_SSE = (TSS - kmeans.inertia_)
print(f"Between-Cluster Sum of Squares (BSS): {between_SSE}")

# What % variance is explained by the model? 
# Higher is better because it means that the clusters are capturing meaningful patterns
variance_explained = between_SSE / TSS
print(f"Variance Explained: {variance_explained:.4f} or {variance_explained*100:.2f}%")

# %%
# Silhouette Score: for each point, it measures how similar it is to its own cluster vs. how similar it is to the nearest other cluster 
# 1: perfect clustering, the point is very close to its own cluster
# 0: the point is on the border between 2 clusters
# -1: the point is probably assigned to the wrong cluster

# Finds the silhouette score for each player and takes the average of it to produce one value for the entire clustering algorithm
ss = silhouette_score(df_standardized[features], kmeans.labels_)
print('Silhouette Score:', ss)

 
# %%
# Calculate within-cluster sum of squares (WCSS) for different values of k to see which number of clusters works best
# WCSS: measures total distance of points from their cluster centers

wcss = []
for i in range(1, 11):
    kmeans_elbow = KMeans(n_clusters=i, random_state=1).fit(df_standardized[features])
    # The within cluster sum of squares for each iteration is added to the wcss list
    wcss.append(kmeans_elbow.inertia_)

# %%
# Visualize the results of the Elbow Method/Curve
# Look for the "elbow" - where adding more clusters doesn't help much
# The elbow indicates optimal k (balance between fit and complexity)
# After the elbow, WCSS decreases slowly, suggesting diminishing returns

# Creates an elbow_data DataFrame with k values and its wcss 
elbow_data = pd.DataFrame({"k": range(1, 11), "wcss": wcss})
fig = px.line(elbow_data, x="k", y="wcss", title="Elbow Method")
fig.show()

#%%
# Calculate Silhouette coefficients for k = 2 through 10
# Note: Silhouette requires at least 2 clusters, so we start at k=2

silhouette_scores = []

for k in range(2, 11):
    kmeans_obj = KMeans(n_clusters=k, algorithm="lloyd", random_state=42)
    kmeans_obj.fit(df_standardized[features])
    
    # Calculate average silhouette score across all points
    silhouette_scores.append(
        silhouette_score(df_standardized[features], kmeans_obj.labels_))

# Find k with highest silhouette score (that's our optimal number)
best_nc = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimal number of clusters by Silhouette Score: {best_nc}")

# Sets the k value with the highest silhouette score to the optimal_k variable
optimal_k = best_nc

# %%
# Plot silhouette scores across different values of k
# Look for the highest point - that's the best number of clusters
# Unlike elbow method, this gives a clear maximum to choose

fig = go.Figure(data=go.Scatter(
    x=list(range(2, 11)),
    y=silhouette_scores,
    mode='lines+markers'))

fig.update_layout(
    title="Silhouette Score by Number of Clusters",
    xaxis_title="Number of Clusters (k)",
    yaxis_title="Silhouette Score")

fig.show(renderer = 'browser')

# %%
# Based on the elbow curve, we will retrain the model with 2 clusters

# Make a copy of the standardized cleaned dataset
df_copied = df_standardized.copy()

# Rerun the algorithm with the optimal k value (2)
kmeans_retrain = KMeans(n_clusters=optimal_k, random_state=42).fit(df_copied[features])
df_copied['Cluster'] = kmeans_retrain.labels_

df_copied.head()

# Check cluster counts
print(df_copied['Cluster'].value_counts())

# %%
# 2D Visualization 
plt.figure(figsize=(8,6))
plt.scatter(df_copied['eFG%'], df_copied['PTS'], c=df_copied['Cluster'], cmap='viridis')
plt.xlabel("eFG%")
plt.ylabel("Points (PTS)")
plt.show()

# %%
# Total variance explained 

total_retrain = (df_copied[features] - df_copied[features].mean()) ** 2
TSS_retrain = np.sum(total)
print(f"Total Sum of Squares (TSS): {TSS_retrain}")

bss_retrain = (TSS_retrain - kmeans_retrain.inertia_)
print(f"Between-Cluster Sum of Squares (BSS): {bss_retrain}")

variance_explained_retrain = bss_retrain / TSS_retrain
print(f"Variance Explained: {variance_explained_retrain:.4f} or {variance_explained_retrain*100:.2f}%")

# %%
# Silhouette Score
ss_retrain = silhouette_score(df_copied[features], kmeans_retrain.labels_)
print('Silhouette Score:', ss_retrain)

# %%
# 3D Visualization of the model to select players for Mr. Rooney to consider from
fig = px.scatter_3d(
    df_copied,
    x = 'eFG%',
    y = 'PTS',
    z = 'G',
    color = '2025-26',
    symbol = kmeans_retrain.labels_,
    hover_data = ['Player'],
    title = 'NBA Player Clusters'
)
fig.show(renderer = 'browser')

# %%
