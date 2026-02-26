# %%
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import plotly.express as px

# %% [markdown]
## Loading the datasets 

# %%
salary_data = pd.read_csv('2025_salaries.csv', header = 1, encoding = 'latin-1')
salary_data.head()

# %%
stats = pd.read_csv('nba_2025.txt', sep = ",", encoding = 'latin-1')
stats.head()   

# %%
# Merging the two data sets by player
merged_data = pd.merge(salary_data, stats, on = 'Player', how = 'inner')
merged_data.head()

# %%
# Duplicates in the 'Player' column
duplicates = merged_data[merged_data.duplicated(subset = 'Player', keep = False)]
duplicates.head()

# %%
# Check column names
merged_data.columns

# %%
# Sort players by games played (G) in descending order
merged_data = merged_data.sort_values(by = 'G', ascending = False)
merged_data.head()

# %%
merged_data = merged_data.drop_duplicates(subset = 'Player', keep = 'first')
len(merged_data)

# %%
# Drop unneccessary variables 
columns_to_keep = ['Player', '2025-26', 'Rk', 'Age', 'G', '3P%', '2P%', 'eFG%', 'FT%', 'TRB', 'PTS', 'Trp-Dbl']
cleaned_data = merged_data[columns_to_keep]
cleaned_data.head()

# %%
# Drop missing values 
cleaned_data = cleaned_data.dropna()

# %%
# Remove the $ sign and comma from 2025-26 salary column
cleaned_data['2025-26'] = cleaned_data['2025-26'].astype(str).str.replace("[$,]", '', regex = True).astype(float)
cleaned_data.head()

# %%
# Select all the data with type = floats (all columns except for Player)
x = cleaned_data.select_dtypes(include = 'float64')

# Standardizing 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

# Create a new dataframe with the standardized values 
df_standardized = pd.DataFrame(X_scaled, columns = x.columns, index = x.index)
df_standardized['Player'] = cleaned_data['Player']
df_standardized.head()

# %%
kmeans = KMeans(n_clusters = 5, random_state = 42, verbose = 1)
kmeans.fit(df_standardized[['Rk', 'Age', 'G', '3P%', '2P%', 'eFG%', 'FT%', 'TRB', 'PTS', 'Trp-Dbl']])

# %%
print(kmeans.cluster_centers_)
print(kmeans.inertia_)

# %%
df_standardized['Cluster'] = kmeans.labels_
df_standardized[['Player', 'Cluster']].head()

# %%
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    df_standardized['eFG%'],
    df_standardized['PTS'],              
    c=df_standardized['Cluster'], 
    cmap='viridis',                    
)
plt.xlabel("eFG%")
plt.ylabel("Points (PTS)")
plt.title("NBA Players: Performance vs Salary")
cbar = plt.colorbar(scatter)
cbar.set_label("Salary (2025-26)")
plt.show()

# %%
fig = px.scatter_3d(
    df_standardized,
    x = 'eFG%',
    y = 'PTS',
    z = 'G',
    color = '2025-26',
    symbol = kmeans.labels_,
    hover_data = ['Player'],
    title = 'NBA Player Clusters'
)
fig.show(renderer = 'browser')

# %%
# Evaluate the quality of the clustering using total variance explained 
features = ['Rk', 'Age', 'G', '3P%', '2P%', 'eFG%', 'FT%', 'TRB', 'PTS', 'Trp-Dbl']
total = (df_standardized[features] - df_standardized[features].mean()) ** 2
TSS = np.sum(total)
print(f"Total Sum of Squares (TSS): {TSS}")

between_SSE = (TSS - kmeans.inertia_)
print(f"Between-Cluster Sum of Squares (BSS): {between_SSE}")

variance_explained = between_SSE / TSS
print(f"Variance Explained: {variance_explained:.4f} or {variance_explained*100:.2f}%")

# %%
from sklearn.metrics import silhouette_score
ss = silhouette_score(df_standardized[features], kmeans.labels_)
print('Silhouette Score:', ss)

# %%
# Calculate within-cluster sum of squares (WCSS) for different values of k
# WCSS measures total distance of points from their cluster centers
# We test k from 1 to 10 to see which number of clusters works best
wcss = []
for i in range(1, 11):
    kmeans_elbow = KMeans(n_clusters=i, random_state=1).fit(df_standardized[features])
    wcss.append(kmeans_elbow.inertia_)

# %%
# Plot the Elbow Curve
# Look for the "elbow" - where adding more clusters doesn't help much
# The elbow indicates optimal k (balance between fit and complexity)
# After the elbow, WCSS decreases slowly, suggesting diminishing returns
elbow_data = pd.DataFrame({"k": range(1, 11), "wcss": wcss})
fig = px.line(elbow_data, x="k", y="wcss", title="Elbow Method")
fig.show()

#%%
import plotly.graph_objects as go

# Calculate silhouette score for k = 2 through 10
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
df_copied = df_standardized.copy()
kmeans_retrain = KMeans(n_clusters=optimal_k, random_state=42)
df_copied['Cluster'] = kmeans_retrain.fit_predict(df_copied[features])

# Check cluster counts
print(df_copied['Cluster'].value_counts())

# %%
# 2D scatter
plt.figure(figsize=(8,6))
plt.scatter(df_copied['eFG%'], df_copied['PTS'], c=df_copied['Cluster'], cmap='viridis')
plt.xlabel("eFG%")
plt.ylabel("Points (PTS)")
plt.title(f"NBA Player Clusters (k={optimal_k})")
plt.colorbar(label="Cluster")
plt.show()

# %%
# 3D scatter
fig = px.scatter_3d(
    df_copied,
    x = 'eFG%',
    y = 'PTS',
    z = 'G',
    color = 'Cluster',
    hover_data = ['Player'],
    title = 'NBA Player Clusters'
)
fig.show(renderer = 'browser')

# %%
total_retrain = (df_copied[features] - df_copied[features].mean()) ** 2
TSS_retrain = np.sum(total)
print(f"Total Sum of Squares (TSS): {TSS_retrain}")

bss_retrain = (TSS_retrain - kmeans_retrain.inertia_)
print(f"Between-Cluster Sum of Squares (BSS): {bss_retrain}")

variance_explained_retrain = bss_retrain / TSS_retrain
print(f"Variance Explained: {variance_explained_retrain:.4f} or {variance_explained_retrain*100:.2f}%")

# %%
from sklearn.metrics import silhouette_score
ss_retrain = silhouette_score(df_copied[features], kmeans_retrain.labels_)
print('Silhouette Score:', ss_retrain)

# %%
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
# Pick members
# Examples that are not good choices: 
# %%
