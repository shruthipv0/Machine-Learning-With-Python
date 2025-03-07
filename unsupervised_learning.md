
# K-means Clustering

![image](https://github.com/user-attachments/assets/2bd1896b-22e9-4120-8881-435de17a868e)

Groups unlabelled data into groups or clusters.

## Working principle

1. Choosing the number of clusters 
The first step is to define the K number of clusters in which we will group the data. let K=3.

2. Initializing centroids
Centroid is the center of a cluster but initially, the exact center of data points will be unknown so, select random data points and define them as centroids for each cluster. Lets initialize 3 centroids.

![image](https://github.com/user-attachments/assets/7cb8f57b-01fa-4979-b78a-7f2230cf4d5c)

3. Assign data points to the nearest cluster 
Assign data points X_n to their closest cluster centroid C_k

![image](https://github.com/user-attachments/assets/bc66900d-eff9-4968-a3f0-0d07e9e72c97)

calculate the distance between data point X and centroid C using Euclidean Distance metric.

![image](https://github.com/user-attachments/assets/19ee332c-2995-4ebc-ba48-8877a1f1ecbd)

Assign each point to the centroid such that the distance between the point and the centroid is less. 

![image](https://github.com/user-attachments/assets/66ad05b3-00ef-49ad-b544-d7a1283a7e30)

4. Re-initialise centroids
re-initialise the centroids by calculating the average of all points that cluster.

![image](https://github.com/user-attachments/assets/885d87a5-5e3e-41b5-ba9b-fdd5146f0927)

5. Repeat steps 3 and 4
keep repeating steps 3 and 4 until we have optimal centroids and the assignments of data points to correct clusters are not changing anymore.

![image](https://github.com/user-attachments/assets/2ee60285-9883-415c-ac2a-3722ae171294)

## K-Means on a synthetic data set

Creating synthetic data set using random.seed, make_blobs.

- random.seed : initialising random seed ensures the output of the random number generated will always be the same.
- make_blobs : creates synthetic __clustered data__ (used for testing k means usually).

```python
np.random.seed(0)

#centers = no of distinct clusters
X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std=0.9)

#visualising the clusers
plt.scatter(X[:, 0], X[:, 1], marker='.',alpha=0.3,ec='k',s=80)
```
![image](https://github.com/user-attachments/assets/00122f77-df39-40d5-91d4-13dba3c1e07c)

## Setting up k-means

__init__ = "random" picks random points for centroids

__init__ = "k-means++" is more better initialisation method

__n_init__ = Number of times the k-means algorithm will be run with different centroid seeds.

1️⃣ First, it picks a random data point as the first centroid.
2️⃣ For each remaining point, calculate its distance to the nearest already chosen centroid.
3️⃣ Assign a probability to each point, where farther points have a higher probability of being chosen.
4️⃣ Randomly select the next centroid based on these probabilities.
5️⃣ Repeat until K centroids are chosen.

![image](https://github.com/user-attachments/assets/aead8434-90ec-4822-a133-7537bf693c78)


``` python

#Defining the model 
k_means = KMeans(init = "k-means++", n_clusters = 4, n_init = 12)

#training the model 
k_means.fit(X)

#labels for each point
# After clustering , every data point belongs to one of K clusters.This label stores which cluster each point belongs to.
k_means_labels = k_means.labels_

#co ord of each cluster
k_means_cluster_centers = k_means.cluster_centers_

#Plotting
# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(6, 4))

# plt.cm.tab10 is a colour map with 10 colors.
#np.linspace(0,1, N) generates N evenly spaced numbers between 0 and 1. You need to specify N (the number of clusters).
#len(set(k_means_labels)) gives the number of unique clusters, not the total number of data points in k-means.

#need lin space to select different colours for different clusters.

colors = plt.cm.tab10(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4, 4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data points that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)

    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]

    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.',ms=10)

    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()

```
![image](https://github.com/user-attachments/assets/28d430ed-e2c8-4271-a0d7-5705595af975)


##  DBSCAN and HDBSCAN clustering

