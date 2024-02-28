# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

# Definition of the KMeans class
class KMeans:
    
    # Constructor method
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        # Initializing parameters
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        
        # List of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        
        # The centers (mean vector) for each cluster
        self.centroids = []

    # Method to predict cluster labels for given data
    def predict(self, X):
        # Storing the input data
        self.X = X
        # Getting the number of samples and features
        self.n_samples, self.n_features = X.shape
        
        # Initializing centroids
        random_sample_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idxs]
        
        # Optimizing clusters
        for _ in range(self.max_iters):
            # Assigning samples to closest centroids
            self.clusters = self._create_clusters(self.centroids)
            
            # Updating centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            
            # Checking for convergence
            if self._is_converged(centroids_old, self.centroids):
                break
                
        # Returning cluster labels
        return self._get_cluster_labels(self.clusters)

    # Method to get cluster labels for each sample
    def _get_cluster_labels(self, clusters):
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels

    # Method to assign samples to the closest centroids
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    # Method to find the index of the closest centroid to a sample
    def _closest_centroid(self, sample, centroids):
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    # Method to calculate new centroids from the clusters
    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    # Method to check convergence of centroids
    def _is_converged(self, centroids_old, centroids):
        distances = [euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    # Method to plot clusters and centroids
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plotting samples in each cluster
        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)
        
        # Plotting centroids
        for point in self.centroids:
            ax.scatter(*point, marker="x", color="black", linewidth=2)
        
        # Displaying the plot
        plt.show()

# Testing the algorithm
if __name__ == "__main__":
    # Setting random seed for reproducibility
    np.random.seed(42)
    
    # Generating sample data
    from sklearn.datasets import make_blobs
    X, y = make_blobs(
        centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40
    )
    
    # Getting the number of clusters from the unique labels
    clusters = len(np.unique(y))
    
    # Initializing and running KMeans algorithm
    k = KMeans(K=clusters, max_iters=150, plot_steps=True)
    y_pred = k.predict(X)
    
    # Plotting the final clusters and centroids
    k.plot()
