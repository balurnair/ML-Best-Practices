## Clustering

<div class="list-group" id="list-tab" role="tablist">
  <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Notebook Content</h3>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Overview" role="tab" aria-controls="settings"> Overview<span class="badge badge-primary badge-pill"></span></a><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering-Dataset" role="tab" aria-controls="settings">Dataset Description<span class="badge badge-primary badge-pill"></span></a><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Affinity-Propagation" role="tab" aria-controls="settings">Affinity Propagation<span class="badge badge-primary badge-pill"></span></a><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Agglomerative-Clustering" role="tab" aria-controls="settings">Agglomerative Clustering<span class="badge badge-primary badge-pill"></span></a><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#BIRCH" role="tab" aria-controls="settings">BIRCH<span class="badge badge-primary badge-pill"></span></a><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#DBSCAN" role="tab" aria-controls="settings">DBSCAN<span class="badge badge-primary badge-pill"></span></a><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#K-Means" role="tab" aria-controls="settings">K-Means Clustering<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#Mini-Batch-K-Means" role="tab" aria-controls="settings">Mini Batch K-Means Clustering<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#Mean-Shift" role="tab" aria-controls="settings">Mean Shift<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#OPTICS" role="tab" aria-controls="settings">OPTICS<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#Spectral-Clustering" role="tab" aria-controls="settings">Spectral Clustering<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#Gaussian-Mixture-Model" role="tab" aria-controls="settings">Gaussian Mixture Model<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#K-Mode" role="tab" aria-controls="settings">K-Mode<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#K-Prototype" role="tab" aria-controls="settings">K-Prototype<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#k-medoidpampartitioning-around-medoid" role="tab" aria-controls="settings">K-Medoid/PAM(Partitioning around Medoid)<span class="badge badge-primary badge-pill"></span></a><br>
     <a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering-Validity-Measures" role="tab" aria-controls="settings">Clustering Validity Measures<span class="badge badge-primary badge-pill"></span></a><br>
    
   </div>

# Overview

Clustering or cluster analysis is an **unsupervised learning** problem. Sometimes, rather than ‘making predictions’, we instead want to **categorize data** into buckets. This is termed “unsupervised learning.”

Clustering techniques apply when there is no class to be predicted but rather when the instances are to be divided into natural groups.

A **cluster is often an area of density in the feature space** where examples from the domain (observations or rows of data) are closer to the cluster than other clusters. The cluster may have a center (the centroid) that is a sample or a point feature space and may have a boundary or extent. These clusters presumably reflect some mechanism at work in the domain from which instances are drawn, a mechanism that causes some instances to bear a stronger resemblance to each other than they do to the remaining instances.

It is often used as a data analysis technique for **discovering interesting patterns in data**.

There are many clustering algorithms to choose from and no single best clustering algorithm for all cases. Instead, it is a good idea to explore a range of clustering algorithms and different configurations for each algorithm.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

# Clustering Algorithms
There are many types of clustering algorithms.

Many algorithms use similarity or distance measures between examples in the feature space in an effort to discover dense regions of observations. As such, it is often good practice to scale data prior to using clustering algorithms.

Central to all of the goals of cluster analysis is the notion of the degree of similarity (or dissimilarity) between the individual objects being clustered. A clustering method attempts to group the objects based on the definition of similarity supplied to it.

Some clustering algorithms require you to specify or guess at the number of clusters to discover in the data, whereas others require the specification of some minimum distance between observations in which examples may be considered “close” or “connected.”

As such, cluster analysis is an iterative process where subjective evaluation of the identified clusters is fed back into changes to algorithm configuration until a desired or appropriate result is achieved.

The scikit-learn library provides a suite of different clustering algorithms to choose from. There is no best clustering algorithm, and no easy way to find the best algorithm for the data without using controlled experiments.

# Clustering Dataset
We will use the **make_classification() function to create a test binary classification dataset**.

The dataset will have 1,000 examples, with two input features and one cluster per class. The clusters are visually obvious in two dimensions so that we can plot the data with a scatter plot and color the points in the plot by the assigned cluster. This will help to see, at least on the test problem, how “well” the clusters were identified.

The clusters in this test problem are based on a multivariate Gaussian, and not all clustering algorithms will be effective at identifying these types of clusters. As such, **the results should not be used as the basis for comparing the methods generally**.

An example of creating and summarizing the synthetic clustering dataset is listed below.


```python
# synthetic classification dataset
from numpy import where
from sklearn.datasets import make_classification
from matplotlib import pyplot
# define dataset
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# create scatter plot for samples from each class
for class_value in range(2):
    # get row indexes for samples with this class
    row_ix = where(y == class_value)
    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/output_6_0.png)


Running the code creates the synthetic clustering dataset, then creates a scatter plot of the input data with points colored by class label (idealized clusters). Two distinct groups of data in two dimensions can be clearly seen.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

### Affinity Propagation
Affinity Propagation involves finding a **set of exemplars** that best summarize the data.

It takes as input measures of similarity between pairs of data points. Real-valued messages are exchanged between data points until a high-quality set of exemplars and corresponding clusters gradually emerges.

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/download.png)

Affinity Propagation can be **interesting as it chooses the number of clusters based on the data provided**. For this purpose, the two important parameters are the **preference**, which controls how many exemplars are used, and the **damping factor** which damps the responsibility and availability messages to avoid numerical oscillations when updating these messages.

The **main drawback** of Affinity Propagation is its complexity. Further, the memory complexity is also high if a dense similarity matrix is used, but reducible if a sparse similarity matrix is used. This makes Affinity Propagation **most appropriate for small to medium sized datasets**.

The complete example is listed below.


```python
# affinity propagation clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = AffinityPropagation(damping=0.9)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
```

    C:\Users\duhita\anaconda3\lib\site-packages\sklearn\cluster\_affinity_propagation.py:152: FutureWarning: 'random_state' has been introduced in 0.23. It will be set to None starting from 0.25 which means that results will differ at every function call. Set 'random_state' to None to silence this warning, or to 0 to keep the behavior of versions <0.23.
      FutureWarning)
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/output_10_1.png)


Running the example fits the model on the training dataset and predicts a cluster for each example in the dataset. A scatter plot is then created with points colored by their assigned cluster.

In this case, Good results could not be achieved.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

### Agglomerative Clustering
Agglomerative clustering involves merging examples until the desired number of clusters is achieved.

It is a part of a broader class of hierarchical clustering methods. It is implemented via the AgglomerativeClustering class and the main configuration to tune is the “n_clusters” set, an estimate of the number of clusters in the data, e.g. 2.

The AgglomerativeClustering object performs a hierarchical clustering using a bottom up approach: each observation starts in its own cluster, and clusters are successively merged together. The linkage criteria determines the metric used for the merge strategy:
-**Ward** minimizes the sum of squared differences within all clusters. It is a variance-minimizing approach and in this sense is similar to the k-means objective function but tackled with an agglomerative hierarchical approach.
-**Maximum** or **complete linkage** minimizes the maximum distance between observations of pairs of clusters.
-**Average linkage** minimizes the average of the distances between all observations of pairs of clusters.
-**Single linkage** minimizes the distance between the closest observations of pairs of clusters.

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/download(1).png)

Agglomerative cluster has a “rich get richer” behavior that leads to uneven cluster sizes. In this regard, single linkage is the worst strategy, and Ward gives the most regular sizes. However, the affinity (or distance used in clustering) cannot be varied with Ward, thus for non Euclidean metrics, average linkage is a good alternative. Single linkage, while not robust to noisy data, can be computed very efficiently and can therefore be useful to provide hierarchical clustering of larger datasets. Single linkage can also perform well on non-globular data.

AgglomerativeClustering can also scale to large number of samples when it is used jointly with a connectivity matrix, but is computationally expensive when no connectivity constraints are added between samples: it considers at each step all the possible merges.

The complete example is listed below.


```python
# agglomerative clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = AgglomerativeClustering(n_clusters=2)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/output_14_0.png)


Running the example fits the model on the training dataset and predicts a cluster for each example in the dataset. A scatter plot is then created with points colored by their assigned cluster.

In this case, a reasonable grouping is found.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

### BIRCH
BIRCH Clustering (BIRCH is short for Balanced Iterative Reducing and Clustering using
Hierarchies) involves constructing a tree structure from which cluster centroids are extracted.

BIRCH incrementally and dynamically clusters incoming multi-dimensional metric data points to try to produce the best quality clustering with the available resources (i. e., available memory and time constraints).

The Birch algorithm has two parameters, the threshold and the branching factor. The branching factor limits the number of subclusters in a node and the threshold limits the distance between the entering sample and the existing subclusters.

This algorithm can be viewed as an instance or data reduction method, since it reduces the input data to a set of subclusters which are obtained directly from the leaves of the CFT. This reduced data can be further processed by feeding it into a global clusterer. This global clusterer can be set by n_clusters. 

Birch does not scale very well to high dimensional data. As a rule of thumb if n_features is greater than twenty, it is generally better to use other algorithms like MiniBatchKMeans.

If the number of instances of data needs to be reduced, or if one wants a large number of subclusters either as a preprocessing step or otherwise, Birch is a useful algorithm.

The complete example is listed below.


```python
# birch clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import Birch
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = Birch(threshold=0.01, n_clusters=2)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/output_18_0.png)


Running the example fits the model on the training dataset and predicts a cluster for each example in the dataset. A scatter plot is then created with points colored by their assigned cluster.

In this case, an excellent grouping is found.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

### DBSCAN
The DBSCAN algorithm views **clusters as areas of high density** separated by areas of low density. Due to this rather generic view, clusters found by DBSCAN can be any shape, as opposed to k-means which assumes that clusters are convex shaped. 

The central component to the DBSCAN is the concept of core samples, which are samples that are in areas of high density. A cluster is therefore a set of core samples, each close to each other (measured by some distance measure) and a set of non-core samples that are close to a core sample (but are not themselves core samples). 

There are two parameters to the algorithm, **min_samples** and **eps**, which define formally what we mean when we say dense. Higher min_samples or lower eps indicate higher density necessary to form a cluster.

Any core sample is part of a cluster, by definition. Any sample that is not a core sample, and is at least eps in distance from any core sample, is considered an outlier by the algorithm. In the figure below, the color indicates cluster membership, with large circles indicating core samples found by the algorithm. Smaller circles are non-core samples that are still part of a cluster. Moreover, the outliers are indicated by black points below.

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/download(2).png)

The complete example is listed below.


```python
# dbscan clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = DBSCAN(eps=0.30, min_samples=9)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/output_22_0.png)


Running the example fits the model on the training dataset and predicts a cluster for each example in the dataset. A scatter plot is then created with points colored by their assigned cluster.

In this case, a reasonable grouping is found, although more tuning is required.

The DBSCAN algorithm is **deterministic, always generating the same clusters when given the same data in the same order**. However, the results can differ when data is provided in a different order. First, even though the core samples will always be assigned to the same clusters, the labels of those clusters will depend on the order in which those samples are encountered in the data. 

Second and more importantly, the clusters to which non-core samples are assigned can differ depending on the data order. This would happen when a non-core sample has a distance lower than eps to two core samples in different clusters. By the triangular inequality, those two core samples must be more distant than eps from each other, or they would be in the same cluster. The non-core sample is assigned to whichever cluster is generated first in a pass through the data, and so the results will depend on the data ordering.

This implementation is by default not memory efficient because it constructs a full pairwise similarity matrix in the case where kd-trees or ball-trees cannot be used (e.g., with sparse matrices). This matrix will consume n^2 floats. A couple of mechanisms for getting around this are:

OPTICS clustering in conjunction with the extract_dbscan method is more memory efficient. OPTICS clustering also calculates the full pairwise matrix, but only keeps one row in memory at a time (memory complexity n).

A sparse radius neighborhood graph (where missing entries are presumed to be out of eps) can be precomputed in a memory-efficient way and dbscan can be run over this with metric='precomputed'.

The dataset can be compressed, either by removing exact duplicates if these occur in the data, or by using BIRCH. This gives a relatively small number of representatives for a large number of points.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

## K-Means
K-Means Clustering may be the **most widely known clustering algorithm** and involves assigning examples to clusters in an effort to minimize the variance within each cluster.

The KMeans algorithm clusters data by trying to **separate samples in n groups of equal variance**, **minimizing** a criterion known as the **inertia or within-cluster sum-of-squares**. This algorithm requires the number of clusters to be specified. It scales well to large number of samples.



### How to Choose the Right Number of Clusters in K-Means Clustering?

1.Elbow Method: Elbow method gives us an idea on what a good k number of clusters would be based on the sum of squared distance (SSE)/Inertia between data points and their assigned clusters’ centroids. We pick k at the spot where SSE starts to flatten out and forming an elbow. We’ll evaluate SSE for different values of k and see where the curve might form an elbow and flatten out.

2.The Silhoutte Method:The silhouette value measures how similar a point is to its own cluster (cohesion) compared to other clusters (separation). The range of the Silhouette value is between +1 and -1. A high value is desirable and indicates that the point is placed in the correct cluster. If many points have a negative Silhouette value, it may indicate that we have created too many or too few clusters.

### Elbow Method


```python
from sklearn.datasets import make_blobs

# Create dataset with 3 random cluster centers and 1000 datapoints
x, y = make_blobs(n_samples = 1000, centers = 3, n_features=2, shuffle=True, random_state=31)
```


```python
# function returns WSS score for k values from 1 to kmax
def calculate_WSS(points, kmax):
  sse = []
  for k in range(1, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(points)
    centroids = kmeans.cluster_centers_
    pred_clusters = kmeans.predict(points)
    curr_sse = 0
    
    # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
    for i in range(len(points)):
      curr_center = centroids[pred_clusters[i]]
      curr_sse += (points[i, 0] - curr_center[0]) ** 2 + (points[i, 1] - curr_center[1]) ** 2
      
    sse.append(curr_sse)
  return sse
```


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
wss = calculate_WSS(x, 10)
y = np.linspace(1, 10, 10)
plt.plot(y, wss)
```




    [<matplotlib.lines.Line2D at 0x1cc116326d8>]




![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/output_31_1.png)


As expected, the plot looks like an arm with a clear elbow at k = 3.

### Silhoutte Method


```python
from sklearn.metrics import silhouette_score

sil = []
kmax = 10

# dissimilarity would not be defined for a single cluster, thus, minimum number of clusters should be 2
for k in range(2, kmax+1):
  kmeans = KMeans(n_clusters = k).fit(x)
  labels = kmeans.labels_
  sil.append(silhouette_score(x, labels, metric = 'euclidean'))
```


```python
sil
```




    [0.7554555291482525,
     0.8430349122094365,
     0.677298976447064,
     0.5128163623725102,
     0.3255367643263419,
     0.3381489462981078,
     0.344553007437787,
     0.34138764028291346,
     0.3379530986267173]



The Silhouette Score reaches its global maximum at the optimal k. This should ideally appear as a peak in the Silhouette Value-versus-k plot.


```python
y1 = np.linspace(2, 10, 9)
plt.plot(y1, sil)
```




    [<matplotlib.lines.Line2D at 0x1cc115c2080>]




![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/output_37_1.png)


There is a clear peak at k = 3. Hence, it is optimal.
Finally, the data can be optimally clustered into 3 clusters

### Implementing k-Means using Elbow Method on a Dataset

The aim of this problem is to segment the clients of a wholesale distributor based on their annual spending on diverse product categories like milk, grocery, region, etc.
We will be working on a wholesale customer segmentation problem. You can download the dataset using https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv link. The data is hosted on the UCI Machine Learning repository.


```python
data=pd.read_csv("C:/Users/amit/Documents/Phase-2/dataset/Wholesale customers data.csv")
data.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Channel</th>
      <th>Region</th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicassen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>3</td>
      <td>12669</td>
      <td>9656</td>
      <td>7561</td>
      <td>214</td>
      <td>2674</td>
      <td>1338</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>3</td>
      <td>7057</td>
      <td>9810</td>
      <td>9568</td>
      <td>1762</td>
      <td>3293</td>
      <td>1776</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>3</td>
      <td>6353</td>
      <td>8808</td>
      <td>7684</td>
      <td>2405</td>
      <td>3516</td>
      <td>7844</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>3</td>
      <td>13265</td>
      <td>1196</td>
      <td>4221</td>
      <td>6404</td>
      <td>507</td>
      <td>1788</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2</td>
      <td>3</td>
      <td>22615</td>
      <td>5410</td>
      <td>7198</td>
      <td>3915</td>
      <td>1777</td>
      <td>5185</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Channel</th>
      <th>Region</th>
      <th>Fresh</th>
      <th>Milk</th>
      <th>Grocery</th>
      <th>Frozen</th>
      <th>Detergents_Paper</th>
      <th>Delicassen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
      <td>440.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.322727</td>
      <td>2.543182</td>
      <td>12000.297727</td>
      <td>5796.265909</td>
      <td>7951.277273</td>
      <td>3071.931818</td>
      <td>2881.493182</td>
      <td>1524.870455</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.468052</td>
      <td>0.774272</td>
      <td>12647.328865</td>
      <td>7380.377175</td>
      <td>9503.162829</td>
      <td>4854.673333</td>
      <td>4767.854448</td>
      <td>2820.105937</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>55.000000</td>
      <td>3.000000</td>
      <td>25.000000</td>
      <td>3.000000</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>3127.750000</td>
      <td>1533.000000</td>
      <td>2153.000000</td>
      <td>742.250000</td>
      <td>256.750000</td>
      <td>408.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>8504.000000</td>
      <td>3627.000000</td>
      <td>4755.500000</td>
      <td>1526.000000</td>
      <td>816.500000</td>
      <td>965.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>16933.750000</td>
      <td>7190.250000</td>
      <td>10655.750000</td>
      <td>3554.250000</td>
      <td>3922.000000</td>
      <td>1820.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>112151.000000</td>
      <td>73498.000000</td>
      <td>92780.000000</td>
      <td>60869.000000</td>
      <td>40827.000000</td>
      <td>47943.000000</td>
    </tr>
  </tbody>
</table>
</div>



Since K-Means is a distance-based algorithm, this difference of magnitude can create a problem. So let’s first bring all the variables to the same magnitude:


```python
# standardizing the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# statistics of scaled data
pd.DataFrame(data_scaled).describe()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>4.400000e+02</td>
      <td>4.400000e+02</td>
      <td>4.400000e+02</td>
      <td>4.400000e+02</td>
      <td>4.400000e+02</td>
      <td>4.400000e+02</td>
      <td>4.400000e+02</td>
      <td>4.400000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-2.452584e-16</td>
      <td>-5.737834e-16</td>
      <td>-2.422305e-17</td>
      <td>-1.589638e-17</td>
      <td>-6.030530e-17</td>
      <td>1.135455e-17</td>
      <td>-1.917658e-17</td>
      <td>-8.276208e-17</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.001138e+00</td>
      <td>1.001138e+00</td>
      <td>1.001138e+00</td>
      <td>1.001138e+00</td>
      <td>1.001138e+00</td>
      <td>1.001138e+00</td>
      <td>1.001138e+00</td>
      <td>1.001138e+00</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-6.902971e-01</td>
      <td>-1.995342e+00</td>
      <td>-9.496831e-01</td>
      <td>-7.787951e-01</td>
      <td>-8.373344e-01</td>
      <td>-6.283430e-01</td>
      <td>-6.044165e-01</td>
      <td>-5.402644e-01</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-6.902971e-01</td>
      <td>-7.023369e-01</td>
      <td>-7.023339e-01</td>
      <td>-5.783063e-01</td>
      <td>-6.108364e-01</td>
      <td>-4.804306e-01</td>
      <td>-5.511349e-01</td>
      <td>-3.964005e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-6.902971e-01</td>
      <td>5.906683e-01</td>
      <td>-2.767602e-01</td>
      <td>-2.942580e-01</td>
      <td>-3.366684e-01</td>
      <td>-3.188045e-01</td>
      <td>-4.336004e-01</td>
      <td>-1.985766e-01</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.448652e+00</td>
      <td>5.906683e-01</td>
      <td>3.905226e-01</td>
      <td>1.890921e-01</td>
      <td>2.849105e-01</td>
      <td>9.946441e-02</td>
      <td>2.184822e-01</td>
      <td>1.048598e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.448652e+00</td>
      <td>5.906683e-01</td>
      <td>7.927738e+00</td>
      <td>9.183650e+00</td>
      <td>8.936528e+00</td>
      <td>1.191900e+01</td>
      <td>7.967672e+00</td>
      <td>1.647845e+01</td>
    </tr>
  </tbody>
</table>
</div>



The magnitude looks similar now. Next, let’s create a kmeans function and fit it on the data:


```python
# defining the kmeans function with initialization as k-means++
kmeans = KMeans(n_clusters=2, init='k-means++')

# fitting the k means algorithm on scaled data
kmeans.fit(data_scaled)
```




    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=None, tol=0.0001, verbose=0)



We have initialized two clusters and pay attention – the initialization is not random here. We have used the k-means++ initialization which generally produces better results.

Let’s evaluate how well the formed clusters are. To do that, we will calculate the inertia of the clusters:


```python
kmeans.inertia_
```




    2599.384423783625



We got an inertia value of almost 2600. Now, let’s see how we can use the elbow curve to determine the optimum number of clusters in Python.

We will first fit multiple k-means models and in each successive model, we will increase the number of clusters. We will store the inertia value of each model and then plot it to visualize the result:


```python
import warnings
warnings.filterwarnings("ignore")
# fitting multiple k-means algorithms and storing the values in an empty list
SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(data_scaled)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them
frame = pd.DataFrame({'Cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
```




    Text(0,0.5,'Inertia')




![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/output_52_1.png)


Looking at the above elbow curve, we can choose any number of clusters between 5 to 8. Let’s set the number of clusters as 5 and fit the model:


```python
# k means using 5 clusters and k-means++ initialization
kmeans = KMeans(n_jobs = -1, n_clusters = 5, init='k-means++')
kmeans.fit(data_scaled)
pred = kmeans.predict(data_scaled)
```


```python
frame = pd.DataFrame(data_scaled)
frame['cluster'] = pred
frame['cluster'].value_counts()
```




    0    209
    1    126
    4     92
    2     12
    3      1
    Name: cluster, dtype: int64




```python
# Visualisation of clusters(K means)
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import KMeans
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = KMeans(n_clusters=2)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/output_56_0.png)


Running the example fits the model on the training dataset and predicts a cluster for each example in the dataset. A scatter plot is then created with points colored by their assigned cluster.

In this case, a reasonable grouping is found, although the unequal equal variance in each dimension makes the method less suited to this dataset.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

### Mini-Batch K-Means
Mini-Batch K-Means is a **modified version of k-means** that **makes updates to the cluster centroids using mini-batches of samples** rather than the entire dataset, which can make it faster for large datasets, and perhaps more robust to statistical noise. It reduces computation cost by orders of magnitude compared to the classic batch algorithm while yielding significantly better solutions than online stochastic gradient descent.

It is implemented via the MiniBatchKMeans class and the main configuration to tune is the “n_clusters” hyperparameter set to the estimated number of clusters in the data.

MiniBatchKMeans converges faster than KMeans, but the quality of the results is reduced. In practice this difference in quality can be quite small, as shown in the example and cited reference.

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/download(3).png)

The complete example is listed below.


```python
# mini-batch k-means clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = MiniBatchKMeans(n_clusters=2)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/output_60_0.png)


Running the example fits the model on the training dataset and predicts a cluster for each example in the dataset. A scatter plot is then created with points colored by their assigned cluster.

In this case, a result equivalent to the standard k-means algorithm is found.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

### Mean Shift
Mean shift clustering involves **finding and adapting centroids based on the density** of examples in the feature space. It is a **centroid based algorithm**, which works by updating candidates for centroids to be the mean of the points within a given region. These candidates are then filtered in a post-processing stage to **eliminate near-duplicates** to form the final set of centroids.

The algorithm is not highly scalable, as it requires multiple nearest neighbor searches during the execution of the algorithm. The algorithm is guaranteed to converge, however the algorithm will stop iterating when the change in centroids is small. Labelling a new sample is performed by finding the nearest centroid for a given sample.

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/download(4).png)

It is implemented via the MeanShift class and the main configuration to tune is the **“bandwidth”** hyperparameter.

The complete example is listed below.


```python
# mean shift clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import MeanShift
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = MeanShift()
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/output_64_0.png)


Running the example fits the model on the training dataset and predicts a cluster for each example in the dataset. A scatter plot is then created with points colored by their assigned cluster.

In this case, a reasonable set of clusters are found in the data.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

### OPTICS
OPTICS clustering (where OPTICS is short for Ordering Points To Identify the Clustering Structure) is a modified version of DBSCAN described above. The OPTICS algorithm shares many similarities with the DBSCAN algorithm, and can be considered a generalization of DBSCAN that relaxes the eps requirement from a single value to a value range. 

The key difference between DBSCAN and OPTICS is that the OPTICS algorithm builds a reachability graph, which assigns each sample both a reachability_ distance, and a spot within the cluster ordering_ attribute; these two attributes are assigned when the model is fitted, and are used to determine cluster membership. 

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/download(5).png)

The **reachability distances generated by OPTICS allow for variable density extraction of clusters** within a single data set. As shown in the above plot, **combining reachability distances and data set ordering_ produces a reachability plot**, where point density is represented on the Y-axis, and points are ordered such that nearby points are adjacent. ‘Cutting’ the reachability plot at a single value produces DBSCAN like results; all points above the ‘cut’ are classified as noise, and each time that there is a break when reading from left to right signifies a new cluster. 

It is implemented via the OPTICS class and the main configuration to tune is the “eps” and “min_samples” hyperparameters.

The complete example is listed below.


```python
# optics clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import OPTICS
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = OPTICS(eps=0.8, min_samples=10)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/output_68_0.png)


A code to make the reachablity plot (with make_classification  data) as show above in the defination is given below.


```python
from sklearn.cluster import OPTICS, cluster_optics_dbscan
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

# define dataset
X, _ = make_classification(n_samples=1500, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = OPTICS(eps=0.8, min_samples=10)

# Run the fit
model.fit(X)

labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=0.5)
labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                   core_distances=clust.core_distances_,
                                   ordering=clust.ordering_, eps=2)

space = np.arange(len(X))
reachability = model.reachability_[model.ordering_]
labels = model.labels_[model.ordering_]

plt.figure(figsize=(10, 7))
G = gridspec.GridSpec(2, 3)
ax1 = plt.subplot(G[0, :])
ax2 = plt.subplot(G[1, 0])
ax3 = plt.subplot(G[1, 1])
ax4 = plt.subplot(G[1, 2])

# Reachability plot
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    ax1.plot(Xk, Rk, color, alpha=0.3)
ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
ax1.plot(space, np.full_like(space, 2., dtype=float), 'k-', alpha=0.5)
ax1.plot(space, np.full_like(space, 0.5, dtype=float), 'k-.', alpha=0.5)
ax1.set_ylabel('Reachability (epsilon distance)')
ax1.set_title('Reachability Plot')

# OPTICS
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = X[model.labels_ == klass]
    ax2.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax2.plot(X[model.labels_ == -1, 0], X[model.labels_ == -1, 1], 'k+', alpha=0.1)
ax2.set_title('Automatic Clustering\nOPTICS')

# DBSCAN at 0.5
colors = ['g', 'greenyellow', 'olive', 'r', 'b', 'c']
for klass, color in zip(range(0, 6), colors):
    Xk = X[labels_050 == klass]
    ax3.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3, marker='.')
ax3.plot(X[labels_050 == -1, 0], X[labels_050 == -1, 1], 'k+', alpha=0.1)
ax3.set_title('Clustering at 0.5 epsilon cut\nDBSCAN')

# DBSCAN at 2.
colors = ['g.', 'm.', 'y.', 'c.']
for klass, color in zip(range(0, 4), colors):
    Xk = X[labels_200 == klass]
    ax4.plot(Xk[:, 0], Xk[:, 1], color, alpha=0.3)
ax4.plot(X[labels_200 == -1, 0], X[labels_200 == -1, 1], 'k+', alpha=0.1)
ax4.set_title('Clustering at 2.0 epsilon cut\nDBSCAN')

plt.tight_layout()
plt.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/output_70_0.png)


Running the example fits the model on the training dataset and predicts a cluster for each example in the dataset. A scatter plot is then created with points colored by their assigned cluster.

In this case, Reasonable results on this dataset could not be achieved.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

### Spectral Clustering
Spectral Clustering is a general class of clustering methods, **drawn from linear algebra**. SpectralClustering performs a **low-dimension embedding** of the affinity matrix between samples, followed by clustering, e.g., by KMeans, of the components of the eigenvectors in the low dimensional space. It is especially computationally efficient if the affinity matrix is sparse and the amg solver is used for the eigenvalue problem.

For two clusters, SpectralClustering solves a convex relaxation of the normalised cuts problem on the similarity graph: cutting the graph in two so that the weight of the edges cut is small compared to the weights of the edges inside each cluster. This criteria is especially interesting when working on images, where graph vertices are pixels, and weights of the edges of the similarity graph are computed using a function of a gradient of the image.

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/download(7).png)

It is implemented via the SpectralClustering class and the main Spectral Clustering is a general class of clustering methods, drawn from linear algebra. to tune is the “n_clusters” hyperparameter used to specify the estimated number of clusters in the data.

The complete example is listed below.


```python
# spectral clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.cluster import SpectralClustering
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = SpectralClustering(n_clusters=2)
# fit model and predict clusters
yhat = model.fit_predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/output_74_0.png)


Running the example fits the model on the training dataset and predicts a cluster for each example in the dataset. A scatter plot is then created with points colored by their assigned cluster.

In this case, reasonable clusters were found.

Different **label assignment strategies** can be used, corresponding to the assign_labels parameter of SpectralClustering. "kmeans" strategy can match finer details, but can be unstable. In particular, unless the random_state is controlled, it may not be reproducible from run-to-run, as it depends on random initialization. The alternative "discretize" strategy is 100% reproducible, but tends to create parcels of fairly even and geometrical shape.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

### Gaussian Mixture Model
A Gaussian mixture model **summarizes a multivariate probability density function** with a mixture of Gaussian probability distributions as its name suggests.

A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. The GaussianMixture object implements the expectation-maximization (EM) algorithm for fitting mixture-of-Gaussian models. One can think of mixture models as generalizing k-means clustering to incorporate information about the covariance structure of the data as well as the centers of the latent Gaussians.

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/download(8).png)

It is implemented via the GaussianMixture class and the main configuration to tune is the “n_clusters” hyperparameter used to specify the estimated number of clusters in the data.

It is the **fastest algorithm for learning mixture models**. As this algorithm maximizes only the likelihood, it will not bias the means towards zero, or bias the cluster sizes to have specific structures that might or might not apply.
When one has insufficiently many points per mixture, estimating the covariance matrices becomes difficult, and the algorithm is known to diverge and find solutions with infinite likelihood unless one regularizes the covariances artificially. This algorithm will always use all the components it has access to, needing held-out data or information theoretical criteria to decide how many components to use in the absence of external cues.

The complete example is listed below.


```python
# gaussian mixture clustering
from numpy import unique
from numpy import where
from sklearn.datasets import make_classification
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot
# define dataset
X, _ = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)
# define the model
model = GaussianMixture(n_components=2)
# fit the model
model.fit(X)
# assign a cluster to each example
yhat = model.predict(X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
    # get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)
    # create scatter of these samples
    pyplot.scatter(X[row_ix, 0], X[row_ix, 1])
# show the plot
pyplot.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/output_78_0.png)


Running the example fits the model on the training dataset and predicts a cluster for each example in the dataset. A scatter plot is then created with points colored by their assigned cluster.

In this case, it can be seen that the clusters were identified perfectly. This is not surprising given that the dataset was generated as a mixture of Gaussians.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

# K-Mode

The basic concept of k-means stands on mathematical calculations (means, euclidian distances). But what if our data is non-numerical or, in other words, categorical? 
We could think of transforming our categorical values in numerical values and eventually apply k-means. But beware: k-means uses numerical distances, so it could consider close two really distant objects that merely have been assigned two close numbers.k-modes is an extension of k-means. Instead of distances it uses dissimilarities (that is, quantification of the total mismatches between two objects: the smaller this number, the more similar the two objects). And instead of means, it uses modes. A mode is a vector of elements that minimizes the dissimilarities between the vector itself and each object of the data. We will have as many modes as the number of clusters we required, since they act as centroids.


```python
import numpy as np
import pandas as pd
k = 3 # Specify the number of clusters

X = np.array([
        [1,1,7], 
        [2,1,8], 
        [2,2,7], 
        [5,3,9], 
        [6,3,9], 
        [5,3,9]])

# Print the number of data and dimension 
n = len(X)
d = len(X[0])
addZeros = np.zeros((n, 1))
X = np.append(X, addZeros, axis=1)
print("The K Modes algorithm: \n")
print("The training data: \n", X)
print("Total number of data: ",n)
print("Total number of features: ",d)
print("Total number of Clusters: ",k)
```

    The K Modes algorithm: 
    
    The training data: 
     [[1. 1. 7. 0.]
     [2. 1. 8. 0.]
     [2. 2. 7. 0.]
     [5. 3. 9. 0.]
     [6. 3. 9. 0.]
     [5. 3. 9. 0.]]
    Total number of data:  6
    Total number of features:  3
    Total number of Clusters:  3
    


```python
# Random selection of initial cluster centers
import numpy as np
dup = np.array([])
while 1:
    ranIndex = np.random.randint(low=1, high=n, size=k)
    u, c = np.unique(ranIndex, return_counts=True)
    dup = u[c > 1]
    if dup.size == 0:
        break
C = X[ranIndex]
print("\n The initial cluster centers: \n", C[:,0:d])
print("\n")
```

    
     The initial cluster centers: 
     [[2. 2. 7.]
     [5. 3. 9.]
     [6. 3. 9.]]
    
    
    


```python
# Function to calculate distance between two nominal data
def distanceNominal(x,y):
   d = len(x)
   sumD = 0
   for i in range(d):
       if (x[i] !=y [i]):
           sumD +=1
   return sumD
```


```python
from scipy import stats
# Main iteration starts
for it in range(10): # Total number of iterations
    for i in range(n): # Iterate each data
        minDist = 9999999999
        for j in range(k): # Iterate each cluster center
            #Distance calculation from centers
            dist = distanceNominal(C[j,0:d], X[i,0:d])
            if (dist<minDist):
                minDist = dist
                clusterNumber = j
                X[i,d] = clusterNumber
                C[j,d] = clusterNumber
     # Group the data to calculate the mean
    for j in range(k):
        result = np.where(X[:,d] == j)
        mode_info = stats.mode(X[result])
        C[j] = np.reshape(mode_info[0],(d+1))  
```


```python
# Calculate cost value
cost = 0
for i in range(n):
    for j in range(k):
        if X[i,d] == C[j,d]:
            cost += distanceNominal(C[j,0:d], X[i,0:d])
cost = cost/n
# 
print("The Final cluster centers: \n", C)
print("\n The data with cluster number: \n", X)
print("\n The cost is: ", np.round(cost,4))
# # End of cost value calculation
```

    The Final cluster centers: 
     [[2. 1. 7. 0.]
     [5. 3. 9. 1.]
     [6. 3. 9. 2.]]
    
     The data with cluster number: 
     [[1. 1. 7. 0.]
     [2. 1. 8. 0.]
     [2. 2. 7. 0.]
     [5. 3. 9. 1.]
     [6. 3. 9. 2.]
     [5. 3. 9. 1.]]
    
     The cost is:  0.5
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

# K-Prototype

Just like K — means where we allocate the record to the closest centroid (a reference point to the cluster), here we allocate the record to the cluster which has the most similar looking reference point a.k.a prototype of the cluster a.k.a centroid of the cluster.

More than similarity the algorithms try to find the dissimilarity between data points and try to group points with less dissimilarity into a cluster
.
The dissimilarity measure for numeric attributes is the square Euclidean distance whereas the similarity measure on categorical attributes is the number of matching attributes between objects and cluster prototypes.


```python
pip install kmodes
```

    
    The following command must be run outside of the IPython shell:
    
        $ pip install kmodes
    
    The Python package manager (pip) can only be used from outside of IPython.
    Please reissue the `pip` command in a separate terminal or command prompt.
    
    See the Python documentation for more information on how to install packages:
    
        https://docs.python.org/3/installing/
    


```python
# importing necessary libraries
import pandas as pd
import numpy as np
from scipy import stats
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt
from matplotlib import style
```


```python
#read the file
marketing_df = pd.read_csv('C:/Users/amit/Documents/Phase-2/dataset/marketing_cva_f.csv')
marketing_df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Customer</th>
      <th>State</th>
      <th>CLV</th>
      <th>Coverage</th>
      <th>Income</th>
      <th>loc_type</th>
      <th>monthly_premium</th>
      <th>months_last_claim</th>
      <th>Months_Since_Policy_Inception</th>
      <th>Total_Claim_Amount</th>
      <th>Vehicle_Class</th>
      <th>avg_vehicle_age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>BU79786</td>
      <td>Washington</td>
      <td>2763.519279</td>
      <td>Basic</td>
      <td>56274</td>
      <td>Suburban</td>
      <td>69</td>
      <td>32</td>
      <td>5</td>
      <td>384.811147</td>
      <td>Two-Door Car</td>
      <td>40.696695</td>
    </tr>
    <tr>
      <th>1</th>
      <td>AI49188</td>
      <td>Nevada</td>
      <td>12887.431650</td>
      <td>Premium</td>
      <td>48767</td>
      <td>Suburban</td>
      <td>108</td>
      <td>18</td>
      <td>38</td>
      <td>566.472247</td>
      <td>Two-Door Car</td>
      <td>48.755298</td>
    </tr>
    <tr>
      <th>2</th>
      <td>HB64268</td>
      <td>Washington</td>
      <td>2813.692575</td>
      <td>Basic</td>
      <td>43836</td>
      <td>Rural</td>
      <td>73</td>
      <td>12</td>
      <td>44</td>
      <td>138.130879</td>
      <td>Four-Door Car</td>
      <td>70.394474</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OC83172</td>
      <td>Oregon</td>
      <td>8256.297800</td>
      <td>Basic</td>
      <td>62902</td>
      <td>Rural</td>
      <td>69</td>
      <td>14</td>
      <td>94</td>
      <td>159.383042</td>
      <td>Two-Door Car</td>
      <td>53.460212</td>
    </tr>
    <tr>
      <th>4</th>
      <td>XZ87318</td>
      <td>Oregon</td>
      <td>5380.898636</td>
      <td>Basic</td>
      <td>55350</td>
      <td>Suburban</td>
      <td>67</td>
      <td>0</td>
      <td>13</td>
      <td>321.600000</td>
      <td>Four-Door Car</td>
      <td>32.811507</td>
    </tr>
  </tbody>
</table>
</div>




```python
marketing_df=marketing_df.drop(['Customer','Vehicle_Class','avg_vehicle_age','months_last_claim','Total_Claim_Amount'],axis=1)
```


```python
mark_array=marketing_df.values
```


```python
mark_array[:, 1] = mark_array[:, 1].astype(float)
mark_array[:, 3] = mark_array[:, 3].astype(float)
mark_array[:, 5] = mark_array[:, 5].astype(float)
mark_array[:, 6] = mark_array[:, 6].astype(float)
```


```python
mark_array
```




    array([['Washington', 2763.519279, 'Basic', ..., 'Suburban', 69.0, 5.0],
           ['Nevada', 12887.43165, 'Premium', ..., 'Suburban', 108.0, 38.0],
           ['Washington', 2813.692575, 'Basic', ..., 'Rural', 73.0, 44.0],
           ...,
           ['California', 23405.98798, 'Basic', ..., 'Urban', 73.0, 89.0],
           ['California', 3096.511217, 'Extended', ..., 'Suburban', 79.0,
            28.0],
           ['California', 7524.442436, 'Extended', ..., 'Suburban', 96.0,
            3.0]], dtype=object)




```python
kproto = KPrototypes(n_clusters=3, verbose=2,max_iter=20)
clusters = kproto.fit_predict(mark_array, categorical=[0, 2, 4])
```

    Init: initializing centroids
    Init: initializing clusters
    Starting iterations...
    Run: 1, iteration: 1/20, moves: 999, ncost: 846501643869.7179
    Run: 1, iteration: 2/20, moves: 521, ncost: 806797750297.4905
    Run: 1, iteration: 3/20, moves: 359, ncost: 791987052888.7139
    Run: 1, iteration: 4/20, moves: 183, ncost: 787836890881.0387
    Run: 1, iteration: 5/20, moves: 104, ncost: 786426844156.4158
    Run: 1, iteration: 6/20, moves: 74, ncost: 785733314562.7207
    Run: 1, iteration: 7/20, moves: 38, ncost: 785571996335.9271
    Run: 1, iteration: 8/20, moves: 11, ncost: 785552822475.9596
    Run: 1, iteration: 9/20, moves: 6, ncost: 785548092552.7047
    Run: 1, iteration: 10/20, moves: 4, ncost: 785545634415.1553
    Run: 1, iteration: 11/20, moves: 0, ncost: 785545634415.1553
    Init: initializing centroids
    Init: initializing clusters
    Starting iterations...
    Run: 2, iteration: 1/20, moves: 1353, ncost: 996592521686.7178
    Run: 2, iteration: 2/20, moves: 1099, ncost: 871463592902.5999
    Run: 2, iteration: 3/20, moves: 645, ncost: 818295889850.6456
    Run: 2, iteration: 4/20, moves: 420, ncost: 796174339905.6849
    Run: 2, iteration: 5/20, moves: 232, ncost: 789407032644.8394
    Run: 2, iteration: 6/20, moves: 135, ncost: 787053472267.3033
    Run: 2, iteration: 7/20, moves: 88, ncost: 786017677597.46
    Run: 2, iteration: 8/20, moves: 58, ncost: 785626178299.5214
    Run: 2, iteration: 9/20, moves: 20, ncost: 785564139538.9407
    Run: 2, iteration: 10/20, moves: 9, ncost: 785551468522.3029
    Run: 2, iteration: 11/20, moves: 6, ncost: 785546422486.7771
    Run: 2, iteration: 12/20, moves: 2, ncost: 785545634415.1554
    Run: 2, iteration: 13/20, moves: 0, ncost: 785545634415.1554
    Init: initializing centroids
    Init: initializing clusters
    Starting iterations...
    Run: 3, iteration: 1/20, moves: 1796, ncost: 947277002519.4674
    Run: 3, iteration: 2/20, moves: 915, ncost: 843623094279.6534
    Run: 3, iteration: 3/20, moves: 560, ncost: 805009557966.1805
    Run: 3, iteration: 4/20, moves: 343, ncost: 791536618295.5936
    Run: 3, iteration: 5/20, moves: 170, ncost: 787758777586.0885
    Run: 3, iteration: 6/20, moves: 100, ncost: 786415917594.5874
    Run: 3, iteration: 7/20, moves: 74, ncost: 785726741417.6898
    Run: 3, iteration: 8/20, moves: 35, ncost: 785576817979.9009
    Run: 3, iteration: 9/20, moves: 12, ncost: 785553835556.5504
    Run: 3, iteration: 10/20, moves: 4, ncost: 785550664653.3425
    Run: 3, iteration: 11/20, moves: 7, ncost: 785545634415.1554
    Run: 3, iteration: 12/20, moves: 0, ncost: 785545634415.1554
    Init: initializing centroids
    Init: initializing clusters
    Starting iterations...
    Run: 4, iteration: 1/20, moves: 1283, ncost: 855866760868.7408
    Run: 4, iteration: 2/20, moves: 586, ncost: 812690969201.948
    Run: 4, iteration: 3/20, moves: 391, ncost: 794094496619.8457
    Run: 4, iteration: 4/20, moves: 202, ncost: 788796801136.1307
    Run: 4, iteration: 5/20, moves: 120, ncost: 786870228602.3743
    Run: 4, iteration: 6/20, moves: 87, ncost: 785903097925.9778
    Run: 4, iteration: 7/20, moves: 51, ncost: 785606912443.4148
    Run: 4, iteration: 8/20, moves: 18, ncost: 785557453367.9037
    Run: 4, iteration: 9/20, moves: 7, ncost: 785550664653.3424
    Run: 4, iteration: 10/20, moves: 7, ncost: 785545634415.1553
    Run: 4, iteration: 11/20, moves: 0, ncost: 785545634415.1553
    Init: initializing centroids
    Init: initializing clusters
    Starting iterations...
    Run: 5, iteration: 1/20, moves: 686, ncost: 786268491116.9663
    Run: 5, iteration: 2/20, moves: 100, ncost: 785676881791.3881
    Run: 5, iteration: 3/20, moves: 15, ncost: 785636565854.424
    Run: 5, iteration: 4/20, moves: 21, ncost: 785592464242.3938
    Run: 5, iteration: 5/20, moves: 9, ncost: 785580196116.0236
    Run: 5, iteration: 6/20, moves: 4, ncost: 785577307058.1284
    Run: 5, iteration: 7/20, moves: 4, ncost: 785574087818.5052
    Run: 5, iteration: 8/20, moves: 3, ncost: 785572843787.3959
    Run: 5, iteration: 9/20, moves: 0, ncost: 785572843787.3959
    Init: initializing centroids
    Init: initializing clusters
    Starting iterations...
    Run: 6, iteration: 1/20, moves: 1256, ncost: 869295047725.5557
    Run: 6, iteration: 2/20, moves: 666, ncost: 810776308868.2255
    Run: 6, iteration: 3/20, moves: 368, ncost: 792832397091.7493
    Run: 6, iteration: 4/20, moves: 202, ncost: 787532909779.4669
    Run: 6, iteration: 5/20, moves: 81, ncost: 786437772137.1063
    Run: 6, iteration: 6/20, moves: 58, ncost: 785945968414.2338
    Run: 6, iteration: 7/20, moves: 45, ncost: 785708436066.95
    Run: 6, iteration: 8/20, moves: 20, ncost: 785655117685.7761
    Run: 6, iteration: 9/20, moves: 15, ncost: 785613961116.1897
    Run: 6, iteration: 10/20, moves: 15, ncost: 785587014534.5708
    Run: 6, iteration: 11/20, moves: 6, ncost: 785580196116.0239
    Run: 6, iteration: 12/20, moves: 4, ncost: 785577307058.1285
    Run: 6, iteration: 13/20, moves: 4, ncost: 785574087818.505
    Run: 6, iteration: 14/20, moves: 3, ncost: 785572843787.3959
    Run: 6, iteration: 15/20, moves: 0, ncost: 785572843787.3959
    Init: initializing centroids
    Init: initializing clusters
    Starting iterations...
    Run: 7, iteration: 1/20, moves: 923, ncost: 792151845715.2307
    Run: 7, iteration: 2/20, moves: 182, ncost: 787840468631.2158
    Run: 7, iteration: 3/20, moves: 105, ncost: 786426844156.4155
    Run: 7, iteration: 4/20, moves: 74, ncost: 785733314562.7208
    Run: 7, iteration: 5/20, moves: 38, ncost: 785571996335.9276
    Run: 7, iteration: 6/20, moves: 11, ncost: 785552822475.9592
    Run: 7, iteration: 7/20, moves: 6, ncost: 785548092552.7047
    Run: 7, iteration: 8/20, moves: 4, ncost: 785545634415.1553
    Run: 7, iteration: 9/20, moves: 0, ncost: 785545634415.1553
    Init: initializing centroids
    Init: initializing clusters
    Starting iterations...
    Run: 8, iteration: 1/20, moves: 742, ncost: 786488271751.0723
    Run: 8, iteration: 2/20, moves: 98, ncost: 785674082470.9698
    Run: 8, iteration: 3/20, moves: 29, ncost: 785568188439.9851
    Run: 8, iteration: 4/20, moves: 11, ncost: 785551468522.3025
    Run: 8, iteration: 5/20, moves: 6, ncost: 785546422486.7771
    Run: 8, iteration: 6/20, moves: 2, ncost: 785545634415.1553
    Run: 8, iteration: 7/20, moves: 0, ncost: 785545634415.1553
    Init: initializing centroids
    Init: initializing clusters
    Starting iterations...
    Run: 9, iteration: 1/20, moves: 537, ncost: 814307424096.8529
    Run: 9, iteration: 2/20, moves: 407, ncost: 794521348329.5726
    Run: 9, iteration: 3/20, moves: 207, ncost: 788967647058.1592
    Run: 9, iteration: 4/20, moves: 126, ncost: 786900443343.6208
    Run: 9, iteration: 5/20, moves: 86, ncost: 785932203890.0603
    Run: 9, iteration: 6/20, moves: 50, ncost: 785621748137.485
    Run: 9, iteration: 7/20, moves: 22, ncost: 785557453367.9037
    Run: 9, iteration: 8/20, moves: 7, ncost: 785550664653.3427
    Run: 9, iteration: 9/20, moves: 7, ncost: 785545634415.1553
    Run: 9, iteration: 10/20, moves: 0, ncost: 785545634415.1553
    Init: initializing centroids
    Init: initializing clusters
    Starting iterations...
    Run: 10, iteration: 1/20, moves: 564, ncost: 819469137179.7957
    Run: 10, iteration: 2/20, moves: 410, ncost: 796551269889.757
    Run: 10, iteration: 3/20, moves: 234, ncost: 789565746800.6543
    Run: 10, iteration: 4/20, moves: 135, ncost: 787156625857.9867
    Run: 10, iteration: 5/20, moves: 90, ncost: 786066485686.8143
    Run: 10, iteration: 6/20, moves: 60, ncost: 785639281247.7822
    Run: 10, iteration: 7/20, moves: 22, ncost: 785566620155.142
    Run: 10, iteration: 8/20, moves: 10, ncost: 785551468522.303
    Run: 10, iteration: 9/20, moves: 6, ncost: 785546422486.7771
    Run: 10, iteration: 10/20, moves: 2, ncost: 785545634415.1554
    Run: 10, iteration: 11/20, moves: 0, ncost: 785545634415.1554
    Best run was number 1
    


```python
print(kproto.cluster_centroids_)
```

    [array([[8.04338174e+03, 8.29341174e+04, 9.09228225e+01, 4.76527012e+01],
           [8.33216831e+03, 5.46449257e+04, 9.51337668e+01, 4.91597771e+01],
           [8.03295458e+03, 2.66208754e+04, 9.26217544e+01, 4.69445614e+01]]), array([['California', 'Basic', 'Rural'],
           ['California', 'Basic', 'Suburban'],
           ['California', 'Basic', 'Suburban']], dtype='<U10')]
    


```python
cluster_dict=[]
for c in clusters:
    cluster_dict.append(c)
```


```python
cluster_dict
```




    [1,
     1,
     1,
     1,
     1,
     2,
     2,
     0,
     0,
     0,
     2,
     2,
     2,
     1,
     1,
     1,
     2,
     2,
     2,
     0,
     2,
     0,
     0,
     2,
     1,
     1,
     2,
     0,
     1,
     1,
     1,
     1,
     1,
     1,
     0,
     2,
     2,
     2,
     2,
     2,
     2,
     2,
     1,
     1,
     2,
     2,
     2,
     2,
     1,
     2,
     2,
     1,
     2,
     0,
     1,
     2,
     0,
     2,
     0,
     2,
     2,
     1,
     1,
     2,
     1,
     2,
     2,
     2,
     1,
     2,
     2,
     2,
     0,
     0,
     1,
     0,
     2,
     2,
     1,
     2,
     1,
     1,
     1,
     2,
     2,
     0,
     1,
     2,
     2,
     1,
     1,
     2,
     2,
     0,
     0,
     1,
     0,
     2,
     2,
     1,
     1,
     2,
     2,
     2,
     1,
     2,
     0,
     0,
     0,
     1,
     1,
     1,
     1,
     2,
     0,
     2,
     0,
     2,
     2,
     0,
     2,
     1,
     2,
     2,
     1,
     0,
     2,
     2,
     0,
     2,
     2,
     2,
     1,
     2,
     1,
     0,
     2,
     2,
     2,
     1,
     0,
     0,
     1,
     1,
     1,
     0,
     1,
     0,
     2,
     1,
     1,
     2,
     2,
     2,
     1,
     1,
     2,
     0,
     0,
     0,
     2,
     0,
     0,
     0,
     0,
     2,
     2,
     0,
     0,
     0,
     1,
     2,
     2,
     2,
     1,
     2,
     2,
     1,
     1,
     0,
     1,
     1,
     1,
     1,
     1,
     1,
     2,
     0,
     1,
     2,
     1,
     2,
     2,
     1,
     2,
     0,
     2,
     2,
     0,
     2,
     2,
     0,
     2,
     2,
     2,
     0,
     0,
     2,
     1,
     2,
     2,
     2,
     0,
     2,
     1,
     2,
     1,
     0,
     0,
     0,
     0,
     0,
     2,
     2,
     0,
     1,
     1,
     2,
     0,
     2,
     2,
     1,
     1,
     2,
     2,
     1,
     0,
     0,
     2,
     0,
     1,
     1,
     2,
     1,
     0,
     2,
     1,
     2,
     2,
     2,
     1,
     1,
     2,
     1,
     1,
     0,
     2,
     2,
     1,
     2,
     2,
     0,
     0,
     1,
     1,
     0,
     0,
     2,
     2,
     1,
     1,
     2,
     2,
     1,
     2,
     2,
     2,
     1,
     1,
     2,
     1,
     2,
     2,
     0,
     0,
     1,
     1,
     0,
     1,
     2,
     2,
     2,
     0,
     1,
     0,
     1,
     1,
     1,
     2,
     1,
     2,
     1,
     2,
     2,
     1,
     2,
     2,
     0,
     0,
     0,
     2,
     1,
     2,
     0,
     1,
     0,
     0,
     1,
     2,
     2,
     1,
     2,
     1,
     2,
     1,
     2,
     2,
     1,
     2,
     1,
     2,
     2,
     0,
     2,
     2,
     1,
     1,
     2,
     2,
     2,
     2,
     2,
     2,
     1,
     2,
     1,
     0,
     1,
     1,
     1,
     2,
     0,
     2,
     1,
     0,
     1,
     1,
     2,
     2,
     1,
     2,
     2,
     1,
     1,
     2,
     1,
     0,
     1,
     2,
     0,
     0,
     1,
     1,
     2,
     0,
     2,
     2,
     2,
     2,
     2,
     0,
     2,
     1,
     1,
     0,
     1,
     2,
     2,
     0,
     2,
     2,
     2,
     0,
     1,
     2,
     2,
     2,
     0,
     2,
     0,
     0,
     1,
     0,
     0,
     2,
     0,
     2,
     2,
     1,
     2,
     2,
     2,
     1,
     2,
     0,
     2,
     0,
     1,
     0,
     2,
     2,
     1,
     1,
     2,
     2,
     0,
     1,
     1,
     1,
     1,
     1,
     1,
     2,
     2,
     2,
     1,
     0,
     2,
     2,
     1,
     0,
     0,
     1,
     1,
     1,
     2,
     1,
     2,
     2,
     0,
     1,
     2,
     0,
     0,
     2,
     0,
     1,
     1,
     1,
     0,
     2,
     2,
     0,
     1,
     2,
     2,
     0,
     2,
     2,
     0,
     2,
     1,
     2,
     2,
     2,
     2,
     1,
     0,
     1,
     1,
     2,
     1,
     1,
     0,
     1,
     2,
     1,
     1,
     0,
     0,
     2,
     0,
     2,
     1,
     2,
     0,
     0,
     2,
     0,
     1,
     0,
     2,
     1,
     1,
     2,
     2,
     2,
     0,
     2,
     2,
     1,
     0,
     1,
     1,
     2,
     0,
     0,
     1,
     2,
     0,
     0,
     2,
     0,
     0,
     2,
     1,
     2,
     0,
     1,
     2,
     1,
     0,
     2,
     2,
     0,
     1,
     2,
     1,
     2,
     2,
     2,
     2,
     1,
     1,
     1,
     1,
     0,
     1,
     2,
     2,
     0,
     0,
     2,
     1,
     2,
     1,
     1,
     1,
     2,
     1,
     2,
     1,
     2,
     1,
     2,
     1,
     2,
     1,
     1,
     0,
     2,
     2,
     1,
     0,
     2,
     0,
     0,
     1,
     0,
     2,
     1,
     0,
     1,
     0,
     2,
     1,
     2,
     0,
     1,
     0,
     1,
     0,
     2,
     1,
     2,
     2,
     2,
     1,
     1,
     0,
     2,
     1,
     2,
     0,
     0,
     0,
     2,
     0,
     0,
     1,
     2,
     0,
     2,
     1,
     2,
     0,
     1,
     0,
     1,
     2,
     2,
     0,
     1,
     0,
     0,
     1,
     1,
     2,
     1,
     2,
     2,
     1,
     0,
     2,
     1,
     1,
     2,
     1,
     0,
     2,
     1,
     1,
     1,
     1,
     1,
     0,
     1,
     2,
     0,
     0,
     0,
     0,
     1,
     2,
     2,
     1,
     0,
     1,
     2,
     2,
     0,
     0,
     2,
     0,
     2,
     0,
     0,
     0,
     2,
     0,
     2,
     0,
     0,
     1,
     2,
     1,
     1,
     2,
     2,
     2,
     2,
     1,
     1,
     1,
     2,
     2,
     1,
     0,
     2,
     1,
     0,
     1,
     2,
     2,
     1,
     0,
     1,
     0,
     2,
     1,
     2,
     0,
     2,
     0,
     0,
     0,
     2,
     1,
     2,
     1,
     2,
     0,
     0,
     1,
     2,
     0,
     0,
     2,
     2,
     0,
     2,
     2,
     1,
     0,
     0,
     0,
     1,
     1,
     2,
     0,
     2,
     2,
     1,
     0,
     2,
     2,
     0,
     2,
     0,
     1,
     0,
     2,
     2,
     2,
     2,
     1,
     2,
     2,
     0,
     2,
     1,
     2,
     2,
     0,
     2,
     0,
     2,
     1,
     1,
     0,
     1,
     0,
     1,
     2,
     2,
     1,
     2,
     1,
     0,
     2,
     2,
     1,
     0,
     0,
     2,
     2,
     1,
     2,
     2,
     2,
     2,
     0,
     1,
     1,
     1,
     0,
     0,
     0,
     0,
     0,
     1,
     1,
     0,
     2,
     0,
     0,
     1,
     1,
     0,
     2,
     2,
     2,
     0,
     2,
     1,
     2,
     0,
     1,
     0,
     0,
     2,
     0,
     2,
     1,
     2,
     1,
     0,
     2,
     2,
     1,
     0,
     2,
     1,
     1,
     0,
     1,
     1,
     1,
     2,
     1,
     2,
     1,
     1,
     2,
     2,
     2,
     0,
     2,
     1,
     2,
     1,
     2,
     2,
     1,
     2,
     0,
     1,
     1,
     2,
     0,
     2,
     1,
     1,
     1,
     0,
     1,
     0,
     1,
     2,
     1,
     0,
     2,
     2,
     1,
     2,
     1,
     2,
     0,
     2,
     2,
     0,
     1,
     2,
     2,
     0,
     0,
     0,
     0,
     0,
     0,
     2,
     1,
     0,
     1,
     2,
     1,
     0,
     2,
     0,
     2,
     2,
     1,
     2,
     2,
     0,
     2,
     2,
     2,
     2,
     0,
     2,
     0,
     1,
     0,
     0,
     2,
     2,
     0,
     2,
     1,
     1,
     2,
     0,
     2,
     0,
     0,
     1,
     2,
     1,
     0,
     1,
     2,
     0,
     2,
     0,
     0,
     2,
     2,
     2,
     1,
     2,
     1,
     2,
     0,
     0,
     1,
     0,
     0,
     2,
     1,
     0,
     2,
     1,
     2,
     0,
     0,
     2,
     2,
     2,
     2,
     2,
     1,
     0,
     2,
     1,
     0,
     2,
     2,
     2,
     2,
     1,
     2,
     2,
     2,
     2,
     2,
     0,
     1,
     2,
     1,
     2,
     1,
     2,
     1,
     2,
     1,
     0,
     1,
     0,
     1,
     0,
     1,
     2,
     1,
     1,
     2,
     1,
     1,
     1,
     2,
     2,
     1,
     2,
     1,
     2,
     0,
     1,
     2,
     1,
     1,
     ...]




```python
marketing_df['cluster']=cluster_dict
```


```python
marketing_df
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>State</th>
      <th>CLV</th>
      <th>Coverage</th>
      <th>Income</th>
      <th>loc_type</th>
      <th>monthly_premium</th>
      <th>Months_Since_Policy_Inception</th>
      <th>cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Washington</td>
      <td>2763.519279</td>
      <td>Basic</td>
      <td>56274</td>
      <td>Suburban</td>
      <td>69</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Nevada</td>
      <td>12887.431650</td>
      <td>Premium</td>
      <td>48767</td>
      <td>Suburban</td>
      <td>108</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Washington</td>
      <td>2813.692575</td>
      <td>Basic</td>
      <td>43836</td>
      <td>Rural</td>
      <td>73</td>
      <td>44</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Oregon</td>
      <td>8256.297800</td>
      <td>Basic</td>
      <td>62902</td>
      <td>Rural</td>
      <td>69</td>
      <td>94</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Oregon</td>
      <td>5380.898636</td>
      <td>Basic</td>
      <td>55350</td>
      <td>Suburban</td>
      <td>67</td>
      <td>13</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Oregon</td>
      <td>24127.504020</td>
      <td>Basic</td>
      <td>14072</td>
      <td>Suburban</td>
      <td>71</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Oregon</td>
      <td>7388.178085</td>
      <td>Extended</td>
      <td>28812</td>
      <td>Urban</td>
      <td>93</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>California</td>
      <td>8798.797003</td>
      <td>Premium</td>
      <td>77026</td>
      <td>Urban</td>
      <td>110</td>
      <td>82</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Arizona</td>
      <td>8819.018934</td>
      <td>Basic</td>
      <td>99845</td>
      <td>Suburban</td>
      <td>110</td>
      <td>25</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>California</td>
      <td>5384.431665</td>
      <td>Basic</td>
      <td>83689</td>
      <td>Urban</td>
      <td>70</td>
      <td>10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Oregon</td>
      <td>7463.139377</td>
      <td>Basic</td>
      <td>24599</td>
      <td>Rural</td>
      <td>64</td>
      <td>50</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Nevada</td>
      <td>2566.867823</td>
      <td>Basic</td>
      <td>25049</td>
      <td>Suburban</td>
      <td>67</td>
      <td>7</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>California</td>
      <td>3945.241604</td>
      <td>Basic</td>
      <td>28855</td>
      <td>Suburban</td>
      <td>101</td>
      <td>59</td>
      <td>2</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Oregon</td>
      <td>5710.333115</td>
      <td>Basic</td>
      <td>51148</td>
      <td>Urban</td>
      <td>72</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>California</td>
      <td>8162.617053</td>
      <td>Premium</td>
      <td>66140</td>
      <td>Suburban</td>
      <td>101</td>
      <td>21</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Oregon</td>
      <td>2872.051273</td>
      <td>Basic</td>
      <td>57749</td>
      <td>Suburban</td>
      <td>74</td>
      <td>21</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Washington</td>
      <td>3041.791561</td>
      <td>Extended</td>
      <td>13789</td>
      <td>Suburban</td>
      <td>79</td>
      <td>49</td>
      <td>2</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Arizona</td>
      <td>24127.504020</td>
      <td>Basic</td>
      <td>14072</td>
      <td>Suburban</td>
      <td>71</td>
      <td>3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>18</th>
      <td>California</td>
      <td>2392.107890</td>
      <td>Basic</td>
      <td>17870</td>
      <td>Suburban</td>
      <td>61</td>
      <td>91</td>
      <td>2</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Oregon</td>
      <td>5802.065978</td>
      <td>Basic</td>
      <td>97541</td>
      <td>Suburban</td>
      <td>72</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Washington</td>
      <td>5346.916576</td>
      <td>Extended</td>
      <td>10511</td>
      <td>Urban</td>
      <td>139</td>
      <td>64</td>
      <td>2</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Arizona</td>
      <td>12902.560140</td>
      <td>Premium</td>
      <td>86584</td>
      <td>Suburban</td>
      <td>111</td>
      <td>54</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Oregon</td>
      <td>3235.360468</td>
      <td>Extended</td>
      <td>75690</td>
      <td>Suburban</td>
      <td>80</td>
      <td>44</td>
      <td>0</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Arizona</td>
      <td>2454.583540</td>
      <td>Basic</td>
      <td>23158</td>
      <td>Suburban</td>
      <td>63</td>
      <td>6</td>
      <td>2</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Nevada</td>
      <td>18975.456110</td>
      <td>Extended</td>
      <td>65999</td>
      <td>Urban</td>
      <td>237</td>
      <td>14</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Washington</td>
      <td>5018.885233</td>
      <td>Basic</td>
      <td>54500</td>
      <td>Suburban</td>
      <td>63</td>
      <td>17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Oregon</td>
      <td>4932.916345</td>
      <td>Basic</td>
      <td>37260</td>
      <td>Rural</td>
      <td>62</td>
      <td>42</td>
      <td>2</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Arizona</td>
      <td>5744.229745</td>
      <td>Basic</td>
      <td>68987</td>
      <td>Urban</td>
      <td>71</td>
      <td>40</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>California</td>
      <td>13891.735670</td>
      <td>Premium</td>
      <td>42305</td>
      <td>Suburban</td>
      <td>117</td>
      <td>62</td>
      <td>1</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Oregon</td>
      <td>7380.976717</td>
      <td>Extended</td>
      <td>65706</td>
      <td>Suburban</td>
      <td>91</td>
      <td>86</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>6787</th>
      <td>California</td>
      <td>9410.670129</td>
      <td>Premium</td>
      <td>96060</td>
      <td>Suburban</td>
      <td>117</td>
      <td>57</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6788</th>
      <td>California</td>
      <td>5878.447428</td>
      <td>Basic</td>
      <td>25398</td>
      <td>Suburban</td>
      <td>74</td>
      <td>2</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6789</th>
      <td>California</td>
      <td>4547.321823</td>
      <td>Basic</td>
      <td>29031</td>
      <td>Suburban</td>
      <td>61</td>
      <td>19</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6790</th>
      <td>California</td>
      <td>5926.385440</td>
      <td>Basic</td>
      <td>92949</td>
      <td>Urban</td>
      <td>74</td>
      <td>84</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6791</th>
      <td>California</td>
      <td>2580.849899</td>
      <td>Basic</td>
      <td>46900</td>
      <td>Suburban</td>
      <td>66</td>
      <td>59</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6792</th>
      <td>California</td>
      <td>7083.642205</td>
      <td>Premium</td>
      <td>97024</td>
      <td>Urban</td>
      <td>177</td>
      <td>68</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6793</th>
      <td>California</td>
      <td>4148.570285</td>
      <td>Basic</td>
      <td>61896</td>
      <td>Urban</td>
      <td>104</td>
      <td>97</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6794</th>
      <td>California</td>
      <td>2518.743544</td>
      <td>Basic</td>
      <td>39317</td>
      <td>Rural</td>
      <td>64</td>
      <td>46</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6795</th>
      <td>California</td>
      <td>3843.965188</td>
      <td>Extended</td>
      <td>43987</td>
      <td>Suburban</td>
      <td>96</td>
      <td>17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6796</th>
      <td>California</td>
      <td>9075.768214</td>
      <td>Basic</td>
      <td>37722</td>
      <td>Rural</td>
      <td>116</td>
      <td>23</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6797</th>
      <td>California</td>
      <td>2619.337376</td>
      <td>Basic</td>
      <td>78618</td>
      <td>Urban</td>
      <td>66</td>
      <td>56</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6798</th>
      <td>California</td>
      <td>15245.254950</td>
      <td>Basic</td>
      <td>30205</td>
      <td>Suburban</td>
      <td>195</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6799</th>
      <td>California</td>
      <td>2615.139220</td>
      <td>Basic</td>
      <td>57023</td>
      <td>Urban</td>
      <td>67</td>
      <td>59</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6800</th>
      <td>California</td>
      <td>5551.398167</td>
      <td>Extended</td>
      <td>36918</td>
      <td>Suburban</td>
      <td>76</td>
      <td>77</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6801</th>
      <td>California</td>
      <td>34611.378960</td>
      <td>Basic</td>
      <td>20090</td>
      <td>Suburban</td>
      <td>109</td>
      <td>59</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6802</th>
      <td>California</td>
      <td>2845.520933</td>
      <td>Basic</td>
      <td>86631</td>
      <td>Suburban</td>
      <td>73</td>
      <td>44</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6803</th>
      <td>California</td>
      <td>5500.577411</td>
      <td>Extended</td>
      <td>44019</td>
      <td>Rural</td>
      <td>138</td>
      <td>60</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6804</th>
      <td>California</td>
      <td>3358.532935</td>
      <td>Extended</td>
      <td>59367</td>
      <td>Rural</td>
      <td>84</td>
      <td>48</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6805</th>
      <td>California</td>
      <td>7501.661322</td>
      <td>Extended</td>
      <td>38874</td>
      <td>Urban</td>
      <td>94</td>
      <td>86</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6806</th>
      <td>California</td>
      <td>5133.397765</td>
      <td>Basic</td>
      <td>28647</td>
      <td>Suburban</td>
      <td>69</td>
      <td>59</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6807</th>
      <td>California</td>
      <td>8732.090534</td>
      <td>Basic</td>
      <td>51205</td>
      <td>Urban</td>
      <td>72</td>
      <td>52</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6808</th>
      <td>California</td>
      <td>9424.256842</td>
      <td>Basic</td>
      <td>46897</td>
      <td>Urban</td>
      <td>118</td>
      <td>11</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6809</th>
      <td>California</td>
      <td>5479.555081</td>
      <td>Basic</td>
      <td>56005</td>
      <td>Suburban</td>
      <td>68</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6810</th>
      <td>California</td>
      <td>25464.820590</td>
      <td>Extended</td>
      <td>13663</td>
      <td>Suburban</td>
      <td>97</td>
      <td>66</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6811</th>
      <td>California</td>
      <td>16261.585500</td>
      <td>Extended</td>
      <td>60646</td>
      <td>Suburban</td>
      <td>134</td>
      <td>42</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6812</th>
      <td>California</td>
      <td>5032.165498</td>
      <td>Basic</td>
      <td>66367</td>
      <td>Suburban</td>
      <td>64</td>
      <td>48</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6813</th>
      <td>California</td>
      <td>4100.398533</td>
      <td>Premium</td>
      <td>47761</td>
      <td>Suburban</td>
      <td>104</td>
      <td>58</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6814</th>
      <td>California</td>
      <td>23405.987980</td>
      <td>Basic</td>
      <td>71941</td>
      <td>Urban</td>
      <td>73</td>
      <td>89</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6815</th>
      <td>California</td>
      <td>3096.511217</td>
      <td>Extended</td>
      <td>21604</td>
      <td>Suburban</td>
      <td>79</td>
      <td>28</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6816</th>
      <td>California</td>
      <td>7524.442436</td>
      <td>Extended</td>
      <td>21941</td>
      <td>Suburban</td>
      <td>96</td>
      <td>3</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
<p>6817 rows × 8 columns</p>
</div>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

# K-Medoid/PAM(Partitioning around Medoid)

Both the k-means and k-medoids algorithms are partitional, which involves breaking the dataset into groups. K-means aims to minimize the total squared error from a central position in each cluster. These central positions are called centroids. On the other hand, k-medoids attempts to minimize the sum of dissimilarities between objects labeled to be in a cluster and one of the objects designated as the representative of that cluster. These representatives are called medoids.
In contrast to the k-means algorithm that the centroids are central, average positions that might not be data points in the set, k-medoids chooses medoids from the data points in the set.


```python
#K-Medoid example on Iris Dataset
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn import datasets
```


```python
# Dataset
iris = datasets.load_iris()
data = pd.DataFrame(iris.data,columns = iris.feature_names)

target = iris.target_names
labels = iris.target
```


```python
#Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
```


```python
#PCA Transformation
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
principalComponents = pca.fit_transform(data)
PCAdf = pd.DataFrame(data = principalComponents , columns = ['principal component 1', 'principal component 2','principal component 3'])

datapoints = PCAdf.values
m, f = datapoints.shape
k = 3
```


```python
#Visualization
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_reduced = datapoints
ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:, 2], c=labels,
           cmap=plt.cm.Set1, edgecolor='k', s=40)
ax.set_title("First three PCA directions")
ax.set_xlabel("principal component 1")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("principal component 1")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("principal component 1")
ax.w_zaxis.set_ticklabels([])
plt.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Clustering/output_111_0.png)



```python
def init_medoids(X, k):
    from numpy.random import choice
    from numpy.random import seed
 
    seed(1)
    samples = choice(len(X), size=k, replace=False)
    return X[samples, :]

medoids_initial = init_medoids(datapoints, 3)

def compute_d_p(X, medoids, p):
    m = len(X)
    medoids_shape = medoids.shape
    # If a 1-D array is provided, 
    # it will be reshaped to a single row 2-D array
    if len(medoids_shape) == 1: 
        medoids = medoids.reshape((1,len(medoids)))
    k = len(medoids)
    
    S = np.empty((m, k))
    
    for i in range(m):
        d_i = np.linalg.norm(X[i, :] - medoids, ord=p, axis=1)
        S[i, :] = d_i**p

    return S
  
S = compute_d_p(datapoints, medoids_initial, 2)


def assign_labels(S):
    return np.argmin(S, axis=1)
  
labels = assign_labels(S)

def update_medoids(X, medoids, p):
    
    S = compute_d_p(datapoints, medoids, p)
    labels = assign_labels(S)
        
    out_medoids = medoids
                
    for i in set(labels):
        
        avg_dissimilarity = np.sum(compute_d_p(datapoints, medoids[i], p))

        cluster_points = datapoints[labels == i]
        
        for datap in cluster_points:
            new_medoid = datap
            new_dissimilarity= np.sum(compute_d_p(datapoints, datap, p))
            
            if new_dissimilarity < avg_dissimilarity :
                avg_dissimilarity = new_dissimilarity
                
                out_medoids[i] = datap
                
    return out_medoids

def has_converged(old_medoids, medoids):
    return set([tuple(x) for x in old_medoids]) == set([tuple(x) for x in medoids])
```


```python
#Full algorithm
def kmedoids(X, k, p, starting_medoids=None, max_steps=np.inf):
    if starting_medoids is None:
        medoids = init_medoids(X, k)
    else:
        medoids = starting_medoids
        
    converged = False
    labels = np.zeros(len(X))
    i = 1
    while (not converged) and (i <= max_steps):
        old_medoids = medoids.copy()
        
        S = compute_d_p(X, medoids, p)
        
        labels = assign_labels(S)
        
        medoids = update_medoids(X, medoids, p)
        
        converged = has_converged(old_medoids, medoids)
        i += 1
    return (medoids,labels)

results = kmedoids(datapoints, 3, 2)
final_medoids = results[0]
data['clusters'] = results[1]

#Count
def mark_matches(a, b, exact=False):
    """
    Given two Numpy arrays of {0, 1} labels, returns a new boolean
    array indicating at which locations the input arrays have the
    same label (i.e., the corresponding entry is True).
    
    This function can consider "inexact" matches. That is, if `exact`
    is False, then the function will assume the {0, 1} labels may be
    regarded as the same up to a swapping of the labels. This feature
    allows
    
      a == [0, 0, 1, 1, 0, 1, 1]
      b == [1, 1, 0, 0, 1, 0, 0]
      
    to be regarded as equal. (That is, use `exact=False` when you
    only care about "relative" labeling.)
    """
    assert a.shape == b.shape
    a_int = a.astype(dtype=int)
    b_int = b.astype(dtype=int)
    all_axes = tuple(range(len(a.shape)))
    assert ((a_int == 0) | (a_int == 1) | (a_int == 2)).all()
    assert ((b_int == 0) | (b_int == 1) | (b_int == 2)).all()
    
    exact_matches = (a_int == b_int)
    if exact:
        return exact_matches

    assert exact == False
    num_exact_matches = np.sum(exact_matches)
    if (2*num_exact_matches) >= np.prod (a.shape):
        return exact_matches
    return exact_matches == False # Invert

def count_matches(a, b, exact=False):
    """
    Given two sets of {0, 1} labels, returns the number of mismatches.
    
    This function can consider "inexact" matches. That is, if `exact`
    is False, then the function will assume the {0, 1} labels may be
    regarded as similar up to a swapping of the labels. This feature
    allows
    
      a == [0, 0, 1, 1, 0, 1, 1]
      b == [1, 1, 0, 0, 1, 0, 0]
      
    to be regarded as equal. (That is, use `exact=False` when you
    only care about "relative" labeling.)
    """
    matches = mark_matches(a, b, exact=exact)
    return np.sum(matches)

n_matches = count_matches(labels, data['clusters'])
print(n_matches,
      "matches out of",
      len(data), "data points",
      "(~ {:.1f}%)".format(100.0 * n_matches / len(labels)))
```

    142 matches out of 150 data points (~ 94.7%)
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

# Clustering Validity Measures

Generally, cluster validity measures are categorized into 3 classes, they are –

Internal cluster validation : The clustering result is evaluated based on the data clustered itself (internal information) without reference to external information. 

External cluster validation : Clustering results are evaluated based on some externally known result, such as externally provided class labels.

Relative cluster validation : The clustering results are evaluated by varying different parameters for the same algorithm (e.g. changing the number of clusters).

## Dunn Index

Dunn Index: The aim of Dunn index is to identify sets of clusters that are compact, with a small variance between members of the cluster, and well separated, where the means of different clusters are sufficiently far apart, as compared to the within cluster variance. Higher the Dunn index value, better is the clustering. The number of clusters that maximizes Dunn index is taken as the optimal number of clusters k.


```python
def delta(ck, cl):    
    values = np.ones([len(ck), len(cl)])*10000
    
    for i in range(0, len(ck)):
        for j in range(0, len(cl)):
            values[i, j] = np.linalg.norm(ck[i]-cl[j])
            
    return np.min(values)
    
def big_delta(ci):
    values = np.zeros([len(ci), len(ci)])
    
    for i in range(0, len(ci)):
        for j in range(0, len(ci)):
            values[i, j] = np.linalg.norm(ci[i]-ci[j])
            
    return np.max(values)
```


```python
def dunn(k_list):
    """ Dunn index [CVI]
    
    Parameters
    ----------
    k_list : list of np.arrays
        A list containing a numpy array for each cluster |c| = number of clusters
        c[K] is np.array([N, p]) (N : number of samples in cluster K, p : sample dimension)
    """
    deltas = np.ones([len(k_list), len(k_list)])*1000000
    big_deltas = np.zeros([len(k_list), 1])
    l_range = list(range(0, len(k_list)))
    
    for k in l_range:
        for l in (l_range[0:k]+l_range[k+1:]):
            deltas[k, l] = delta(k_list[k], k_list[l])
        
        big_deltas[k] = big_delta(k_list[k])

    di = np.min(deltas)/np.max(big_deltas)
    return di
```


```python
# loading the dataset 
X = datasets.load_iris() 
df = pd.DataFrame(X.data) 
  
# K-Means 
from sklearn import cluster 
k_means = cluster.KMeans(n_clusters=3) 
k_means.fit(df) #K-means training 
y_pred = k_means.predict(df) 
  
# We store the K-means results in a dataframe 
pred = pd.DataFrame(y_pred) 
pred.columns = ['Type'] 
  
# we merge this dataframe with df 
prediction = pd.concat([df, pred], axis = 1) 
  
# We store the clusters 
clus0 = prediction.loc[prediction.Type == 0] 
clus1 = prediction.loc[prediction.Type == 1] 
clus2 = prediction.loc[prediction.Type == 2] 
cluster_list = [clus0.values, clus1.values, clus2.values] 
  
print(dunn(cluster_list))
```

    0.38630676272011016
    

## Davies Bouldin Index

Davies Bouldin-Index : It is an internal evaluation scheme, where the validation of how well the clustering has been done is made using quantities and features inherent to the dataset.The DB index captures the intuition that clusters that are (1) well-spaced from each other and (2) themselves very dense are likely a ‘good’ clustering.
Lower the DB index value, better is the clustering. 


```python
from sklearn.cluster import KMeans 
from sklearn.metrics import davies_bouldin_score 
from sklearn.datasets.samples_generator import make_blobs 
  
# loading the dataset 
X, y_true = make_blobs(n_samples=300, centers=4,  
                       cluster_std=0.50, random_state=0) 
  
# K-Means 
kmeans = KMeans(n_clusters=4, random_state=1).fit(X) 
  
# we store the cluster labels 
labels = kmeans.labels_ 
  
print(davies_bouldin_score(X, labels)) 
```

    0.3662877051289654
    

    C:\Users\amit\Anaconda3\lib\site-packages\sklearn\utils\deprecation.py:143: FutureWarning: The sklearn.datasets.samples_generator module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.datasets. Anything that cannot be imported from sklearn.datasets is now part of the private API.
      warnings.warn(message, FutureWarning)
    

## Adjusted RAND Score

Rand Index is a function that computes a similarity measure between two clustering. For this computation rand index considers all pairs of samples and counting pairs that are assigned in the similar or different clusters in the predicted and true clustering. Afterwards, the raw Rand Index score is ‘adjusted for chance’ into the Adjusted Rand Index score by using the following formula −

AdjustedRI=(RI−Expected−RI)/(max(RI)−Expected−RI)
It has two parameters namely labels_true, which is ground truth class labels, and labels_pred, which are clusters label to evaluate.


```python
# loading the dataset 
from sklearn import datasets
import pandas as pd
X = datasets.load_iris()
df = pd.DataFrame(X.data)
```


```python
y_true = X.target
```


```python
# K-Means 
from sklearn import cluster 
k_means = cluster.KMeans(n_clusters=3) 
k_means.fit(df) #K-means training 
y_pred = k_means.predict(df)
```


```python
from sklearn.metrics.cluster import adjusted_rand_score
```


```python
adjusted_rand_score(y_true, y_pred)
```




    0.7302382722834697



## Mutual Info Score

Mutual Information is a function that computes the agreement of the two assignments. It ignores the permutations. 
There are following versions available −
1.Adjusted Mutual Information
2.Normalized Mutual Information


```python
#Adjusted Mutual Information(AMI)
from sklearn import metrics
metrics.adjusted_mutual_info_score(y_true, y_pred)
```




    0.7551191675800484




```python
#Normalized Mutual Information (NMI)
from sklearn.metrics.cluster import normalized_mutual_info_score
   
labels_true = [0, 0, 1, 1, 1, 1]
labels_pred = [0, 0, 2, 2, 3, 3]

normalized_mutual_info_score (y_true, y_pred)

```




    0.7581756800057785



## Homogenity Score, Completeness and V-Measure

Given the knowledge of the ground truth class assignments of the samples, it is possible to define some intuitive metric using conditional entropy analysis.

homogeneity: each cluster contains only members of a single class.

completeness: all members of a given class are assigned to the same cluster.

We can turn those concept as scores homogeneity_score and completeness_score. Both are bounded below by 0.0 and above by 1.0 (higher is better)

Their harmonic mean called V-measure is computed by v_measure_score


```python
metrics.homogeneity_score(y_true, y_pred)
```




    0.7514854021988339




```python
metrics.completeness_score(y_true, y_pred)
```




    0.7649861514489816




```python
metrics.v_measure_score(y_true, y_pred)
```




    0.7581756800057786



## Fowlkes Mallowes Score

The Fowlkes-Mallows score FMI is defined as the geometric mean of the pairwise precision and recall.
The score ranges from 0 to 1. A high value indicates a good similarity between two clusters.


```python
metrics.fowlkes_mallows_score(y_true, y_pred)
```




    0.8208080729114153



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Comparison of the above methods

|Method name|Parameters|Scalability|Usecase|Geometry (metric used)|
|-----------|----------|-----------|-------|----------------------|
|K-Means|number of clusters|Very large n_samples, medium n_clusters with MiniBatch code|General-purpose, even cluster size, flat geometry, not too many clusters|Distances between points|
|Affinity propagation|damping, sample preference|Not scalable with n_samples|Many clusters, uneven cluster size, non-flat geometry|Graph distance (e.g. nearest-neighbor graph)|
|Mean-shift|bandwidth|Not scalable with n_samples|Many clusters, uneven cluster size, non-flat geometry|Distances between points|
|Spectral clustering|number of clusters|Medium n_samples, small n_clusters|Few clusters, even cluster size, non-flat geometry|Graph distance (e.g. nearest-neighbor graph)|
|Ward hierarchical clustering|number of clusters or distance threshold|Large n_samples and n_clusters|Many clusters, possibly connectivity constraints|Distances between points|
|Agglomerative clustering|number of clusters or distance threshold, linkage type, distance|Large n_samples and n_clusters|Many clusters, possibly connectivity constraints, non Euclidean distances|Any pairwise distance|
|DBSCAN|neighborhood size|Very large n_samples, medium n_clusters|Non-flat geometry, uneven cluster sizes|Distances between nearest points|
|OPTICS|minimum cluster membership|Very large n_samples, large n_clusters|Non-flat geometry, uneven cluster sizes, variable cluster density|Distances between points|
|Gaussian mixtures|many|Not scalable|Flat geometry, good for density estimation|Mahalanobis distances to  centers|
|Birch|branching factor, threshold, optional global clusterer.|Large n_clusters and n_samples|Large dataset, outlier removal, data reduction.|Euclidean distance between points|


```python

```
