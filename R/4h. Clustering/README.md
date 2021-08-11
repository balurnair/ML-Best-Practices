
## Clustering

<div class="list-group" id="list-tab" role="tablist">
  <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Notebook Content</h3><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Overview" role="tab" aria-controls="settings"> Overview<span class="badge badge-primary badge-pill"></span></a><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Affinity-Propagation" role="tab" aria-controls="settings">Affinity Propagation<span class="badge badge-primary badge-pill"></span></a><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Agglomerative-Clustering" role="tab" aria-controls="settings">Agglomerative Clustering<span class="badge badge-primary badge-pill"></span></a><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#DBSCAN" role="tab" aria-controls="settings">DBSCAN<span class="badge badge-primary badge-pill"></span></a><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#K-Means" role="tab" aria-controls="settings">K-Means Clustering<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#Mini-Batch-K-Means" role="tab" aria-controls="settings">Mini Batch K-Means Clustering<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#OPTICS" role="tab" aria-controls="settings">OPTICS<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#Spectral-Clustering" role="tab" aria-controls="settings">Spectral Clustering<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#Gaussian-Mixture-Model" role="tab" aria-controls="settings">Gaussian Mixture Model<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#K-Mode" role="tab" aria-controls="settings">K-Mode<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#K-Prototype" role="tab" aria-controls="settings">K-Prototype<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#k-medoidpampartitioning-around-medoid" role="tab" aria-controls="settings">K-Medoid/PAM(Partitioning around Medoid)<span class="badge badge-primary badge-pill"></span></a><br>
     <a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering-Validity-Measures" role="tab" aria-controls="settings">Clustering Validity Measures<span class="badge badge-primary badge-pill"></span></a><br>
    
   

# Overview

Clustering or cluster analysis is an **unsupervised learning** problem. Sometimes, rather than ‘making predictions’, we instead want to **categorize data** into buckets. This is termed “unsupervised learning.”

Clustering techniques apply when there is no class to be predicted but rather when the instances are to be divided into natural groups.

A **cluster is often an area of density in the feature space** where examples from the domain (observations or rows of data) are closer to the cluster than other clusters. The cluster may have a center (the centroid) that is a sample or a point feature space and may have a boundary or extent. These clusters presumably reflect some mechanism at work in the domain from which instances are drawn, a mechanism that causes some instances to bear a stronger resemblance to each other than they do to the remaining instances.

It is often used as a data analysis technique for **discovering interesting patterns in data**.

There are many clustering algorithms to choose from and no single best clustering algorithm for all cases. Instead, it is a good idea to explore a range of clustering algorithms and different configurations for each algorithm.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Clustering Algorithms
There are many types of clustering algorithms.

Many algorithms use similarity or distance measures between examples in the feature space in an effort to discover dense regions of observations. As such, it is often good practice to scale data prior to using clustering algorithms.

Central to all of the goals of cluster analysis is the notion of the degree of similarity (or dissimilarity) between the individual objects being clustered. A clustering method attempts to group the objects based on the definition of similarity supplied to it.

Some clustering algorithms require you to specify or guess at the number of clusters to discover in the data, whereas others require the specification of some minimum distance between observations in which examples may be considered “close” or “connected.”

As such, cluster analysis is an iterative process where subjective evaluation of the identified clusters is fed back into changes to algorithm configuration until a desired or appropriate result is achieved.

The scikit-learn library provides a suite of different clustering algorithms to choose from. There is no best clustering algorithm, and no easy way to find the best algorithm for the data without using controlled experiments.

Running the code creates the synthetic clustering dataset, then creates a scatter plot of the input data with points colored by class label (idealized clusters). Two distinct groups of data in two dimensions can be clearly seen.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Affinity Propagation
Affinity Propagation involves finding a **set of exemplars** that best summarize the data.

It takes as input measures of similarity between pairs of data points. Real-valued messages are exchanged between data points until a high-quality set of exemplars and corresponding clusters gradually emerges.

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/1.png)

Affinity Propagation can be **interesting as it chooses the number of clusters based on the data provided**. For this purpose, the two important parameters are the **preference**, which controls how many exemplars are used, and the **damping factor** which damps the responsibility and availability messages to avoid numerical oscillations when updating these messages.

The **main drawback** of Affinity Propagation is its complexity. Further, the memory complexity is also high if a dense similarity matrix is used, but reducible if a sparse similarity matrix is used. This makes Affinity Propagation **most appropriate for small to medium sized datasets**.

The complete example is listed below.


```R
## create two Gaussian clouds
cl1 <- cbind(rnorm(100, 0.2, 0.05), rnorm(100, 0.8, 0.06))
cl2 <- cbind(rnorm(50, 0.7, 0.08), rnorm(50, 0.3, 0.05))
x <- rbind(cl1, cl2)
```


```R
#installing package
install.packages("apcluster")
```

    package 'apcluster' successfully unpacked and MD5 sums checked
    
    The downloaded binary packages are in
    	C:\Users\sidhartha\AppData\Local\Temp\RtmpInT5gF\downloaded_packages
    


```R
library(apcluster)
```


```R
## compute similarity matrix and run affinity propagation 
## (p defaults to median of similarity)
apres <- apcluster(negDistMat(r=2), x, details=TRUE)
```


```R
## show details of clustering results
show(apres)
```

    
    APResult object
    
    Number of samples     =  150 
    Number of iterations  =  169 
    Input preference      =  -0.02965015 
    Sum of similarities   =  -0.27524 
    Sum of preferences    =  -0.2668514 
    Net similarity        =  -0.5420914 
    Number of clusters    =  9 
    
    Exemplars:
       1 12 26 57 70 122 123 135 141
    Clusters:
       Cluster 1, exemplar 1:
          1 2 3 4 5 8 11 14 16 18 19 23 24 28 29 40 52 53 54 56 59 60 68 69 71 73 
          74 75 76 78 79 80 83 85 88 91 92 95 96
       Cluster 2, exemplar 12:
          6 7 10 12 13 15 21 22 27 30 32 33 36 38 42 43 44 46 47 51 55 67 82 84 87 
          89 93 97 99
       Cluster 3, exemplar 26:
          26 34 39 45 48 49 50 58 62 64 66 94 100
       Cluster 4, exemplar 57:
          9 17 25 41 57 61 63 77 81 86 90 98
       Cluster 5, exemplar 70:
          20 31 35 37 65 70 72
       Cluster 6, exemplar 122:
          102 103 106 108 114 115 118 120 121 122 124 125 126 130 131 134 136 138 
          143 144 145 147 148 149
       Cluster 7, exemplar 123:
          101 105 109 116 123 128 133 142
       Cluster 8, exemplar 135:
          110 111 112 117 135
       Cluster 9, exemplar 141:
          104 107 113 119 127 129 132 137 139 140 141 146 150
    


```R
## plot clustering result
plot(apres, x)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_13_0.png)


Run affinity propagation with default preference of 10% quantile of similarities; this should lead to a smaller number of clusters



```R
#reuse similarity matrix from previous run
apres <- apcluster(s=apres@sim, q=0.1)
show(apres)
plot(apres, x)
```

    
    APResult object
    
    Number of samples     =  150 
    Number of iterations  =  127 
    Input preference      =  -0.5759545 
    Sum of similarities   =  -0.9656939 
    Sum of preferences    =  -1.151909 
    Net similarity        =  -2.117603 
    Number of clusters    =  2 
    
    Exemplars:
       19 136
    Clusters:
       Cluster 1, exemplar 19:
          1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 
          28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 
          52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 
          76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 
          100
       Cluster 2, exemplar 136:
          101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 
          119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 
          137 138 139 140 141 142 143 144 145 146 147 148 149 150
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_15_1.png)


Now try the same with RBF kernel


```R
sim <- expSimMat(x, r=2)
apres <- apcluster(s=sim, q=0.2)
show(apres)
plot(apres, x)
```

    
    APResult object
    
    Number of samples     =  150 
    Number of iterations  =  128 
    Input preference      =  0.6064023 
    Sum of similarities   =  147.0414 
    Sum of preferences    =  1.212805 
    Net similarity        =  148.2542 
    Number of clusters    =  2 
    
    Exemplars:
       19 136
    Clusters:
       Cluster 1, exemplar 19:
          1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 
          28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 
          52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 
          76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 
          100
       Cluster 2, exemplar 136:
          101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 
          119 120 121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 
          137 138 139 140 141 142 143 144 145 146 147 148 149 150
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_17_1.png)



```R
## create sparse similarity matrix
cl1 <- cbind(rnorm(20, 0.2, 0.05), rnorm(20, 0.8, 0.06))
cl2 <- cbind(rnorm(20, 0.7, 0.08), rnorm(20, 0.3, 0.05))
x <- rbind(cl1, cl2)
```


```R
sim <- negDistMat(x, r=2)
ssim <- as.SparseSimilarityMatrix(sim, lower=-0.2)
```

Run apcluster() on the sparse similarity matrix


```R

apres <- apcluster(ssim, q=0)
apres
```


    
    APResult object
    
    Number of samples     =  40 
    Number of iterations  =  130 
    Input preference      =  -0.198375 
    Sum of similarities   =  -0.3808138 
    Sum of preferences    =  -0.39675 
    Net similarity        =  -0.7775637 
    Number of clusters    =  2 
    
    Exemplars:
       15 21
    Clusters:
       Cluster 1, exemplar 15:
          1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
       Cluster 2, exemplar 21:
          21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40



```R
plot(apres, x)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_22_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

### Agglomerative Clustering
Agglomerative clustering involves merging examples until the desired number of clusters is achieved.

It is a part of a broader class of hierarchical clustering methods. It is implemented via the AgglomerativeClustering class and the main configuration to tune is the “n_clusters” set, an estimate of the number of clusters in the data, e.g. 2.

The AgglomerativeClustering object performs a hierarchical clustering using a bottom up approach: each observation starts in its own cluster, and clusters are successively merged together. The linkage criteria determines the metric used for the merge strategy:
-**Ward** minimizes the sum of squared differences within all clusters. It is a variance-minimizing approach and in this sense is similar to the k-means objective function but tackled with an agglomerative hierarchical approach.
-**Maximum** or **complete linkage** minimizes the maximum distance between observations of pairs of clusters.
-**Average linkage** minimizes the average of the distances between all observations of pairs of clusters.
-**Single linkage** minimizes the distance between the closest observations of pairs of clusters.

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/2.png)

Agglomerative cluster has a “rich get richer” behavior that leads to uneven cluster sizes. In this regard, single linkage is the worst strategy, and Ward gives the most regular sizes. However, the affinity (or distance used in clustering) cannot be varied with Ward, thus for non Euclidean metrics, average linkage is a good alternative. Single linkage, while not robust to noisy data, can be computed very efficiently and can therefore be useful to provide hierarchical clustering of larger datasets. Single linkage can also perform well on non-globular data.

AgglomerativeClustering can also scale to large number of samples when it is used jointly with a connectivity matrix, but is computationally expensive when no connectivity constraints are added between samples: it considers at each step all the possible merges.

The complete example is listed below.


```R
data(iris)
```

We can use hclust for hierarchical clustering implementation. hclust requires us to provide the data in the form of a distance matrix. We can do this by using dist. By default, the complete linkage method is used.


```R
# agglomerative clustering
clusters <- hclust(dist(iris[, 3:4]))
plot(clusters)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_27_0.png)


We can see from the figure that the best choices for total number of clusters are either 3 or 4:

To do this, we can cut off the tree at the desired number of clusters using cutree.


```R
clusterCut <- cutree(clusters, 3)
clusterCut
```


<ol class=list-inline>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
</ol>



Now, let us compare it with the original species.


```R
table(clusterCut, iris$Species)
```


              
    clusterCut setosa versicolor virginica
             1     50          0         0
             2      0         21        50
             3      0         29         0


It looks like the algorithm successfully classified all the flowers of species setosa into cluster 1, and virginica into cluster 2, but had trouble with versicolor.

Let us see if we can better by using a different linkage method. This time, we will use the mean linkage method:


```R
clusters <- hclust(dist(iris[, 3:4]), method = 'average')
plot(clusters)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_35_0.png)



```R
clusterCut <- cutree(clusters, 3)
table(clusterCut, iris$Species)
```


              
    clusterCut setosa versicolor virginica
             1     50          0         0
             2      0         45         1
             3      0          5        49


We can see that this time, the algorithm did a much better job of clustering the data, only going wrong with 6 of the data points.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

### DBSCAN
The DBSCAN algorithm views **clusters as areas of high density** separated by areas of low density. Due to this rather generic view, clusters found by DBSCAN can be any shape, as opposed to k-means which assumes that clusters are convex shaped. 

The central component to the DBSCAN is the concept of core samples, which are samples that are in areas of high density. A cluster is therefore a set of core samples, each close to each other (measured by some distance measure) and a set of non-core samples that are close to a core sample (but are not themselves core samples). 

There are two parameters to the algorithm, **min_samples** and **eps**, which define formally what we mean when we say dense. Higher min_samples or lower eps indicate higher density necessary to form a cluster.

Any core sample is part of a cluster, by definition. Any sample that is not a core sample, and is at least eps in distance from any core sample, is considered an outlier by the algorithm. In the figure below, the color indicates cluster membership, with large circles indicating core samples found by the algorithm. Smaller circles are non-core samples that are still part of a cluster. Moreover, the outliers are indicated by black points below.

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/3.png)

The complete example is listed below.


```R
# Loading data 
data(iris)
```


```R
# Structure  
str(iris)
```

    'data.frame':	150 obs. of  5 variables:
     $ Sepal.Length: num  5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
     $ Sepal.Width : num  3.5 3 3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 ...
     $ Petal.Length: num  1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 ...
     $ Petal.Width : num  0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 ...
     $ Species     : Factor w/ 3 levels "setosa","versicolor",..: 1 1 1 1 1 1 1 1 1 1 ...
    

Performing DBScan on Dataset


```R
# Installing Packages 
install.packages("fpc") 
  
# Loading package 
library(fpc) 
  
# Remove label form dataset 
iris_1 <- iris[-5] 
  
# Fitting DBScan clustering Model  
# to training dataset 
set.seed(220)  # Setting seed 
Dbscan_cl <- dbscan(iris_1, eps = 0.45, MinPts = 5) 
Dbscan_cl
```

    package 'fpc' successfully unpacked and MD5 sums checked
    
    The downloaded binary packages are in
    	C:\Users\amit\AppData\Local\Temp\RtmpKOvSNR\downloaded_packages
    

    Warning message:
    "package 'fpc' was built under R version 3.6.3"


    dbscan Pts=150 MinPts=5 eps=0.45
            0  1  2
    border 24  4 13
    seed    0 44 65
    total  24 48 78



```R
# Checking cluster 
Dbscan_cl$cluster
```


<ol class=list-inline>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>0</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>0</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>0</li>
	<li>2</li>
	<li>2</li>
	<li>0</li>
	<li>2</li>
	<li>0</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>0</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>0</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>0</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>0</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>0</li>
	<li>2</li>
	<li>2</li>
	<li>0</li>
	<li>0</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>0</li>
	<li>2</li>
	<li>2</li>
	<li>0</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>0</li>
	<li>0</li>
	<li>0</li>
	<li>2</li>
	<li>2</li>
	<li>0</li>
	<li>0</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
</ol>




```R
# Table 
table(Dbscan_cl$cluster, iris$Species)
```


       
        setosa versicolor virginica
      0      2          7        15
      1     48          0         0
      2      0         43        35



```R
# Plotting Cluster 
# DBScan cluster is plotted with Sepal.Length, Sepal.Width, Petal.Length, Petal.Width.
plot(Dbscan_cl, iris_1, main = "DBScan") 
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_46_0.png)


.The DBSCAN algorithm is **deterministic, always generating the same clusters when given the same data in the same order**. However, the results can differ when data is provided in a different order. First, even though the core samples will always be assigned to the same clusters, the labels of those clusters will depend on the order in which those samples are encountered in the data. 

Second and more importantly, the clusters to which non-core samples are assigned can differ depending on the data order. This would happen when a non-core sample has a distance lower than eps to two core samples in different clusters. By the triangular inequality, those two core samples must be more distant than eps from each other, or they would be in the same cluster. The non-core sample is assigned to whichever cluster is generated first in a pass through the data, and so the results will depend on the data ordering.

This implementation is by default not memory efficient because it constructs a full pairwise similarity matrix in the case where kd-trees or ball-trees cannot be used (e.g., with sparse matrices). This matrix will consume n^2 floats. A couple of mechanisms for getting around this are:

OPTICS clustering in conjunction with the extract_dbscan method is more memory efficient. OPTICS clustering also calculates the full pairwise matrix, but only keeps one row in memory at a time (memory complexity n).

A sparse radius neighborhood graph (where missing entries are presumed to be out of eps) can be precomputed in a memory-efficient way and dbscan can be run over this with metric='precomputed'.

The dataset can be compressed, either by removing exact duplicates if these occur in the data, or by using BIRCH. This gives a relatively small number of representatives for a large number of points.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

## K-Means
K-Means Clustering may be the **most widely known clustering algorithm** and involves assigning examples to clusters in an effort to minimize the variance within each cluster.

The KMeans algorithm clusters data by trying to **separate samples in n groups of equal variance**, **minimizing** a criterion known as the **inertia or within-cluster sum-of-squares**. This algorithm requires the number of clusters to be specified. It scales well to large number of samples.



**K-means algorithm can be summarized as follow:**

1. Specify the number of clusters (K) to be created (by the analyst)
2. Select randomly k objects from the dataset as the initial cluster centers or means
3. Assigns each observation to their closest centroid, based on the Euclidean distance between the object and the centroid
4. For each of the k clusters update the cluster centroid by calculating the new mean values of all the data points in the cluster. The centoid of a Kth cluster is a vector of length p containing the means of all variables for the observations in the kth cluster; p is the number of variables.
5. Iteratively minimize the total within sum of square. That is, iterate steps 3 and 4 until the cluster assignments stop changing or the maximum number of iterations is reached. By default, the R software uses 10 as the default value for the maximum number of iterations.


```R
install.packages("rlang")
```

    package 'rlang' successfully unpacked and MD5 sums checked
    

    Warning message:
    "cannot remove prior installation of package 'rlang'"Warning message in file.copy(savedcopy, lib, recursive = TRUE):
    "problem copying C:\Users\amit\Anaconda3\envs\R practice\Lib\R\library\00LOCK\rlang\libs\x64\rlang.dll to C:\Users\amit\Anaconda3\envs\R practice\Lib\R\library\rlang\libs\x64\rlang.dll: Permission denied"Warning message:
    "restored 'rlang'"

    
    The downloaded binary packages are in
    	C:\Users\amit\AppData\Local\Temp\RtmpKuWwvm\downloaded_packages
    

We’ll use the demo data sets “USArrests”.


```R
# Loading the data set
data("USArrests")  

# Scaling the data
df <- scale(USArrests) 

# View the firt 3 rows of the data
head(df, n = 3)
```


<table>
<thead><tr><th></th><th scope=col>Murder</th><th scope=col>Assault</th><th scope=col>UrbanPop</th><th scope=col>Rape</th></tr></thead>
<tbody>
	<tr><th scope=row>Alabama</th><td>1.24256408  </td><td>0.7828393   </td><td>-0.5209066  </td><td>-0.003416473</td></tr>
	<tr><th scope=row>Alaska</th><td>0.50786248  </td><td>1.1068225   </td><td>-1.2117642  </td><td> 2.484202941</td></tr>
	<tr><th scope=row>Arizona</th><td>0.07163341  </td><td>1.4788032   </td><td> 0.9989801  </td><td> 1.042878388</td></tr>
</tbody>
</table>




```R
#Installing factoextra package as:
install.packages("factoextra")
```

    package 'factoextra' successfully unpacked and MD5 sums checked
    
    The downloaded binary packages are in
    	C:\Users\amit\AppData\Local\Temp\RtmpGws1LK\downloaded_packages
    


```R
#Loading factoextra:
library(factoextra)
```

    Warning message:
    "package 'factoextra' was built under R version 3.6.3"Loading required package: ggplot2
    Warning message:
    "package 'ggplot2' was built under R version 3.6.3"Welcome! Want to learn more? See two factoextra-related books at https://goo.gl/ve3WBa
    


```R
# Elbow method for to determine the number of clusters 
fviz_nbclust(df, kmeans, method = "wss") +
geom_vline(xintercept = 3, linetype = 2)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_57_0.png)



```R
# Compute k-means with k = 4
set.seed(123)
km.res <- kmeans(df, 4, nstart = 25)
```


```R
# Print the results
print(km.res)
```

    K-means clustering with 4 clusters of sizes 8, 13, 16, 13
    
    Cluster means:
          Murder    Assault   UrbanPop        Rape
    1  1.4118898  0.8743346 -0.8145211  0.01927104
    2 -0.9615407 -1.1066010 -0.9301069 -0.96676331
    3 -0.4894375 -0.3826001  0.5758298 -0.26165379
    4  0.6950701  1.0394414  0.7226370  1.27693964
    
    Clustering vector:
           Alabama         Alaska        Arizona       Arkansas     California 
                 1              4              4              1              4 
          Colorado    Connecticut       Delaware        Florida        Georgia 
                 4              3              3              4              1 
            Hawaii          Idaho       Illinois        Indiana           Iowa 
                 3              2              4              3              2 
            Kansas       Kentucky      Louisiana          Maine       Maryland 
                 3              2              1              2              4 
     Massachusetts       Michigan      Minnesota    Mississippi       Missouri 
                 3              4              2              1              4 
           Montana       Nebraska         Nevada  New Hampshire     New Jersey 
                 2              2              4              2              3 
        New Mexico       New York North Carolina   North Dakota           Ohio 
                 4              4              1              2              3 
          Oklahoma         Oregon   Pennsylvania   Rhode Island South Carolina 
                 3              3              3              3              1 
      South Dakota      Tennessee          Texas           Utah        Vermont 
                 2              1              4              3              2 
          Virginia     Washington  West Virginia      Wisconsin        Wyoming 
                 3              3              2              2              3 
    
    Within cluster sum of squares by cluster:
    [1]  8.316061 11.952463 16.212213 19.922437
     (between_SS / total_SS =  71.2 %)
    
    Available components:
    
    [1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
    [6] "betweenss"    "size"         "iter"         "ifault"      
    


```R
aggregate(USArrests, by=list(cluster=km.res$cluster), mean)
```


<table>
<thead><tr><th scope=col>cluster</th><th scope=col>Murder</th><th scope=col>Assault</th><th scope=col>UrbanPop</th><th scope=col>Rape</th></tr></thead>
<tbody>
	<tr><td>1        </td><td>13.93750 </td><td>243.62500</td><td>53.75000 </td><td>21.41250 </td></tr>
	<tr><td>2        </td><td> 3.60000 </td><td> 78.53846</td><td>52.07692 </td><td>12.17692 </td></tr>
	<tr><td>3        </td><td> 5.65625 </td><td>138.87500</td><td>73.87500 </td><td>18.78125 </td></tr>
	<tr><td>4        </td><td>10.81538 </td><td>257.38462</td><td>76.00000 </td><td>33.19231 </td></tr>
</tbody>
</table>




```R
dd <- cbind(USArrests, cluster = km.res$cluster)
head(dd)
```


<table>
<thead><tr><th></th><th scope=col>Murder</th><th scope=col>Assault</th><th scope=col>UrbanPop</th><th scope=col>Rape</th><th scope=col>cluster</th></tr></thead>
<tbody>
	<tr><th scope=row>Alabama</th><td>13.2</td><td>236 </td><td>58  </td><td>21.2</td><td>1   </td></tr>
	<tr><th scope=row>Alaska</th><td>10.0</td><td>263 </td><td>48  </td><td>44.5</td><td>4   </td></tr>
	<tr><th scope=row>Arizona</th><td> 8.1</td><td>294 </td><td>80  </td><td>31.0</td><td>4   </td></tr>
	<tr><th scope=row>Arkansas</th><td> 8.8</td><td>190 </td><td>50  </td><td>19.5</td><td>1   </td></tr>
	<tr><th scope=row>California</th><td> 9.0</td><td>276 </td><td>91  </td><td>40.6</td><td>4   </td></tr>
	<tr><th scope=row>Colorado</th><td> 7.9</td><td>204 </td><td>78  </td><td>38.7</td><td>4   </td></tr>
</tbody>
</table>




```R
#These components can be accessed as follow:

# Cluster number for each of the observations
km.res$cluster
```


<dl class=dl-horizontal>
	<dt>Alabama</dt>
		<dd>1</dd>
	<dt>Alaska</dt>
		<dd>4</dd>
	<dt>Arizona</dt>
		<dd>4</dd>
	<dt>Arkansas</dt>
		<dd>1</dd>
	<dt>California</dt>
		<dd>4</dd>
	<dt>Colorado</dt>
		<dd>4</dd>
	<dt>Connecticut</dt>
		<dd>3</dd>
	<dt>Delaware</dt>
		<dd>3</dd>
	<dt>Florida</dt>
		<dd>4</dd>
	<dt>Georgia</dt>
		<dd>1</dd>
	<dt>Hawaii</dt>
		<dd>3</dd>
	<dt>Idaho</dt>
		<dd>2</dd>
	<dt>Illinois</dt>
		<dd>4</dd>
	<dt>Indiana</dt>
		<dd>3</dd>
	<dt>Iowa</dt>
		<dd>2</dd>
	<dt>Kansas</dt>
		<dd>3</dd>
	<dt>Kentucky</dt>
		<dd>2</dd>
	<dt>Louisiana</dt>
		<dd>1</dd>
	<dt>Maine</dt>
		<dd>2</dd>
	<dt>Maryland</dt>
		<dd>4</dd>
	<dt>Massachusetts</dt>
		<dd>3</dd>
	<dt>Michigan</dt>
		<dd>4</dd>
	<dt>Minnesota</dt>
		<dd>2</dd>
	<dt>Mississippi</dt>
		<dd>1</dd>
	<dt>Missouri</dt>
		<dd>4</dd>
	<dt>Montana</dt>
		<dd>2</dd>
	<dt>Nebraska</dt>
		<dd>2</dd>
	<dt>Nevada</dt>
		<dd>4</dd>
	<dt>New Hampshire</dt>
		<dd>2</dd>
	<dt>New Jersey</dt>
		<dd>3</dd>
	<dt>New Mexico</dt>
		<dd>4</dd>
	<dt>New York</dt>
		<dd>4</dd>
	<dt>North Carolina</dt>
		<dd>1</dd>
	<dt>North Dakota</dt>
		<dd>2</dd>
	<dt>Ohio</dt>
		<dd>3</dd>
	<dt>Oklahoma</dt>
		<dd>3</dd>
	<dt>Oregon</dt>
		<dd>3</dd>
	<dt>Pennsylvania</dt>
		<dd>3</dd>
	<dt>Rhode Island</dt>
		<dd>3</dd>
	<dt>South Carolina</dt>
		<dd>1</dd>
	<dt>South Dakota</dt>
		<dd>2</dd>
	<dt>Tennessee</dt>
		<dd>1</dd>
	<dt>Texas</dt>
		<dd>4</dd>
	<dt>Utah</dt>
		<dd>3</dd>
	<dt>Vermont</dt>
		<dd>2</dd>
	<dt>Virginia</dt>
		<dd>3</dd>
	<dt>Washington</dt>
		<dd>3</dd>
	<dt>West Virginia</dt>
		<dd>2</dd>
	<dt>Wisconsin</dt>
		<dd>2</dd>
	<dt>Wyoming</dt>
		<dd>3</dd>
</dl>




```R
head(km.res$cluster, 4)
```


<dl class=dl-horizontal>
	<dt>Alabama</dt>
		<dd>1</dd>
	<dt>Alaska</dt>
		<dd>4</dd>
	<dt>Arizona</dt>
		<dd>4</dd>
	<dt>Arkansas</dt>
		<dd>1</dd>
</dl>




```R
# Cluster size
km.res$size
```


<ol class=list-inline>
	<li>8</li>
	<li>13</li>
	<li>16</li>
	<li>13</li>
</ol>




```R
# Cluster means
km.res$centers
```


<table>
<thead><tr><th scope=col>Murder</th><th scope=col>Assault</th><th scope=col>UrbanPop</th><th scope=col>Rape</th></tr></thead>
<tbody>
	<tr><td> 1.4118898 </td><td> 0.8743346 </td><td>-0.8145211 </td><td> 0.01927104</td></tr>
	<tr><td>-0.9615407 </td><td>-1.1066010 </td><td>-0.9301069 </td><td>-0.96676331</td></tr>
	<tr><td>-0.4894375 </td><td>-0.3826001 </td><td> 0.5758298 </td><td>-0.26165379</td></tr>
	<tr><td> 0.6950701 </td><td> 1.0394414 </td><td> 0.7226370 </td><td> 1.27693964</td></tr>
</tbody>
</table>




```R
# Visualize kmeans clustering
repel = TRUE
fviz_cluster(km.res, dd[, -5], ellipse.type = "norm")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_66_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

### Mini-Batch K-Means
Mini-Batch K-Means is a **modified version of k-means** that **makes updates to the cluster centroids using mini-batches of samples** rather than the entire dataset, which can make it faster for large datasets, and perhaps more robust to statistical noise. It reduces computation cost by orders of magnitude compared to the classic batch algorithm while yielding significantly better solutions than online stochastic gradient descent.

It is implemented via the MiniBatchKMeans class and the main configuration to tune is the “n_clusters” hyperparameter set to the estimated number of clusters in the data.

MiniBatchKMeans converges faster than KMeans, but the quality of the results is reduced. In practice this difference in quality can be quite small, as shown in the example and cited reference.

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/4.png)

The complete example is listed below.


```R
#installing package
install.packages("ClusterR")
```

    also installing the dependency 'gmp'
    
    
    

    package 'gmp' successfully unpacked and MD5 sums checked
    package 'ClusterR' successfully unpacked and MD5 sums checked
    
    The downloaded binary packages are in
    	C:\Users\sidhartha\AppData\Local\Temp\RtmpInT5gF\downloaded_packages
    


```R
library(ClusterR)
data(dietary_survey_IBS)
```

    Warning message:
    "package 'ClusterR' was built under R version 4.0.3"
    Loading required package: gtools
    
    


```R
params_mbkm = list(batch_size = 10, init_fraction = 0.3, early_stop_iter = 10)
```


```R
#Running MiniBatchKmeans with bacth size 20 and number of clusters 2
dat = dietary_survey_IBS[, -ncol(dietary_survey_IBS)]
dat = center_scale(dat)
MbatchKm = MiniBatchKmeans(dat, clusters = 2, batch_size = 20, num_init = 5, early_stop_iter = 10)
pr = predict_MBatchKMeans(dat, MbatchKm$centroids, fuzzy = FALSE)
```


```R
#indentifying the optimal number of cluster
opt_mbkm = Optimal_Clusters_KMeans(dat, max_clusters = 10, criterion = "distortion_fK",
plot_clusters = TRUE , mini_batch_params = params_mbkm)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_73_0.png)



```R
print(MbatchKm)
```

    $centroids
                [,1]       [,2]        [,3]       [,4]       [,5]        [,6]
    [1,] -0.56409733 -0.6021368 -0.03540576 -0.6895054 -0.5846429 -0.47645186
    [2,] -0.06886823  1.1376426  0.16118625  0.2678604  0.3719690 -0.04538045
               [,7]       [,8]       [,9]      [,10]      [,11]      [,12]
    [1,] -0.5984357 -0.6908949 -0.6908949 -0.7469200 -0.7074831 -0.4726476
    [2,]  0.6986602  0.9393802  0.8593108  0.6400135  0.9297550  1.2870261
              [,13]      [,14]      [,15]      [,16]      [,17]      [,18]
    [1,] -0.6921132 -0.6161957 -0.6941489 -0.5032040 -0.6908949 -0.6860449
    [2,]  0.9993712  1.0383063  1.0887595  0.1570629  0.3557043  0.1914889
              [,19]      [,20]      [,21]      [,22]      [,23]      [,24]
    [1,] -0.6667894 -0.7044828 -0.3715609 -0.5363581 -0.6921265 -0.3807551
    [2,]  0.8429355  0.7607037  0.4874801  0.1831353  1.4534991  0.8371739
              [,25]      [,26]      [,27]      [,28]       [,29]     [,30]
    [1,] -0.6908949 -0.6091218 -0.2382442 -0.6895054 -0.68604491 0.2429197
    [2,]  0.7300952  0.6450601  0.6077861  0.4847640 -0.02169388 0.5341924
              [,31]      [,32]      [,33]      [,34]      [,35]      [,36]
    [1,] -0.6992525 -0.6917698 -0.6895054 -0.5617559 -0.4743129 -0.5525849
    [2,]  0.8522499  1.7976849  0.8573702  0.1960639  0.3045455  0.3241364
              [,37]      [,38]      [,39]      [,40]       [,41]       [,42]
    [1,] -0.5177370 -0.3927000 -0.3401336 -0.6941489 -0.62757422 -0.60034258
    [2,]  0.1689377  0.5137204  0.4970929  0.3492672  0.04044678 -0.08803905
    
    $WCSS_per_cluster
             [,1]     [,2]
    [1,] 131.8175 265.5992
    
    $best_initialization
    [1] 4
    
    $iters_per_initialization
         [,1] [,2] [,3] [,4] [,5]
    [1,]   14   13   12   20   13
    
    attr(,"class")
    [1] "k-means clustering"
    


```R
print(pr)
```

      [1] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     [38] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
     [75] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
    [112] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
    [149] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
    [186] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    [223] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    [260] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    [297] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    [334] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    [371] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
    attr(,"class")
    [1] "k-means clustering"
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

### OPTICS
OPTICS clustering (where OPTICS is short for Ordering Points To Identify the Clustering Structure) is a modified version of DBSCAN described above. The OPTICS algorithm shares many similarities with the DBSCAN algorithm, and can be considered a generalization of DBSCAN that relaxes the eps requirement from a single value to a value range. 

The key difference between DBSCAN and OPTICS is that the OPTICS algorithm builds a reachability graph, which assigns each sample both a reachability_ distance, and a spot within the cluster ordering_ attribute; these two attributes are assigned when the model is fitted, and are used to determine cluster membership. 

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/5.png)

The **reachability distances generated by OPTICS allow for variable density extraction of clusters** within a single data set. As shown in the above plot, **combining reachability distances and data set ordering_ produces a reachability plot**, where point density is represented on the Y-axis, and points are ordered such that nearby points are adjacent. ‘Cutting’ the reachability plot at a single value produces DBSCAN like results; all points above the ‘cut’ are classified as noise, and each time that there is a break when reading from left to right signifies a new cluster. 

It is implemented via the OPTICS class and the main configuration to tune is the “eps” and “min_samples” hyperparameters.

The complete example is listed below.


```R
#Loading package
library(dbscan)
```


```R
#loading data
n <- 400
x <- cbind(
  x = runif(4, 0, 1) + rnorm(n, sd=0.1),
  y = runif(4, 0, 1) + rnorm(n, sd=0.1)
  )
```


```R
plot(x, col=rep(1:4, time = 100))
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_80_0.png)



```R
### run OPTICS (Note: we use the default eps calculation)
res <- optics(x, minPts = 10)
res
```


    OPTICS ordering/clustering for 400 objects.
    Parameters: minPts = 10, eps = 0.171044041996972, eps_cl = NA, xi = NA
    Available fields: order, reachdist, coredist, predecessor, minPts, eps,
                      eps_cl, xi



```R
### get order
res$order
```


<ol class=list-inline>
	<li>1</li>
	<li>393</li>
	<li>353</li>
	<li>337</li>
	<li>317</li>
	<li>241</li>
	<li>225</li>
	<li>157</li>
	<li>137</li>
	<li>117</li>
	<li>69</li>
	<li>37</li>
	<li>33</li>
	<li>345</li>
	<li>221</li>
	<li>101</li>
	<li>381</li>
	<li>233</li>
	<li>297</li>
	<li>265</li>
	<li>189</li>
	<li>145</li>
	<li>25</li>
	<li>9</li>
	<li>197</li>
	<li>281</li>
	<li>377</li>
	<li>329</li>
	<li>277</li>
	<li>209</li>
	<li>205</li>
	<li>201</li>
	<li>65</li>
	<li>53</li>
	<li>173</li>
	<li>93</li>
	<li>77</li>
	<li>325</li>
	<li>13</li>
	<li>177</li>
	<li>289</li>
	<li>285</li>
	<li>273</li>
	<li>213</li>
	<li>133</li>
	<li>109</li>
	<li>357</li>
	<li>169</li>
	<li>129</li>
	<li>365</li>
	<li>309</li>
	<li>193</li>
	<li>185</li>
	<li>181</li>
	<li>125</li>
	<li>121</li>
	<li>73</li>
	<li>61</li>
	<li>21</li>
	<li>81</li>
	<li>89</li>
	<li>45</li>
	<li>361</li>
	<li>341</li>
	<li>321</li>
	<li>269</li>
	<li>261</li>
	<li>105</li>
	<li>29</li>
	<li>369</li>
	<li>161</li>
	<li>165</li>
	<li>5</li>
	<li>17</li>
	<li>397</li>
	<li>385</li>
	<li>257</li>
	<li>97</li>
	<li>41</li>
	<li>153</li>
	<li>349</li>
	<li>305</li>
	<li>57</li>
	<li>49</li>
	<li>293</li>
	<li>149</li>
	<li>141</li>
	<li>301</li>
	<li>384</li>
	<li>372</li>
	<li>368</li>
	<li>352</li>
	<li>322</li>
	<li>300</li>
	<li>336</li>
	<li>292</li>
	<li>208</li>
	<li>398</li>
	<li>268</li>
	<li>196</li>
	<li>168</li>
	<li>156</li>
	<li>64</li>
	<li>56</li>
	<li>28</li>
	<li>382</li>
	<li>312</li>
	<li>270</li>
	<li>60</li>
	<li>296</li>
	<li>232</li>
	<li>52</li>
	<li>48</li>
	<li>42</li>
	<li>20</li>
	<li>100</li>
	<li>63</li>
	<li>164</li>
	<li>400</li>
	<li>266</li>
	<li>186</li>
	<li>142</li>
	<li>14</li>
	<li>274</li>
	<li>172</li>
	<li>112</li>
	<li>8</li>
	<li>216</li>
	<li>46</li>
	<li>386</li>
	<li>320</li>
	<li>288</li>
	<li>256</li>
	<li>122</li>
	<li>304</li>
	<li>324</li>
	<li>350</li>
	<li>318</li>
	<li>355</li>
	<li>259</li>
	<li>211</li>
	<li>160</li>
	<li>140</li>
	<li>255</li>
	<li>126</li>
	<li>111</li>
	<li>59</li>
	<li>290</li>
	<li>267</li>
	<li>194</li>
	<li>280</li>
	<li>294</li>
	<li>184</li>
	<li>130</li>
	<li>128</li>
	<li>119</li>
	<li>94</li>
	<li>371</li>
	<li>370</li>
	<li>183</li>
	<li>328</li>
	<li>179</li>
	<li>158</li>
	<li>118</li>
	<li>86</li>
	<li>74</li>
	<li>18</li>
	<li>50</li>
	<li>70</li>
	<li>346</li>
	<li>263</li>
	<li>162</li>
	<li>51</li>
	<li>235</li>
	<li>358</li>
	<li>315</li>
	<li>231</li>
	<li>135</li>
	<li>79</li>
	<li>334</li>
	<li>146</li>
	<li>114</li>
	<li>123</li>
	<li>62</li>
	<li>262</li>
	<li>54</li>
	<li>22</li>
	<li>171</li>
	<li>134</li>
	<li>150</li>
	<li>399</li>
	<li>343</li>
	<li>314</li>
	<li>260</li>
	<li>224</li>
	<li>180</li>
	<li>159</li>
	<li>275</li>
	<li>282</li>
	<li>246</li>
	<li>238</li>
	<li>331</li>
	<li>338</li>
	<li>243</li>
	<li>178</li>
	<li>174</li>
	<li>102</li>
	<li>88</li>
	<li>71</li>
	<li>27</li>
	<li>139</li>
	<li>40</li>
	<li>367</li>
	<li>264</li>
	<li>215</li>
	<li>98</li>
	<li>218</li>
	<li>36</li>
	<li>203</li>
	<li>223</li>
	<li>219</li>
	<li>34</li>
	<li>84</li>
	<li>106</li>
	<li>99</li>
	<li>250</li>
	<li>351</li>
	<li>44</li>
	<li>188</li>
	<li>2</li>
	<li>234</li>
	<li>80</li>
	<li>68</li>
	<li>6</li>
	<li>326</li>
	<li>310</li>
	<li>258</li>
	<li>242</li>
	<li>24</li>
	<li>4</li>
	<li>340</li>
	<li>200</li>
	<li>202</li>
	<li>182</li>
	<li>90</li>
	<li>376</li>
	<li>251</li>
	<li>311</li>
	<li>279</li>
	<li>240</li>
	<li>147</li>
	<li>82</li>
	<li>66</li>
	<li>43</li>
	<li>387</li>
	<li>335</li>
	<li>207</li>
	<li>287</li>
	<li>210</li>
	<li>11</li>
	<li>163</li>
	<li>155</li>
	<li>226</li>
	<li>379</li>
	<li>319</li>
	<li>190</li>
	<li>67</li>
	<li>55</li>
	<li>170</li>
	<li>23</li>
	<li>191</li>
	<li>19</li>
	<li>7</li>
	<li>291</li>
	<li>239</li>
	<li>39</li>
	<li>247</li>
	<li>308</li>
	<li>252</li>
	<li>236</li>
	<li>212</li>
	<li>392</li>
	<li>344</li>
	<li>26</li>
	<li>30</li>
	<li>154</li>
	<li>16</li>
	<li>348</li>
	<li>228</li>
	<li>220</li>
	<li>396</li>
	<li>108</li>
	<li>244</li>
	<li>330</li>
	<li>276</li>
	<li>187</li>
	<li>284</li>
	<li>176</li>
	<li>332</li>
	<li>104</li>
	<li>307</li>
	<li>3</li>
	<li>35</li>
	<li>354</li>
	<li>143</li>
	<li>32</li>
	<li>391</li>
	<li>214</li>
	<li>204</li>
	<li>222</li>
	<li>394</li>
	<li>92</li>
	<li>375</li>
	<li>110</li>
	<li>364</li>
	<li>76</li>
	<li>316</li>
	<li>230</li>
	<li>356</li>
	<li>278</li>
	<li>144</li>
	<li>124</li>
	<li>116</li>
	<li>78</li>
	<li>38</li>
	<li>198</li>
	<li>138</li>
	<li>286</li>
	<li>10</li>
	<li>166</li>
	<li>72</li>
	<li>360</li>
	<li>248</li>
	<li>120</li>
	<li>96</li>
	<li>254</li>
	<li>192</li>
	<li>339</li>
	<li>323</li>
	<li>302</li>
	<li>390</li>
	<li>12</li>
	<li>362</li>
	<li>175</li>
	<li>299</li>
	<li>195</li>
	<li>327</li>
	<li>303</li>
	<li>115</li>
	<li>91</li>
	<li>83</li>
	<li>15</li>
	<li>167</li>
	<li>366</li>
	<li>342</li>
	<li>206</li>
	<li>298</li>
	<li>295</li>
	<li>271</li>
	<li>95</li>
	<li>131</li>
	<li>75</li>
	<li>87</li>
	<li>152</li>
	<li>313</li>
	<li>148</li>
	<li>132</li>
	<li>47</li>
	<li>374</li>
	<li>245</li>
	<li>151</li>
	<li>103</li>
	<li>136</li>
	<li>107</li>
	<li>272</li>
	<li>283</li>
	<li>395</li>
	<li>347</li>
	<li>378</li>
	<li>359</li>
	<li>306</li>
	<li>31</li>
	<li>58</li>
	<li>113</li>
	<li>199</li>
	<li>227</li>
	<li>229</li>
	<li>127</li>
	<li>249</li>
	<li>383</li>
	<li>373</li>
	<li>380</li>
	<li>388</li>
	<li>253</li>
	<li>389</li>
	<li>363</li>
	<li>217</li>
	<li>85</li>
	<li>237</li>
	<li>333</li>
</ol>




```R
### plot produces a reachability plot
plot(res)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_83_0.png)



```R
### plot the order of points in the reachability plot
plot(x, col = "grey")
polygon(x[res$order,])
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_84_0.png)



```R
### extract a DBSCAN clustering by cutting the reachability plot at eps_cl
res <- extractDBSCAN(res, eps_cl = .065)
res
```


    OPTICS ordering/clustering for 400 objects.
    Parameters: minPts = 10, eps = 0.171044041996972, eps_cl = 0.065, xi = NA
    The clustering contains 2 cluster(s) and 32 noise points.
    
      0   1   2 
     32  84 284 
    
    Available fields: order, reachdist, coredist, predecessor, minPts, eps,
                      eps_cl, xi, cluster



```R
plot(res)  ## black is noise
hullplot(x, res)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_86_0.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_86_1.png)



```R
### re-cut at a higher eps threshold
res <- extractDBSCAN(res, eps_cl = .1)
res
plot(res)
hullplot(x, res)
```


    OPTICS ordering/clustering for 400 objects.
    Parameters: minPts = 10, eps = 0.171044041996972, eps_cl = 0.1, xi = NA
    The clustering contains 1 cluster(s) and 6 noise points.
    
      0   1 
      6 394 
    
    Available fields: order, reachdist, coredist, predecessor, minPts, eps,
                      eps_cl, xi, cluster



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_87_1.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_87_2.png)



```R
### extract hierarchical clustering of varying density using the Xi method
res <- extractXi(res, xi = 0.05)
res
```


    OPTICS ordering/clustering for 400 objects.
    Parameters: minPts = 10, eps = 0.171044041996972, eps_cl = NA, xi = 0.05
    The clustering contains 4 cluster(s) and 0 noise points.
    
    Available fields: order, reachdist, coredist, predecessor, minPts, eps,
                      eps_cl, xi, cluster, clusters_xi



```R
plot(res)
hullplot(x, res)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_89_0.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_89_1.png)



```R
# Xi cluster structure
res$clusters_xi
```


<table>
<thead><tr><th scope=col>start</th><th scope=col>end</th><th scope=col>cluster_id</th></tr></thead>
<tbody>
	<tr><td>  1</td><td>400</td><td>1  </td></tr>
	<tr><td> 49</td><td> 70</td><td>2  </td></tr>
	<tr><td>158</td><td>171</td><td>3  </td></tr>
	<tr><td>198</td><td>218</td><td>4  </td></tr>
</tbody>
</table>




```R
### use OPTICS on a precomputed distance matrix
d <- dist(x)
res <- optics(d, minPts = 10)
plot(res)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_91_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

### Spectral Clustering
Spectral Clustering is a general class of clustering methods, drawn from linear algebra. Spectral clustering is a technique with roots in graph theory, where the approach is used to identify communities of nodes in a graph based on the edges connecting them.

Spectral clustering uses information from the eigenvalues (spectrum) of special matrices built from the graph or the data set.

It treats each data point as a graph-node and thus transforms the clustering problem into a graph-partitioning problem.

We don’t cluster data points directly in their native data space but instead form a similarity matrix where the (i,j)th entry is some similarity distance you define between the  ith and  jth data points in your dataset.



```R
#loading data
data(iris)
```


```R
head(iris)
```


<table>
<thead><tr><th scope=col>Sepal.Length</th><th scope=col>Sepal.Width</th><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th><th scope=col>Species</th></tr></thead>
<tbody>
	<tr><td>5.1   </td><td>3.5   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.9   </td><td>3.0   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.7   </td><td>3.2   </td><td>1.3   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.6   </td><td>3.1   </td><td>1.5   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.0   </td><td>3.6   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.4   </td><td>3.9   </td><td>1.7   </td><td>0.4   </td><td>setosa</td></tr>
</tbody>
</table>




```R
#installing package
install.packages("kknn")
```

    package 'kknn' successfully unpacked and MD5 sums checked
    
    The downloaded binary packages are in
    	C:\Users\amit\AppData\Local\Temp\Rtmpkb78Lt\downloaded_packages
    


```R
library(kknn)
```

    Warning message:
    "package 'kknn' was built under R version 3.6.3"


```R
cl <- specClust(iris[,1:4], 3, nn=5)
pcol <- as.character(as.numeric(iris$Species))
pairs(iris[1:4], pch = pcol, col = c("green", "red", "blue")[cl$cluster])
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_98_0.png)



```R
table(iris[,5], cl$cluster)
```


                
                  1  2  3
      setosa     50  0  0
      versicolor  0 48  2
      virginica   0  4 46


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

### Gaussian Mixture Model
A Gaussian mixture model **summarizes a multivariate probability density function** with a mixture of Gaussian probability distributions as its name suggests.

A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. The GaussianMixture object implements the expectation-maximization (EM) algorithm for fitting mixture-of-Gaussian models. One can think of mixture models as generalizing k-means clustering to incorporate information about the covariance structure of the data as well as the centers of the latent Gaussians.

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/6.png)

It is implemented via the GaussianMixture class and the main configuration to tune is the “n_clusters” hyperparameter used to specify the estimated number of clusters in the data.

It is the **fastest algorithm for learning mixture models**. As this algorithm maximizes only the likelihood, it will not bias the means towards zero, or bias the cluster sizes to have specific structures that might or might not apply.
When one has insufficiently many points per mixture, estimating the covariance matrices becomes difficult, and the algorithm is known to diverge and find solutions with infinite likelihood unless one regularizes the covariances artificially. This algorithm will always use all the components it has access to, needing held-out data or information theoretical criteria to decide how many components to use in the absence of external cues.



#### Implementation

The galaxies data in the MASS package is a frequently used example for
Gaussian mixture models. It contains the velocities of 82 galaxies from a redshift survey in the Corona
Borealis region. Clustering of galaxy velocities reveals information about the large scale structure of the
universe.


```R
library(MASS)
data(galaxies)
X = galaxies / 1000
print(X)
```

     [1]  9.172  9.350  9.483  9.558  9.775 10.227 10.406 16.084 16.170 18.419
    [11] 18.552 18.600 18.927 19.052 19.070 19.330 19.343 19.349 19.440 19.473
    [21] 19.529 19.541 19.547 19.663 19.846 19.856 19.863 19.914 19.918 19.973
    [31] 19.989 20.166 20.175 20.179 20.196 20.215 20.221 20.415 20.629 20.795
    [41] 20.821 20.846 20.875 20.986 21.137 21.492 21.701 21.814 21.921 21.960
    [51] 22.185 22.209 22.242 22.249 22.314 22.374 22.495 22.746 22.747 22.888
    [61] 22.914 23.206 23.241 23.263 23.484 23.538 23.542 23.666 23.706 23.711
    [71] 24.129 24.285 24.289 24.366 24.717 24.990 25.633 26.690 26.995 32.065
    [81] 32.789 34.279
    

The Mclust function from the mclust package (Fraley et al, 2012) is used to fit Gaussian mixture models.
The code below fits a model with G=4 components to the galaxies data, allowing the variances to be unequal
(model="V").



```R
library(mclust, quietly=TRUE)
```


```R
fit = Mclust(X, G=4, model="V")
summary(fit)
```


    ---------------------------------------------------- 
    Gaussian finite mixture model fitted by EM algorithm 
    ---------------------------------------------------- 
    
    Mclust V (univariate, unequal variance) model with 4 components: 
    
     log-likelihood  n df       BIC      ICL
          -199.2545 82 11 -446.9829 -466.264
    
    Clustering table:
     1  2  3  4 
     7 35 32  8 



```R
plot(fit, what="density", main="", xlab="Velocity (Mm/s)")
rug(X)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_107_0.png)


$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$ <b>Figure 1: Density estimate for galaxies data from a 4-component mixture model

Section 6.2 of Drton and Plummer (2017) considers singular BIC for Gaussian mixture models using the
galaxies data set as an example. Singularities occur when two mixture components coincide (i.e. they have
the same mean and variance) or on the boundary of the parameter space where the prior probability of a
mixture component is zero.
<br>The GaussianMixtures() function creates an object representing a family of mixture models up to a specified
maximum number of components (maxNumComponents=10 in this example). The phi parameter controls the
penalty to be used for sBIC (See below) and the restarts parameter determines the number of times each
model is fitted starting from randomly chosen starting points. Due to the multi-modal likelihood surface for
mixture models, multiple restarts are used to find a good (local) maximum.



```R
library(sBIC)
gMix = GaussianMixtures(maxNumComponents=10, phi=1, restarts=100)
```

Learning coefficients are known exactly for Gaussian mixtures with known and equal variances, but this
model is rarely applied in practice. For unequal variances, the learning coefficients are unknown, but upper
bounds are given by Drton and Plummer (2017, equation 6.11). These bounds are implemented by setting
the penalty parameter phi=1 in the GaussianMixtures() function. We refer to the singular BIC using these
approximate penalties as sBIC1. It is calculated by supplying the data X and the model set gMix to the
sBIC() function. The RNG seed is set for reproducibility, due to the random restarts.


```R
set.seed(1234)
m = sBIC(X, gMix)
print(m)
```

    $logLike
     [1] -240.3379 -220.2445 -203.1792 -197.4621 -190.0724 -186.8674 -185.8390
     [8] -186.7764 -184.0937 -185.8294
    
    $sBIC
     [1] -244.7446 -231.2612 -220.8038 -219.0979 -216.4564 -216.2255 -217.5358
     [8] -220.6820 -220.2114 -224.1505
    
    $BIC
     [1] -244.7446 -231.2613 -220.8061 -221.6990 -220.9195 -224.3245 -229.9061
     [8] -237.4537 -241.3811 -249.7268
    
    $modelPoset
    [1] "GaussianMixtures: 0x000000001a68e4b0"
    
    


```R
matplot(
cbind(m$BIC - m$BIC[1], m$sBIC - m$sBIC[1]),
pch = c(1, 3),
col = "black",
xlab = "Number of components",
ylab = expression(BIC - BIC(M[1])),
las=1, xaxt="n"
)
axis(1, at = 1:10)
legend("topleft",
c(expression(BIC), expression(bar(sBIC)[1])),
pch = c(1, 3),
y.intersp = 1.2)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_113_0.png)


$\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;$  <b>Figure 2: Comparison of singular BIC with BIC for choosing the number of components in the galaxies data

Figure 2 compares BIC with sBIC1. Both criteria have been standardized so that the value for the 1-component
model is 0. This figures reproduces Figure 7 of Drton and Plummer (2017). The reproduction is not exact
because, in the interests of speed, we have reduced the number of restarts from 5000 to 100. This mainly
affects the models with larger number of components.


```R
post.MCMC = c(0.000, 0.000, 0.061, 0.128, 0.182, 0.199, 0.160,
0.109, 0.071, 0.040, 0.023, 0.013, 0.006, 0.003)[1:10]
post.MCMC = post.MCMC / sum(post.MCMC)
```

The posterior probabilities from BIC and sBIC1 are derived by exponentiating and then renormalizing using
the helper function postBIC().


```R
postBIC <- function(BIC) {
prob <- exp(BIC - max(BIC))
prob/sum(prob)
}
normalizedProbs = rbind("BIC"=postBIC(m$BIC), "sBIC1"=postBIC(m$sBIC), "MCMC"=post.MCMC)
```


```R
barplot(
normalizedProbs,
beside = TRUE,
col = c("white","grey","black"),
legend = c(expression(BIC), expression(bar(sBIC)[1]), expression(MCMC)),
xlab = "Number of components",
ylab = "Probability",
args.legend = list(y.intersp = 1.2),
names.arg = 1:10
)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_119_0.png)


$\;\;\;\;\;\;\;\;\;$<b>Figure3.Posterior distribution of the number of components in a Gaussian mixture model with unequal
variances applied to the galaxies data


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

# K-Mode

The basic concept of k-means stands on mathematical calculations (means, euclidian distances). But what if our data is non-numerical or, in other words, categorical? 
We could think of transforming our categorical values in numerical values and eventually apply k-means. But beware: k-means uses numerical distances, so it could consider close two really distant objects that merely have been assigned two close numbers.k-modes is an extension of k-means. Instead of distances it uses dissimilarities (that is, quantification of the total mismatches between two objects: the smaller this number, the more similar the two objects). And instead of means, it uses modes. A mode is a vector of elements that minimizes the dissimilarities between the vector itself and each object of the data. We will have as many modes as the number of clusters we required, since they act as centroids.

K-modes clustering implementation on categorical data


```R
install.packages("klaR")
```

    also installing the dependencies 'R.utils', 'R.cache', 'rematch2', 'e1071', 'haven', 'miniUI', 'styler', 'classInt', 'labelled', 'questionr'
    
    

    package 'R.utils' successfully unpacked and MD5 sums checked
    package 'R.cache' successfully unpacked and MD5 sums checked
    package 'rematch2' successfully unpacked and MD5 sums checked
    package 'e1071' successfully unpacked and MD5 sums checked
    package 'haven' successfully unpacked and MD5 sums checked
    package 'miniUI' successfully unpacked and MD5 sums checked
    package 'styler' successfully unpacked and MD5 sums checked
    package 'classInt' successfully unpacked and MD5 sums checked
    package 'labelled' successfully unpacked and MD5 sums checked
    package 'questionr' successfully unpacked and MD5 sums checked
    package 'klaR' successfully unpacked and MD5 sums checked
    
    The downloaded binary packages are in
    	C:\Users\amit\AppData\Local\Temp\RtmpOmtb4o\downloaded_packages
    


```R
library(klaR)
```

    Warning message:
    "package 'klaR' was built under R version 3.6.3"Loading required package: MASS
    


```R
## generate data set with two groups of data:
set.seed(1)
x <- rbind(matrix(rbinom(250, 2, 0.25), ncol = 5),
           matrix(rbinom(250, 2, 0.75), ncol = 5))
colnames(x) <- c("a", "b", "c", "d", "e")
```


```R
print(x)
```

           a b c d e
      [1,] 0 0 1 1 0
      [2,] 0 1 0 0 0
      [3,] 1 0 0 0 0
      [4,] 1 0 2 0 0
      [5,] 0 0 1 0 0
      [6,] 1 0 0 0 0
      [7,] 2 0 0 0 1
      [8,] 1 0 0 0 0
      [9,] 1 1 1 0 0
     [10,] 0 0 1 0 1
     [11,] 0 1 2 0 2
     [12,] 0 0 1 1 0
     [13,] 1 0 0 0 1
     [14,] 0 0 0 1 2
     [15,] 1 1 0 1 1
     [16,] 0 0 0 0 0
     [17,] 1 0 1 0 1
     [18,] 2 1 0 0 2
     [19,] 0 0 0 1 2
     [20,] 1 1 1 0 0
     [21,] 1 0 2 1 0
     [22,] 0 1 0 1 0
     [23,] 1 0 0 1 0
     [24,] 0 0 0 0 0
     [25,] 0 0 1 0 1
     [26,] 0 1 0 1 0
     [27,] 0 1 0 1 0
     [28,] 0 0 0 1 0
     [29,] 1 1 0 1 0
     [30,] 0 2 1 1 1
     [31,] 0 0 1 0 0
     [32,] 1 1 0 0 0
     [33,] 0 0 0 1 0
     [34,] 0 0 1 0 1
     [35,] 1 1 1 1 0
     [36,] 1 0 1 0 1
     [37,] 1 1 0 1 1
     [38,] 0 0 0 1 0
     [39,] 1 0 2 2 0
     [40,] 0 0 0 0 0
     [41,] 1 0 1 1 0
     [42,] 1 0 1 0 1
     [43,] 1 1 0 0 1
     [44,] 0 1 0 1 0
     [45,] 0 1 1 0 0
     [46,] 1 1 0 1 0
     [47,] 0 0 0 0 0
     [48,] 0 0 1 1 0
     [49,] 1 1 0 0 0
     [50,] 1 1 1 1 2
     [51,] 1 1 2 1 0
     [52,] 1 2 2 2 2
     [53,] 2 2 1 0 2
     [54,] 1 2 2 1 2
     [55,] 2 2 0 0 2
     [56,] 2 0 2 1 1
     [57,] 2 2 1 2 1
     [58,] 2 1 2 1 0
     [59,] 2 2 1 2 1
     [60,] 1 2 2 0 2
     [61,] 2 2 2 0 2
     [62,] 2 2 2 2 2
     [63,] 2 1 1 1 2
     [64,] 1 1 2 1 2
     [65,] 1 2 1 1 2
     [66,] 2 2 1 2 0
     [67,] 2 2 2 2 1
     [68,] 1 2 1 2 2
     [69,] 1 0 1 1 2
     [70,] 2 1 2 1 1
     [71,] 2 1 2 2 2
     [72,] 2 2 2 2 0
     [73,] 2 2 1 1 1
     [74,] 2 0 2 2 2
     [75,] 2 1 2 2 2
     [76,] 2 0 2 2 2
     [77,] 2 1 2 2 1
     [78,] 2 1 1 1 2
     [79,] 2 2 2 1 2
     [80,] 1 1 2 1 1
     [81,] 2 0 2 1 1
     [82,] 2 2 1 2 2
     [83,] 1 2 1 2 1
     [84,] 2 1 0 2 0
     [85,] 2 2 2 2 1
     [86,] 2 2 2 1 2
     [87,] 2 2 1 1 2
     [88,] 2 2 1 2 2
     [89,] 2 1 2 2 1
     [90,] 2 0 0 1 2
     [91,] 2 0 2 2 2
     [92,] 2 2 2 2 0
     [93,] 0 2 1 2 2
     [94,] 2 2 2 1 2
     [95,] 2 1 2 2 2
     [96,] 1 2 2 2 2
     [97,] 2 2 1 2 2
     [98,] 2 2 2 0 2
     [99,] 2 2 2 2 1
    [100,] 1 1 2 2 2
    


```R
## Run K mode on x:
(cl <- kmodes(x, 2))
```


    K-modes clustering with 2 clusters of sizes 53, 47
    
    Cluster modes:
      a b c d e
    1 1 0 0 0 0
    2 2 2 2 2 2
    
    Clustering vector:
      [1] 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1
     [38] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2
     [75] 2 2 2 2 2 1 2 2 2 1 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2
    
    Within cluster simple-matching distance by cluster:
    [1] 116  84
    
    Available components:
    [1] "cluster"    "size"       "modes"      "withindiff" "iterations"
    [6] "weighted"  



```R
cl$cluster
```


<ol class=list-inline>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>2</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>2</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>1</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>1</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>1</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>1</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
</ol>




```R
##  visualize with some jitter:
plot(jitter(x), col = cl$cluster)
points(cl$modes, col = 1:5, pch = 8)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_131_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

# K-Prototype

Just like K — means where we allocate the record to the closest centroid (a reference point to the cluster), here we allocate the record to the cluster which has the most similar looking reference point a.k.a prototype of the cluster a.k.a centroid of the cluster.

More than similarity the algorithms try to find the dissimilarity between data points and try to group points with less dissimilarity into a cluster
.
The dissimilarity measure for numeric attributes is the square Euclidean distance whereas the similarity measure on categorical attributes is the number of matching attributes between objects and cluster prototypes.


```R
#installing package 
install.packages("clustMixType")
```

    package 'clustMixType' successfully unpacked and MD5 sums checked
    
    The downloaded binary packages are in
    	C:\Users\amit\AppData\Local\Temp\RtmpGo9GTP\downloaded_packages
    


```R
library(clustMixType)
```

    Warning message:
    "package 'clustMixType' was built under R version 3.6.3"


```R
# generate toy data with factors and numerics

n   <- 100
prb <- 0.9
muk <- 1.5 
clusid <- rep(1:4, each = n)

x1 <- sample(c("A","B"), 2*n, replace = TRUE, prob = c(prb, 1-prb))
x1 <- c(x1, sample(c("A","B"), 2*n, replace = TRUE, prob = c(1-prb, prb)))
x1 <- as.factor(x1)

x2 <- sample(c("A","B"), 2*n, replace = TRUE, prob = c(prb, 1-prb))
x2 <- c(x2, sample(c("A","B"), 2*n, replace = TRUE, prob = c(1-prb, prb)))
x2 <- as.factor(x2)

x3 <- c(rnorm(n, mean = -muk), rnorm(n, mean = muk), rnorm(n, mean = -muk), rnorm(n, mean = muk))
x4 <- c(rnorm(n, mean = -muk), rnorm(n, mean = muk), rnorm(n, mean = -muk), rnorm(n, mean = muk))

x <- data.frame(x1,x2,x3,x4)
```


```R
print(x)
```

        x1 x2           x3            x4
    1    A  A -0.521360498 -1.4281026147
    2    A  A -1.162025535 -1.3590592653
    3    A  A  0.232753140 -1.6282661844
    4    A  A -0.799464641 -2.3714210524
    5    A  A -0.443648070 -1.2866200357
    6    A  A -1.067972708 -1.9075549461
    7    B  B -2.714482906 -2.0629742398
    8    A  A -2.458724429 -2.0938778526
    9    A  A -0.005504827 -3.9762607411
    10   B  A -2.480787592 -1.5102295130
    11   A  A -0.760390946 -0.9810155844
    12   A  A -3.284994658 -0.1378991786
    13   A  A -0.648339497 -1.3092143972
    14   A  A -2.547054765 -0.2494424100
    15   B  A -2.019913977 -0.7253254377
    16   A  B -1.033189164 -0.1636846824
    17   A  A -0.994836372 -3.4329769393
    18   A  A -0.096591477 -0.9647562027
    19   A  A -0.939563423 -3.6808733597
    20   A  A -0.942573701 -1.7993502445
    21   A  A -2.409282689 -1.2357051887
    22   A  A -0.880463787 -1.0254061074
    23   A  A -1.125263144 -0.7985519787
    24   A  A -2.208764069 -2.0199540531
    25   A  A  0.750710819 -0.8651810004
    26   A  A -0.437069309 -0.5178935316
    27   A  A -3.393687115 -0.0353609514
    28   A  A -2.200384697 -2.2225416797
    29   A  A -2.214052901 -2.1601964005
    30   A  A -1.938916114 -0.9253620964
    31   A  A -1.033900281 -2.1535513770
    32   A  A -2.479991606 -1.4729583766
    33   A  A -1.853035482 -2.5162158182
    34   A  B -0.570691144 -2.1224724210
    35   A  B  0.621010335 -1.5891168420
    36   A  B -2.406803310 -1.0606765981
    37   A  A -2.079443979 -0.3597824841
    38   A  A -1.153926215 -0.9539480860
    39   A  A -0.888421183 -1.4373755104
    40   A  A -1.070462890 -1.0774520364
    41   A  A -0.920287946 -0.7131376328
    42   A  A -1.298052245  0.0185542763
    43   A  A  0.605831778 -0.4887189824
    44   A  A -1.433854807 -2.1015094482
    45   A  B -1.437507148 -1.0619710538
    46   A  A -2.662667647 -1.6053657728
    47   A  A -2.003255865 -1.7423083913
    48   A  A -0.962907623 -2.3766726505
    49   A  A -2.317495402 -1.6213048469
    50   A  A -2.209708744 -2.8984939122
    51   A  A -1.442131992 -1.2068855336
    52   A  A -0.760295883  0.2771235360
    53   A  A -1.736922255 -2.1024489442
    54   A  A -2.239302158 -1.3311687273
    55   A  A -2.115345263 -2.1749420177
    56   A  A -2.931959002 -0.6302757655
    57   A  A -2.225988976 -1.5987650201
    58   A  A -0.695173130 -0.3832589739
    59   A  A -0.260974945  0.0225569177
    60   A  A -1.733116565 -2.0216630241
    61   A  A -0.385966719 -2.3888779668
    62   A  A -2.291748024 -0.4506125213
    63   A  A -1.472579512 -1.1855717593
    64   A  A -2.173988538 -0.5151298308
    65   A  A -1.647832599 -1.4634640032
    66   A  A -1.788630874 -1.9900438034
    67   A  A -0.869399446 -1.4718148018
    68   A  A -1.533090186 -1.2749421583
    69   A  A -3.676107226 -2.4655899200
    70   A  A -2.121627510 -2.2088435818
    71   B  A -2.154528815 -1.6357277202
    72   A  A  0.121738950 -0.6587025441
    73   A  A -1.161694187 -2.4434276167
    74   A  A  0.495565412 -2.6541323330
    75   A  A -1.376832664 -1.5288149423
    76   A  A -0.837906970 -2.0601422310
    77   A  A -0.404118540 -1.5407009116
    78   A  A  0.216173536 -2.2887419167
    79   A  A -0.562480116 -1.1352326412
    80   A  A -0.321377631 -2.3272504184
    81   A  A -0.799157647 -2.2288299907
    82   A  A -2.338677120 -1.3145316619
    83   A  A -0.280475877 -2.3715906520
    84   A  A -1.898227242 -2.3610173179
    85   A  A -2.273934958 -1.3732924612
    86   B  A -0.766861044 -2.0747701001
    87   A  A -1.030082878 -3.8868936360
    88   A  A -1.534567909 -1.1023324032
    89   A  A  0.996243287 -2.3604419716
    90   A  B -2.816497058  0.0048283355
    91   A  A -2.702868280 -0.2587337323
    92   A  B  1.135494094 -1.9322850579
    93   A  A -2.876423068 -2.6343417990
    94   A  A -1.463860577 -2.1441677906
    95   B  A -2.739775063 -0.7326324207
    96   A  A -3.284809431 -1.8471146292
    97   A  A  1.475048746 -2.7603843642
    98   A  A  1.072431613 -1.7484057172
    99   A  B -2.669873428 -1.9316132330
    100  A  A -2.048408449 -1.7228092024
    101  A  A  2.153857902 -0.9643759467
    102  A  A  0.497318729  1.6644507171
    103  A  A  2.555170508  1.2368941709
    104  B  A  1.811733016  0.8410747258
    105  A  A  1.302921660  1.7092984021
    106  A  A  2.066309764  0.8552134560
    107  A  A  1.317651653  1.4262007290
    108  A  A -0.804136791  2.0054537961
    109  A  A  1.079038302  2.2129570599
    110  A  A  1.691482550  0.3795185717
    111  A  A -0.087845839  1.8329249032
    112  A  B  2.070920173  2.6310754650
    113  A  A  1.618782704  2.3865104304
    114  A  A  0.682448414  3.2348089851
    115  A  A  2.327474053  3.1936359805
    116  A  B  3.768326934  0.7561561295
    117  A  A  3.160128802  2.9582708231
    118  A  A  2.059921934  0.5607898721
    119  B  A  1.307391902  0.8145325942
    120  A  A  1.147662394  0.8797562435
    121  A  A -0.102312390  0.2090047352
    122  A  A  1.892474644  0.6670476580
    123  A  A  0.981221596  1.1283720293
    124  B  B  1.582789151  2.0159623101
    125  A  A  0.549426652 -0.8850235386
    126  A  A  1.979850854  0.5400369581
    127  A  A  1.001748728  2.5358483234
    128  A  A  1.162154875  1.4227401657
    129  A  A  2.084872719  1.1108181829
    130  A  A  3.052371386  0.7205040510
    131  A  A  2.026799290  1.4215529311
    132  A  A  1.395274645  0.9081800019
    133  A  A  2.964428450  1.3033578946
    134  A  A  0.156643635  1.6099044306
    135  A  B -0.052243891  0.4992938701
    136  B  A  0.921391987  1.1357047272
    137  A  A  2.159888896  1.5392310668
    138  B  A  0.476815871 -0.0602578752
    139  A  A  0.295883904 -0.4112301225
    140  A  A -1.418019096  1.7233704010
    141  A  A  1.258466841  1.4610973189
    142  A  A  0.837150762  0.2049491024
    143  A  A  3.278234678  3.5165836403
    144  A  A  1.301761535  1.8953878277
    145  A  A  1.826750623  1.6195806968
    146  A  A  0.838135575  1.8119850314
    147  A  B  2.404308577  0.5296448626
    148  A  A  1.784926622  2.0101092173
    149  A  A  3.067954267  0.7045402514
    150  A  A -0.414425581  1.3043637401
    151  A  A  1.558414708  2.9261421121
    152  A  A  0.170185546  2.3390104973
    153  A  A  3.365988592  2.1913403046
    154  A  A  1.184755386  0.3664586421
    155  A  A  1.955477275  2.2617098433
    156  A  A  2.969287379  0.3114464159
    157  A  A  1.733453660  1.3127030841
    158  A  A  0.827761734  1.2755953403
    159  A  A  3.011459721  0.9841268798
    160  A  A  0.993527056  1.4678060813
    161  A  A  0.543010903  2.4659056003
    162  A  A  2.084085715  2.4626274547
    163  A  A  1.952506385  2.0490953596
    164  A  A  2.048222637  2.9901529451
    165  A  A  1.211215724  1.2667531651
    166  A  A  1.516957124  1.5497915283
    167  A  A -0.425457223  0.8574180366
    168  A  A  2.284600636  1.7137258449
    169  B  A  1.159464629  0.8854124906
    170  A  A  2.924873606  1.4572722412
    171  A  A  2.213765524  1.4133493583
    172  A  A  1.576466775  2.6405859293
    173  A  A  0.268283969  3.1784366093
    174  A  A  0.382891495  1.2417561381
    175  A  A  2.050130972  1.6823771449
    176  A  A  3.196798283  2.2791483471
    177  B  A  2.084392209  0.2655176520
    178  A  A  1.989145860  0.9403774443
    179  A  A  0.409875436  1.0034684198
    180  A  A  0.599820134  2.0672227943
    181  A  A  4.052116935  0.9144489302
    182  A  A  2.180787832  2.8722868749
    183  A  A  2.432723003  0.3410504387
    184  A  A  3.238038382  2.2659622190
    185  A  A  2.386624133  1.1652540585
    186  A  A  2.149019911  3.0848130733
    187  A  A  0.642416876  0.8753721637
    188  A  A  1.762370943  0.5266803206
    189  A  A  1.556399875  1.5639636959
    190  A  A  1.400633133  1.6997604651
    191  B  A  3.777049683  2.3113813367
    192  A  A  2.110617082  0.6870528172
    193  A  A  0.598075291  0.5728745505
    194  A  A  1.169649192 -0.1737029755
    195  A  A -0.916442410  0.6335840047
    196  B  A -0.048210095  0.8945123948
    197  A  B -0.602037567  1.5538897917
    198  A  A  1.585044572  0.5972391613
    199  B  A  0.932830929  1.5424812239
    200  A  A  1.839032736  1.9939243683
    201  B  B -0.263064058 -2.7255368533
    202  A  B  0.164228541 -4.0214985273
    203  B  B -1.944201052 -2.0978893276
    204  B  A -0.380736183  0.6320505570
    205  B  B -0.172683508 -1.4772580167
    206  B  A -1.399172914 -1.8330725438
    207  B  B -0.898890438 -3.3175332149
    208  B  B -1.470212229 -0.9470207946
    209  B  B -2.880840516 -0.8735319300
    210  B  B -2.130443672 -2.6855862849
    211  B  A -1.246404146 -0.8799519230
    212  B  B -1.915397456 -0.4708886958
    213  B  B -1.794625911 -2.0958332980
    214  B  B  0.963618293 -1.5256129285
    215  B  B -0.151633603 -1.7847104959
    216  B  A -2.269108823 -3.0959801211
    217  B  B -1.878484612 -2.4440310054
    218  B  B -0.977357405 -1.1131909565
    219  B  B -1.540117187 -2.4165803392
    220  B  B -0.689720736 -1.3320398760
    221  B  B -3.193176419 -1.8621568385
    222  A  B -1.770206465 -2.2437274131
    223  B  B -1.096336370 -0.0913402437
    224  B  B  0.773251391 -0.5152537852
    225  B  B -1.337739316 -2.0996586860
    226  B  B -1.677917383 -1.6501996552
    227  B  B  0.477212341 -0.7820927308
    228  B  B -2.434383709 -3.0735181220
    229  B  B -3.089987362 -2.0572995732
    230  A  B -0.757190275 -1.9626091532
    231  B  B -2.219975625 -0.6523834972
    232  B  B -1.944783079 -0.5363181865
    233  B  B -3.743593542 -1.7242507335
    234  B  A -1.367817759 -1.4389490708
    235  B  A -2.262037517 -3.0525271434
    236  B  B -2.056524243 -2.3585643512
    237  B  A -1.274021428 -0.7320155680
    238  B  A -2.244525440  1.2202778941
    239  B  B -1.580231468 -1.8846964668
    240  B  A -1.023851519 -0.6123766363
    241  B  B -2.940961913 -1.3526750436
    242  A  B -2.463725067 -2.5449663771
    243  B  B -0.748951803 -1.5226301406
    244  B  B -0.378616894 -2.1068347125
    245  B  B -2.780519585 -1.3720405026
    246  B  B  0.261198111 -2.7522555220
    247  B  B -1.101929581 -1.7764612776
    248  B  B -2.189325678 -1.6062088374
    249  A  B -2.183840839  0.5702419063
    250  B  B  0.570322015  1.0366411903
    251  B  B -0.404180455 -0.8103020852
    252  B  B -1.610190579 -2.5399333095
    253  B  B -2.527642218 -1.5367355528
    254  B  B -0.950479196 -3.0817047037
    255  A  B -0.479014116 -2.8688356731
    256  B  B -1.250162292 -2.2360444876
    257  B  B -0.826398561 -2.1850882857
    258  B  B -1.210666303 -2.3683803128
    259  B  B -0.983925500  1.6429369593
    260  B  B -2.034265708 -1.5032589337
    261  B  B -0.665890176 -1.3231290274
    262  B  B -2.834745789 -1.2918307669
    263  B  B -0.933515851 -0.8648483626
    264  B  B -2.230778201 -1.8897458915
    265  B  B -0.787647329 -0.2384857178
    266  A  B  0.103473548 -3.1399215488
    267  B  B -1.002869048  0.1546135946
    268  B  B -0.206514334 -0.7305039922
    269  B  B -1.118522123 -2.4734576277
    270  B  B -1.689222255 -3.0089709866
    271  B  A -2.605025887 -1.8017092359
    272  B  B -0.902174770 -2.1623270025
    273  B  B -2.257299635 -1.0895187925
    274  B  B -1.518143010 -1.8403624192
    275  B  B -0.323405812 -0.8898819031
    276  B  B -1.810384786 -1.7815900473
    277  B  B -2.250757587 -1.8018517820
    278  B  B -1.730266461 -2.8422613455
    279  B  B -1.926723291 -1.0161838990
    280  B  B -1.822145239 -1.0029780113
    281  B  B -1.024154831  1.0100453910
    282  B  B -1.319220244 -0.2090571051
    283  B  B -1.720233187 -1.8882164049
    284  B  B -0.978172738  0.4055301274
    285  B  A -2.665906522 -0.9527100509
    286  B  B -1.601474981  0.0915811837
    287  A  B -2.207451192 -2.0022484990
    288  B  B -1.117011207 -2.5699579913
    289  B  A -0.581467128 -2.4215710746
    290  B  A -2.835146593 -2.1596765983
    291  B  B -1.074886887 -0.4322314134
    292  B  B -0.601431421 -1.0865653075
    293  A  B -1.360866633 -2.6548362661
    294  B  B -2.052309882 -1.8502736857
    295  B  B -1.459040909 -0.5084886453
    296  B  B -1.976051036 -0.7957221416
    297  B  A  2.259324942 -2.5252362648
    298  B  B -2.066025369 -0.4597545351
    299  B  B -2.298626756 -0.3895275192
    300  B  B -1.172302701 -0.6084379561
    301  A  B  0.774968173  3.2301264790
    302  B  B  1.902070671  2.5740698500
    303  B  B  1.917632926  1.7011859524
    304  B  B  1.539985291 -1.0127872979
    305  B  B  2.054749802  0.9216751215
    306  B  B  0.845360552  2.1364299680
    307  B  B -0.587986222  1.2052801716
    308  B  A  1.843232000  3.7762987187
    309  B  A  2.119013317  0.3330280601
    310  B  B  0.394665098  2.6679762919
    311  A  B  2.428154216  0.5945171564
    312  B  B  2.433059774  0.8640759464
    313  B  B  2.798899202  0.7135071241
    314  A  B  0.521271633  2.8905146275
    315  B  A  1.723174597  3.4864825221
    316  B  B  2.175660409  2.1491000185
    317  B  B  0.888223485  1.7514246728
    318  B  B  2.332757944  0.3619046545
    319  B  B -1.238441759  1.1505912370
    320  B  B  0.355214792  1.0468707798
    321  B  B  0.617528445  1.5955294650
    322  B  B  0.896553716  1.1661070136
    323  B  B -0.511005529  0.0517910209
    324  B  B  2.235034238  0.3257766236
    325  B  A  1.230897023  0.1420030875
    326  B  B  0.661326020  0.0270292362
    327  B  B  1.246049341  1.2860024873
    328  B  B  2.550526957  3.5339443721
    329  A  B  1.436450811  0.3131389239
    330  B  B  1.169206365  0.2087436487
    331  A  B -0.310035411  2.1512446639
    332  B  B  0.847129425 -0.1044311423
    333  B  B  1.311271603  1.4450617214
    334  A  B  3.696241397  2.2896095635
    335  B  B  1.698794134  2.3836150182
    336  B  B  0.977932933  2.3731612471
    337  B  B  1.762376869  2.2490262165
    338  B  A  1.885655560  4.0461633381
    339  B  B  1.422535416  0.4970499009
    340  B  B  2.081442689  1.6375529587
    341  B  B  1.730523780  2.2582981943
    342  B  B  3.356023347  2.0191764237
    343  A  B  0.962087729  2.4585008600
    344  B  B  0.512242983  1.3933260837
    345  B  B  0.878004932  1.4389087670
    346  B  B  2.045734261  0.7328323457
    347  A  B  3.685324449  1.4339105264
    348  B  A  1.518310857  2.9275277120
    349  B  B  3.456691949  1.4081729992
    350  B  B -1.037491110  1.4087453459
    351  B  B  0.958317749  1.3258660616
    352  B  B  1.513422268  2.5093723297
    353  B  A  1.677605837  1.9260823818
    354  B  B  0.597867461  0.6758424129
    355  B  B  1.754017764  3.1140007654
    356  B  B  1.178034952  1.0624398178
    357  B  B -0.004412398  2.3648526341
    358  B  B  3.890117391  1.8534599304
    359  B  B  0.982469160  0.5295660100
    360  B  B  0.810374402  0.0390148418
    361  B  B  1.271572048 -0.3646739267
    362  B  B  0.189300755  0.3349824194
    363  B  B  2.315637547  0.4919204092
    364  B  B  2.453919758  2.0839128413
    365  B  B  1.449746522  0.9925535301
    366  A  B  3.154100818  0.5323381105
    367  B  B  0.192958557  3.0441193046
    368  B  A  2.413564118  2.2077473265
    369  B  B  2.367153069  2.0238800909
    370  B  B  1.974889533  0.1328200428
    371  B  B  1.401608416  1.9346769519
    372  B  B  0.451375050  1.7708080976
    373  A  B  0.670190849  1.0900296491
    374  B  B  1.275878153  0.8536563734
    375  B  B -0.393533108  0.9506490728
    376  B  B  2.236143090  1.4459945878
    377  B  B  2.127309178  1.5879962011
    378  B  B  0.963395952  0.4659758733
    379  B  B  1.383624349  1.9157940541
    380  B  B  2.916630585  0.8986613051
    381  B  B  0.533140283  2.0720676040
    382  B  B  0.022493753  1.2602470003
    383  B  A  4.169087795  1.9548426463
    384  B  B  0.945889163  1.1673701299
    385  B  B  1.719589085  1.5790538858
    386  B  B  1.776849329  0.8746158143
    387  B  A  1.608258780  1.1939518615
    388  A  B  0.393180095  3.3118039929
    389  B  B  0.380918839  0.6347413953
    390  B  B  2.909777452  2.6446014932
    391  B  B  2.631087698  3.5138500530
    392  B  B  0.882533975  0.0003695591
    393  B  B  1.147894499  1.6335613390
    394  B  B  2.298407357  1.3253381832
    395  B  B  2.283781470  2.7122590223
    396  B  B  0.830637970  1.1005304957
    397  B  B  0.550687449 -0.3227471442
    398  B  B  0.599611074  2.5934468227
    399  B  A  3.055295219  2.4596248354
    400  B  B  1.944303064  1.7689753971
    


```R
# apply k-prototypes
kpres <- kproto(x, 4)
clprofiles(kpres, x)
```

    # NAs in variables:
    x1 x2 x3 x4 
     0  0  0  0 
    0 observation(s) with NAs.
    
    Estimated lambda: 6.359521 
    
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_139_1.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_139_2.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_139_3.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_139_4.png)



```R
# In real world clusters are often not as clear cut
# By variation of lambda the emphasize is shifted towards factor/numeric variables    
kpres <- kproto(x, 2)
clprofiles(kpres, x)
```

    # NAs in variables:
    x1 x2 x3 x4 
     0  0  0  0 
    0 observation(s) with NAs.
    
    Estimated lambda: 6.359521 
    
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_140_1.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_140_2.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_140_3.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_140_4.png)



```R
kpres <- kproto(x, 2, lambda = 0.1)
clprofiles(kpres, x)
```

    # NAs in variables:
    x1 x2 x3 x4 
     0  0  0  0 
    0 observation(s) with NAs.
    
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_141_1.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_141_2.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_141_3.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_141_4.png)



```R
kpres <- kproto(x, 2, lambda = 25)
clprofiles(kpres, x)
```

    # NAs in variables:
    x1 x2 x3 x4 
     0  0  0  0 
    0 observation(s) with NAs.
    
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_142_1.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_142_2.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_142_3.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_142_4.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

# K-Medoid/PAM(Partitioning around Medoid)

Both the k-means and k-medoids algorithms are partitional, which involves breaking the dataset into groups. K-means aims to minimize the total squared error from a central position in each cluster. These central positions are called centroids. On the other hand, k-medoids attempts to minimize the sum of dissimilarities between objects labeled to be in a cluster and one of the objects designated as the representative of that cluster. These representatives are called medoids.
In contrast to the k-means algorithm that the centroids are central, average positions that might not be data points in the set, k-medoids chooses medoids from the data points in the set.


```R
install.packages("kmed")
```

    package 'kmed' successfully unpacked and MD5 sums checked
    
    The downloaded binary packages are in
    	C:\Users\amit\AppData\Local\Temp\RtmpGo9GTP\downloaded_packages
    


```R
library(kmed)
```

    Warning message:
    "package 'kmed' was built under R version 3.6.3"

The distNumeric function can be applied to calculate numerical distances.
The distNumeric function provides method in which the desired distance method can be selected. The default method is Manhattan weighted by range (mrw).


```R
#Using IRIS datatset
iris[1:10,]
```


<table>
<thead><tr><th scope=col>Sepal.Length</th><th scope=col>Sepal.Width</th><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th><th scope=col>Species</th></tr></thead>
<tbody>
	<tr><td>5.1   </td><td>3.5   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.9   </td><td>3.0   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.7   </td><td>3.2   </td><td>1.3   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.6   </td><td>3.1   </td><td>1.5   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.0   </td><td>3.6   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.4   </td><td>3.9   </td><td>1.7   </td><td>0.4   </td><td>setosa</td></tr>
	<tr><td>4.6   </td><td>3.4   </td><td>1.4   </td><td>0.3   </td><td>setosa</td></tr>
	<tr><td>5.0   </td><td>3.4   </td><td>1.5   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.4   </td><td>2.9   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.9   </td><td>3.1   </td><td>1.5   </td><td>0.1   </td><td>setosa</td></tr>
</tbody>
</table>



By applying the distNumeric function with method = "mrw", the distance among objects in the iris data set can be obtained.


```R
num <- as.matrix(iris[,1:4])
rownames(num) <- rownames(iris)

#calculate the Manhattan weighted by range distance of all iris objects
mrwdist <- distNumeric(num, num)

#show the distance among objects 1 to 3
mrwdist[1:3,1:3]
```


<table>
<thead><tr><th scope=col>1</th><th scope=col>2</th><th scope=col>3</th></tr></thead>
<tbody>
	<tr><td>0.0000000</td><td>0.2638889</td><td>0.2530603</td></tr>
	<tr><td>0.2638889</td><td>0.0000000</td><td>0.1558380</td></tr>
	<tr><td>0.2530603</td><td>0.1558380</td><td>0.0000000</td></tr>
</tbody>
</table>



The Manhattan weighted by range distance between objects 1 and 2 is 0.2638889. To calculate this distance, the range of each variable is computed.


```R
#extract the range of each variable
apply(num, 2, function(x) max(x)-min(x))
```


<dl class=dl-horizontal>
	<dt>Sepal.Length</dt>
		<dd>3.6</dd>
	<dt>Sepal.Width</dt>
		<dd>2.4</dd>
	<dt>Petal.Length</dt>
		<dd>5.9</dd>
	<dt>Petal.Width</dt>
		<dd>2.4</dd>
</dl>




```R
#Then, the distance between objects 1 and 2 is
abs(5.1-4.9)/3.6 + abs(3.5 - 3.0)/2.4 + abs(1.4-1.4)/5.9 + abs(0.2-0.2)/2.4
```


0.263888888888889


There are four k-medoids presented, namely the simple and fast k-medoids, k-medoids, ranked k-medoids, and increasing number of clusters in k-medoids.

#### 1. Simple and fast k-medoids algorithm (fastkmed)

The fastkmed function runs this algorithm to cluster the objects. The compulsory inputs are a distance matrix or distance object and a number of clusters. Hence, the SFKM algorithm for the iris data set is mentioned below.


```R
#run the sfkm algorihtm on iris data set with mrw distance
sfkm <- fastkmed(mrwdist, ncluster = 3, iterate = 50)
print(sfkm)
```

    $cluster
      1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20 
      1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1 
     21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40 
      1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1 
     41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60 
      1   1   1   1   1   1   1   1   1   1   3   3   3   2   3   2   3   2   2   2 
     61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80 
      2   2   2   2   2   3   2   2   2   2   3   2   2   2   2   3   3   3   2   2 
     81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 
      2   2   2   2   2   2   3   2   2   2   2   2   2   2   2   2   2   2   2   2 
    101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 
      3   3   3   3   3   3   2   3   3   3   3   3   3   3   3   3   3   3   3   2 
    121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 
      3   3   3   3   3   3   3   3   3   3   3   3   3   3   2   3   3   3   3   3 
    141 142 143 144 145 146 147 148 149 150 
      3   3   3   3   3   3   3   3   3   3 
    
    $medoid
    [1]   8  95 148
    
    $minimum_distance
    [1] 48.76718
    
    


```R
#A classification table is obtained.
(sfkmtable <- table(sfkm$cluster, iris[,5]))
```


       
        setosa versicolor virginica
      1     50          0         0
      2      0         39         3
      3      0         11        47



```R
#Applying the SFKM algorithm in iris data set with the Manhattan weighted by range, the misclassification rate is

(3+11)/sum(sfkmtable)
```


0.0933333333333333


#### 2. K-medoids algorithm


```R
#set the initial medoids
set.seed(1)
(kminit <- sample(1:nrow(iris), 3))
```


<ol class=list-inline>
	<li>68</li>
	<li>129</li>
	<li>43</li>
</ol>




```R
#run the km algorihtm on iris data set with mrw distance
km <- fastkmed(mrwdist, ncluster = 3, iterate = 50, init = kminit)
print(km)
```

    $cluster
      1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20 
      3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3 
     21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40 
      3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3 
     41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60 
      3   3   3   3   3   3   3   3   3   3   2   2   2   1   1   1   2   1   1   1 
     61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80 
      1   1   1   1   1   2   1   1   1   1   2   1   1   1   1   2   1   2   1   1 
     81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 
      1   1   1   1   1   1   2   1   1   1   1   1   1   1   1   1   1   1   1   1 
    101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 
      2   2   2   2   2   2   1   2   2   2   2   2   2   2   2   2   2   2   2   1 
    121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 
      2   2   2   2   2   2   2   2   2   2   2   2   2   2   1   2   2   2   2   2 
    141 142 143 144 145 146 147 148 149 150 
      2   2   2   2   2   2   2   2   2   2 
    
    $medoid
    [1] 100 148   8
    
    $minimum_distance
    [1] 48.8411
    
    


```R
#The classification table of the KM algorithm is
(kmtable <- table(km$cluster, iris[,5]))

```


       
        setosa versicolor virginica
      1      0         41         3
      2      0          9        47
      3     50          0         0



```R
# misclassification rate

(3+9)/sum(kmtable)
```


0.08


#### 3.  Rank k-medoids algorithm (rankkmed)

The rankkmed function runs the RKM algorithm. The m argument is introduced to calculate a hostility score. The m indicates how many closest objects is selected. The selected objects as initial medoids in the RKM is randomly assigned. The RKM algorithm for the iris data is run set by setting m = 10 


```R
#run the rkm algorihtm on iris data set with mrw distance and m = 10
rkm <- rankkmed(mrwdist, ncluster = 3, m = 10, iterate = 50)
print(rkm)
```

    $cluster
      1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20 
      1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1 
     21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40 
      1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1   1 
     41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60 
      1   1   1   1   1   1   1   1   1   1   2   2   2   3   2   3   2   3   2   3 
     61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80 
      3   3   3   3   3   2   3   3   3   3   3   3   2   3   2   2   2   2   3   3 
     81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 
      3   3   3   3   3   3   2   2   3   3   3   3   3   3   3   3   3   3   3   3 
    101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 
      2   3   2   2   2   2   3   2   2   2   2   2   2   3   3   2   2   2   2   3 
    121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 
      2   3   2   2   2   2   2   3   2   2   2   2   2   2   3   2   2   2   3   2 
    141 142 143 144 145 146 147 148 149 150 
      2   2   3   2   2   2   2   2   3   3 
    
    $medoid
    [1] "50" "55" "79"
    
    $minimum_distance
    [1] 60.89713
    
    


```R
#classification table is attained by
(rkmtable <- table(rkm$cluster, iris[,5]))
```


       
        setosa versicolor virginica
      1     50          0         0
      2      0         14        38
      3      0         36        12



```R
#The misclassification proportion is
(3+3)/sum(rkmtable)
```


0.04


With 4% misclassification rate, the RKM algorithm is the best among the three algorithm.

#### 4. Increasing number of clusters k-medoids algorithm (inckmed)


```R
#Run the inckm algorihtm on iris data set with mrw distance and alpha = 1.2
inckm <- inckmed(mrwdist, ncluster = 3, alpha = 1.1, iterate = 50)
print(inckm)
```

    $cluster
      1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20 
      3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3 
     21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40 
      3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3   3 
     41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60 
      3   3   3   3   3   3   3   3   3   3   2   2   2   1   1   1   2   1   1   1 
     61  62  63  64  65  66  67  68  69  70  71  72  73  74  75  76  77  78  79  80 
      1   1   1   1   1   2   1   1   1   1   2   1   1   1   1   2   1   2   1   1 
     81  82  83  84  85  86  87  88  89  90  91  92  93  94  95  96  97  98  99 100 
      1   1   1   1   1   1   2   1   1   1   1   1   1   1   1   1   1   1   1   1 
    101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 
      2   2   2   2   2   2   1   2   2   2   2   2   2   2   2   2   2   2   2   1 
    121 122 123 124 125 126 127 128 129 130 131 132 133 134 135 136 137 138 139 140 
      2   2   2   2   2   2   2   2   2   2   2   2   2   2   1   2   2   2   2   2 
    141 142 143 144 145 146 147 148 149 150 
      2   2   2   2   2   2   2   2   2   2 
    
    $medoid
    [1] 100 148   8
    
    $minimum_distance
    [1] 48.8411
    
    

The alpha argument indicates a stretch factor to select the initial medoids. The SFKM, KM and INCKM are similar algorithm with a different way to select the initial medoids.


```R
#The classification table can be attained.
(inckmtable <- table(inckm$cluster, iris[,5]))
```


       
        setosa versicolor virginica
      1      0         41         3
      2      0          9        47
      3     50          0         0



```R
#The misclassification rate is
(9+3)/sum(inckmtable)
```


0.08


The algorithm has 8% misclassification rate.
#### Conclusion: The RKM algorithm performs the best among the four algorithms in the iris data set with the mrw distance.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a>

# Clustering Validity Measures

Generally, cluster validity measures are categorized into 3 classes, they are –

Internal cluster validation : The clustering result is evaluated based on the data clustered itself (internal information) without reference to external information. 

External cluster validation : Clustering results are evaluated based on some externally known result, such as externally provided class labels.

Relative cluster validation : The clustering results are evaluated by varying different parameters for the same algorithm (e.g. changing the number of clusters).

#### Data preparation for clustering and its validation

The following packages will be used:
<br>-cluster for computing PAM clustering and for analyzing cluster silhouettes
<br>-factoextra for simplifying clustering workflows and for visualizing clusters using ggplot2 plotting system
<br>-NbClust for determining the optimal number of clusters in the data
<br>-fpc for computing clustering validation statistics



```R
#Install factoextra package as follow:
if(!require(devtools)) install.packages("devtools")
devtools::install_github("kassambara/factoextra")
```

    Loading required package: devtools
    Warning message in library(package, lib.loc = lib.loc, character.only = TRUE, logical.return = TRUE, :
    "there is no package called 'devtools'"also installing the dependencies 'ini', 'fs', 'gh', 'git2r', 'processx', 'rex', 'xopen', 'brew', 'commonmark', 'usethis', 'callr', 'cli', 'covr', 'jsonlite', 'memoise', 'rcmdcheck', 'remotes', 'roxygen2', 'rstudioapi', 'rversions', 'sessioninfo', 'withr'
    
    

    package 'ini' successfully unpacked and MD5 sums checked
    package 'fs' successfully unpacked and MD5 sums checked
    package 'gh' successfully unpacked and MD5 sums checked
    package 'git2r' successfully unpacked and MD5 sums checked
    package 'processx' successfully unpacked and MD5 sums checked
    package 'rex' successfully unpacked and MD5 sums checked
    package 'xopen' successfully unpacked and MD5 sums checked
    package 'brew' successfully unpacked and MD5 sums checked
    package 'commonmark' successfully unpacked and MD5 sums checked
    package 'usethis' successfully unpacked and MD5 sums checked
    package 'callr' successfully unpacked and MD5 sums checked
    package 'cli' successfully unpacked and MD5 sums checked
    package 'covr' successfully unpacked and MD5 sums checked
    package 'jsonlite' successfully unpacked and MD5 sums checked
    

    Warning message:
    "cannot remove prior installation of package 'jsonlite'"Warning message in file.copy(savedcopy, lib, recursive = TRUE):
    "problem copying C:\Users\amit\Anaconda3\envs\R practice\Lib\R\library\00LOCK\jsonlite\libs\x64\jsonlite.dll to C:\Users\amit\Anaconda3\envs\R practice\Lib\R\library\jsonlite\libs\x64\jsonlite.dll: Permission denied"Warning message:
    "restored 'jsonlite'"

    package 'memoise' successfully unpacked and MD5 sums checked
    package 'rcmdcheck' successfully unpacked and MD5 sums checked
    package 'remotes' successfully unpacked and MD5 sums checked
    package 'roxygen2' successfully unpacked and MD5 sums checked
    package 'rstudioapi' successfully unpacked and MD5 sums checked
    package 'rversions' successfully unpacked and MD5 sums checked
    package 'sessioninfo' successfully unpacked and MD5 sums checked
    package 'withr' successfully unpacked and MD5 sums checked
    package 'devtools' successfully unpacked and MD5 sums checked
    
    The downloaded binary packages are in
    	C:\Users\amit\AppData\Local\Temp\RtmpuiYr9s\downloaded_packages
    

    WARNING: Rtools is required to build R packages, but is not currently installed.
    
    Please download and install Rtools 3.5 from https://cran.r-project.org/bin/windows/Rtools/.
    Downloading GitHub repo kassambara/factoextra@HEAD
    Warning message in utils::untar(tarfile, ...):
    "'tar.exe -xf "C:\Users\amit\AppData\Local\Temp\RtmpuiYr9s\file73542f278ba.tar.gz" -C "C:/Users/amit/AppData/Local/Temp/RtmpuiYr9s/remotes735478b45650"' returned error code 1"Warning message in system(cmd, intern = TRUE):
    "running command 'tar.exe -tf "C:\Users\amit\AppData\Local\Temp\RtmpuiYr9s\file73542f278ba.tar.gz"' had status 1"


    Error: Failed to install 'factoextra' from GitHub:
      length(file_list) > 0 is not TRUE
    Traceback:
    

    1. devtools::install_github("kassambara/factoextra")

    2. pkgbuild::with_build_tools({
     .     ellipsis::check_dots_used(action = getOption("devtools.ellipsis_action", 
     .         rlang::warn))
     .     {
     .         remotes <- lapply(repo, github_remote, ref = ref, subdir = subdir, 
     .             auth_token = auth_token, host = host)
     .         install_remotes(remotes, auth_token = auth_token, host = host, 
     .             dependencies = dependencies, upgrade = upgrade, force = force, 
     .             quiet = quiet, build = build, build_opts = build_opts, 
     .             build_manual = build_manual, build_vignettes = build_vignettes, 
     .             repos = repos, type = type, ...)
     .     }
     . }, required = FALSE)

    3. install_remotes(remotes, auth_token = auth_token, host = host, 
     .     dependencies = dependencies, upgrade = upgrade, force = force, 
     .     quiet = quiet, build = build, build_opts = build_opts, build_manual = build_manual, 
     .     build_vignettes = build_vignettes, repos = repos, type = type, 
     .     ...)

    4. tryCatch(res[[i]] <- install_remote(remotes[[i]], ...), error = function(e) {
     .     stop(remote_install_error(remotes[[i]], e))
     . })

    5. tryCatchList(expr, classes, parentenv, handlers)

    6. tryCatchOne(expr, names, parentenv, handlers[[1L]])

    7. value[[3L]](cond)



```R
pkgs <- c("cluster", "fpc", "NbClust")
install.packages(pkgs)
```

    Warning message:
    "package 'fpc' is in use and will not be installed"

    package 'cluster' successfully unpacked and MD5 sums checked
    

    Warning message:
    "cannot remove prior installation of package 'cluster'"Warning message in file.copy(savedcopy, lib, recursive = TRUE):
    "problem copying C:\Users\amit\Anaconda3\envs\R practice\Lib\R\library\00LOCK\cluster\libs\x64\cluster.dll to C:\Users\amit\Anaconda3\envs\R practice\Lib\R\library\cluster\libs\x64\cluster.dll: Permission denied"Warning message:
    "restored 'cluster'"

    package 'NbClust' successfully unpacked and MD5 sums checked
    
    The downloaded binary packages are in
    	C:\Users\amit\AppData\Local\Temp\RtmpuiYr9s\downloaded_packages
    


```R
#Loadning packages
library(factoextra)
library(cluster)
library(fpc)
library(NbClust)
```

    Warning message:
    "package 'factoextra' was built under R version 3.6.3"Loading required package: ggplot2
    Warning message:
    "package 'ggplot2' was built under R version 3.6.3"Welcome! Want to learn more? See two factoextra-related books at https://goo.gl/ve3WBa
    Warning message:
    "package 'fpc' was built under R version 3.6.3"


```R
# Load the data
data("iris")
head(iris)
```


<table>
<thead><tr><th scope=col>Sepal.Length</th><th scope=col>Sepal.Width</th><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th><th scope=col>Species</th></tr></thead>
<tbody>
	<tr><td>5.1   </td><td>3.5   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.9   </td><td>3.0   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.7   </td><td>3.2   </td><td>1.3   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>4.6   </td><td>3.1   </td><td>1.5   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.0   </td><td>3.6   </td><td>1.4   </td><td>0.2   </td><td>setosa</td></tr>
	<tr><td>5.4   </td><td>3.9   </td><td>1.7   </td><td>0.4   </td><td>setosa</td></tr>
</tbody>
</table>




```R
# Remove species column (5) and scale the data
iris.scaled <- scale(iris[, -5])
```


```R
# Compute the number of clusters
library(NbClust)
nb <- NbClust(iris.scaled, distance = "euclidean", min.nc = 2,
        max.nc = 10, method = "complete", index ="all")
```

    *** : The Hubert index is a graphical method of determining the number of clusters.
                    In the plot of Hubert index, we seek a significant knee that corresponds to a 
                    significant increase of the value of the measure i.e the significant peak in Hubert
                    index second differences plot. 
     
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_188_1.png)


    *** : The D index is a graphical method of determining the number of clusters. 
                    In the plot of D index, we seek a significant knee (the significant peak in Dindex
                    second differences plot) that corresponds to a significant increase of the value of
                    the measure. 
     
    ******************************************************************* 
    * Among all indices:                                                
    * 2 proposed 2 as the best number of clusters 
    * 18 proposed 3 as the best number of clusters 
    * 3 proposed 10 as the best number of clusters 
    
                       ***** Conclusion *****                            
     
    * According to the majority rule, the best number of clusters is  3 
     
     
    ******************************************************************* 
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_188_3.png)



```R
# Visualize the result
library(factoextra)
fviz_nbclust(nb) + theme_minimal()
```

    Among all indices: 
    ===================
    * 2 proposed  0 as the best number of clusters
    * 1 proposed  1 as the best number of clusters
    * 2 proposed  2 as the best number of clusters
    * 18 proposed  3 as the best number of clusters
    * 3 proposed  10 as the best number of clusters
    
    Conclusion
    =========================
    * According to the majority rule, the best number of clusters is  3 .
    
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_189_1.png)



```R
# K-means clustering
km.res <- eclust(iris.scaled, "kmeans", k = 3,
                 nstart = 25, graph = FALSE)
# k-means group number of each observation
km.res$cluster
```


<ol class=list-inline>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
</ol>




```R
# Visualize k-means clusters
fviz_cluster(km.res, geom = "point", frame.type = "norm")
```

    Warning message:
    "argument frame is deprecated; please use ellipse instead."Warning message:
    "argument frame.type is deprecated; please use ellipse.type instead."


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_191_1.png)



```R
# PAM clustering
pam.res <- eclust(iris.scaled, "pam", k = 3, graph = FALSE)
pam.res$cluster
```


<ol class=list-inline>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
	<li>2</li>
	<li>2</li>
	<li>3</li>
</ol>




```R
# Visualize pam clusters
fviz_cluster(pam.res, geom = "point", frame.type = "norm")
```

    Warning message:
    "argument frame is deprecated; please use ellipse instead."Warning message:
    "argument frame.type is deprecated; please use ellipse.type instead."


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_193_1.png)



```R
# Enhanced hierarchical clustering
res.hc <- eclust(iris.scaled, "hclust", k = 3,
                method = "complete", graph = FALSE) 
head(res.hc$cluster, 15)
```


<ol class=list-inline>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
</ol>




```R
# Dendrogram
fviz_dend(res.hc, rect = TRUE, show_labels = FALSE) 
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_195_0.png)


### Internal clustering validation measures
we want the average distance within cluster to be as small as possible; and the average distance between clusters to be as large as possible.

Internal validation measures reflect often the compactness, the connectedness and separation of the cluster partitions.

<b>Compactness</b> measures evaluate how close are the objects within the same cluster. A lower within-cluster variation is an indicator of a good compactness (i.e., a good clustering). The different indices for evaluating the compactness of clusters are base on distance measures such as the cluster-wise within average/median distances between observations.

<b>Separation</b> measures determine how well-separated a cluster is from other clusters. The indices used as separation measures include:
    <br>-distances between cluster centers
    <br>-the pairwise minimum distances between objects in different clusters
 
<b>Connectivity</b> corresponds to what extent items are placed in the same cluster as their nearest neighbors in the data space. The connectivity has a value between 0 and infinity and should be minimized.

Generally most of the indices used for internal clustering validation combine compactness and separation measures as follow:
<b><br>Index=(α×Separation)/(β×Compactness)</b>

Next, we’ll describe the two commonly used indices for assessing the goodness of clustering: Silhouette width and Dunn index.

## 1.Silhouette analysis

Silhouette analysis measures how well an observation is clustered and it estimates the average distance between clusters. The silhouette plot displays a measure of how close each point in one cluster is to points in the neighboring cluster

<b>Interpretation of silhouette width:</b>
<br>Silhouette width can be interpreted as follow:
        <br>-Observations with a large Si (almost 1) are very well clustered
        <br>-A small Si (around 0) means that the observation lies between two clusters
        <br>-Observations with a negative Si are probably placed in the wrong cluster.

The code below computes silhouette analysis and draw the result using R base plot:


```R
# Silhouette coefficient of observations
library("cluster")
sil <- silhouette(km.res$cluster, dist(iris.scaled))
head(sil[, 1:3], 10)
```


<table>
<thead><tr><th scope=col>cluster</th><th scope=col>neighbor</th><th scope=col>sil_width</th></tr></thead>
<tbody>
	<tr><td>1        </td><td>2        </td><td>0.7341949</td></tr>
	<tr><td>1        </td><td>2        </td><td>0.5682739</td></tr>
	<tr><td>1        </td><td>2        </td><td>0.6775472</td></tr>
	<tr><td>1        </td><td>2        </td><td>0.6205016</td></tr>
	<tr><td>1        </td><td>2        </td><td>0.7284741</td></tr>
	<tr><td>1        </td><td>2        </td><td>0.6098848</td></tr>
	<tr><td>1        </td><td>2        </td><td>0.6983835</td></tr>
	<tr><td>1        </td><td>2        </td><td>0.7308169</td></tr>
	<tr><td>1        </td><td>2        </td><td>0.4882100</td></tr>
	<tr><td>1        </td><td>2        </td><td>0.6315409</td></tr>
</tbody>
</table>




```R
# Silhouette plot
plot(sil, main ="Silhouette plot - K-means")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_204_0.png)



```R
#Use factoextra for elegant data visualization:
library(factoextra)
fviz_silhouette(sil)
```

      cluster size ave.sil.width
    1       1   50          0.64
    2       2   53          0.39
    3       3   47          0.35
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_205_1.png)



```R
# Summary of silhouette analysis
si.sum <- summary(sil)
# Average silhouette width of each cluster
si.sum$clus.avg.widths
```


<dl class=dl-horizontal>
	<dt>1</dt>
		<dd>0.636316174439295</dd>
	<dt>2</dt>
		<dd>0.393377210558143</dd>
	<dt>3</dt>
		<dd>0.347392234026205</dd>
</dl>




```R
# The total average (mean of all individual silhouette widths)
si.sum$avg.width
```


0.459948239205186



```R
# The size of each clusters
si.sum$clus.sizes
```


    cl
     1  2  3 
    50 53 47 


### Silhouette plot for k-means clustering


```R
# Default plot
fviz_silhouette(km.res)
```

      cluster size ave.sil.width
    1       1   50          0.64
    2       2   53          0.39
    3       3   47          0.35
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_210_1.png)



```R
# Change the theme and color
fviz_silhouette(km.res, print.summary = FALSE) +
  scale_fill_brewer(palette = "Dark2") +
  scale_color_brewer(palette = "Dark2") +
  theme_minimal()+
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_211_0.png)



```R
# Silhouette information
silinfo <- km.res$silinfo
names(silinfo)
```


<ol class=list-inline>
	<li>'widths'</li>
	<li>'clus.avg.widths'</li>
	<li>'avg.width'</li>
</ol>




```R
# Silhouette widths of each observation
head(silinfo$widths[, 1:3], 10)
```


<table>
<thead><tr><th></th><th scope=col>cluster</th><th scope=col>neighbor</th><th scope=col>sil_width</th></tr></thead>
<tbody>
	<tr><th scope=row>1</th><td>1        </td><td>2        </td><td>0.7341949</td></tr>
	<tr><th scope=row>41</th><td>1        </td><td>2        </td><td>0.7333345</td></tr>
	<tr><th scope=row>8</th><td>1        </td><td>2        </td><td>0.7308169</td></tr>
	<tr><th scope=row>18</th><td>1        </td><td>2        </td><td>0.7287522</td></tr>
	<tr><th scope=row>5</th><td>1        </td><td>2        </td><td>0.7284741</td></tr>
	<tr><th scope=row>40</th><td>1        </td><td>2        </td><td>0.7247047</td></tr>
	<tr><th scope=row>38</th><td>1        </td><td>2        </td><td>0.7244191</td></tr>
	<tr><th scope=row>12</th><td>1        </td><td>2        </td><td>0.7217939</td></tr>
	<tr><th scope=row>28</th><td>1        </td><td>2        </td><td>0.7215103</td></tr>
	<tr><th scope=row>29</th><td>1        </td><td>2        </td><td>0.7145192</td></tr>
</tbody>
</table>




```R
# Average silhouette width of each cluster
silinfo$clus.avg.widths
```


<ol class=list-inline>
	<li>0.636316174439295</li>
	<li>0.393377210558143</li>
	<li>0.347392234026205</li>
</ol>




```R
# The total average (mean of all individual silhouette widths)
silinfo$avg.width
```


0.459948239205186



```R
# The size of each clusters
km.res$size
```


<ol class=list-inline>
	<li>50</li>
	<li>53</li>
	<li>47</li>
</ol>



### Silhouette plot for PAM clustering


```R
fviz_silhouette(pam.res)
```

      cluster size ave.sil.width
    1       1   50          0.63
    2       2   45          0.35
    3       3   55          0.38
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_218_1.png)


### Silhouette plot for hierarchical clustering


```R
fviz_silhouette(res.hc)
```

      cluster size ave.sil.width
    1       1   49          0.63
    2       2   30          0.44
    3       3   71          0.32
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Clustering/output_220_1.png)


<b>Samples with a negative silhouette coefficient<b/>

It can be seen that several samples have a negative silhouette coefficient in the hierarchical clustering. This means that they are not in the right cluster.

We can find the name of these samples and determine the clusters they are closer (neighbor cluster), as follow:


```R
# Silhouette width of observation
sil <- res.hc$silinfo$widths[, 1:3]
# Objects with negative silhouette
neg_sil_index <- which(sil[, 'sil_width'] < 0)
sil[neg_sil_index, , drop = FALSE]
```


<table>
<thead><tr><th></th><th scope=col>cluster</th><th scope=col>neighbor</th><th scope=col>sil_width</th></tr></thead>
<tbody>
	<tr><th scope=row>84</th><td>3          </td><td>2          </td><td>-0.01269799</td></tr>
	<tr><th scope=row>122</th><td>3          </td><td>2          </td><td>-0.01789603</td></tr>
	<tr><th scope=row>62</th><td>3          </td><td>2          </td><td>-0.04756835</td></tr>
	<tr><th scope=row>135</th><td>3          </td><td>2          </td><td>-0.05302402</td></tr>
	<tr><th scope=row>73</th><td>3          </td><td>2          </td><td>-0.10091884</td></tr>
	<tr><th scope=row>74</th><td>3          </td><td>2          </td><td>-0.14761137</td></tr>
	<tr><th scope=row>114</th><td>3          </td><td>2          </td><td>-0.16107155</td></tr>
	<tr><th scope=row>72</th><td>3          </td><td>2          </td><td>-0.23036371</td></tr>
</tbody>
</table>



## 2.Dunn Index

Dunn index is another internal clustering validation measure which can be computed as follow:

For each cluster, compute the distance between each of the objects in the cluster and the objects in the other clusters
Use the minimum of this pairwise distance as the inter-cluster separation (min.separation)
For each cluster, compute the distance between the objects in the same cluster.
Use the maximal intra-cluster distance (i.e maximum diameter) as the intra-cluster compactness

Calculate Dunn index (D) as follow:
#### D= (min.separation)/(max.diameter)

If the data set contains compact and well-separated clusters, the diameter of the clusters is expected to be small and the distance between the clusters is expected to be large. Thus, Dunn index should be maximized.

#### R function for computing Dunn index
The function cluster.stats() [in fpc package] and the function NbClust() [in NbClust package] can be used to compute Dunn index and many other indices.

The function cluster.stats() is described below.

The simplified format is:
<b><br>cluster.stats</b>(d = NULL, clustering, al.clustering = NULL)
<b><br>-d: </b>a distance object between cases as generated by the dist() function
<b><br>-clustering: </b>vector containing the cluster number of each observation
<b><br>-alt.clustering: </b>vector such as for clustering, indicating an alternative clustering

The function <b>cluster.stats()</b> returns a list containing many components useful for analyzing the intrinsic characteristics of a clustering:
<b><br>-cluster.number:</b> number of clusters
<b><br>-cluster.size:</b> vector containing the number of points in each cluster
<b><br>-average.distance, median.distance:</b>vector containing the cluster-wise within average/median distances
<b><br>-average.between:</b> average distance between clusters. We want it to be as large as possible
<b><br>-average.within: </b>average distance within clusters. We want it to be as small as possible
<b><br>-clus.avg.silwidths:</b> vector of cluster average silhouette widths. Recall that, the silhouette width is also an estimate of the average distance between clusters. Its value is comprised between 1 and -1 with a value of 1 indicating a very good cluster.
<b><br>-within.cluster.ss:</b>a generalization of the within clusters sum of squares (k-means objective function), which is obtained if d is a Euclidean distance matrix.
dunn, dunn2: Dunn index
<b><br>-corrected.rand, vi:</b>Two indexes to assess the similarity of two clustering: the corrected Rand index and Meila’s VI
<br>All the above elements can be used to evaluate the internal quality of clustering.
<br>In the following sections, we’ll compute the clustering quality statistics for k-means, pam and hierarchical clustering. Look at the within.cluster.ss (within clusters sum of squares), the average.within (average distance within clusters) and<b><br>-clus.avg.silwidths</b> (vector of cluster average silhouette widths).


#### Cluster statistics for k-means clustering


```R
library(fpc)
# Compute pairwise-distance matrices
dd <- dist(iris.scaled, method ="euclidean")
# Statistics for k-means clustering
km_stats <- cluster.stats(dd,  km.res$cluster)
# (k-means) within clusters sum of squares
km_stats$within.cluster.ss
```


138.888359717351



```R
# (k-means) cluster average silhouette widths
km_stats$clus.avg.silwidths
```


<dl class=dl-horizontal>
	<dt>1</dt>
		<dd>0.636316174439295</dd>
	<dt>2</dt>
		<dd>0.393377210558143</dd>
	<dt>3</dt>
		<dd>0.347392234026205</dd>
</dl>




```R
# Display all statistics
km_stats
```


<dl>
	<dt>$n</dt>
		<dd>150</dd>
	<dt>$cluster.number</dt>
		<dd>3</dd>
	<dt>$cluster.size</dt>
		<dd><ol class=list-inline>
	<li>50</li>
	<li>53</li>
	<li>47</li>
</ol>
</dd>
	<dt>$min.cluster.size</dt>
		<dd>47</dd>
	<dt>$noisen</dt>
		<dd>0</dd>
	<dt>$diameter</dt>
		<dd><ol class=list-inline>
	<li>5.03419823027391</li>
	<li>2.92237122641746</li>
	<li>3.34367052678603</li>
</ol>
</dd>
	<dt>$average.distance</dt>
		<dd><ol class=list-inline>
	<li>1.17515504803729</li>
	<li>1.19706068909887</li>
	<li>1.30771576344607</li>
</ol>
</dd>
	<dt>$median.distance</dt>
		<dd><ol class=list-inline>
	<li>0.988417748745722</li>
	<li>1.15598869239352</li>
	<li>1.23835312822661</li>
</ol>
</dd>
	<dt>$separation</dt>
		<dd><ol class=list-inline>
	<li>1.55335915855837</li>
	<li>0.133389398453682</li>
	<li>0.133389398453682</li>
</ol>
</dd>
	<dt>$average.toother</dt>
		<dd><ol class=list-inline>
	<li>3.64791248751346</li>
	<li>2.67429770843926</li>
	<li>3.0812116450103</li>
</ol>
</dd>
	<dt>$separation.matrix</dt>
		<dd><table>
<tbody>
	<tr><td>0.000000 </td><td>1.5533592</td><td>2.4150235</td></tr>
	<tr><td>1.553359 </td><td>0.0000000</td><td>0.1333894</td></tr>
	<tr><td>2.415024 </td><td>0.1333894</td><td>0.0000000</td></tr>
</tbody>
</table>
</dd>
	<dt>$ave.between.matrix</dt>
		<dd><table>
<tbody>
	<tr><td>0.000000</td><td>3.221129</td><td>4.129179</td></tr>
	<tr><td>3.221129</td><td>0.000000</td><td>2.092563</td></tr>
	<tr><td>4.129179</td><td>2.092563</td><td>0.000000</td></tr>
</tbody>
</table>
</dd>
	<dt>$average.between</dt>
		<dd>3.13070835203233</dd>
	<dt>$average.within</dt>
		<dd>1.22443073204046</dd>
	<dt>$n.between</dt>
		<dd>7491</dd>
	<dt>$n.within</dt>
		<dd>3684</dd>
	<dt>$max.diameter</dt>
		<dd>5.03419823027391</dd>
	<dt>$min.separation</dt>
		<dd>0.133389398453682</dd>
	<dt>$within.cluster.ss</dt>
		<dd>138.888359717351</dd>
	<dt>$clus.avg.silwidths</dt>
		<dd><dl class=dl-horizontal>
	<dt>1</dt>
		<dd>0.636316174439295</dd>
	<dt>2</dt>
		<dd>0.393377210558143</dd>
	<dt>3</dt>
		<dd>0.347392234026205</dd>
</dl>
</dd>
	<dt>$avg.silwidth</dt>
		<dd>0.459948239205186</dd>
	<dt>$g2</dt>
		<dd>NULL</dd>
	<dt>$g3</dt>
		<dd>NULL</dd>
	<dt>$pearsongamma</dt>
		<dd>0.679695973313445</dd>
	<dt>$dunn</dt>
		<dd>0.0264966519696275</dd>
	<dt>$dunn2</dt>
		<dd>1.60016634762696</dd>
	<dt>$entropy</dt>
		<dd>1.09741156762931</dd>
	<dt>$wb.ratio</dt>
		<dd>0.391103416338865</dd>
	<dt>$ch</dt>
		<dd>241.904401701832</dd>
	<dt>$cwidegap</dt>
		<dd><ol class=list-inline>
	<li>1.38922509363362</li>
	<li>0.782450764228991</li>
	<li>0.943224898892106</li>
</ol>
</dd>
	<dt>$widestgap</dt>
		<dd>1.38922509363362</dd>
	<dt>$sindex</dt>
		<dd>0.352481197744096</dd>
	<dt>$corrected.rand</dt>
		<dd>NULL</dd>
	<dt>$vi</dt>
		<dd>NULL</dd>
</dl>



#### Cluster statistics for PAM clustering


```R
# Statistics for pam clustering
pam_stats <- cluster.stats(dd,  pam.res$cluster)
# (pam) within clusters sum of squares
pam_stats$within.cluster.ss
```


140.285581233191



```R
# (pam) cluster average silhouette widths
pam_stats$clus.avg.silwidths
```


<dl class=dl-horizontal>
	<dt>1</dt>
		<dd>0.634639677757573</dd>
	<dt>2</dt>
		<dd>0.349633243416986</dd>
	<dt>3</dt>
		<dd>0.382381690873668</dd>
</dl>



#### Cluster statistics for hierarchical clustering


```R
# Statistics for hierarchical clustering
hc_stats <- cluster.stats(dd,  res.hc$cluster)
# (HCLUST) within clusters sum of squares
hc_stats$within.cluster.ss
```


147.883747407715



```R
# (HCLUST) cluster average silhouette widths
hc_stats$clus.avg.silwidths
```


<dl class=dl-horizontal>
	<dt>1</dt>
		<dd>0.634745619407596</dd>
	<dt>2</dt>
		<dd>0.436705035030708</dd>
	<dt>3</dt>
		<dd>0.321122109188664</dd>
</dl>



### External clustering validation

The aim is to compare the identified clusters (by k-means, pam or hierarchical clustering) to a reference.



```R
table(iris$Species, km.res$cluster)
```


                
                  1  2  3
      setosa     50  0  0
      versicolor  0 39 11
      virginica   0 14 36


It can be seen that:
<br>All setosa species (n = 50) has been classified in cluster 1
<br>A large number of versicor species (n = 39 ) has been classified in cluster 3. Some of them ( n = 11) have been classified in cluster 2.
<br>A large number of virginica species (n = 36 ) has been classified in cluster 2. Some of them (n = 14) have been classified in cluster 3.

It’s possible to quantify the agreement between Species and k-means clusters using either the corrected Rand index and Meila’s VI provided as follow:


```R
library("fpc")
# Compute cluster stats
species <- as.numeric(iris$Species)
clust_stats <- cluster.stats(d = dist(iris.scaled), 
                             species, km.res$cluster)
# Corrected Rand index
clust_stats$corrected.rand
```


0.620135180887038



```R
clust_stats$vi
```


0.74777490695804


The corrected Rand index provides a measure for assessing the similarity between two partitions, adjusted for chance. Its range is -1 (no agreement) to 1 (perfect agreement). Agreement between the specie types and the cluster solution is 0.62 using Rand index and 0.748 using Meila’s VI

The same analysis can be computed for both pam and hierarchical clustering:


```R
# Agreement between species and pam clusters
table(iris$Species, pam.res$cluster)
```


                
                  1  2  3
      setosa     50  0  0
      versicolor  0  9 41
      virginica   0 36 14



```R
cluster.stats(d = dist(iris.scaled), 
              species, pam.res$cluster)$vi
```


0.712903447402617



```R
# Agreement between species and HC clusters
table(iris$Species, res.hc$cluster)
```


                
                  1  2  3
      setosa     49  1  0
      versicolor  0 27 23
      virginica   0  2 48



```R
cluster.stats(d = dist(iris.scaled), 
              species, res.hc$cluster)$vi
```


0.694497589734848


External clustering validation, can be used to select suitable clustering algorithm for a given dataset.

### Comparison of the above methods

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

#### Thank you
