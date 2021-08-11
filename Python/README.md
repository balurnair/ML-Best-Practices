# ML Best Practices Standardized Codes-Python

[Guide for .ipynb file download](#How-to-Download-a-single-file-without-downloading-the-entire-repository)<br>

# Table of Contents
[1. **EDA**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/Python/1.EDA)
* Univariate Analysis
    * Categorical Variables
        - Pie Chart
        - Bar Plot
        - Frequency Table
    * Numerical Variables
        - Histograms
        - Box Plot
        - Density Plot
        - Table    

* Bivariate Analysis
    * Numerical-Numerical 
        - Scatter plot
    * Categorical-Numerical
        - Bar Plots
        - Box Plots
    * Categorical-Categorical
        - Cross Tab
        - Z Test
        - T Test
        - Chi Square Test
* Multivariate Analysis  
    * Correlation
    * Combination Chart
    * Scatter Plot
    * Pair Plot
* Outlier Treatment
    * Z Score
    * IQR Score
    
[2. **Pre-Processing**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/Python/2.Pre-Processing)
* Missing Value Treatment
    * Missing Value Treatment with mean
    * Forward and Backward fill
    * Nearest neighbors imputation
    * MultiOutput Regressor
    * IterativeImpute
    * Time-Series Specific Methods
* Rescalling
    * MinMax Scaler
    * MaxAbs Scaler
    * Robust Scaler
    * StandardScaler
* Data Transformation
    * Quantile Transformation
    * Power Transformation
    * Custom Transformation
* Data Normalization
* Handling Categorical Variable
    * One Hot Encoding
    * Label Encoding
    * Hashing
    * Backward Difference Encoding
* Embedding
    * CountVectorizer
    * DictVectorizer
    * TF-IDF Vectorizer
    * Stemming
    * Lemmatization
    * Word2Vec
    * Doc2Vec
    * Visualize Word Embedding

[3. **Pre Modelling**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/Python/3.Pre-Modelling)
* Sampling
    * Simple Random Sampling
    * Stratified Sampling
    * Random Undersampling and Oversampling
    * Undersampling
    * Oversampling
    * Reservoir Sampling
    * Undersampling and Oversampling Using Imbalanced-learn
        * Undersampling Methods
            * Cluster Centroids
            * Tomek Links
        * Oversampling Methods
            * SMOTE
            * ADASYN
            * Borderline Smote
            * SVM SMOTE
            * KMeans SMOTE
* Feature Importance
    * Univariate Selection
    * Recursive Feature Elimination
    * Removing Highly Corelated Variables
    * Boruta
    * Variance Inflation Factor (VIF)
    * Principal Component Analysis (PCA)
    * Linear Discriminant Analysis (LDA)
    * Feature Importance Using Random Forest



[4a. **Classification**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/Python/4a.Classification)
* Overview
* Library and Data
* Binary Classification
    * Logistic Regression Classifier
    * Support Vector Classifier
    * Multinomial Naive Bayes Classifier
    * Bernoulli Naive Bayes Classifier
    * Gradient Boost Classifier
    * XGBoost Classifier
    * Stochastic Gradient Descent
    * Decision Tree
    * Random Forest Classifier
    * KNN Classifier
* Classifiers Report
* Multiclass Classification
    * Multiclass Logistic Regression
    * Support Vector Classifier
    * Multinomial Naive Bayes Classifier
    * Bernoulli Naive Bayes Classifier
    * Gaussian Naive Bayes Classifier
    * Decision Tree Classifier
    * Random Forest
    * Gradient Boosting
    * Extreme Gradient Boosting Classifier
    * Stochastic Gradient Descent
    * KNN Classifier
* Classifiers Report (multiclass classiification)
* Hyperparameter Tuning
    * Grid-Search 
    * Random Search
    * Bayesian Optimization
* Multi-Label Classification
    * Data Preprocessing for Multi-label Classification
    * OneVsRest
    * Binary Relevance
    * Classifier Chains
    * Label Powerset
    * Adapted Algorithm

[4b. **Text Classification**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/Python/4b.%20Text_Classification)
* Library and Data
* Logistic Regression Classifier
* Support Vector Classifier
* Multinomial Naive Bayes Classifier
* Bernoulli Naive Bayes Classifier
* Gradient Boost Classifier
* Extreme Gradient Boosting Classifier
* Random Forest Classifier

[4c. **Regression**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/Python/4c.%20Regression)
* Overview & Data
* Linear Regression
* Polynomial Regression
* Quantile Regression
* Ridge Regression
* Lasso Regression
* Elastic Net Regression
* Multiple Linear Regression
* Support Vector Regression
* Decision Tree - CART
* Random Forest Regression
* Gradient Boosting (GBM)
* Stochastic Gradient Descent
* KNN Regressor
* Extreme Gradient Boosting Regressor
* Light Gradient Boosting Machine
* Regressors Report
* Hyperparameter Tuning
    * Grid-Search
    * Random Search
    * Bayesian Optimization

[4d. **Time Series**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/Python/4d.%20TimeSeries)
* Overview
* Time Serires Plot
* Test for Stationary
* Differencing
* Decomposing
* Univariate Forecasting techniques
    * Naive Approach
    * Simple Average
    * Exponential Smoothing
    * Holt’s Method
    * Holt-Winters’ Method
    * ACF and PACF
    * ARIMA Model
    * Seasonal ARIMA
    * TBATS model
    * Univariate fbprophet
* Multivariate Time Series
    * Multivariate Time Series – VAR
    * Multivariate fbprophet
    
4e. [**Anomaly Detection**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/Python/4e.%20Anomaly%20Detection)   
  * One-Class SVM
  * Isolation Forests
  * Elliptic Envelope
  * DBSCAN
  * PCA Based Anomaly Detection
  * Local Outlier Factor
  * Feature Bagging
  * KNN
  * HBOS
  * CBLOF
	
4f. [**Association Rules**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/Python/4f.%20Association%20Rules)   
  * Apriori Algorithm
  * FP Growth Algorithm
	
4g. [**Bayesian**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/Python/4g.%20Bayesian)   
  * Bayesian Time Series 
  * Bayesian Belief Network
	
4h. [**Clustering**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/Python/4h.%20Clustering)  
  * Affinity Propagation
  * Agglomerative Clustering
  * BIRCH
  * DBSCAN
  * K-Means
  * Mini-Batch K-Means
  * Mean Shift
  * OPTICS
  * Spectral Clustering
  * Gaussian Mixture Model
  * K- Mode
  * K-Prototypes
	
4i. [**Dimensionality Reduction Technique**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/Python/4i.%20Dimensionality%20Reduction%20Technique)   
  * PCA
  * LDA
  * SVD
  * PLSR
  * t-SNE
  * Factor Analysis
  * Isomap

4j. [**Recommender System**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/Python/4j.%20Recommender%20System)   
 * Content based filtering
 * Collaborative filtering
   * Item based Collaborative filtering
   * User based Collaborative filtering
 * Hybrid recommender system

[5. **Post- Modelling**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/Python/5.%20Post-Modelling)
* Evaluation Metrics for Regression
    * Explained Variance Score
    * Max error
    * Mean Absolute Error
    * Mean Squared Error
    * Mean Squared Logarithmic Error
    * Median Absolute Error
    * R² score
    * Mean Percentage Error
    * Mean Absolute Percentage Error
    * Weighted Mean Absolute Percentage Error
    * Tips for Metric Selection
    * Cross Validation
    * StratifiedKFold
* Accuracy Metrics for Binary Classification
    * Cohen’s kappa
    * Hamming Loss
    * Confusion Matrix
    * Precision-Recall (PR) Curve
    * ROC (Receiver Operating Characteristics)
    * AUC (Area Under the Curve)
* Accuracy Metrics for Multi-Class Classification
    * Accuarcy Score
    * Confusion Matrix for MultiClass
    * ROC and AUC
    * Precision Recall Curve for Multiclass
    * Classification Report


# How to Download a single file without downloading the entire repository
Please follow the following step to download any particular notebook without downloading the entire repository:
   1. Copy the raw codes from the desired notebook
   2. Paste the coded in a text editor, eg notepad++
   3. Go to save as and save the file as <name>.ipynb in the desired location
	
By providing the .ipynb extension with the name you can save it as a notebook even if the file type is not available in the drop down list.

