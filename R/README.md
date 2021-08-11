# ML Best Practices Standardized Codes-R

[Guide for .ipynb file download](#How-to-Download-a-single-file-without-downloading-the-entire-repository)<br>
[Guide for using R notebooks in your system](https://docs.anaconda.com/anaconda/navigator/tutorials/r-lang/#:~:text=To%20install%20and%20run%20R%20in%20a%20Jupyter,following%20code%20into%20the%20first%20...%20More%20items)

## Table of Contents

1. [**EDA**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/R/1.%20EDA)
  * Exploratory Data Analysis
  * Univariate Analysis
    * Categorical Variables
      -Pie Chart
      -Bar Plot
      -Frequency Table
    * Numerical Variables
      * Histograms
      * Density Plot
      * Boxplot
      * Table
  * Bivariate analysis
    * Numerical- Numerical
      * Scatter plot
      * Line Plot
    * Categorical-Numerical
      * Bar Plot
      * Box Plot
      * Violin Plot
      * T Test
      * Chi Square Test
    * Categorical-Categorical
      * Stacked bar chart
      * Grouped bar chart
      * Grouped kernel density plot
      * Ridgeline Plot
      * Z-Test
  * Multivariate Analysis
    * Grouping
    * Faceting
    * Correlation
    * Mosaic Plots
    * 3D Scatter Plot
    * Bubble Plot
    * Scatter Plot Matrix
    * Chi Square Test(categorical data)
  * Outlier Treatment 
    * Z Score
    * IQR Score
    * Removing outliers - quick & dirty
    
2. [**Pre-Processing**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/R/2.%20Pre-Processing)
  * Data Preprocessing
  * Missing Values Treatment
	   * Missing Value Treatment with mean
	   * Forward and Backward fill
	   * Nearest neighbors imputation
	   * Multivaraiate Imputation
  * Data Transformations
	   * Scale Transformation
	   * Centre Transformation
	   * Standardize Transformation
	   * Data Normalization
	   * Box-Cox Transform
	   * Yeo-Johnson Transform
	   * Principal Component Analysis
	   * Tips For Data Transforms
  * Encoding
	   * Handling Categorical Variable
	   * One Hot Encoding
	   * Label Encoding
	   * Hashing
  * Embedding
	   * Embedding
	   * CountVectorizer
	   * TF-IDF Vectorizer
	   * Stemming
	   * Lemmatization
3. [**Pre-Modelling**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/R/3.%20Pre-Modelling)
  * Sampling
    * Simple Random Sampling
    * Stratified Sampling
    * Random Undersampling and Oversampling
    * Undersampling
    * Oversampling
    * Tomek Links
    * SMOTE
    * ADASYN
  * Feature Importance
    * Removing Highly Corelated Variables
    * Boruta
    * Variance Inflation Factor (VIF)
    * Principal Component Analysis (PCA)
    * Linear Discriminant Analysis (LDA)
    * Feature importance
    * Chi Square Test
    
4a. [**Classification**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/R/4a.%20Classification)
  * Classification
    * Logistic Regression Classifier
    * Support Vector Classifier
    * Naive Bayes Classifier
    * Multinomial Naive Bayes Classifier
    * Gradient Boosting Classifier
    * XGBoost Classifier
    * Gradient Descent
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
    * Decision Tree Classifier
    * Random Forest
    * Gradient Boosting
    * XGBoost Classifier
    * KNN Classifier
  * Multi-label Classification
    * Binary Relevance 
    * Label Powerset
    * Classifier Chains
  * Hyperparameter Tuning
    * Grid-Search
    * Random-Search
    
4b. [**Test Classification**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/R/4b.%20Text_Classification)
  * Logistic Regression Classifier
  * Support Vector Classifier
  * Gradient Boosting Classifier
  * XGBoost Classifier
  * Gradient Descen
  * Stochastic Gradient Descent
  * Decision Tree
  * Random Forest Classifier
  * KNN Classifier
  * Classifiers Report
      
4c. [**Regression**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/R/4c.%20Regression)
  * Multiple Linear Regression
  * Polynomial Regression
  * Quantile Regression
  * Ridge Regression
  * Lasso Regression
  * Elastic Net Regression
  * Support Vector Regression
  * Decision Tree - CART
  * Random Forest Regression
  * Gradient Boosting (GBM)
  * Stochastic Gradient Descent
  * KNN Regressor
  * XGB Regressor
  * Regressors Report
  * Cross Validation
  * Grid Search CV
  * Random Search CV
   
4d. [**TimeSeries**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/R/4d.%20TimeSeries)   
  * Time Series forecasting
    * Time Series
    * Time Serires Plot
  * Stationarity of a Time Series
    * Test for Stationary
    * Differencing
    * Decomposing
  * Univariate Forecasting techniques
    * Naive Approach
    * Simple Moving Average
    * Exponential Smoothing
    * Holt’s Method
    * Holt-Winters’ Method
    * ACF & PACF
    * ARIMA Model
    * Seasonal ARIMA
    * TBATS model 
  * Multivariate Time Series
    * Multivariate Time Series – VAR
    
4e. [**Anomaly Detection**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/R/4e.%20Anomaly%20Detection)   
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
	
4f. [**Association Rules**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/R/4f.%20Association%20Rules)   
  * Apriori Algorithm
  * FP Growth Algorithm
	
4g. [**Bayesian**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/R/4g.%20Bayesian)   
  * Bayesian Time Series 
  * Bayesian Belief Network
	
4h. [**Clustering**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/R/4h.%20Clustering)  
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
	
4i. [**Dimensionality Reduction Technique**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/R/4i.%20Dimensionality%20Reduction%20Technique)   
  * PCA
  * LDA
  * SVD
  * PLSR
  * t-SNE
  * Factor Analysis
  * Isomap

4j. [**Recommender System**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/R/4j.%20Recommender%20System)   
 * Content based filtering
 * Collaborative filtering
   * Item based Collaborative filtering
   * User based Collaborative filtering
 * Hybrid recommender system
	
5. [**Post-Modelling**](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/tree/master/R/5.Post-Modelling)   
  * Evaluation Metrics for Regression
    * Mean Absolute Error
    * Mean Squared Error
    * Mean Squared Logarithmic Error
    * Median Absolute Error
    * Mean Percentage Error
    * Mean Absolute Percentage Error
    * Weighted Mean Absolute Percentage Error
    * Metric Selection
    * Cross Validation
  * Accuracy Metrics for Binary Classification
    * Confusion Matrix
    * Accuracy and Cohen’s kappa
    * Recall
    * Precision
    * F1 Score
    * ROC (Receiver Operating Characteristics)
    * AUC (Area Under the Curve)
  * Accuracy Metrics for Multi-Class Classification
    * Accuarcy Score
    * Confusion Matrix
    * Logarithmic Loss
    * ROC and AUC
    * Precision Recall Curve
    
    
# How to Download a single file without downloading the entire repository
Please follow the following step to download any particular notebook without downloading the entire repository:
   1. Copy the raw codes from the desired notebook
   2. Paste the coded in a text editor, eg notepad++
   3. Go to save as and save the file as <name>.ipynb in the desired location
	
By providing the .ipynb extension with the name you can save it as a notebook even if the file type is not available in the drop down list.


 
