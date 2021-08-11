# Pre Modeling

# Notebook Content
[Libraries used](#Library)<br> 
## Sampling
[Simple Random Sampling](#Simple-Random-Sampling)<br>
[Stratified Sampling](#Stratified-Sampling)<br>
[Random Undersampling and Oversampling](#Random-Undersampling-and-Oversampling)<br>
[Undersampling](#Under-Sampling)<br>
[Oversampling](#Over-Sampling)<br>
[Tomek Links](#Tomek-Links)<br>
[SMOTE](#SMOTE)<br>
[ADASYN](#ADASYN)<br>
## Feature Importance
[Removing Highly Corelated Variables](#Removing-Highly-Corelated-Variables)<br>
[Boruta](#Boruta)<br>
[Variance Inflation Factor (VIF)](#1)<br>
[Principal Component Analysis (PCA)](#2)<br>
[Linear Discriminant Analysis (LDA)](#3)<br>
[Feature importance](#4)<br>
[Chi Square Test](#Chi-Square-Test)

# Library


```python
# install.packages("DMwR")
# install.packages("tseries")
# install.packages("imbalance")
# install.packages("unbalanced")
# install.packages("FactoMineR")
# install.packages("UBL")
# install.packages("PRROC")
# install.packages("ROCR")
# install.packages('dplyr')
# install.packages("ROSE")
# install.packages('ggcorrplot')
# install.packages('VIF')
# install.packages('Boruta')
# install.packages('TH.data')
# install.packages('caret')
# install.packages('earth')
# install.packages('splitstackshape')
```

The download links for the automobile dataset that have been used in the notebook is given below. Please download the dataset before proceeding
https://www.kaggle.com/toramky/automobile-dataset

The other datasets that are present in the notebook are common in-built r datasets.


```python
auto_mobile = read.csv('dataset/auto_mobile.csv')
```


```python
head(auto_mobile)
```


<table>
<caption>A data.frame: 6 × 27</caption>
<thead>
	<tr><th></th><th scope=col>X</th><th scope=col>symboling</th><th scope=col>normalized.losses</th><th scope=col>make</th><th scope=col>fuel.type</th><th scope=col>aspiration</th><th scope=col>num.of.doors</th><th scope=col>body.style</th><th scope=col>drive.wheels</th><th scope=col>engine.location</th><th scope=col>...</th><th scope=col>engine.size</th><th scope=col>fuel.system</th><th scope=col>bore</th><th scope=col>stroke</th><th scope=col>compression.ratio</th><th scope=col>horsepower</th><th scope=col>peak.rpm</th><th scope=col>city.mpg</th><th scope=col>highway.mpg</th><th scope=col>price</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>...</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0</td><td>3</td><td>122</td><td>alfa-romero</td><td>gas</td><td>std</td><td>two </td><td>convertible</td><td>rwd</td><td>front</td><td>...</td><td>130</td><td>mpfi</td><td>3.47</td><td>2.68</td><td> 9.0</td><td>111</td><td>5000</td><td>21</td><td>27</td><td>13495</td></tr>
	<tr><th scope=row>2</th><td>1</td><td>3</td><td>122</td><td>alfa-romero</td><td>gas</td><td>std</td><td>two </td><td>convertible</td><td>rwd</td><td>front</td><td>...</td><td>130</td><td>mpfi</td><td>3.47</td><td>2.68</td><td> 9.0</td><td>111</td><td>5000</td><td>21</td><td>27</td><td>16500</td></tr>
	<tr><th scope=row>3</th><td>2</td><td>1</td><td>122</td><td>alfa-romero</td><td>gas</td><td>std</td><td>two </td><td>hatchback  </td><td>rwd</td><td>front</td><td>...</td><td>152</td><td>mpfi</td><td>2.68</td><td>3.47</td><td> 9.0</td><td>154</td><td>5000</td><td>19</td><td>26</td><td>16500</td></tr>
	<tr><th scope=row>4</th><td>3</td><td>2</td><td>164</td><td>audi       </td><td>gas</td><td>std</td><td>four</td><td>sedan      </td><td>fwd</td><td>front</td><td>...</td><td>109</td><td>mpfi</td><td>3.19</td><td>3.40</td><td>10.0</td><td>102</td><td>5500</td><td>24</td><td>30</td><td>13950</td></tr>
	<tr><th scope=row>5</th><td>4</td><td>2</td><td>164</td><td>audi       </td><td>gas</td><td>std</td><td>four</td><td>sedan      </td><td>4wd</td><td>front</td><td>...</td><td>136</td><td>mpfi</td><td>3.19</td><td>3.40</td><td> 8.0</td><td>115</td><td>5500</td><td>18</td><td>22</td><td>17450</td></tr>
	<tr><th scope=row>6</th><td>5</td><td>2</td><td>122</td><td>audi       </td><td>gas</td><td>std</td><td>two </td><td>sedan      </td><td>fwd</td><td>front</td><td>...</td><td>136</td><td>mpfi</td><td>3.19</td><td>3.40</td><td> 8.5</td><td>110</td><td>5500</td><td>19</td><td>25</td><td>15250</td></tr>
</tbody>
</table>



# Sampling
Data sampling refers to **statistical methods for selecting observations from the domain** with the objective of estimating a population parameter. Whereas data resampling refers to methods for economically using a collected dataset to improve the estimate of the population parameter and help to quantify the uncertainty of the estimate.

## Simple Random Sampling

Simple Random Sampling can be used to select a subset of a population in which each member of the subset has an equal probability of being chosen.
Below is a code to select 100 sample points from a dataset.


```python
auto_df <- auto_mobile
set.seed(0)
auto_df = auto_df[sample(nrow(auto_df), 100),]
dim(auto_mobile)
dim(auto_df)
```



<ol class=list-inline><li>203</li><li>27</li></ol>





<ol class=list-inline><li>100</li><li>27</li></ol>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 



## Stratified Sampling
Stratified random sampling of dataframe in R:
Sample_n() along with group_by() function is used to get the stratified random sampling of dataframe in R as shown below. The iris dataset is used:


```python
# install.packages("splitstackshape")
library(splitstackshape)
options(warn=-1)
```

    Warning message:
    "package 'splitstackshape' was built under R version 3.6.3"
    


```python
## Sample data
set.seed(1)
n <- 1e4
d <- data.frame(age = sample(1:5, n, T), 
                lc = rbinom(n, 1 , .5),
                ants = rbinom(n, 1, .7))

library(splitstackshape)
set.seed(1)
out <- stratified(d, c("age", "lc"), 30)
head(out)
```


<table>
<caption>A data.table: 6 × 3</caption>
<thead>
	<tr><th scope=col>age</th><th scope=col>lc</th><th scope=col>ants</th></tr>
	<tr><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><td>1</td><td>1</td><td>1</td></tr>
	<tr><td>1</td><td>1</td><td>1</td></tr>
	<tr><td>1</td><td>1</td><td>0</td></tr>
	<tr><td>1</td><td>1</td><td>0</td></tr>
	<tr><td>1</td><td>1</td><td>0</td></tr>
	<tr><td>1</td><td>1</td><td>1</td></tr>
</tbody>
</table>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 



```python
# or we can do by this method also
# install.packages("dplyr")
library(dplyr)
options(warn=-1)

set.seed(1)
out2 <- d %>%
  group_by(age, lc) %>%
  sample_n(30)
head(out2)
```

    
    Attaching package: 'dplyr'
    
    
    The following objects are masked from 'package:stats':
    
        filter, lag
    
    
    The following objects are masked from 'package:base':
    
        intersect, setdiff, setequal, union
    
    
    


<table>
<caption>A grouped_df: 6 × 3</caption>
<thead>
	<tr><th scope=col>age</th><th scope=col>lc</th><th scope=col>ants</th></tr>
	<tr><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><td>1</td><td>0</td><td>1</td></tr>
	<tr><td>1</td><td>0</td><td>1</td></tr>
	<tr><td>1</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>0</td></tr>
	<tr><td>1</td><td>0</td><td>1</td></tr>
</tbody>
</table>



here it can be observed that species have equa number of possibilities

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 



# Random Undersampling and Oversampling

**Imbalanced data:** A dataset is imbalanced if at least one of the classes constitutes only a very small minority. Imbalanced data prevail in banking, insurance, engineering, and many other fields. It is common in fraud detection that the imbalance is on the order of 100 to 1

The issue of class imbalance can result in a serious bias towards the majority class, reducing the classification performance and increasing the number of false negatives. How can we alleviate the issue? The most commonly used techniques are data resampling either under-sampling the majority of the class, or oversampling the minority class, or a mix of both. This will result in improved classification performance.

**The Problem with Imbalanced Classes**
Most machine learning algorithms work best when the number of samples in each class are about equal. This is because most algorithms are designed to maximize accuracy and reduce error.

![Undersampling%20and%20oversampling.png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Pre-Modeling/Undersampling%20and%20oversampling.png)

**It is too often that we encounter an imbalanced dataset**<br>
A widely adopted technique for dealing with highly imbalanced datasets is called resampling. It consists of removing samples from the majority class (under-sampling) and/or adding more examples from the minority class (over-sampling).


```python
# install.packages("ROSE")
library(ROSE)
options(warn=-1)
# rose library contains inbuilt data set called hacide
data(hacide)
str(hacide.train)
#check table
table(hacide.train$cls)
# can also represent it in a visual form 
# barplot(table(hacide.train$cls))
```

    Loaded ROSE 0.0-3
    
    
    

    'data.frame':	1000 obs. of  3 variables:
     $ cls: Factor w/ 2 levels "0","1": 1 1 1 1 1 1 1 1 1 1 ...
     $ x1 : num  0.2008 0.0166 0.2287 0.1264 0.6008 ...
     $ x2 : num  0.678 1.5766 -0.5595 -0.0938 -0.2984 ...
    


    
      0   1 
    980  20 


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


## Over Sampling


```python
#over sampling
data_balanced_over <- ovun.sample(cls ~ ., data = hacide.train, method = "over",N = 1960)$data
table(data_balanced_over$cls)
```


    
      0   1 
    980 980 


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


## Under Sampling


```python
data_balanced_under <- ovun.sample(cls ~ ., data = hacide.train, method = "under", N = 40, seed = 1)$data
table(data_balanced_under$cls)
```


    
     0  1 
    20 20 


Now the data set is balanced. But, you see that we’ve lost significant information from the sample. Let’s do both under-sampling and oversampling on this imbalanced data. This can be achieved using method = “both“. In this case, the minority class is oversampled with replacement and majority class is under-sampled without replacement.


```python
data_balanced_both <- ovun.sample(cls ~ ., data = hacide.train, method = "both", p=0.5,N=1000, seed = 1)$data
table(data_balanced_both$cls)
```


    
      0   1 
    520 480 


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


##  Tomek Links
The most coming under sampling method is the Tomek Links method and hence we will be showing it in detail.<br>
Tomek links are pairs of examples of opposite classes in close vicinity.<br>
In this algorithm, we end up removing the majority element from the Tomek link which provides a better decision boundary for a classifier.<br>

![Tomeklinks.png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Pre-Modeling/Tomeklinks.png)


```python
# install.packages("DMwR")
# install.packages("UBL")
library(DMwR)
options(warn=-1)

library(UBL)
options(warn=-1)

data(algae)
options(warn=-1)

clean.algae <- algae[complete.cases(algae), ]
alg.HVDM1 <- TomekClassif(season~., clean.algae, dist = "HVDM", 
                          Cl = c("winter", "spring"), rem = "both")
alg.HVDM2 <- TomekClassif(season~., clean.algae, dist = "HVDM", rem = "maj")
# removes only examples from class summer which are the majority class in the link
alg.EuM <- TomekClassif(season~., clean.algae, dist = "HEOM", 
                        Cl = "summer", rem = "maj")

# removes only examples from class summer in every link they appear
alg.EuB <- TomekClassif(season~., clean.algae, dist = "HEOM",
                        Cl = "summer", rem = "both")
                          
summary(clean.algae$season)
summary(alg.HVDM1[[1]]$season)
summary(alg.HVDM2[[1]]$season)
summary(alg.EuM[[1]]$season)
summary(alg.EuB[[1]]$season)
  
# check which were the indexes of the examples removed in alg.EuM
alg.EuM[[2]]

```


<dl class=dl-inline><dt>autumn</dt><dd>36</dd><dt>spring</dt><dd>48</dd><dt>summer</dt><dd>43</dd><dt>winter</dt><dd>57</dd></dl>




<dl class=dl-inline><dt>autumn</dt><dd>36</dd><dt>spring</dt><dd>31</dd><dt>summer</dt><dd>43</dd><dt>winter</dt><dd>36</dd></dl>




<dl class=dl-inline><dt>autumn</dt><dd>36</dd><dt>spring</dt><dd>43</dd><dt>summer</dt><dd>37</dd><dt>winter</dt><dd>36</dd></dl>




<dl class=dl-inline><dt>autumn</dt><dd>36</dd><dt>spring</dt><dd>48</dd><dt>summer</dt><dd>37</dd><dt>winter</dt><dd>57</dd></dl>




<dl class=dl-inline><dt>autumn</dt><dd>36</dd><dt>spring</dt><dd>48</dd><dt>summer</dt><dd>27</dd><dt>winter</dt><dd>57</dd></dl>




<ol class=list-inline><li>7</li><li>35</li><li>75</li><li>95</li><li>165</li><li>180</li></ol>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


## SMOTE
In SMOTE (Synthetic Minority Oversampling Technique) we synthesize elements for the minority class, in the vicinity of already existing elements.<br>
Supports multi-class resampling.

![smote.png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Pre-Modeling/SMOTE.png)


```python
## A small example with a data set created artificially from the IRIS
## data 
# install.packages("imbalance")
library(imbalance)
data(iris)
# head(iris)
# Oversample glass0 to get an imbalance ratio of 0.8
imbalanceRatio(iris, classAttr = "Species")

newDataset <- oversample(iris, classAttr = "Species", ratio = 0.8, method = "SMOTE")
imbalanceRatio(newDataset, classAttr = "Species")
# # }

## Checking visually the created data
p<-par(mfrow = c(1, 2))
plot(iris[, 1], iris[, 2], pch = 19 + as.integer(iris[, 3]),
     main = "Original Data")
plot(newDataset[, 1], newDataset[, 2], pch = 19 + as.integer(newDataset[,3]),
     main = "SMOTE'd Data")


```


0.5



0.384615384615385



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Pre-Modeling/output_39_2.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


## ADASYN
We can perform over-sampling using Adaptive Synthetic (ADASYN) sampling approach for imbalanced datasets.


```python
# install.packages("imbalance")
library(imbalance)
data(glass0)

# imbalance ratio 0.8
imbalanceRatio(glass0)
```


0.486111111111111



```python
head(glass0)
```


<table>
<caption>A data.frame: 6 × 10</caption>
<thead>
	<tr><th></th><th scope=col>RI</th><th scope=col>Na</th><th scope=col>Mg</th><th scope=col>Al</th><th scope=col>Si</th><th scope=col>K</th><th scope=col>Ca</th><th scope=col>Ba</th><th scope=col>Fe</th><th scope=col>Class</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;fct&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>1.515888</td><td>12.87795</td><td>3.43036</td><td>1.40066</td><td>73.2820</td><td>0.68931</td><td> 8.04468</td><td>0</td><td>0.1224</td><td>positive</td></tr>
	<tr><th scope=row>2</th><td>1.517642</td><td>12.97770</td><td>3.53812</td><td>1.21127</td><td>73.0020</td><td>0.65205</td><td> 8.52888</td><td>0</td><td>0.0000</td><td>positive</td></tr>
	<tr><th scope=row>3</th><td>1.522130</td><td>14.20795</td><td>3.82099</td><td>0.46976</td><td>71.7700</td><td>0.11178</td><td> 9.57260</td><td>0</td><td>0.0000</td><td>positive</td></tr>
	<tr><th scope=row>4</th><td>1.522221</td><td>13.21045</td><td>3.77160</td><td>0.79076</td><td>71.9884</td><td>0.13041</td><td>10.24520</td><td>0</td><td>0.0000</td><td>positive</td></tr>
	<tr><th scope=row>5</th><td>1.517551</td><td>13.39000</td><td>3.65935</td><td>1.18880</td><td>72.7892</td><td>0.57132</td><td> 8.27064</td><td>0</td><td>0.0561</td><td>positive</td></tr>
	<tr><th scope=row>6</th><td>1.520991</td><td>13.68925</td><td>3.59200</td><td>1.12139</td><td>71.9604</td><td>0.08694</td><td> 9.40044</td><td>0</td><td>0.0000</td><td>positive</td></tr>
</tbody>
</table>




```python
newDataset <- oversample(glass0, method = "ADASYN")
imbalanceRatio(newDataset)
```


0.972222222222222



```python
par(mfrow = c(1, 2))
plot(glass0[ ,1], glass0[ ,2], pch = 19 + as.integer(glass0[, 10]),
     main = "Original Data")
plot(glass0[ ,1], glass0[ ,2], pch = 19 + as.integer(newDataset[, 10]),
     main = "ADASYN'd Data")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Pre-Modeling/output_45_0.png)


• While the RandomOverSampler is over-sampling by duplicating some of the original samples of the minority class, **SMOTE and ADASYN generate new samples in by interpolation**<br>
• The samples used to interpolate/generate new synthetic samples differ. In fact, ADASYN focuses on generating samples next to the original samples which are wrongly classified using a k-Nearest Neighbors classifier while the basic implementation of SMOTE will not make any distinction between easy and hard samples to be classified using the nearest neighbors rule <br>


![ADASYN-SMOTE.png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Pre-Modeling/ADASYN%20and%20SMOTE.png)

**SMOTE might connect inliers and outliers while ADASYN might focus solely on outliers which, in both cases, might lead to a sub-optimal decision function**.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


# Feature Importance 
The performance of machine learning model is directly proportional to the data features used to train it. The performance of ML model will be affected negatively if the data features provided to it are irrelevant. On the other hand, **use of relevant data features can increase the accuracy of your ML model** especially linear and logistic regression.

Performing feature selection before data modeling will reduce the overfitting.<br>
Performing feature selection before data modeling will increases the accuracy of ML model.<br>
Performing feature selection before data modeling will reduce the training time.<br>

## Removing Highly Corelated Variables

Correlation is a statistical term which in common usage refers to how close two variables are to having a linear relationship with each other.
<br>For example, two variables which are linearly dependent (say, x and y which depend on each other as x = 2y) will have a higher correlation than two variables which are non-linearly dependent (say, u and v which depend on each other as u = v^2)

**How does correlation help in feature selection?**
<br>Features with high correlation are more linearly dependent and hence have almost the same effect on the dependent variable(multicollinearity). So, when two features have high correlation, we can drop one of the two features<br>
We will be showing how it works using the data for breast cancer.<br> The link to download dataset:https://www.kaggle.com/uciml/breast-cancer-wisconsin-data


```python
num_cols <- unlist(lapply(auto_mobile, is.numeric))         # Identify numeric columns
data <- auto_mobile[ , num_cols]                        # Subset numeric columns of data 
r <- cor(data, use="complete.obs")
round(r,2)
```


<table>
<caption>A matrix: 17 × 17 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>X</th><th scope=col>symboling</th><th scope=col>normalized.losses</th><th scope=col>wheel.base</th><th scope=col>length</th><th scope=col>width</th><th scope=col>height</th><th scope=col>curb.weight</th><th scope=col>engine.size</th><th scope=col>bore</th><th scope=col>stroke</th><th scope=col>compression.ratio</th><th scope=col>horsepower</th><th scope=col>peak.rpm</th><th scope=col>city.mpg</th><th scope=col>highway.mpg</th><th scope=col>price</th></tr>
</thead>
<tbody>
	<tr><th scope=row>X</th><td> 1.00</td><td>-0.13</td><td>-0.22</td><td> 0.12</td><td> 0.16</td><td> 0.04</td><td> 0.24</td><td> 0.06</td><td>-0.06</td><td> 0.26</td><td>-0.17</td><td> 0.16</td><td>-0.02</td><td>-0.18</td><td> 0.00</td><td> 0.00</td><td>-0.12</td></tr>
	<tr><th scope=row>symboling</th><td>-0.13</td><td> 1.00</td><td> 0.45</td><td>-0.53</td><td>-0.36</td><td>-0.24</td><td>-0.51</td><td>-0.23</td><td>-0.06</td><td>-0.13</td><td>-0.01</td><td>-0.17</td><td> 0.07</td><td> 0.22</td><td> 0.02</td><td> 0.08</td><td>-0.08</td></tr>
	<tr><th scope=row>normalized.losses</th><td>-0.22</td><td> 0.45</td><td> 1.00</td><td>-0.04</td><td> 0.03</td><td> 0.09</td><td>-0.35</td><td> 0.11</td><td> 0.14</td><td>-0.03</td><td> 0.05</td><td>-0.11</td><td> 0.20</td><td> 0.21</td><td>-0.20</td><td>-0.16</td><td> 0.14</td></tr>
	<tr><th scope=row>wheel.base</th><td> 0.12</td><td>-0.53</td><td>-0.04</td><td> 1.00</td><td> 0.88</td><td> 0.80</td><td> 0.59</td><td> 0.78</td><td> 0.57</td><td> 0.49</td><td> 0.18</td><td> 0.25</td><td> 0.36</td><td>-0.35</td><td>-0.50</td><td>-0.57</td><td> 0.58</td></tr>
	<tr><th scope=row>length</th><td> 0.16</td><td>-0.36</td><td> 0.03</td><td> 0.88</td><td> 1.00</td><td> 0.84</td><td> 0.49</td><td> 0.88</td><td> 0.69</td><td> 0.60</td><td> 0.13</td><td> 0.15</td><td> 0.56</td><td>-0.28</td><td>-0.71</td><td>-0.74</td><td> 0.69</td></tr>
	<tr><th scope=row>width</th><td> 0.04</td><td>-0.24</td><td> 0.09</td><td> 0.80</td><td> 0.84</td><td> 1.00</td><td> 0.28</td><td> 0.87</td><td> 0.75</td><td> 0.56</td><td> 0.18</td><td> 0.18</td><td> 0.65</td><td>-0.22</td><td>-0.67</td><td>-0.70</td><td> 0.73</td></tr>
	<tr><th scope=row>height</th><td> 0.24</td><td>-0.51</td><td>-0.35</td><td> 0.59</td><td> 0.49</td><td> 0.28</td><td> 1.00</td><td> 0.29</td><td> 0.02</td><td> 0.17</td><td>-0.05</td><td> 0.26</td><td>-0.11</td><td>-0.27</td><td>-0.11</td><td>-0.16</td><td> 0.14</td></tr>
	<tr><th scope=row>curb.weight</th><td> 0.06</td><td>-0.23</td><td> 0.11</td><td> 0.78</td><td> 0.88</td><td> 0.87</td><td> 0.29</td><td> 1.00</td><td> 0.86</td><td> 0.65</td><td> 0.18</td><td> 0.16</td><td> 0.75</td><td>-0.27</td><td>-0.78</td><td>-0.82</td><td> 0.82</td></tr>
	<tr><th scope=row>engine.size</th><td>-0.06</td><td>-0.06</td><td> 0.14</td><td> 0.57</td><td> 0.69</td><td> 0.75</td><td> 0.02</td><td> 0.86</td><td> 1.00</td><td> 0.59</td><td> 0.21</td><td> 0.03</td><td> 0.83</td><td>-0.21</td><td>-0.72</td><td>-0.73</td><td> 0.88</td></tr>
	<tr><th scope=row>bore</th><td> 0.26</td><td>-0.13</td><td>-0.03</td><td> 0.49</td><td> 0.60</td><td> 0.56</td><td> 0.17</td><td> 0.65</td><td> 0.59</td><td> 1.00</td><td>-0.07</td><td> 0.00</td><td> 0.58</td><td>-0.26</td><td>-0.60</td><td>-0.60</td><td> 0.54</td></tr>
	<tr><th scope=row>stroke</th><td>-0.17</td><td>-0.01</td><td> 0.05</td><td> 0.18</td><td> 0.13</td><td> 0.18</td><td>-0.05</td><td> 0.18</td><td> 0.21</td><td>-0.07</td><td> 1.00</td><td> 0.20</td><td> 0.09</td><td>-0.07</td><td>-0.04</td><td>-0.05</td><td> 0.10</td></tr>
	<tr><th scope=row>compression.ratio</th><td> 0.16</td><td>-0.17</td><td>-0.11</td><td> 0.25</td><td> 0.15</td><td> 0.18</td><td> 0.26</td><td> 0.16</td><td> 0.03</td><td> 0.00</td><td> 0.20</td><td> 1.00</td><td>-0.20</td><td>-0.44</td><td> 0.31</td><td> 0.25</td><td> 0.07</td></tr>
	<tr><th scope=row>horsepower</th><td>-0.02</td><td> 0.07</td><td> 0.20</td><td> 0.36</td><td> 0.56</td><td> 0.65</td><td>-0.11</td><td> 0.75</td><td> 0.83</td><td> 0.58</td><td> 0.09</td><td>-0.20</td><td> 1.00</td><td> 0.13</td><td>-0.81</td><td>-0.78</td><td> 0.76</td></tr>
	<tr><th scope=row>peak.rpm</th><td>-0.18</td><td> 0.22</td><td> 0.21</td><td>-0.35</td><td>-0.28</td><td>-0.22</td><td>-0.27</td><td>-0.27</td><td>-0.21</td><td>-0.26</td><td>-0.07</td><td>-0.44</td><td> 0.13</td><td> 1.00</td><td>-0.06</td><td>-0.01</td><td>-0.10</td></tr>
	<tr><th scope=row>city.mpg</th><td> 0.00</td><td> 0.02</td><td>-0.20</td><td>-0.50</td><td>-0.71</td><td>-0.67</td><td>-0.11</td><td>-0.78</td><td>-0.72</td><td>-0.60</td><td>-0.04</td><td> 0.31</td><td>-0.81</td><td>-0.06</td><td> 1.00</td><td> 0.97</td><td>-0.69</td></tr>
	<tr><th scope=row>highway.mpg</th><td> 0.00</td><td> 0.08</td><td>-0.16</td><td>-0.57</td><td>-0.74</td><td>-0.70</td><td>-0.16</td><td>-0.82</td><td>-0.73</td><td>-0.60</td><td>-0.05</td><td> 0.25</td><td>-0.78</td><td>-0.01</td><td> 0.97</td><td> 1.00</td><td>-0.70</td></tr>
	<tr><th scope=row>price</th><td>-0.12</td><td>-0.08</td><td> 0.14</td><td> 0.58</td><td> 0.69</td><td> 0.73</td><td> 0.14</td><td> 0.82</td><td> 0.88</td><td> 0.54</td><td> 0.10</td><td> 0.07</td><td> 0.76</td><td>-0.10</td><td>-0.69</td><td>-0.70</td><td> 1.00</td></tr>
</tbody>
</table>




```python
library(ggplot2)
library(ggcorrplot)
options(warn=-1)

ggcorrplot(r,
           lab = TRUE,
           lab_size = 2,
           outline.color = "white",
           title = "Correlation of numeric variables")
```

    
    Attaching package: 'ggplot2'
    
    
    The following object is masked from 'package:randomForest':
    
        margin
    
    
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Pre-Modeling/output_55_1.png)



```python
for (i in 1:nrow(r)){
  correlations <-  which((r[i,] > 0.5) & (r[i,] != 1))
    
  }
correlations
```


<dl class=dl-inline><dt>wheel.base</dt><dd>4</dd><dt>length</dt><dd>5</dd><dt>width</dt><dd>6</dd><dt>curb.weight</dt><dd>8</dd><dt>engine.size</dt><dd>9</dd><dt>bore</dt><dd>10</dd><dt>horsepower</dt><dd>13</dd></dl>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


## Boruta
The Boruta algorithm is a wrapper built around the random forest classification algorithm. It tries to capture all the important, interesting features you might have in your data set with respect to an outcome variable.

**Methodology:**
Follow the steps below to understand the algorithm -
1. Create duplicate copies of all independent variables. When the number of independent variables in the original data is less than 5, create at least 5 copies using existing variables
2. Shuffle the values of added duplicate copies to remove their correlations with the target variable. It is called shadow features or permuted copies
3. Combine the original ones with shuffled copies
4. Run a random forest classifier on the combined dataset and performs a variable importance measure (the default is Mean Decrease Accuracy) to evaluate the importance of each variable where higher means more important
5. Then Z score is computed. It means mean of accuracy loss divided by standard deviation of accuracy loss
6. Find the maximum Z score among shadow attributes (MZSA)
7. Tag the variables as 'unimportant' when they have importance significantly lower than MZSA. Then we permanently remove them from the process
8. Tag the variables as 'important' when they have importance significantly higher than MZSA
9. Repeat the above steps for predefined number of iterations (random forest runs), or until all attributes are either tagged 'unimportant' or 'important', whichever comes first

it is important to treat missing or blank values prior to using boruta package, otherwise it throws an error.


```python
# install.packages("Baruta")
library(Boruta)
```


```python
traindata = read.csv('dataset/train.csv')
traindata[traindata == ""] <- NA
traindata <- traindata[complete.cases(traindata),]
convert <- c(2:6, 11:13)
traindata[,convert] <- data.frame(apply(traindata[convert], 2, as.factor))
```


```python
colnames(traindata)
```


<ol class=list-inline><li>'Loan_ID'</li><li>'Gender'</li><li>'Married'</li><li>'Dependents'</li><li>'Education'</li><li>'Self_Employed'</li><li>'ApplicantIncome'</li><li>'CoapplicantIncome'</li><li>'LoanAmount'</li><li>'Loan_Amount_Term'</li><li>'Credit_History'</li><li>'Property_Area'</li><li>'Loan_Status'</li></ol>




```python
set.seed(123)
boruta.train <- Boruta(Loan_Status~.-Loan_ID, data = traindata, doTrace = 2)

print(boruta.train)
```

     1. run of importance source...
    
     2. run of importance source...
    
     3. run of importance source...
    
     4. run of importance source...
    
     5. run of importance source...
    
     6. run of importance source...
    
     7. run of importance source...
    
     8. run of importance source...
    
     9. run of importance source...
    
     10. run of importance source...
    
     11. run of importance source...
    
    After 11 iterations, +4.2 secs: 
    
     confirmed 1 attribute: Credit_History;
    
     rejected 1 attribute: Dependents;
    
     still have 9 attributes left.
    
    
     12. run of importance source...
    
     13. run of importance source...
    
     14. run of importance source...
    
     15. run of importance source...
    
    After 15 iterations, +5.3 secs: 
    
     rejected 1 attribute: Education;
    
     still have 8 attributes left.
    
    
     16. run of importance source...
    
     17. run of importance source...
    
     18. run of importance source...
    
    After 18 iterations, +6.1 secs: 
    
     rejected 1 attribute: Self_Employed;
    
     still have 7 attributes left.
    
    
     19. run of importance source...
    
     20. run of importance source...
    
     21. run of importance source...
    
     22. run of importance source...
    
     23. run of importance source...
    
     24. run of importance source...
    
    After 24 iterations, +7.7 secs: 
    
     confirmed 1 attribute: ApplicantIncome;
    
     still have 6 attributes left.
    
    
     25. run of importance source...
    
     26. run of importance source...
    
     27. run of importance source...
    
    After 27 iterations, +8.5 secs: 
    
     rejected 1 attribute: Gender;
    
     still have 5 attributes left.
    
    
     28. run of importance source...
    
     29. run of importance source...
    
     30. run of importance source...
    
    After 30 iterations, +9.2 secs: 
    
     confirmed 2 attributes: CoapplicantIncome, LoanAmount;
    
     still have 3 attributes left.
    
    
     31. run of importance source...
    
     32. run of importance source...
    
     33. run of importance source...
    
    After 33 iterations, +9.9 secs: 
    
     confirmed 1 attribute: Loan_Amount_Term;
    
     still have 2 attributes left.
    
    
     34. run of importance source...
    
     35. run of importance source...
    
     36. run of importance source...
    
     37. run of importance source...
    
     38. run of importance source...
    
     39. run of importance source...
    
     40. run of importance source...
    
     41. run of importance source...
    
     42. run of importance source...
    
     43. run of importance source...
    
     44. run of importance source...
    
     45. run of importance source...
    
     46. run of importance source...
    
     47. run of importance source...
    
     48. run of importance source...
    
     49. run of importance source...
    
     50. run of importance source...
    
     51. run of importance source...
    
     52. run of importance source...
    
     53. run of importance source...
    
     54. run of importance source...
    
     55. run of importance source...
    
     56. run of importance source...
    
     57. run of importance source...
    
     58. run of importance source...
    
     59. run of importance source...
    
     60. run of importance source...
    
     61. run of importance source...
    
     62. run of importance source...
    
     63. run of importance source...
    
     64. run of importance source...
    
     65. run of importance source...
    
     66. run of importance source...
    
     67. run of importance source...
    
     68. run of importance source...
    
     69. run of importance source...
    
     70. run of importance source...
    
     71. run of importance source...
    
     72. run of importance source...
    
     73. run of importance source...
    
     74. run of importance source...
    
     75. run of importance source...
    
     76. run of importance source...
    
     77. run of importance source...
    
     78. run of importance source...
    
     79. run of importance source...
    
     80. run of importance source...
    
     81. run of importance source...
    
     82. run of importance source...
    
     83. run of importance source...
    
     84. run of importance source...
    
     85. run of importance source...
    
     86. run of importance source...
    
     87. run of importance source...
    
     88. run of importance source...
    
     89. run of importance source...
    
     90. run of importance source...
    
     91. run of importance source...
    
     92. run of importance source...
    
     93. run of importance source...
    
     94. run of importance source...
    
     95. run of importance source...
    
     96. run of importance source...
    
     97. run of importance source...
    
     98. run of importance source...
    
     99. run of importance source...
    
    

    Boruta performed 99 iterations in 24.9815 secs.
     5 attributes confirmed important: ApplicantIncome, CoapplicantIncome,
    Credit_History, Loan_Amount_Term, LoanAmount;
     4 attributes confirmed unimportant: Dependents, Education, Gender,
    Self_Employed;
     2 tentative attributes left: Married, Property_Area;
    

Boruta performed 99 iterations in 25.87992 secs.<br>
 5 attributes confirmed important: ApplicantIncome, CoapplicantIncome,
Credit_History, Loan_Amount_Term, LoanAmount;<br>
 4 attributes confirmed unimportant: Dependents, Education, Gender,
Self_Employed;<br>
 2 tentative attributes left: Married, Property_Area;

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


## MARS
The earth package implements variable importance based on Generalized cross validation (GCV), number of subset models the variable occurs (nsubsets) and residual sum of squares (RSS).


```python
# install.packages('earth')  
library(earth)
options(warn=-1)

marsModel <- earth(Loan_Status ~ ., data=traindata) # build model
ev <- evimp (marsModel) # estimate variable importance
head(ev)
```

    Loading required package: Formula
    
    Loading required package: plotmo
    
    Loading required package: plotrix
    
    Loading required package: TeachingDemos
    
    


<ol class=list-inline><li>625</li><li>286</li><li>622</li><li>518</li><li>326</li><li>195</li></ol>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

<a id='1'></a>
## Variance Inflation Factor (VIF)

Variance Inflation Factor (VIF) is used to detect the presence of multicollinearity. Variance inflation factors (VIF) measure how much the variance of the estimated regression coefficients is inflated as compared to when the predictor variables are not linearly related.

It is obtained by regressing each independent variable, say X on the remaining independent variables (say Y and Z) and checking how much of it (of X) is explained by these variables.


  **VIF = 1/(1- R^2)**

**Application & Interpretation:**<br>
1. From the list of variables, we select the variables with high VIF as collinear variables. But to decide which variable to select, we look at the Condition Index of the variables or the final regression coefficient table.
2. As a thumb rule, any variable with VIF > 2 is avoided in a regression analysis. Sometimes the condition is relaxed to 5, instead of 2.
3. It is not suitable for categorical variables


```python
head(data)
```


<table>
<caption>A data.frame: 6 × 17</caption>
<thead>
	<tr><th></th><th scope=col>X</th><th scope=col>symboling</th><th scope=col>normalized.losses</th><th scope=col>wheel.base</th><th scope=col>length</th><th scope=col>width</th><th scope=col>height</th><th scope=col>curb.weight</th><th scope=col>engine.size</th><th scope=col>bore</th><th scope=col>stroke</th><th scope=col>compression.ratio</th><th scope=col>horsepower</th><th scope=col>peak.rpm</th><th scope=col>city.mpg</th><th scope=col>highway.mpg</th><th scope=col>price</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0</td><td>3</td><td>122</td><td>88.6</td><td>168.8</td><td>64.1</td><td>48.8</td><td>2548</td><td>130</td><td>3.47</td><td>2.68</td><td> 9.0</td><td>111</td><td>5000</td><td>21</td><td>27</td><td>13495</td></tr>
	<tr><th scope=row>2</th><td>1</td><td>3</td><td>122</td><td>88.6</td><td>168.8</td><td>64.1</td><td>48.8</td><td>2548</td><td>130</td><td>3.47</td><td>2.68</td><td> 9.0</td><td>111</td><td>5000</td><td>21</td><td>27</td><td>16500</td></tr>
	<tr><th scope=row>3</th><td>2</td><td>1</td><td>122</td><td>94.5</td><td>171.2</td><td>65.5</td><td>52.4</td><td>2823</td><td>152</td><td>2.68</td><td>3.47</td><td> 9.0</td><td>154</td><td>5000</td><td>19</td><td>26</td><td>16500</td></tr>
	<tr><th scope=row>4</th><td>3</td><td>2</td><td>164</td><td>99.8</td><td>176.6</td><td>66.2</td><td>54.3</td><td>2337</td><td>109</td><td>3.19</td><td>3.40</td><td>10.0</td><td>102</td><td>5500</td><td>24</td><td>30</td><td>13950</td></tr>
	<tr><th scope=row>5</th><td>4</td><td>2</td><td>164</td><td>99.4</td><td>176.6</td><td>66.4</td><td>54.3</td><td>2824</td><td>136</td><td>3.19</td><td>3.40</td><td> 8.0</td><td>115</td><td>5500</td><td>18</td><td>22</td><td>17450</td></tr>
	<tr><th scope=row>6</th><td>5</td><td>2</td><td>122</td><td>99.8</td><td>177.3</td><td>66.3</td><td>53.1</td><td>2507</td><td>136</td><td>3.19</td><td>3.40</td><td> 8.5</td><td>110</td><td>5500</td><td>19</td><td>25</td><td>15250</td></tr>
</tbody>
</table>




```python
data <- subset(data, select = -c(symboling,X))
```


```python
# library(VIF)
model2 <- lm(price~., data)

```


```python
# onthe basis of vif cutoff we can drop the columns with higher vif values 
car::vif(model2)
```


<dl class=dl-inline><dt>normalized.losses</dt><dd>1.34547204219814</dd><dt>wheel.base</dt><dd>7.89493750706869</dd><dt>length</dt><dd>9.7040302054627</dd><dt>width</dt><dd>5.6664832357184</dd><dt>height</dt><dd>2.63207762830906</dd><dt>curb.weight</dt><dd>16.5644808580086</dd><dt>engine.size</dt><dd>7.082215867018</dd><dt>bore</dt><dd>2.1769374179109</dd><dt>stroke</dt><dd>1.20950020426134</dd><dt>compression.ratio</dt><dd>2.26130704067139</dd><dt>horsepower</dt><dd>8.71492541162748</dd><dt>peak.rpm</dt><dd>1.98379004535762</dd><dt>city.mpg</dt><dd>27.6224165717629</dd><dt>highway.mpg</dt><dd>24.0929440290376</dd></dl>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

<a id='2'></a>
## Principal Component Analysis (PCA)
PCA, generally called **data reduction technique**, is very useful feature selection technique as it uses linear algebra to transform the dataset into a compressed form. We can select number of principal components in the output.

**The components are formed such that it explains the maximum variation in the dataset**.

The dataset used is USArrests and can be imported by the code below:



```python
data(USArrests)
```

### Option 1: using prcomp()
The function prcomp() comes with the default "stats" package, which means that you don’t have to install anything. It is perhaps the quickest way to do a PCA if you don’t want to install other packages.


```python
# PCA with function prcomp
pca1 = prcomp(USArrests, scale. = TRUE)

# sqrt of eigenvalues
pca1$sdev
```


<ol class=list-inline><li>1.57487827439123</li><li>0.994869414817764</li><li>0.597129115502527</li><li>0.41644938195396</li></ol>




```python
# loadings
head(pca1$rotation)
```


<table>
<caption>A matrix: 4 × 4 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>PC1</th><th scope=col>PC2</th><th scope=col>PC3</th><th scope=col>PC4</th></tr>
</thead>
<tbody>
	<tr><th scope=row>Murder</th><td>-0.5358995</td><td> 0.4181809</td><td>-0.3412327</td><td> 0.64922780</td></tr>
	<tr><th scope=row>Assault</th><td>-0.5831836</td><td> 0.1879856</td><td>-0.2681484</td><td>-0.74340748</td></tr>
	<tr><th scope=row>UrbanPop</th><td>-0.2781909</td><td>-0.8728062</td><td>-0.3780158</td><td> 0.13387773</td></tr>
	<tr><th scope=row>Rape</th><td>-0.5434321</td><td>-0.1673186</td><td> 0.8177779</td><td> 0.08902432</td></tr>
</tbody>
</table>




```python
# PCs (aka scores)
head(pca1$x)
```


<table>
<caption>A matrix: 6 × 4 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>PC1</th><th scope=col>PC2</th><th scope=col>PC3</th><th scope=col>PC4</th></tr>
</thead>
<tbody>
	<tr><th scope=row>Alabama</th><td>-0.9756604</td><td> 1.1220012</td><td>-0.43980366</td><td> 0.154696581</td></tr>
	<tr><th scope=row>Alaska</th><td>-1.9305379</td><td> 1.0624269</td><td> 2.01950027</td><td>-0.434175454</td></tr>
	<tr><th scope=row>Arizona</th><td>-1.7454429</td><td>-0.7384595</td><td> 0.05423025</td><td>-0.826264240</td></tr>
	<tr><th scope=row>Arkansas</th><td> 0.1399989</td><td> 1.1085423</td><td> 0.11342217</td><td>-0.180973554</td></tr>
	<tr><th scope=row>California</th><td>-2.4986128</td><td>-1.5274267</td><td> 0.59254100</td><td>-0.338559240</td></tr>
	<tr><th scope=row>Colorado</th><td>-1.4993407</td><td>-0.9776297</td><td> 1.08400162</td><td> 0.001450164</td></tr>
</tbody>
</table>



### Option 2: using PCA()
A highly recommended option, especially if you want more detailed results and assessing tools, is the PCA() function from the package "FactoMineR". It is by far the best PCA function in R and it comes with a number of parameters that allow you to tweak the analysis in a very nice way.


```python
# PCA with function PCA
# install.packages("FactoMineR")
library(FactoMineR)

# apply PCA
pca2 = PCA(USArrests, graph = TRUE)

# matrix with eigenvalues
pca2$eig
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Pre-Modeling/output_85_0.png)



<table>
<caption>A matrix: 4 × 3 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>eigenvalue</th><th scope=col>percentage of variance</th><th scope=col>cumulative percentage of variance</th></tr>
</thead>
<tbody>
	<tr><th scope=row>comp 1</th><td>2.4802416</td><td>62.006039</td><td> 62.00604</td></tr>
	<tr><th scope=row>comp 2</th><td>0.9897652</td><td>24.744129</td><td> 86.75017</td></tr>
	<tr><th scope=row>comp 3</th><td>0.3565632</td><td> 8.914080</td><td> 95.66425</td></tr>
	<tr><th scope=row>comp 4</th><td>0.1734301</td><td> 4.335752</td><td>100.00000</td></tr>
</tbody>
</table>




![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Pre-Modeling/output_85_2.png)



```python
# correlations between variables and PCs
pca2$var$coord
```


<table>
<caption>A matrix: 4 × 4 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>Dim.1</th><th scope=col>Dim.2</th><th scope=col>Dim.3</th><th scope=col>Dim.4</th></tr>
</thead>
<tbody>
	<tr><th scope=row>Murder</th><td>0.8439764</td><td>-0.4160354</td><td> 0.2037600</td><td> 0.27037052</td></tr>
	<tr><th scope=row>Assault</th><td>0.9184432</td><td>-0.1870211</td><td> 0.1601192</td><td>-0.30959159</td></tr>
	<tr><th scope=row>UrbanPop</th><td>0.4381168</td><td> 0.8683282</td><td> 0.2257242</td><td> 0.05575330</td></tr>
	<tr><th scope=row>Rape</th><td>0.8558394</td><td> 0.1664602</td><td>-0.4883190</td><td> 0.03707412</td></tr>
</tbody>
</table>




```python
# PCs (aka scores)
head(pca2$ind$coord)
```


<table>
<caption>A matrix: 6 × 4 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>Dim.1</th><th scope=col>Dim.2</th><th scope=col>Dim.3</th><th scope=col>Dim.4</th></tr>
</thead>
<tbody>
	<tr><th scope=row>Alabama</th><td> 0.9855659</td><td>-1.1333924</td><td> 0.44426879</td><td> 0.156267145</td></tr>
	<tr><th scope=row>Alaska</th><td> 1.9501378</td><td>-1.0732133</td><td>-2.04000333</td><td>-0.438583440</td></tr>
	<tr><th scope=row>Arizona</th><td> 1.7631635</td><td> 0.7459568</td><td>-0.05478082</td><td>-0.834652924</td></tr>
	<tr><th scope=row>Arkansas</th><td>-0.1414203</td><td>-1.1197968</td><td>-0.11457369</td><td>-0.182810896</td></tr>
	<tr><th scope=row>California</th><td> 2.5239801</td><td> 1.5429340</td><td>-0.59855680</td><td>-0.341996478</td></tr>
	<tr><th scope=row>Colorado</th><td> 1.5145629</td><td> 0.9875551</td><td>-1.09500699</td><td> 0.001464887</td></tr>
</tbody>
</table>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### PCA plots
PCA is commonly used for data visualization even though other methods like ggplots are also avaliable. 


```python
# load ggplot2
# install.packages("ggplot2")
library(ggplot2)
options(warn=-1)

# create data frame with scores
scores = as.data.frame(pca1$x)

# plot of observations
ggplot(data = scores, aes(x = PC1, y = PC2, label = rownames(scores))) +
  geom_hline(yintercept = 0, colour = "gray65") +
  geom_vline(xintercept = 0, colour = "gray65") +
  geom_text(colour = "tomato", alpha = 0.8, size = 4) +
  ggtitle("PCA plot of USA States - Crime Rates")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Pre-Modeling/output_90_0.png)


PCA does not reak pick features for further modeling, it reduces large numer of features to the number of components required. <br>
PCA does not provide much interpretability

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

<a id='3'></a>
## Linear Discriminant Analysis (LDA)
LDA is a type of Linear combination, a mathematical process using various data items and applying a function to that site to separately analyse multiple classes of objects or items.



An example of LDA is given below:


```python
# install.packages("MASS")
require(MASS)

# Load data
data(iris)
options(warn=-1)
 
head(iris, 3)
```

    Loading required package: MASS
    
    
    Attaching package: 'MASS'
    
    
    The following object is masked from 'package:dplyr':
    
        select
    
    
    


<table>
<caption>A data.frame: 3 × 5</caption>
<thead>
	<tr><th></th><th scope=col>Sepal.Length</th><th scope=col>Sepal.Width</th><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th><th scope=col>Species</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;fct&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>5.1</td><td>3.5</td><td>1.4</td><td>0.2</td><td>setosa</td></tr>
	<tr><th scope=row>2</th><td>4.9</td><td>3.0</td><td>1.4</td><td>0.2</td><td>setosa</td></tr>
	<tr><th scope=row>3</th><td>4.7</td><td>3.2</td><td>1.3</td><td>0.2</td><td>setosa</td></tr>
</tbody>
</table>




```python
lda <- lda(formula = Species ~ ., 
         data = iris, 
         prior = c(1,1,1)/3)
```


```python
lda$prior
lda$counts
lda$means
lda$scaling
lda$svd
```


<dl class=dl-inline><dt>setosa</dt><dd>0.333333333333333</dd><dt>versicolor</dt><dd>0.333333333333333</dd><dt>virginica</dt><dd>0.333333333333333</dd></dl>




<dl class=dl-inline><dt>setosa</dt><dd>50</dd><dt>versicolor</dt><dd>50</dd><dt>virginica</dt><dd>50</dd></dl>




<table>
<caption>A matrix: 3 × 4 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>Sepal.Length</th><th scope=col>Sepal.Width</th><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th></tr>
</thead>
<tbody>
	<tr><th scope=row>setosa</th><td>5.006</td><td>3.428</td><td>1.462</td><td>0.246</td></tr>
	<tr><th scope=row>versicolor</th><td>5.936</td><td>2.770</td><td>4.260</td><td>1.326</td></tr>
	<tr><th scope=row>virginica</th><td>6.588</td><td>2.974</td><td>5.552</td><td>2.026</td></tr>
</tbody>
</table>




<table>
<caption>A matrix: 4 × 2 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>LD1</th><th scope=col>LD2</th></tr>
</thead>
<tbody>
	<tr><th scope=row>Sepal.Length</th><td> 0.8293776</td><td> 0.02410215</td></tr>
	<tr><th scope=row>Sepal.Width</th><td> 1.5344731</td><td> 2.16452123</td></tr>
	<tr><th scope=row>Petal.Length</th><td>-2.2012117</td><td>-0.93192121</td></tr>
	<tr><th scope=row>Petal.Width</th><td>-2.8104603</td><td> 2.83918785</td></tr>
</tbody>
</table>




<ol class=list-inline><li>48.6426438022589</li><li>4.57998271097129</li></ol>



As seen above, a call to LDA returns the prior probability of each class, the counts for each class in the data, the class-specific means for each covariate, the linear combination coefficients (scaling) for each linear discriminant (remember that in this case with 3 classes at most two linear discriminants is possible) and the singular values (svd) that gives the ratio of the between- and within-group standard deviations on the linear discriminant variables.


```python
prop = lda$svd^2/sum(lda$svd^2)
prop
```


<ol class=list-inline><li>0.991212604965367</li><li>0.00878739503463279</li></ol>



The singular values can be used to compute the amount of the between-group variance that is explained by each linear discriminant. In the example it can be seen that the first linear discriminant explains more than {99\%} of the between-group variance in the iris dataset.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


Principal Component Analysis (**PCA**) applied to this data **identifies the
combination of attributes (principal components, or directions in the
feature space) that account for the most variance** in the data. Here we
plot the different samples on the 2 first principal components.

Linear Discriminant Analysis (**LDA**) tries to **identify attributes that
account for the most variance** between classes. In particular,
LDA, in contrast to PCA, is a supervised method, using known class labels.

The Iris dataset represents 3 kind of Iris flowers (Setosa, Versicolour
and Virginica) with 4 attributes: sepal length, sepal width, petal length
and petal width.


```python
require(MASS)
require(ggplot2)
require(scales)
require(gridExtra)
options(warn=-1) #Suppress warnings

pca <- prcomp(iris[,-5],
              center = TRUE,
              scale. = TRUE) 

prop.pca = pca$sdev^2/sum(pca$sdev^2)

lda <- lda(Species ~ ., 
           iris, 
           prior = c(1,1,1)/3)

prop.lda = lda$svd^2/sum(lda$svd^2)

plda <- predict(object = lda,
                newdata = iris)

dataset = data.frame(species = iris[,"Species"],
                     pca = pca$x, lda = plda$x)

p1 <- ggplot(dataset) + geom_point(aes(lda.LD1, lda.LD2, colour = species, shape = species), size = 2.5) + 
  labs(x = paste("LD1 (", percent(prop.lda[1]), ")", sep=""),
       y = paste("LD2 (", percent(prop.lda[2]), ")", sep=""))

p2 <- ggplot(dataset) + geom_point(aes(pca.PC1, pca.PC2, colour = species, shape = species), size = 2.5) +
  labs(x = paste("PC1 (", percent(prop.pca[1]), ")", sep=""),
       y = paste("PC2 (", percent(prop.pca[2]), ")", sep=""))

grid.arrange(p1, p2)
options(warn=-1) #Suppress warnings
```
![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Pre-Modeling/output_103_1.png)

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

<a id='4'></a>
## Feature importance
The importance of features can be estimated from data by building a model. Some methods like decision trees have a built-in mechanism to report on variable importance. For other algorithms, the importance can be estimated using a ROC curve analysis conducted for each attribute. <br>

In the example below, we will use RandomForestClassifier to select features.


```python
data(iris)
# Set random seed to make results reproducible:
set.seed(17)
# Calculate the size of each of the data sets:
data_set_size <- floor(nrow(iris)/2)
# Generate a random sample of "data_set_size" indexes
indexes <- sample(1:nrow(iris), size = data_set_size)
# Assign the data to the correct sets
training <- iris[indexes,]
validation1 <- iris[-indexes,]
```


```python
#import the package
library(randomForest)
# Perform training:
rf_classifier = randomForest(Species ~ ., data=training, ntree=100, mtry=2, importance=TRUE)
```


```python
rf_classifier
```


    
    Call:
     randomForest(formula = Species ~ ., data = training, ntree = 100,      mtry = 2, importance = TRUE) 
                   Type of random forest: classification
                         Number of trees: 100
    No. of variables tried at each split: 2
    
            OOB estimate of  error rate: 6.67%
    Confusion matrix:
               setosa versicolor virginica class.error
    setosa         27          0         0  0.00000000
    versicolor      0         22         2  0.08333333
    virginica       0          3        21  0.12500000



```python
varImpPlot(rf_classifier)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Pre-Modeling/output_109_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


## Chi Square Test
The Chi-square test of independence works by comparing the observed frequencies (so the frequencies observed in your sample) to the expected frequencies if there was no relationship between the two categorical variables (so the expected frequencies if the null hypothesis was true)


```python
dat <- iris
dat$size <- ifelse(dat$Sepal.Length < median(dat$Sepal.Length),
  "small", "big")

table(dat$Species, dat$size)
```


                
                 big small
      setosa       1    49
      versicolor  29    21
      virginica   47     3


The contingency table gives the observed number of cases in each subgroup.


```python
test <- chisq.test(table(dat$Species, dat$size))
test
```


    
    	Pearson's Chi-squared test
    
    data:  table(dat$Species, dat$size)
    X-squared = 86.035, df = 2, p-value < 2.2e-16
    



```python
test$statistic
test$p.value
```


<strong>X-squared:</strong> 86.0345134317737



2.07894395533151e-19


From the output and from test$p.value we see that the p-value is less than the significance level of 0.05. Like any other statistical test, if the p-value is less than the significance level, the null hypothesis can be rejected.
<br><br>In this context, rejecting the null hypothesis for the Chi-square test of independence means that there is a significant relationship between the Species and the sepal length. Therefore, knowing the value of one variable helps to predict the value of the other variable.

Chi square test can be used to find the feature importance of a categorical feature as follows:


```python
auto_mobile <- read.csv('dataset/auto_mobile.csv')
dat <- auto_mobile

table(dat$body.style, dat$num.of.doors)
```


                 
                  four two
      convertible    0   6
      hardtop        0   8
      hatchback     10  60
      sedan         79  15
      wagon         25   0



```python
test <- chisq.test(table(dat$body.style, dat$num.of.doors))
test
```


    
    	Pearson's Chi-squared test
    
    data:  table(dat$body.style, dat$num.of.doors)
    X-squared = 116.98, df = 4, p-value < 2.2e-16
    



```python
test$statistic
test$p.value
```


<strong>X-squared:</strong> 116.984187249141



2.35323747910589e-24


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Modeling" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 





