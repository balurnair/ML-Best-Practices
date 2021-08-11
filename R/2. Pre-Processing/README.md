# Pre Processing

# Notebook Content
[Libraries used](#Libraries)<br>
[Data Preprocessing](#Data-Preprocessing)<br>
## Missing Values Treatment
[Missing Value Treatment](#Missing-Value-Treatment)<br>
[Missing Value Treatment with mean](#Mean-or-median-or-other-summary-statistic-substitution)<br>
[Forward and Backward fill](#20)<br>
[Nearest neighbors imputation](#Nearest-neighbors-imputation)<br>
[Multivaraiate Imputation](#Multivariate-Imputation)<br>
## Data Transformations
[Data Transformations](#Data-Transformation)<br>
[Scale Transformation](#Scale-Transformation)<br>
[Centre Transformation](#Centre-Transformation)<br>
[Standardize Transformation](#Standardize-Transformation)<br>
[Data Normalization](#Normalization)<br>
[Box-Cox Transform](#Box-Cox-Transform)<br>
[Yeo-Johnson Transform](#Yeo-Johnson-Transform)<br>
[Principal Component Analysis](#Principal-Component-Analysis)<br>
[Tips For Data Transforms](#Tips-For-Data-Transforms)<br>
## Encoding
[Handling Categorical Variable](#Handling-Categorical-Variable)<br>
[One Hot Encoding](#One-Hot-Encoding)<br>
[Label Encoding](#Label-Encoding)<br>
[Hashing](#Hashing)<br>
## Embedding
[Embedding](#Embedding-methods)<br>
[CountVectorizer](#CountVectorizer)<br>
[TF-IDF Vectorizer](#TF-IDF-Vectorizer)<br>
[Stemming](#Stemming)<br>
[Lemmatization](#Lemmatization)<br>

# Libraries


```python
# install.packages('caTools')
# install.packages("ROCR")
# install.packages('glmnet')
# install.packages('rpart')
# install.packages('caret')
# install.packages('rpart.plot')
# install.packages('rattle')
# install.packages('tidyverse')
# install.packages('nnet')
# install.packages('caret')
# install.packages('e1071')
# install.packages('class')
# install.packages('gmodels')
# install.packages("mlbench")
# install.packages("tidyr")
# install.packages("DMwR")
# install.packages("mice")
# install.packages("superml")
# install.packages('devtools')
# install.packages("textmineR")
# install.packages("textstem")
# install.packages("qdap")
# install.packages("tm")
# options(warn=-1)
```

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# Data Preprocessing

Pre-processing refers to the **transformations applied to data** before feeding it to the algorithm. Data Preprocessing is a technique that is used to **convert the raw data into a clean data set**. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis, pre-processing heps us to bring our data to a desired format.

## Need for Data Preprocessing

**For achieving better results** from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning models need information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set. Another aspect is that data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithms are executed in one data set, and best out of them is chosen.

## Different data preprocesses

The different pre processing techniques are listed below; we will look into each of it in detail:

![Types%20of%20Pre_Processing.PNG](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Pre-Processing/Types%20of%20Pre_Processing.PNG)

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# __Missing Value Treatment__
The options we have for handling missing values are as follows:<br>
1. Drop missing values
2. Fill missing value 
3. Predict missing value with maching learning algoritm

## Imputation vs Removing Data
Before jumping to the methods of data imputation, we have to understand the reason why data goes missing.
1. **Missing completely at random**: This is a case when the probability of missing variable is same for all observations. For example: respondents of data collection process decide that they will declare their earning after tossing a fair coin. If an head occurs, respondent declares his / her earnings & vice versa. Here each observation has equal chance of missing value.
2. **Missing at random**: This is a case when variable is missing at random and missing ratio varies for different values / level of other input variables. For example: We are collecting data for age and female has higher missing value compare to male.
3. **Missing that depends on unobserved predictors**: This is a case when the missing values are not random and are related to the unobserved input variable. For example: In a medical study, if a particular diagnostic causes discomfort, then there is higher chance of drop out from the study. This missing value is not at random unless we have included “discomfort” as an input variable for all patients.
4. **Missing that depends on the missing value itself**: This is a case when the probability of missing value is directly correlated with missing value itself. For example: People with higher or lower income are likely to provide non-response to their earning.
 

## Drop missing values

**Simple approaches**<br>
A number of simple approaches exist. For basic use cases, these are often enough.<br><br>
**Dropping rows with null values**
1. If the number of data points is sufficiently high that dropping some of them will not cause lose generalizability in the models built (to determine whether or not this is the case, a learning curve can be used)
2. Dropping too much data is also dangerous
3. If in a large data set is present and missinng values is in range of 5-3%; then droping missing values is feasible


```python
library(tidyr)
options(warn=-1)
Country <- c('France','Spain','Germany','Spain')
Salary <- c(44000, NA, 35000,55000)
Age <- c(44,35,NA,55)
employ<- data.frame(Country, Salary,Age)
print('Dataset with Null values')
employ

print('Null values dropped')
employ %>% drop_na()
```

    [1] "Dataset with Null values"
    


<table>
<caption>A data.frame: 4 × 3</caption>
<thead>
	<tr><th scope=col>Country</th><th scope=col>Salary</th><th scope=col>Age</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>France </td><td>44000</td><td>44</td></tr>
	<tr><td>Spain  </td><td>   NA</td><td>35</td></tr>
	<tr><td>Germany</td><td>35000</td><td>NA</td></tr>
	<tr><td>Spain  </td><td>55000</td><td>55</td></tr>
</tbody>
</table>



    [1] "Null values dropped"
    


<table>
<caption>A data.frame: 2 × 3</caption>
<thead>
	<tr><th scope=col>Country</th><th scope=col>Salary</th><th scope=col>Age</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>France</td><td>44000</td><td>44</td></tr>
	<tr><td>Spain </td><td>55000</td><td>55</td></tr>
</tbody>
</table>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

## Mean or median or other summary statistic substitution
When to use example:
1. Check outlier, if less outliers is present then use mean imputation 
2. When outliers are more median impuation can be used 
3. For categorical variables use mode

<br>**NOTE:**- Ok to use if missing data is less than 3%, otherwise introduces too much bias and artificially lowers variability of data


```python
Country <- c('France','Spain','Germany','Spain')
Salary <- c(44000, NA, 35000,55000)
Age <- c(44,35,NA,55)
employ<- data.frame(Country, Salary,Age)
print('Dataset with NA values in Salary & Age')
employ


employ$Age <- ifelse(is.na(employ$Age), 
                      ave(employ$Age, FUN = function(x) 
                        mean(x, na.rm = TRUE)), 
                      employ$Age)

employ$Salary <- ifelse(is.na(employ$Salary), 
                      ave(employ$Salary, FUN = function(x) 
                        mean(x, na.rm = TRUE)), 
                      employ$Salary)
print('Missing values handled with its mean value')  
employ                        
```

    [1] "Dataset with NA values in Salary & Age"
    


<table>
<caption>A data.frame: 4 × 3</caption>
<thead>
	<tr><th scope=col>Country</th><th scope=col>Salary</th><th scope=col>Age</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>France </td><td>44000</td><td>44</td></tr>
	<tr><td>Spain  </td><td>   NA</td><td>35</td></tr>
	<tr><td>Germany</td><td>35000</td><td>NA</td></tr>
	<tr><td>Spain  </td><td>55000</td><td>55</td></tr>
</tbody>
</table>



    [1] "Missing values handled with its mean value"
    


<table>
<caption>A data.frame: 4 × 3</caption>
<thead>
	<tr><th scope=col>Country</th><th scope=col>Salary</th><th scope=col>Age</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>France </td><td>44000.00</td><td>44.00000</td></tr>
	<tr><td>Spain  </td><td>44666.67</td><td>35.00000</td></tr>
	<tr><td>Germany</td><td>35000.00</td><td>44.66667</td></tr>
	<tr><td>Spain  </td><td>55000.00</td><td>55.00000</td></tr>
</tbody>
</table>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>
<a id='20'></a>
## Forward fill and backward fill (can be used according to business problem)
Forward filling means fill missing values with previous data. Backward filling means fill missing values with next data point.


```python
library(data.table) 
options(warn=-1)
dataset <- structure(list(id = c("foo", "foo", "foo", "foo", "foo", "bar", 
        "bar", "bar", "bar", "bar"), value = c("blue", NA, NA, "red", 
            NA, "green", "green", NA, NA, NA), timestamp = structure(c(1571348572.31003, 
                1571348632.31003, 1571348692.31003, 1571348752.31003, 1571348812.31003, 
                1571348872.31003, 1571348932.31003, 1571348992.31003, 1571349052.31003, 
                1571349112.31003), class = c("POSIXct", "POSIXt"))), row.names = c(NA, 
                    -10L), class = "data.frame")
print('Dataset')
dataset

setDT(dataset)[, value := zoo::na.locf(value, FALSE), id]
print(" NA values handled by forward fill")
dataset
```

    [1] "Dataset"
    


<table>
<caption>A data.frame: 10 × 3</caption>
<thead>
	<tr><th scope=col>id</th><th scope=col>value</th><th scope=col>timestamp</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dttm&gt;</th></tr>
</thead>
<tbody>
	<tr><td>foo</td><td>blue </td><td>2019-10-18 03:12:52</td></tr>
	<tr><td>foo</td><td>NA   </td><td>2019-10-18 03:13:52</td></tr>
	<tr><td>foo</td><td>NA   </td><td>2019-10-18 03:14:52</td></tr>
	<tr><td>foo</td><td>red  </td><td>2019-10-18 03:15:52</td></tr>
	<tr><td>foo</td><td>NA   </td><td>2019-10-18 03:16:52</td></tr>
	<tr><td>bar</td><td>green</td><td>2019-10-18 03:17:52</td></tr>
	<tr><td>bar</td><td>green</td><td>2019-10-18 03:18:52</td></tr>
	<tr><td>bar</td><td>NA   </td><td>2019-10-18 03:19:52</td></tr>
	<tr><td>bar</td><td>NA   </td><td>2019-10-18 03:20:52</td></tr>
	<tr><td>bar</td><td>NA   </td><td>2019-10-18 03:21:52</td></tr>
</tbody>
</table>



    [1] " NA values handled by forward fill"
    


<table>
<caption>A data.table: 10 × 3</caption>
<thead>
	<tr><th scope=col>id</th><th scope=col>value</th><th scope=col>timestamp</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dttm&gt;</th></tr>
</thead>
<tbody>
	<tr><td>foo</td><td>blue </td><td>2019-10-18 03:12:52</td></tr>
	<tr><td>foo</td><td>blue </td><td>2019-10-18 03:13:52</td></tr>
	<tr><td>foo</td><td>blue </td><td>2019-10-18 03:14:52</td></tr>
	<tr><td>foo</td><td>red  </td><td>2019-10-18 03:15:52</td></tr>
	<tr><td>foo</td><td>red  </td><td>2019-10-18 03:16:52</td></tr>
	<tr><td>bar</td><td>green</td><td>2019-10-18 03:17:52</td></tr>
	<tr><td>bar</td><td>green</td><td>2019-10-18 03:18:52</td></tr>
	<tr><td>bar</td><td>green</td><td>2019-10-18 03:19:52</td></tr>
	<tr><td>bar</td><td>green</td><td>2019-10-18 03:20:52</td></tr>
	<tr><td>bar</td><td>green</td><td>2019-10-18 03:21:52</td></tr>
</tbody>
</table>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

### Nearest neighbors imputation
It can be used for data that are continuous, discrete, ordinal and categorical which makes it particularly useful for dealing with all kind of missing data. The assumption behind using KNN for missing values is that a point value can be approximated by the values of the points that are closest to it, based on other variables. <br><br>The distance metric varies according to the type of data:
1. **Continuous Data**: The commonly used distance metrics for continuous data are Euclidean, Manhattan and Cosine
2. **Categorical Data**: Hamming distance is generally used in this case. It takes all the categorical attributes 
    
**DMwR::knnImputation** uses k-Nearest Neighbours approach to impute missing values. What kNN imputation does in simpler terms is as follows: For every observation to be imputed, it identifies ‘k’ closest observations based on the euclidean distance and computes the weighted average (weighted based on distance) of these ‘k’ obs.
The advantage is that you could impute all the missing values in all variables with one call to the function. It takes the whole data frame as the argument and you don’t even have to specify which variable you want to impute
    


```python
# initialize the data
data ("BostonHousing", package="mlbench")
original <- BostonHousing  # backup original data

# Introduce missing values
set.seed(10)
BostonHousing[sample(1:nrow(BostonHousing), 40), "rad"] <- NA
BostonHousing[sample(1:nrow(BostonHousing), 40), "ptratio"]

print('Boston Housing Dataset')
head(BostonHousing, n = 10L)


```



<ol class=list-inline><li>20.2</li><li>14.7</li><li>19.6</li><li>18.2</li><li>17.8</li><li>18.5</li><li>18.4</li><li>17.8</li><li>20.2</li><li>16.6</li><li>16.6</li><li>13</li><li>20.2</li><li>20.2</li><li>20.2</li><li>17.8</li><li>18.8</li><li>17.9</li><li>20.2</li><li>15.2</li><li>20.1</li><li>20.2</li><li>16.4</li><li>17</li><li>14.7</li><li>20.2</li><li>21</li><li>17.6</li><li>18.6</li><li>17.9</li><li>14.7</li><li>21</li><li>20.2</li><li>20.2</li><li>21</li><li>18.4</li><li>21.2</li><li>17.4</li><li>18.6</li><li>20.2</li></ol>



    [1] "Boston Housing Dataset"
    


<table>
<caption>A data.frame: 10 × 14</caption>
<thead>
	<tr><th></th><th scope=col>crim</th><th scope=col>zn</th><th scope=col>indus</th><th scope=col>chas</th><th scope=col>nox</th><th scope=col>rm</th><th scope=col>age</th><th scope=col>dis</th><th scope=col>rad</th><th scope=col>tax</th><th scope=col>ptratio</th><th scope=col>b</th><th scope=col>lstat</th><th scope=col>medv</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0.00632</td><td>18.0</td><td>2.31</td><td>0</td><td>0.538</td><td>6.575</td><td> 65.2</td><td>4.0900</td><td>1</td><td>296</td><td>15.3</td><td>396.90</td><td> 4.98</td><td>24.0</td></tr>
	<tr><th scope=row>2</th><td>0.02731</td><td> 0.0</td><td>7.07</td><td>0</td><td>0.469</td><td>6.421</td><td> 78.9</td><td>4.9671</td><td>2</td><td>242</td><td>17.8</td><td>396.90</td><td> 9.14</td><td>21.6</td></tr>
	<tr><th scope=row>3</th><td>0.02729</td><td> 0.0</td><td>7.07</td><td>0</td><td>0.469</td><td>7.185</td><td> 61.1</td><td>4.9671</td><td>2</td><td>242</td><td>17.8</td><td>392.83</td><td> 4.03</td><td>34.7</td></tr>
	<tr><th scope=row>4</th><td>0.03237</td><td> 0.0</td><td>2.18</td><td>0</td><td>0.458</td><td>6.998</td><td> 45.8</td><td>6.0622</td><td>3</td><td>222</td><td>18.7</td><td>394.63</td><td> 2.94</td><td>33.4</td></tr>
	<tr><th scope=row>5</th><td>0.06905</td><td> 0.0</td><td>2.18</td><td>0</td><td>0.458</td><td>7.147</td><td> 54.2</td><td>6.0622</td><td>3</td><td>222</td><td>18.7</td><td>396.90</td><td> 5.33</td><td>36.2</td></tr>
	<tr><th scope=row>6</th><td>0.02985</td><td> 0.0</td><td>2.18</td><td>0</td><td>0.458</td><td>6.430</td><td> 58.7</td><td>6.0622</td><td>3</td><td>222</td><td>18.7</td><td>394.12</td><td> 5.21</td><td>28.7</td></tr>
	<tr><th scope=row>7</th><td>0.08829</td><td>12.5</td><td>7.87</td><td>0</td><td>0.524</td><td>6.012</td><td> 66.6</td><td>5.5605</td><td>5</td><td>311</td><td>15.2</td><td>395.60</td><td>12.43</td><td>22.9</td></tr>
	<tr><th scope=row>8</th><td>0.14455</td><td>12.5</td><td>7.87</td><td>0</td><td>0.524</td><td>6.172</td><td> 96.1</td><td>5.9505</td><td>5</td><td>311</td><td>15.2</td><td>396.90</td><td>19.15</td><td>27.1</td></tr>
	<tr><th scope=row>9</th><td>0.21124</td><td>12.5</td><td>7.87</td><td>0</td><td>0.524</td><td>5.631</td><td>100.0</td><td>6.0821</td><td>5</td><td>311</td><td>15.2</td><td>386.63</td><td>29.93</td><td>16.5</td></tr>
	<tr><th scope=row>10</th><td>0.17004</td><td>12.5</td><td>7.87</td><td>0</td><td>0.524</td><td>6.004</td><td> 85.9</td><td>6.5921</td><td>5</td><td>311</td><td>15.2</td><td>386.71</td><td>17.10</td><td>18.9</td></tr>
</tbody>
</table>




```python
# perform knn imputation.
library(DMwR)
options(warn=-1)
knnOutput <- knnImputation(BostonHousing[, !names(BostonHousing) %in% "medv"])  
print('Imputed using DMwR library for KNN')
head(knnOutput)
```

    Loading required package: lattice
    
    Loading required package: grid
    
    Registered S3 method overwritten by 'quantmod':
      method            from
      as.zoo.data.frame zoo 
    
    

    [1] "Imputed using DMwR library for KNN"
    


<table>
<caption>A data.frame: 6 × 13</caption>
<thead>
	<tr><th></th><th scope=col>crim</th><th scope=col>zn</th><th scope=col>indus</th><th scope=col>chas</th><th scope=col>nox</th><th scope=col>rm</th><th scope=col>age</th><th scope=col>dis</th><th scope=col>rad</th><th scope=col>tax</th><th scope=col>ptratio</th><th scope=col>b</th><th scope=col>lstat</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0.00632</td><td>18</td><td>2.31</td><td>0</td><td>0.538</td><td>6.575</td><td>65.2</td><td>4.0900</td><td>1</td><td>296</td><td>15.3</td><td>396.90</td><td>4.98</td></tr>
	<tr><th scope=row>2</th><td>0.02731</td><td> 0</td><td>7.07</td><td>0</td><td>0.469</td><td>6.421</td><td>78.9</td><td>4.9671</td><td>2</td><td>242</td><td>17.8</td><td>396.90</td><td>9.14</td></tr>
	<tr><th scope=row>3</th><td>0.02729</td><td> 0</td><td>7.07</td><td>0</td><td>0.469</td><td>7.185</td><td>61.1</td><td>4.9671</td><td>2</td><td>242</td><td>17.8</td><td>392.83</td><td>4.03</td></tr>
	<tr><th scope=row>4</th><td>0.03237</td><td> 0</td><td>2.18</td><td>0</td><td>0.458</td><td>6.998</td><td>45.8</td><td>6.0622</td><td>3</td><td>222</td><td>18.7</td><td>394.63</td><td>2.94</td></tr>
	<tr><th scope=row>5</th><td>0.06905</td><td> 0</td><td>2.18</td><td>0</td><td>0.458</td><td>7.147</td><td>54.2</td><td>6.0622</td><td>3</td><td>222</td><td>18.7</td><td>396.90</td><td>5.33</td></tr>
	<tr><th scope=row>6</th><td>0.02985</td><td> 0</td><td>2.18</td><td>0</td><td>0.458</td><td>6.430</td><td>58.7</td><td>6.0622</td><td>3</td><td>222</td><td>18.7</td><td>394.12</td><td>5.21</td></tr>
</tbody>
</table>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

## Multivariate Imputation

Mice short for Multivariate Imputation by Chained Equations is an R package that provides advanced features for missing value treatment. It uses a slightly uncommon way of implementing the imputation in 2-steps, using mice() to build the model and complete() to generate the completed data. The mice(df) function produces multiple complete copies of df, each with different imputations of the missing data. The complete() function returns one or several of these data sets, with the default being the first. Lets see how to impute ‘rad’ and ‘ptratio’:


```python
library(mice)
options(warn=-1)
miceMod <- mice(BostonHousing[, !names(BostonHousing) %in% "medv"], method="rf")  # perform mice imputation, based on random forests.
miceOutput <- complete(miceMod)  # generate the completed data.
head(miceOutput)
```

    
    Attaching package: 'mice'
    
    
    The following objects are masked from 'package:base':
    
        cbind, rbind
    
    
    

    
     iter imp variable
      1   1  rad
      1   2  rad
      1   3  rad
      1   4  rad
      1   5  rad
      2   1  rad
      2   2  rad
      2   3  rad
      2   4  rad
      2   5  rad
      3   1  rad
      3   2  rad
      3   3  rad
      3   4  rad
      3   5  rad
      4   1  rad
      4   2  rad
      4   3  rad
      4   4  rad
      4   5  rad
      5   1  rad
      5   2  rad
      5   3  rad
      5   4  rad
      5   5  rad
    


<table>
<caption>A data.frame: 6 × 13</caption>
<thead>
	<tr><th></th><th scope=col>crim</th><th scope=col>zn</th><th scope=col>indus</th><th scope=col>chas</th><th scope=col>nox</th><th scope=col>rm</th><th scope=col>age</th><th scope=col>dis</th><th scope=col>rad</th><th scope=col>tax</th><th scope=col>ptratio</th><th scope=col>b</th><th scope=col>lstat</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0.00632</td><td>18</td><td>2.31</td><td>0</td><td>0.538</td><td>6.575</td><td>65.2</td><td>4.0900</td><td>1</td><td>296</td><td>15.3</td><td>396.90</td><td>4.98</td></tr>
	<tr><th scope=row>2</th><td>0.02731</td><td> 0</td><td>7.07</td><td>0</td><td>0.469</td><td>6.421</td><td>78.9</td><td>4.9671</td><td>2</td><td>242</td><td>17.8</td><td>396.90</td><td>9.14</td></tr>
	<tr><th scope=row>3</th><td>0.02729</td><td> 0</td><td>7.07</td><td>0</td><td>0.469</td><td>7.185</td><td>61.1</td><td>4.9671</td><td>2</td><td>242</td><td>17.8</td><td>392.83</td><td>4.03</td></tr>
	<tr><th scope=row>4</th><td>0.03237</td><td> 0</td><td>2.18</td><td>0</td><td>0.458</td><td>6.998</td><td>45.8</td><td>6.0622</td><td>3</td><td>222</td><td>18.7</td><td>394.63</td><td>2.94</td></tr>
	<tr><th scope=row>5</th><td>0.06905</td><td> 0</td><td>2.18</td><td>0</td><td>0.458</td><td>7.147</td><td>54.2</td><td>6.0622</td><td>3</td><td>222</td><td>18.7</td><td>396.90</td><td>5.33</td></tr>
	<tr><th scope=row>6</th><td>0.02985</td><td> 0</td><td>2.18</td><td>0</td><td>0.458</td><td>6.430</td><td>58.7</td><td>6.0622</td><td>3</td><td>222</td><td>18.7</td><td>394.12</td><td>5.21</td></tr>
</tbody>
</table>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# Data Transformation

When data is comprised of attributes with varying scales, many machine learning algorithms can benefit from transforming the attributes to all have the same scale. This is useful for optimization algorithms used in the core of machine learning algorithms like gradient descent.
It is also useful for algorithms that weight inputs like regression and neural networks and algorithms that use distance measures like K-Nearest Neighbors. We can rescale data using different techniques, some of which are listed below.

Transforms can be used in two ways.<br>
- **Standalone:** Transforms can be modeled from training data and applied to multiple datasets.<br>The model of the transform is prepared using the preProcess() function and applied to a dataset using the predict() function.<br>

- **Training:** Transforms can prepared and applied automatically during model evaluation.<br>Transforms applied during training are prepared using the preProcess() and passed to the train() function via the preProcess argument.



**Note:**
Before performing the below transformations one should check for oultiers and Treat them.<br>
</t></t> _Check the EDA Notebook for various outlier treament method_

# Scale Transformation
The scale transform calculates the standard deviation for an attribute and divides each value by that standard deviation.



```python
# load libraries
library(caret)
options(warn=-1)
# load the dataset
data(iris)
# summarize data
print('RawData')
summary(iris[,1:4])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(iris[,1:4], method=c("scale"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, iris[,1:4])
# summarize the transformed dataset
print('Scale Transofrmed Data')
summary(transformed)
```

    Loading required package: ggplot2
    
    

    [1] "RawData"
    


      Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
     Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
     1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
     Median :5.800   Median :3.000   Median :4.350   Median :1.300  
     Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199  
     3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
     Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  


    Created from 150 samples and 4 variables
    
    Pre-processing:
      - ignored (0)
      - scaled (4)
    
    [1] "Scale Transofrmed Data"
    


      Sepal.Length    Sepal.Width      Petal.Length     Petal.Width    
     Min.   :5.193   Min.   : 4.589   Min.   :0.5665   Min.   :0.1312  
     1st Qu.:6.159   1st Qu.: 6.424   1st Qu.:0.9064   1st Qu.:0.3936  
     Median :7.004   Median : 6.883   Median :2.4642   Median :1.7055  
     Mean   :7.057   Mean   : 7.014   Mean   :2.1288   Mean   :1.5734  
     3rd Qu.:7.729   3rd Qu.: 7.571   3rd Qu.:2.8890   3rd Qu.:2.3615  
     Max.   :9.540   Max.   :10.095   Max.   :3.9087   Max.   :3.2798  


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# Centre Transformation

The center transform calculates the mean for an attribute and subtracts it from each value.


```python
# load libraries
library(caret)
# load the dataset
data(iris)
# summarize data
print('RawData')
summary(iris[,1:4])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(iris[,1:4], method=c("center"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, iris[,1:4])
# summarize the transformed dataset
print('Centre Transformed Data')
summary(transformed)
```

    [1] "RawData"
    


      Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
     Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
     1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
     Median :5.800   Median :3.000   Median :4.350   Median :1.300  
     Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199  
     3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
     Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  


    Created from 150 samples and 4 variables
    
    Pre-processing:
      - centered (4)
      - ignored (0)
    
    [1] "Centre Transformed Data"
    


      Sepal.Length       Sepal.Width        Petal.Length     Petal.Width     
     Min.   :-1.54333   Min.   :-1.05733   Min.   :-2.758   Min.   :-1.0993  
     1st Qu.:-0.74333   1st Qu.:-0.25733   1st Qu.:-2.158   1st Qu.:-0.8993  
     Median :-0.04333   Median :-0.05733   Median : 0.592   Median : 0.1007  
     Mean   : 0.00000   Mean   : 0.00000   Mean   : 0.000   Mean   : 0.0000  
     3rd Qu.: 0.55667   3rd Qu.: 0.24267   3rd Qu.: 1.342   3rd Qu.: 0.6007  
     Max.   : 2.05667   Max.   : 1.34267   Max.   : 3.142   Max.   : 1.3007  


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# Standardize Transformation
Combining the scale and center transforms will standardize your data. Attributes will have a mean value of 0 and a standard deviation of 1


```python
# load libraries
library(caret)
# load the dataset
data(iris)
# summarize data
print('Raw data')
summary(iris[,1:4])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(iris[,1:4], method=c("center", "scale"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, iris[,1:4])
# summarize the transformed dataset
print('Standardized Transformation')
summary(transformed)
```

    [1] "Raw data"
    


      Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
     Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
     1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
     Median :5.800   Median :3.000   Median :4.350   Median :1.300  
     Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199  
     3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
     Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  


    Created from 150 samples and 4 variables
    
    Pre-processing:
      - centered (4)
      - ignored (0)
      - scaled (4)
    
    [1] "Standardized Transformation"
    


      Sepal.Length       Sepal.Width       Petal.Length      Petal.Width     
     Min.   :-1.86378   Min.   :-2.4258   Min.   :-1.5623   Min.   :-1.4422  
     1st Qu.:-0.89767   1st Qu.:-0.5904   1st Qu.:-1.2225   1st Qu.:-1.1799  
     Median :-0.05233   Median :-0.1315   Median : 0.3354   Median : 0.1321  
     Mean   : 0.00000   Mean   : 0.0000   Mean   : 0.0000   Mean   : 0.0000  
     3rd Qu.: 0.67225   3rd Qu.: 0.5567   3rd Qu.: 0.7602   3rd Qu.: 0.7880  
     Max.   : 2.48370   Max.   : 3.0805   Max.   : 1.7799   Max.   : 1.7064  


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# Normalization
Is the process of **scaling individual samples to have unit norm**. This process can be useful if you plan to use a quadratic form such as the dot-product or any other kernel to quantify the similarity of any pair of samples.<br>
The function normalize provides a quick and easy way to perform this operation on a single array-like dataset, either using the l1 or l2 norms. Normalizer __works on the rows, not the columns!__ 



```python
# load libraries
library(caret)
# load the dataset
data(iris)
# summarize data
print('Raw Data')
summary(iris[,1:4])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(iris[,1:4], method=c("range"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, iris[,1:4])
# summarize the transformed dataset
print('Normalized Data')
summary(transformed)
```

    [1] "Raw Data"
    


      Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
     Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
     1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
     Median :5.800   Median :3.000   Median :4.350   Median :1.300  
     Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199  
     3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
     Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  


    Created from 150 samples and 4 variables
    
    Pre-processing:
      - ignored (0)
      - re-scaling to [0, 1] (4)
    
    [1] "Normalized Data"
    


      Sepal.Length     Sepal.Width      Petal.Length     Petal.Width     
     Min.   :0.0000   Min.   :0.0000   Min.   :0.0000   Min.   :0.00000  
     1st Qu.:0.2222   1st Qu.:0.3333   1st Qu.:0.1017   1st Qu.:0.08333  
     Median :0.4167   Median :0.4167   Median :0.5678   Median :0.50000  
     Mean   :0.4287   Mean   :0.4406   Mean   :0.4675   Mean   :0.45806  
     3rd Qu.:0.5833   3rd Qu.:0.5417   3rd Qu.:0.6949   3rd Qu.:0.70833  
     Max.   :1.0000   Max.   :1.0000   Max.   :1.0000   Max.   :1.00000  


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# Box-Cox Transform
When an attribute has a Gaussian-like distribution but is shifted, this is called a skew. The distribution of an attribute can be shifted to reduce the skew and make it more Gaussian. The BoxCox transform can perform this operation (assumes all values are positive).


```python
# load libraries
# install.packages("mlbench")
# install.packages(caret)
library(mlbench)
library(caret)
# load the dataset
print('Raw Data')
data(PimaIndiansDiabetes)
# summarize pedigree and age
summary(PimaIndiansDiabetes[,7:8])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(PimaIndiansDiabetes[,7:8], method=c("BoxCox"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, PimaIndiansDiabetes[,7:8])
# summarize the transformed dataset (note pedigree and age)
print('Box Transformed data')
summary(transformed)
```

    [1] "Raw Data"
    


        pedigree           age       
     Min.   :0.0780   Min.   :21.00  
     1st Qu.:0.2437   1st Qu.:24.00  
     Median :0.3725   Median :29.00  
     Mean   :0.4719   Mean   :33.24  
     3rd Qu.:0.6262   3rd Qu.:41.00  
     Max.   :2.4200   Max.   :81.00  


    Created from 768 samples and 2 variables
    
    Pre-processing:
      - Box-Cox transformation (2)
      - ignored (0)
    
    Lambda estimates for Box-Cox transformation:
    -0.1, -1.1
    [1] "Box Transformed data"
    


        pedigree            age        
     Min.   :-2.5510   Min.   :0.8772  
     1st Qu.:-1.4116   1st Qu.:0.8815  
     Median :-0.9875   Median :0.8867  
     Mean   :-0.9599   Mean   :0.8874  
     3rd Qu.:-0.4680   3rd Qu.:0.8938  
     Max.   : 0.8838   Max.   :0.9019  


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# Yeo-Johnson Transform
Another power-transform like the Box-Cox transform, but it supports raw values that are equal to zero and negative.


```python
# load the dataset
data(PimaIndiansDiabetes)
# load libraries
library(mlbench)
library(caret)
# load the dataset
print('Raw Data')
data(PimaIndiansDiabetes)
# summarize pedigree and age
summary(PimaIndiansDiabetes[,7:8])
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(PimaIndiansDiabetes[,7:8], method=c("YeoJohnson"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, PimaIndiansDiabetes[,7:8])
# summarize the transformed dataset (note pedigree and age)
print('Yeo Johnson Transformed')
summary(transformed)
```

    [1] "Raw Data"
    


        pedigree           age       
     Min.   :0.0780   Min.   :21.00  
     1st Qu.:0.2437   1st Qu.:24.00  
     Median :0.3725   Median :29.00  
     Mean   :0.4719   Mean   :33.24  
     3rd Qu.:0.6262   3rd Qu.:41.00  
     Max.   :2.4200   Max.   :81.00  


    Created from 768 samples and 2 variables
    
    Pre-processing:
      - ignored (0)
      - Yeo-Johnson transformation (2)
    
    Lambda estimates for Yeo-Johnson transformation:
    -2.25, -1.15
    [1] "Yeo Johnson Transformed"
    


        pedigree           age        
     Min.   :0.0691   Min.   :0.8450  
     1st Qu.:0.1724   1st Qu.:0.8484  
     Median :0.2265   Median :0.8524  
     Mean   :0.2317   Mean   :0.8530  
     3rd Qu.:0.2956   3rd Qu.:0.8580  
     Max.   :0.4164   Max.   :0.8644  


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# Principal Component Analysis
Transform the data to the principal components. The transform keeps components above the variance threshold (default=0.95) or the number of components can be specified (pcaComp). The result is attributes that are uncorrelated, useful for algorithms like linear and generalized linear regression.


```python
# load the libraries
library(mlbench)
# load the dataset
data(iris)
# summarize dataset
print('Raw Data')
summary(iris)
# calculate the pre-process parameters from the dataset
preprocessParams <- preProcess(iris, method=c("center", "scale", "pca"))
# summarize transform parameters
print(preprocessParams)
# transform the dataset using the parameters
transformed <- predict(preprocessParams, iris)
# summarize the transformed dataset
print('Transformed Data')
summary(transformed)
```

    [1] "Raw Data"
    


      Sepal.Length    Sepal.Width     Petal.Length    Petal.Width   
     Min.   :4.300   Min.   :2.000   Min.   :1.000   Min.   :0.100  
     1st Qu.:5.100   1st Qu.:2.800   1st Qu.:1.600   1st Qu.:0.300  
     Median :5.800   Median :3.000   Median :4.350   Median :1.300  
     Mean   :5.843   Mean   :3.057   Mean   :3.758   Mean   :1.199  
     3rd Qu.:6.400   3rd Qu.:3.300   3rd Qu.:5.100   3rd Qu.:1.800  
     Max.   :7.900   Max.   :4.400   Max.   :6.900   Max.   :2.500  
           Species  
     setosa    :50  
     versicolor:50  
     virginica :50  
                    
                    
                    


    Created from 150 samples and 5 variables
    
    Pre-processing:
      - centered (4)
      - ignored (1)
      - principal component signal extraction (4)
      - scaled (4)
    
    PCA needed 2 components to capture 95 percent of the variance
    [1] "Transformed Data"
    


           Species        PC1               PC2          
     setosa    :50   Min.   :-2.7651   Min.   :-2.67732  
     versicolor:50   1st Qu.:-2.0957   1st Qu.:-0.59205  
     virginica :50   Median : 0.4169   Median :-0.01744  
                     Mean   : 0.0000   Mean   : 0.00000  
                     3rd Qu.: 1.3385   3rd Qu.: 0.59649  
                     Max.   : 3.2996   Max.   : 2.64521  


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# Tips For Data Transforms
<br>Below are some tips for getting the most out of data transforms.

- Actually Use Them. You are a step ahead if you are thinking about and using data transforms to prepare your data.
 It is an easy step to forget or skip over and often has a huge impact on the accuracy of your final models.

- Use a Variety. Try a number of different data transforms on your data with a suite of different machine learning algorithms.

- Review a Summary. It is a good idea to summarize your data before and after a transform to understand the effect it had. The summary() function can be very useful.

- Visualize Data. It is also a good idea to visualize the distribution of your data before and after to get a spatial intuition for the effect of the transform.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

 

# __Handling Categorical Variable__

## One Hot Encoding
In this method, we map each category to a vector that contains 1 and 0 denoting the presence or absence of the feature. The number of vectors depends on the number of categories for features.<br><br>This method produces a lot of columns that slows down the learning significantly if the number of the category is very high for the feature.<br><br>One Hot Encoding is very popular. We can represent all categories by **N-1 (N= No of Category)** as that is sufficient to encode the one that is not included. Usually, for **Regression, we use N-1** (drop first or last column of One Hot Coded new feature ), **but for classification, the recommendation is to use all N columns without as most of the tree-based algorithm builds a tree based on all available variables**


```python
customers <- data.frame(
  id=c(10, 20, 30, 40, 50),
  gender=c('male', 'female', 'female', 'male', 'female'),
  mood=c('happy', 'sad', 'happy', 'sad','happy'),
  outcome=c(1, 1, 0, 0, 0))
print('Customer Dataset')
customers



# dummify the data
dmy <- dummyVars(" ~ .", data = customers)
trsf <- data.frame(predict(dmy, newdata = customers))
print('One hot encoding of Dataset')
trsf
```

    [1] "Customer Dataset"
    


<table>
<caption>A data.frame: 5 × 4</caption>
<thead>
	<tr><th scope=col>id</th><th scope=col>gender</th><th scope=col>mood</th><th scope=col>outcome</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>10</td><td>male  </td><td>happy</td><td>1</td></tr>
	<tr><td>20</td><td>female</td><td>sad  </td><td>1</td></tr>
	<tr><td>30</td><td>female</td><td>happy</td><td>0</td></tr>
	<tr><td>40</td><td>male  </td><td>sad  </td><td>0</td></tr>
	<tr><td>50</td><td>female</td><td>happy</td><td>0</td></tr>
</tbody>
</table>



    [1] "One hot encoding of Dataset"
    


<table>
<caption>A data.frame: 5 × 6</caption>
<thead>
	<tr><th></th><th scope=col>id</th><th scope=col>gender.female</th><th scope=col>gender.male</th><th scope=col>mood.happy</th><th scope=col>mood.sad</th><th scope=col>outcome</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>10</td><td>0</td><td>1</td><td>1</td><td>0</td><td>1</td></tr>
	<tr><th scope=row>2</th><td>20</td><td>1</td><td>0</td><td>0</td><td>1</td><td>1</td></tr>
	<tr><th scope=row>3</th><td>30</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td></tr>
	<tr><th scope=row>4</th><td>40</td><td>0</td><td>1</td><td>0</td><td>1</td><td>0</td></tr>
	<tr><th scope=row>5</th><td>50</td><td>1</td><td>0</td><td>1</td><td>0</td><td>0</td></tr>
</tbody>
</table>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

## Label Encoding
In this encoding, __each category is assigned a value from 1 through N__; here N is the number of categories for the feature. One major issue with this approach is that there is no relation or order between these classes, but the algorithm might consider them as some order, or there is some relationship


```python
sample_dat <- data.frame(a_str=c('Red','Blue','Blue','Red','Green'))
sample_dat
```


<table>
<caption>A data.frame: 5 × 1</caption>
<thead>
	<tr><th scope=col>a_str</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Red  </td></tr>
	<tr><td>Blue </td></tr>
	<tr><td>Blue </td></tr>
	<tr><td>Red  </td></tr>
	<tr><td>Green</td></tr>
</tbody>
</table>




```python
sample_dat$a_int<-as.integer(as.factor(sample_dat$a_str))
sample_dat
```


<table>
<caption>A data.frame: 5 × 2</caption>
<thead>
	<tr><th scope=col>a_str</th><th scope=col>a_int</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Red  </td><td>3</td></tr>
	<tr><td>Blue </td><td>1</td></tr>
	<tr><td>Blue </td><td>1</td></tr>
	<tr><td>Red  </td><td>3</td></tr>
	<tr><td>Green</td><td>2</td></tr>
</tbody>
</table>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

## Hashing
Hashing converts categorical variables to a higher dimensional space of integers, where the distance between two vectors of categorical variables in approximately maintained the transformed numerical dimensional space.
<br><br>With Hashing, the __number of dimensions will be far less__ than the number of dimensions with encoding like One Hot Encoding. This method is **advantageous when the cardinality of categorical is very high**.



```python
# vectorize assign, get and exists for convenience
assign_hash <- Vectorize(assign, vectorize.args = c("x", "value"))
get_hash <- Vectorize(get, vectorize.args = "x")

```


```python
# initialize hash
hash <- new.env(hash = TRUE, parent = emptyenv(), size = 100L)

 
# keys and values
key <- c("Blue", "Green", "Yellow")
value <- c(1, 22, 333)
 
# assign values to keys
assign_hash(key, value, hash)

```


<dl class=dl-inline><dt>Blue</dt><dd>1</dd><dt>Green</dt><dd>22</dd><dt>Yellow</dt><dd>333</dd></dl>




```python
# get values for keys
get_hash(c("Blue"), hash)

```


<strong>Blue:</strong> 1



```python
# show all keys with values
get_hash(ls(hash), hash)
```


<dl class=dl-inline><dt>Blue</dt><dd>1</dd><dt>Green</dt><dd>22</dd><dt>Yellow</dt><dd>333</dd></dl>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

# Embedding methods

**Text feature extraction**
<br>Text Analysis is a major application field for machine learning algorithms. However the raw data, a sequence of symbols cannot be fed directly to the algorithms themselves as most of them expect numerical feature vectors with a fixed size rather than the raw text documents with variable length.
In order to address this, scikit-learn provides utilities for the most common ways to extract numerical features from text content, namely:

**a. tokenizing:** strings and giving an integer id for each possible token, for instance by using white-spaces and punctuation as token separators

**b. counting:** the occurrences of tokens in each document

**c. normalizing:** weighting with diminishing importance tokens that occur in the majority of samples / documents

## CountVectorizer
The most straightforward one, it counts the number of times a token shows up in the document and uses this value as its weight.


```python
library(superml)
options(warn=-1)
sents = c('i am alone in dark.','mother_mary a lot',
         'alone in the dark?', 'many mothers in the lot....')
cv <- CountVectorizer$new(min_df=0.1)
cv_count_matrix <- cv$fit_transform(sents)
cv_count_matrix
```

    Loading required package: R6
    
    


<table>
<caption>A matrix: 4 × 7 of type dbl</caption>
<thead>
	<tr><th scope=col>alone</th><th scope=col>dark</th><th scope=col>lot</th><th scope=col>many</th><th scope=col>mary</th><th scope=col>mother</th><th scope=col>mothers</th></tr>
</thead>
<tbody>
	<tr><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0</td><td>0</td><td>1</td><td>0</td><td>1</td><td>1</td><td>0</td></tr>
	<tr><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr>
	<tr><td>0</td><td>0</td><td>1</td><td>1</td><td>0</td><td>0</td><td>1</td></tr>
</tbody>
</table>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

## TF-IDF Vectorizer
It will **transform the text into the feature vectors** and used as input to the estimator. The vocabulary is the dictionary that will convert each token or word in the matrix and it will get the feature index. In **CountVectorizer** we only count the number of times a word appears in the document which results in biasing in favour of most frequent words. this ends up in ignoring rare words which could have helped is in processing our data more efficiently. To overcome this , we use TfidfVectorizer. <br>
<br>In **TfidfVectorizer** we consider overall document weightage of a word. It helps us in dealing with most frequent words. Using it we can penalize them. TfidfVectorizer weights the word counts by a measure of how often they appear in the documents.


```python
df <- data.frame(sents = c('i am alone in dark.',
                           'mother_mary a lot',
                           'alone in the dark?',
                           'many mothers in the lot....'))
tf <- TfIdfVectorizer$new(smooth_idf = TRUE, min_df = 0.1)
tf_features <- tf$fit_transform(df$sents)
tf_features
```


<table>
<caption>A matrix: 4 × 7 of type dbl</caption>
<thead>
	<tr><th scope=col>alone</th><th scope=col>dark</th><th scope=col>lot</th><th scope=col>many</th><th scope=col>mary</th><th scope=col>mother</th><th scope=col>mothers</th></tr>
</thead>
<tbody>
	<tr><td>0.7071068</td><td>0.7071068</td><td>0.0000000</td><td>0.0000000</td><td>0.0000000</td><td>0.0000000</td><td>0.0000000</td></tr>
	<tr><td>0.0000000</td><td>0.0000000</td><td>0.4869343</td><td>0.0000000</td><td>0.6176144</td><td>0.6176144</td><td>0.0000000</td></tr>
	<tr><td>0.7071068</td><td>0.7071068</td><td>0.0000000</td><td>0.0000000</td><td>0.0000000</td><td>0.0000000</td><td>0.0000000</td></tr>
	<tr><td>0.0000000</td><td>0.0000000</td><td>0.4869343</td><td>0.6176144</td><td>0.0000000</td><td>0.0000000</td><td>0.6176144</td></tr>
</tbody>
</table>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

## Stemming
Stemming is a kind of normalization for words. **Normalization** is a technique where a set of words in a sentence are converted into a sequence to shorten its lookup. The words which have the same meaning but have some variation according to the context or sentence are normalized. In another word, there is one root word, but there are many variations of the same words 
For example, the root word is "eat" and it's variations are "eats, eating, eaten and like so". In the same way, with the help of Stemming, we can find the root word of any variations. <br>
<br>NLTK has an algorithm named as "PorterStemmer". This algorithm accepts the list of tokenized word and stems it into root word 


```python
library(qdap)
library(tm)
corpus <- "Text mining usually involves the process of structuring the input text. The overarching goal is, essentially, to turn text into data for analysis, via application of natural language processing (NLP) and analytical methods."
# Remove punctuation: rm_punc
rm_punc <- removePunctuation(corpus)
# Create character vector: n_char_vec
n_char_vec <- unlist(strsplit(rm_punc, split = ' '))
# Perform word stemming: stem_doc
stem_doc <- stemDocument(n_char_vec)
# Print stem_doc
print("Below is Stemming performed on corpus")
stem_doc


```

    [1] "Below is Stemming performed on corpus"
    


<ol class=list-inline><li>'Text'</li><li>'mine'</li><li>'usual'</li><li>'involv'</li><li>'the'</li><li>'process'</li><li>'of'</li><li>'structur'</li><li>'the'</li><li>'input'</li><li>'text'</li><li>'The'</li><li>'overarch'</li><li>'goal'</li><li>'is'</li><li>'essenti'</li><li>'to'</li><li>'turn'</li><li>'text'</li><li>'into'</li><li>'data'</li><li>'for'</li><li>'analysi'</li><li>'via'</li><li>'applic'</li><li>'of'</li><li>'natur'</li><li>'languag'</li><li>'process'</li><li>'NLP'</li><li>'and'</li><li>'analyt'</li><li>'method'</li></ol>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>

## Lemmatization
Lemmatization is the algorithmic process of **finding the lemma of a word depending on their meaning**. Lemmatization usually refers to the morphological analysis of words, which aims to remove inflectional endings. It helps in returning the base or dictionary form of a word, which is known as the lemma. <br><br>


__Lemmatization is preferred over the former because of the below reason:__
Stemming algorithm works by cutting the suffix from the word. 
In a broader sense cuts either the beginning or end of the word. 
On the contrary, Lemmatization is a more powerful operation, and it takes into consideration morphological analysis 
of the words. It returns the lemma which is the base form of all its inflectional forms. 
In-depth linguistic knowledge is required to create dictionaries and look for the proper form of the word. 
Stemming is a general operation while lemmatization is an intelligent operation where the proper form will be 
looked in the dictionary. Hence, lemmatization helps in forming better machine learning features.


```python
library(textstem)
corpus <- c("run", "ran", "running","jogging","walking","fishing","fishes")
print("Below is Lemmatization performed on corpus")
lemmatize_words(corpus)
```

    [1] "Below is Lemmatization performed on corpus"
    


<ol class=list-inline><li>'run'</li><li>'run'</li><li>'run'</li><li>'jog'</li><li>'walk'</li><li>'fish'</li><li>'fish'</li></ol>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Pre-Processing" role="tab" aria-controls="settings">Go to Top<span class="badge badge-primary badge-pill"></span></a>





