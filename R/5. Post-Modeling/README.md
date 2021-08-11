# Model Evaluation

# Notebook Content
[Libraries used](#Library)<br>
## Evaluation Metrics for Regression
[Mean Absolute Error](#Mean-Absolute-Error)<br>
[Mean Squared Error](#Mean-Squared-Error)<br>
[Mean Squared Logarithmic Error](#Mean-Squared-Logarithmic-Error)<br>
[Median Absolute Error](#Median-Absolute-Error)<br>
[R² score](#1)<br>
[Mean Percentage Error](#Mean-Percentage-Error)<br>
[Mean Absolute Percentage Error](#Mean-Absolute-Percentage-Error)<br>
[Weighted Mean Absolute Percentage Error](#Weighted-Mean-Absolute-Error)<br>
[Metric Selection](#Tips-for-Metric-Selection)<br>
[Cross Validation](#Cross-Validation)<br>
## Accuracy Metrics for Binary Classification
[Confusion Matrix](#Confusion-Matrix)<br>
[Accuracy and Cohen’s kappa](#2)<br>
[Recall](#Recall)<br>
[Precision](#Precision)<br>
[F1 Score](#F1-Score)<br>
[ROC (Receiver Operating Characteristics)](#3)<br>
[AUC (Area Under the Curve)](#4)<br>
## Accuracy Metrics for Multi-Class Classification
[Accuarcy Score](#Accuarcy-Score)<br>
[Confusion Matrix](#Confusion-Matrix)<br>
[Logarithmic Loss](#Logarithmic-Loss)<br>
[ROC and AUC](#ROC-and-AUC)<br>
[Precision Recall Curve](#Precision-Recall-Curve)<br>

# Library


```python
# install.packages("Metrics")
# install.packages("MLmetrics")
# install.packages("klaR")
# install.packages("CORElearn")
# install.packages("mnormt")
# install.packages("irr")
# install.packages("MLeval")
# install.packages("ROSE")
# install.packages("AUC")
# install.packages("multiROC")
# install.packages('caret')
# install.packages('randomForest')
# install.packages('devtools')
# install.packages('multiROC')
# install.packages('dummies')
```

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Evaluation Metrics for Regression

## Mean Absolute Error
The mean_absolute_error function computes mean absolute error, a risk metric corresponding to the expected value of the absolute error loss or *l1*-norm loss.In statistics, mean absolute error (MAE) is a **measure of errors between paired observations** expressing the same phenomenon.

If <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i"> is the predicted value of the i-th sample, and <img src="https://render.githubusercontent.com/render/math?math=\y_i"> is the corresponding true value, then the mean absolute error (MAE) estimated over <img src="https://render.githubusercontent.com/render/math?math=n_{\text{samples}}"> is defined as:<br>

<img src="https://render.githubusercontent.com/render/math?math=\text{MAE}(y, \hat{y}) = \frac{1}{n_{\text{samples}}} \sum_{i=0}^{n_{\text{samples}}-1} \left| y_i - \hat{y}_i \right|">.<br><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Code" role="tab" aria-controls="messages">Go to Code<span class="badge badge-primary badge-pill"> </span></a>

## Mean Squared Error
The mean_squared_error function computes mean square error, a risk metric **corresponding to the expected value of the squared (quadratic) error or loss**.

If <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i"> is the predicted value of the *i*-th sample, and <img src="https://render.githubusercontent.com/render/math?math=\y_i"> is the corresponding true value, then the mean squared error (MSE) estimated over <img src="https://render.githubusercontent.com/render/math?math=n_{\text{samples}}"> is defined as

<img src="https://render.githubusercontent.com/render/math?math=\text{MSE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (y_i - \hat{y}_i)^2"><br><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Code" role="tab" aria-controls="messages">Go to Code<span class="badge badge-primary badge-pill"> </span></a>


## Mean Squared Logarithmic Error
The mean_squared_log_error function computes a **risk metric corresponding to the expected value of the squared logarithmic (quadratic) error or loss.**

If <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i"> is the predicted value of the *i*-th sample, and <img src="https://render.githubusercontent.com/render/math?math=\y_i"> is the corresponding true value, then the mean squared logarithmic error (MSLE) estimated over <img src="https://render.githubusercontent.com/render/math?math=n_{\text{samples}}"> is defined as:<br>

<img src="https://render.githubusercontent.com/render/math?math=\text{MSLE}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples} - 1} (\log_e (1 + y_i) - \log_e (1 + \hat{y}_i) )^2"><br>
Where <img src="https://render.githubusercontent.com/render/math?math=\log_e (x)"> means the natural logarithm of *x*. <br>
**This metric is best to use when targets having exponential growth, such as population counts**, average sales of a commodity over a span of years etc. Note that this metric penalizes an under-predicted estimate greater than an over-predicted estimate.<br><br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Code" role="tab" aria-controls="messages">Go to Code<span class="badge badge-primary badge-pill"> </span></a>

## Median Absolute Error
The median_absolute_error is particularly interesting because **it is robust to outliers**. The loss is calculated by taking the **median of all absolute differences between the target and the prediction**.<br><br>
If <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i"> is the predicted value of the *i*-th sample and <img src="https://render.githubusercontent.com/render/math?math=y_i"> is the corresponding true value, then the median absolute error (MedAE) estimated over <img src="https://render.githubusercontent.com/render/math?math=n_{\text{samples}}"> is defined as<br>

<img src="https://render.githubusercontent.com/render/math?math=\text{MedAE}(y, \hat{y}) = \text{median}(\mid y_1 - \hat{y}_1 \mid, \ldots, \mid y_n - \hat{y}_n \mid)">

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Code" role="tab" aria-controls="messages">Go to Code<span class="badge badge-primary badge-pill"> </span></a>

<a id = '1'></a>

## R² score
The r2_score function **computes the coefficient of determination**, usually denoted as R².<br><br>
It represents the **proportion of variance (of y) that has been explained by the independent variables** in the model. It provides an **indication of goodness of fit** and therefore a measure of how well unseen samples are likely to be predicted by the model, through the proportion of explained variance.<br><br>
As such variance is dataset dependent, **R² may not be meaningfully comparable across different datasets**. <br>Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R² score of 0.0.

If <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i"> is the predicted value of the *i*-th sample and <img src="https://render.githubusercontent.com/render/math?math=\y_i"> is the corresponding true value for total $n$ samples, the estimated R² is defined as:

<img src="https://render.githubusercontent.com/render/math?math=R^2(y, \hat{y}) = 1 - \frac{\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}{\sum_{i=1}^{n} (y_i - \bar{y})^2}"> <br>
where <img src="https://render.githubusercontent.com/render/math?math=\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i$ and $\sum_{i=1}^{n} (y_i - \hat{y}_i)^2 = \sum_{i=1}^{n} \epsilon_i^2"><br>


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Code" role="tab" aria-controls="messages">Go to Code<span class="badge badge-primary badge-pill"> </span></a>

## Mean Percentage Error
In statistics, the mean percentage error (MPE) is the computed average of percentage errors by which forecasts of a model differ from actual values of the quantity being forecast.

The formula for the mean percentage error is:
![MPE.svg](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/dataset/Mean%20Percentage%20Error.svg) <br>
where <img src="https://render.githubusercontent.com/render/math?math=a_t"> is the actual value of the quantity being forecast, <img src="https://render.githubusercontent.com/render/math?math=f_t"> is the forecast, and n is the number of different times for which the variable is forecast.

Because actual rather than absolute values of the forecast errors are used in the formula, positive and negative forecast errors can offset each other; as a result the formula can be used as a measure of the bias in the forecasts.

A disadvantage of this measure is that it is undefined whenever a single actual value is zero.<br>
The fuction defination of MPE is given below

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Code" role="tab" aria-controls="messages">Go to Code<span class="badge badge-primary badge-pill"> </span></a>

## Mean Absolute Percentage Error
It is a simple average of absolute percentage errors. The MAPE calculation is as follows:

where <img src="https://render.githubusercontent.com/render/math?math=A_t"> is the actual value and <img src="https://render.githubusercontent.com/render/math?math=F_t"> is the forecast value. The MAPE is also sometimes reported as a percentage, which is the above equation multiplied by 100. The difference between <img src="https://render.githubusercontent.com/render/math?math=A_t"> and <img src="https://render.githubusercontent.com/render/math?math=F_t"> is divided by the actual value At again. <br>
It **cannot be used if there are zero values** (which sometimes happens for example in demand data) because there would be a division by zero.<br>
For forecasts which are too low the percentage error cannot exceed 100%, but **for forecasts which are too high there is no upper limit to the percentage error**.<br>


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>

# Weighted Mean Absolute Error

The wMAPE is the metric in which the sales are weighted by sales volumne. Weighted Mean Absolute Percentage Error, as the name suggests, is a **measure that gives greater importance to faster selling products**. Thus it overcomes one of the potential drawbacks of MAPE. There is a very simple way to calculate WMAPE. This involves adding together the absolute errors at the detailed level, then calculating the total of the errors as a percentage of total sales.  This method of calculation leads to the additional benefit that it is robust to individual instances when the base is zero, thus overcoming the divide by zero problem that often occurs with MAPE.

WMAPE is a highly useful measure and is becoming increasingly popular both in corporate KPIs and for operational use. It is easily calculated and gives a concise forecast accuracy measurement that can be used to summarise performance at any detailed level across any grouping of products and/or time periods. If a measure of accuracy required this is calculated as 100 - WMAPE.

Now, in order to show how to run the different Accuracy metrics, we will be prforming Random Forest Regression on a dataset the download link for which is given below:

https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009<br>
We will start by importing the necessary libraries and the required dataset.


```python
# Loading the data
winedata = read.csv('dataset/winequality-red.csv')
head(winedata)
```


<table>
<caption>A data.frame: 6 × 12</caption>
<thead>
	<tr><th></th><th scope=col>fixed.acidity</th><th scope=col>volatile.acidity</th><th scope=col>citric.acid</th><th scope=col>residual.sugar</th><th scope=col>chlorides</th><th scope=col>free.sulfur.dioxide</th><th scope=col>total.sulfur.dioxide</th><th scope=col>density</th><th scope=col>pH</th><th scope=col>sulphates</th><th scope=col>alcohol</th><th scope=col>quality</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td> 7.4</td><td>0.70</td><td>0.00</td><td>1.9</td><td>0.076</td><td>11</td><td>34</td><td>0.9978</td><td>3.51</td><td>0.56</td><td>9.4</td><td>5</td></tr>
	<tr><th scope=row>2</th><td> 7.8</td><td>0.88</td><td>0.00</td><td>2.6</td><td>0.098</td><td>25</td><td>67</td><td>0.9968</td><td>3.20</td><td>0.68</td><td>9.8</td><td>5</td></tr>
	<tr><th scope=row>3</th><td> 7.8</td><td>0.76</td><td>0.04</td><td>2.3</td><td>0.092</td><td>15</td><td>54</td><td>0.9970</td><td>3.26</td><td>0.65</td><td>9.8</td><td>5</td></tr>
	<tr><th scope=row>4</th><td>11.2</td><td>0.28</td><td>0.56</td><td>1.9</td><td>0.075</td><td>17</td><td>60</td><td>0.9980</td><td>3.16</td><td>0.58</td><td>9.8</td><td>6</td></tr>
	<tr><th scope=row>5</th><td> 7.4</td><td>0.70</td><td>0.00</td><td>1.9</td><td>0.076</td><td>11</td><td>34</td><td>0.9978</td><td>3.51</td><td>0.56</td><td>9.4</td><td>5</td></tr>
	<tr><th scope=row>6</th><td> 7.4</td><td>0.66</td><td>0.00</td><td>1.8</td><td>0.075</td><td>13</td><td>40</td><td>0.9978</td><td>3.51</td><td>0.56</td><td>9.4</td><td>5</td></tr>
</tbody>
</table>



Next, split 80% of the data to the training set while 20% of the data to test set using below code.


```python
# Number of rows to take in train and test
N_all = nrow(winedata)
N_train = round(0.8*(N_all))
N_test = N_all-N_train

# Dividing training set and testing set
wine_train <- winedata[1:N_train,]
wine_test <- winedata[(N_train+1): N_all,]
```

Now, random forest regression on the training set.


```python
# Required libraries
library(randomForest)
options(warn=-1)

# Fitting the model
rf <- randomForest(quality ~ ., data = wine_train, ntree=100)
summary(rf)
```

    randomForest 4.6-14
    Type rfNews() to see new features/changes/bug fixes.
    


                    Length Class  Mode     
    call               4   -none- call     
    type               1   -none- character
    predicted       1279   -none- numeric  
    mse              100   -none- numeric  
    rsq              100   -none- numeric  
    oob.times       1279   -none- numeric  
    importance        11   -none- numeric  
    importanceSD       0   -none- NULL     
    localImportance    0   -none- NULL     
    proximity          0   -none- NULL     
    ntree              1   -none- numeric  
    mtry               1   -none- numeric  
    forest            11   -none- list     
    coefs              0   -none- NULL     
    y               1279   -none- numeric  
    test               0   -none- NULL     
    inbag              0   -none- NULL     
    terms              3   terms  call     



```python
# Predicting using the model
predictions <- predict(rf,newdata=wine_test)[1:N_test]

```

### _Code_
Now we can calculate the accuracy of our model using the different accuracy metrics that we have explained above. 


```python
mpe_fnc <- function(y_true, y_pred) {
    x <- mean((y_true - y_pred) / y_true)
    return(x)}

# Calculate the errors
library(Metrics)
options(warn=-1)
rmse <- rmse(wine_test$quality,predictions)
mse <- mse(wine_test$quality,predictions)
mae <- mae(wine_test$quality,predictions)
msle <- msle(wine_test$quality,predictions)
mdae <- mdae(wine_test$quality,predictions)
mape <- mape(wine_test$quality,predictions)
mpe <- mpe_fnc(wine_test$quality,predictions)

library(MLmetrics)
options(warn=-1)
r2_score <- R2_Score(predictions,wine_test$quality)

# Print scores
print(paste("Mean Absolute Error: ", mae))      ##Make them round 2
print(paste("Mean Square Error: ", mse))
print(paste("Root Mean Square Error: ", rmse))
print(paste("Mean Log Square Error: ", msle))
print(paste("Median Absolute Error: ", mdae))
print(paste("Mean Absolute Percentage Error: ", mape))
print(paste("Mean Absolute Percentage Error: ", mpe))
print(paste("R2 Score: ", r2_score))
```

    
    Attaching package: 'MLmetrics'
    
    The following object is masked from 'package:base':
    
        Recall
    
    

    [1] "Mean Absolute Error:  0.498466666666667"
    [1] "Mean Square Error:  0.428438608333333"
    [1] "Root Mean Square Error:  0.654552219714618"
    [1] "Mean Log Square Error:  0.0111499833183455"
    [1] "Median Absolute Error:  0.372333333333333"
    [1] "Mean Absolute Percentage Error:  0.0969457564484127"
    [1] "Mean Absolute Percentage Error:  -0.0372608581349206"
    [1] "R2 Score:  0.292567829377365"
    


```python
#Dataframe of errors given by all the metrics explained above
Error_Metrics = c('Mean Absolute Error','Mean Square Error','Root Mean Square Error',
              'Mean Log Square Error','Median Absolute Error','Mean Absolute Percentage Error','Mean Absolute Percentage Error','R2 Score')
Score = c(mae, mse, rmse, msle, mdae, mape, mpe, r2_score)

report = data.frame(Error_Metrics,Score)
report
```


<table>
<thead><tr><th scope=col>Error_Metrics</th><th scope=col>Score</th></tr></thead>
<tbody>
	<tr><td>Mean Absolute Error           </td><td> 0.49846667                   </td></tr>
	<tr><td>Mean Square Error             </td><td> 0.42843861                   </td></tr>
	<tr><td>Root Mean Square Error        </td><td> 0.65455222                   </td></tr>
	<tr><td>Mean Log Square Error         </td><td> 0.01114998                   </td></tr>
	<tr><td>Median Absolute Error         </td><td> 0.37233333                   </td></tr>
	<tr><td>Mean Absolute Percentage Error</td><td> 0.09694576                   </td></tr>
	<tr><td>Mean Absolute Percentage Error</td><td>-0.03726086                   </td></tr>
	<tr><td>R2 Score                      </td><td> 0.29256783                   </td></tr>
</tbody>
</table>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Tips for Metric Selection
- The **MAE is also the most intuitive of the metrics** one should by just looking at the absolute difference between the data and the model’s predictions<br>


- MAE does not indicate underperformance or overperformance of the model (whether or not the model under or overshoots actual data). Each residual contributes proportionally to the total amount of error, meaning that larger errors will contribute linearly to the overall error<br> 


- A **small MAE** suggests the model is great at prediction, while a **large MAE** suggests that, model may have trouble in certain areas. A MAE of 0 means that your model is a perfect predictor of the outputs (but this will almost never happen)<br><br>


- While the MAE is easily interpretable, using the absolute value of the residual often is not as desirable as squaring this difference. Depending on how the model to treat outliers, or extreme values, in the data, one may want to bring more attention to these outliers or downplay them. The issue of outliers can play a major role in which error metric you use<br>


- Because of squaring the difference, the **MSE will almost always be bigger than the MAE**. For this reason, we cannot directly compare the MAE to the MSE. **one can only compare model’s error metrics to those of a competing model**<br>

- The effect of the square term in the MSE equation is most apparent with the presence of outliers in our data. **While each residual in MAE contributes proportionally to the total error, the error grows quadratically in MSE**<br><br>


- RMSE is the square root of the MSE. Because the **MSE is squared, its units do not match that of the original output**. Researchers will often use **RMSE to convert the error metric** back into similar units, making interpretation easier<br>


- Taking the square root before they are averaged, **RMSE gives a relatively high weight to large errors**, so RMSE should be useful when large errors are undesirable.<br><br>


- Just as MAE is the average magnitude of error produced by your model, the **MAPE is how far the model’s predictions are off** from their corresponding outputs on average. Like MAE, MAPE also has a clear interpretation since percentages are easier for people to conceptualize. **Both MAPE and MAE are robust to the effects of outliers**<br>


- Many of MAPE’s weaknesses actually stem from use division operation. MAPE is **undefined for data points where the value is 0**. Similarly, the MAPE can grow unexpectedly large if the actual values are exceptionally small themselves<br>


- Finally, the MAPE is **biased towards predictions** that are systematically less than the actual values themselves. That is to say, MAPE will be lower when the prediction is lower than the actual compared to a prediction that is higher by the same amount. 

The table given below can also be helpful in regard to metric understanding:


|Acroynm|Full Name|Residual Operation?|Robust To Outliers|
|-------|---------|-------------------|------------------|
|MAE|Mean Absolute Error|Absolute Value|Yes|
|MSE|Mean Squared Error	Square|No|No|
|RMSE|Root Mean Squared Error|Square|No|
|MAPE|Mean Absolute Percentage Error|Absolute Value|Yes|
|MPE|Mean Percentage Error|N/A|Yes|


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Cross Validation
Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.
Cross-validation is primarily used in applied machine learning to estimate the skill of a machine learning model on unseen data. That is, to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.

## Leave one out cross validation - LOOCV
Leave out one data point and build the model on the rest of the data set
Test the model against the data point that is left out at step 1 and record the test error associated with the prediction
Repeat the process for all data points
Compute the overall prediction error by taking the average of all these test error estimates recorded at step 2.
Practical example in R using the caret package:


```python
# Loading required libraries and data
library(caret)
options(warn=-1)
data(iris)

# Define train control for LOOCV cross validation
train_control <- trainControl(method="LOOCV")

# Fit Naive Bayes Model
model <- train(Species~., data=iris, trControl=train_control, method="nb")


# Summarise Results
print(model)
```

    Loading required package: lattice
    Loading required package: ggplot2
    Registered S3 methods overwritten by 'ggplot2':
      method         from 
      [.quosures     rlang
      c.quosures     rlang
      print.quosures rlang
    
    Attaching package: 'ggplot2'
    
    The following object is masked from 'package:randomForest':
    
        margin
    
    
    Attaching package: 'caret'
    
    The following objects are masked from 'package:MLmetrics':
    
        MAE, RMSE
    
    The following objects are masked from 'package:Metrics':
    
        precision, recall
    
    

    Naive Bayes 
    
    150 samples
      4 predictor
      3 classes: 'setosa', 'versicolor', 'virginica' 
    
    No pre-processing
    Resampling: Leave-One-Out Cross-Validation 
    Summary of sample sizes: 149, 149, 149, 149, 149, 149, ... 
    Resampling results across tuning parameters:
    
      usekernel  Accuracy   Kappa
      FALSE      0.9533333  0.93 
       TRUE      0.9600000  0.94 
    
    Tuning parameter 'fL' was held constant at a value of 0
    Tuning
     parameter 'adjust' was held constant at a value of 1
    Accuracy was used to select the optimal model using the largest value.
    The final values used for the model were fL = 0, usekernel = TRUE and adjust
     = 1.
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>

## K-Folds Cross Validation
In K-Folds Cross Validation we split our data into k different subsets (or folds). We use k-1 subsets to train our data and leave the last subset (or the last fold) as test data. We then average the model against each of the folds and then finalize our model. After that we test it against the test set.<br><br>

![image.png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/dataset/K-Fold%20Cross%20Validation.png)


```python
# Loading required libraries and data
library(caret)
options(warn=-1)
data(iris)

# Define train control for k fold cross validation
train_control <- trainControl(method="cv", number=10)

# Fit Naive Bayes Model
model <- train(Species~., data=iris, trControl=train_control, method="nb")

# Summarise Results
print(model)
```

    Naive Bayes 
    
    150 samples
      4 predictor
      3 classes: 'setosa', 'versicolor', 'virginica' 
    
    No pre-processing
    Resampling: Cross-Validated (10 fold) 
    Summary of sample sizes: 135, 135, 135, 135, 135, 135, ... 
    Resampling results across tuning parameters:
    
      usekernel  Accuracy   Kappa
      FALSE      0.9533333  0.93 
       TRUE      0.9533333  0.93 
    
    Tuning parameter 'fL' was held constant at a value of 0
    Tuning
     parameter 'adjust' was held constant at a value of 1
    Accuracy was used to select the optimal model using the largest value.
    The final values used for the model were fL = 0, usekernel = FALSE and adjust
     = 1.
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>

# StratifiedKFold
StratifiedKFold is a variation of KFold. First, StratifiedKFold shuffles your data, after that splits the data into n_splits parts and Done. Now, it will use each part as a test set. Note that it only and always shuffles data one time before splitting.



![image.png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/dataset/Stratified%20K-Fold.png)


```python
# Stratified sampling using CORElearn library
library(CORElearn)

# Loading iris data
data(iris)

# Stratified sampling
folds <- 10
foldIdx <- cvGenStratified(iris$Species, k=folds)
evalCore<-list()
for (j in 1:folds) {
    dTrain <- iris[foldIdx!=j,]
    dTest  <- iris[foldIdx==j,]
    modelCore <- CoreModel(Species~., dTrain, model="rf") 
    predCore <- predict(modelCore, dTest)
    evalCore[[j]] <- modelEval(modelCore, correctClass=dTest$Species,
              predictedClass=predCore$class, predictedProb=predCore$prob ) 
    destroyModels(modelCore)
}
results <- gatherFromList(evalCore)
sapply(results, mean)
```


<dl class=dl-horizontal>
	<dt>accuracy</dt>
		<dd>0.96</dd>
	<dt>averageCost</dt>
		<dd>0.04</dd>
	<dt>informationScore</dt>
		<dd>1.45398382803449</dd>
	<dt>AUC</dt>
		<dd>0.996</dd>
	<dt>predictionMatrix</dt>
		<dd>0</dd>
	<dt>sensitivity</dt>
		<dd>0</dd>
	<dt>specificity</dt>
		<dd>0</dd>
	<dt>brierScore</dt>
		<dd>0.067426584829932</dd>
	<dt>kappa</dt>
		<dd>0.94</dd>
	<dt>precision</dt>
		<dd>0</dd>
	<dt>recall</dt>
		<dd>0</dd>
	<dt>Fmeasure</dt>
		<dd>0</dd>
	<dt>Gmean</dt>
		<dd>0</dd>
	<dt>KS</dt>
		<dd>0</dd>
	<dt>TPR</dt>
		<dd>0</dd>
	<dt>FPR</dt>
		<dd>0</dd>
</dl>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Repeated K-fold cross-validation
The process of splitting the data into k-folds can be repeated a number of times, this is called repeated k-fold cross validation.

The final model error is taken as the mean error from the number of repeats.

The following example uses 10-fold cross validation with 3 repeats:


```python
# Loading required libraries and data
library(caret)
options(warn=-1)
data(iris)

# Define train control for repeated k fold cross validation
train_control <- trainControl(method="repeatedcv", number=10)

# Fit Naive Bayes Model
model <- train(Species~., data=iris, trControl=train_control, method="nb")

# Summarise Results
print(model)
```

    Naive Bayes 
    
    150 samples
      4 predictor
      3 classes: 'setosa', 'versicolor', 'virginica' 
    
    No pre-processing
    Resampling: Cross-Validated (10 fold, repeated 1 times) 
    Summary of sample sizes: 135, 135, 135, 135, 135, 135, ... 
    Resampling results across tuning parameters:
    
      usekernel  Accuracy   Kappa
      FALSE      0.9533333  0.93 
       TRUE      0.9600000  0.94 
    
    Tuning parameter 'fL' was held constant at a value of 0
    Tuning
     parameter 'adjust' was held constant at a value of 1
    Accuracy was used to select the optimal model using the largest value.
    The final values used for the model were fL = 0, usekernel = TRUE and adjust
     = 1.
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Accuracy Metrics for Binary Classification

Pima Indian Diabetes classification dataset will be used to show how model evaluation for Binary Classification Can be done. It is an inbuilt r dataset and can also be downloaded from the link below:<br>
https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset


```python
# loading data
# load libraries
library(caret)
library(mlbench)
data(PimaIndiansDiabetes)

# train test split
library(caret)
set.seed(3456)
trainIndex <- createDataPartition(PimaIndiansDiabetes$diabetes, p = .8, 
                                  list = FALSE, 
                                  times = 1)
head(trainIndex)

train <- PimaIndiansDiabetes[ trainIndex,]
test  <- PimaIndiansDiabetes[-trainIndex,] 
x <- train[ ,1:8]
y <- train[ ,9]
```


<table>
<thead><tr><th scope=col>Resample1</th></tr></thead>
<tbody>
	<tr><td>1</td></tr>
	<tr><td>2</td></tr>
	<tr><td>3</td></tr>
	<tr><td>4</td></tr>
	<tr><td>5</td></tr>
	<tr><td>6</td></tr>
</tbody>
</table>




```python
# Loading required libraries
# install.packages('randomForest')
library(randomForest)

# Fitting the model
model_rf = randomForest(x, y, ntree = 20, method="class")
model_rf
```


    
    Call:
     randomForest(x = x, y = y, ntree = 20, method = "class") 
                   Type of random forest: classification
                         Number of trees: 20
    No. of variables tried at each split: 2
    
            OOB estimate of  error rate: 28.18%
    Confusion matrix:
        neg pos class.error
    neg 329  70   0.1754386
    pos 103 112   0.4790698



```python
predictions <- predict(model_rf, test)
```

The confusionMatrix function gives the values for many evaluation metrics as can be seen below:


```python
#confusion matrix
library(e1071)
library(caret)
results <- confusionMatrix(as.factor(predictions), test$diabetes)
results
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction neg pos
           neg  80  18
           pos  20  35
                                              
                   Accuracy : 0.7516          
                     95% CI : (0.6754, 0.8179)
        No Information Rate : 0.6536          
        P-Value [Acc > NIR] : 0.005891        
                                              
                      Kappa : 0.4563          
                                              
     Mcnemar's Test P-Value : 0.871131        
                                              
                Sensitivity : 0.8000          
                Specificity : 0.6604          
             Pos Pred Value : 0.8163          
             Neg Pred Value : 0.6364          
                 Prevalence : 0.6536          
             Detection Rate : 0.5229          
       Detection Prevalence : 0.6405          
          Balanced Accuracy : 0.7302          
                                              
           'Positive' Class : neg             
                                              


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Confusion Matrix

A confusion matrix is a summary of prediction results on a classification problem. The number of correct and incorrect predictions are summarized with count values and broken down by each class. This is the key to the confusion matrix. The confusion matrix shows the ways in which your classification model is confused when it makes predictions. It gives us insight not only into the errors being made by a classifier but more importantly the types of errors that are being made.<br>

![confusion%20matrix.PNG](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/dataset/confusion_matrix.png)

Here,

<br>Class 1 : Positive
<br>Class 2 : Negative

**Definition of the Terms:**

<br>Positive (P) : Observation is positive (for example: is an apple).
<br>Negative (N) : Observation is not positive (for example: is not an apple).
<br>True Positive (TP) : Observation is positive, and is predicted to be positive.
<br>False Negative (FN) : Observation is positive, but is predicted negative.
<br>True Negative (TN) : Observation is negative, and is predicted to be negative.
<br>False Positive (FP) : Observation is negative, but is predicted positive.


```python
table_ = data.frame(results$table)
```


```python
ggplot(table_, aes(x=table_$Reference, y=table_$Prediction, fill=table_$Freq)) +
  geom_tile() + theme_bw() + coord_equal() +
#   guides(fill= TRUE) + # removing legend for `fill`
  labs(title = "Confusion Matrix")+ 
  geom_text(aes(label=table_$Freq), color="black") 
```


![png](https://github.com/Affineindia/ML-Best-Practices/blob/master/R/Images/Post-Modeling/output_55_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>

<a id = '2'></a>

# Accuracy and Cohen’s kappa
These are the default metrics used to evaluate algorithms on binary and multi-class classification datasets in caret.

Accuracy is the percentage of correctly classified instances out of all instances. It is **more useful on a binary classification than multi-class classification problems** because it can be less clear exactly how the accuracy breaks down across those classes (e.g. you need to go deeper with a confusion matrix).

Kappa or Cohen’s Kappa is like classification accuracy, except that **it is normalized at the baseline** of random chance on your dataset. It is a more **useful measure to use on problems that have an imbalance in the classes** (e.g. 70-30 split for classes 0 and 1 and you can achieve 70% accuracy by predicting all instances are for class 0).

It is a statistic that **measures inter-annotator agreement**.

This function computes Cohen’s kappa, a **score that expresses the level of agreement between two annotators** on a classification problem. It is defined as:<<br>
<img src="https://render.githubusercontent.com/render/math?math=\kappa = (p_o - p_e) / (1 - p_e)"><br>
where  is the empirical probability of agreement on the label assigned to any sample (the observed agreement ratio), and  is the expected agreement when both annotators assign labels randomly.  is estimated using a per-annotator empirical prior over the class labels


```python
# load libraries
library(caret)

# prepare resampling method
control <- trainControl(method="cv", number=5)
set.seed(7)

fit <- train(diabetes~., data=train, method="glm", metric="Accuracy", trControl=control)

# display results
print(fit)
```

    Generalized Linear Model 
    
    615 samples
      8 predictor
      2 classes: 'neg', 'pos' 
    
    No pre-processing
    Resampling: Cross-Validated (5 fold) 
    Summary of sample sizes: 492, 492, 492, 492, 492 
    Resampling results:
    
      Accuracy   Kappa    
      0.7756098  0.4754314
    
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Recall
Recall quantifies the number of positive class predictions made out of all positive examples in the dataset.<br>
Recall used When there is a high cost associated with false negatives. E.g. — fraud detection or sick patient detection.
![image.png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/dataset/Recall.png)


```python
recall_score <- Recall(test$diabetes, predictions)
recall_score
```


0.8


## Precision
Precision quantifies the number of positive class predictions that actually belong to the positive class.<br>
Precision is a good measure to determine when the cost of false positives is high. E.g. — email spam detection.
![image.png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/dataset/Precision.png)


```python
precision_score <- Precision(test$diabetes, predictions)
precision_score
```


0.816326530612245


## F1 Score 
The F1 score is the **harmonic mean of the precision and recall**, where an F1 score reaches its best value at 1 (perfect precision and recall). F-Measure provides a single score that balances both the concerns of precision and recall in one number.

![image.png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/dataset/F1Score.png)


```python
F1_Score <- F1_Score(test$diabetes, predictions)
F1_Score
```


0.808080808080808


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


<a id = '3'></a>

# ROC (Receiver Operating Characteristics)

ROC Curves are used to see how well your classifier can separate positive and negative examples and to identify the best threshold for separating them.

In ROC curves, the true positive rate (TPR, y-axis) is plotted against the false positive rate (FPR, x-axis). These quantities are defined as follows:<br>
**TPR(True Positve Rate) = TP/(TP+FN)<br>
FPR(False Positve Rate) = FP/(TN+FP)**



```python
# prepare resampling method
control <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=twoClassSummary)
set.seed(7)
fit <- train(diabetes~., data=train, method="glm", metric="ROC", trControl=control)
# display results
print(fit)
```

    Generalized Linear Model 
    
    615 samples
      8 predictor
      2 classes: 'neg', 'pos' 
    
    No pre-processing
    Resampling: Cross-Validated (5 fold) 
    Summary of sample sizes: 492, 492, 492, 492, 492 
    Resampling results:
    
      ROC        Sens    Spec     
      0.8258721  0.8975  0.5488372
    
    


```python
# Loading required libraries
library(ROSE)

# check imbalance on training set
table(train$diabetes)

# model estimation using logistic regression
fit.PimaIndiansDiabetes  <- glm(diabetes~., data=train, family="binomial")

# prediction on training set
pred.PimaIndiansDiabetes <- predict(fit.PimaIndiansDiabetes, newdata=train)

# plot the ROC curve (training set)
roc <- roc.curve(train$diabetes, pred.PimaIndiansDiabetes, 
          main="ROC curve 
 (Half circle depleted data)")

# check imbalance on test set 
table(test$diabetes)

# prediction using test set
PimaIndiansDiabetes.test <- predict(fit.PimaIndiansDiabetes, newdata=test)

# add the ROC curve (test set)
roc <- roc.curve(test$diabetes, PimaIndiansDiabetes.test, add=TRUE, col=2, 
          lwd=2, lty=2)
legend("topleft", c("on train set", "on test set"), 
        col=1:2, lty=1:2, lwd=2)
```

    Loaded ROSE 0.0-3
    
    


    
    neg pos 
    400 215 



    
    neg pos 
    100  53 



![png](https://github.com/Affineindia/ML-Best-Practices/blob/master/R/Images/Post-Modeling/output_71_3.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>

We can also use the ROCR package to plot the following:
1. true positive vs false positive
2. precision vs recall
3. sensitivity vs specitivity


```python
library(ROCR)

pred <- prediction( PimaIndiansDiabetes.test, test$diabetes )
perf <- performance( pred, "tpr", "fpr")
plot(perf, col="green")

## precision/recall curve (x-axis: recall, y-axis: precision)
perf1 <- performance(pred, "prec", "rec")
plot(perf1, col="blue")

## sensitivity/specificity curve (x-axis: specificity,
## y-axis: sensitivity)
perf1 <- performance(pred, "sens", "spec")
plot(perf1, col="red")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Post-Modeling/output_74_0.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Post-Modeling/output_74_1.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Post-Modeling/output_74_2.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>

<a id = '4'></a>

# AUC (Area Under the Curve)
The model performance is determined by looking at the area under the ROC curve (or AUC). **An excellent model has AUC near to the 1.0**, which means it has a good measure of separability. For this model, the AUC is the combined area of blue, green and purple rectangles,
<br>so the AUC = 0.4 x 0.6 + 0.2 x 0.8 + 0.4 x 1.0 = 0.80.

![AUC.PNG](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/dataset/AUC.png)


```python
library(pROC)
options(warn=-1)

roc1 <- roc(test$diabetes, PimaIndiansDiabetes.test, plot=TRUE,
           # arguments for auc, 
            print.auc=TRUE,
            col = "red")
roc1
```

    Type 'citation("pROC")' for a citation.
    
    Attaching package: 'pROC'
    
    The following object is masked from 'package:Metrics':
    
        auc
    
    The following objects are masked from 'package:stats':
    
        cov, smooth, var
    
    Setting levels: control = neg, case = pos
    Setting direction: controls < cases
    


    
    Call:
    roc.default(response = test$diabetes, predictor = PimaIndiansDiabetes.test,     plot = TRUE, print.auc = TRUE, col = "red")
    
    Data: PimaIndiansDiabetes.test in 100 controls (test$diabetes neg) < 53 cases (test$diabetes pos).
    Area under the curve: 0.8564



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Post-Modeling/output_78_2.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


```python
# plotting a ROC curve:
library(ROCR)

pred <- prediction( PimaIndiansDiabetes.test, test$diabetes )
perf <- performance( pred, "tpr", "fpr" )

# For a simple ROC Curve
plot(perf)

par(bg="lightblue", mai=c(1.2,1.5,1,1))
plot(perf, main="ROC curve", colorize=TRUE,
  xlab="False Positive Rate", ylab="True Positive Rate", box.lty=7, box.lwd=5,
  box.col="black", lwd=17, colorkey.relwidth=0.5, xaxis.cex.axis=2,
  xaxis.col='black', xaxis.col.axis="black", yaxis.col='black', yaxis.cex.axis=2,
  yaxis.at=c(0,0.5,0.8,0.85,0.9,1), yaxis.las=0.5, xaxis.lwd=2, yaxis.lwd=3,
  yaxis.col.axis="black", cex.lab=2, cex.main=2)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Post-Modeling/output_80_0.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Post-Modeling/output_80_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>

# Accuracy Metrics for Multi-Class Classification
In multiclass and multilabel classification task, the notions of precision, recall, and F-measures can be applied to each label independently. There are a few ways to combine results across labels, specified by the average argument to the average_precision_score (multilabel only), f1_score, fbeta_score, precision_recall_fscore_support, precision_score and recall_score functions, as described above. <br><br>
Note that if all labels are included, “micro”-averaging in a multiclass setting will produce precision, recall and $F$ that are all identical to accuracy. Also note that “weighted” averaging may produce an F-score that is not between precision and recall.

## Accuarcy Score
The accuracy_score function **computes the accuracy, either the fraction (default) or the count (normalize=False) of correct predictions**.

**In multilabel classification, the function returns the subset accuracy**. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.

If <img src="https://render.githubusercontent.com/render/math?math=\hat{y}_i"> is the predicted value of the *i*-th sample and <img src="https://render.githubusercontent.com/render/math?math=y_i"> is the corresponding true value, then the fraction of correct predictions over <img src="https://render.githubusercontent.com/render/math?math=n_\text{samples}"> is defined as:<br>
<img src="https://render.githubusercontent.com/render/math?math=\texttt{accuracy}(y, \hat{y}) = \frac{1}{n_\text{samples}} \sum_{i=0}^{n_\text{samples}-1} 1(\hat{y}_i = y_i)"><br>
where *(x)* is the indicator function.

## Precision, recall and F-measures
The precision is the ratio **tp / (tp + fp)** where tp is the number of true positives and fp the number of false positives. The precision is intuitively the **ability of the classifier not to label as positive a sample that is negative**.<br><br>
The best value is 1 and the worst value is 0.


The **F-measure** (<img src="https://render.githubusercontent.com/render/math?math=F_\beta"> and <img src="https://render.githubusercontent.com/render/math?math=F_1"> measures) can be interpreted as a weighted harmonic mean of the precision and recall. A <img src="https://render.githubusercontent.com/render/math?math=F_\beta"> measure reaches its best value at 1 and its worst score at 0. With <img src="https://render.githubusercontent.com/render/math?math=\beta = 1">, <img src="https://render.githubusercontent.com/render/math?math=F_\beta"> and <img src="https://render.githubusercontent.com/render/math?math=F_1"> are equivalent, and the recall and the precision are equally important.

The **average_precision_score** function computes the average precision (AP) from prediction scores. The value is between 0 and 1 and higher is better. AP is defined as: <br>
<img src="https://render.githubusercontent.com/render/math?math=\text{AP} = \sum_n (R_n - R_{n-1}) P_n"><br>
where <img src="https://render.githubusercontent.com/render/math?math=P_n"> and <img src="https://render.githubusercontent.com/render/math?math=R_n"> are the precision and recall at the nth threshold. With random predictions, the AP is the fraction of positive samples.

To show how these metrics can evaluate our model, we first import the dataset for multiclass classification. We will be using the fruit dataset for classification of fruits on the basis of the features such as mass, height, etc. The dataframe can be downloaded from the following link:<br> https://github.com/susanli2016/Machine-Learning-with-Python/blob/master/fruit_data_with_colors.txt


```python
fruit = read.table('fruit.txt',header = TRUE)
fruit =  subset(fruit, select = -(fruit_subtype))
fruit =  subset(fruit, select = -(fruit_name))
fruit$fruit_label = as.factor(fruit$fruit_label)
class(fruit$fruit_label)
```


'factor'



```python
library(caret)
library(randomForest)

alpha=0.8
d = sort(sample(nrow(fruit), nrow(fruit)*alpha))
train = fruit[d,]
test = fruit[-d,]
# Training with Random forest model
modfit.rf <- randomForest(fruit_label ~., data=train)
```


```python
# Predict the testing set with the trained model
pred_rf <- predict(modfit.rf, test, type = "class")

```


```python
# table(pred_rf,test$fruit_label)
# Accuracy and other metrics
result = confusionMatrix(pred_rf, test$fruit_label)
# confusionMatrix(pred_rf, test$fruit_label)
result
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction 1 2 3 4
             1 3 0 0 0
             2 0 2 0 0
             3 0 0 1 0
             4 1 0 1 4
    
    Overall Statistics
                                              
                   Accuracy : 0.8333          
                     95% CI : (0.5159, 0.9791)
        No Information Rate : 0.3333          
        P-Value [Acc > NIR] : 0.0005438       
                                              
                      Kappa : 0.7647          
                                              
     Mcnemar's Test P-Value : NA              
    
    Statistics by Class:
    
                         Class: 1 Class: 2 Class: 3 Class: 4
    Sensitivity            0.7500   1.0000  0.50000   1.0000
    Specificity            1.0000   1.0000  1.00000   0.7500
    Pos Pred Value         1.0000   1.0000  1.00000   0.6667
    Neg Pred Value         0.8889   1.0000  0.90909   1.0000
    Prevalence             0.3333   0.1667  0.16667   0.3333
    Detection Rate         0.2500   0.1667  0.08333   0.3333
    Detection Prevalence   0.2500   0.1667  0.08333   0.5000
    Balanced Accuracy      0.8750   1.0000  0.75000   0.8750



```python
table_ = data.frame(result$table)
```


```python
ggplot(table_, aes(x=table_$Reference, y=table_$Prediction, fill=table_$Freq)) +
  geom_tile() + theme_bw() + coord_equal() +
#   guides(fill= TRUE) + # removing legend for `fill`
  labs(title = "Confusion Matrix")+  # using a title instead
  geom_text(aes(label=table_$Freq), color="white") # printing values
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Post-Modeling/output_92_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Logarithmic Loss
Logarithmic Loss or LogLoss is used to evaluate binary classification but it is **more common for multi-class classification** algorithms. Specifically, it evaluates the probabilities estimated by the algorithms. Learn more about log loss here.

In this case we see logloss calculated for the iris flower multi-class classification problem


```python
# load libraries
library(caret)
# load the dataset
data(iris)
# prepare resampling method
control <- trainControl(method="cv", number=5, classProbs=TRUE, summaryFunction=mnLogLoss)
set.seed(7)
fit <- train(Species~., data=iris, method="rpart", metric="logLoss", trControl=control)
# display results
print(fit)
```

    CART 
    
    150 samples
      4 predictor
      3 classes: 'setosa', 'versicolor', 'virginica' 
    
    No pre-processing
    Resampling: Cross-Validated (5 fold) 
    Summary of sample sizes: 120, 120, 120, 120, 120 
    Resampling results across tuning parameters:
    
      cp    logLoss  
      0.00  0.4074697
      0.44  0.3603594
      0.50  1.0986123
    
    logLoss was used to select the optimal model using the smallest value.
    The final value used for the model was cp = 0.44.
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>

## ROC and AUC
An example of Receiver Operating Characteristic (ROC) metric to evaluate classifier output quality.

**ROC curves typically feature true positive rate on the Y axis, and false positive rate on the X axis**. This means that the top left corner of the plot is the “ideal” point - a false positive rate of zero, and a true positive rate of one. This is not very realistic, but it does mean that a larger area under the curve (AUC) is usually better.

The “steepness” of ROC curves is also important, since it is ideal to maximize the true positive rate while minimizing the false positive rate.

**ROC curves are typically used in binary classification** to study the output of a classifier. In order to extend ROC curve and ROC area to multi-label classification, **it is necessary to binarize the output**. **One ROC curve can be drawn per label**, but one can also draw a ROC curve by considering each element of the label indicator matrix as a binary prediction (micro-averaging).

**Another evaluation measure for multi-label classification is macro-averaging, which gives equal weight to the classification of each label**.


```python
# plot for multiclass at once
# Loading data and required libraries
library(ROCR)
library(klaR)
data(iris)

lvls = levels(iris$Species)
testidx = which(1:length(iris[, 1]) %% 5 == 0) 
iris.train = iris[testidx, ]
iris.test = iris[-testidx, ]

aucs = c()
plot(x=NA, y=NA, xlim=c(0,1), ylim=c(0,1),
     ylab='True Positive Rate',
     xlab='False Positive Rate',
     bty='n')

for (type.id in 1:3) {
  type = as.factor(iris.train$Species == lvls[type.id])

  nbmodel = NaiveBayes(type ~ ., data=iris.train[, -5])
  nbprediction = predict(nbmodel, iris.test[,-5], type='raw')

  score = nbprediction$posterior[, 'TRUE']
  actual.class = iris.test$Species == lvls[type.id]

  pred = prediction(score, actual.class)
  nbperf = performance(pred, "tpr", "fpr")

  roc.x = unlist(nbperf@x.values)
  roc.y = unlist(nbperf@y.values)
  lines(roc.y ~ roc.x, col=type.id+1, lwd=2)

  nbauc = performance(pred, "auc")
  nbauc = unlist(slot(nbauc, "y.values"))
  aucs[type.id] = nbauc
}

lines(x=c(0,1), c(0,1))

mean(aucs)
```

    Loading required package: MASS
    


0.9871875



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Post-Modeling/output_98_2.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>


```python
# Loading required libraries
library(dummies)
library(multiROC)
# library(devtools)

# Split the dataset
set.seed(123456)
total_number <- nrow(iris)
train_idx <- sample(total_number, round(total_number*0.6))
train_df <- iris[train_idx, ]
test_df <- iris[-train_idx, ]

# modeling 
rf_res <- randomForest::randomForest(Species~., data = train_df, ntree = 100)
rf_pred <- predict(rf_res, test_df, type = 'prob') 
rf_pred <- data.frame(rf_pred)
colnames(rf_pred) <- paste(colnames(rf_pred), "_pred_RF")

# for plotting necessary check need to be made("mandatory")
true_label <- dummies::dummy(test_df$Species, sep = ".")
true_label <- data.frame(true_label)
colnames(true_label) <- gsub(".*?\\.", "", colnames(true_label))
colnames(true_label) <- paste(colnames(true_label), "_true")
final_df <- cbind(true_label, rf_pred)

roc_res <- multi_roc(final_df, force_diag=T)
pr_res <- multi_pr(final_df, force_diag=T)
```

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>

# Precision Recall Curve


```python
# Loading required libraries
library(dummies)

# Plotting
plot_pr_df <- plot_pr_data(pr_res)
ggplot(plot_pr_df, aes(x=Recall, y=Precision)) + 
  geom_path(aes(color = Group, linetype=Method), size=1.5) + 
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), 
                 legend.justification=c(1, 0), legend.position=c(.95, .05),
                 legend.title=element_blank(), 
                 legend.background = element_rect(fill=NULL, size=0.5, 
                                                           linetype="solid", colour ="black"))

```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Post-Modeling/output_103_0.png)



```python
roc_res <- multi_roc(final_df, force_diag=T)
plot_roc_df <- plot_roc_data(roc_res)
require(ggplot2)
ggplot(plot_roc_df, aes(x = 1-Specificity, y=Sensitivity)) +
  geom_path(aes(color = Group, linetype=Method), size=1.5) +
  geom_segment(aes(x = 0, y = 0, xend = 1, yend = 1), 
                        colour='grey', linetype = 'dotdash') +
  theme_bw() + 
  theme(plot.title = element_text(hjust = 0.5), 
                 legend.justification=c(1, 0), legend.position=c(.95, .05),
                 legend.title=element_blank(), 
                 legend.background = element_rect(fill=NULL, size=0.5, 
                                                           linetype="solid", colour ="black"))
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Post-Modeling/output_104_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Evaluation" role="tab" aria-controls="messages">Go to top<span class="badge badge-primary badge-pill"></span></a>
