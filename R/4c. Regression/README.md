# Regression

## Notebook Content
[Libraries used](#Library)<br>
## Regression
[Overview](#Overview)<br>
[Multiple Linear Regression](#Multiple-Linear-Regression)<br>
[Polynomial Regression](#Polynomial-Regression)<br>
[Quantile Regression](#Quantile-Regression)<br>
[Ridge Regression](#Ridge-Regression)<br>
[Lasso Regression](#Lasso-Regression)<br>
[Elastic Net Regression](#Elastic-Net-Regression)<br>
[Support Vector Regression](#Support-Vector-Regression)<br>
[Decision Tree - CART](#Decision-Tree---CART)<br>
[Random Forest Regression](#Random-Forest-Regression)<br>
[Gradient Boosting (GBM)](#1)<br>
[Stochastic Gradient Descent](#Stochastic-Gradient-Descent)<br>
[KNN Regressor](#KNN-Regressor)<br>
[XGB Regressor](#XGB-Regressor)<br>
[Regressors Report](#Regressors-Report)<br>
## Cross Validation
[Grid Search CV](#Grid-Search-Cross-Validation)<br>
[Random Search CV](#Random-Search-Cross-Validation)<br>

# Library


```python
# #Loading required libraries
# install.packages("psych")
# install.packages("quantreg")
# install.packages("glmnet")
# install.packages("e1071")
# install.packages("caTools")
# install.packages("e1071")
# install.packages("Metrics")
# install.packages("reshape")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("randomForest")
# install.packages("gbm")
# install.packages("gradDescent")
# install.packages("class")
# install.packages("ggplot2")
```

# Overview

"In statistical modeling, regression analysis is a set of statistical processes for estimating the relationships between a continues dependent variable and one or more independent variables. Following are the Regression Algorithms widely used -"

## Multiple Linear Regression
In statistical modeling, regression analysis is a set of statistical processes for estimating the relationships between a continues dependent variable and one or more independent variables. In this Notebook, we will briefly study what linear regression is and how it can be implemented for both two variables and multiple variables using Scikit-Learn

The dataset we will be using here can be downloaded from - https://www.kaggle.com/zaraavagyan/weathercsv/data#

The dataset contains information on weather conditions recorded on each day at various weather stations around the world. Information includes precipitation, snowfall, temperatures, wind speed and whether the day included thunderstorms or other poor weather conditions. So our task is to predict the maximum temperature taking input feature as the minimum temperature.
The following command imports the CSV dataset: 


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



Let's Split our dataset to train and test sets


```python
# Number of rows to take in train and test
N_all = nrow(winedata)
#N_train = round(0.75*(N_all))
#N_test = N_all-N_train

N_train=sample(1:N_all,.75*N_all)
N_test=setdiff(1:N_all,N_train)

# Dividing training set and testing set
wine_train <- winedata[N_train,]
wine_test <- winedata[N_test,]
```

Fitting the linear regression model


```python
# Fitting data to linear regression model
linear_reg <- lm(quality~.,data = wine_train)
summary(linear_reg)
```


    
    Call:
    lm(formula = quality ~ ., data = wine_train)
    
    Residuals:
        Min      1Q  Median      3Q     Max 
    -2.6941 -0.3615 -0.0538  0.4386  1.9554 
    
    Coefficients:
                          Estimate Std. Error t value Pr(>|t|)    
    (Intercept)           8.695161  24.365497   0.357 0.721257    
    fixed.acidity         0.032281   0.029529   1.093 0.274532    
    volatile.acidity     -1.072256   0.139224  -7.702 2.82e-14 ***
    citric.acid          -0.303135   0.170347  -1.780 0.075411 .  
    residual.sugar        0.010359   0.017132   0.605 0.545540    
    chlorides            -1.670432   0.480088  -3.479 0.000521 ***
    free.sulfur.dioxide   0.004105   0.002436   1.685 0.092211 .  
    total.sulfur.dioxide -0.003078   0.000846  -3.639 0.000286 ***
    density              -5.055355  24.850635  -0.203 0.838834    
    pH                   -0.388182   0.218650  -1.775 0.076095 .  
    sulphates             0.825833   0.131606   6.275 4.89e-10 ***
    alcohol               0.317767   0.030277  10.495  < 2e-16 ***
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    Residual standard error: 0.6375 on 1187 degrees of freedom
    Multiple R-squared:  0.3708,	Adjusted R-squared:  0.365 
    F-statistic: 63.59 on 11 and 1187 DF,  p-value: < 2.2e-16
    


Let us make predictions on our test data using the linear regression model


```python
# Predict using Linear Regression Model
predictions = predict(linear_reg, wine_test)
coeff_df = data.frame(linear_reg$coefficients)
coeff_df

# Dataframe to show actual and predicted quality
df_lm  = data.frame("Actual Value" = wine_test$quality,"Predicted Values" = predictions)
head(df_lm, n = 5)

```


<table>
<caption>A data.frame: 12 × 1</caption>
<thead>
	<tr><th></th><th scope=col>linear_reg.coefficients</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>(Intercept)</th><td> 8.695161545</td></tr>
	<tr><th scope=row>fixed.acidity</th><td> 0.032280793</td></tr>
	<tr><th scope=row>volatile.acidity</th><td>-1.072255892</td></tr>
	<tr><th scope=row>citric.acid</th><td>-0.303134801</td></tr>
	<tr><th scope=row>residual.sugar</th><td> 0.010358684</td></tr>
	<tr><th scope=row>chlorides</th><td>-1.670432499</td></tr>
	<tr><th scope=row>free.sulfur.dioxide</th><td> 0.004104899</td></tr>
	<tr><th scope=row>total.sulfur.dioxide</th><td>-0.003078361</td></tr>
	<tr><th scope=row>density</th><td>-5.055355207</td></tr>
	<tr><th scope=row>pH</th><td>-0.388181859</td></tr>
	<tr><th scope=row>sulphates</th><td> 0.825833489</td></tr>
	<tr><th scope=row>alcohol</th><td> 0.317766918</td></tr>
</tbody>
</table>




<table>
<caption>A data.frame: 5 × 2</caption>
<thead>
	<tr><th></th><th scope=col>Actual.Value</th><th scope=col>Predicted.Values</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>10</th><td>5</td><td>5.644966</td></tr>
	<tr><th scope=row>12</th><td>5</td><td>5.644966</td></tr>
	<tr><th scope=row>13</th><td>5</td><td>5.127552</td></tr>
	<tr><th scope=row>20</th><td>6</td><td>5.394482</td></tr>
	<tr><th scope=row>24</th><td>5</td><td>5.279987</td></tr>
</tbody>
</table>




```python
test2 <- rbind(df_head$Actual.Value,df_head$Predicted.Values)
as = colnames(df_head)
barplot(test2,beside=T,legend.text = as)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Regression/output_12_0.png)


The RMSE, MPE and MAPE for the model are as follows:



```python
# Calculate the errors
library(Metrics)
rmse_lr <- rmse(wine_test$quality,predictions)
mse_lr <- mse(wine_test$quality,predictions)
mae_lr <- mae(wine_test$quality,predictions)

# Print scores
print(paste("Mean Absolute Error: ", round(mae_lr,2)))
print(paste("Mean Square Error: ", round(mse_lr,2)))
print(paste("Root Mean Square Error: ", round(rmse_lr,2)))
```

    [1] "Mean Absolute Error:  0.53"
    [1] "Mean Square Error:  0.47"
    [1] "Root Mean Square Error:  0.68"
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Polynomial Regression
If your data points clearly will not fit a linear regression (a straight line through all data points), it might be ideal for polynomial regression.
Polynomial regression, like linear regression, uses the relationship between the variables x and y to find the best way to draw a line through the data points.
Download Eaxample dataset from - https://media.geeksforgeeks.org/wp-content/uploads/data.csv


```python
# Loading the dataset    ##Why there are only 6 rows in the dataset  
data = read.csv("dataset/data.csv")
head(data)
train = data['Temperature']
test = data['Pressure']

# Fitting the data to quadratic model
quadratic_model <-lm(data$Pressure~ data$Temperature + I(data$Temperature^2))   


# Predict using model
pred_quadratic_model = predict(quadratic_model,test )
df  = data.frame("Actual Value" = test,"Predicted Values" = pred_quadratic_model)
head(df)

summary(quadratic_model)
```


<table>
<caption>A data.frame: 6 × 3</caption>
<thead>
	<tr><th></th><th scope=col>sno</th><th scope=col>Temperature</th><th scope=col>Pressure</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>1</td><td>  0</td><td>0.0002</td></tr>
	<tr><th scope=row>2</th><td>2</td><td> 20</td><td>0.0012</td></tr>
	<tr><th scope=row>3</th><td>3</td><td> 40</td><td>0.0060</td></tr>
	<tr><th scope=row>4</th><td>4</td><td> 60</td><td>0.0300</td></tr>
	<tr><th scope=row>5</th><td>5</td><td> 80</td><td>0.0900</td></tr>
	<tr><th scope=row>6</th><td>6</td><td>100</td><td>0.2700</td></tr>
</tbody>
</table>




<table>
<caption>A data.frame: 6 × 2</caption>
<thead>
	<tr><th></th><th scope=col>Pressure</th><th scope=col>Predicted.Values</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0.0002</td><td> 0.01555</td></tr>
	<tr><th scope=row>2</th><td>0.0012</td><td>-0.01731</td></tr>
	<tr><th scope=row>3</th><td>0.0060</td><td>-0.01032</td></tr>
	<tr><th scope=row>4</th><td>0.0300</td><td> 0.03652</td></tr>
	<tr><th scope=row>5</th><td>0.0900</td><td> 0.12321</td></tr>
	<tr><th scope=row>6</th><td>0.2700</td><td> 0.24975</td></tr>
</tbody>
</table>




    
    Call:
    lm(formula = data$Pressure ~ data$Temperature + I(data$Temperature^2))
    
    Residuals:
           1        2        3        4        5        6 
    -0.01535  0.01851  0.01632 -0.00652 -0.03321  0.02025 
    
    Coefficients:
                            Estimate Std. Error t value Pr(>|t|)  
    (Intercept)            1.555e-02  2.564e-02   0.607   0.5869  
    data$Temperature      -2.639e-03  1.206e-03  -2.189   0.1164  
    I(data$Temperature^2)  4.981e-05  1.157e-05   4.304   0.0231 *
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    Residual standard error: 0.02828 on 3 degrees of freedom
    Multiple R-squared:  0.9568,	Adjusted R-squared:  0.9281 
    F-statistic: 33.26 on 2 and 3 DF,  p-value: 0.008965
    



```python
# Calculate the errors
library(Metrics)    #Don't Load it again
rmse_pr <- rmse(test$Pressure,pred_quadratic_model)
mse_pr <- mse(test$Pressure,pred_quadratic_model)
mae_pr <- mae(test$Pressure,pred_quadratic_model)

# Print scores
print(paste("Mean Absolute Error: ", round(mae_pr,2)))
print(paste("Mean Square Error: ", round(mse_pr,2)))
print(paste("Root Mean Square Error: ", round(rmse_pr,2)))
```

    [1] "Mean Absolute Error:  0.02"
    [1] "Mean Square Error:  0"
    [1] "Root Mean Square Error:  0.02"
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Quantile Regression
Quantile regression is the extension of linear regression and we generally use it when outliers, high skeweness and heteroscedasticity exist in the data.   __This needs to be more elaborative__


```python
# Loading Required libraries
library(quantreg)
options(warn=-1)

# Loading the dataset
engel = read.csv("dataset/engel.csv")
head(engel)

# Fitting the datat to quantile regression
quantile_reg <- rq( foodexp~ income, data = engel)
summary(quantile_reg)
```


<table>
<caption>A data.frame: 6 × 3</caption>
<thead>
	<tr><th></th><th scope=col>X</th><th scope=col>income</th><th scope=col>foodexp</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>0</td><td>420.1577</td><td>255.8394</td></tr>
	<tr><th scope=row>2</th><td>1</td><td>541.4117</td><td>310.9587</td></tr>
	<tr><th scope=row>3</th><td>2</td><td>901.1575</td><td>485.6800</td></tr>
	<tr><th scope=row>4</th><td>3</td><td>639.0802</td><td>402.9974</td></tr>
	<tr><th scope=row>5</th><td>4</td><td>750.8756</td><td>495.5608</td></tr>
	<tr><th scope=row>6</th><td>5</td><td>945.7989</td><td>633.7978</td></tr>
</tbody>
</table>




    
    Call: rq(formula = foodexp ~ income, data = engel)
    
    tau: [1] 0.5
    
    Coefficients:
                coefficients lower bd  upper bd 
    (Intercept)  81.48225     53.25915 114.01156
    income        0.56018      0.48702   0.60199



```python
plot(foodexp ~ income, data = engel, pch = 16, main = "foodexp ~ income")
abline(lm(foodexp ~ income, data = engel), col = "red", lty = 2)
abline(rq(foodexp ~ income, data = engel), col = "blue", lty = 2)
legend("topright", legend = c("lm", "rq"), col = c("red", "blue"), lty = 2)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Regression/output_22_0.png)


We estimate the quantile regression model for many quantiles between .05 and .95, and compare best fit line from each of these models to Ordinary Least Squares results.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Ridge Regression
Ridge Regression is particularly useful to mitigate the problem of multicollinearity in linear regression,
which commonly occurs in models with large numbers of parameters.
Removing predictors from the model can be seen as settings their coefficients to zero.
Instead of forcing them to be exactly zero, let's penalize them if they are too far from zero,
thus enforcing them to be small in a continuous way.
This way, we decrease model complexity while keeping all variables in the model.
It applies L2 Regularization



```python
# Loading required libraries
library(glmnet)

# Setting alpha = 0 implements ridge regression
lambdas_to_try <- 0.01
ridge <- glmnet(as.matrix(wine_train), wine_train$quality, alpha = 0, lambda = lambdas_to_try,
                      standardize = TRUE)
```


```python
# Predict using model
predictions <- predict(ridge, lambdas_to_try, newx = as.matrix(wine_test))
df  = data.frame("Actual Value" = wine_test$quality,"Predicted Values" = predictions)
head(df)
```


<table>
<caption>A data.frame: 6 × 2</caption>
<thead>
	<tr><th></th><th scope=col>Actual.Value</th><th scope=col>X1</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>10</th><td>5</td><td>5.012410</td></tr>
	<tr><th scope=row>12</th><td>5</td><td>5.012410</td></tr>
	<tr><th scope=row>13</th><td>5</td><td>5.002032</td></tr>
	<tr><th scope=row>20</th><td>6</td><td>5.988535</td></tr>
	<tr><th scope=row>24</th><td>5</td><td>5.005399</td></tr>
	<tr><th scope=row>25</th><td>6</td><td>5.991057</td></tr>
</tbody>
</table>




```python
# Calculate the errors
library(Metrics)
rmse_rr <- rmse(wine_test$quality,predictions)
mse_rr <- mse(wine_test$quality,predictions)
mae_rr <- mae(wine_test$quality,predictions)

# Print scores
print(paste("Mean Absolute Error: ", round(mae_rr,2)))
print(paste("Mean Square Error: ", round(mse_rr,2)))
print(paste("Root Mean Square Error: ", round(rmse_rr,2)))
```

    [1] "Mean Absolute Error:  0.72"
    [1] "Mean Square Error:  0.81"
    [1] "Root Mean Square Error:  0.9"
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Lasso Regression
Lasso (least absolute shrinkage and selection operator) is a regression analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the statistical model it produces
Here the lambda value is 0.01


```python
# Loading required libraries
library(glmnet)

# Setting alpha = 0 implements ridge regression
lambdas_to_try <- 0.01
lasso <- glmnet(as.matrix(wine_train), wine_train$quality, alpha = 1, lambda = lambdas_to_try,
                standardize = TRUE)
```


```python
# Predict using model
predictions <- predict(lasso, lambdas_to_try, newx = as.matrix(wine_test))
df  = data.frame("Actual Value" = wine_test$quality,"Predicted Values" = predictions)
head(df)
```


<table>
<caption>A data.frame: 6 × 2</caption>
<thead>
	<tr><th></th><th scope=col>Actual.Value</th><th scope=col>X1</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>10</th><td>5</td><td>5.008063</td></tr>
	<tr><th scope=row>12</th><td>5</td><td>5.008063</td></tr>
	<tr><th scope=row>13</th><td>5</td><td>5.008063</td></tr>
	<tr><th scope=row>20</th><td>6</td><td>5.995557</td></tr>
	<tr><th scope=row>24</th><td>5</td><td>5.008063</td></tr>
	<tr><th scope=row>25</th><td>6</td><td>5.995557</td></tr>
</tbody>
</table>




```python
# Calculate the errors
library(Metrics)
rmse_lasso <- rmse(wine_test$quality,predictions)
mse_lasso <- mse(wine_test$quality,predictions)
mae_lasso <- mae(wine_test$quality,predictions)

# Print scores
print(paste("Mean Absolute Error: ", round(mae_lasso,2)))
print(paste("Mean Square Error: ", round(mse_lasso,2)))
print(paste("Root Mean Square Error: ", round(rmse_lasso,2)))


```

    [1] "Mean Absolute Error:  0.72"
    [1] "Mean Square Error:  0.81"
    [1] "Root Mean Square Error:  0.9"
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Elastic Net Regression
ElasticNet Regression is regularized regression method that linearly combines the L1 and L2 penalties of the lasso and ridge methods.


```python
# install glmnet library 
lambdas_to_try <- 0.03
# Splitting the data into test and train
# Setting alpha = it can be any value between 0 & 1 
elastic_net <- glmnet(as.matrix(wine_train),wine_train$quality, alpha = 0.4, lambda = lambdas_to_try,
                standardize = TRUE)
```


```python
# Predict using elastic net
predictions <- predict(elastic_net, lambdas_to_try, newx = as.matrix(wine_test))
df  = data.frame("Actual Value" = wine_test$quality,"Predicted Values" = predictions)
head(df)
```


<table>
<caption>A data.frame: 6 × 2</caption>
<thead>
	<tr><th></th><th scope=col>Actual.Value</th><th scope=col>X1</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>10</th><td>5</td><td>5.025245</td></tr>
	<tr><th scope=row>12</th><td>5</td><td>5.025245</td></tr>
	<tr><th scope=row>13</th><td>5</td><td>5.023320</td></tr>
	<tr><th scope=row>20</th><td>6</td><td>5.982328</td></tr>
	<tr><th scope=row>24</th><td>5</td><td>5.021716</td></tr>
	<tr><th scope=row>25</th><td>6</td><td>5.983932</td></tr>
</tbody>
</table>




```python
# Calculate the errors
library(Metrics)
rmse_en <- rmse(wine_test$quality,predictions)
mse_en <- mse(wine_test$quality,predictions)
mae_en <- mae(wine_test$quality,predictions)

# Print scores
print(paste("Mean Absolute Error: ", round(mae_en,2)))
print(paste("Mean Square Error: ", round(mse_en,2)))
print(paste("Root Mean Square Error: ", round(rmse_en,2)))

```

    [1] "Mean Absolute Error:  0.72"
    [1] "Mean Square Error:  0.81"
    [1] "Root Mean Square Error:  0.9"
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Support Vector Regression
Support vector regression can solve both linear and non-linear models. SVM uses different kernel functions (such as linear,polynomial,radial basis,sigmoid ) to find the optimal solution for non-linear models.
SVMs assume that the data it works with is in a standard range, usually either 0 to 1, or -1 to 1 (roughly). So the normalization of feature vectors prior to feeding them to the SVM is very important. (This is often called whitening, although there are different types of whitening.) You want to make sure that for each dimension, the values are scaled to lie roughly within this range. Otherwise, if e.g. dimension 1 is from 0-1000 and dimension 2 is from 0-1.2, then dimension 1 becomes much more important than dimension 2, which will skew results.
    
    


```python
# required libraries
library(e1071)
library(caTools)

# Fitting the model
SVM = svm(as.matrix(wine_train), y =wine_train$quality ,kernel = 'linear')  
```


```python
# Predict using the model
predictions = predict(SVM, as.matrix(wine_test))
df  = data.frame("Actual Value" = wine_test$quality,"Predicted Values" = predictions)
head(df)
```


<table>
<caption>A data.frame: 6 × 2</caption>
<thead>
	<tr><th></th><th scope=col>Actual.Value</th><th scope=col>Predicted.Values</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>10</th><td>5</td><td>5.021103</td></tr>
	<tr><th scope=row>12</th><td>5</td><td>5.021103</td></tr>
	<tr><th scope=row>13</th><td>5</td><td>4.995930</td></tr>
	<tr><th scope=row>20</th><td>6</td><td>5.988695</td></tr>
	<tr><th scope=row>24</th><td>5</td><td>5.024304</td></tr>
	<tr><th scope=row>25</th><td>6</td><td>5.965476</td></tr>
</tbody>
</table>




```python
# Calculate the errors
library(Metrics)
rmse_svm <- rmse(wine_test$quality,predictions)
mse_svm <- mse(wine_test$quality,predictions)
mae_svm <- mae(wine_test$quality,predictions)

# Print scores

print(paste("Mean Absolute Error: ", round(mae_svm,2)))
print(paste("Mean Square Error: ", round(mse_svm,2)))
print(paste("Root Mean Square Error: ", round(rmse_svm,2)))
```

    [1] "Mean Absolute Error:  0.72"
    [1] "Mean Square Error:  0.81"
    [1] "Root Mean Square Error:  0.9"
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

# Decision Tree - CART 
Decision tree regression observes features of an object and trains a model in the structure of a tree to predict data in the future to produce meaningful continuous output. Continuous output means that the output/result is not discrete, i.e., it is not represented just by a discrete, known set of numbers or values.

Let’s see the Step-by-Step implementation –


```python
# Required library
library(rpart)

#Creating regressor object and fitting with X and Y data
decision_tree = rpart(quality~., data=wine_test)
summary(decision_tree)
```

    Call:
    rpart(formula = quality ~ ., data = wine_test)
      n= 400 
    
               CP nsplit rel error    xerror       xstd
    1  0.17062842      0 1.0000000 1.0091300 0.07173145
    2  0.09876073      1 0.8293716 0.9495460 0.07538482
    3  0.03659812      2 0.7306108 0.8481914 0.06759793
    4  0.03268095      3 0.6940127 0.8558523 0.06968053
    5  0.02875098      4 0.6613318 0.8593629 0.07029898
    6  0.02320904      5 0.6325808 0.8496688 0.07074195
    7  0.01855050      6 0.6093718 0.8292981 0.06981687
    8  0.01723007      7 0.5908213 0.8236812 0.07035636
    9  0.01393212      8 0.5735912 0.8355560 0.07090740
    10 0.01259060      9 0.5596591 0.8433696 0.07130243
    11 0.01168732     10 0.5470685 0.8591997 0.07266491
    12 0.01158693     12 0.5236938 0.8607596 0.07270697
    13 0.01000000     13 0.5121069 0.8451087 0.07253793
    
    Variable importance
                 alcohol     volatile.acidity              density 
                      24                   16                   12 
           fixed.acidity            sulphates            chlorides 
                      11                    9                    7 
             citric.acid                   pH total.sulfur.dioxide 
                       6                    6                    4 
     free.sulfur.dioxide       residual.sugar 
                       4                    1 
    
    Node number 1: 400 observations,    complexity param=0.1706284
      mean=5.61, MSE=0.6879 
      left son=2 (323 obs) right son=3 (77 obs)
      Primary splits:
          alcohol          < 11.45    to the left,  improve=0.17062840, (0 missing)
          volatile.acidity < 0.365    to the right, improve=0.16332630, (0 missing)
          sulphates        < 0.615    to the left,  improve=0.13243910, (0 missing)
          citric.acid      < 0.305    to the left,  improve=0.08922573, (0 missing)
          density          < 0.995565 to the right, improve=0.04401157, (0 missing)
      Surrogate splits:
          density        < 0.993815 to the right, agree=0.858, adj=0.260, (0 split)
          fixed.acidity  < 5.7      to the right, agree=0.822, adj=0.078, (0 split)
          chlorides      < 0.0555   to the right, agree=0.820, adj=0.065, (0 split)
          pH             < 3.675    to the left,  agree=0.818, adj=0.052, (0 split)
          residual.sugar < 1.25     to the right, agree=0.815, adj=0.039, (0 split)
    
    Node number 2: 323 observations,    complexity param=0.09876073
      mean=5.442724, MSE=0.5687009 
      left son=4 (263 obs) right son=5 (60 obs)
      Primary splits:
          volatile.acidity < 0.375    to the right, improve=0.14793920, (0 missing)
          sulphates        < 0.615    to the left,  improve=0.11869380, (0 missing)
          alcohol          < 9.95     to the left,  improve=0.07540900, (0 missing)
          citric.acid      < 0.305    to the left,  improve=0.04576861, (0 missing)
          chlorides        < 0.0665   to the right, improve=0.04513746, (0 missing)
      Surrogate splits:
          free.sulfur.dioxide < 3.5      to the right, agree=0.820, adj=0.033, (0 split)
          alcohol             < 11.35    to the left,  agree=0.820, adj=0.033, (0 split)
          chlorides           < 0.055    to the right, agree=0.817, adj=0.017, (0 split)
    
    Node number 3: 77 observations,    complexity param=0.03268095
      mean=6.311688, MSE=0.5781751 
      left son=6 (39 obs) right son=7 (38 obs)
      Primary splits:
          fixed.acidity    < 8.25     to the left,  improve=0.2019900, (0 missing)
          pH               < 3.425    to the right, improve=0.2009627, (0 missing)
          citric.acid      < 0.27     to the left,  improve=0.1910942, (0 missing)
          sulphates        < 0.625    to the left,  improve=0.1580385, (0 missing)
          volatile.acidity < 0.495    to the right, improve=0.1549594, (0 missing)
      Surrogate splits:
          density     < 0.995085 to the left,  agree=0.857, adj=0.711, (0 split)
          citric.acid < 0.285    to the left,  agree=0.844, adj=0.684, (0 split)
          sulphates   < 0.635    to the left,  agree=0.831, adj=0.658, (0 split)
          pH          < 3.335    to the right, agree=0.792, adj=0.579, (0 split)
          chlorides   < 0.0705   to the left,  agree=0.740, adj=0.474, (0 split)
    
    Node number 4: 263 observations,    complexity param=0.03659812
      mean=5.304183, MSE=0.447397 
      left son=8 (106 obs) right son=9 (157 obs)
      Primary splits:
          sulphates            < 0.575    to the left,  improve=0.08558454, (0 missing)
          volatile.acidity     < 0.9125   to the right, improve=0.06910727, (0 missing)
          alcohol              < 9.85     to the left,  improve=0.05451070, (0 missing)
          chlorides            < 0.0645   to the right, improve=0.04522456, (0 missing)
          total.sulfur.dioxide < 10.5     to the left,  improve=0.03281568, (0 missing)
      Surrogate splits:
          volatile.acidity     < 0.8      to the right, agree=0.639, adj=0.104, (0 split)
          density              < 0.99479  to the left,  agree=0.631, adj=0.085, (0 split)
          free.sulfur.dioxide  < 5.5      to the left,  agree=0.620, adj=0.057, (0 split)
          total.sulfur.dioxide < 10.5     to the left,  agree=0.616, adj=0.047, (0 split)
          chlorides            < 0.0575   to the left,  agree=0.612, adj=0.038, (0 split)
    
    Node number 5: 60 observations,    complexity param=0.02320904
      mean=6.05, MSE=0.6475 
      left son=10 (34 obs) right son=11 (26 obs)
      Primary splits:
          sulphates            < 0.775    to the left,  improve=0.16438090, (0 missing)
          total.sulfur.dioxide < 49.5     to the right, improve=0.12749480, (0 missing)
          alcohol              < 9.75     to the left,  improve=0.12530170, (0 missing)
          fixed.acidity        < 10.45    to the left,  improve=0.12026310, (0 missing)
          chlorides            < 0.0585   to the left,  improve=0.07877072, (0 missing)
      Surrogate splits:
          volatile.acidity     < 0.265    to the right, agree=0.717, adj=0.346, (0 split)
          chlorides            < 0.0785   to the right, agree=0.683, adj=0.269, (0 split)
          free.sulfur.dioxide  < 4        to the right, agree=0.617, adj=0.115, (0 split)
          alcohol              < 10.15    to the left,  agree=0.617, adj=0.115, (0 split)
          total.sulfur.dioxide < 36.5     to the right, agree=0.600, adj=0.077, (0 split)
    
    Node number 6: 39 observations,    complexity param=0.01723007
      mean=5.974359, MSE=0.486522 
      left son=12 (15 obs) right son=13 (24 obs)
      Primary splits:
          free.sulfur.dioxide  < 8.5      to the left,  improve=0.24986490, (0 missing)
          total.sulfur.dioxide < 79.5     to the left,  improve=0.16028600, (0 missing)
          volatile.acidity     < 0.515    to the right, improve=0.11448190, (0 missing)
          pH                   < 3.425    to the right, improve=0.10823680, (0 missing)
          chlorides            < 0.0495   to the right, improve=0.08514058, (0 missing)
      Surrogate splits:
          total.sulfur.dioxide < 16.5     to the left,  agree=0.897, adj=0.733, (0 split)
          volatile.acidity     < 0.5625   to the right, agree=0.769, adj=0.400, (0 split)
          chlorides            < 0.078    to the right, agree=0.692, adj=0.200, (0 split)
          citric.acid          < 0.005    to the left,  agree=0.667, adj=0.133, (0 split)
          density              < 0.99503  to the right, agree=0.667, adj=0.133, (0 split)
    
    Node number 7: 38 observations,    complexity param=0.0185505
      mean=6.657895, MSE=0.4355956 
      left son=14 (9 obs) right son=15 (29 obs)
      Primary splits:
          fixed.acidity        < 10.55    to the right, improve=0.3083713, (0 missing)
          total.sulfur.dioxide < 43       to the right, improve=0.2556765, (0 missing)
          chlorides            < 0.095    to the right, improve=0.2119820, (0 missing)
          volatile.acidity     < 0.35     to the right, improve=0.1954040, (0 missing)
          free.sulfur.dioxide  < 13.5     to the right, improve=0.1762871, (0 missing)
      Surrogate splits:
          density     < 1.0005   to the right, agree=0.868, adj=0.444, (0 split)
          citric.acid < 0.57     to the right, agree=0.816, adj=0.222, (0 split)
          chlorides   < 0.102    to the right, agree=0.816, adj=0.222, (0 split)
          pH          < 3.1      to the left,  agree=0.816, adj=0.222, (0 split)
          alcohol     < 12.65    to the right, agree=0.816, adj=0.222, (0 split)
    
    Node number 8: 106 observations,    complexity param=0.01393212
      mean=5.066038, MSE=0.3258277 
      left son=16 (27 obs) right son=17 (79 obs)
      Primary splits:
          chlorides            < 0.0875   to the right, improve=0.11099640, (0 missing)
          citric.acid          < 0.005    to the left,  improve=0.11002600, (0 missing)
          pH                   < 3.295    to the right, improve=0.09727543, (0 missing)
          total.sulfur.dioxide < 14.5     to the left,  improve=0.09129099, (0 missing)
          volatile.acidity     < 0.9125   to the right, improve=0.07420684, (0 missing)
      Surrogate splits:
          pH               < 2.935    to the left,  agree=0.764, adj=0.074, (0 split)
          fixed.acidity    < 10.6     to the right, agree=0.755, adj=0.037, (0 split)
          volatile.acidity < 0.395    to the left,  agree=0.755, adj=0.037, (0 split)
          citric.acid      < 0.53     to the right, agree=0.755, adj=0.037, (0 split)
    
    Node number 9: 157 observations,    complexity param=0.02875098
      mean=5.464968, MSE=0.4653333 
      left son=18 (83 obs) right son=19 (74 obs)
      Primary splits:
          alcohol              < 9.95     to the left,  improve=0.10828650, (0 missing)
          total.sulfur.dioxide < 60.5     to the right, improve=0.07977459, (0 missing)
          density              < 0.995155 to the right, improve=0.05895350, (0 missing)
          volatile.acidity     < 0.615    to the right, improve=0.05609015, (0 missing)
          pH                   < 3.565    to the right, improve=0.04666234, (0 missing)
      Surrogate splits:
          density              < 0.99631  to the right, agree=0.656, adj=0.270, (0 split)
          total.sulfur.dioxide < 55.5     to the right, agree=0.643, adj=0.243, (0 split)
          fixed.acidity        < 9.25     to the left,  agree=0.637, adj=0.230, (0 split)
          volatile.acidity     < 0.485    to the right, agree=0.618, adj=0.189, (0 split)
          residual.sugar       < 1.95     to the left,  agree=0.605, adj=0.162, (0 split)
    
    Node number 10: 34 observations
      mean=5.764706, MSE=0.4740484 
    
    Node number 11: 26 observations,    complexity param=0.01158693
      mean=6.423077, MSE=0.6286982 
      left son=22 (19 obs) right son=23 (7 obs)
      Primary splits:
          fixed.acidity        < 10.45    to the left,  improve=0.19504640, (0 missing)
          total.sulfur.dioxide < 45.5     to the right, improve=0.18768690, (0 missing)
          free.sulfur.dioxide  < 16.5     to the right, improve=0.12653590, (0 missing)
          citric.acid          < 0.435    to the left,  improve=0.06789305, (0 missing)
          chlorides            < 0.0695   to the left,  improve=0.06789305, (0 missing)
      Surrogate splits:
          density     < 0.9975   to the left,  agree=0.846, adj=0.429, (0 split)
          pH          < 3.19     to the right, agree=0.808, adj=0.286, (0 split)
          citric.acid < 0.52     to the left,  agree=0.769, adj=0.143, (0 split)
    
    Node number 12: 15 observations
      mean=5.533333, MSE=0.6488889 
    
    Node number 13: 24 observations
      mean=6.25, MSE=0.1875 
    
    Node number 14: 9 observations
      mean=6, MSE=0.4444444 
    
    Node number 15: 29 observations
      mean=6.862069, MSE=0.2568371 
    
    Node number 16: 27 observations
      mean=4.740741, MSE=0.266118 
    
    Node number 17: 79 observations,    complexity param=0.0125906
      mean=5.177215, MSE=0.2977087 
      left son=34 (41 obs) right son=35 (38 obs)
      Primary splits:
          pH                   < 3.295    to the right, improve=0.14730350, (0 missing)
          citric.acid          < 0.015    to the left,  improve=0.11984320, (0 missing)
          volatile.acidity     < 0.6075   to the right, improve=0.08391232, (0 missing)
          fixed.acidity        < 6.75     to the left,  improve=0.06998479, (0 missing)
          total.sulfur.dioxide < 27       to the left,  improve=0.05878428, (0 missing)
      Surrogate splits:
          fixed.acidity       < 7.75     to the left,  agree=0.785, adj=0.553, (0 split)
          citric.acid         < 0.195    to the left,  agree=0.722, adj=0.421, (0 split)
          volatile.acidity    < 0.6025   to the right, agree=0.658, adj=0.289, (0 split)
          alcohol             < 9.55     to the right, agree=0.646, adj=0.263, (0 split)
          free.sulfur.dioxide < 13.5     to the right, agree=0.608, adj=0.184, (0 split)
    
    Node number 18: 83 observations
      mean=5.253012, MSE=0.3335753 
    
    Node number 19: 74 observations,    complexity param=0.01168732
      mean=5.702703, MSE=0.5062089 
      left son=38 (24 obs) right son=39 (50 obs)
      Primary splits:
          volatile.acidity     < 0.62     to the right, improve=0.07758057, (0 missing)
          total.sulfur.dioxide < 68.5     to the right, improve=0.06468892, (0 missing)
          density              < 0.9995   to the right, improve=0.06468892, (0 missing)
          citric.acid          < 0.46     to the right, improve=0.05216528, (0 missing)
          residual.sugar       < 3.3      to the right, improve=0.05216528, (0 missing)
      Surrogate splits:
          free.sulfur.dioxide  < 11.5     to the left,  agree=0.770, adj=0.292, (0 split)
          density              < 1.00065  to the right, agree=0.730, adj=0.167, (0 split)
          total.sulfur.dioxide < 13.5     to the left,  agree=0.716, adj=0.125, (0 split)
          fixed.acidity        < 14.05    to the right, agree=0.703, adj=0.083, (0 split)
          chlorides            < 0.1025   to the right, agree=0.703, adj=0.083, (0 split)
    
    Node number 22: 19 observations
      mean=6.210526, MSE=0.5872576 
    
    Node number 23: 7 observations
      mean=7, MSE=0.2857143 
    
    Node number 34: 41 observations
      mean=4.97561, MSE=0.2189173 
    
    Node number 35: 38 observations
      mean=5.394737, MSE=0.2915512 
    
    Node number 38: 24 observations,    complexity param=0.01168732
      mean=5.416667, MSE=0.5763889 
      left son=76 (11 obs) right son=77 (13 obs)
      Primary splits:
          citric.acid          < 0.13     to the right, improve=0.25486560, (0 missing)
          total.sulfur.dioxide < 22.5     to the right, improve=0.18209980, (0 missing)
          fixed.acidity        < 8.75     to the right, improve=0.18072290, (0 missing)
          sulphates            < 0.645    to the left,  improve=0.10843370, (0 missing)
          density              < 0.99651  to the right, improve=0.09948365, (0 missing)
      Surrogate splits:
          fixed.acidity        < 8.75     to the right, agree=0.833, adj=0.636, (0 split)
          density              < 0.99691  to the right, agree=0.833, adj=0.636, (0 split)
          chlorides            < 0.0825   to the right, agree=0.792, adj=0.545, (0 split)
          total.sulfur.dioxide < 22.5     to the right, agree=0.792, adj=0.545, (0 split)
          volatile.acidity     < 0.685    to the left,  agree=0.708, adj=0.364, (0 split)
    
    Node number 39: 50 observations
      mean=5.84, MSE=0.4144 
    
    Node number 76: 11 observations
      mean=5, MSE=0.3636364 
    
    Node number 77: 13 observations
      mean=5.769231, MSE=0.4852071 
    
    


```python
#We can plot and see the decision tree as below.
library(rpart.plot)

rpart.plot(decision_tree)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Regression/output_47_0.png)



```python
# Predicting using the model
predictions <- predict(decision_tree,newdata=wine_test)[1:N_test]   
```


```python
# Calculate the errors
library(Metrics)
rmse_dt <- rmse(wine_test$quality,predictions)
mse_dt <- mse(wine_test$quality,predictions)
mae_dt <- mae(wine_test$quality,predictions)

# Print scores
print(paste("Mean Absolute Error: ", round(mae_dt,2)))
print(paste("Mean Square Error: ", round(mse_dt,2)))
print(paste("Root Mean Square Error: ", round(rmse_dt,2)))
```

    [1] "Mean Absolute Error:  0.79"
    [1] "Mean Square Error:  0.98"
    [1] "Root Mean Square Error:  0.99"
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Random Forest Regression
Random Forest is a collection of decision trees and average/majority vote of the forest is selected as the predicted output.

We import the random forest regression model from skicit-learn, instantiate the model, and fit (scikit-learn’s name for training) the model on the training data.


```python
# Required libraries
library(randomForest)

# Fitting the model
rf <- randomForest(quality ~ ., data = wine_train, ntree=100)   
summary(rf)
```


                    Length Class  Mode     
    call               4   -none- call     
    type               1   -none- character
    predicted       1199   -none- numeric  
    mse              100   -none- numeric  
    rsq              100   -none- numeric  
    oob.times       1199   -none- numeric  
    importance        11   -none- numeric  
    importanceSD       0   -none- NULL     
    localImportance    0   -none- NULL     
    proximity          0   -none- NULL     
    ntree              1   -none- numeric  
    mtry               1   -none- numeric  
    forest            11   -none- list     
    coefs              0   -none- NULL     
    y               1199   -none- numeric  
    test               0   -none- NULL     
    inbag              0   -none- NULL     
    terms              3   terms  call     



```python
# Predicting using the model
predictions <- predict(rf,newdata=wine_test)[1:N_test]
```


```python
# Calculate the errors
library(Metrics)
rmse_rf <- rmse(wine_test$quality,predictions)
mse_rf <- mse(wine_test$quality,predictions)
mae_rf <- mae(wine_test$quality,predictions)

# Print scores
print(paste("Mean Absolute Error: ", round(mae_rf,2)))
print(paste("Mean Square Error: ", round(mse_rf,2)))
print(paste("Root Mean Square Error: ", round(rmse_rf,2)))
```

    [1] "Mean Absolute Error:  0.72"
    [1] "Mean Square Error:  0.81"
    [1] "Root Mean Square Error:  0.9"
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>
<a id='1'></a>
## Gradient Boosting Machine

In gradient boosting, the ensemble model we try to build is also a weighted sum of weak learners

Boosting is a sequential technique which works on the principle of ensemble. It combines a set of weak learners and delivers improved prediction accuracy. At any instant t, the model outcomes are weighed based on the outcomes of previous instant t-1. The outcomes predicted correctly are given a lower weight and the ones miss-classified are weighted higher. This technique is followed for a classification problem while a similar technique is used for regression.


```python
# Required libraries
library(gbm)

# Fitting the model
gb <- gbm(quality ~ ., data = wine_train, distribution = "gaussian", shrinkage = 0.01, interaction.depth = 4)
summary(gb)
```


<table>
<caption>A data.frame: 11 × 2</caption>
<thead>
	<tr><th></th><th scope=col>var</th><th scope=col>rel.inf</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>alcohol</th><td>alcohol             </td><td>54.3732559</td></tr>
	<tr><th scope=row>sulphates</th><td>sulphates           </td><td>20.1286166</td></tr>
	<tr><th scope=row>volatile.acidity</th><td>volatile.acidity    </td><td>16.6958670</td></tr>
	<tr><th scope=row>total.sulfur.dioxide</th><td>total.sulfur.dioxide</td><td> 3.5694214</td></tr>
	<tr><th scope=row>chlorides</th><td>chlorides           </td><td> 1.5009597</td></tr>
	<tr><th scope=row>pH</th><td>pH                  </td><td> 1.1958543</td></tr>
	<tr><th scope=row>free.sulfur.dioxide</th><td>free.sulfur.dioxide </td><td> 0.5935582</td></tr>
	<tr><th scope=row>fixed.acidity</th><td>fixed.acidity       </td><td> 0.5384384</td></tr>
	<tr><th scope=row>density</th><td>density             </td><td> 0.5186587</td></tr>
	<tr><th scope=row>citric.acid</th><td>citric.acid         </td><td> 0.4845031</td></tr>
	<tr><th scope=row>residual.sugar</th><td>residual.sugar      </td><td> 0.4008667</td></tr>
</tbody>
</table>




![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Regression/output_57_1.png)



```python
#Predict using model
predictions <- predict.gbm(gb, wine_test, n.trees = 100)
```


```python
# Calculate the errors
library(Metrics)
rmse_gbm <- rmse(wine_test$quality,predictions)
mse_gbm <- mse(wine_test$quality,predictions)
mae_gbm <- mae(wine_test$quality,predictions)

# Print scores
print(paste("Mean Absolute Error: ", round(mae_gbm,2)))
print(paste("Mean Square Error: ", round(mse_gbm,2)))
print(paste("Root Mean Square Error: ", round(rmse_gbm,2)))
```

    [1] "Mean Absolute Error:  0.72"
    [1] "Mean Square Error:  0.81"
    [1] "Root Mean Square Error:  0.9"
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Gradient Descent (GBM)

Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.


```python
# Loading required libraries
library(gradDescent)

#winedata is scaled and then split
featureScalingResult <- varianceScaling(winedata)
splitedDataset <- splitData(featureScalingResult$scaledDataSet)

# # Fit the train from splitted dataset data to GD model
model <- GD(splitedDataset$dataTrain)
```


```python
# Test data input
dataTestInput <- (splitedDataset$dataTest)[,1:(ncol(splitedDataset$dataTest)-1)]

# Predict using model
predictions <- prediction(model, dataTestInput)
```


```python
# Calculating errors
library(Metrics)
rmse_gd <- rmse(splitedDataset$dataTest[,ncol(splitedDataset$dataTest)],predictions[,ncol(predictions)])
mae_gd <- mae(splitedDataset$dataTest[,ncol(splitedDataset$dataTest)],predictions[,ncol(predictions)])
mse_gd <- mse(splitedDataset$dataTest[,ncol(splitedDataset$dataTest)],predictions[,ncol(predictions)])

# Print scores
print(paste("Mean Absolute Error: ", mae_gd))
print(paste("Mean Square Error: ", mse_gd))
print(paste("Root Mean Square Error: ", rmse_gd))
```

    [1] "Mean Absolute Error:  0.791272754541309"
    [1] "Mean Square Error:  1.1116292980384"
    [1] "Root Mean Square Error:  1.05433832237968"
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Stochastic Gradient Descent

Stochastic Gradient Descent (SGD) regressor basically implements a plain SGD learning routine supporting various loss functions and penalties to fit linear regression models.


```python
# Fitting the model
SGDmodel <- SGD(splitedDataset$dataTrain)

#show result
print(SGDmodel)
```

              [,1]      [,2]       [,3]      [,4]        [,5]      [,6]       [,7]
    [1,] 0.3876573 0.4291433 -0.2580326 0.9828201 0.001271909 0.7428889 -0.2227534
              [,8]      [,9]       [,10]     [,11]     [,12]
    [1,] 0.4159231 0.2981105 -0.02304137 0.2418506 0.5050303
    


```python
# Test data input
dataTestInput <- (splitedDataset$dataTest)[,1:ncol(splitedDataset$dataTest)-1]

# Predict using model
predictions <- prediction(model, dataTestInput)
```


```python
# Calculating errors
library(Metrics)
rmse_sgd <- rmse(splitedDataset$dataTest[,ncol(splitedDataset$dataTest)],predictions[,ncol(predictions)])
mae_sgd <- mae(splitedDataset$dataTest[,ncol(splitedDataset$dataTest)],predictions[,ncol(predictions)])
mse_sgd <- mse(splitedDataset$dataTest[,ncol(splitedDataset$dataTest)],predictions[,ncol(predictions)])

# Print scores
print(paste("Mean Absolute Error: ", round(mae_sgd,2)))
print(paste("Mean Square Error: ", round(mse_sgd,2)))
print(paste("Root Mean Square Error: ", round(rmse_sgd,2)))
```

    [1] "Mean Absolute Error:  0.79"
    [1] "Mean Square Error:  1.11"
    [1] "Root Mean Square Error:  1.05"
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

## KNN Regressor

Regression based on k-nearest neighbors.
The target is predicted by local interpolation of the targets associated of the nearest neighbors in the training set.
A simple implementation of KNN regression is to calculate the average of the numerical target of the K nearest neighbors.


```python
# Remove the null values if any
wine_train <- na.omit(wine_train)
wine_test <- na.omit(wine_test)

#Loading required libraries
library(class)

# Fitting the model
knn_model <- knn(train=wine_train, test=wine_test, cl=wine_train$quality, k=26)
```


```python
# Accuracy of the model
accuracy <- 100 * sum(wine_test['quality'][1:N_test,] ==
                           knn_model)/NROW(wine_test['quality'])
accuracy
```


48.25


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

# XGB Regressor


```python
# We need to pass matrix as input for xgb, hence we do the train test split again and define X and Y
data = read.csv('dataset/winequality-red.csv')
alpha=0.8
d = sort(sample(nrow(data), nrow(data)*alpha))
train = data[d,]
test = data[-d,]
X_train = as.matrix(train[,1:11])
X_test = as.matrix(test[,1:11])
y_train = train$quality
y_test = test$quality
```


```python
# Load required libraries
library(xgboost)

#Prepare the data
dtrain = xgb.DMatrix(data = X_train, label = y_train)
dtest = xgb.DMatrix(data = X_test, label = y_test)
```


```python
#Fit the model
model_xgb <- xgboost(data = dtrain, # the data   
                 nround = 15, # max number of boosting iterations
                 max_depth = 3)
```

    [1]	train-rmse:3.675007 
    [2]	train-rmse:2.617607 
    [3]	train-rmse:1.889074 
    [4]	train-rmse:1.395113 
    [5]	train-rmse:1.067960 
    [6]	train-rmse:0.858540 
    [7]	train-rmse:0.732016 
    [8]	train-rmse:0.657595 
    [9]	train-rmse:0.612948 
    [10]	train-rmse:0.586373 
    [11]	train-rmse:0.569531 
    [12]	train-rmse:0.559085 
    [13]	train-rmse:0.551950 
    [14]	train-rmse:0.545688 
    [15]	train-rmse:0.542357 
    


```python
# Make predictions
predictions = predict(model_xgb, dtest)
```


```python
# Calculating errors
library(Metrics)
rmse_xgb <- rmse(y_test, predictions)
mae_xgb <- mae(y_test, predictions)
mse_xgb <- mse(y_test, predictions)

# Print scores
print(paste("Mean Absolute Error: ", round(mae_xgb,2)))
print(paste("Mean Square Error: ", round(mse_xgb,2)))
print(paste("Root Mean Square Error: ", round(rmse_xgb,2)))
```

    [1] "Mean Absolute Error:  0.48"
    [1] "Mean Square Error:  0.39"
    [1] "Root Mean Square Error:  0.62"
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

# Regressors Report


```python
Algorithm = c('Multiple Linear Regression','Ridge Regularisation','Lasso Reguralisation',
              'Elastic net','Support Vector Regression','Decision Tree','Gradient Descent','Stochastic Gradient Descent','Random Forest', 'XGBoost')

Mean_Absolute_Error = c(mae_lr,mae_rr,mae_lasso,mae_en,mae_svm, mae_dt, mae_gd,mae_sgd,mae_rf, mae_xgb)
                      
Mean_Squared_Error = c(mse_lr,mse_rr, mse_lasso,mse_en,mse_svm, mse_dt,mae_gd,mae_sgd, mse_rf, mse_xgb)         

Root_Mean_Squared_Error = c(rmse_lr,rmse_rr, rmse_lasso,rmse_en,rmse_svm, rmse_gd,mae_sgd,mae_dt, rmse_rf, rmse_xgb)          

report = data.frame(Algorithm,Mean_Absolute_Error,Mean_Squared_Error,Root_Mean_Squared_Error)
report

```


<table>
<caption>A data.frame: 10 × 4</caption>
<thead>
	<tr><th scope=col>Algorithm</th><th scope=col>Mean_Absolute_Error</th><th scope=col>Mean_Squared_Error</th><th scope=col>Root_Mean_Squared_Error</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>Multiple Linear Regression </td><td>0.533237724</td><td>0.4663782949</td><td>0.68291895</td></tr>
	<tr><td>Ridge Regularisation       </td><td>0.010343730</td><td>0.0001743837</td><td>0.01320544</td></tr>
	<tr><td>Lasso Reguralisation       </td><td>0.008824855</td><td>0.0001077731</td><td>0.01038138</td></tr>
	<tr><td>Elastic net                </td><td>0.025867776</td><td>0.0009522315</td><td>0.03085825</td></tr>
	<tr><td>Support Vector Regression  </td><td>0.023676095</td><td>0.0008768431</td><td>0.02961154</td></tr>
	<tr><td>Decision Tree              </td><td>0.793092692</td><td>0.9754048259</td><td>1.00940175</td></tr>
	<tr><td>Gradient Descent           </td><td>0.770021312</td><td>0.7700213120</td><td>0.77002131</td></tr>
	<tr><td>Stochastic Gradient Descent</td><td>0.770021312</td><td>0.7700213120</td><td>0.79309269</td></tr>
	<tr><td>Random Forest              </td><td>0.719360000</td><td>0.8136320028</td><td>0.90201552</td></tr>
	<tr><td>XGBoost                    </td><td>0.493619740</td><td>0.4040405121</td><td>0.63564181</td></tr>
</tbody>
</table>




```python
library(ggplot2)
options(repr.plot.width=10, repr.plot.height=5)
ggplot(data=report, aes(x=Algorithm, y=Mean_Absolute_Error))+geom_bar(position="dodge",stat="identity",width=0.7,color="yellowgreen",fill="yellowgreen") +
  coord_flip() +theme(axis.text.x = element_text(face="bold", color="black",
                                                 size=10, angle=0),
                      axis.text.y = element_text(face="bold", color="black",
                                                 size=10, angle=0))
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Regression/output_84_0.png)



```python
library(ggplot2)
options(repr.plot.width=10, repr.plot.height=5)
ggplot(data=report, aes(x=Algorithm, y=Mean_Squared_Error))+geom_bar(position="dodge",stat="identity",width=0.7,color="lightblue",fill="lightskyblue") +
  coord_flip() +theme(axis.text.x = element_text(face="bold", color="black",
                                                 size=7, angle=0),
                      axis.text.y = element_text(face="bold", color="black",
                                                 size=7, angle=0))
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Regression/output_85_0.png)



```python
library(ggplot2)
options(repr.plot.width=10, repr.plot.height=5)
ggplot(data=report, aes(x=Algorithm, y=Root_Mean_Squared_Error))+geom_bar(position="dodge",stat="identity",width=0.7,color="gold",fill="gold") +
  coord_flip() +theme(axis.text.x = element_text(face="bold", color="black",
                                                 size=8, angle=0),
                      axis.text.y = element_text(face="bold", color="black",
                                                 size=8, angle=0))
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Regression/output_86_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

# Grid Search
Grid-searching is the process of scanning the data to configure optimal parameters for a given model. Depending on the type of model utilized, certain parameters are necessary. Grid-searching does NOT only apply to one model type. Grid-searching can be applied across machine learning to calculate the best parameters to use for any given model.

It is important to note that Grid-searching can be extremely computationally expensive and may take your machine quite a long time to run. Grid-Search will build a model on each parameter combination possible. It iterates through every parameter combination and stores a model for each combination.

**Note:** Here Grid search is demonstrated for only one model but it can be replicated across all the model with changing its respective hyperparameters


**Cross Validation**<br>
The technique of cross validation (CV) is best explained by example using the most common method, K-Fold CV. When we approach a machine learning problem, we make sure to split our data into a training and a testing set. In K-Fold CV, we further split our training set into K number of subsets, called folds. We then iteratively fit the model K times, each time training the data on K-1 of the folds and evaluating on the Kth fold (called the validation data). As an example, consider fitting a model with K = 5. The first iteration we train on the first four folds and evaluate on the fifth. The second time we train on the first, second, third, and fifth fold and evaluate on the fourth. We repeat this procedure 3 more times, each time evaluating on a different fold. At the very end of training, we average the performance on each of the folds to come up with final validation metrics for the model.

![CV.PNG](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Regression/CV.PNG)

For hyperparameter tuning, we perform many iterations of the entire K-Fold CV process, each time using different model settings. We then compare all of the models, select the best one, train it on the full training set, and then evaluate on the testing set. This sounds like an awfully tedious process! Each time we want to assess a different set of hyperparameters, we have to split our training data into K fold and train and evaluate K times. If we have 10 sets of hyperparameters and are using 5-Fold CV, that represents 50 training loops.

Usually, we only have a vague idea of the best hyperparameters and thus the best approach to narrow our search is to evaluate a wide range of values for each hyperparameter


```python
# Loading required libraries
library(randomForest)
library(mlbench)
library(caret)
```


```python
# Parameter grid to tune
gbmGrid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                        n.trees = (1:30)*50, 
                        shrinkage = 0.1,
                        n.minobsinnode = 20)
                        
nrow(gbmGrid)
```


90


### Grid Search Cross Validation


```python
# Hyperparameter tuning using Grid Search
control <- trainControl(method="repeatedcv", number=3, repeats=2, search="grid")
seed <- 7
set.seed(seed)
metric <- "RMSE"
set.seed(825)
gbm_grid <- train(quality ~ ., data = wine_train, 
                 method = "gbm", 
                 trControl = control, 
                 verbose = FALSE, 
                 ## Now specify the exact models 
                 ## to evaluate:
                 tuneGrid = gbmGrid)
gbm_grid
```


    Stochastic Gradient Boosting 
    
    1199 samples
      11 predictor
    
    No pre-processing
    Resampling: Cross-Validated (3 fold, repeated 2 times) 
    Summary of sample sizes: 798, 801, 799, 799, 799, 800, ... 
    Resampling results across tuning parameters:
    
      interaction.depth  n.trees  RMSE       Rsquared   MAE      
      1                    50     0.6398199  0.3677149  0.5023971
      1                   100     0.6313090  0.3784887  0.4902834
      1                   150     0.6315701  0.3783206  0.4888666
      1                   200     0.6335229  0.3755945  0.4905652
      1                   250     0.6340230  0.3744474  0.4897549
      1                   300     0.6345593  0.3736941  0.4907706
      1                   350     0.6347825  0.3734800  0.4915388
      1                   400     0.6342240  0.3746369  0.4908707
      1                   450     0.6350995  0.3733186  0.4921503
      1                   500     0.6363861  0.3730019  0.4928628
      1                   550     0.6362754  0.3732484  0.4922965
      1                   600     0.6370490  0.3721252  0.4931196
      1                   650     0.6375340  0.3710958  0.4935871
      1                   700     0.6373383  0.3717976  0.4937793
      1                   750     0.6378789  0.3712591  0.4940874
      1                   800     0.6378544  0.3712167  0.4933727
      1                   850     0.6406303  0.3659938  0.4953157
      1                   900     0.6410201  0.3656608  0.4962356
      1                   950     0.6423801  0.3639685  0.4970852
      1                  1000     0.6431042  0.3636590  0.4979074
      1                  1050     0.6426346  0.3650525  0.4975443
      1                  1100     0.6443564  0.3626413  0.4991835
      1                  1150     0.6437946  0.3637202  0.4992216
      1                  1200     0.6455398  0.3613819  0.5009525
      1                  1250     0.6469191  0.3595019  0.5018842
      1                  1300     0.6477665  0.3571774  0.5022827
      1                  1350     0.6482648  0.3569405  0.5019665
      1                  1400     0.6497061  0.3547192  0.5037904
      1                  1450     0.6494711  0.3558652  0.5045945
      1                  1500     0.6491065  0.3563922  0.5040236
      5                    50     0.6208462  0.3996792  0.4823816
      5                   100     0.6225939  0.3995318  0.4806193
      5                   150     0.6215471  0.4039594  0.4785687
      5                   200     0.6218804  0.4051120  0.4790988
      5                   250     0.6214770  0.4076626  0.4777257
      5                   300     0.6209485  0.4102208  0.4759345
      5                   350     0.6205321  0.4133582  0.4738314
      5                   400     0.6222900  0.4113941  0.4735160
      5                   450     0.6244427  0.4094497  0.4734310
      5                   500     0.6247562  0.4100898  0.4713431
      5                   550     0.6264550  0.4093058  0.4710840
      5                   600     0.6280525  0.4074841  0.4715375
      5                   650     0.6283743  0.4080280  0.4709649
      5                   700     0.6296379  0.4071897  0.4699773
      5                   750     0.6295009  0.4092589  0.4693879
      5                   800     0.6307583  0.4085343  0.4691149
      5                   850     0.6309236  0.4083776  0.4680417
      5                   900     0.6328242  0.4067492  0.4684635
      5                   950     0.6340063  0.4054208  0.4697520
      5                  1000     0.6349283  0.4050569  0.4697170
      5                  1050     0.6351341  0.4047305  0.4695550
      5                  1100     0.6355741  0.4043568  0.4688364
      5                  1150     0.6360909  0.4041481  0.4682794
      5                  1200     0.6363789  0.4043272  0.4679155
      5                  1250     0.6377561  0.4029695  0.4684625
      5                  1300     0.6378954  0.4027062  0.4681118
      5                  1350     0.6383957  0.4024368  0.4679193
      5                  1400     0.6388773  0.4023778  0.4677124
      5                  1450     0.6394889  0.4017215  0.4679233
      5                  1500     0.6401776  0.4013870  0.4679169
      9                    50     0.6089592  0.4218985  0.4680416
      9                   100     0.6081869  0.4272941  0.4666285
      9                   150     0.6061936  0.4330503  0.4636346
      9                   200     0.6051265  0.4371483  0.4598212
      9                   250     0.6068725  0.4359282  0.4588141
      9                   300     0.6087091  0.4340360  0.4572849
      9                   350     0.6087263  0.4355861  0.4548237
      9                   400     0.6108615  0.4342395  0.4555246
      9                   450     0.6111084  0.4347526  0.4539560
      9                   500     0.6131540  0.4325035  0.4542557
      9                   550     0.6150818  0.4304779  0.4531884
      9                   600     0.6163628  0.4297215  0.4529389
      9                   650     0.6179298  0.4281280  0.4521389
      9                   700     0.6181427  0.4285874  0.4511694
      9                   750     0.6194419  0.4274045  0.4513299
      9                   800     0.6206392  0.4262361  0.4516407
      9                   850     0.6211409  0.4258003  0.4510831
      9                   900     0.6215097  0.4256512  0.4505073
      9                   950     0.6220697  0.4254552  0.4505510
      9                  1000     0.6222686  0.4252384  0.4499312
      9                  1050     0.6227233  0.4251909  0.4497423
      9                  1100     0.6229171  0.4251413  0.4493469
      9                  1150     0.6231433  0.4250974  0.4490526
      9                  1200     0.6235408  0.4247370  0.4490544
      9                  1250     0.6240324  0.4243917  0.4490192
      9                  1300     0.6241654  0.4243940  0.4488486
      9                  1350     0.6241988  0.4244947  0.4485010
      9                  1400     0.6244816  0.4243189  0.4484478
      9                  1450     0.6247304  0.4240163  0.4483704
      9                  1500     0.6247797  0.4241100  0.4480581
    
    Tuning parameter 'shrinkage' was held constant at a value of 0.1
    
    Tuning parameter 'n.minobsinnode' was held constant at a value of 20
    RMSE was used to select the optimal model using the smallest value.
    The final values used for the model were n.trees = 200, interaction.depth =
     9, shrinkage = 0.1 and n.minobsinnode = 20.


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

### Random Search Cross Validation
In contrast to GridSearchCV, not all parameter values are tried out, but rather a fixed number of parameter settings is sampled from the specified distributions.


```python
# Hyperparameter tuning using Random Search
control <- trainControl(method="repeatedcv", number=3, repeats=2, search="random")
seed <- 7
set.seed(seed)
metric <- "RMSE"
set.seed(825)
gbm_random <- train(quality ~ ., data = wine_train, 
                 method = "gbm", 
                 trControl = control, 
                 verbose = FALSE, 
                 tuneGrid = gbmGrid)
gbm_random
```


    Stochastic Gradient Boosting 
    
    1199 samples
      11 predictor
    
    No pre-processing
    Resampling: Cross-Validated (3 fold, repeated 2 times) 
    Summary of sample sizes: 798, 801, 799, 799, 799, 800, ... 
    Resampling results across tuning parameters:
    
      interaction.depth  n.trees  RMSE       Rsquared   MAE      
      1                    50     0.6398199  0.3677149  0.5023971
      1                   100     0.6313090  0.3784887  0.4902834
      1                   150     0.6315701  0.3783206  0.4888666
      1                   200     0.6335229  0.3755945  0.4905652
      1                   250     0.6340230  0.3744474  0.4897549
      1                   300     0.6345593  0.3736941  0.4907706
      1                   350     0.6347825  0.3734800  0.4915388
      1                   400     0.6342240  0.3746369  0.4908707
      1                   450     0.6350995  0.3733186  0.4921503
      1                   500     0.6363861  0.3730019  0.4928628
      1                   550     0.6362754  0.3732484  0.4922965
      1                   600     0.6370490  0.3721252  0.4931196
      1                   650     0.6375340  0.3710958  0.4935871
      1                   700     0.6373383  0.3717976  0.4937793
      1                   750     0.6378789  0.3712591  0.4940874
      1                   800     0.6378544  0.3712167  0.4933727
      1                   850     0.6406303  0.3659938  0.4953157
      1                   900     0.6410201  0.3656608  0.4962356
      1                   950     0.6423801  0.3639685  0.4970852
      1                  1000     0.6431042  0.3636590  0.4979074
      1                  1050     0.6426346  0.3650525  0.4975443
      1                  1100     0.6443564  0.3626413  0.4991835
      1                  1150     0.6437946  0.3637202  0.4992216
      1                  1200     0.6455398  0.3613819  0.5009525
      1                  1250     0.6469191  0.3595019  0.5018842
      1                  1300     0.6477665  0.3571774  0.5022827
      1                  1350     0.6482648  0.3569405  0.5019665
      1                  1400     0.6497061  0.3547192  0.5037904
      1                  1450     0.6494711  0.3558652  0.5045945
      1                  1500     0.6491065  0.3563922  0.5040236
      5                    50     0.6208462  0.3996792  0.4823816
      5                   100     0.6225939  0.3995318  0.4806193
      5                   150     0.6215471  0.4039594  0.4785687
      5                   200     0.6218804  0.4051120  0.4790988
      5                   250     0.6214770  0.4076626  0.4777257
      5                   300     0.6209485  0.4102208  0.4759345
      5                   350     0.6205321  0.4133582  0.4738314
      5                   400     0.6222900  0.4113941  0.4735160
      5                   450     0.6244427  0.4094497  0.4734310
      5                   500     0.6247562  0.4100898  0.4713431
      5                   550     0.6264550  0.4093058  0.4710840
      5                   600     0.6280525  0.4074841  0.4715375
      5                   650     0.6283743  0.4080280  0.4709649
      5                   700     0.6296379  0.4071897  0.4699773
      5                   750     0.6295009  0.4092589  0.4693879
      5                   800     0.6307583  0.4085343  0.4691149
      5                   850     0.6309236  0.4083776  0.4680417
      5                   900     0.6328242  0.4067492  0.4684635
      5                   950     0.6340063  0.4054208  0.4697520
      5                  1000     0.6349283  0.4050569  0.4697170
      5                  1050     0.6351341  0.4047305  0.4695550
      5                  1100     0.6355741  0.4043568  0.4688364
      5                  1150     0.6360909  0.4041481  0.4682794
      5                  1200     0.6363789  0.4043272  0.4679155
      5                  1250     0.6377561  0.4029695  0.4684625
      5                  1300     0.6378954  0.4027062  0.4681118
      5                  1350     0.6383957  0.4024368  0.4679193
      5                  1400     0.6388773  0.4023778  0.4677124
      5                  1450     0.6394889  0.4017215  0.4679233
      5                  1500     0.6401776  0.4013870  0.4679169
      9                    50     0.6089592  0.4218985  0.4680416
      9                   100     0.6081869  0.4272941  0.4666285
      9                   150     0.6061936  0.4330503  0.4636346
      9                   200     0.6051265  0.4371483  0.4598212
      9                   250     0.6068725  0.4359282  0.4588141
      9                   300     0.6087091  0.4340360  0.4572849
      9                   350     0.6087263  0.4355861  0.4548237
      9                   400     0.6108615  0.4342395  0.4555246
      9                   450     0.6111084  0.4347526  0.4539560
      9                   500     0.6131540  0.4325035  0.4542557
      9                   550     0.6150818  0.4304779  0.4531884
      9                   600     0.6163628  0.4297215  0.4529389
      9                   650     0.6179298  0.4281280  0.4521389
      9                   700     0.6181427  0.4285874  0.4511694
      9                   750     0.6194419  0.4274045  0.4513299
      9                   800     0.6206392  0.4262361  0.4516407
      9                   850     0.6211409  0.4258003  0.4510831
      9                   900     0.6215097  0.4256512  0.4505073
      9                   950     0.6220697  0.4254552  0.4505510
      9                  1000     0.6222686  0.4252384  0.4499312
      9                  1050     0.6227233  0.4251909  0.4497423
      9                  1100     0.6229171  0.4251413  0.4493469
      9                  1150     0.6231433  0.4250974  0.4490526
      9                  1200     0.6235408  0.4247370  0.4490544
      9                  1250     0.6240324  0.4243917  0.4490192
      9                  1300     0.6241654  0.4243940  0.4488486
      9                  1350     0.6241988  0.4244947  0.4485010
      9                  1400     0.6244816  0.4243189  0.4484478
      9                  1450     0.6247304  0.4240163  0.4483704
      9                  1500     0.6247797  0.4241100  0.4480581
    
    Tuning parameter 'shrinkage' was held constant at a value of 0.1
    
    Tuning parameter 'n.minobsinnode' was held constant at a value of 20
    RMSE was used to select the optimal model using the smallest value.
    The final values used for the model were n.trees = 200, interaction.depth =
     9, shrinkage = 0.1 and n.minobsinnode = 20.


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Regression" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>
