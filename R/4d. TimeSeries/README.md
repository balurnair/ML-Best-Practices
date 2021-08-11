# Forecasting 


<div class="list-group" id="list-tab" role="tablist">
 <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Notebook Content</h3>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Library" role="tab" aria-controls="profile">Library<span class="badge badge-primary badge-pill"></span></a>
<div class="list-group" id="list-tab" role="tablist">
<h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Time Series forecasting</h3>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Time-Series" role="tab" aria-controls="profile">Time Series<span class="badge badge-primary badge-pill"></span></a></br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Plotting-Data" role="tab" aria-controls="profile">Time Serires Plot<span class="badge badge-primary badge-pill"></span></a>
<div class="list-group" id="list-tab" role="tablist">
<h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Stationarity of a Time Series</h3>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Test-for-Stationary" role="tab" aria-controls="profile">Test for Stationary<span class="badge badge-primary badge-pill"></span></a></br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Differencing" role="tab" aria-controls="profile">Differencing<span class="badge badge-primary badge-pill"></span></a></br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Decomposing" role="tab" aria-controls="profile">Decomposing<span class="badge badge-primary badge-pill"></span></a>
<div class="list-group" id="list-tab" role="tablist">
<h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Univariate Forecasting techniques </h3>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Naive-Approach" role="tab" aria-controls="profile">Naive Approach<span class="badge badge-primary badge-pill"></span></a></br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Simple-Moving-Average" role="tab" aria-controls="profile">Simple Moving Average<span class="badge badge-primary badge-pill"></span></a></br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Exponential-Smoothing" role="tab" aria-controls="profile">Exponential Smoothing<span class="badge badge-primary badge-pill"></span></a></br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Holt’s-Method" role="tab" aria-controls="profile">Holt’s Method<span class="badge badge-primary badge-pill"></span></a></br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Holt-Winters’-Method" role="tab" aria-controls="profile">Holt-Winters’ Method<span class="badge badge-primary badge-pill"></span></a></br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#ACF-&-PACF" role="tab" aria-controls="profile">ACF & PACF<span class="badge badge-primary badge-pill"></span></a></br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#ARIMA-Model" role="tab" aria-controls="profile">ARIMA Model<span class="badge badge-primary badge-pill"></span></a></br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Seasonal-ARIMA" role="tab" aria-controls="profile">Seasonal ARIMA<span class="badge badge-primary badge-pill"></span></a></br>
<a class="list-group-item list-group-item-action" data-toggle="list" href="#Auto-ARIMA" role="tab" aria-controls="profile">Auto ARIMA<span class="badge badge-primary badge-pill"></span></a></br>
 <a class="list-group-item list-group-item-action" data-toggle="list" href="#TBATS-model" role="tab" aria-controls="profile">TBATS model<span class="badge badge-primary badge-pill"></span></a></br>
<div class="list-group" id="list-tab" role="tablist">
 <a class="list-group-item list-group-item-action" data-toggle="list" href="#Model-Selection" role="tab" aria-controls="profile">Model Selection<span class="badge badge-primary badge-pill"></span></a></br>
    
    
 <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Multivariate Time Series</h3>
 <a class="list-group-item list-group-item-action" data-toggle="list" href="#Multivariate-Time-Series-–-VAR" role="tab" aria-controls="profile">Multivariate Time Series – VAR<span class="badge badge-primary badge-pill"></span></a></br>
 <a class="list-group-item list-group-item-action" data-toggle="list" href="#Arimax/Sarimax" role="tab" aria-controls="profile">Arimax/Sarimax<span class="badge badge-primary badge-pill"></span></a>


# Library


```R
# install.packages("forecast")
# install.packages("uroot")
# install.packages('tseries')
# install.packages('vars')
# install.packages('fpp2')
# library(forecast)
# library(fpp2)
# library(TTR)
# library(tseries)
# library(uroot)
# options(warn=-1) 
```

# Time Series forecasting

### Time Series
<br>A time series is usually modelled through a stochastic process Y(t), i.e. a sequence of random variables. In a forecasting setting we find ourselves at time t and we are interested in estimating Y(t+h), using only information available at time t.


```R
#Importing data
data <- read.csv('dataset/AirPassengers.csv')
#Printing head
head(data)
```


<table>
<caption>A data.frame: 6 × 2</caption>
<thead>
	<tr><th></th><th scope=col>Month</th><th scope=col>Passengers</th></tr>
	<tr><th></th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>1949-01</td><td>112</td></tr>
	<tr><th scope=row>2</th><td>1949-02</td><td>118</td></tr>
	<tr><th scope=row>3</th><td>1949-03</td><td>132</td></tr>
	<tr><th scope=row>4</th><td>1949-04</td><td>129</td></tr>
	<tr><th scope=row>5</th><td>1949-05</td><td>121</td></tr>
	<tr><th scope=row>6</th><td>1949-06</td><td>135</td></tr>
</tbody>
</table>




```R
nrow(data)

```


144



```R
# Number of rows to take in train and test
N_all = nrow(data)
N_train = 132
N_test = 12

# Dividing training set and testing set
train <- data[1:132,]
test <- data[133: 144,]
test
```


<table>
<caption>A data.frame: 12 × 2</caption>
<thead>
	<tr><th></th><th scope=col>Month</th><th scope=col>Passengers</th></tr>
	<tr><th></th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>133</th><td>1960-01</td><td>417</td></tr>
	<tr><th scope=row>134</th><td>1960-02</td><td>391</td></tr>
	<tr><th scope=row>135</th><td>1960-03</td><td>419</td></tr>
	<tr><th scope=row>136</th><td>1960-04</td><td>461</td></tr>
	<tr><th scope=row>137</th><td>1960-05</td><td>472</td></tr>
	<tr><th scope=row>138</th><td>1960-06</td><td>535</td></tr>
	<tr><th scope=row>139</th><td>1960-07</td><td>622</td></tr>
	<tr><th scope=row>140</th><td>1960-08</td><td>606</td></tr>
	<tr><th scope=row>141</th><td>1960-09</td><td>508</td></tr>
	<tr><th scope=row>142</th><td>1960-10</td><td>461</td></tr>
	<tr><th scope=row>143</th><td>1960-11</td><td>390</td></tr>
	<tr><th scope=row>144</th><td>1960-12</td><td>432</td></tr>
</tbody>
</table>




```R
data_ts <- ts(data[, 2], start = c(1949, 1), end = c(1960, 12), frequency = 12)
data_ts
```


<table>
<caption>A Time Series: 12 × 12</caption>
<thead>
	<tr><th></th><th scope=col>Jan</th><th scope=col>Feb</th><th scope=col>Mar</th><th scope=col>Apr</th><th scope=col>May</th><th scope=col>Jun</th><th scope=col>Jul</th><th scope=col>Aug</th><th scope=col>Sep</th><th scope=col>Oct</th><th scope=col>Nov</th><th scope=col>Dec</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1949</th><td>112</td><td>118</td><td>132</td><td>129</td><td>121</td><td>135</td><td>148</td><td>148</td><td>136</td><td>119</td><td>104</td><td>118</td></tr>
	<tr><th scope=row>1950</th><td>115</td><td>126</td><td>141</td><td>135</td><td>125</td><td>149</td><td>170</td><td>170</td><td>158</td><td>133</td><td>114</td><td>140</td></tr>
	<tr><th scope=row>1951</th><td>145</td><td>150</td><td>178</td><td>163</td><td>172</td><td>178</td><td>199</td><td>199</td><td>184</td><td>162</td><td>146</td><td>166</td></tr>
	<tr><th scope=row>1952</th><td>171</td><td>180</td><td>193</td><td>181</td><td>183</td><td>218</td><td>230</td><td>242</td><td>209</td><td>191</td><td>172</td><td>194</td></tr>
	<tr><th scope=row>1953</th><td>196</td><td>196</td><td>236</td><td>235</td><td>229</td><td>243</td><td>264</td><td>272</td><td>237</td><td>211</td><td>180</td><td>201</td></tr>
	<tr><th scope=row>1954</th><td>204</td><td>188</td><td>235</td><td>227</td><td>234</td><td>264</td><td>302</td><td>293</td><td>259</td><td>229</td><td>203</td><td>229</td></tr>
	<tr><th scope=row>1955</th><td>242</td><td>233</td><td>267</td><td>269</td><td>270</td><td>315</td><td>364</td><td>347</td><td>312</td><td>274</td><td>237</td><td>278</td></tr>
	<tr><th scope=row>1956</th><td>284</td><td>277</td><td>317</td><td>313</td><td>318</td><td>374</td><td>413</td><td>405</td><td>355</td><td>306</td><td>271</td><td>306</td></tr>
	<tr><th scope=row>1957</th><td>315</td><td>301</td><td>356</td><td>348</td><td>355</td><td>422</td><td>465</td><td>467</td><td>404</td><td>347</td><td>305</td><td>336</td></tr>
	<tr><th scope=row>1958</th><td>340</td><td>318</td><td>362</td><td>348</td><td>363</td><td>435</td><td>491</td><td>505</td><td>404</td><td>359</td><td>310</td><td>337</td></tr>
	<tr><th scope=row>1959</th><td>360</td><td>342</td><td>406</td><td>396</td><td>420</td><td>472</td><td>548</td><td>559</td><td>463</td><td>407</td><td>362</td><td>405</td></tr>
	<tr><th scope=row>1960</th><td>417</td><td>391</td><td>419</td><td>461</td><td>472</td><td>535</td><td>622</td><td>606</td><td>508</td><td>461</td><td>390</td><td>432</td></tr>
</tbody>
</table>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Plotting Data


```R
#Check the cycle of data and plot the raw data
x = as.data.frame(data_ts)
plot(x, ylab="Passengers (1000s)", type="o")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_12_0.png)


### Preparing the Time Series Object
To run the forecasting models in 'R', one must convert the data into a time series object which is done in the first line of code below. The 'start' and 'end' argument specifies the time of the first and the last observation, respectively. The argument 'frequency' specifies the number of observations per unit of time basically it defines how the data is distributed. For example if its a monthly data then we give frequency = 12

Then create a utility function for calculating Mean Absolute Percentage Error (or MAPE), which will be used to evaluate the performance of the forecasting models. The lower the MAPE value, the better the forecasting model. This is done in the second to fourth lines of code.


```R
dat_ts <- ts(train[, 2], start = c(1949, 1), end = c(1959, 12), frequency = 12)
dat_ts
```


<table>
<caption>A Time Series: 11 × 12</caption>
<thead>
	<tr><th></th><th scope=col>Jan</th><th scope=col>Feb</th><th scope=col>Mar</th><th scope=col>Apr</th><th scope=col>May</th><th scope=col>Jun</th><th scope=col>Jul</th><th scope=col>Aug</th><th scope=col>Sep</th><th scope=col>Oct</th><th scope=col>Nov</th><th scope=col>Dec</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1949</th><td>112</td><td>118</td><td>132</td><td>129</td><td>121</td><td>135</td><td>148</td><td>148</td><td>136</td><td>119</td><td>104</td><td>118</td></tr>
	<tr><th scope=row>1950</th><td>115</td><td>126</td><td>141</td><td>135</td><td>125</td><td>149</td><td>170</td><td>170</td><td>158</td><td>133</td><td>114</td><td>140</td></tr>
	<tr><th scope=row>1951</th><td>145</td><td>150</td><td>178</td><td>163</td><td>172</td><td>178</td><td>199</td><td>199</td><td>184</td><td>162</td><td>146</td><td>166</td></tr>
	<tr><th scope=row>1952</th><td>171</td><td>180</td><td>193</td><td>181</td><td>183</td><td>218</td><td>230</td><td>242</td><td>209</td><td>191</td><td>172</td><td>194</td></tr>
	<tr><th scope=row>1953</th><td>196</td><td>196</td><td>236</td><td>235</td><td>229</td><td>243</td><td>264</td><td>272</td><td>237</td><td>211</td><td>180</td><td>201</td></tr>
	<tr><th scope=row>1954</th><td>204</td><td>188</td><td>235</td><td>227</td><td>234</td><td>264</td><td>302</td><td>293</td><td>259</td><td>229</td><td>203</td><td>229</td></tr>
	<tr><th scope=row>1955</th><td>242</td><td>233</td><td>267</td><td>269</td><td>270</td><td>315</td><td>364</td><td>347</td><td>312</td><td>274</td><td>237</td><td>278</td></tr>
	<tr><th scope=row>1956</th><td>284</td><td>277</td><td>317</td><td>313</td><td>318</td><td>374</td><td>413</td><td>405</td><td>355</td><td>306</td><td>271</td><td>306</td></tr>
	<tr><th scope=row>1957</th><td>315</td><td>301</td><td>356</td><td>348</td><td>355</td><td>422</td><td>465</td><td>467</td><td>404</td><td>347</td><td>305</td><td>336</td></tr>
	<tr><th scope=row>1958</th><td>340</td><td>318</td><td>362</td><td>348</td><td>363</td><td>435</td><td>491</td><td>505</td><td>404</td><td>359</td><td>310</td><td>337</td></tr>
	<tr><th scope=row>1959</th><td>360</td><td>342</td><td>406</td><td>396</td><td>420</td><td>472</td><td>548</td><td>559</td><td>463</td><td>407</td><td>362</td><td>405</td></tr>
</tbody>
</table>




```R
# Mape
mape <- function(actual,pred){
  mape <- mean(abs((actual - pred)/actual))*100
  return (mape)
}
```

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Stationarity of a Time Series
A TS is said to be stationary if its statistical properties such as mean, variance remain constant over time.
We can sat that if a TS has a particular behaviour over time, there is a very high probability that it will follow the same in the future. Also, the theories related to stationary series are more mature and easier to implement as compared to non-stationary series.

## Test for Stationary

It is clearly evident that there is an overall increasing trend in the data along with some seasonal variations. However, it might not always be possible to make such visual inferences (we’ll see such cases later). So, more formally, we can check stationarity using the following:<br>
**Dickey-Fuller Test:** This is one of the statistical tests for checking stationarity.
Note: for further study click on link https://www.analyticsvidhya.com/blog/2015/12/complete-tutorial-time-series-modeling/ 


```R
# install.packages('tseries')
# install.packages("uroot")
library(tseries)
options(warn=-1)
```


```R
#Testing the stationarity of the data
#Augmented Dickey-Fuller Test
adf.test(data_ts)
```


    
    	Augmented Dickey-Fuller Test
    
    data:  data_ts
    Dickey-Fuller = -7.3186, Lag order = 5, p-value = 0.01
    alternative hypothesis: stationary
    


the p-value is 0.01 which is <0.05, therefore, we reject the null hypothesis and hence time series is stationary.



### How to make a Time Series Stationary?
A time series can be broken down into 3 major components.<br>
**Trend:** Upward & downward movement of the data with time over a large period of time (i.e. house appreciation)<br>
**Seasonality:** Seasonal variance (i.e. an increase in demand for ice cream during summer)<br>
**Noise:** Spikes & troughs at random intervals


<br>**Eliminating Trend and Seasonality**
The simple trend reduction techniques discussed before don’t work in all cases, particularly the ones with high seasonality. Lets discuss two ways of removing trend and seasonality:

1. Differencing – taking the difference with a particular time lag
2. Decomposition – modeling both trend and seasonality and removing them from the model.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


## Differencing
One of the most common methods of dealing with both trend and seasonality is differencing. In this technique, we take the difference of the observation at a particular instant with that at the previous instant. This mostly works well in improving stationarity. First order differencing can be done in Pandas as:


```R
## twice-difference the CO2 data
co2.D2 <- diff(data_ts, differences = 2)   
# differences -- means the data has diffrenced for two time to make it stationary
## plot the differenced data
plot(co2.D2)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_26_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


## Decomposing
In this approach, both trend and seasonality are modeled separately and the remaining part of the series is returned.


```R
#Decomposing the data into its trend, seasonal, and random error components
# The multiplicative model is useful when the seasonal variation increases over time
# The additive model is useful when the seasonal variation is relatively constant over time.
# from multiplicative and additive can be used as per the TS 
tsdata_decom <- decompose(data_ts, type = "multiplicative")
plot(tsdata_decom)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_29_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Univariate Forecasting techniques 

# Naive Approach

In the Naïve model, the forecasts for every horizon correspond to the last observed value.
<br>**Ŷ(t+h|t) = Y(t)**
<br>This kind of forecast assumes that the stochastic model generating the time series is a random walk.


```R
library(forecast)

naive_mod <- naive(dat_ts, h=12)
summary(naive_mod)
plot(forecast(naive_mod), col="red", xlab="Time", ylab="Passengers")
```

    
    Forecast method: Naive method
    
    Model Information:
    Call: naive(y = dat_ts, h = 12) 
    
    Residual sd: 31.3321 
    
    Error measures:
                       ME     RMSE      MAE       MPE     MAPE     MASE      ACF1
    Training set 2.236641 31.33213 24.08397 0.4168179 8.979488 0.790935 0.2863378
    
    Forecasts:
             Point Forecast    Lo 80    Hi 80    Lo 95    Hi 95
    Jan 1960            405 364.8463 445.1537 343.5902 466.4098
    Feb 1960            405 348.2140 461.7860 318.1534 491.8466
    Mar 1960            405 335.4517 474.5483 298.6350 511.3650
    Apr 1960            405 324.6925 485.3075 282.1803 527.8197
    May 1960            405 315.2135 494.7865 267.6834 542.3166
    Jun 1960            405 306.6438 503.3562 254.5772 555.4228
    Jul 1960            405 298.7632 511.2368 242.5248 567.4752
    Aug 1960            405 291.4281 518.5719 231.3067 578.6933
    Sep 1960            405 284.5388 525.4612 220.7705 589.2295
    Oct 1960            405 278.0227 531.9773 210.8050 599.1950
    Nov 1960            405 271.8251 538.1749 201.3266 608.6734
    Dec 1960            405 265.9034 544.0966 192.2701 617.7299
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_34_1.png)


The output above shows that the naive method predicts the same value for the entire forecasting horizon. compute the Forecasted value and evaluate the model performance on the test data.


```R
# MAPE 
test$naive = 405
mape(test$Passengers, test$naive) 
```


14.2513384867722


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Simple Moving Average
Simple Moving Average (SMA) takes the average over some set number of time periods. So a 10 period SMA would be over 10 periods (usually meaning 10 trading days).
The Simple Moving Average formula is a very basic arithmetic mean over the number of periods.

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/SMA.png)


```R
# install.packages('TTR')
library("TTR")
```


```R
sma =  SMA(dat_ts)
pred = forecast(sma,h=12)
pred
```


             Point Forecast    Lo 80    Hi 80    Lo 95    Hi 95
    Jan 1960       441.2759 436.3153 446.2364 433.6893 448.8624
    Feb 1960       440.0994 432.8813 447.3175 429.0602 451.1386
    Mar 1960       445.5066 436.2860 454.7271 431.4050 459.6081
    Apr 1960       443.8833 432.9668 454.7999 427.1879 460.5787
    May 1960       436.9862 424.6360 449.3364 418.0982 455.8741
    Jun 1960       437.4568 423.5514 451.3621 416.1904 458.7232
    Jul 1960       448.5365 432.7417 464.3312 424.3805 472.6925
    Aug 1960       465.4364 447.4834 483.3894 437.9797 492.8931
    Sep 1960       479.3940 459.3168 499.4713 448.6885 510.0996
    Oct 1960       482.9280 461.1233 504.7327 449.5806 516.2754
    Nov 1960       477.1322 454.0424 500.2220 441.8195 512.4449
    Dec 1960       478.0376 453.3651 502.7101 440.3043 515.7709



```R
# MAPE
df_sma = as.data.frame(sma)
df1_complete <- na.omit(df_sma)
test$Passengers
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>417</li><li>391</li><li>419</li><li>461</li><li>472</li><li>535</li><li>622</li><li>606</li><li>508</li><li>461</li><li>390</li><li>432</li></ol>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Exponential Smoothing

Exponential Smoothing Methods are a family of forecasting models. **it uses weighted averages of past observations to forecast new values**. Here, the idea is to give more importance to recent values in the series. Thus, as observations get older (in time), the importance of these values get exponentially smaller.

Simple exponential smoothing is a good choice for forecasting data with no clear trend or seasonal pattern. Forecasts are calculated using weighted averages, which means the largest weights are associated with most recent observations, while the smallest weights are associated with the oldest observations:
<br>where 0≤ α ≤1 is the smoothing parameter.


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/ES.png)

```R
library(forecast)
se_model <- ses(dat_ts, h = 12)
summary(se_model)
plot(forecast(se_model), col="red", xlab="Time", ylab="Passengers")
```

    
    Forecast method: Simple exponential smoothing
    
    Model Information:
    Simple exponential smoothing 
    
    Call:
     ses(y = dat_ts, h = 12) 
    
      Smoothing parameters:
        alpha = 0.9999 
    
      Initial states:
        l = 112.015 
    
      sigma:  31.4533
    
         AIC     AICc      BIC 
    1558.920 1559.107 1567.568 
    
    Error measures:
                       ME     RMSE      MAE       MPE     MAPE      MASE      ACF1
    Training set 2.219773 31.21412 23.90229 0.4135676 8.911792 0.7849683 0.2863262
    
    Forecasts:
             Point Forecast    Lo 80    Hi 80    Lo 95    Hi 95
    Jan 1960       404.9957 364.6867 445.3047 343.3483 466.6431
    Feb 1960       404.9957 347.9930 461.9984 317.8175 492.1739
    Mar 1960       404.9957 335.1830 474.8084 298.2264 511.7649
    Apr 1960       404.9957 324.3837 485.6077 281.7102 528.2812
    May 1960       404.9957 314.8691 495.1223 267.1590 542.8324
    Jun 1960       404.9957 306.2673 503.7241 254.0037 555.9877
    Jul 1960       404.9957 298.3571 511.6343 241.9061 568.0853
    Aug 1960       404.9957 290.9945 518.9969 230.6459 579.3455
    Sep 1960       404.9957 284.0793 525.9121 220.0700 589.9214
    Oct 1960       404.9957 277.5388 532.4526 210.0672 599.9242
    Nov 1960       404.9957 271.3179 538.6735 200.5531 609.4383
    Dec 1960       404.9957 265.3739 544.6175 191.4625 618.5289
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_46_1.png)


evaluation of model performance on the test data.

The first line of code below stores the output of the model in a data frame. The second line adds a new variable, simplexp, in the test data which contains the forecasted value from the simple exponential model


```R
# MAPE
df_fc = as.data.frame(se_model)
test$simplexp = df_fc$`Point Forecast`
mape(test$Passengers, test$simplexp)
```


14.2518951121403


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Holt’s Method
Holt extended simple exponential smoothing (solution to data with no clear trend or seasonality) to **allow the forecasting of data with trends**. Holt’s method involves a forecast equation and two smoothing equations (one for the level and one for the trend)<bR>
where 0≤ α ≤1 is the level smoothing parameter, and 0≤ β* ≤1 is the trend smoothing parameter.


```R
library(forecast) 
holt_model <- holt(dat_ts, h = 12)
summary(holt_model)
plot(forecast(holt_model), col="red", xlab="Time", ylab="Passengers")
```

    
    Forecast method: Holt's method
    
    Model Information:
    Holt's method 
    
    Call:
     holt(y = dat_ts, h = 12) 
    
      Smoothing parameters:
        alpha = 0.9996 
        beta  = 1e-04 
    
      Initial states:
        l = 120.1255 
        b = 2.0887 
    
      sigma:  31.6347
    
         AIC     AICc      BIC 
    1562.391 1562.868 1576.805 
    
    Error measures:
                         ME     RMSE     MAE        MPE     MAPE     MASE      ACF1
    Training set 0.06902756 31.15172 23.8295 -0.5842125 8.965372 0.782578 0.2860526
    
    Forecasts:
             Point Forecast    Lo 80    Hi 80    Lo 95    Hi 95
    Jan 1960       407.0734 366.5319 447.6149 345.0705 469.0763
    Feb 1960       409.1630 351.8371 466.4888 321.4906 496.8353
    Mar 1960       411.2525 341.0441 481.4610 303.8780 518.6271
    Apr 1960       413.3421 332.2710 494.4132 289.3546 537.3297
    May 1960       415.4317 324.7887 506.0747 276.8052 554.0582
    Jun 1960       417.5213 318.2232 516.8194 265.6580 569.3846
    Jul 1960       419.6109 312.3524 526.8694 255.5731 583.6486
    Aug 1960       421.7005 307.0314 536.3696 246.3292 597.0717
    Sep 1960       423.7900 302.1597 545.4204 237.7724 609.8076
    Oct 1960       425.8796 297.6641 554.0951 229.7909 621.9683
    Nov 1960       427.9692 293.4894 562.4490 222.3001 633.6383
    Dec 1960       430.0588 289.5926 570.5250 215.2343 644.8833
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_51_1.png)



```R
# MAPE
df_holt = as.data.frame(holt_model)
test$holt = df_holt$`Point Forecast`
mape(test$Passengers, test$holt)
```


12.5405983033639


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Holt-Winters’ Method
Holt-Winters’ Method is suitable for data with trends and seasonalities which includes a seasonality smoothing parameter γ. There are two variations to this method:
1. Additive method: the seasonal variations are roughly constant through the series.
2. Multiplicative method: the seasonal variations are changing proportionally to the level of the series.
<br>full Holt-Winters’ method including a trend component and a seasonal component. Statsmodels allows for all the combinations including as shown in the examples below:


```R
library(forecast)
holtw <- HoltWinters(dat_ts)
plot(forecast(holtw,12), col="red", xlab="Time", ylab="Passengers")
summary(holtw)
```


                 Length Class  Mode     
    fitted       480    mts    numeric  
    x            132    ts     numeric  
    alpha          1    -none- numeric  
    beta           1    -none- numeric  
    gamma          1    -none- numeric  
    coefficients  14    -none- numeric  
    seasonal       1    -none- character
    SSE            1    -none- numeric  
    call           2    -none- call     



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_55_1.png)



```R
# MAPE
df_holt = as.data.frame(forecast(holtw))
mape(test$Passengers, df_holt$'Point Forecast')
```


5.64150258428401


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# ACF & PACF

**Autocorrelation function plot (ACF):**<br>
Autocorrelation refers to how correlated a time series is with its past values whereas the ACF is the plot used to see the correlation between the points, up to and including the lag unit. In ACF, the correlation coefficient is in the x-axis whereas the number of lags is shown in the y-axis.

**Partial Autocorrelation Function plots (PACF):** A partial autocorrelation is a summary of the relationship between an observation in a time series with observations at prior time steps with the relationships of intervening observations removed or the partial autocorrelation at lag k is the correlation that results after removing the effect of any correlations due to the terms at shorter lags.


```R
library(forecast)
Acf(dat_ts)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_59_0.png)



```R
# library(forecast)
Pacf(dat_ts)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_60_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# AR Model
RIMA, short for ‘Auto Regressive Integrated Moving Average’ is actually a class of models that ‘explains’ a given time series based on its own past values, that is, its own lags and the lagged forecast errors, so that equation can be used to forecast future values.<br>
There are seasonal and Non-seasonal ARIMA models that can be used for forecasting
The predictors depend on the parameters (p,d,q) of the ARIMA model:

**Number of AR (Auto-Regressive) terms (p):** Partial autocorrelation can be imagined as the correlation between the series and its lag, after excluding the contributions from the intermediate lags. So, PACF sort of conveys the pure correlation between a lag and the series. For instance if p is 5, the predictors for x(t) will be x(t-1)….x(t-5)<br>


```R
library(forecast)
ar_model<-arima(dat_ts,c(1,0,0))
ar_model
```


    
    Call:
    arima(x = dat_ts, order = c(1, 0, 0))
    
    Coefficients:
             ar1  intercept
          0.9618   261.5006
    s.e.  0.0233    60.2964
    
    sigma^2 estimated as 965.5:  log likelihood = -642.19,  aic = 1290.38



```R
# Predicting the test value using ar model
ar_forecast <- as.data.frame(predict(ar_model, n.ahead=12))

```

**Plot the Forecast**<br>
plot.ts() function use to first plot the original data and then add points for the forecasted values using the points() function as shown below:


```R
plot.ts(data_ts)
points(ar_forecast$pred , type = "l", col = 2)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_66_0.png)



```R
# MAPE
mape(test$Passengers, ar_forecast$pred)
```


19.7095219916492


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# MA Model
**Number of MA (Moving Average) terms (q):** MA terms are lagged forecast errors in prediction equation. For instance if q is 5, the predictors for x(t) will be e(t-1)….e(t-5) where e(i) is the difference between the moving average at ith instant and actual value<br>



```R
library(forecast)

plot(dat_ts)
ma_model <- ma(dat_ts,order = 12)
lines(ma_model, col="red")
summary(ma_model)
```


       Min. 1st Qu.  Median    Mean 3rd Qu.    Max.    NA's 
      126.8   182.4   239.1   260.5   345.1   425.5      12 



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_70_1.png)



```R
f_cast = forecast(ma_model,18) 
# MAPE
mape(test$Passengers, f_cast$mean[7:18])
```


13.0586092814793


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# ARIMA Model

ARIMA, short for ‘Auto Regressive Integrated Moving Average’ is actually a class of models that ‘explains’ a given time series based on its own past values, that is, its own lags and the lagged forecast errors, so that equation can be used to forecast future values.
There are seasonal and Non-seasonal ARIMA models that can be used for forecasting


```R
library(forecast)
arima_model <- arima(dat_ts,order = c(4,1,4))
summary(arima_model)
plot(arima_model)
autoplot(arima_model)
```

    
    Call:
    arima(x = dat_ts, order = c(4, 1, 4))
    
    Coefficients:
            ar1     ar2      ar3      ar4      ma1      ma2     ma3     ma4
          0.013  0.9893  -0.1402  -0.7799  -0.0095  -1.4628  0.0634  0.8804
    s.e.  0.080  0.0873   0.0718   0.0721   0.0862   0.0903  0.0724  0.0767
    
    sigma^2 estimated as 518.7:  log likelihood = -598.53,  aic = 1215.06
    
    Training set error measures:
                       ME     RMSE      MAE      MPE     MAPE      MASE       ACF1
    Training set 5.109689 22.68943 16.96178 1.581097 6.483126 0.7042769 0.04521609
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_74_1.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_74_2.png)



```R
fore_arima = forecast::forecast(arima_model, h=12)
df_arima = as.data.frame(fore_arima)
test$arima = df_arima$`Point Forecast`
# MAPE
mape(test$Passengers, test$arima)
```


7.86737184419184



```R
plot.ts(data_ts)
points(fore_arima$mean , type = "l", col = 2)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_76_0.png)


Taking it back to original scale
Since the combined model gave best result, lets scale it back to the original values and see how well it performs there. First step would be to store the predicted results as a separate series and observe it.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Seasonal ARIMA 

An extension to ARIMA that supports the direct modeling of the seasonal component of the series is called SARIMA.
1. The limitations of ARIMA when it comes to seasonal data.
2. The SARIMA extension of ARIMA that explicitly models the seasonal element in univariate data.
<br>It adds three new hyperparameters to specify the autoregression (AR), differencing (I) and moving average (MA) for the seasonal component of the series, as well as an additional parameter for the period of the seasonality
<br> Here the Seasonal Component are added i.e. P, Q, D and m (seasonal period)


```R
pdqParam = c(0, 1, 1)
manualFit <- arima(dat_ts, pdqParam, seasonal = list(order = pdqParam, period = 12))

```


```R
summary(manualFit)
```

    
    Call:
    arima(x = dat_ts, order = pdqParam, seasonal = list(order = pdqParam, period = 12))
    
    Coefficients:
              ma1     sma1
          -0.2167  -0.0843
    s.e.   0.0913   0.0849
    
    sigma^2 estimated as 108.6:  log likelihood = -447.86,  aic = 901.72
    
    Training set error measures:
                        ME     RMSE      MAE       MPE     MAPE      MASE      ACF1
    Training set 0.5901905 9.896706 7.502005 0.1194352 2.865133 0.3114937 -0.012993
    


```R
arimaorder(manualFit)
```


<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>p</dt><dd>0</dd><dt>d</dt><dd>1</dd><dt>q</dt><dd>1</dd><dt>P</dt><dd>0</dd><dt>D</dt><dd>1</dd><dt>Q</dt><dd>1</dd><dt>Frequency</dt><dd>12</dd></dl>




```R
autoPred = forecast(manualFit, h = 12)

```


```R
autoPred
```


             Point Forecast    Lo 80    Hi 80    Lo 95    Hi 95
    Jan 1960       422.9845 409.6266 436.3423 402.5554 443.4135
    Feb 1960       404.7081 387.7403 421.6760 378.7580 430.6583
    Mar 1960       467.0908 447.1563 487.0254 436.6036 497.5781
    Apr 1960       456.7989 434.2853 479.3125 422.3673 491.2305
    May 1960       479.9818 455.1556 504.8080 442.0134 517.9502
    Jun 1960       533.6253 506.6844 560.5663 492.4227 574.8280
    Jul 1960       607.8447 578.9432 636.7461 563.6438 652.0456
    Aug 1960       619.0059 588.2688 649.7430 571.9976 666.0142
    Sep 1960       522.8630 490.3939 555.3322 473.2057 572.5203
    Oct 1960       467.7106 433.5972 501.8239 415.5387 519.8825
    Nov 1960       422.4272 386.7453 458.1091 367.8564 476.9980
    Dec 1960       464.1092 426.9249 501.2936 407.2407 520.9778



```R
df = as.data.frame(autoPred)
test$simplexp = df$'Point Forecast'
mape(test$Passengers, test$simplexp)

```


3.65238733494184



```R
plot.ts(data_ts)
points(autoPred$mean , type = "l", col = 2)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_87_0.png)


 <a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Auto-ARIMA

Auto ARIMA/SARIMA is a powerful function provided by R's forecast package. As discussed above we were required to build plots and charts to find p,d and q and then impute the values in arima() but lets say we had more then 1000 time series.In that case it would be very difficult to find p,d and q of individual time series and hence we use auto.arima() which automatically finds the best p,d and q values using Hyndman-Khandakar algorithm.For further understanding you can refer the following link:
https://otexts.com/fpp2/arima-r.html


```R

```


```R
library(forecast)
```


```R
class(dat_ts) 
```


'ts'



```R
model = auto.arima(dat_ts,seasonal = FALSE)
```


```R
autoPred_non_seasonal = forecast(model,h=12)
```

## To find the ARIMA order we can use arima order function which returns p,d,q values


```R
arimaorder(model)
```


<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>p</dt><dd>4</dd><dt>d</dt><dd>1</dd><dt>q</dt><dd>4</dd></dl>




```R
mape(test$Passengers, autoPred_non_seasonal$mean)
```


9.64751057669067



```R
model = auto.arima(dat_ts,seasonal = TRUE)
autoPred_seasonal = forecast(model,h=12)
mape(test$Passengers, autoPred_seasonal$mean)
```


4.18239527895942



```R
# When the seasonal parameter is given true the values of p,d,q and P,D,Q is also available now
arimaorder(model)
```


<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>p</dt><dd>1</dd><dt>d</dt><dd>1</dd><dt>q</dt><dd>0</dd><dt>P</dt><dd>0</dd><dt>D</dt><dd>1</dd><dt>Q</dt><dd>0</dd><dt>Frequency</dt><dd>12</dd></dl>




```R
autoPred_non_seasonal
```


             Point Forecast    Lo 80    Hi 80    Lo 95    Hi 95
    Jan 1960       425.5702 396.3473 454.7931 380.8777 470.2627
    Feb 1960       432.2525 391.8385 472.6665 370.4446 494.0604
    Mar 1960       489.0878 446.7324 531.4432 424.3108 553.8648
    Apr 1960       499.4493 456.1639 542.7347 433.2499 565.6486
    May 1960       539.9511 496.5550 583.3472 473.5825 606.3197
    Jun 1960       539.7919 496.2399 583.3438 473.1848 606.3989
    Jul 1960       536.0011 492.3466 579.6556 469.2373 602.7649
    Aug 1960       525.4080 481.7460 569.0699 458.6328 592.1831
    Sep 1960       492.9301 447.9391 537.9212 424.1222 561.7381
    Oct 1960       486.2330 438.8782 533.5877 413.8101 558.6558
    Nov 1960       461.6871 408.8568 514.5173 380.8901 542.4840
    Dec 1960       470.2665 412.6859 527.8470 382.2046 558.3283



```R
# plot of both seasonal auto arima and non seasonal auto arima
plot.ts(data_ts)
points(autoPred_seasonal$mean , type = "l", col = 2)
points(autoPred_non_seasonal$mean , type = "l", col = 3)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_101_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


The green color indicates non seasonal auto arima where as the red one denotes the seasonal auto arima 

# TBATS model

Time-series forecasting refers to the use of a model to predict future values based on previously observed values. Many researchers are familiar with time-series forecasting yet they struggle with specific types of time-series data. One such type of data is data with seasonality. There can be many types of seasonalities present (e.g., time of day, daily, weekly, monthly, yearly).<BR>

TBATS is a forecasting method to model time series data.The main aim of this is to forecast time series with complex seasonal patterns using exponential smoothing.

TBATS is an acronym for key features of the model:

1. T: Trigonometric seasonality
2. B: Box-Cox transformation
3. A: ARIMA errors
4. T: Trend
5. S: Seasonal components
<br>TBATS makes it easy for users to handle data with multiple seasonal patterns. This model is preferable when the seasonality changes over time.


```R
library(forecast)
model_tbats <- tbats(dat_ts)
summary(model_tbats)
```


                      Length Class  Mode     
    lambda               1   -none- numeric  
    alpha                1   -none- numeric  
    beta                 1   -none- numeric  
    damping.parameter    1   -none- numeric  
    gamma.one.values     1   -none- numeric  
    gamma.two.values     1   -none- numeric  
    ar.coefficients      0   -none- NULL     
    ma.coefficients      0   -none- NULL     
    likelihood           1   -none- numeric  
    optim.return.code    1   -none- numeric  
    variance             1   -none- numeric  
    AIC                  1   -none- numeric  
    parameters           2   -none- list     
    seed.states         12   -none- numeric  
    fitted.values      132   ts     numeric  
    errors             132   ts     numeric  
    x                 1584   -none- numeric  
    seasonal.periods     1   -none- numeric  
    k.vector             1   -none- numeric  
    y                  132   ts     numeric  
    p                    1   -none- numeric  
    q                    1   -none- numeric  
    call                 2   -none- call     
    series               1   -none- character
    method               1   -none- character



```R
for_tbats <- forecast::forecast(model_tbats, h = 12)
df_tbats = as.data.frame(for_tbats)
test$tbats = df_tbats$`Point Forecast`
mape(test$Passengers, test$tbats) 
```


3.35414104003087



```R
plot.ts(data_ts)
points(for_tbats$mean , type = "l", col = 2)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_109_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Model-Selection

Model-selection is used in cases where we are dealing with multiple time series models and is required to select any one from them. In cases like these we use the approach of model selection based on a value called AIC. 

The Akaike information criterion (AIC) is a mathematical method for evaluating how well a model fits the data it was generated from. In statistics, AIC is used to compare different possible models and determine which one is the best fit for the data. AIC is calculated from:<br>

the number of independent variables used to build the model.
<br>
the maximum likelihood estimate of the model (how well the model reproduces the data).
<br>

The best-fit model according to AIC is the one that explains the greatest amount of variation using the fewest possible independent variables.

AIC determines the relative information value of the model using the maximum likelihood estimate and the number of parameters (independent variables) in the model. The formula for AIC is:

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/Model-Selection.png)


K is the number of independent variables used and L is the log-likelihood estimate (a.k.a. the likelihood that the model could have produced your observed y-values). The default K is always 2, so if your model uses one independent variable your K will be 3, if it uses two independent variables your K will be 4, and so on.

To compare models using AIC, you need to calculate the AIC of each model. If a model is more than 2 AIC units lower than another, then it is considered significantly better than that model.

You can easily calculate AIC by hand if you have the log-likelihood of your model, but calculating log-likelihood is complicated! Most statistical software will include a function for calculating AIC.


```R
# AIC is available in Model 
sarima = manualFit$aic
model_auto_arima = model$aic
```


```R
sarima
```


901.721052361976



```R
model_auto_arima
```


899.902124377601


#### As we see Model auto arima is having less AIC value and hence we should be going for Model auto arima 

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Multivariate Time Series – VAR

A Multivariate time series has more than one time-dependent variable. Each variable depends not only on its past values but also has some dependency on other variables.<br>
Consider the above example. Now suppose our dataset includes perspiration percent, dew point, wind speed, cloud cover percentage, etc. along with the temperature value for the past two years. In this case, there are multiple variables to be considered to optimally predict temperature. A series like this would fall under the category of multivariate time series. Below is an illustration of this:

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/MTS-VAR.png)

Now that we understand what a multivariate time series looks like, let us understand how can we use it to build a forecast.

**Dealing with a Multivariate Time Series – VAR**
In this section, I will introduce you to one of the most commonly used methods for multivariate time series forecasting – Vector Auto Regression (VAR).

In a VAR model, each variable is a linear function of the past values of itself and the past values of all the other variables.<br>
for further unnderstanding for theory:https://www.analyticsvidhya.com/blog/2018/09/multivariate-time-series-guide-forecasting-modeling-python-codes/


```R
# install.packages('vars')
library(vars)
options(warn=-1)
```

    Loading required package: MASS
    
    Loading required package: strucchange
    
    Loading required package: zoo
    
    
    Attaching package: 'zoo'
    
    
    The following objects are masked from 'package:base':
    
        as.Date, as.Date.numeric
    
    
    Loading required package: sandwich
    
    Loading required package: urca
    
    Loading required package: lmtest
    
    

In order to estimate the VAR model I use the vars package by Pfaff (2008). The relevant function is VAR and its use is straightforward. You just have to load the package and specify the data (y), order (p) and the type of the model. The option type determines whether to include an intercept term, a trend or both in the model. Since the artificial sample does not contain any deterministic term, we neglect it in the estimation by setting type = "none".


```R
data(EuStockMarkets)
# EuStockMarkets data set for VAR modeling 
```


```R
plot(EuStockMarkets)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_123_0.png)



```R
# Main packages - problem: both have different functions VAR
## Testing for stationarity
### tseries - standard test adt.test
# apply(EuStockMarkets, 2, adf.test)
stnry = diff(EuStockMarkets) #difference operation on a vector of time series. Default order of differencing is 1.
apply(stnry, 2, adf.test)


# Lag order identification
#We will use two different functions, from two different packages to identify the lag order for the VAR model. Both functions are quite similar to each other but differ in the output they produce. vars::VAR is a more powerful and convinient function to identify the correct lag order. 
VARselect(stnry, 
          type = "none", #type of deterministic regressors to include. We use none becasue the time series was made stationary using differencing above. 
          lag.max = 10) #highest lag order
```


    $DAX
    
    	Augmented Dickey-Fuller Test
    
    data:  newX[, i]
    Dickey-Fuller = -9.9997, Lag order = 12, p-value = 0.01
    alternative hypothesis: stationary
    
    
    $SMI
    
    	Augmented Dickey-Fuller Test
    
    data:  newX[, i]
    Dickey-Fuller = -10.769, Lag order = 12, p-value = 0.01
    alternative hypothesis: stationary
    
    
    $CAC
    
    	Augmented Dickey-Fuller Test
    
    data:  newX[, i]
    Dickey-Fuller = -11.447, Lag order = 12, p-value = 0.01
    alternative hypothesis: stationary
    
    
    $FTSE
    
    	Augmented Dickey-Fuller Test
    
    data:  newX[, i]
    Dickey-Fuller = -10.838, Lag order = 12, p-value = 0.01
    alternative hypothesis: stationary
    
    



<dl>
	<dt>$selection</dt>
		<dd><style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>AIC(n)</dt><dd>9</dd><dt>HQ(n)</dt><dd>1</dd><dt>SC(n)</dt><dd>1</dd><dt>FPE(n)</dt><dd>9</dd></dl>
</dd>
	<dt>$criteria</dt>
		<dd><table>
<caption>A matrix: 4 × 10 of type dbl</caption>
<thead>
	<tr><th></th><th scope=col>1</th><th scope=col>2</th><th scope=col>3</th><th scope=col>4</th><th scope=col>5</th><th scope=col>6</th><th scope=col>7</th><th scope=col>8</th><th scope=col>9</th><th scope=col>10</th></tr>
</thead>
<tbody>
	<tr><th scope=row>AIC(n)</th><td>2.527062e+01</td><td>2.527564e+01</td><td>2.526566e+01</td><td>2.525844e+01</td><td>2.525725e+01</td><td>2.525408e+01</td><td>2.525692e+01</td><td>2.525696e+01</td><td>2.525073e+01</td><td>2.525455e+01</td></tr>
	<tr><th scope=row>HQ(n)</th><td>2.528823e+01</td><td>2.531088e+01</td><td>2.531850e+01</td><td>2.532891e+01</td><td>2.534534e+01</td><td>2.535978e+01</td><td>2.538023e+01</td><td>2.539789e+01</td><td>2.540927e+01</td><td>2.543071e+01</td></tr>
	<tr><th scope=row>SC(n)</th><td>2.531840e+01</td><td>2.537122e+01</td><td>2.540902e+01</td><td>2.544959e+01</td><td>2.549619e+01</td><td>2.554080e+01</td><td>2.559143e+01</td><td>2.563926e+01</td><td>2.568081e+01</td><td>2.573242e+01</td></tr>
	<tr><th scope=row>FPE(n)</th><td>9.438206e+10</td><td>9.485771e+10</td><td>9.391500e+10</td><td>9.324010e+10</td><td>9.312964e+10</td><td>9.283467e+10</td><td>9.309877e+10</td><td>9.310329e+10</td><td>9.252533e+10</td><td>9.288047e+10</td></tr>
</tbody>
</table>
</dd>
</dl>




```R
# Creating a VAR model with vars
# install.packages('vars')
library(vars)
var.a <- vars::VAR(stnry,
                   lag.max = 10, #highest lag order for lag length selection according to the choosen ic
                   ic = "AIC", #information criterion
                   type = "none") #type of deterministic regressors to include
summary(var.a)
```


    
    VAR Estimation Results:
    ========================= 
    Endogenous variables: DAX, SMI, CAC, FTSE 
    Deterministic variables: none 
    Sample size: 1850 
    Log Likelihood: -33712.408 
    Roots of the characteristic polynomial:
    0.817 0.817 0.8116 0.8116 0.7915 0.7915 0.7864 0.7864 0.7784 0.7784 0.7579 0.7579 0.7541 0.7541 0.7537 0.7537 0.7473 0.7421 0.7421 0.7295 0.7295 0.7285 0.7153 0.7153 0.6723 0.6723 0.6696 0.6696 0.6616 0.6616 0.6551 0.577 0.577 0.4544 0.289 0.1213
    Call:
    vars::VAR(y = stnry, type = "none", lag.max = 10, ic = "AIC")
    
    
    Estimation results for equation DAX: 
    ==================================== 
    DAX = DAX.l1 + SMI.l1 + CAC.l1 + FTSE.l1 + DAX.l2 + SMI.l2 + CAC.l2 + FTSE.l2 + DAX.l3 + SMI.l3 + CAC.l3 + FTSE.l3 + DAX.l4 + SMI.l4 + CAC.l4 + FTSE.l4 + DAX.l5 + SMI.l5 + CAC.l5 + FTSE.l5 + DAX.l6 + SMI.l6 + CAC.l6 + FTSE.l6 + DAX.l7 + SMI.l7 + CAC.l7 + FTSE.l7 + DAX.l8 + SMI.l8 + CAC.l8 + FTSE.l8 + DAX.l9 + SMI.l9 + CAC.l9 + FTSE.l9 
    
              Estimate Std. Error t value Pr(>|t|)    
    DAX.l1   0.0096570  0.0424492   0.227 0.820065    
    SMI.l1  -0.1008170  0.0297641  -3.387 0.000721 ***
    CAC.l1   0.0752689  0.0465795   1.616 0.106285    
    FTSE.l1  0.0730055  0.0366170   1.994 0.046328 *  
    DAX.l2   0.0190453  0.0423265   0.450 0.652792    
    SMI.l2  -0.0172409  0.0298939  -0.577 0.564188    
    CAC.l2   0.0687124  0.0465965   1.475 0.140487    
    FTSE.l2 -0.0804753  0.0369389  -2.179 0.029489 *  
    DAX.l3  -0.0676359  0.0423179  -1.598 0.110155    
    SMI.l3   0.0135412  0.0299928   0.451 0.651696    
    CAC.l3   0.0484694  0.0466586   1.039 0.299032    
    FTSE.l3  0.0409793  0.0369675   1.109 0.267783    
    DAX.l4  -0.0501669  0.0422723  -1.187 0.235480    
    SMI.l4   0.0162536  0.0300860   0.540 0.589099    
    CAC.l4   0.1001510  0.0469324   2.134 0.032981 *  
    FTSE.l4 -0.0451988  0.0369319  -1.224 0.221170    
    DAX.l5   0.0109497  0.0424940   0.258 0.796687    
    SMI.l5  -0.0978623  0.0303192  -3.228 0.001270 ** 
    CAC.l5   0.0731622  0.0469140   1.559 0.119054    
    FTSE.l5 -0.0254787  0.0369942  -0.689 0.491086    
    DAX.l6  -0.0121897  0.0424062  -0.287 0.773800    
    SMI.l6   0.0246183  0.0303677   0.811 0.417660    
    CAC.l6   0.0871855  0.0468724   1.860 0.063039 .  
    FTSE.l6  0.0007736  0.0369967   0.021 0.983320    
    DAX.l7   0.0786601  0.0425103   1.850 0.064421 .  
    SMI.l7  -0.0050826  0.0302543  -0.168 0.866604    
    CAC.l7  -0.0691098  0.0466880  -1.480 0.138981    
    FTSE.l7 -0.0418380  0.0370855  -1.128 0.259406    
    DAX.l8  -0.0336346  0.0425857  -0.790 0.429743    
    SMI.l8   0.0963209  0.0304325   3.165 0.001576 ** 
    CAC.l8  -0.1180253  0.0466645  -2.529 0.011515 *  
    FTSE.l8  0.0517022  0.0371047   1.393 0.163666    
    DAX.l9  -0.0262047  0.0423049  -0.619 0.535714    
    SMI.l9   0.0052002  0.0303603   0.171 0.864022    
    CAC.l9   0.1359369  0.0467408   2.908 0.003678 ** 
    FTSE.l9 -0.0068091  0.0369929  -0.184 0.853983    
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    
    Residual standard error: 32.09 on 1814 degrees of freedom
    Multiple R-Squared: 0.05129,	Adjusted R-squared: 0.03246 
    F-statistic: 2.724 on 36 and 1814 DF,  p-value: 2.04e-07 
    
    
    Estimation results for equation SMI: 
    ==================================== 
    SMI = DAX.l1 + SMI.l1 + CAC.l1 + FTSE.l1 + DAX.l2 + SMI.l2 + CAC.l2 + FTSE.l2 + DAX.l3 + SMI.l3 + CAC.l3 + FTSE.l3 + DAX.l4 + SMI.l4 + CAC.l4 + FTSE.l4 + DAX.l5 + SMI.l5 + CAC.l5 + FTSE.l5 + DAX.l6 + SMI.l6 + CAC.l6 + FTSE.l6 + DAX.l7 + SMI.l7 + CAC.l7 + FTSE.l7 + DAX.l8 + SMI.l8 + CAC.l8 + FTSE.l8 + DAX.l9 + SMI.l9 + CAC.l9 + FTSE.l9 
    
             Estimate Std. Error t value Pr(>|t|)    
    DAX.l1   0.035132   0.052025   0.675 0.499578    
    SMI.l1  -0.038299   0.036478  -1.050 0.293895    
    CAC.l1   0.046647   0.057087   0.817 0.413969    
    FTSE.l1  0.127516   0.044877   2.841 0.004541 ** 
    DAX.l2   0.006278   0.051875   0.121 0.903691    
    SMI.l2   0.018350   0.036637   0.501 0.616532    
    CAC.l2   0.104672   0.057108   1.833 0.066983 .  
    FTSE.l2 -0.096675   0.045272  -2.135 0.032859 *  
    DAX.l3  -0.148622   0.051864  -2.866 0.004210 ** 
    SMI.l3   0.004229   0.036759   0.115 0.908422    
    CAC.l3   0.094768   0.057184   1.657 0.097644 .  
    FTSE.l3  0.131679   0.045307   2.906 0.003701 ** 
    DAX.l4  -0.175243   0.051808  -3.383 0.000733 ***
    SMI.l4   0.029175   0.036873   0.791 0.428904    
    CAC.l4   0.124249   0.057520   2.160 0.030895 *  
    FTSE.l4  0.011514   0.045263   0.254 0.799239    
    DAX.l5   0.007207   0.052080   0.138 0.889954    
    SMI.l5  -0.089506   0.037159  -2.409 0.016106 *  
    CAC.l5   0.070892   0.057497   1.233 0.217751    
    FTSE.l5 -0.037913   0.045339  -0.836 0.403156    
    DAX.l6  -0.072106   0.051972  -1.387 0.165490    
    SMI.l6   0.011650   0.037218   0.313 0.754308    
    CAC.l6   0.102452   0.057446   1.783 0.074681 .  
    FTSE.l6 -0.001026   0.045343  -0.023 0.981944    
    DAX.l7   0.147987   0.052100   2.840 0.004555 ** 
    SMI.l7  -0.012999   0.037079  -0.351 0.725941    
    CAC.l7  -0.123208   0.057220  -2.153 0.031432 *  
    FTSE.l7 -0.049168   0.045451  -1.082 0.279502    
    DAX.l8   0.008599   0.052192   0.165 0.869153    
    SMI.l8   0.089777   0.037298   2.407 0.016182 *  
    CAC.l8  -0.099393   0.057191  -1.738 0.082397 .  
    FTSE.l8 -0.019262   0.045475  -0.424 0.671921    
    DAX.l9   0.072664   0.051848   1.401 0.161245    
    SMI.l9  -0.091853   0.037209  -2.469 0.013657 *  
    CAC.l9   0.081425   0.057285   1.421 0.155371    
    FTSE.l9  0.068442   0.045338   1.510 0.131322    
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    
    Residual standard error: 39.33 on 1814 degrees of freedom
    Multiple R-Squared: 0.05981,	Adjusted R-squared: 0.04115 
    F-statistic: 3.206 on 36 and 1814 DF,  p-value: 7.227e-10 
    
    
    Estimation results for equation CAC: 
    ==================================== 
    CAC = DAX.l1 + SMI.l1 + CAC.l1 + FTSE.l1 + DAX.l2 + SMI.l2 + CAC.l2 + FTSE.l2 + DAX.l3 + SMI.l3 + CAC.l3 + FTSE.l3 + DAX.l4 + SMI.l4 + CAC.l4 + FTSE.l4 + DAX.l5 + SMI.l5 + CAC.l5 + FTSE.l5 + DAX.l6 + SMI.l6 + CAC.l6 + FTSE.l6 + DAX.l7 + SMI.l7 + CAC.l7 + FTSE.l7 + DAX.l8 + SMI.l8 + CAC.l8 + FTSE.l8 + DAX.l9 + SMI.l9 + CAC.l9 + FTSE.l9 
    
             Estimate Std. Error t value Pr(>|t|)   
    DAX.l1  -0.001041   0.034340  -0.030  0.97582   
    SMI.l1  -0.071184   0.024078  -2.956  0.00315 **
    CAC.l1   0.043492   0.037681   1.154  0.24857   
    FTSE.l1  0.082268   0.029622   2.777  0.00554 **
    DAX.l2   0.014488   0.034241   0.423  0.67225   
    SMI.l2  -0.027912   0.024183  -1.154  0.24857   
    CAC.l2   0.083842   0.037695   2.224  0.02626 * 
    FTSE.l2 -0.063578   0.029882  -2.128  0.03350 * 
    DAX.l3  -0.032437   0.034234  -0.948  0.34350   
    SMI.l3   0.031449   0.024263   1.296  0.19508   
    CAC.l3  -0.059480   0.037745  -1.576  0.11524   
    FTSE.l3  0.023769   0.029905   0.795  0.42683   
    DAX.l4  -0.112680   0.034197  -3.295  0.00100 **
    SMI.l4   0.045902   0.024338   1.886  0.05946 . 
    CAC.l4   0.071056   0.037967   1.872  0.06143 . 
    FTSE.l4 -0.020521   0.029877  -0.687  0.49225   
    DAX.l5  -0.040047   0.034376  -1.165  0.24418   
    SMI.l5  -0.040002   0.024527  -1.631  0.10308   
    CAC.l5   0.044130   0.037952   1.163  0.24507   
    FTSE.l5 -0.011466   0.029927  -0.383  0.70166   
    DAX.l6  -0.010487   0.034305  -0.306  0.75987   
    SMI.l6   0.017464   0.024566   0.711  0.47724   
    CAC.l6   0.046108   0.037918   1.216  0.22415   
    FTSE.l6 -0.002253   0.029929  -0.075  0.94000   
    DAX.l7   0.093443   0.034389   2.717  0.00665 **
    SMI.l7  -0.011696   0.024475  -0.478  0.63280   
    CAC.l7  -0.058576   0.037769  -1.551  0.12110   
    FTSE.l7 -0.059667   0.030001  -1.989  0.04687 * 
    DAX.l8   0.012292   0.034450   0.357  0.72128   
    SMI.l8   0.026246   0.024619   1.066  0.28653   
    CAC.l8  -0.102523   0.037750  -2.716  0.00667 **
    FTSE.l8  0.048842   0.030016   1.627  0.10387   
    DAX.l9   0.019936   0.034223   0.583  0.56028   
    SMI.l9  -0.025465   0.024560  -1.037  0.29995   
    CAC.l9   0.048093   0.037812   1.272  0.20357   
    FTSE.l9 -0.003901   0.029926  -0.130  0.89630   
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    
    Residual standard error: 25.96 on 1814 degrees of freedom
    Multiple R-Squared: 0.04515,	Adjusted R-squared: 0.0262 
    F-statistic: 2.382 on 36 and 1814 DF,  p-value: 8.732e-06 
    
    
    Estimation results for equation FTSE: 
    ===================================== 
    FTSE = DAX.l1 + SMI.l1 + CAC.l1 + FTSE.l1 + DAX.l2 + SMI.l2 + CAC.l2 + FTSE.l2 + DAX.l3 + SMI.l3 + CAC.l3 + FTSE.l3 + DAX.l4 + SMI.l4 + CAC.l4 + FTSE.l4 + DAX.l5 + SMI.l5 + CAC.l5 + FTSE.l5 + DAX.l6 + SMI.l6 + CAC.l6 + FTSE.l6 + DAX.l7 + SMI.l7 + CAC.l7 + FTSE.l7 + DAX.l8 + SMI.l8 + CAC.l8 + FTSE.l8 + DAX.l9 + SMI.l9 + CAC.l9 + FTSE.l9 
    
             Estimate Std. Error t value Pr(>|t|)    
    DAX.l1   0.025600   0.039796   0.643  0.52012    
    SMI.l1  -0.085874   0.027904  -3.078  0.00212 ** 
    CAC.l1  -0.003879   0.043668  -0.089  0.92923    
    FTSE.l1  0.165616   0.034328   4.824 1.52e-06 ***
    DAX.l2   0.023464   0.039681   0.591  0.55438    
    SMI.l2  -0.023240   0.028025  -0.829  0.40708    
    CAC.l2   0.028324   0.043684   0.648  0.51683    
    FTSE.l2 -0.031301   0.034630  -0.904  0.36619    
    DAX.l3  -0.052914   0.039673  -1.334  0.18245    
    SMI.l3   0.012312   0.028118   0.438  0.66154    
    CAC.l3   0.057729   0.043742   1.320  0.18709    
    FTSE.l3  0.007780   0.034657   0.224  0.82240    
    DAX.l4  -0.054187   0.039630  -1.367  0.17170    
    SMI.l4   0.043084   0.028206   1.527  0.12681    
    CAC.l4   0.078160   0.043999   1.776  0.07583 .  
    FTSE.l4 -0.083589   0.034624  -2.414  0.01587 *  
    DAX.l5   0.001615   0.039838   0.041  0.96767    
    SMI.l5  -0.042176   0.028424  -1.484  0.13803    
    CAC.l5   0.102931   0.043982   2.340  0.01938 *  
    FTSE.l5 -0.069017   0.034682  -1.990  0.04674 *  
    DAX.l6  -0.027039   0.039756  -0.680  0.49651    
    SMI.l6   0.058310   0.028470   2.048  0.04069 *  
    CAC.l6   0.094202   0.043943   2.144  0.03219 *  
    FTSE.l6 -0.088315   0.034684  -2.546  0.01097 *  
    DAX.l7   0.054056   0.039853   1.356  0.17514    
    SMI.l7   0.052084   0.028363   1.836  0.06648 .  
    CAC.l7  -0.065521   0.043770  -1.497  0.13458    
    FTSE.l7 -0.055592   0.034768  -1.599  0.11000    
    DAX.l8  -0.004926   0.039924  -0.123  0.90181    
    SMI.l8   0.057267   0.028530   2.007  0.04488 *  
    CAC.l8  -0.080907   0.043748  -1.849  0.06456 .  
    FTSE.l8  0.017291   0.034786   0.497  0.61919    
    DAX.l9   0.003630   0.039661   0.092  0.92709    
    SMI.l9  -0.017471   0.028463  -0.614  0.53942    
    CAC.l9   0.069078   0.043819   1.576  0.11510    
    FTSE.l9  0.012475   0.034681   0.360  0.71911    
    ---
    Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    
    
    Residual standard error: 30.08 on 1814 degrees of freedom
    Multiple R-Squared: 0.05941,	Adjusted R-squared: 0.04075 
    F-statistic: 3.183 on 36 and 1814 DF,  p-value: 9.501e-10 
    
    
    
    Covariance matrix of residuals:
            DAX    SMI   CAC  FTSE
    DAX  1025.7  940.4 616.8 647.6
    SMI   940.4 1537.5 650.9 722.0
    CAC   616.8  650.9 672.0 522.2
    FTSE  647.6  722.0 522.2 903.2
    
    Correlation matrix of residuals:
            DAX    SMI    CAC   FTSE
    DAX  1.0000 0.7488 0.7430 0.6729
    SMI  0.7488 1.0000 0.6403 0.6127
    CAC  0.7430 0.6403 1.0000 0.6703
    FTSE 0.6729 0.6127 0.6703 1.0000
    
    



```R
# Residual diagnostics
#serial.test function takes the VAR model as the input.  
serial.test(var.a)

#selecting the variables
# Granger test for causality
#for causality function to give reliable results we need all the variables of the multivariate time series to be stationary. 
causality(var.a, #VAR model
          cause = c("DAX")) #cause variable. If not specified then first column of x is used. Multiple variables can be used. 

```


    
    	Portmanteau Test (asymptotic)
    
    data:  Residuals of VAR object var.a
    Chi-squared = 183.36, df = 112, p-value = 2.444e-05
    
    $serial
    
    	Portmanteau Test (asymptotic)
    
    data:  Residuals of VAR object var.a
    Chi-squared = 183.36, df = 112, p-value = 2.444e-05
    
    



    $Granger
    
    	Granger causality H0: DAX do not Granger-cause SMI CAC FTSE
    
    data:  VAR object var.a
    F-Test = 1.7314, df1 = 27, df2 = 7256, p-value = 0.01074
    
    
    $Instant
    
    	H0: No instantaneous causality between: DAX and SMI CAC FTSE
    
    data:  VAR object var.a
    Chi-squared = 759.19, df = 3, p-value < 2.2e-16
    
    



```R
## Forecasting VAR models
fcast = predict(var.a, n.ahead = 25) # we forecast over a short horizon because beyond short horizon prediction becomes unreliable or uniform
par(mar = c(2.5,2.5,2.5,2.5))
plot(fcast)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_127_0.png)



```R
# Forecasting the DAX index
DAX = fcast$fcst[1]; DAX # type list


# Extracting the forecast column
x = DAX$DAX[,1]; x
```


<strong>$DAX</strong> = <table>
<caption>A matrix: 25 × 4 of type dbl</caption>
<thead>
	<tr><th scope=col>fcst</th><th scope=col>lower</th><th scope=col>upper</th><th scope=col>CI</th></tr>
</thead>
<tbody>
	<tr><td>  5.60084304</td><td>-57.29384</td><td>68.49552</td><td>62.89468</td></tr>
	<tr><td> -5.91965377</td><td>-69.10567</td><td>57.26636</td><td>63.18601</td></tr>
	<tr><td>-13.00325572</td><td>-76.29843</td><td>50.29191</td><td>63.29517</td></tr>
	<tr><td> 13.72698427</td><td>-49.63068</td><td>77.08465</td><td>63.35767</td></tr>
	<tr><td>-36.35436670</td><td>-99.79645</td><td>27.08772</td><td>63.44208</td></tr>
	<tr><td> -0.64933493</td><td>-64.37759</td><td>63.07892</td><td>63.72826</td></tr>
	<tr><td>  5.96623225</td><td>-57.93847</td><td>69.87093</td><td>63.90470</td></tr>
	<tr><td>  8.82501176</td><td>-55.17502</td><td>72.82505</td><td>64.00004</td></tr>
	<tr><td>  2.15548901</td><td>-62.14270</td><td>66.45368</td><td>64.29819</td></tr>
	<tr><td>  2.12089142</td><td>-62.42840</td><td>66.67018</td><td>64.54929</td></tr>
	<tr><td> -0.80619025</td><td>-65.36242</td><td>63.75003</td><td>64.55622</td></tr>
	<tr><td> -4.24776524</td><td>-68.81260</td><td>60.31707</td><td>64.56484</td></tr>
	<tr><td>  0.31282719</td><td>-64.25708</td><td>64.88274</td><td>64.56991</td></tr>
	<tr><td> -2.10767272</td><td>-66.68072</td><td>62.46537</td><td>64.57304</td></tr>
	<tr><td>  1.30500015</td><td>-63.28275</td><td>65.89275</td><td>64.58775</td></tr>
	<tr><td>  0.23742105</td><td>-64.35338</td><td>64.82822</td><td>64.59080</td></tr>
	<tr><td>  1.78179652</td><td>-62.81456</td><td>66.37815</td><td>64.59635</td></tr>
	<tr><td> -0.36003725</td><td>-64.96020</td><td>64.24012</td><td>64.60016</td></tr>
	<tr><td>  0.72242303</td><td>-63.87835</td><td>65.32319</td><td>64.60077</td></tr>
	<tr><td> -0.69067282</td><td>-65.29176</td><td>63.91041</td><td>64.60109</td></tr>
	<tr><td> -0.43392018</td><td>-65.03547</td><td>64.16762</td><td>64.60155</td></tr>
	<tr><td> -0.39358579</td><td>-64.99520</td><td>64.20802</td><td>64.60161</td></tr>
	<tr><td> -0.08492216</td><td>-64.68679</td><td>64.51694</td><td>64.60186</td></tr>
	<tr><td>  0.19373467</td><td>-64.40828</td><td>64.79575</td><td>64.60202</td></tr>
	<tr><td>  0.22812144</td><td>-64.37395</td><td>64.83020</td><td>64.60207</td></tr>
</tbody>
</table>




<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>5.60084303897527</li><li>-5.91965376521413</li><li>-13.0032557241428</li><li>13.7269842717171</li><li>-36.3543667033214</li><li>-0.649334929627913</li><li>5.96623225099533</li><li>8.82501175697222</li><li>2.15548900858749</li><li>2.12089142411205</li><li>-0.806190247248811</li><li>-4.24776523827309</li><li>0.312827190104973</li><li>-2.10767272423435</li><li>1.30500014822655</li><li>0.237421048159101</li><li>1.78179652381606</li><li>-0.360037247830662</li><li>0.722423028321822</li><li>-0.690672822500132</li><li>-0.433920180409265</li><li>-0.393585786698533</li><li>-0.0849221629418965</li><li>0.193734671217351</li><li>0.228121437536343</li></ol>




```R
# Inverting the differencing
#To get the data to the original scale we invert the time series
#since the values are just difference from the previous value, to get the values on the original scale we add the last value from the DAX time series to the predicted values.
#the plot of the predicted values will also show that over longer horizon the predicted values are not reliable
x = cumsum(x) + 5473.72
par(mar = c(2.5,2.5,1,2.5)) #bottom, left, top, and right
plot.ts(x)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_129_0.png)



```R
# Adding data and forecast to one time series
DAXinv =ts(c(EuStockMarkets[,1], x),
           start = c(1991,130), 
           frequency = 260)
plot(DAXinv)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_130_0.png)



```R
library(ggplot2)
DAXinv_datframe <- as.data.frame(DAXinv[1786:1885]) 
colnames(DAXinv_datframe) <- c("x")
head(DAXinv_datframe)
ggplot() + 
  geom_line(data = as.data.frame(DAXinv_datframe[1:75,]), aes(y = get("DAXinv_datframe[1:75, ]"), x = seq(1, 75)), color = "green") +
  geom_line(data = as.data.frame(DAXinv_datframe[76:100,]), aes(y = get("DAXinv_datframe[76:100, ]"), x = seq(76, 100)), color = "red") +
  ggtitle("Plot of forecast of the VAR model on `EuStockMarkets''s DAX time series") +
  theme(plot.title = element_text(hjust = 0.5)) +
  xlab("Time") + ylab("Value")
```


<table>
<caption>A data.frame: 6 × 1</caption>
<thead>
	<tr><th></th><th scope=col>x</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>1</th><td>5337.75</td></tr>
	<tr><th scope=row>2</th><td>5226.20</td></tr>
	<tr><th scope=row>3</th><td>5264.62</td></tr>
	<tr><th scope=row>4</th><td>5164.89</td></tr>
	<tr><th scope=row>5</th><td>5270.61</td></tr>
	<tr><th scope=row>6</th><td>5348.75</td></tr>
</tbody>
</table>




![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_131_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


# Arimax/Sarimax

An Autoregressive Integrated Moving Average with Explanatory Variable (ARIMAX) model can be viewed as a multiple regression model with one or more autoregressive (AR) terms and/or one or more moving average (MA) terms. This method is suitable for forecasting when data is stationary/non stationary, and multivariate with any type of data pattern


```R
library(dplyr)
```

    
    Attaching package: 'dplyr'
    
    
    The following object is masked from 'package:MASS':
    
        select
    
    
    The following objects are masked from 'package:stats':
    
        filter, lag
    
    
    The following objects are masked from 'package:base':
    
        intersect, setdiff, setequal, union
    
    
    


```R
data(EuStockMarkets)
```


```R
df = as.data.frame(EuStockMarkets)
```

## About the data 
The EuStockMarkets data set contains the daily closing prices (except for weekends/holidays) of four European stock exchanges: the DAX (Germany), the SMI (Switzerland), the CAC (France), and the FTSE (UK). An important characteristic of these data is that they represent stock market points, which have different interpretations depending on the exchange


```R
X = df %>% select(SMI,CAC,FTSE)
y = df$DAX
```


```R
dim(df)
```


<style>
.list-inline {list-style: none; margin:0; padding: 0}
.list-inline>li {display: inline-block}
.list-inline>li:not(:last-child)::after {content: "\00b7"; padding: 0 .5ex}
</style>
<ol class=list-inline><li>1860</li><li>4</li></ol>




```R
x_train = X[1:1600,]
x_test = X[1601:nrow(df),]
y_train = y[1:1600]
y_test = y[1601:nrow(df)]
```


```R
# Toggle the seasonal parameter from TRUE to FALSE to build ARIMAX / SARIMAX models

arimax = auto.arima(y_train,xreg = as.matrix(x_train))
arima_mod = auto.arima(y_train,seasonal = TRUE)
```


```R
f_cast = forecast(arimax , xreg = as.matrix(x_test),h=260)
f_cast_arima = forecast(arima_mod,h=260)
```


```R
mape(y_test, f_cast$mean)
mape(y_test,f_cast_arima$mean)
```


3.49926065519492



252.57480258955



```R
plot.ts(df$DAX)
points(f_cast$mean , type = "l", col = 2)
points(f_cast_arima$mean , type = "l", col = 3)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Time%20Series/output_145_0.png)


Arimax is denoted in Red and Green one is simple Arima

Arima  model detects a dip in that particular point keeps on giving a downward trend, as the model does not know or finds any relationship in its past data

## Using Arimax model we are able to predict quite accurately the DAX (Germany) Market hence it is quite clear that if you have a dateset which have dependency on other series those can be used in xreg parameter for better and high accuracy.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Forecasting" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

