
# Anomaly Detection Algorithms

<div class="list-group" id="list-tab" role="tablist">
  <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Notebook Content</h3><br>
   <a class="list-group-item list-group-item-action" data-toggle="list" href="#Introduction" role="tab" aria-controls="settings">Introduction<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#One-Class-SVM" role="tab" aria-controls="settings">One-Class SVM<span class="badge badge-primary badge-pill"></span></a><br>
        <a class="list-group-item list-group-item-action" data-toggle="list" href="#Local-Outlier-Factor" role="tab" aria-controls="settings">Local Outlier Factor<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#Isolation-Forests" role="tab" aria-controls="settings">Isolation Forests<span class="badge badge-primary badge-pill"></span></a><br>
	<a class="list-group-item list-group-item-action" data-toggle="list" href="#dbscan-density-based-spatial-clustering-of-applications-with-noise" role="tab" aria-controls="settings">DBSCAN<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#Clustering-Based-Outlier-Detection-Technique" role="tab" aria-controls="settings">Clustering Based Outlier Detection Technique<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#Feature-Bagging" role="tab" aria-controls="settings">Feature Bagging<span class="badge badge-primary badge-pill"></span></a><br>
	<a class="list-group-item list-group-item-action" data-toggle="list" href="#KNN" role="tab" aria-controls="settings">KNN<span class="badge badge-primary badge-pill"></span></a><br>
      <a class="list-group-item list-group-item-action" data-toggle="list" href="#HBOS" role="tab" aria-controls="settings">HBOS<span class="badge badge-primary badge-pill"></span></a><br>
        

# Introduction

What is Anomaly?

Anomalies are defined as events that deviate from the standard, happen rarely, and do not follow the rest of the “pattern”. Anomalies are also referred to as outliers, surprise, aberrant, deviation, peculiarity, etc. 

<div>
    
![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Anomaly%20Detection/download.png)


</div>


Real-time Applications: 

Intrusion detection, fraud detection, system health monitoring, event detection in sensor networks, and detecting eco-system disturbances

Challenges in Anomaly Detection:

•	The difficulty to achieve high anomaly detection recall rate

•	Anomaly detection in high-dimensional space - Anomalies often exhibit evident abnormal characteristics in a low-dimensional space yet become hidden and unnoticeable in a high-dimensional space 

•	Due to the difficulty and cost of collecting large-scale labelled anomaly data, it is important to have data-efficient learning of normality/abnormality

**Unsupervised methods are the best choice for anomaly detection, since they recognize new and unknown objects whereas supervised methods can detect only pre-known abnormal cases. 

# R Implementation starts from here

# One Class SVM
One Class SVM i.e. One-Class Support Vector Machine is an unsupervised algorithm that learns a decision function to identify outliers. We will be using the Iris dataset.

One-Class SVM is similar to Standard SVM, but instead of using a hyperplane to separate two classes of instances, it uses a hypersphere to encompass all of the instances. Now think of the "margin" as referring to the outside of the hypersphere -- so by "the largest possible margin", we mean "the smallest possible hypersphere".

That's about it. Note the following facts, true of SVM, still apply to One-Class SVM:

* If we insist that there are no margin violations, by seeking the smallest hypersphere, the margin will end up touching a small number of instances. These are the "support vectors", and they fully determine the model. As long as they are within the hypersphere, all of the other instances can be changed without affecting the model.

* We can allow for some margin violations if we don't want the model to be too sensitive to noise.

* We can do this in the original space, or in an enlarged feature space (implicitly, using the kernel trick), which can result in a boundary with a complex shape in the original space.

It is an algorithm for novelty detection and can be used for time series data as well.

It learns the boundaries of these points and is therefore able to classify any points that lie outside the boundary i.e. outliers.

Novelty detection - Classifying new data even it was not captured in training data. The idea of novelty detection is to detect rare events, i.e. events that happen rarely, and hence, of which you have very little samples.


```R
#We add preliminary library for svm function.
library(e1071)
```

    Warning message:
    "package 'e1071' was built under R version 3.6.3"


```R
#We take the four independent variables of the iris dataset.
iris_data <- iris
iris_X <- iris_data[,1:4]
```


```R
head(iris_data)
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
head(iris_X)
```


<table>
<thead><tr><th scope=col>Sepal.Length</th><th scope=col>Sepal.Width</th><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th></tr></thead>
<tbody>
	<tr><td>5.1</td><td>3.5</td><td>1.4</td><td>0.2</td></tr>
	<tr><td>4.9</td><td>3.0</td><td>1.4</td><td>0.2</td></tr>
	<tr><td>4.7</td><td>3.2</td><td>1.3</td><td>0.2</td></tr>
	<tr><td>4.6</td><td>3.1</td><td>1.5</td><td>0.2</td></tr>
	<tr><td>5.0</td><td>3.6</td><td>1.4</td><td>0.2</td></tr>
	<tr><td>5.4</td><td>3.9</td><td>1.7</td><td>0.4</td></tr>
</tbody>
</table>



## Building and Fitting Model

* In the following code, we first declare the value of nu which defines the upper bound of the fraction of outliers. We then initialize the model and fit it on the iris dataset.


```R
model_oneclasssvm <- svm(iris_X,type='one-classification',kernel = "radial",gamma=0.05,nu=0.05)
model_oneclasssvm
```


    
    Call:
    svm.default(x = iris_X, type = "one-classification", kernel = "radial", 
        gamma = 0.05, nu = 0.05)
    
    
    Parameters:
       SVM-Type:  one-classification 
     SVM-Kernel:  radial 
          gamma:  0.05 
             nu:  0.05 
    
    Number of Support Vectors:  9
    



```R
pred_oneclasssvm <- predict(model_oneclasssvm,iris_X)
pred_oneclasssvm
```


<dl class=dl-horizontal>
	<dt>1</dt>
		<dd>TRUE</dd>
	<dt>2</dt>
		<dd>TRUE</dd>
	<dt>3</dt>
		<dd>TRUE</dd>
	<dt>4</dt>
		<dd>TRUE</dd>
	<dt>5</dt>
		<dd>TRUE</dd>
	<dt>6</dt>
		<dd>TRUE</dd>
	<dt>7</dt>
		<dd>TRUE</dd>
	<dt>8</dt>
		<dd>TRUE</dd>
	<dt>9</dt>
		<dd>TRUE</dd>
	<dt>10</dt>
		<dd>TRUE</dd>
	<dt>11</dt>
		<dd>TRUE</dd>
	<dt>12</dt>
		<dd>TRUE</dd>
	<dt>13</dt>
		<dd>TRUE</dd>
	<dt>14</dt>
		<dd>FALSE</dd>
	<dt>15</dt>
		<dd>TRUE</dd>
	<dt>16</dt>
		<dd>FALSE</dd>
	<dt>17</dt>
		<dd>TRUE</dd>
	<dt>18</dt>
		<dd>TRUE</dd>
	<dt>19</dt>
		<dd>TRUE</dd>
	<dt>20</dt>
		<dd>TRUE</dd>
	<dt>21</dt>
		<dd>TRUE</dd>
	<dt>22</dt>
		<dd>TRUE</dd>
	<dt>23</dt>
		<dd>TRUE</dd>
	<dt>24</dt>
		<dd>TRUE</dd>
	<dt>25</dt>
		<dd>TRUE</dd>
	<dt>26</dt>
		<dd>TRUE</dd>
	<dt>27</dt>
		<dd>TRUE</dd>
	<dt>28</dt>
		<dd>TRUE</dd>
	<dt>29</dt>
		<dd>TRUE</dd>
	<dt>30</dt>
		<dd>TRUE</dd>
	<dt>31</dt>
		<dd>TRUE</dd>
	<dt>32</dt>
		<dd>TRUE</dd>
	<dt>33</dt>
		<dd>TRUE</dd>
	<dt>34</dt>
		<dd>FALSE</dd>
	<dt>35</dt>
		<dd>TRUE</dd>
	<dt>36</dt>
		<dd>TRUE</dd>
	<dt>37</dt>
		<dd>TRUE</dd>
	<dt>38</dt>
		<dd>TRUE</dd>
	<dt>39</dt>
		<dd>TRUE</dd>
	<dt>40</dt>
		<dd>TRUE</dd>
	<dt>41</dt>
		<dd>TRUE</dd>
	<dt>42</dt>
		<dd>FALSE</dd>
	<dt>43</dt>
		<dd>TRUE</dd>
	<dt>44</dt>
		<dd>TRUE</dd>
	<dt>45</dt>
		<dd>TRUE</dd>
	<dt>46</dt>
		<dd>TRUE</dd>
	<dt>47</dt>
		<dd>TRUE</dd>
	<dt>48</dt>
		<dd>TRUE</dd>
	<dt>49</dt>
		<dd>TRUE</dd>
	<dt>50</dt>
		<dd>TRUE</dd>
	<dt>51</dt>
		<dd>TRUE</dd>
	<dt>52</dt>
		<dd>TRUE</dd>
	<dt>53</dt>
		<dd>TRUE</dd>
	<dt>54</dt>
		<dd>TRUE</dd>
	<dt>55</dt>
		<dd>TRUE</dd>
	<dt>56</dt>
		<dd>TRUE</dd>
	<dt>57</dt>
		<dd>TRUE</dd>
	<dt>58</dt>
		<dd>TRUE</dd>
	<dt>59</dt>
		<dd>TRUE</dd>
	<dt>60</dt>
		<dd>TRUE</dd>
	<dt>61</dt>
		<dd>FALSE</dd>
	<dt>62</dt>
		<dd>TRUE</dd>
	<dt>63</dt>
		<dd>TRUE</dd>
	<dt>64</dt>
		<dd>TRUE</dd>
	<dt>65</dt>
		<dd>TRUE</dd>
	<dt>66</dt>
		<dd>TRUE</dd>
	<dt>67</dt>
		<dd>TRUE</dd>
	<dt>68</dt>
		<dd>TRUE</dd>
	<dt>69</dt>
		<dd>TRUE</dd>
	<dt>70</dt>
		<dd>TRUE</dd>
	<dt>71</dt>
		<dd>TRUE</dd>
	<dt>72</dt>
		<dd>TRUE</dd>
	<dt>73</dt>
		<dd>TRUE</dd>
	<dt>74</dt>
		<dd>TRUE</dd>
	<dt>75</dt>
		<dd>TRUE</dd>
	<dt>76</dt>
		<dd>TRUE</dd>
	<dt>77</dt>
		<dd>TRUE</dd>
	<dt>78</dt>
		<dd>TRUE</dd>
	<dt>79</dt>
		<dd>TRUE</dd>
	<dt>80</dt>
		<dd>TRUE</dd>
	<dt>81</dt>
		<dd>TRUE</dd>
	<dt>82</dt>
		<dd>TRUE</dd>
	<dt>83</dt>
		<dd>TRUE</dd>
	<dt>84</dt>
		<dd>TRUE</dd>
	<dt>85</dt>
		<dd>TRUE</dd>
	<dt>86</dt>
		<dd>TRUE</dd>
	<dt>87</dt>
		<dd>TRUE</dd>
	<dt>88</dt>
		<dd>TRUE</dd>
	<dt>89</dt>
		<dd>TRUE</dd>
	<dt>90</dt>
		<dd>TRUE</dd>
	<dt>91</dt>
		<dd>TRUE</dd>
	<dt>92</dt>
		<dd>TRUE</dd>
	<dt>93</dt>
		<dd>TRUE</dd>
	<dt>94</dt>
		<dd>TRUE</dd>
	<dt>95</dt>
		<dd>TRUE</dd>
	<dt>96</dt>
		<dd>TRUE</dd>
	<dt>97</dt>
		<dd>TRUE</dd>
	<dt>98</dt>
		<dd>TRUE</dd>
	<dt>99</dt>
		<dd>TRUE</dd>
	<dt>100</dt>
		<dd>TRUE</dd>
	<dt>101</dt>
		<dd>TRUE</dd>
	<dt>102</dt>
		<dd>TRUE</dd>
	<dt>103</dt>
		<dd>TRUE</dd>
	<dt>104</dt>
		<dd>TRUE</dd>
	<dt>105</dt>
		<dd>TRUE</dd>
	<dt>106</dt>
		<dd>TRUE</dd>
	<dt>107</dt>
		<dd>TRUE</dd>
	<dt>108</dt>
		<dd>TRUE</dd>
	<dt>109</dt>
		<dd>TRUE</dd>
	<dt>110</dt>
		<dd>TRUE</dd>
	<dt>111</dt>
		<dd>TRUE</dd>
	<dt>112</dt>
		<dd>TRUE</dd>
	<dt>113</dt>
		<dd>TRUE</dd>
	<dt>114</dt>
		<dd>TRUE</dd>
	<dt>115</dt>
		<dd>TRUE</dd>
	<dt>116</dt>
		<dd>TRUE</dd>
	<dt>117</dt>
		<dd>TRUE</dd>
	<dt>118</dt>
		<dd>FALSE</dd>
	<dt>119</dt>
		<dd>FALSE</dd>
	<dt>120</dt>
		<dd>TRUE</dd>
	<dt>121</dt>
		<dd>TRUE</dd>
	<dt>122</dt>
		<dd>TRUE</dd>
	<dt>123</dt>
		<dd>FALSE</dd>
	<dt>124</dt>
		<dd>TRUE</dd>
	<dt>125</dt>
		<dd>TRUE</dd>
	<dt>126</dt>
		<dd>TRUE</dd>
	<dt>127</dt>
		<dd>TRUE</dd>
	<dt>128</dt>
		<dd>TRUE</dd>
	<dt>129</dt>
		<dd>TRUE</dd>
	<dt>130</dt>
		<dd>TRUE</dd>
	<dt>131</dt>
		<dd>TRUE</dd>
	<dt>132</dt>
		<dd>TRUE</dd>
	<dt>133</dt>
		<dd>TRUE</dd>
	<dt>134</dt>
		<dd>TRUE</dd>
	<dt>135</dt>
		<dd>TRUE</dd>
	<dt>136</dt>
		<dd>TRUE</dd>
	<dt>137</dt>
		<dd>TRUE</dd>
	<dt>138</dt>
		<dd>TRUE</dd>
	<dt>139</dt>
		<dd>TRUE</dd>
	<dt>140</dt>
		<dd>TRUE</dd>
	<dt>141</dt>
		<dd>TRUE</dd>
	<dt>142</dt>
		<dd>TRUE</dd>
	<dt>143</dt>
		<dd>TRUE</dd>
	<dt>144</dt>
		<dd>TRUE</dd>
	<dt>145</dt>
		<dd>TRUE</dd>
	<dt>146</dt>
		<dd>TRUE</dd>
	<dt>147</dt>
		<dd>TRUE</dd>
	<dt>148</dt>
		<dd>TRUE</dd>
	<dt>149</dt>
		<dd>TRUE</dd>
	<dt>150</dt>
		<dd>TRUE</dd>
</dl>



## Computing Summary

* We can also compute the summary using the summary command.

All values that are equal to False are outliers.


```R
summary(pred_oneclasssvm)
```


       Mode   FALSE    TRUE 
    logical       8     142 


# Local Outlier Factor

LOF (Local Outlier Factor) is an algorithm for identifying density-based local outliers. With LOF, the local density of a point is compared with that of its neighbors. If the former is significantly lower than the latter (with an LOF value greater than one), the point is in a sparser region than its neighbors, which suggests it be an outlier.

The higher the LOF value for an observation, the more anomalous the observation.

Function lofactor(data, k) in packages DMwR and dprep calculates local outlier factors using the LOF algorithm, where k is the number of neighbors used in the calculation of the local outlier factors.


```R
library(DMwR)
# remove "Species", which is a categorical column
outlier.scores <- lofactor(iris_X, k=5)
plot(density(outlier.scores))
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Anomaly%20Detection/output_17_0.png)



```R
head(outlier.scores)
```


<ol class=list-inline>
	<li>0.997970710762121</li>
	<li>1.02158807687099</li>
	<li>1.05452953463337</li>
	<li>0.993617710523857</li>
	<li>1.00497304016236</li>
	<li>1.22883436823715</li>
</ol>



### Visualize Outliers with Plots
Next, we show outliers with a biplot of the first two principal components.


```R
n <- nrow(iris_X)
labels <- 1:n
labels[-outliers] <- "."
biplot(prcomp(iris_X), cex=.8, xlabs=labels)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Anomaly%20Detection/output_20_0.png)


We can also show outliers with a pairs plot as below, where outliers are labeled with "+" in red.


```R
pch <- rep(".", n)
pch[outliers] <- "+"
col <- rep("black", n)
col[outliers] <- "red"
pairs(iris_X, pch=pch, col=col)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Anomaly%20Detection/output_22_0.png)



```R
outliers <- order(outlier.scores, decreasing=T)[1:5]
```


```R
print(outliers)
```

    [1]  42 107  23 110  63
    

### Parallel Computation of LOF Scores
Package Rlof provides function lof(), a parallel implementation of the LOF algorithm. Its usage is similar to the above lofactor(), but lof() has two additional features of supporting multiple values of k and several choices of distance metrics. Below is an example of lof().


```R
library(Rlof)
```

    Warning message:
    "package 'Rlof' was built under R version 3.6.3"Loading required package: doParallel
    Warning message:
    "package 'doParallel' was built under R version 3.6.3"Loading required package: foreach
    Loading required package: iterators
    Loading required package: parallel
    


```R
outlier.scores <- lof(iris_X, k=c(5:10))
```


```R
head(outlier.scores)
```


<table>
<thead><tr><th scope=col>5</th><th scope=col>6</th><th scope=col>7</th><th scope=col>8</th><th scope=col>9</th><th scope=col>10</th></tr></thead>
<tbody>
	<tr><td>0.9979707</td><td>0.9974838</td><td>1.0138815</td><td>0.9786037</td><td>0.9902176</td><td>0.9749183</td></tr>
	<tr><td>1.0215881</td><td>0.9979364</td><td>0.9935776</td><td>0.9863154</td><td>1.0170826</td><td>0.9933587</td></tr>
	<tr><td>1.0545295</td><td>1.0169330</td><td>1.0053840</td><td>1.0352726</td><td>0.9855317</td><td>0.9971526</td></tr>
	<tr><td>0.9936177</td><td>1.0089740</td><td>0.9712264</td><td>0.9918824</td><td>1.0096213</td><td>1.0082478</td></tr>
	<tr><td>1.0049730</td><td>1.0157466</td><td>0.9729420</td><td>0.9868866</td><td>0.9983219</td><td>0.9976917</td></tr>
	<tr><td>1.2288344</td><td>1.1939532</td><td>1.1497786</td><td>1.1463403</td><td>1.1219691</td><td>1.1171744</td></tr>
</tbody>
</table>



In conclusion, the LOF of a point tells the density of this point compared to the density of its neighbors. If the density of a point is much smaller than the densities of its neighbors (LOF ≫1), the point is far from dense areas and, hence, an outlier.

# Isolation Forests

Isolation Forest is unsupervised algorithm and it is based on the Decision Tree algorithm.

It identifies anomaly by isolating outliers in the data.

Isolation Forest uses an ensemble of Isolation Trees for the given data points to isolate anomalies.

### Methodology behind Isolation Forests
For each observation, do the following:
1. Randomly select a feature and randomly select a value for that feature within its range.
2. If the observation’s feature value falls above (below) the selected value, then this value becomes the new min (max) of that feature’s range.
3. Check if at least one other observation has values in the range of each feature in the dataset, where some ranges were altered via step 2. If no, then the observation is isolated.
4. Repeat steps 1–3 until the observation is isolated. The number of times you had to go through these steps is the isolation number. The lower the number, the more anomalous the observation is.


```R
#import packages
library(ggplot2)
library(solitude)
```

    Warning message:
    "package 'solitude' was built under R version 3.6.3"


```R
data(iris)
iris2 <- iris[,1:4]
```


```R
str(iris2)
```

    'data.frame':	150 obs. of  4 variables:
     $ Sepal.Length: num  5.1 4.9 4.7 4.6 5 5.4 4.6 5 4.4 4.9 ...
     $ Sepal.Width : num  3.5 3 3.2 3.1 3.6 3.9 3.4 3.4 2.9 3.1 ...
     $ Petal.Length: num  1.4 1.4 1.3 1.5 1.4 1.7 1.4 1.5 1.4 1.5 ...
     $ Petal.Width : num  0.2 0.2 0.2 0.2 0.2 0.4 0.3 0.2 0.2 0.1 ...
    


```R
set.seed(1)
index = sample(ceiling(nrow(iris2) * 0.5))
```


```R
# initiate an isolation forest
iso = isolationForest$new(sample_size = length(index))
# fit for attrition data
iso$fit(iris2[index, ])
```

    INFO  [13:35:43.553] Building Isolation Forest ...  
    INFO  [13:35:45.322] done 
    INFO  [13:35:45.332] Computing depth of terminal nodes ...  
    INFO  [13:35:47.587] done 
    INFO  [13:35:47.696] Completed growing isolation forest 
    


```R
# Obtain anomaly scores
scores_train = iso$predict(iris2[index, ])
head(scores_train[order(anomaly_score, decreasing = TRUE)])
```


<table>
<thead><tr><th scope=col>id</th><th scope=col>average_depth</th><th scope=col>anomaly_score</th></tr></thead>
<tbody>
	<tr><td>22       </td><td>4.67     </td><td>0.6599617</td></tr>
	<tr><td>74       </td><td>4.98     </td><td>0.6420046</td></tr>
	<tr><td> 6       </td><td>5.01     </td><td>0.6402930</td></tr>
	<tr><td> 8       </td><td>5.39     </td><td>0.6190032</td></tr>
	<tr><td> 1       </td><td>5.67     </td><td>0.6037703</td></tr>
	<tr><td>25       </td><td>5.69     </td><td>0.6026967</td></tr>
</tbody>
</table>




```R
# predict scores for unseen data (50% sample)
scores_unseen = iso$predict(iris2[-index, ])
head(scores_unseen[order(anomaly_score, decreasing = TRUE)])
```


<table>
<thead><tr><th scope=col>id</th><th scope=col>average_depth</th><th scope=col>anomaly_score</th></tr></thead>
<tbody>
	<tr><td>35       </td><td>4.40     </td><td>0.6760104</td></tr>
	<tr><td>43       </td><td>4.40     </td><td>0.6760104</td></tr>
	<tr><td>57       </td><td>4.40     </td><td>0.6760104</td></tr>
	<tr><td>32       </td><td>4.87     </td><td>0.6483198</td></tr>
	<tr><td>51       </td><td>4.92     </td><td>0.6454416</td></tr>
	<tr><td>50       </td><td>4.97     </td><td>0.6425762</td></tr>
</tbody>
</table>



If the score is closer to 1 for a some observations, they are likely outliers. If the score for all observations hover around 0.5, there might not be outliers at all.

By observing the quantiles, we might arrive at the a threshold on the anomaly scores and investigate the outlier suspects.


```R
# quantiles of anomaly scores
quantile(scores_unseen$anomaly_score, probs = seq(0.5, 1, length.out = 11))
```


<dl class=dl-horizontal>
	<dt>50%</dt>
		<dd>0.558888919236839</dd>
	<dt>55%</dt>
		<dd>0.559801479288359</dd>
	<dt>60%</dt>
		<dd>0.561704461232994</dd>
	<dt>65%</dt>
		<dd>0.563301341591528</dd>
	<dt>70%</dt>
		<dd>0.565308396267705</dd>
	<dt>75%</dt>
		<dd>0.567526301777609</dd>
	<dt>80%</dt>
		<dd>0.571242164771264</dd>
	<dt>85%</dt>
		<dd>0.579389692080076</dd>
	<dt>90%</dt>
		<dd>0.588997055131807</dd>
	<dt>95%</dt>
		<dd>0.601272472611482</dd>
	<dt>100%</dt>
		<dd>0.645388124164687</dd>
</dl>



The understanding of why is an observation an anomaly might require a combination of domain understanding and techniques like lime (Local Interpretable Model-agnostic Explanations), Rule based systems etc

# Clustering Based Outlier Detection Technique

In this section we will discuss about the k-means algorithm for detecting the outliers.

In the k-means based outlier detection technique the data are partitioned in to k groups by assigning them to the closest cluster centers.

Once assigned we can compute the distance or dissimilarity between each object and its cluster center, and pick those with largest distances as outliers.

An anomaly score is computed by the distance of each instance to its cluster center multiplied by the instances belonging to its cluster. 

Here we will look in to an example to illustrate the k-means technique to detect the outlier using the Iris data set as we used to illustrate the proximity based outlier detection technique.

From the Iris data set create a subset in R using the following command.


```R
iris2 <- iris[,1:4]
```

On this subset of data perform a k-means cluster using the kmeans() function with k=3


```R
kmeans.result <- kmeans(iris2, centers=3)
```

To view the results of the k-means cluster type the following command


```R
kmeans.result$centers
```


<table>
<thead><tr><th scope=col>Sepal.Length</th><th scope=col>Sepal.Width</th><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th></tr></thead>
<tbody>
	<tr><td>6.850000</td><td>3.073684</td><td>5.742105</td><td>2.071053</td></tr>
	<tr><td>5.901613</td><td>2.748387</td><td>4.393548</td><td>1.433871</td></tr>
	<tr><td>5.006000</td><td>3.428000</td><td>1.462000</td><td>0.246000</td></tr>
</tbody>
</table>



To obtain the cluster ID’s type the following command


```R
kmeans.result$cluster
```


<ol class=list-inline>
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
	<li>2</li>
	<li>2</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>2</li>
	<li>1</li>
	<li>2</li>
	<li>1</li>
	<li>2</li>
	<li>1</li>
	<li>1</li>
	<li>2</li>
	<li>2</li>
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
	<li>2</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>2</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
	<li>2</li>
	<li>1</li>
	<li>1</li>
	<li>2</li>
</ol>



In the next step we will calculate the distance between the objects and cluster centers to determine the outliers and identify 5 largest distances which are outliers. Finally we will print the five outliers.

The R commands for the following steps are provided below:


```R
centers <- kmeans.result$centers[kmeans.result$cluster, ] # "centers" is a data frame of 3 centers but the length of iris dataset so we can canlculate distance difference easily.

distances <- sqrt(rowSums((iris2 - centers)^2))

outliers <- order(distances, decreasing=T)[1:5]

print(outliers) # these rows are 5 top outliers
```

    [1]  99  58  94  61 119
    

To print the details about the outliers use the following command


```R
print(iris2[outliers,])
```

        Sepal.Length Sepal.Width Petal.Length Petal.Width
    99           5.1         2.5          3.0         1.1
    58           4.9         2.4          3.3         1.0
    94           5.0         2.3          3.3         1.0
    61           5.0         2.0          3.5         1.0
    119          7.7         2.6          6.9         2.3
    

Using the following commands provided below you should be able to plot the clusters with the “+” representing the outliers and the asterisks “*" representing the cluster center.


```R
plot(iris2[,c("Sepal.Length", "Sepal.Width")], pch=19, col=kmeans.result$cluster, cex=1)

points(kmeans.result$centers[,c("Sepal.Length", "Sepal.Width")], col=1:3, pch=15, cex=2)

points(iris2[outliers, c("Sepal.Length", "Sepal.Width")], pch="+", col=4, cex=3)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Anomaly%20Detection/output_56_0.png)


# Feature Bagging
Feature bagging approach combines results
from multiple outlier detection algorithms that are applied using different set of features. 

A feature bagging detector fits a number of base detectors on various sub-samples of the dataset.

It uses averaging or other combination methods to improve the prediction accuracy. This brings out the diversity of base estimators.


```R
library('HighDimOut')
```

    Warning message:
    "package 'HighDimOut' was built under R version 3.6.3"


```R
data(iris)
```


```R
data.temp <- iris[,1:4]
```


```R
head(data.temp)
```


<table>
<thead><tr><th scope=col>Sepal.Length</th><th scope=col>Sepal.Width</th><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th></tr></thead>
<tbody>
	<tr><td>5.1</td><td>3.5</td><td>1.4</td><td>0.2</td></tr>
	<tr><td>4.9</td><td>3.0</td><td>1.4</td><td>0.2</td></tr>
	<tr><td>4.7</td><td>3.2</td><td>1.3</td><td>0.2</td></tr>
	<tr><td>4.6</td><td>3.1</td><td>1.5</td><td>0.2</td></tr>
	<tr><td>5.0</td><td>3.6</td><td>1.4</td><td>0.2</td></tr>
	<tr><td>5.4</td><td>3.9</td><td>1.7</td><td>0.4</td></tr>
</tbody>
</table>




```R
library(ggplot2)
res.FBOD <- Func.FBOD(data = iris[,1:4], iter=10, k.nn=5)
data.temp$Ind <- NA
data.temp[order(res.FBOD, decreasing = TRUE)[1:10],"Ind"] <- "Outlier"
data.temp[is.na(data.temp$Ind),"Ind"] <- "Inlier"
data.temp$Ind <- factor(data.temp$Ind)
ggplot(data = data.temp) + geom_point(aes(x = Sepal.Length, y = Sepal.Width, color=Ind, shape=Ind))
```

    Warning message:
    "executing %dopar% sequentially: no parallel backend registered"Registered S3 method overwritten by 'xts':
      method     from
      as.zoo.xts zoo 
    Registered S3 method overwritten by 'quantmod':
      method            from
      as.zoo.data.frame zoo 
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Anomaly%20Detection/output_62_1.png)



```R
head(data.temp,50)
```


<table>
<thead><tr><th scope=col>Sepal.Length</th><th scope=col>Sepal.Width</th><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th><th scope=col>Ind</th></tr></thead>
<tbody>
	<tr><td>5.1    </td><td>3.5    </td><td>1.4    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>4.9    </td><td>3.0    </td><td>1.4    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>4.7    </td><td>3.2    </td><td>1.3    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>4.6    </td><td>3.1    </td><td>1.5    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>5.0    </td><td>3.6    </td><td>1.4    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>5.4    </td><td>3.9    </td><td>1.7    </td><td>0.4    </td><td>Inlier </td></tr>
	<tr><td>4.6    </td><td>3.4    </td><td>1.4    </td><td>0.3    </td><td>Inlier </td></tr>
	<tr><td>5.0    </td><td>3.4    </td><td>1.5    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>4.4    </td><td>2.9    </td><td>1.4    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>4.9    </td><td>3.1    </td><td>1.5    </td><td>0.1    </td><td>Inlier </td></tr>
	<tr><td>5.4    </td><td>3.7    </td><td>1.5    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>4.8    </td><td>3.4    </td><td>1.6    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>4.8    </td><td>3.0    </td><td>1.4    </td><td>0.1    </td><td>Inlier </td></tr>
	<tr><td>4.3    </td><td>3.0    </td><td>1.1    </td><td>0.1    </td><td>Inlier </td></tr>
	<tr><td>5.8    </td><td>4.0    </td><td>1.2    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>5.7    </td><td>4.4    </td><td>1.5    </td><td>0.4    </td><td>Inlier </td></tr>
	<tr><td>5.4    </td><td>3.9    </td><td>1.3    </td><td>0.4    </td><td>Inlier </td></tr>
	<tr><td>5.1    </td><td>3.5    </td><td>1.4    </td><td>0.3    </td><td>Inlier </td></tr>
	<tr><td>5.7    </td><td>3.8    </td><td>1.7    </td><td>0.3    </td><td>Inlier </td></tr>
	<tr><td>5.1    </td><td>3.8    </td><td>1.5    </td><td>0.3    </td><td>Inlier </td></tr>
	<tr><td>5.4    </td><td>3.4    </td><td>1.7    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>5.1    </td><td>3.7    </td><td>1.5    </td><td>0.4    </td><td>Inlier </td></tr>
	<tr><td>4.6    </td><td>3.6    </td><td>1.0    </td><td>0.2    </td><td>Outlier</td></tr>
	<tr><td>5.1    </td><td>3.3    </td><td>1.7    </td><td>0.5    </td><td>Inlier </td></tr>
	<tr><td>4.8    </td><td>3.4    </td><td>1.9    </td><td>0.2    </td><td>Outlier</td></tr>
	<tr><td>5.0    </td><td>3.0    </td><td>1.6    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>5.0    </td><td>3.4    </td><td>1.6    </td><td>0.4    </td><td>Inlier </td></tr>
	<tr><td>5.2    </td><td>3.5    </td><td>1.5    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>5.2    </td><td>3.4    </td><td>1.4    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>4.7    </td><td>3.2    </td><td>1.6    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>4.8    </td><td>3.1    </td><td>1.6    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>5.4    </td><td>3.4    </td><td>1.5    </td><td>0.4    </td><td>Inlier </td></tr>
	<tr><td>5.2    </td><td>4.1    </td><td>1.5    </td><td>0.1    </td><td>Outlier</td></tr>
	<tr><td>5.5    </td><td>4.2    </td><td>1.4    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>4.9    </td><td>3.1    </td><td>1.5    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>5.0    </td><td>3.2    </td><td>1.2    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>5.5    </td><td>3.5    </td><td>1.3    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>4.9    </td><td>3.6    </td><td>1.4    </td><td>0.1    </td><td>Outlier</td></tr>
	<tr><td>4.4    </td><td>3.0    </td><td>1.3    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>5.1    </td><td>3.4    </td><td>1.5    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>5.0    </td><td>3.5    </td><td>1.3    </td><td>0.3    </td><td>Inlier </td></tr>
	<tr><td>4.5    </td><td>2.3    </td><td>1.3    </td><td>0.3    </td><td>Outlier</td></tr>
	<tr><td>4.4    </td><td>3.2    </td><td>1.3    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>5.0    </td><td>3.5    </td><td>1.6    </td><td>0.6    </td><td>Inlier </td></tr>
	<tr><td>5.1    </td><td>3.8    </td><td>1.9    </td><td>0.4    </td><td>Inlier </td></tr>
	<tr><td>4.8    </td><td>3.0    </td><td>1.4    </td><td>0.3    </td><td>Inlier </td></tr>
	<tr><td>5.1    </td><td>3.8    </td><td>1.6    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>4.6    </td><td>3.2    </td><td>1.4    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>5.3    </td><td>3.7    </td><td>1.5    </td><td>0.2    </td><td>Inlier </td></tr>
	<tr><td>5.0    </td><td>3.3    </td><td>1.4    </td><td>0.2    </td><td>Inlier </td></tr>
</tbody>
</table>



# DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
DBSCAN algorithm groups together the points that are closely packed together and marks the low density points far apart as outliers.

#### The method, step-by-step:
1. Randomly select a point not already assigned to a cluster or designated as an outlier. Determine if it’s a core point by seeing if there are at least min_samples points around it within epsilon distance.
2. Create a cluster of this core point and all points within epsilon distance of it (all directly reachable points).
3. Find all points that are within epsilon distance of each point in the cluster and add them to the cluster. Find all points that are within epsilon distance of all newly added points and add these to the cluster. Rinse and repeat. (i.e. perform “neighborhood jumps” to find all density-reachable points and add them to the cluster).


```R
library(dbscan)
library(factoextra)
```

    Warning message:
    "package 'dbscan' was built under R version 3.6.3"Warning message:
    "package 'factoextra' was built under R version 3.6.3"Loading required package: ggplot2
    Warning message:
    "package 'ggplot2' was built under R version 3.6.3"Welcome! Want to learn more? See two factoextra-related books at https://goo.gl/ve3WBa
    


```R
# load and prepare the data
data(iris)
data.temp <- iris[,1:4]
```


```R
# plot the distribution of distances to the fifth nearest neighbors 
kNNdistplot(data.temp, k = 5)
abline(h = 0.4, col = "red")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Anomaly%20Detection/output_68_0.png)



```R
# find clusters
db_clusters <- dbscan(data.temp, eps=0.4, minPts=5)
print(db_clusters)
```

    DBSCAN clustering for 150 objects.
    Parameters: eps = 0.4, minPts = 5
    The clustering contains 4 cluster(s) and 32 noise points.
    
     0  1  2  3  4 
    32 46 36 14 22 
    
    Available fields: cluster, eps, minPts
    


```R
fviz_cluster(db_clusters, data.temp, ellipse = FALSE, geom = "point")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Anomaly%20Detection/output_70_0.png)


# KNN
KNN algorithm identified anomalies based on three approaches:

Largest: Uses the distance of the kth neighbor as the outlier score
Mean: Uses the average of all k neighbors as the outlier score
Median: Uses the median of the distance to k neighbors as the outlier score


```R
library(adamethods)
X=iris[,1:4]
outl <- do_knno(X, 3,5)
outl
data[outl,]
```

    Warning message:
    "package 'adamethods' was built under R version 3.6.3"


<ol class=list-inline>
	<li>132</li>
	<li>119</li>
	<li>118</li>
	<li>107</li>
	<li>42</li>
</ol>




<table>
<thead><tr><th scope=col>Sepal.Length</th><th scope=col>Sepal.Width</th><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th></tr></thead>
<tbody>
	<tr><td>7.9</td><td>3.8</td><td>6.4</td><td>2.0</td></tr>
	<tr><td>7.7</td><td>2.6</td><td>6.9</td><td>2.3</td></tr>
	<tr><td>7.7</td><td>3.8</td><td>6.7</td><td>2.2</td></tr>
	<tr><td>4.9</td><td>2.5</td><td>4.5</td><td>1.7</td></tr>
	<tr><td>4.5</td><td>2.3</td><td>1.3</td><td>0.3</td></tr>
</tbody>
</table>




```R
library(OutlierDetection)
X=iris[,1:4]
nn(X,k=5)
```

    Warning message:
    "package 'OutlierDetection' was built under R version 3.6.3"Registered S3 method overwritten by 'spatstat':
      method     from
      print.boxx cli 
    Warning message:
    "`line.width` does not currently support multiple values."Warning message:
    "`line.width` does not currently support multiple values."Warning message:
    "`line.width` does not currently support multiple values."Warning message:
    "`line.width` does not currently support multiple values."


<dl>
	<dt>$`Outlier Observations`</dt>
		<dd><table>
<thead><tr><th></th><th scope=col>Sepal.Length</th><th scope=col>Sepal.Width</th><th scope=col>Petal.Length</th><th scope=col>Petal.Width</th></tr></thead>
<tbody>
	<tr><th scope=row>42</th><td>4.5</td><td>2.3</td><td>1.3</td><td>0.3</td></tr>
	<tr><th scope=row>107</th><td>4.9</td><td>2.5</td><td>4.5</td><td>1.7</td></tr>
	<tr><th scope=row>118</th><td>7.7</td><td>3.8</td><td>6.7</td><td>2.2</td></tr>
	<tr><th scope=row>132</th><td>7.9</td><td>3.8</td><td>6.4</td><td>2.0</td></tr>
</tbody>
</table>
</dd>
	<dt>$`Location of Outlier`</dt>
		<dd><ol class=list-inline>
	<li>42</li>
	<li>107</li>
	<li>118</li>
	<li>132</li>
</ol>
</dd>
	<dt>$`Outlier Probability`</dt>
		<dd><ol class=list-inline>
	<li>0.96</li>
	<li>1</li>
	<li>1</li>
	<li>1</li>
</ol>
</dd>
	<dt>$`3Dplot`</dt>
		<dd><!doctype html>
</dd>
</dl>



# HBOS
HBOS is histogram-based anomaly detection algorithm.

HBOS assumes the feature independence and calculates the degree of anomalies by building histograms.

In multivariate anomaly detection, a histogram for each single feature can be computed, scored individually and combined at the end.

Usecase: Best choice for anomaly detection in computer networks due to its low computational time


```R
#Histogram based approach using IQR method
X=iris[,1:4]
hist(X$Sepal.Width, breaks = nclass.scottRob)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Anomaly%20Detection/output_75_0.png)



```R
trim_t <- function(x){
  x[(x > quantile(x, 0.25)-1.5*IQR(x)) & (x < quantile(x, 0.75)+1.5*IQR(x))]
}

hist(trim_t(X$Sepal.Width),breaks = nclass.scottRob)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Anomaly%20Detection/output_76_0.png)



```R
boxplot(X$Sepal.Width,data=X, main="Values of Sepal Width")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Anomaly%20Detection/output_77_0.png)

