## Text Classification

# Notebook Content
## Text Classification
[Data Preparation](#Preparing-Data-for-Modeling)<br>
[Logistic Regression Classifier](#Logistic-Regression)<br>
[Support Vector Classifier](#1)<br>
[Gradient Boosting Classifier](#2)<br>
[Decision Tree](#3)<br>
[Random Forest Classifier](#4)<br>
[KNN Classifier](#5)<br>
[Gradient Descent](#6)<br>
[Stochastic Gradient Descent](#7)<br>
[XGBoost Classifier](#8)<br>

## Text Classification
The domain of analytics that addresses how computers understand text is called Natural Language Processing (NLP). NLP has multiple applications like sentiment analysis, chatbots, AI agents, social media analytics, as well as text classification. In this guide, you will learn how to build a supervised machine learning model on text data, using the popular statistical programming language, 'R'.

## Library


```python
# install.packages("kernlab")
# install.packages("caret")
# install.packages("tm")
# install.packages("dplyr")
# install.packages("splitstackshape")
# install.packages("e1071")
# install.packages("SnowballC")
# install.packages("rattle")
# install.packages("gmodels")
# install.packages('randomForest')
```

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Text-Classification" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Data
The data we’ll be using in this guide comes from Kaggle, a machine learning competition website. This is a women's clothing e-commerce data, consisting of the reviews written by the customers. In this guide, we will take up the task of predicting whether the customer will recommend the product or not. In this guide, we are taking a sample of the original dataset. The sampled data contains 500 rows and three variables, as described below: 1. Clothing ID: This is the unique ID. 2. Review Text: Text containing reviews by the customer. 3. Recommended IND: Binary variable stating where the customer recommends the product ("1") or not ("0"). This is the target variable. Let us start by loading the required libraries and the data.


```python
library(readr)
library(dplyr)
 
#Text mining packages
library(tm)
library(SnowballC)
options(warn=-1) #To supress waarning messages

#loading the data
t1 <- read.csv('dataset/Womens Clothing E-Commerce Reviews.csv')
glimpse(t1) 
```

    Warning message:
    "package 'readr' was built under R version 3.6.3"
    Warning message:
    "package 'SnowballC' was built under R version 3.6.3"
    

    Rows: 23,486
    Columns: 11
    $ X                       [3m[90m<int>[39m[23m 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1...
    $ Clothing.ID             [3m[90m<int>[39m[23m 767, 1080, 1077, 1049, 847, 1080, 858, 858,...
    $ Age                     [3m[90m<int>[39m[23m 33, 34, 60, 50, 47, 49, 39, 39, 24, 34, 53,...
    $ Title                   [3m[90m<fct>[39m[23m "", "", "Some major design flaws", "My favo...
    $ Review.Text             [3m[90m<fct>[39m[23m "Absolutely wonderful - silky and sexy and ...
    $ Rating                  [3m[90m<int>[39m[23m 4, 5, 3, 5, 5, 2, 5, 4, 5, 5, 3, 5, 5, 5, 3...
    $ Recommended.IND         [3m[90m<int>[39m[23m 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1...
    $ Positive.Feedback.Count [3m[90m<int>[39m[23m 0, 4, 0, 0, 6, 4, 1, 4, 0, 0, 14, 2, 2, 0, ...
    $ Division.Name           [3m[90m<fct>[39m[23m Initmates, General, General, General Petite...
    $ Department.Name         [3m[90m<fct>[39m[23m Intimate, Dresses, Dresses, Bottoms, Tops, ...
    $ Class.Name              [3m[90m<fct>[39m[23m Intimates, Dresses, Dresses, Pants, Blouses...
    

The above output shows that the data has other variables, but the important ones are the variables 'Review_Text', and 'Recommended_IND'.

## Preparing Data for Modeling
Since the text data is not in the traditional format of observations in rows, and variables in columns, we will have to perform certain text-specific steps. The list of such steps is discussed in the subsequent sections.

**Step 1 - Create the Text Corpus**<br>
The variable containing text needs to be converted to a corpus for preprocessing. A corpus is a collection of documents. The first line of code below performs this task. The second line prints the content of the first corpus, while the third line prints the corresponding recommendation score.


```python
corpuss = Corpus(VectorSource(t1$Review.Text))
corpuss[[1]][1]
t1$Recommended.IND[1]
```


<strong>$content</strong> = 'Absolutely wonderful - silky and sexy and comfortable'



1


Looking at the review text, it is obvious that the customer not happy with the product, and hence gave the recommendation score of one.

**Step 2 - Conversion to Lowercase**<br>
The model needs to treat Words like 'soft' and 'Soft' as same. Hence, all the words are converted to lowercase with the lines of code below.


```python
corpuss = tm_map(corpuss, PlainTextDocument)
corpuss = tm_map(corpuss, tolower)
corpuss[[1]][1] 
```


<strong>$content</strong> = 'absolutely wonderful - silky and sexy and comfortable'


**Step 3 - Removing Punctuation**<br>
The idea here is to remove everything that isn't a standard number or letter.


```python
corpuss = tm_map(corpuss, removePunctuation)
corpuss[[1]][1]
```


<strong>$content</strong> = <span style=white-space:pre-wrap>'absolutely wonderful  silky and sexy and comfortable'</span>


**Step 4 - Removing Stopwords**<br>
Stopwords are unhelpful words like 'i', 'is', 'at', 'me', 'our'. These are not helpful because the frequency of such stopwords is high in the corpus, but they don't help in differentiating the target classes. The removal of Stopwords is therefore important.

The line of code below uses the tm_map function on the 'corpus' and removes stopwords, as well as the word 'cloth'. The word 'cloth' is removed because this dataset is on clothing review, so this word will not add any predictive power to the model.


```python
corpuss = tm_map(corpuss, stemDocument)
corpuss[[1]][1]
```


<strong>$content</strong> = 'absolut wonder silki and sexi and comfort'


**Step 5 - Stemming**<br>
The idea behind stemming is to reduce the number of inflectional forms of words appearing in the text. For example, words such as "argue", "argued", "arguing", "argues" are reduced to their common stem "argu". This helps in decreasing the size of the vocabulary space. The lines of code below perform the stemming on the corpus.

**Create Document Term Matrix**<br>
The most commonly used text preprocessing steps are complete. Now we are ready to extract the word frequencies, which will be used as features in our prediction problem. The line of code below uses the function called DocumentTermMatrix from the tm package and generates a matrix. The rows in the matrix correspond to the documents, in our case reviews, and the columns correspond to words in those reviews. The values in the matrix are the frequency of the word across the document.


```python
frequencies = DocumentTermMatrix(corpuss)
```

The above command results in a matrix that contains zeroes in many of the cells, a problem called sparsity. It is advisable to remove such words that have a lot of zeroes across the documents. The following lines of code perform this task.


```python
sparse = removeSparseTerms(frequencies, 0.995)
```

The final data preparation step is to convert the matrix into a data frame, a format widely used in 'R' for predictive modeling. The first line of code below converts the matrix into dataframe, called 'tSparse'. The second line makes all the variable names R-friendly, while the third line of code adds the dependent variable to the data set.


```python
tSparse = as.data.frame(as.matrix(sparse))
colnames(tSparse) = make.names(colnames(tSparse))
tSparse$recommended_id = t1$Recommended.IND
```

Now we are ready for building the predictive model. But before that, it is always a good idea to set the baseline accuracy of the model. The baseline accuracy, in the case of a classification problem, is the proportion of the majority label in the target variable. The line of code below prints the proportion of the labels in the target variable, 'recommended_id'.


```python
prop.table(table(tSparse$recommended_id))
```


    
            0         1 
    0.1776377 0.8223623 


The above output shows that 82.23 percent of the reviews are from customers who recommended the product. This becomes the baseline accuracy for predictive modeling.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Text-Classification" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Creating Training and Test Data for Machine Learning
For evaluating how the predictive model is performing, we will divide the data into training and test data. The first line of code below loads the caTools package, which will be used for creating the training and test data. The second line sets the 'random seed' so that the results are reproducible.
The third line creates the data partition in the manner that it keeps 70% of the data for training the model. The fourth and fifth lines of code create the training ('trainSparse') and testing ('testSparse') dataset.


```python
library(caTools)
set.seed(100)
split = sample.split(tSparse$recommended_id, SplitRatio = 0.7)
trainSparse = subset(tSparse, split==TRUE)
testSparse = subset(tSparse, split==FALSE)
```

You can train you model on all the rows of the dataset, we have subsetted here to reduce the training time.


```python
trainSparse = trainSparse[1:1000,]
testSparse = testSparse[1:400,]
```


```python
set.seed(100)
trainSparse$recommended_id = as.factor(trainSparse$recommended_id)
testSparse$recommended_id = as.factor(testSparse$recommended_id)
```

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Text-Classification" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Logistic Regression
In this algorithm, the probabilities describing the possible outcomes of a single trial are modelled using a logistic function.
It is used to model the probability of a certain class or event existing 
such as pass/fail, win/lose, alive/dead or healthy/sick.


```python
glmnet_classifier = glm(recommended_id ~ ., data=trainSparse,
    family = 'binomial')
```


```python
predictLR = predict(glmnet_classifier, newdata=testSparse)
predictions = ifelse(predictLR >.5, 1, 0)
table(testSparse$recommended_id, predictions)
```


       predictions
          0   1
      0  44  29
      1 111 216



```python
#confusion matrix
library(e1071)
library(caret)
confusionMatrix(as.factor(predictions), testSparse$recommended_id)
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction   0   1
             0  44 111
             1  29 216
                                             
                   Accuracy : 0.65           
                     95% CI : (0.601, 0.6967)
        No Information Rate : 0.8175         
        P-Value [Acc > NIR] : 1              
                                             
                      Kappa : 0.1833         
                                             
     Mcnemar's Test P-Value : 7.608e-12      
                                             
                Sensitivity : 0.6027         
                Specificity : 0.6606         
             Pos Pred Value : 0.2839         
             Neg Pred Value : 0.8816         
                 Prevalence : 0.1825         
             Detection Rate : 0.1100         
       Detection Prevalence : 0.3875         
          Balanced Accuracy : 0.6316         
                                             
           'Positive' Class : 0              
                                             


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Text-Classification" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

<a id="1"></a>

## Support Vector Classifier
The support vector machine is a classifier that represents the training data as points in space separated into categories by a gap as wide as possible. 
New points are then added to space by predicting which category they fall into and which space they will belong to.

More often text classification use cases will have linearly separable data and LinearSVC is apt for such scenarios


```python
#Load Library
library(e1071)

# Fitting the model
model_svm = svm(recommended_id ~ ., data=trainSparse)

# Making 
predictions = predict(model_svm, newdata=testSparse, method="C-classification", kernal="radial", 
          gamma=0.1, cost=10)
```


```python
predictions = ifelse(predictLR >.5, 1, 0)
table(predictions, testSparse$recommended_id)
```


               
    predictions   0   1
              0  44 111
              1  29 216



```python
#confusion matrix
library(e1071)
library(caret)
confusionMatrix(as.factor(predictions), testSparse$recommended_id)
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction   0   1
             0  44 111
             1  29 216
                                             
                   Accuracy : 0.65           
                     95% CI : (0.601, 0.6967)
        No Information Rate : 0.8175         
        P-Value [Acc > NIR] : 1              
                                             
                      Kappa : 0.1833         
                                             
     Mcnemar's Test P-Value : 7.608e-12      
                                             
                Sensitivity : 0.6027         
                Specificity : 0.6606         
             Pos Pred Value : 0.2839         
             Neg Pred Value : 0.8816         
                 Prevalence : 0.1825         
             Detection Rate : 0.1100         
       Detection Prevalence : 0.3875         
          Balanced Accuracy : 0.6316         
                                             
           'Positive' Class : 0              
                                             


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Text-Classification" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

<a id="2"></a>

## Gradient Boosting Classifier
GB builds an additive model in a forward stage-wise fashion
It allows for the optimization of arbitrary differentiable loss functions. 
Binary classification is a special case where only a single regression tree is induced.


```python
# Load the required libraries
library(gbm)

# Fit the model
model_gbm = gbm(recommended_id ~ ., data=trainSparse,  distribution = "gaussian",n.trees = 50,
                  shrinkage = 0.01, interaction.depth = 4)

#variance importance plt
summary(model_gbm)
```

    Loaded gbm 2.1.5
    
    


<table>
<caption>A data.frame: 815 × 2</caption>
<thead>
	<tr><th></th><th scope=col>var</th><th scope=col>rel.inf</th></tr>
	<tr><th></th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>was</th><td>was       </td><td>25.4493493</td></tr>
	<tr><th scope=row>return</th><td>return    </td><td>16.9791012</td></tr>
	<tr><th scope=row>back</th><td>back      </td><td>11.1731467</td></tr>
	<tr><th scope=row>disappoint</th><td>disappoint</td><td>10.9386761</td></tr>
	<tr><th scope=row>unfortun</th><td>unfortun  </td><td> 6.3792747</td></tr>
	<tr><th scope=row>with</th><td>with      </td><td> 5.1806575</td></tr>
	<tr><th scope=row>wide</th><td>wide      </td><td> 3.2389267</td></tr>
	<tr><th scope=row>size</th><td>size      </td><td> 2.0544616</td></tr>
	<tr><th scope=row>thought</th><td>thought   </td><td> 1.8285234</td></tr>
	<tr><th scope=row>wear</th><td>wear      </td><td> 1.5252181</td></tr>
	<tr><th scope=row>littl</th><td>littl     </td><td> 1.4774547</td></tr>
	<tr><th scope=row>the</th><td>the       </td><td> 1.4560079</td></tr>
	<tr><th scope=row>howev</th><td>howev     </td><td> 1.1137237</td></tr>
	<tr><th scope=row>cut</th><td>cut       </td><td> 1.0466238</td></tr>
	<tr><th scope=row>excit</th><td>excit     </td><td> 0.7743964</td></tr>
	<tr><th scope=row>odd</th><td>odd       </td><td> 0.7383801</td></tr>
	<tr><th scope=row>want</th><td>want      </td><td> 0.6333873</td></tr>
	<tr><th scope=row>top</th><td>top       </td><td> 0.5909084</td></tr>
	<tr><th scope=row>perfect</th><td>perfect   </td><td> 0.5567859</td></tr>
	<tr><th scope=row>thin</th><td>thin      </td><td> 0.4849028</td></tr>
	<tr><th scope=row>appear</th><td>appear    </td><td> 0.4833759</td></tr>
	<tr><th scope=row>huge</th><td>huge      </td><td> 0.4391831</td></tr>
	<tr><th scope=row>realli</th><td>realli    </td><td> 0.4173475</td></tr>
	<tr><th scope=row>were</th><td>were      </td><td> 0.3925811</td></tr>
	<tr><th scope=row>just</th><td>just      </td><td> 0.3909910</td></tr>
	<tr><th scope=row>especi</th><td>especi    </td><td> 0.3663345</td></tr>
	<tr><th scope=row>would</th><td>would     </td><td> 0.3656446</td></tr>
	<tr><th scope=row>shorter</th><td>shorter   </td><td> 0.3452103</td></tr>
	<tr><th scope=row>made</th><td>made      </td><td> 0.3384843</td></tr>
	<tr><th scope=row>didnt</th><td>didnt     </td><td> 0.3196429</td></tr>
	<tr><th scope=row>...</th><td>...</td><td>...</td></tr>
	<tr><th scope=row>build</th><td>build  </td><td>0</td></tr>
	<tr><th scope=row>excel</th><td>excel  </td><td>0</td></tr>
	<tr><th scope=row>attract</th><td>attract</td><td>0</td></tr>
	<tr><th scope=row>mid</th><td>mid    </td><td>0</td></tr>
	<tr><th scope=row>bag</th><td>bag    </td><td>0</td></tr>
	<tr><th scope=row>complet</th><td>complet</td><td>0</td></tr>
	<tr><th scope=row>booti</th><td>booti  </td><td>0</td></tr>
	<tr><th scope=row>husband</th><td>husband</td><td>0</td></tr>
	<tr><th scope=row>classi</th><td>classi </td><td>0</td></tr>
	<tr><th scope=row>offer</th><td>offer  </td><td>0</td></tr>
	<tr><th scope=row>stomach</th><td>stomach</td><td>0</td></tr>
	<tr><th scope=row>often</th><td>often  </td><td>0</td></tr>
	<tr><th scope=row>relax</th><td>relax  </td><td>0</td></tr>
	<tr><th scope=row>deal</th><td>deal   </td><td>0</td></tr>
	<tr><th scope=row>mix</th><td>mix    </td><td>0</td></tr>
	<tr><th scope=row>togeth</th><td>togeth </td><td>0</td></tr>
	<tr><th scope=row>meant</th><td>meant  </td><td>0</td></tr>
	<tr><th scope=row>glove</th><td>glove  </td><td>0</td></tr>
	<tr><th scope=row>beach</th><td>beach  </td><td>0</td></tr>
	<tr><th scope=row>shop</th><td>shop   </td><td>0</td></tr>
	<tr><th scope=row>everyon</th><td>everyon</td><td>0</td></tr>
	<tr><th scope=row>becom</th><td>becom  </td><td>0</td></tr>
	<tr><th scope=row>expens</th><td>expens </td><td>0</td></tr>
	<tr><th scope=row>frumpi</th><td>frumpi </td><td>0</td></tr>
	<tr><th scope=row>money</th><td>money  </td><td>0</td></tr>
	<tr><th scope=row>suggest</th><td>suggest</td><td>0</td></tr>
	<tr><th scope=row>requir</th><td>requir </td><td>0</td></tr>
	<tr><th scope=row>silk</th><td>silk   </td><td>0</td></tr>
	<tr><th scope=row>purpl</th><td>purpl  </td><td>0</td></tr>
	<tr><th scope=row>swingi</th><td>swingi </td><td>0</td></tr>
</tbody>
</table>




![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Text%20Classification/output_48_2.png)



```python
#Generating a Prediction matrix for each Tree
predictions <- predict(model_gbm, testSparse, n.trees = 50, type="response")
```


```python
predictions = ifelse(predictions >1.78, 1, 0)
table(predictions, testSparse$recommended_id)
```


               
    predictions   0   1
              0  48  51
              1  25 276



```python
#confusion matrix
library(e1071)
library(caret)
confusionMatrix(as.factor(predictions), testSparse$recommended_id)
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction   0   1
             0  48  51
             1  25 276
                                              
                   Accuracy : 0.81            
                     95% CI : (0.7681, 0.8473)
        No Information Rate : 0.8175          
        P-Value [Acc > NIR] : 0.678628        
                                              
                      Kappa : 0.4406          
                                              
     Mcnemar's Test P-Value : 0.004135        
                                              
                Sensitivity : 0.6575          
                Specificity : 0.8440          
             Pos Pred Value : 0.4848          
             Neg Pred Value : 0.9169          
                 Prevalence : 0.1825          
             Detection Rate : 0.1200          
       Detection Prevalence : 0.2475          
          Balanced Accuracy : 0.7508          
                                              
           'Positive' Class : 0               
                                              


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Text-Classification" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

<a id="3"></a>

## Decision Tree
The decision tree algorithm builds the classification model in the form of a tree structure. 
It utilizes the if-then rules which are equally exhaustive and mutually exclusive in classification.

We can use decision tree when there are missing values in the data and when pre processing time is to be reduced as it does not require pre processing


```python
#Loading libraries
library(rpart,quietly = TRUE)
#library(caret,quietly = TRUE)
library(rpart.plot,quietly = TRUE)
library(rattle)

#building the classification tree with rpart
library(rpart)
model_dt <- rpart(recommended_id ~ ., data=trainSparse,
                method = "class")
```

    Loading required package: tibble
    
    Loading required package: bitops
    
    Rattle: A free graphical interface for data science with R.
    Version 5.4.0 Copyright (c) 2006-2020 Togaware Pty Ltd.
    Type 'rattle()' to shake, rattle, and roll your data.
    
    


```python
# Visualize the decision tree with rpart.plot
library(rpart.plot)
rpart.plot(model_dt, nn=TRUE)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Text%20Classification/output_56_0.png)



```python
#Testing the model
predictions <- predict(model_dt, testSparse, type="class")
```


```python
table(predictions, testSparse$recommended_id)
```


               
    predictions   0   1
              0  30  23
              1  43 304



```python
#confusion matrix
library(e1071)
library(caret)
confusionMatrix(as.factor(predictions), testSparse$recommended_id)
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction   0   1
             0  30  23
             1  43 304
                                            
                   Accuracy : 0.835         
                     95% CI : (0.7949, 0.87)
        No Information Rate : 0.8175        
        P-Value [Acc > NIR] : 0.20117       
                                            
                      Kappa : 0.3812        
                                            
     Mcnemar's Test P-Value : 0.01935       
                                            
                Sensitivity : 0.4110        
                Specificity : 0.9297        
             Pos Pred Value : 0.5660        
             Neg Pred Value : 0.8761        
                 Prevalence : 0.1825        
             Detection Rate : 0.0750        
       Detection Prevalence : 0.1325        
          Balanced Accuracy : 0.6703        
                                            
           'Positive' Class : 0             
                                            


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Text-Classification" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

<a id="4"></a>

## Random Forest Classifier
Random forest are an ensemble learning method.
It operates by constructing a multitude of decision trees at training time and outputs the class that is the mode of the classes of the individual trees.
A random forest is a meta-estimator that fits a number of trees on various subsamples of data sets and then uses an average to improve the accuracy in the model’s predictive nature.
The sub-sample size is always the same as that of the original input size but the samples are often drawn with replacements.

We should use this algorithm when we need high accuracy while working with large datasets with higher dimensions. We can also use it if there are missing values in the dataset. We should not use it if we have less time for modeling or if large computational costs and memory space are a constrain.


```python
library(randomForest)
options(warn=-1) #To supress waarning messages

RF_model = randomForest(recommended_id ~ ., data=trainSparse)
predictRF = predict(RF_model, newdata=testSparse)
table(testSparse$recommended_id, predictRF)
```

    randomForest 4.6-14
    
    Type rfNews() to see new features/changes/bug fixes.
    
    
    Attaching package: 'randomForest'
    
    
    The following object is masked from 'package:rattle':
    
        importance
    
    
    The following object is masked from 'package:dplyr':
    
        combine
    
    
    The following object is masked from 'package:ggplot2':
    
        margin
    
    
    


       predictRF
          0   1
      0  11  62
      1   0 327


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Text-Classification" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

<a id="5"></a>

## KNN Classifier
It is a lazy learning algorithm that stores all instances corresponding to training data in n-dimensional space
To label a new point, it looks at the labeled points closest to that new point also known as its nearest neighbors

We should use KNN when the dataset is small and speed is a priority (real-time)


```python
library(class)
predictions <- knn(train = trainSparse, test = testSparse, cl = trainSparse$recommended_id, k=10)
```


```python
library(gmodels)
CrossTable(x = testSparse$recommended_id, y = predictions, prop.chisq=FALSE )
```

    
     
       Cell Contents
    |-------------------------|
    |                       N |
    |           N / Row Total |
    |           N / Col Total |
    |         N / Table Total |
    |-------------------------|
    
     
    Total Observations in Table:  400 
    
     
                              | predictions 
    testSparse$recommended_id |         0 |         1 | Row Total | 
    --------------------------|-----------|-----------|-----------|
                            0 |        16 |        57 |        73 | 
                              |     0.219 |     0.781 |     0.182 | 
                              |     0.696 |     0.151 |           | 
                              |     0.040 |     0.142 |           | 
    --------------------------|-----------|-----------|-----------|
                            1 |         7 |       320 |       327 | 
                              |     0.021 |     0.979 |     0.818 | 
                              |     0.304 |     0.849 |           | 
                              |     0.018 |     0.800 |           | 
    --------------------------|-----------|-----------|-----------|
                 Column Total |        23 |       377 |       400 | 
                              |     0.058 |     0.942 |           | 
    --------------------------|-----------|-----------|-----------|
    
     
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Text-Classification" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

<a id="6"></a>

## Gradient Descent
Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function.


```python
# Loading required libraries
library(gradDescent)
# data = read.csv("pima-indians-diabetes.csv")
#winedata is scaled and then split
featureScalingResult <- varianceScaling(tSparse)
data = tSparse[1:1000,]
splitedDataset <- splitData(data)

# # Fit the train from splitted dataset data to GD model
model <- GD(splitedDataset$dataTrain)
```

    
    Attaching package: 'gradDescent'
    
    
    The following object is masked from 'package:caret':
    
        RMSE
    
    
    


```python
print(model)
```

             [,1]     [,2]     [,3]     [,4]     [,5]      [,6]     [,7]     [,8]
    [1,] 1104.043 50.83314 3011.386 164.8758 20.44298 0.9135144 30.80802 37.50258
             [,9]    [,10]    [,11]    [,12]    [,13]    [,14]    [,15]    [,16]
    [1,] 139.7761 81.95206 92.30288 685.8664 57.40612 30.76642 2.110403 492.1616
            [,17]    [,18]    [,19]   [,20]    [,21]   [,22]    [,23]    [,24]
    [1,] 52.42501 395.7703 67.74398 149.628 232.4159 538.979 23.36495 129.1613
            [,25]    [,26]    [,27]    [,28]    [,29]    [,30]    [,31]    [,32]
    [1,] 274.2111 201.1285 123.2423 32.42281 147.9792 5299.139 1463.896 105.8012
            [,33]    [,34]   [,35]    [,36]    [,37]    [,38]    [,39]    [,40]
    [1,] 2.593996 37.71189 352.128 102.0485 1048.341 24.14679 95.54462 73.61231
            [,41]    [,42]    [,43]    [,44]    [,45]    [,46]    [,47]    [,48]
    [1,] 22.14137 662.1039 11.79398 767.5894 57.93305 185.2428 10.65327 53.31942
            [,49]    [,50]    [,51]    [,52]    [,53]    [,54]    [,55]    [,56]
    [1,] 65.82279 42.03687 52.72252 134.8153 181.3109 611.0357 74.83121 47.19449
            [,57]    [,58]    [,59]    [,60]    [,61]    [,62]    [,63]    [,64]
    [1,] 214.6612 11.87896 10.46854 618.0465 247.3904 15.18645 21.60665 595.9097
            [,65]    [,66]    [,67]    [,68]    [,69]    [,70]    [,71]    [,72]
    [1,] 94.44755 554.7067 48.69954 189.1471 481.2199 126.4163 848.4356 128.3926
            [,73]    [,74]    [,75]   [,76]    [,77]    [,78]    [,79]    [,80]
    [1,] 169.5988 18.82965 28.24258 42.3676 34.08568 6.132093 45.75898 115.6386
            [,81]    [,82]    [,83]    [,84]    [,85]    [,86]    [,87]    [,88]
    [1,] 274.8858 6.941012 23.75453 80.48946 412.1913 8.686665 139.8521 46.16002
            [,89]    [,90]    [,91]    [,92]    [,93]    [,94]    [,95]    [,96]
    [1,] 16.87461 19.16791 204.2492 93.62503 60.63997 43.01751 251.2996 182.5662
            [,97]    [,98]    [,99]   [,100]   [,101]   [,102]   [,103]   [,104]
    [1,] 27.15573 179.9142 673.9082 3.907801 10.43794 19.85283 386.8471 14.10124
           [,105]   [,106]   [,107]   [,108]   [,109]   [,110]   [,111]   [,112]
    [1,] 112.7458 41.37441 2.599214 128.5662 100.9178 11.97339 202.1708 206.6384
           [,113]   [,114]   [,115]   [,116]   [,117]   [,118]   [,119]   [,120]
    [1,] 9.378759 19.50453 68.61307 172.4919 132.3187 38.24464 36.71223 22.59077
           [,121]   [,122]   [,123]   [,124]   [,125]   [,126]   [,127]   [,128]
    [1,] 263.8462 58.40278 31.14916 44.06336 17.26318 25.52396 475.5452 581.3883
           [,129] [,130]   [,131]   [,132]   [,133]   [,134]   [,135]   [,136]
    [1,] 121.1303 140.11 74.31988 8.731397 33.31808 88.32417 94.87911 280.7693
           [,137]   [,138]   [,139]   [,140]   [,141]   [,142]   [,143]   [,144]
    [1,] 22.38683 62.47388 89.97956 18.49022 178.4156 20.47591 442.1273 100.7627
           [,145]   [,146]   [,147]   [,148]   [,149]   [,150]  [,151]   [,152]
    [1,] 203.7193 64.49243 41.49116 71.59467 58.51326 63.12869 135.902 35.70118
            [,153]   [,154]   [,155]   [,156]   [,157]  [,158]   [,159]   [,160]
    [1,] 0.5632669 14.82675 27.42743 59.04552 33.79772 137.831 131.5825 30.72672
           [,161]   [,162]   [,163]   [,164]   [,165]   [,166]   [,167]   [,168]
    [1,] 16.99522 14.08836 15.27548 6.104773 167.3553 368.2127 71.99476 22.64866
           [,169]   [,170]   [,171]   [,172]   [,173]   [,174]   [,175]   [,176]
    [1,] 69.43105 131.7455 11.59194 92.84487 64.86362 69.57288 47.10647 383.7345
           [,177]  [,178]   [,179]   [,180]   [,181]   [,182]  [,183]   [,184]
    [1,] 32.16729 47.0013 45.67617 203.3325 246.4652 62.20649 46.1706 177.3364
           [,185]   [,186]   [,187]   [,188]   [,189]   [,190]   [,191]   [,192]
    [1,] 60.33436 156.8041 134.3097 156.9245 44.09503 56.24415 31.66056 50.03313
           [,193]   [,194]  [,195]   [,196]   [,197]   [,198]  [,199]   [,200]
    [1,] 41.52967 164.8086 97.7251 44.89394 11.78137 85.98205 105.696 95.49443
           [,201]   [,202]  [,203]   [,204]   [,205]   [,206]   [,207]   [,208]
    [1,] 118.6713 25.64367 288.263 62.39263 26.42518 12.71649 11.30017 62.92325
           [,209]   [,210]   [,211]   [,212]   [,213]   [,214]   [,215]   [,216]
    [1,] 27.03028 128.1024 178.6165 95.78342 16.64304 27.01755 258.2355 26.04509
           [,217]   [,218]   [,219]   [,220]   [,221]   [,222]   [,223]   [,224]
    [1,] 83.94741 98.55488 9.518673 30.22853 56.29897 116.9504 28.48618 31.52528
           [,225]   [,226]   [,227]  [,228]   [,229]   [,230]   [,231]   [,232]
    [1,] 15.96398 19.22118 23.37847 24.0277 92.79163 96.59878 13.87271 45.37645
           [,233]   [,234]   [,235]   [,236] [,237]   [,238]   [,239]   [,240]
    [1,] 17.19485 10.51892 16.83735 29.68104 3.4092 33.05352 15.24692 151.8932
           [,241]   [,242]   [,243]   [,244]   [,245]   [,246]   [,247]   [,248]
    [1,] 27.69512 46.25128 46.18136 29.41614 115.3969 11.25419 35.90069 20.82538
           [,249]   [,250]  [,251]   [,252]  [,253]   [,254]   [,255]   [,256]
    [1,] 57.23795 117.7761 19.6785 6.687215 92.7942 145.4716 28.93972 119.9467
           [,257]  [,258]   [,259]   [,260]   [,261]   [,262]   [,263]   [,264]
    [1,] 27.67238 9.33813 28.15884 16.20859 175.8862 95.86174 13.55817 54.45791
           [,265]   [,266]  [,267]   [,268]   [,269]   [,270]   [,271]   [,272]
    [1,] 15.99157 113.6421 18.8989 53.52179 40.70634 21.81285 14.63584 122.3974
           [,273]   [,274]   [,275]   [,276]   [,277]   [,278]  [,279]   [,280]
    [1,] 142.7509 73.27216 46.12622 8.677079 12.43577 34.24581 95.4125 102.3173
           [,281]   [,282]   [,283]   [,284]   [,285]  [,286]   [,287]   [,288]
    [1,] 13.44228 86.10376 2.654133 7.542842 63.18913 9.81492 27.86386 73.86875
           [,289]   [,290]  [,291]   [,292]   [,293]  [,294]  [,295]   [,296]
    [1,] 17.08667 95.97222 15.1248 85.67462 51.88745 7.29218 15.8114 23.50304
           [,297]   [,298]   [,299]   [,300]   [,301]   [,302]   [,303]   [,304]
    [1,] 52.68299 15.52805 9.191301 16.02152 86.97042 24.83264 24.48464 27.81837
           [,305]   [,306]   [,307]   [,308]   [,309]   [,310]   [,311]   [,312]
    [1,] 80.11226 64.75417 8.144297 128.9804 70.67828 11.00475 41.90225 30.72811
           [,313]   [,314]   [,315]   [,316]   [,317]   [,318]   [,319]   [,320]
    [1,] 108.0586 173.2392 56.70787 30.82505 14.29815 121.2751 144.7119 8.665354
           [,321]   [,322]   [,323]   [,324]  [,325]   [,326]   [,327]   [,328]
    [1,] 12.26463 132.8132 21.54034 24.00687 55.4782 12.27278 33.32589 190.4245
          [,329]   [,330]   [,331]  [,332]   [,333]   [,334]   [,335]   [,336]
    [1,] 100.431 92.04593 28.86501 9.00055 6.023497 94.71386 4.936477 100.9374
          [,337]   [,338]   [,339]   [,340]   [,341]  [,342]   [,343]   [,344]
    [1,] 71.0754 9.879819 6.278944 10.79887 57.27989 28.2116 51.65827 70.36233
           [,345]   [,346]   [,347]   [,348]   [,349]   [,350]   [,351]   [,352]
    [1,] 18.43336 58.71241 19.45527 158.6614 208.0529 18.33004 20.28024 33.16926
           [,353]   [,354]   [,355]   [,356]   [,357]   [,358]   [,359]   [,360]
    [1,] 58.88393 5.333966 174.7287 130.8712 15.48957 37.90782 14.65963 56.76289
           [,361]   [,362]   [,363]   [,364]   [,365]   [,366]   [,367]   [,368]
    [1,] 32.22148 18.04928 47.42825 12.50071 31.77655 21.73939 45.59683 48.59985
           [,369]   [,370]   [,371]   [,372]   [,373]   [,374]   [,375]   [,376]
    [1,] 31.71231 71.53089 38.51268 15.18438 43.99992 21.29057 59.28404 18.14987
           [,377]  [,378]   [,379]   [,380]  [,381]   [,382]   [,383]   [,384]
    [1,] 19.11837 34.3169 24.77385 97.66534 15.0246 20.41301 149.5528 58.94482
          [,385]   [,386]   [,387]   [,388]  [,389]   [,390]   [,391]  [,392]
    [1,] 42.0296 43.76096 30.18345 8.921286 13.7027 7.775299 11.56046 53.7891
           [,393]   [,394]   [,395]   [,396]   [,397]   [,398]   [,399]   [,400]
    [1,] 39.17329 14.81475 6.651319 3.755604 32.42072 4.946616 36.96134 24.29502
           [,401]   [,402]   [,403]   [,404]   [,405]   [,406]   [,407]   [,408]
    [1,] 24.28048 21.45501 70.16206 28.32565 24.85451 27.34372 20.91589 152.9279
           [,409]   [,410]  [,411]   [,412]   [,413]   [,414]   [,415]   [,416]
    [1,] 21.81682 31.21976 53.8207 11.79002 33.97705 20.45091 23.75228 67.96273
           [,417]   [,418]   [,419]   [,420]   [,421]   [,422]   [,423]   [,424]
    [1,] 27.68237 16.91727 54.91118 53.53385 57.80384 95.05443 26.52124 63.81566
            [,425]   [,426]   [,427]   [,428]   [,429]   [,430] [,431]   [,432]
    [1,] 0.4055024 25.21954 107.5231 8.488358 15.67028 13.47842 61.309 25.75934
           [,433]   [,434]   [,435]   [,436]   [,437]   [,438]   [,439]   [,440]
    [1,] 13.78896 20.02232 28.39381 21.39885 11.99808 27.80104 3.522029 14.87349
           [,441]   [,442]   [,443]   [,444]   [,445]   [,446]   [,447]   [,448]
    [1,] 24.85251 15.14637 39.93845 18.37774 9.980862 13.24645 17.47611 49.33642
           [,449]  [,450]   [,451] [,452]  [,453]   [,454]   [,455]   [,456]
    [1,] 10.67831 16.1431 28.90872 47.117 22.6916 23.18494 25.44181 2.749836
           [,457]   [,458]   [,459]   [,460]  [,461]   [,462]   [,463]   [,464]
    [1,] 67.45458 36.02712 10.49791 9.298558 50.5091 24.89281 24.17613 84.66119
           [,465]  [,466]   [,467]  [,468]   [,469]   [,470]   [,471]   [,472]
    [1,] 24.50888 57.0963 2.682085 49.1565 13.73453 29.40602 11.81654 47.22168
           [,473]   [,474]   [,475]    [,476]   [,477]   [,478]   [,479]   [,480]
    [1,] 32.03671 3.823616 57.69841 0.9277733 27.82158 26.86732 31.85696 49.67528
           [,481]   [,482]   [,483]   [,484]   [,485]   [,486]   [,487]   [,488]
    [1,] 10.52468 43.96173 25.09565 58.14913 15.36141 19.07196 19.76041 43.53846
           [,489]  [,490]   [,491]   [,492]   [,493]   [,494]   [,495]   [,496]
    [1,] 28.72089 26.4811 39.76684 67.70582 26.51269 85.19007 17.72252 13.79793
          [,497]   [,498]   [,499]   [,500]   [,501]   [,502]   [,503]   [,504]
    [1,] 19.1293 41.71058 12.51597 22.17338 17.28694 17.61615 5.090223 8.889266
           [,505]   [,506]  [,507]   [,508]   [,509]   [,510]   [,511]  [,512]
    [1,] 5.833748 3.346682 12.3403 19.17915 6.916533 20.25087 67.38708 15.7498
           [,513]   [,514]  [,515]   [,516]   [,517]   [,518]   [,519]   [,520]
    [1,] 14.34347 65.08446 90.5076 8.159663 52.48485 12.16992 2.228247 39.41229
           [,521]   [,522]   [,523]   [,524]   [,525]   [,526]   [,527]   [,528]
    [1,] 37.00626 20.46767 68.84123 15.94441 33.24237 12.21188 33.97798 23.74078
           [,529]   [,530]   [,531]   [,532]  [,533]   [,534]   [,535]   [,536]
    [1,] 26.66913 4.586215 11.22484 21.37524 40.1906 35.01562 3.141903 11.22299
           [,537]   [,538]   [,539]  [,540]   [,541]   [,542]   [,543]   [,544]
    [1,] 14.80676 27.87326 14.26955 4.47857 23.19328 11.76807 27.58798 23.84548
           [,545]   [,546]   [,547]   [,548]   [,549]   [,550]   [,551]   [,552]
    [1,] 37.57769 12.70666 53.37096 20.24429 5.849793 5.239459 20.54346 49.74094
           [,553]   [,554]   [,555]   [,556]   [,557]   [,558]  [,559]   [,560]
    [1,] 13.20808 7.165959 8.363347 11.08767 38.49659 57.60811 15.6911 4.610541
           [,561]   [,562]   [,563]   [,564]   [,565]   [,566]   [,567]  [,568]
    [1,] 52.65687 43.98111 102.5441 51.20322 85.29798 10.05669 14.22094 15.7802
           [,569]   [,570]   [,571]   [,572]   [,573]   [,574]   [,575]   [,576]
    [1,] 16.96778 21.78219 19.69431 13.46855 12.44354 48.77938 13.93382 18.08658
           [,577]   [,578]   [,579]  [,580]   [,581]   [,582]   [,583]   [,584]
    [1,] 9.282487 30.39583 29.11345 7.11424 21.76473 4.106983 14.31791 15.90411
           [,585]   [,586]   [,587]  [,588]   [,589]   [,590]   [,591]   [,592]
    [1,] 11.15686 31.60416 8.094276 24.5243 21.84714 33.71089 36.85202 14.22402
           [,593]   [,594]   [,595]   [,596]   [,597]   [,598]   [,599]   [,600]
    [1,] 65.11053 6.273954 12.71348 7.591962 10.03005 8.275329 13.86145 7.291769
           [,601]  [,602]   [,603]   [,604]   [,605]   [,606]   [,607]   [,608]
    [1,] 6.481633 19.2591 29.61002 10.96815 38.02603 20.89852 5.666901 9.172564
           [,609]   [,610]   [,611]   [,612]   [,613]   [,614]   [,615]   [,616]
    [1,] 3.733043 10.36608 62.81607 42.11078 35.18909 8.487094 12.87981 16.00295
           [,617]   [,618]  [,619]   [,620]   [,621]   [,622]  [,623]   [,624]
    [1,] 20.36855 13.21894 13.9237 28.23145 20.15321 8.198554 23.1139 17.87382
          [,625]   [,626]  [,627]   [,628]   [,629]   [,630]   [,631]   [,632]
    [1,] 23.3701 9.200787 10.0661 7.909423 26.20027 16.77972 10.97964 25.78826
           [,633]   [,634]   [,635]   [,636]   [,637]   [,638]   [,639]   [,640]
    [1,] 4.251269 11.11109 18.27491 35.48849 5.680937 25.73711 4.525623 23.52607
            [,641]   [,642]     [,643]   [,644]   [,645]   [,646]   [,647]   [,648]
    [1,] 0.6785339 18.03261 0.08464099 4.485771 7.538388 11.49709 45.55096 8.198895
           [,649]   [,650]   [,651]   [,652]   [,653]   [,654]   [,655]   [,656]
    [1,] 16.50716 16.25647 24.00423 8.408709 17.43779 24.98089 10.07307 8.520382
           [,657]   [,658]   [,659]   [,660]   [,661]   [,662]   [,663]  [,664]
    [1,] 11.22366 3.860565 8.773663 5.328586 31.54215 18.50802 12.41735 9.27285
           [,665]   [,666]  [,667]   [,668]   [,669]   [,670]   [,671]   [,672]
    [1,] 12.82941 17.16002 34.0901 20.69002 35.05479 2.705529 7.289508 28.77511
           [,673]   [,674]  [,675]    [,676]   [,677]   [,678]   [,679]   [,680]
    [1,] 26.79363 6.185509 16.7212 0.3619762 9.234209 9.246012 31.98722 8.853654
           [,681]   [,682]   [,683]   [,684]   [,685]   [,686]   [,687]   [,688]
    [1,] 4.054556 17.87797 51.53339 25.32941 6.601817 48.22346 59.95561 27.53009
           [,689]   [,690]   [,691]   [,692]   [,693]   [,694]   [,695]   [,696]
    [1,] 8.044949 7.118262 11.35481 7.106548 20.60959 3.957762 29.38001 3.750257
           [,697]   [,698]   [,699]   [,700]   [,701]    [,702]   [,703]   [,704]
    [1,] 14.00216 18.59472 9.013905 11.06271 4.318946 0.4324191 3.259206 1.749581
           [,705]   [,706]   [,707]   [,708]   [,709]   [,710]   [,711]   [,712]
    [1,] 10.44767 17.05054 8.879296 18.18082 11.98749 21.93655 13.63362 7.279666
           [,713]   [,714]   [,715]   [,716]   [,717]   [,718]  [,719]   [,720]
    [1,] 51.60344 7.175761 16.82015 10.02736 8.193686 19.72937 20.7779 14.46071
           [,721]   [,722]   [,723]   [,724]  [,725]   [,726]   [,727]   [,728]
    [1,] 19.90386 12.07829 3.960839 4.614798 1.74035 8.509866 15.11923 6.757484
           [,729]  [,730]  [,731]  [,732]   [,733]   [,734]   [,735]   [,736]
    [1,] 30.49922 1.92834 0.30497 3.68209 16.57544 11.50433 7.165384 12.04648
           [,737]   [,738]  [,739]  [,740]   [,741]   [,742]   [,743]   [,744]
    [1,] 1.431354 3.758531 6.29229 32.2555 1.648161 7.494341 5.387158 5.805199
           [,745]   [,746]   [,747]   [,748]  [,749] [,750]   [,751]    [,752]
    [1,] 13.85448 6.664699 2.337477 8.008155 18.8187 9.4965 3.490196 0.9935558
           [,753]   [,754]   [,755]   [,756]   [,757]   [,758]  [,759]   [,760]
    [1,] 3.310304 5.285009 12.06545 14.29096 12.04672 12.82312 13.2867 3.209104
           [,761]  [,762]   [,763]   [,764]   [,765]   [,766]   [,767]   [,768]
    [1,] 6.435179 5.20118 3.228665 6.647633 4.985047 15.56344 14.78197 15.69711
           [,769]   [,770]   [,771]   [,772]   [,773]   [,774]   [,775]   [,776]
    [1,] 6.439911 4.268761 12.01814 7.255677 0.353724 8.030407 14.91461 8.830584
           [,777]  [,778]    [,779] [,780]   [,781]   [,782]   [,783]   [,784]
    [1,] 8.828261 9.21668 0.8512717 11.305 3.806063 8.280021 10.54053 4.166527
           [,785]   [,786]  [,787]   [,788]   [,789]  [,790]   [,791]   [,792]
    [1,] 4.629536 4.770062 5.24642 3.299294 1.671525 6.90669 11.38528 13.76006
           [,793]   [,794]   [,795]    [,796]   [,797]   [,798]   [,799]   [,800]
    [1,] 8.412153 9.761937 7.164802 0.1505894 1.965987 7.773871 1.097846 5.965673
           [,801]   [,802]   [,803]   [,804]   [,805]   [,806]   [,807]   [,808]
    [1,] 4.940414 12.42875 4.513655 4.411961 8.965521 5.355039 2.941832 5.633351
           [,809]   [,810]   [,811]   [,812]   [,813]   [,814]  [,815]   [,816]
    [1,] 4.419179 11.65194 5.217865 6.347364 2.660317 2.254304 0.76002 1.931021
    


```python
# Test data input
dataTestInput <- (splitedDataset$dataTest)[,1:ncol(splitedDataset$dataTest)-1]

# Predict using linear model
predictions <- prediction(model, dataTestInput)
# predictions
```

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Text-Classification" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

<a id="7"></a>

## Stochastic Gradient Descent
SGD has been successfully applied to large-scale and sparse machine learning problems often encountered in text classification and natural language processing. 
Given that the data is sparse, the classifiers in this module easily scale to problems with more than 10^5 training examples and more than 10^5 features.


```python
# Fitting the model
SGDmodel <- SGD(splitedDataset$dataTrain)

#show result
print(SGDmodel)
```

             [,1]      [,2]      [,3]      [,4]       [,5]      [,6]      [,7]
    [1,] 23.60562 0.3570986 -67.58038 -0.852198 0.03072518 0.8345339 0.9337756
              [,8]     [,9]     [,10]    [,11]    [,12]      [,13]    [,14]
    [1,] -9.687577 5.757197 0.5544197 2.434515 218.7185 0.02718541 2.821232
             [,15]    [,16]     [,17]    [,18]     [,19]     [,20]     [,21]
    [1,] 0.1708038 95.77304 0.5149022 108.2886 -9.624174 -9.302528 0.4614279
             [,22]      [,23]     [,24]    [,25]    [,26]    [,27]     [,28]
    [1,] -122.7846 0.08188764 0.4605931 39.41364 37.43043 44.64199 0.8321347
            [,29]   [,30]    [,31]    [,32]     [,33]     [,34]   [,35]     [,36]
    [1,] 39.70286 357.437 -31.7594 56.69234 0.1221462 0.8760759 3.77568 0.7951142
            [,37]     [,38]     [,39]     [,40]     [,41]    [,42]        [,43]
    [1,] 129.1638 0.4764346 0.9601035 -54.24243 0.3773622 8.425283 0.0002425576
             [,44]     [,45]    [,46]    [,47]     [,48]     [,49]     [,50]
    [1,] -76.49155 -9.050672 54.16413 0.765125 0.9672685 0.8658245 -54.34238
             [,51]    [,52]     [,53]    [,54]     [,55]     [,56]    [,57]
    [1,] 0.6879208 54.52628 0.2416542 37.88372 0.6180421 0.9920516 38.08354
             [,58]     [,59]    [,60]    [,61]     [,62]     [,63]    [,64]
    [1,] 0.3891173 0.5775674 135.1941 54.51453 0.4605191 0.5020112 94.19323
             [,65]    [,66]     [,67]    [,68]    [,69]     [,70]    [,71]    [,72]
    [1,] 0.8994822 152.6695 0.2938767 36.78791 56.78613 0.4597743 262.9811 53.07947
           [,73]     [,74]     [,75]    [,76]      [,77]     [,78]    [,79]
    [1,] 33.3643 0.2787512 0.8685358 3.857186 0.03002933 0.7842817 3.484301
            [,80]     [,81]     [,82]     [,83]     [,84]     [,85]     [,86]
    [1,] 2.321414 -69.20415 0.8747994 0.9910504 0.9368901 -3.872205 0.1781989
            [,87]    [,88]     [,89]    [,90]    [,91]    [,92]     [,93]    [,94]
    [1,] 34.79035 2.792285 0.7903988 3.186819 32.81748 26.24489 0.7374766 -53.7374
            [,95]    [,96]    [,97]     [,98]     [,99]    [,100]    [,101]
    [1,] 1.910976 9.068228 0.521333 -63.04926 -24.25136 0.4754189 0.7418872
            [,102]   [,103]  [,104]   [,105]    [,106]    [,107]    [,108]
    [1,] 0.4619187 95.02252 1.95456 39.30255 0.6544949 0.4552929 -9.893216
             [,109]   [,110]   [,111]   [,112]    [,113]    [,114]     [,115]
    [1,] -0.4673874 -9.40764 53.98408 70.90319 0.8702192 0.6735939 -0.8771706
            [,116]   [,117]    [,118]    [,119]    [,120]   [,121]    [,122]
    [1,] -64.42812 36.84781 0.5648529 0.9153522 0.4123449 51.83104 0.3674489
            [,123]   [,124]     [,125]    [,126]     [,127]    [,128]   [,129]
    [1,] 0.2964017 0.149087 0.07301086 0.5172744 -0.1860319 -110.7669 54.61319
            [,130]    [,131]    [,132]    [,133]  [,134]    [,135]    [,136]
    [1,] -2.902512 0.9621213 0.8501003 0.6092324 52.3924 0.8189104 -9.207152
           [,137]    [,138]    [,139]   [,140]   [,141]    [,142]   [,143]
    [1,] 0.973949 0.5161228 0.6289291 2.860819 27.03445 0.6424748 17.00232
            [,144]    [,145]    [,146]    [,147]     [,148]      [,149]   [,150]
    [1,] -9.033919 0.8798873 0.2266025 0.5594111 0.04532433 0.003925996 36.55439
             [,151]   [,152]    [,153]    [,154]    [,155]    [,156]    [,157]
    [1,] 0.01169498 34.62632 0.6575782 0.6813325 0.8242191 0.4306993 0.3975265
           [,158]   [,159]    [,160]    [,161]   [,162]    [,163]   [,164]
    [1,] 37.04472 2.998414 0.5824295 0.1321748 0.204676 0.6422911 0.570649
             [,165]    [,166]    [,167]    [,168]    [,169]    [,170]    [,171]
    [1,] 0.05940507 0.4813431 0.8130488 0.8062153 0.1020342 0.6371674 0.7684485
           [,172]   [,173]   [,174]    [,175]   [,176]     [,177]    [,178]
    [1,] 52.20519 -19.4643 -9.13025 0.8488405 30.66976 0.07296489 0.9479024
           [,179]    [,180]    [,181]    [,182]   [,183]    [,184]    [,185]
    [1,] 38.69631 -54.38153 -0.796826 0.4913092 0.261375 0.1278634 0.3165107
            [,186]   [,187]   [,188]    [,189]  [,190]    [,191]    [,192]
    [1,] 0.4272731 3.496065 1.267449 0.6429083 0.64841 0.2334508 0.1027684
            [,193]    [,194]     [,195]      [,196]    [,197]    [,198]   [,199]
    [1,] 0.6102615 0.3914789 0.03991663 0.004142025 0.3411284 0.9470514 -54.2327
           [,200]    [,201]   [,202]    [,203]    [,204]    [,205]    [,206]
    [1,] 0.883437 0.6478292 0.498982 0.4325998 0.1887695 0.3264992 0.2676997
            [,207]    [,208]    [,209]    [,210]   [,211]    [,212]  [,213]
    [1,] 0.4221403 0.9277647 0.4573093 0.4769879 -9.79152 -18.78512 0.38847
            [,214]    [,215]   [,216]    [,217]   [,218]    [,219]   [,220]
    [1,] 0.2904944 -9.081623 0.591658 0.9819354 6.182231 0.6781226 0.134353
            [,221]   [,222]    [,223]    [,224]    [,225]    [,226]    [,227]
    [1,] 0.2652792 77.59463 0.2534381 0.2533911 0.1642077 0.2950522 0.3836324
            [,228]   [,229]    [,230]    [,231]    [,232]    [,233]    [,234]
    [1,] 0.9110431 1.404938 -3.963133 0.1210214 0.8081758 0.5838853 0.7098347
            [,235]    [,236]    [,237]    [,238]    [,239]    [,240]    [,241]
    [1,] 0.5622992 0.9704517 0.6295528 0.8482321 0.7178358 0.3814704 -8.989286
           [,242]    [,243]   [,244]   [,245]   [,246]    [,247]    [,248]
    [1,] 3.736783 0.4486428 0.227219 0.756026 0.452647 0.1684857 0.1926618
            [,249]    [,250]    [,251]   [,252]    [,253]     [,254]   [,255]
    [1,] 0.1516911 0.9526753 0.5875371 0.669806 0.8643789 0.03078368 1.279619
           [,256]   [,257]    [,258]     [,259]   [,260]    [,261]    [,262]
    [1,] 36.92195 53.77073 0.9454462 0.08471986 0.794897 -58.06716 0.2845983
            [,263]   [,264]    [,265]    [,266]    [,267]    [,268]    [,269]
    [1,] 0.2475854 54.62443 0.8468467 0.2737608 0.4565805 0.9490438 0.2434134
            [,270]    [,271]    [,272]   [,273]   [,274]   [,275]   [,276]   [,277]
    [1,] 0.4422175 0.4932821 0.4452557 91.39622 1.701225 53.10922 1.040707 1.376919
           [,278]   [,279]    [,280]   [,281]    [,282]   [,283]    [,284]   [,285]
    [1,] 1.374194 4.713276 0.9933677 1.287426 -2.409956 1.679314 0.8235253 53.81712
            [,286]   [,287]   [,288]    [,289]    [,290]    [,291]   [,292]
    [1,] 0.5812705 0.158755 0.749115 0.8861045 -58.04186 0.5691661 52.61763
            [,293]    [,294]    [,295]    [,296]    [,297]    [,298]    [,299]
    [1,] 0.1215307 0.8590732 0.5440546 0.6939862 0.3365277 0.2251847 0.7723363
           [,300]    [,301]    [,302]    [,303]    [,304]    [,305]   [,306]
    [1,] 3.525883 -9.525097 0.1440876 0.2895581 0.6437364 -16.36587 0.882377
            [,307]    [,308]    [,309]    [,310]   [,311]    [,312]   [,313]
    [1,] 0.1993837 0.3001659 0.7317593 0.7041954 0.662249 0.6608902 2.920869
            [,314]   [,315]     [,316]    [,317]    [,318]    [,319]    [,320]
    [1,] 0.8164905 0.873967 0.07613502 0.2240052 -3.952272 0.7143281 0.5692557
            [,321]    [,322]     [,323]    [,324]    [,325]     [,326]     [,327]
    [1,] 0.4724966 -4.326482 0.05478512 0.7647785 -9.589077 0.01719883 -0.1441658
            [,328]    [,329]      [,330]    [,331]    [,332]    [,333]    [,334]
    [1,] -20.54632 0.7525548 0.005916652 0.3038771 0.6224343 0.3510238 0.2750569
            [,335]   [,336]   [,337]   [,338]    [,339]   [,340]    [,341]
    [1,] 0.3184519 2.160787 54.28531 0.215817 0.8420078 0.446159 0.9920704
            [,342]     [,343]   [,344]    [,345]    [,346]  [,347]   [,348]
    [1,] 0.6893662 0.04387941 2.967607 0.7449657 0.1066361 0.21464 1.623362
            [,349]    [,350]     [,351]   [,352]     [,353]     [,354]  [,355]
    [1,] 0.2673324 0.9098155 0.06065648 90.53151 0.08524673 0.03988672 72.9319
            [,356]    [,357]    [,358]    [,359]     [,360]    [,361]   [,362]
    [1,] 0.5740021 0.4522591 0.1395448 0.2057234 0.04334045 0.2156782 2.172435
            [,363]    [,364]    [,365]    [,366]    [,367]    [,368]   [,369]
    [1,] 0.5305207 0.4191271 0.9980539 0.4192283 -1.125247 0.5957958 1.428872
            [,370]    [,371]    [,372]   [,373]    [,374]   [,375]    [,376]
    [1,] -54.37816 0.5320239 0.5804279 0.614608 0.1198383 36.36933 0.7633454
            [,377]    [,378]    [,379]    [,380]     [,381]    [,382]  [,383]
    [1,] 0.1827978 0.3663288 0.5762953 0.1397546 0.05293653 0.2661078 36.9951
          [,384]  [,385]      [,386]    [,387]    [,388]     [,389]    [,390]
    [1,] 44.2382 3.36379 0.004852339 0.4791286 0.5519983 0.08731702 0.7221536
           [,391]     [,392]    [,393]   [,394]   [,395]    [,396]    [,397]
    [1,] 2.281876 0.01206622 0.5730218 0.114262 0.348104 0.1570402 0.9622078
            [,398]    [,399]    [,400]    [,401]    [,402]    [,403]    [,404]
    [1,] 0.1952793 0.5374645 0.6580595 0.9360799 0.5840123 0.4080274 0.4750066
            [,405]    [,406]    [,407]    [,408]    [,409]    [,410]    [,411]
    [1,] 0.3356705 0.7343912 0.9547231 0.4318179 0.3217732 0.6389249 0.5145499
            [,412]   [,413]    [,414]    [,415]   [,416]   [,417]   [,418]
    [1,] 0.8542226 0.214801 0.1055238 0.7106181 0.029428 -3.18162 0.821522
            [,419]    [,420]    [,421]    [,422]    [,423]   [,424]    [,425]
    [1,] 0.3278897 0.6866931 -1.484615 0.3605391 0.4686853 3.537407 0.3090991
            [,426]      [,427]    [,428]    [,429]    [,430]     [,431]    [,432]
    [1,] 0.2056136 0.006757624 0.5885244 -54.79742 0.7130944 0.06638885 0.6930577
            [,433]    [,434]   [,435]   [,436]    [,437]    [,438]    [,439]
    [1,] 0.8183524 0.9777949 54.49681 52.82557 -54.55072 -54.57387 0.6390923
            [,440]    [,441]    [,442]   [,443]   [,444]   [,445]    [,446]
    [1,] 0.0428694 0.7019717 0.3449066 0.872504 54.62856 0.446846 0.6509709
            [,447]   [,448]    [,449]    [,450] [,451]   [,452]   [,453]    [,454]
    [1,] 0.5084779 54.18323 0.8269632 0.8652802 54.079 3.215145 3.289135 0.5621816
            [,455]    [,456]    [,457]    [,458]    [,459]    [,460]    [,461]
    [1,] 0.9587897 0.3431173 0.8593804 0.4676938 0.3661641 0.4277255 0.1490823
            [,462]    [,463]    [,464]    [,465]   [,466]    [,467]    [,468]
    [1,] 0.8917802 0.2845346 0.6610065 0.4185378 54.56471 0.4700222 0.1117174
            [,469]    [,470]    [,471]    [,472]    [,473]    [,474]   [,475]
    [1,] 0.2961615 0.2254125 0.7254775 0.9557228 0.3775876 0.1591906 2.811061
           [,476]    [,477]     [,478]    [,479]   [,480]    [,481]    [,482]
    [1,] 0.404257 0.8222442 -0.2879336 0.8548743 3.294134 0.8471537 0.1626322
             [,483]   [,484]    [,485]    [,486]   [,487]    [,488]    [,489]
    [1,] 0.06883513 0.956065 0.7977274 0.5211907 0.927076 0.9328582 0.6032267
            [,490]   [,491]    [,492]    [,493]   [,494]    [,495]   [,496]
    [1,] -3.166191 2.889199 0.1758869 0.1257089 4.683524 0.2759668 54.12483
            [,497]   [,498]    [,499]    [,500]   [,501]    [,502]    [,503]
    [1,] 0.8465822 2.278863 0.8436468 0.8982456 0.824457 0.4288045 0.6813808
           [,504]    [,505]    [,506]    [,507]    [,508]    [,509]    [,510]
    [1,] 36.63796 0.9108074 0.2424892 0.6993776 0.3827807 0.6284622 0.5435364
             [,511]   [,512]    [,513]    [,514]    [,515]    [,516]   [,517]
    [1,] -0.1519977 53.71199 0.1504855 0.9990227 0.2558501 0.3530459 36.81581
            [,518]    [,519]    [,520]    [,521]    [,522]    [,523]    [,524]
    [1,] 0.1844492 0.6846284 0.0209931 0.8558512 0.7425172 0.8000002 0.1638317
           [,525]    [,526]    [,527]     [,528]   [,529]    [,530]    [,531]
    [1,] 36.08353 0.5703951 0.7599262 0.01679807 0.823488 0.7195744 0.0337039
               [,532]   [,533]    [,534]    [,535]    [,536]    [,537]     [,538]
    [1,] 0.0008920347 36.41591 0.5356336 0.4775492 0.5019933 0.1800747 0.03850299
            [,539]    [,540]    [,541]    [,542]    [,543]    [,544]    [,545]
    [1,] 0.7738097 0.9433589 0.6554129 0.1676579 0.6287296 0.5455586 0.1319756
            [,546]    [,547]    [,548]    [,549]  [,550]    [,551]   [,552]
    [1,] 0.5949354 -3.340814 0.6091795 0.6787335 0.07301 0.2961503 54.19338
            [,553]    [,554]    [,555]     [,556]    [,557]   [,558]    [,559]
    [1,] 0.0350657 0.2391474 0.2524753 0.05943466 0.3444783 0.842866 0.7896791
           [,560]    [,561]    [,562]     [,563]     [,564]    [,565]    [,566]
    [1,] 0.285297 0.7280526 0.5729898 0.08354686 0.09817646 0.9857197 0.1975667
            [,567]    [,568]    [,569]     [,570]    [,571]  [,572]    [,573]
    [1,] 0.9106267 0.9763107 0.5044034 0.08789984 0.4186767 0.11117 0.5314098
          [,574]      [,575]    [,576]    [,577]    [,578]     [,579]   [,580]
    [1,] 161.096 0.008176162 0.2126799 0.1065851 -1.777526 0.07045163 0.590367
            [,581]    [,582]    [,583]    [,584]    [,585]    [,586]    [,587]
    [1,] 0.8704083 0.5430169 0.3654241 0.7136541 0.2917693 0.7185972 0.2672193
            [,588]     [,589]     [,590]    [,591]   [,592]    [,593]    [,594]
    [1,] 0.8026071 0.01762329 0.06933411 0.4487043 54.49062 0.2173366 0.6090006
           [,595]    [,596]    [,597]    [,598]    [,599]    [,600]    [,601]
    [1,] 0.376565 0.7055704 0.1701808 0.9191247 0.2848393 0.8391489 0.3102132
             [,602]    [,603]  [,604]   [,605]    [,606]    [,607]    [,608]
    [1,] 0.04849312 0.7178432 0.78798 3.584025 0.4934095 0.2712362 0.1332291
            [,609]    [,610]    [,611]   [,612]     [,613]    [,614]    [,615]
    [1,] 0.9734684 0.7524586 0.9958229 1.257791 0.07947731 0.3387191 0.6061505
            [,616]  [,617]  [,618]    [,619]    [,620]   [,621]   [,622]    [,623]
    [1,] 0.2428089 0.66393 2.89834 0.9823512 0.5673164 3.182657 0.395912 0.5653358
           [,624]    [,625]    [,626]     [,627]    [,628]    [,629]    [,630]
    [1,] 2.018898 0.9806253 0.8180014 0.09621762 0.9057858 0.6752547 0.3521976
            [,631]    [,632]    [,633]    [,634]    [,635]    [,636]    [,637]
    [1,] 0.8368114 0.7037896 0.8539715 0.7265436 0.8437282 0.8940544 0.7695675
            [,638]    [,639]    [,640]    [,641]    [,642]    [,643]    [,644]
    [1,] 0.6809114 0.2284092 0.7291223 0.4352981 0.7887072 0.1288686 0.4240306
            [,645]    [,646]    [,647]    [,648]    [,649]    [,650]    [,651]
    [1,] 0.5491393 0.7396165 0.9384728 0.6105456 0.3704755 0.4780935 0.2671887
            [,652]   [,653]   [,654]    [,655]    [,656]    [,657]    [,658]
    [1,] 0.6365891 0.419795 0.426164 0.7403611 0.8846566 0.3975038 0.7411627
            [,659]      [,660]   [,661]    [,662]    [,663]    [,664]    [,665]
    [1,] 0.7878624 0.007089933 2.908128 0.5204147 0.2001751 0.1497934 0.1500444
            [,666]    [,667]    [,668]    [,669]    [,670]    [,671]    [,672]
    [1,] 0.5640897 0.6413469 0.6605934 0.9296227 0.4119989 0.9045458 0.1050772
            [,673]    [,674]    [,675]    [,676]    [,677]    [,678]    [,679]
    [1,] -9.138069 0.6794469 0.8710793 0.8151215 0.2429956 0.4357712 0.4633673
            [,680]     [,681]   [,682]    [,683]    [,684]    [,685]    [,686]
    [1,] 0.1236505 0.05739229 0.820601 0.3693939 0.3848126 0.5581832 0.2817257
          [,687]    [,688]    [,689]    [,690]    [,691]    [,692]    [,693]
    [1,] 0.17404 0.4240804 0.7984321 0.7091013 0.1138026 0.5662536 0.7374595
            [,694]    [,695]    [,696]    [,697]   [,698]    [,699]    [,700]
    [1,] 0.9651158 0.4154674 0.6776953 0.4294787 0.741266 0.7111233 0.9687364
            [,701]    [,702]    [,703]    [,704]    [,705]    [,706]   [,707]
    [1,] 0.7881835 0.1084563 0.5340319 0.8951118 0.3947766 0.5911712 0.583513
            [,708]    [,709]    [,710]     [,711]    [,712]     [,713]    [,714]
    [1,] 0.6443132 0.7901244 0.1758731 0.08920131 0.2188061 0.05890103 0.7338132
            [,715]     [,716]    [,717]    [,718]    [,719]   [,720]    [,721]
    [1,] 0.6886632 0.03998076 0.1277119 0.4618286 0.8596008 0.389891 0.1468767
            [,722]    [,723]     [,724]    [,725]   [,726]   [,727]    [,728]
    [1,] 0.7287269 0.9229372 0.09617984 0.2735251 0.036668 0.723342 0.4422697
            [,729]    [,730]    [,731]    [,732]    [,733]    [,734]    [,735]
    [1,] 0.4163194 0.2845917 0.1131161 0.6740074 0.2849642 0.1696746 0.3071223
            [,736]    [,737]   [,738]    [,739]    [,740]    [,741]    [,742]
    [1,] 0.3032873 0.4290417 0.313266 0.2467031 0.3128382 0.6794484 0.9001779
            [,743]    [,744]    [,745]    [,746]    [,747]    [,748]    [,749]
    [1,] -1.346819 0.4382826 0.4095833 0.7841907 0.1507095 0.9558495 0.1012325
           [,750]     [,751]    [,752]    [,753]    [,754]    [,755]    [,756]
    [1,] 0.360678 0.01843806 0.6469623 0.3265035 0.1167944 0.2407651 -1.418194
            [,757]   [,758]    [,759]     [,760]   [,761]    [,762]    [,763]
    [1,] 0.9724322 0.962063 0.2396939 0.09160018 0.203002 0.7996892 0.3345962
             [,764]    [,765]     [,766]     [,767]    [,768]    [,769]    [,770]
    [1,] 0.02894356 0.0481163 0.02715385 0.08916973 0.7052583 0.4249919 0.8248898
            [,771]    [,772]    [,773]    [,774]    [,775]    [,776]    [,777]
    [1,] 0.7992636 0.9367087 0.8820491 0.4405789 0.6445042 -1.107578 0.2872721
             [,778]     [,779]    [,780]    [,781]    [,782]   [,783]    [,784]
    [1,] 0.03951718 -0.6389252 0.2304038 0.8485967 0.6146888 0.415718 0.1000977
            [,785]   [,786]    [,787]    [,788]    [,789]    [,790]    [,791]
    [1,] 0.1419843 0.322006 0.5306572 0.2788665 0.7719757 0.5690119 0.1653155
          [,792]    [,793]    [,794]    [,795]    [,796]    [,797]    [,798]
    [1,] 0.77488 0.8474875 0.2470246 0.3120023 0.3297168 0.7156025 0.5768983
            [,799]    [,800]    [,801]    [,802]    [,803]      [,804]      [,805]
    [1,] 0.9344137 0.5415367 0.8433688 0.2825161 0.4910543 0.002892039 0.007447552
            [,806]    [,807]    [,808]    [,809]    [,810]    [,811]    [,812]
    [1,] 0.8339375 0.2330449 0.5751446 0.9849012 0.5561591 0.4579069 0.9336282
            [,813]   [,814]    [,815]    [,816]
    [1,] 0.6729015 0.097868 0.2917804 0.5151714
    


```python
# Test data input
dataTestInput <- (splitedDataset$dataTest)[,1:ncol(splitedDataset$dataTest)-1]

# Predict using model
predictions <- prediction(model, dataTestInput)
```

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Text-Classification" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

<a id="8"></a>

## XGBoost Classifier
XGBoost is a decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. 
Rather than training all the models in isolation of one another, boosting trains models in succession
with each new model being trained to correct the errors made by the previous ones

In a standard ensemble method where models are trained in isolation, all of the models might simply end up making the same mistakes.
We should use this algorithm when we require fast and accurate predictions after the model is deployed


```python
indexes = createDataPartition(data$recommended_id, p=.9, list=F)
train = data[indexes, ]
test = data[-indexes, ]
```


```python
length(train)
length(train$recommended_id)
```


816



900



```python
train_x = data.matrix(train[,-816])
train_y = train[,816]
 
test_x = data.matrix(test[,-816])
test_y = test[,816]
```


```python
library(xgboost)

xgb_train = xgb.DMatrix(data=train_x, label=train_y)
xgb_test = xgb.DMatrix(data=test_x, label=test_y)
```

    
    Attaching package: 'xgboost'
    
    
    The following object is masked from 'package:rattle':
    
        xgboost
    
    
    The following object is masked from 'package:dplyr':
    
        slice
    
    
    


```python
xgbc = xgboost(data=xgb_train, max.depth=3, nrounds=10, objective="binary:logistic")
```

    [1]	train-error:0.156667 
    [2]	train-error:0.148889 
    [3]	train-error:0.140000 
    [4]	train-error:0.137778 
    [5]	train-error:0.136667 
    [6]	train-error:0.137778 
    [7]	train-error:0.132222 
    [8]	train-error:0.134444 
    [9]	train-error:0.128889 
    [10]	train-error:0.125556 
    


```python
print(xgbc)
```

    ##### xgb.Booster
    raw: 9.1 Kb 
    call:
      xgb.train(params = params, data = dtrain, nrounds = nrounds, 
        watchlist = watchlist, verbose = verbose, print_every_n = print_every_n, 
        early_stopping_rounds = early_stopping_rounds, maximize = maximize, 
        save_period = save_period, save_name = save_name, xgb_model = xgb_model, 
        callbacks = callbacks, max.depth = 3, objective = "binary:logistic")
    params (as set within xgb.train):
      max_depth = "3", objective = "binary:logistic", validate_parameters = "TRUE"
    xgb.attributes:
      niter
    callbacks:
      cb.print.evaluation(period = print_every_n)
      cb.evaluation.log()
    # of features: 815 
    niter: 10
    nfeatures : 815 
    evaluation_log:
        iter train_error
           1    0.156667
           2    0.148889
    ---                 
           9    0.128889
          10    0.125556
    


```python
pred = predict(xgbc, xgb_test)
```


```python
prediction <- as.numeric(pred > 0.5)
```


```python
cm = confusionMatrix(as.factor(test_y), as.factor(prediction))
cm
```


    Confusion Matrix and Statistics
    
              Reference
    Prediction  0  1
             0  2 13
             1  3 82
                                              
                   Accuracy : 0.84            
                     95% CI : (0.7532, 0.9057)
        No Information Rate : 0.95            
        P-Value [Acc > NIR] : 0.99999         
                                              
                      Kappa : 0.1351          
                                              
     Mcnemar's Test P-Value : 0.02445         
                                              
                Sensitivity : 0.4000          
                Specificity : 0.8632          
             Pos Pred Value : 0.1333          
             Neg Pred Value : 0.9647          
                 Prevalence : 0.0500          
             Detection Rate : 0.0200          
       Detection Prevalence : 0.1500          
          Balanced Accuracy : 0.6316          
                                              
           'Positive' Class : 0               
                                              


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Text-Classification" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>
