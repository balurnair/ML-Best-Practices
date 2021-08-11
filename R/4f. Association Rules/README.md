
# Association Rules

<div class="list-group" id="list-tab" role="tablist">
  <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Notebook Content</h3><br>
   <a class="list-group-item list-group-item-action" data-toggle="list" href="#Introduction" role="tab" aria-controls="settings">Introduction<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#Market-Basket-Analysis" role="tab" aria-controls="settings">Market Basket Analysis<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#Apriori-Algorithm" role="tab" aria-controls="settings">Apriori Algorithm<span class="badge badge-primary badge-pill"></span></a> <br>
    <a class="list-group-item list-group-item-action" data-toggle="list" href="#FP-Growth-Algorithm" role="tab" aria-controls="settings">FP growth Algorithm<span class="badge badge-primary badge-pill"></span></a><br>


# Introduction

Association Rules are widely used to analyze retail basket or transaction data, and are intended to identify strong rules discovered in transaction data using measures of interestingness, based on the concept of strong rules

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Association%20Rules/img.png)

An example of Association Rules<br>
<br>Assume there are 100 customers
<br>10 of them bought milk, 8 bought butter and 6 bought both of them.
<br>bought milk => bought butter
<br>support = P(Milk & Butter) = 6/100 = 0.06
<br>confidence = support/P(Butter) = 0.06/0.08 = 0.75
<br>lift = confidence/P(Milk) = 0.75/0.10 = 7.5
<br>**Note:** this example is extremely small. In practice, a rule needs the support of several hundred transactions, before it can be considered statistically significant, and datasets often contain thousands or millions of transactions.

**Difference between Association and Recommendation**
Association rules do not extract an individual's preference, rather find relationships between sets of elements of every distinct transaction. This is what makes them different techniques used in recommender systems.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Association-Rules" role="tab" aria-controls="settings">Go To Top<span class="badge badge-primary badge-pill"></span></a>

# Market Basket Analysis

Market Basket Analysis is a technique which identifies the strength of association between pairs of products purchased together and identify patterns of co-occurrence. A co-occurrence is when two or more things take place together.

Market Basket Analysis creates If-Then scenario rules, for example, if item A is purchased then item B is likely to be purchased. The rules are probabilistic in nature or, in other words, they are derived from the frequencies of co-occurrence in the observations. Frequency is the proportion of baskets that contain the items of interest. The rules can be used in pricing strategies, product placement, and various types of cross-selling strategies.


**Practical Applications of Market Basket Analysis**

When one hears Market Basket Analysis, one thinks of shopping carts and supermarket shoppers. It is important to realize that there are many other areas in which Market Basket Analysis can be applied. An example of Market Basket Analysis for a majority of Internet users is a list of potentially interesting products for Amazon. Amazon informs the customer that people who bought the item being purchased by them, also reviewed or bought another list of items. A list of applications of Market Basket Analysis in various industries is listed below:

1. **Retail:** In Retail, Market Basket Analysis can help determine what items are purchased together, purchased sequentially, and purchased by season. This can assist retailers to determine product placement and promotion optimization (for instance, combining product incentives). Does it make sense to sell soda and chips or soda and crackers?

2. **Telecommunications:** In Telecommunications, where high churn rates continue to be a growing concern, Market Basket Analysis can be used to determine what services are being utilized and what packages customers are purchasing. They can use that knowledge to direct marketing efforts at customers who are more likely to follow the same path.
For instance, Telecommunications these days is also offering TV and Internet. Creating bundles for purchases can be determined from an analysis of what customers purchase, thereby giving the company an idea of how to price the bundles. This analysis might also lead to determining the capacity requirements.

3. **Banks:** In Financial (banking for instance), Market Basket Analysis can be used to analyze credit card purchases of customers to build profiles for fraud detection purposes and cross-selling opportunities.

4. **Insurance:** In Insurance, Market Basket Analysis can be used to build profiles to detect medical insurance claim fraud. By building profiles of claims, you are able to then use the profiles to determine if more than 1 claim belongs to a particular claimee within a specified period of time.

5. **Medical:** In Healthcare or Medical, Market Basket Analysis can be used for comorbid conditions and symptom analysis, with which a profile of illness can be better identified. It can also be used to reveal biologically relevant associations between different genes or between environmental effects and gene expression.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Association-Rules" role="tab" aria-controls="settings">Go To Top<span class="badge badge-primary badge-pill"></span></a>

# Apriori Algorithm

Apriori Algorithm is a Machine Learning algorithm which is used to gain insight into the structured relationships between different items involved. The most prominent practical application of the algorithm is to recommend products based on the products already present in the user’s cart. Walmart especially has made great use of the algorithm in suggesting products to it’s users.

**Problem Statement**

Lets find the most frequent items that are bought together and can be placed on common shelf to provide better user experience.

**Step 1:** Importing the required libraries


```R
options(warn=-1)
```


```R
#importing the required libraries
library(arules)
library(arulesViz)
library('RColorBrewer')
```

    Loading required package: Matrix
    
    Attaching package: 'arules'
    
    The following objects are masked from 'package:base':
    
        abbreviate, write
    
    Loading required package: grid
    Registered S3 methods overwritten by 'ggplot2':
      method         from 
      [.quosures     rlang
      c.quosures     rlang
      print.quosures rlang
    Registered S3 method overwritten by 'seriation':
      method         from 
      reorder.hclust gclus
    

**Step 2:** Loading and exploring the data

The first part of any analysis is to bring in the dataset. We will be using an inbuilt dataset “Groceries” from the ‘arules’ package to simplify our analysis.

All stores and retailers store their information of transactions in a specific type of dataset called the “Transaction” type dataset.


```R
# Loading the Data 
data("Groceries")
```

Before we begin applying the “Apriori” algorithm on our dataset, we need to make sure that it is of the type “Transactions”.


```R
# Exploring the data 
str(Groceries)
```

    Formal class 'transactions' [package "arules"] with 3 slots
      ..@ data       :Formal class 'ngCMatrix' [package "Matrix"] with 5 slots
      .. .. ..@ i       : int [1:43367] 13 60 69 78 14 29 98 24 15 29 ...
      .. .. ..@ p       : int [1:9836] 0 4 7 8 12 16 21 22 27 28 ...
      .. .. ..@ Dim     : int [1:2] 169 9835
      .. .. ..@ Dimnames:List of 2
      .. .. .. ..$ : NULL
      .. .. .. ..$ : NULL
      .. .. ..@ factors : list()
      ..@ itemInfo   :'data.frame':	169 obs. of  3 variables:
      .. ..$ labels: chr [1:169] "frankfurter" "sausage" "liver loaf" "ham" ...
      .. ..$ level2: Factor w/ 55 levels "baby food","bags",..: 44 44 44 44 44 44 44 42 42 41 ...
      .. ..$ level1: Factor w/ 10 levels "canned food",..: 6 6 6 6 6 6 6 6 6 6 ...
      ..@ itemsetInfo:'data.frame':	0 obs. of  0 variables
    

The structure of our transaction type dataset shows us that it is internally divided into three slots: Data, itemInfo and itemsetInfo.

The slot “Data” contains the dimensions, dimension names and other numerical values of number of products sold by every transaction made.


```R
# Exploring the different transactions 
head(Groceries@itemInfo, n=20)
```


<table>
<thead><tr><th scope=col>labels</th><th scope=col>level2</th><th scope=col>level1</th></tr></thead>
<tbody>
	<tr><td>frankfurter         </td><td>sausage             </td><td>meat and sausage    </td></tr>
	<tr><td>sausage             </td><td>sausage             </td><td>meat and sausage    </td></tr>
	<tr><td>liver loaf          </td><td>sausage             </td><td>meat and sausage    </td></tr>
	<tr><td>ham                 </td><td>sausage             </td><td>meat and sausage    </td></tr>
	<tr><td>meat                </td><td>sausage             </td><td>meat and sausage    </td></tr>
	<tr><td>finished products   </td><td>sausage             </td><td>meat and sausage    </td></tr>
	<tr><td>organic sausage     </td><td>sausage             </td><td>meat and sausage    </td></tr>
	<tr><td>chicken             </td><td>poultry             </td><td>meat and sausage    </td></tr>
	<tr><td>turkey              </td><td>poultry             </td><td>meat and sausage    </td></tr>
	<tr><td>pork                </td><td>pork                </td><td>meat and sausage    </td></tr>
	<tr><td>beef                </td><td>beef                </td><td>meat and sausage    </td></tr>
	<tr><td>hamburger meat      </td><td>beef                </td><td>meat and sausage    </td></tr>
	<tr><td>fish                </td><td>fish                </td><td>meat and sausage    </td></tr>
	<tr><td>citrus fruit        </td><td>fruit               </td><td>fruit and vegetables</td></tr>
	<tr><td>tropical fruit      </td><td>fruit               </td><td>fruit and vegetables</td></tr>
	<tr><td>pip fruit           </td><td>fruit               </td><td>fruit and vegetables</td></tr>
	<tr><td>grapes              </td><td>fruit               </td><td>fruit and vegetables</td></tr>
	<tr><td>berries             </td><td>fruit               </td><td>fruit and vegetables</td></tr>
	<tr><td>nuts/prunes         </td><td>fruit               </td><td>fruit and vegetables</td></tr>
	<tr><td>root vegetables     </td><td>vegetables          </td><td>fruit and vegetables</td></tr>
</tbody>
</table>



**Step 3:** Summary of the Data

Let us check the most frequently purchased products using the summary function.


```R
summary(Groceries)
```


    transactions as itemMatrix in sparse format with
     9835 rows (elements/itemsets/transactions) and
     169 columns (items) and a density of 0.02609146 
    
    most frequent items:
          whole milk other vegetables       rolls/buns             soda 
                2513             1903             1809             1715 
              yogurt          (Other) 
                1372            34055 
    
    element (itemset/transaction) length distribution:
    sizes
       1    2    3    4    5    6    7    8    9   10   11   12   13   14   15   16 
    2159 1643 1299 1005  855  645  545  438  350  246  182  117   78   77   55   46 
      17   18   19   20   21   22   23   24   26   27   28   29   32 
      29   14   14    9   11    4    6    1    1    1    1    3    1 
    
       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
      1.000   2.000   3.000   4.409   6.000  32.000 
    
    includes extended item information - examples:
           labels  level2           level1
    1 frankfurter sausage meat and sausage
    2     sausage sausage meat and sausage
    3  liver loaf sausage meat and sausage


The summary statistics show us the top 5 items sold in our transaction set as “Whole Milk”,”Other Vegetables”,”Rolls/Buns”,”Soda” and “Yogurt”.

To parse to Transaction type, make sure your dataset has similar slots and then use the as() function in R.

**Step 4:** Generating association rules using the Apriori Algorithm


```R
rules <- apriori(Groceries,parameter = list(supp = 0.001, conf = 0.80))
```

    Apriori
    
    Parameter specification:
     confidence minval smax arem  aval originalSupport maxtime support minlen
            0.8    0.1    1 none FALSE            TRUE       5   0.001      1
     maxlen target  ext
         10  rules TRUE
    
    Algorithmic control:
     filter tree heap memopt load sort verbose
        0.1 TRUE TRUE  FALSE TRUE    2    TRUE
    
    Absolute minimum support count: 9 
    
    set item appearances ...[0 item(s)] done [0.00s].
    set transactions ...[169 item(s), 9835 transaction(s)] done [0.01s].
    sorting and recoding items ... [157 item(s)] done [0.00s].
    creating transaction tree ... done [0.01s].
    checking subsets of size 1 2 3 4 5 6 done [0.03s].
    writing ... [410 rule(s)] done [0.00s].
    creating S4 object  ... done [0.01s].
    

We will set minimum support parameter (minSup) to .001.

We can set minimum confidence (minConf) to anywhere between 0.75 and 0.85 for varied results.

**Support:** Support is the basic probability of an event to occur. If we have an event to buy product A, Support(A) is the number of transactions which includes A divided by total number of transactions.

**Confidence:** The confidence of an event is the conditional probability of the occurrence; the chances of A happening given B has already happened.

**Lift:** This is the ratio of confidence to expected confidence.The probability of all of the items in a rule occurring together (otherwise known as the support) divided by the product of the probabilities of the items on the left and right side occurring as if there was no association between them.

The lift value tells us how much better a rule is at predicting something than randomly guessing. The higher the lift, the stronger the association.

Let’s find out the top 10 rules arranged by lift.

**Step 5:** Lets inspect the Rules


```R
inspect(rules[1:10])
```

         lhs                         rhs                    support confidence    coverage      lift count
    [1]  {liquor,                                                                                         
          red/blush wine}         => {bottled beer}     0.001931876  0.9047619 0.002135231 11.235269    19
    [2]  {curd,                                                                                           
          cereals}                => {whole milk}       0.001016777  0.9090909 0.001118454  3.557863    10
    [3]  {yogurt,                                                                                         
          cereals}                => {whole milk}       0.001728521  0.8095238 0.002135231  3.168192    17
    [4]  {butter,                                                                                         
          jam}                    => {whole milk}       0.001016777  0.8333333 0.001220132  3.261374    10
    [5]  {soups,                                                                                          
          bottled beer}           => {whole milk}       0.001118454  0.9166667 0.001220132  3.587512    11
    [6]  {napkins,                                                                                        
          house keeping products} => {whole milk}       0.001321810  0.8125000 0.001626843  3.179840    13
    [7]  {whipped/sour cream,                                                                             
          house keeping products} => {whole milk}       0.001220132  0.9230769 0.001321810  3.612599    12
    [8]  {pastry,                                                                                         
          sweet spreads}          => {whole milk}       0.001016777  0.9090909 0.001118454  3.557863    10
    [9]  {turkey,                                                                                         
          curd}                   => {other vegetables} 0.001220132  0.8000000 0.001525165  4.134524    12
    [10] {rice,                                                                                           
          sugar}                  => {whole milk}       0.001220132  1.0000000 0.001220132  3.913649    12
    

The first rule shows that if we buy Liquor and Red Wine, we are very likely to buy bottled beer. We can rank the rules based on top 10 from either lift, support or confidence.

# Analyzing the results using the graphical method


```R
arules::itemFrequencyPlot(Groceries,topN=20,col=brewer.pal(8,'Pastel2'),main='Relative Item Frequency Plot',type="relative",ylab="Item Frequency (Relative)")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Association%20Rules/output_30_0.png)


These histograms depict how many times an item has occurred in our dataset as compared to the others.

The relative frequency plot accounts for the fact that “Whole Milk” and “Other Vegetables” constitute around half of the transaction dataset; half the sales of the store are of these items.

This would mean that a lot of people are buying milk and vegetables!

What other objects can we place around the more frequently purchased objects to enhance those sales too?

For example, to boost sales of eggs I can place it beside my milk and vegetables.

Moving forward in the visualisation, we can use a graph to highlight the support and lifts of various items in our repository but mostly to see which product is associated with which one in the sales environment.

**The size of graph nodes is based on support levels and the colour on lift ratios. The incoming lines show the Antecedants or the LHS and the RHS is represented by names of items.**


```R
plot(rules[1:20], method = "graph",control = list(type = "items"))
```

    Available control parameters (with default values):
    main	 =  Graph for 20 rules
    nodeColors	 =  c("#66CC6680", "#9999CC80")
    nodeCol	 =  c("#EE0000FF", "#EE0303FF", "#EE0606FF", "#EE0909FF", "#EE0C0CFF", "#EE0F0FFF", "#EE1212FF", "#EE1515FF", "#EE1818FF", "#EE1B1BFF", "#EE1E1EFF", "#EE2222FF", "#EE2525FF", "#EE2828FF", "#EE2B2BFF", "#EE2E2EFF", "#EE3131FF", "#EE3434FF", "#EE3737FF", "#EE3A3AFF", "#EE3D3DFF", "#EE4040FF", "#EE4444FF", "#EE4747FF", "#EE4A4AFF", "#EE4D4DFF", "#EE5050FF", "#EE5353FF", "#EE5656FF", "#EE5959FF", "#EE5C5CFF", "#EE5F5FFF", "#EE6262FF", "#EE6666FF", "#EE6969FF", "#EE6C6CFF", "#EE6F6FFF", "#EE7272FF", "#EE7575FF",  "#EE7878FF", "#EE7B7BFF", "#EE7E7EFF", "#EE8181FF", "#EE8484FF", "#EE8888FF", "#EE8B8BFF", "#EE8E8EFF", "#EE9191FF", "#EE9494FF", "#EE9797FF", "#EE9999FF", "#EE9B9BFF", "#EE9D9DFF", "#EE9F9FFF", "#EEA0A0FF", "#EEA2A2FF", "#EEA4A4FF", "#EEA5A5FF", "#EEA7A7FF", "#EEA9A9FF", "#EEABABFF", "#EEACACFF", "#EEAEAEFF", "#EEB0B0FF", "#EEB1B1FF", "#EEB3B3FF", "#EEB5B5FF", "#EEB7B7FF", "#EEB8B8FF", "#EEBABAFF", "#EEBCBCFF", "#EEBDBDFF", "#EEBFBFFF", "#EEC1C1FF", "#EEC3C3FF", "#EEC4C4FF", "#EEC6C6FF", "#EEC8C8FF",  "#EEC9C9FF", "#EECBCBFF", "#EECDCDFF", "#EECFCFFF", "#EED0D0FF", "#EED2D2FF", "#EED4D4FF", "#EED5D5FF", "#EED7D7FF", "#EED9D9FF", "#EEDBDBFF", "#EEDCDCFF", "#EEDEDEFF", "#EEE0E0FF", "#EEE1E1FF", "#EEE3E3FF", "#EEE5E5FF", "#EEE7E7FF", "#EEE8E8FF", "#EEEAEAFF", "#EEECECFF", "#EEEEEEFF")
    edgeCol	 =  c("#474747FF", "#494949FF", "#4B4B4BFF", "#4D4D4DFF", "#4F4F4FFF", "#515151FF", "#535353FF", "#555555FF", "#575757FF", "#595959FF", "#5B5B5BFF", "#5E5E5EFF", "#606060FF", "#626262FF", "#646464FF", "#666666FF", "#686868FF", "#6A6A6AFF", "#6C6C6CFF", "#6E6E6EFF", "#707070FF", "#727272FF", "#747474FF", "#767676FF", "#787878FF", "#7A7A7AFF", "#7C7C7CFF", "#7E7E7EFF", "#808080FF", "#828282FF", "#848484FF", "#868686FF", "#888888FF", "#8A8A8AFF", "#8C8C8CFF", "#8D8D8DFF", "#8F8F8FFF", "#919191FF", "#939393FF",  "#959595FF", "#979797FF", "#999999FF", "#9A9A9AFF", "#9C9C9CFF", "#9E9E9EFF", "#A0A0A0FF", "#A2A2A2FF", "#A3A3A3FF", "#A5A5A5FF", "#A7A7A7FF", "#A9A9A9FF", "#AAAAAAFF", "#ACACACFF", "#AEAEAEFF", "#AFAFAFFF", "#B1B1B1FF", "#B3B3B3FF", "#B4B4B4FF", "#B6B6B6FF", "#B7B7B7FF", "#B9B9B9FF", "#BBBBBBFF", "#BCBCBCFF", "#BEBEBEFF", "#BFBFBFFF", "#C1C1C1FF", "#C2C2C2FF", "#C3C3C4FF", "#C5C5C5FF", "#C6C6C6FF", "#C8C8C8FF", "#C9C9C9FF", "#CACACAFF", "#CCCCCCFF", "#CDCDCDFF", "#CECECEFF", "#CFCFCFFF", "#D1D1D1FF",  "#D2D2D2FF", "#D3D3D3FF", "#D4D4D4FF", "#D5D5D5FF", "#D6D6D6FF", "#D7D7D7FF", "#D8D8D8FF", "#D9D9D9FF", "#DADADAFF", "#DBDBDBFF", "#DCDCDCFF", "#DDDDDDFF", "#DEDEDEFF", "#DEDEDEFF", "#DFDFDFFF", "#E0E0E0FF", "#E0E0E0FF", "#E1E1E1FF", "#E1E1E1FF", "#E2E2E2FF", "#E2E2E2FF", "#E2E2E2FF")
    alpha	 =  0.5
    cex	 =  1
    itemLabels	 =  TRUE
    labelCol	 =  #000000B3
    measureLabels	 =  FALSE
    precision	 =  3
    layout	 =  NULL
    layoutParams	 =  list()
    arrowSize	 =  0.5
    engine	 =  igraph
    plot	 =  TRUE
    plot_options	 =  list()
    max	 =  100
    verbose	 =  FALSE
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Association%20Rules/output_32_1.png)


The above graph shows us that most of our transactions were consolidated around “Whole Milk”.

We also see that all liquor and wine are very strongly associated so we must place these together.

Another association we see from this graph is that the people who buy tropical fruits and herbs also buy rolls and buns. We should place these in an aisle together.

The next plot offers us a parallel coordinate system of visualisation. It would help us clearly see that which products along with which ones, result in what kinds of sales.

As mentioned above, the RHS is the Consequent or the item we propose the customer will buy; the positions are in the LHS where 2 is the most recent addition to our basket and 1 is the item we previously had.


```R
plot(rules[1:20],method = "paracoord",control = list(reorder = TRUE))
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Association%20Rules/output_35_0.png)


The topmost rule shows us that when I have whole milk and soups in my shopping cart, I am highly likely to buy other vegetables to go along with those as well.


```R
plot(rules[1:20],method = "matrix",control = list(reorder = 'support/confidence' ))
```

    Itemsets in Antecedent (LHS)
     [1] "{herbs,rolls/buns}"                         
     [2] "{tropical fruit,herbs}"                     
     [3] "{liquor,red/blush wine}"                    
     [4] "{yogurt,rice}"                              
     [5] "{herbs,shopping bags}"                      
     [6] "{yogurt,cereals}"                           
     [7] "{butter,rice}"                              
     [8] "{napkins,house keeping products}"           
     [9] "{whipped/sour cream,house keeping products}"
    [10] "{turkey,curd}"                              
    [11] "{rice,sugar}"                               
    [12] "{rice,bottled water}"                       
    [13] "{oil,mustard}"                              
    [14] "{herbs,fruit/vegetable juice}"              
    [15] "{soups,bottled beer}"                       
    [16] "{domestic eggs,rice}"                       
    [17] "{canned fish,hygiene articles}"             
    [18] "{curd,cereals}"                             
    [19] "{butter,jam}"                               
    [20] "{pastry,sweet spreads}"                     
    Itemsets in Consequent (RHS)
    [1] "{other vegetables}" "{whole milk}"       "{bottled beer}"    
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Association%20Rules/output_37_1.png)



```R
arulesViz::plotly_arules(rules)
```

    To reduce overplotting, jitter is added! Use jitter = 0 to prevent jitter.
    


The plot uses the arulesViz package and plotly to generate an interactive plot. We can hover over each rule and see the Support, Confidence and Lift.

As the interactive plot suggests, one rule that has a confidence of 1 is the one above. It has an exceptionally high lift as well, at 5.17.

# Conclusion

As shown above the top 20 items sets that can be part of a user's cart. Hence association rules generate rules for these associations and the number of rules can be decided on different values of support, confidence and lift.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Association-Rules" role="tab" aria-controls="settings">Go To Top<span class="badge badge-primary badge-pill"></span></a>

### FP Growth Algorithm

FP-growth is an improved version of the Apriori Algorithm which is widely used for frequent pattern mining(AKA Association Rule Mining). It is used as an analytical process that finds frequent patterns or associations from data sets. For example, grocery store transaction data might have a frequent pattern that people usually buy chips and beer together. The Apriori Algorithm produces frequent patterns by generating itemsets and discovering the most frequent itemset over a threshold “minimal support count”. It greatly reduces the size of the itemset in the database by one simple principle:
If an itemset is frequent, then all of its subsets must also be frequent.

**Problem Statement**

To determine the association of income levels with various factors

**Step 1:** Importing the required libraries


```R
#importing fpgrowth algorithm
library("rCBA")
```

    Loading required package: rJava
    

**Step 2:** Loading and exploring the data

The first part of any analysis is to bring in the dataset. We will be using an inbuilt dataset “Adult” from the ‘rCBA’ package to simplify our analysis.

All reasons and income levels are present in a specific type of dataset called the “Transaction” type dataset.


```R
data(Adult)
```

Before we begin applying the “FP Growth” algorithm on our dataset, we need to make sure that it is of the type “Transactions”.


```R
# Exploring the data 
str(Adult)
```

    Formal class 'transactions' [package "arules"] with 3 slots
      ..@ data       :Formal class 'ngCMatrix' [package "Matrix"] with 5 slots
      .. .. ..@ i       : int [1:612200] 1 10 25 32 35 50 59 61 63 65 ...
      .. .. ..@ p       : int [1:48843] 0 13 26 39 52 65 78 91 104 117 ...
      .. .. ..@ Dim     : int [1:2] 115 48842
      .. .. ..@ Dimnames:List of 2
      .. .. .. ..$ : NULL
      .. .. .. ..$ : NULL
      .. .. ..@ factors : list()
      ..@ itemInfo   :'data.frame':	115 obs. of  3 variables:
      .. ..$ labels   : chr [1:115] "age=Young" "age=Middle-aged" "age=Senior" "age=Old" ...
      .. ..$ variables: Factor w/ 13 levels "age","capital-gain",..: 1 1 1 1 13 13 13 13 13 13 ...
      .. ..$ levels   : Factor w/ 112 levels "10th","11th",..: 111 63 92 69 30 54 65 82 90 91 ...
      ..@ itemsetInfo:'data.frame':	48842 obs. of  1 variable:
      .. ..$ transactionID: chr [1:48842] "1" "2" "3" "4" ...
    

The structure of our transaction type dataset shows us that it is internally divided into three slots: Data, itemInfo and itemsetInfo.

The slot “Data” contains the dimensions, dimension names and other factors which determines the income level of an individual.


```R
# Exploring the different transactions 
head(Adult@itemInfo, n=20)
```


<table>
<thead><tr><th scope=col>labels</th><th scope=col>variables</th><th scope=col>levels</th></tr></thead>
<tbody>
	<tr><td>age=Young                 </td><td>age                       </td><td>Young                     </td></tr>
	<tr><td>age=Middle-aged           </td><td>age                       </td><td>Middle-aged               </td></tr>
	<tr><td>age=Senior                </td><td>age                       </td><td>Senior                    </td></tr>
	<tr><td>age=Old                   </td><td>age                       </td><td>Old                       </td></tr>
	<tr><td>workclass=Federal-gov     </td><td>workclass                 </td><td>Federal-gov               </td></tr>
	<tr><td>workclass=Local-gov       </td><td>workclass                 </td><td>Local-gov                 </td></tr>
	<tr><td>workclass=Never-worked    </td><td>workclass                 </td><td>Never-worked              </td></tr>
	<tr><td>workclass=Private         </td><td>workclass                 </td><td>Private                   </td></tr>
	<tr><td>workclass=Self-emp-inc    </td><td>workclass                 </td><td>Self-emp-inc              </td></tr>
	<tr><td>workclass=Self-emp-not-inc</td><td>workclass                 </td><td>Self-emp-not-inc          </td></tr>
	<tr><td>workclass=State-gov       </td><td>workclass                 </td><td>State-gov                 </td></tr>
	<tr><td>workclass=Without-pay     </td><td>workclass                 </td><td>Without-pay               </td></tr>
	<tr><td>education=Preschool       </td><td>education                 </td><td>Preschool                 </td></tr>
	<tr><td>education=1st-4th         </td><td>education                 </td><td>1st-4th                   </td></tr>
	<tr><td>education=5th-6th         </td><td>education                 </td><td>5th-6th                   </td></tr>
	<tr><td>education=7th-8th         </td><td>education                 </td><td>7th-8th                   </td></tr>
	<tr><td>education=9th             </td><td>education                 </td><td>9th                       </td></tr>
	<tr><td>education=10th            </td><td>education                 </td><td>10th                      </td></tr>
	<tr><td>education=11th            </td><td>education                 </td><td>11th                      </td></tr>
	<tr><td>education=12th            </td><td>education                 </td><td>12th                      </td></tr>
</tbody>
</table>



**Step 3:** Summary of the Data

Let us check the most frequently purchased products using the summary function.


```R
summary(Adult)
```


    transactions as itemMatrix in sparse format with
     48842 rows (elements/itemsets/transactions) and
     115 columns (items) and a density of 0.1089939 
    
    most frequent items:
               capital-loss=None            capital-gain=None 
                           46560                        44807 
    native-country=United-States                   race=White 
                           43832                        41762 
               workclass=Private                      (Other) 
                           33906                       401333 
    
    element (itemset/transaction) length distribution:
    sizes
        9    10    11    12    13 
       19   971  2067 15623 30162 
    
       Min. 1st Qu.  Median    Mean 3rd Qu.    Max. 
       9.00   12.00   13.00   12.53   13.00   13.00 
    
    includes extended item information - examples:
               labels variables      levels
    1       age=Young       age       Young
    2 age=Middle-aged       age Middle-aged
    3      age=Senior       age      Senior
    
    includes extended transaction information - examples:
      transactionID
    1             1
    2             2
    3             3


The summary statistics show us the top 5 factors that are common in defining the average income of an individual with respect to age, geography and demography.

To parse to Transaction type, make sure your dataset has similar slots and then use the as() function in R.

**Step 4:** Generating association rules using the FP Growth Algorithm


```R
rules = rCBA::fpgrowth(Adult, support=0.03, confidence=0.03,consequent="income",parallel=FALSE)
```

    2020-09-18 16:51:53 rCBA: initialized
    2020-09-18 16:52:52 rCBA: data 48842x115
    	 took: 58.62  s
    2020-09-18 16:52:57 rCBA: rules 2970
    	 took: 5.28  s
    

We will set minimum support parameter (minSup) to .03.

We can set minimum confidence (minConf) to anywhere between 0.03 and 0.05 for varied results.

We will set the consequent as income as we have to find the income with levels small and large (>50K).

**Support:** Support is the basic probability of an event to occur. If we have an event to buy product A, Support(A) is the number of transactions which includes A divided by total number of transactions.

**Confidence:** The confidence of an event is the conditional probability of the occurrence; the chances of A happening given B has already happened.

**Lift:** This is the ratio of confidence to expected confidence.The probability of all of the items in a rule occurring together (otherwise known as the support) divided by the product of the probabilities of the items on the left and right side occurring as if there was no association between them.

The lift value tells us how much better a rule is at predicting something than randomly guessing. The higher the lift, the stronger the association.

Let’s find out the top 10 rules arranged by lift.

**Step 5:** Lets inspect the Rules


```R
inspect(rules[1:10])
```

         lhs                               rhs               support confidence      lift
    [1]  {occupation=Machine-op-inspct} => {income=small} 0.03587077  0.5797485 1.1454724
    [2]  {occupation=Machine-op-inspct,                                                  
          capital-gain=None}            => {income=small} 0.03421236  0.5900424 1.1658111
    [3]  {occupation=Machine-op-inspct,                                                  
          capital-gain=None,                                                             
          workclass=Private}            => {income=small} 0.03296343  0.5940959 1.1738201
    [4]  {occupation=Machine-op-inspct,                                                  
          capital-gain=None,                                                             
          workclass=Private,                                                             
          capital-loss=None}            => {income=small} 0.03200115  0.5986212 1.1827612
    [5]  {occupation=Machine-op-inspct,                                                  
          capital-gain=None,                                                             
          capital-loss=None}            => {income=small} 0.03322960  0.5951595 1.1759216
    [6]  {occupation=Machine-op-inspct,                                                  
          workclass=Private}            => {income=small} 0.03456042  0.5857044 1.1572400
    [7]  {occupation=Machine-op-inspct,                                                  
          workclass=Private,                                                             
          capital-loss=None}            => {income=small} 0.03359813  0.5896515 1.1650387
    [8]  {occupation=Machine-op-inspct,                                                  
          capital-loss=None}            => {income=small} 0.03488801  0.5841618 1.1541922
    [9]  {workclass=Local-gov}          => {income=small} 0.03021989  0.4706633 0.9299407
    [10] {workclass=Self-emp-not-inc}   => {income=small} 0.03720159  0.4704816 0.9295818
    


```R
arules::itemFrequencyPlot(Adult,topN=20,col=brewer.pal(8,'Pastel2'),main='Relative Item Frequency Plot',type="relative",ylab="Item Frequency (Relative)")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Association%20Rules/output_60_0.png)


These histograms depict how many times an item has occurred in our dataset as compared to the others.

Moving forward in the visualisation, we can use a graph to highlight the support and lifts of various items in our repository but mostly to see which product is associated with which one in the sales environment.

**The size of graph nodes is based on support levels and the colour on lift ratios. The incoming lines show the Antecedants or the LHS and the RHS is represented by names of items.**


```R
plot(rules[1:20], method = "graph",control = list(type = "items"))
```

    Available control parameters (with default values):
    main	 =  Graph for 20 rules
    nodeColors	 =  c("#66CC6680", "#9999CC80")
    nodeCol	 =  c("#EE0000FF", "#EE0303FF", "#EE0606FF", "#EE0909FF", "#EE0C0CFF", "#EE0F0FFF", "#EE1212FF", "#EE1515FF", "#EE1818FF", "#EE1B1BFF", "#EE1E1EFF", "#EE2222FF", "#EE2525FF", "#EE2828FF", "#EE2B2BFF", "#EE2E2EFF", "#EE3131FF", "#EE3434FF", "#EE3737FF", "#EE3A3AFF", "#EE3D3DFF", "#EE4040FF", "#EE4444FF", "#EE4747FF", "#EE4A4AFF", "#EE4D4DFF", "#EE5050FF", "#EE5353FF", "#EE5656FF", "#EE5959FF", "#EE5C5CFF", "#EE5F5FFF", "#EE6262FF", "#EE6666FF", "#EE6969FF", "#EE6C6CFF", "#EE6F6FFF", "#EE7272FF", "#EE7575FF",  "#EE7878FF", "#EE7B7BFF", "#EE7E7EFF", "#EE8181FF", "#EE8484FF", "#EE8888FF", "#EE8B8BFF", "#EE8E8EFF", "#EE9191FF", "#EE9494FF", "#EE9797FF", "#EE9999FF", "#EE9B9BFF", "#EE9D9DFF", "#EE9F9FFF", "#EEA0A0FF", "#EEA2A2FF", "#EEA4A4FF", "#EEA5A5FF", "#EEA7A7FF", "#EEA9A9FF", "#EEABABFF", "#EEACACFF", "#EEAEAEFF", "#EEB0B0FF", "#EEB1B1FF", "#EEB3B3FF", "#EEB5B5FF", "#EEB7B7FF", "#EEB8B8FF", "#EEBABAFF", "#EEBCBCFF", "#EEBDBDFF", "#EEBFBFFF", "#EEC1C1FF", "#EEC3C3FF", "#EEC4C4FF", "#EEC6C6FF", "#EEC8C8FF",  "#EEC9C9FF", "#EECBCBFF", "#EECDCDFF", "#EECFCFFF", "#EED0D0FF", "#EED2D2FF", "#EED4D4FF", "#EED5D5FF", "#EED7D7FF", "#EED9D9FF", "#EEDBDBFF", "#EEDCDCFF", "#EEDEDEFF", "#EEE0E0FF", "#EEE1E1FF", "#EEE3E3FF", "#EEE5E5FF", "#EEE7E7FF", "#EEE8E8FF", "#EEEAEAFF", "#EEECECFF", "#EEEEEEFF")
    edgeCol	 =  c("#474747FF", "#494949FF", "#4B4B4BFF", "#4D4D4DFF", "#4F4F4FFF", "#515151FF", "#535353FF", "#555555FF", "#575757FF", "#595959FF", "#5B5B5BFF", "#5E5E5EFF", "#606060FF", "#626262FF", "#646464FF", "#666666FF", "#686868FF", "#6A6A6AFF", "#6C6C6CFF", "#6E6E6EFF", "#707070FF", "#727272FF", "#747474FF", "#767676FF", "#787878FF", "#7A7A7AFF", "#7C7C7CFF", "#7E7E7EFF", "#808080FF", "#828282FF", "#848484FF", "#868686FF", "#888888FF", "#8A8A8AFF", "#8C8C8CFF", "#8D8D8DFF", "#8F8F8FFF", "#919191FF", "#939393FF",  "#959595FF", "#979797FF", "#999999FF", "#9A9A9AFF", "#9C9C9CFF", "#9E9E9EFF", "#A0A0A0FF", "#A2A2A2FF", "#A3A3A3FF", "#A5A5A5FF", "#A7A7A7FF", "#A9A9A9FF", "#AAAAAAFF", "#ACACACFF", "#AEAEAEFF", "#AFAFAFFF", "#B1B1B1FF", "#B3B3B3FF", "#B4B4B4FF", "#B6B6B6FF", "#B7B7B7FF", "#B9B9B9FF", "#BBBBBBFF", "#BCBCBCFF", "#BEBEBEFF", "#BFBFBFFF", "#C1C1C1FF", "#C2C2C2FF", "#C3C3C4FF", "#C5C5C5FF", "#C6C6C6FF", "#C8C8C8FF", "#C9C9C9FF", "#CACACAFF", "#CCCCCCFF", "#CDCDCDFF", "#CECECEFF", "#CFCFCFFF", "#D1D1D1FF",  "#D2D2D2FF", "#D3D3D3FF", "#D4D4D4FF", "#D5D5D5FF", "#D6D6D6FF", "#D7D7D7FF", "#D8D8D8FF", "#D9D9D9FF", "#DADADAFF", "#DBDBDBFF", "#DCDCDCFF", "#DDDDDDFF", "#DEDEDEFF", "#DEDEDEFF", "#DFDFDFFF", "#E0E0E0FF", "#E0E0E0FF", "#E1E1E1FF", "#E1E1E1FF", "#E2E2E2FF", "#E2E2E2FF", "#E2E2E2FF")
    alpha	 =  0.5
    cex	 =  1
    itemLabels	 =  TRUE
    labelCol	 =  #000000B3
    measureLabels	 =  FALSE
    precision	 =  3
    layout	 =  NULL
    layoutParams	 =  list()
    arrowSize	 =  0.5
    engine	 =  igraph
    plot	 =  TRUE
    plot_options	 =  list()
    max	 =  100
    verbose	 =  FALSE
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Association%20Rules/output_62_1.png)


The above graph shows us that most of our transactions were consolidated around “Income”.

We also see that factors like occupation and workclass are strongly associated with income and hence are placed close to each other.

Another association we see from this graph is that the white people who are male depicts more data in income.


```R
plot(rules[1:20],method = "paracoord",control = list(reorder = TRUE))
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/Association%20Rules/output_64_0.png)



```R
arulesViz::plotly_arules(rules)
```

    To reduce overplotting, jitter is added! Use jitter = 0 to prevent jitter.
    




# Conclusion

As demostrated above we can understand the dependence of income level over various factors like demographics, age, sex etc. Hence association rules generate rules for these associations and the number of rules can be decided on different values of support, confidence and lift.
