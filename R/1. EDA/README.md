# EDA

# Notebook Content
[Libraries required](#Library)<br>
[Exploratory Data Analysis](#Exploratory-Data-Analysis)<br>
## Univariate Analysis
[Categorical Variables](#Categorical-Variables)<br>
[Pie Chart](#Pie-Chart)<br>
[Bar Plot](#Bar-Plot)<br>
[Frequency Table](#Frequency-Table)<br>
[Numerical Variables](#Numerical-Variables)<br>
[Histograms](#Histograms)<br>
[Boxplot](#Boxplot)<br>
[Density Plot](#Density-Plot)<br>
[Table](#Table)<br>
## Bivariate Analysis
### Numerical-Numerical
[Scatter plot](#Scatter-plot)<br>
[Adding Best Fit Lines](#Adding-best-fit-lines)<br>
[Line Plot](#Line-Plot)<br>
## Categorical-Numerical
[Bar Plot](#Bar-Plot)<br>
[Box Plot](#Box-Plot)<br>
[Violin Plot](#Violin-Plot)<br>
[T-Test](#T-Test)<br>
[Chi Square Test](#Chi-Square-Test)<br>
## Categorical-Categorical
[Stacked Bar Chart](#Stacked-bar-chart)<br>
[Grouped Bar Chart](#Grouped-bar-chart)<br>
[Density Plot](#Grouped-kernel-density-plot)<br>
[Ridgeline Plot](#Ridgeline-Plot)<br>
[Z Tests](#Z-Test)<br>
## Multivariate Analysis
[Grouping](#Grouping)<br>
[Faceting](#Faceting)<br>
[Correlation](#Correlation)<br>
[Mosaic Plots](#Mosaic-Plots)<br>
[3D Scatter Plot](#3D-Scatter-Plot)<br>
[Bubble Plot](#Bubble-Plot)<br>
[Scatter Plot Matrix](#Scatter-Plot-Matrix)<br>
[Chi Square Test(categorical data)](#10)<br>
[Z-Score](#Z-Score)<br>
[IQR Score](#IQR-Score)<br>
[Removing outliers - quick & dirty](#20)<br>

# Library


```python
# install.packages("scatterplot3d")
# install.packages("ggridges")
# install.packages("ggcorrplot")
# install.packages("PCAmixdata")
# install.packages("vcd")
# install.packages("readr")
# install.packages("gapminder")
# install.packages("BSDA")
# install.packages("GGally")
# install.packages("rcompanion")
# install.packages('GGally')
# install.packages('plotscale')
# install.packages('GGally')
# install.packages('outliers')
# library(dplyr)
# library(ggplot2)
# library(plotscale)
```

# Exploratory Data Analysis
Exploratory Data Analysis refers to the critical process of performing initial investigations on data to discover patterns, to spot anomalies, to test hypothesis and to check assumptions with the help of summary statistics and graphical representations. There are three types of EDA:<br>
1. Univariate Analysis<br>
2. Bivariate Analysis<br>
3. Multivariate Analysis<br>

Each of these will be explained in detail below:

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

The dataset can be downloaded from the link given below:<br>
https://www.kaggle.com/toramky/automobile-dataset

## Load the requred Libraries and Dataset


```python
auto_mobile = read.csv('dataset/auto_mobile.csv')
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



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

# UNIVARIATE ANALYSIS 

Univariate analysis explores variables (attributes) one by one. Variables could be either categorical or numerical. There are different statistical and visualization techniques of investigation for each type of variable. Numerical variables can be transformed into categorical counterparts by a process called binning or discretization. It is also possible to transform a categorical variable into its numerical counterpart by a process called encoding. <br>
<br>**Univariate Analysis** can be further classified in two categories :
1. Categorical
2. Numerical



### Categorical Variables
A categorical or discrete variable is one that has two or more categories (values). There are two types of categorical variables, nominal and ordinal.  A nominal variable has no intrinsic ordering to its categories. For example, gender is a categorical variable having two categories (male and female) with no intrinsic ordering to the categories. An ordinal variable has a clear ordering. For example, temperature as a variable with three orderly categories (low, medium and high).<br>A frequency table is a way of counting how often each category of the variable in question occurs. It may be enhanced by the addition of percentages that fall into each category:<br>

|Statistics|Visualization|Description|
|----------|-------------|-----------|
|Count|Bar Chart|The number of values of the specified variable|
|Count%|Pie Chart|The percentage of values of the specified variable|



### Pie Chart
A pie chart is a **circular statistical diagram**. The area of the whole chart represents 100% or the whole of the data. The **areas of the pies present in the Pie chart represent the percentage of parts of data**. The parts of a pie chart are called **wedges**. The length of the arc of a wedge determines the area of a wedge in a pie chart. The area of the wedges determines the relative quantum or percentage of a part with respect to a whole. Pie charts are frequently used in business presentations as they give quick summary of the business activities like sales, operations and so on. Pie charts are also used heavily in survey results, news articles, resource usage diagrams like disk and memory.


```python
size = table(auto_mobile$body.style)
contri  <- round(100 * size / sum(size))
contri

```


    
    convertible     hardtop   hatchback       sedan       wagon 
              3           4          34          46          12 



```python
y = paste(names(contri))
pie(contri,labels = contri,col = rainbow(length(y)),
   main="Pie Chart of body-style\n (with sample sizes)")
legend('topright', c(y), fill = rainbow(length(y)))

```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_15_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#EDA" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


###  Bar Plot
A bar chart or bar graph is a chart or graph that **presents categorical data with rectangular bars with heights or lengths proportional to the values that they represent**.<br>

The bars can be plotted vertically or horizontally.<br>

A bar graph shows comparisons among discrete categories. One axis of the chart shows the specific categories being compared, and the other axis represents a measured value.


```python
count = table(auto_mobile$num.of.cylinders)
barplot(count, main="Count of number of cylinder in each cylinder type",  xlab="Number of cylinder",ylab="Count")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_18_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#EDA" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Frequency Table
Frequency table checks the **count of each category in a particular column of the data** and along with this percentage of data in each category can also be found out.


```python
make = table(auto_mobile$make)

precentage_of_make  <- round(100 * make / sum(make),2)
make_df = data.frame(make,precentage_of_make)
make_df = select(make_df, -Var1.1)
colnames(make_df) = c("make_type", "make_freq", "percentage(%) of make")
make_df

```


<table>
<caption>A data.frame: 22 × 3</caption>
<thead>
	<tr><th scope=col>make_type</th><th scope=col>make_freq</th><th scope=col>percentage(%) of make</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>alfa-romero  </td><td> 3</td><td> 1.48</td></tr>
	<tr><td>audi         </td><td> 7</td><td> 3.45</td></tr>
	<tr><td>bmw          </td><td> 8</td><td> 3.94</td></tr>
	<tr><td>chevrolet    </td><td> 3</td><td> 1.48</td></tr>
	<tr><td>dodge        </td><td> 8</td><td> 3.94</td></tr>
	<tr><td>honda        </td><td>13</td><td> 6.40</td></tr>
	<tr><td>isuzu        </td><td> 4</td><td> 1.97</td></tr>
	<tr><td>jaguar       </td><td> 3</td><td> 1.48</td></tr>
	<tr><td>mazda        </td><td>16</td><td> 7.88</td></tr>
	<tr><td>mercedes-benz</td><td> 8</td><td> 3.94</td></tr>
	<tr><td>mercury      </td><td> 1</td><td> 0.49</td></tr>
	<tr><td>mitsubishi   </td><td>13</td><td> 6.40</td></tr>
	<tr><td>nissan       </td><td>18</td><td> 8.87</td></tr>
	<tr><td>peugot       </td><td>11</td><td> 5.42</td></tr>
	<tr><td>plymouth     </td><td> 7</td><td> 3.45</td></tr>
	<tr><td>porsche      </td><td> 5</td><td> 2.46</td></tr>
	<tr><td>renault      </td><td> 2</td><td> 0.99</td></tr>
	<tr><td>saab         </td><td> 6</td><td> 2.96</td></tr>
	<tr><td>subaru       </td><td>12</td><td> 5.91</td></tr>
	<tr><td>toyota       </td><td>32</td><td>15.76</td></tr>
	<tr><td>volkswagen   </td><td>12</td><td> 5.91</td></tr>
	<tr><td>volvo        </td><td>11</td><td> 5.42</td></tr>
</tbody>
</table>



From the table above it can be said that toyota makes the highest and mercury makes the lowest number of cars. Similarly such tables can be made for all the categorical variables.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#EDA" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


# Numerical Variables

### Histograms
The purpose of a histogram is to **graphically summarize the distribution of a univariate data set**.<br><br>
**The histogram graphically shows the following:** centre  (i.e., the location) of the data, spread (i.e., the scale) of the data, skewness of the data, presence of outliers, presence of multiple modes in the data.<br>
The most common form of the histogram is obtained by splitting the range of the data into equal-sized bins (called classes). Then for each bin, the number of points from the data set that fall into each bin are counted. That is;<br>
Vertical axis: Frequency (i.e., counts for each bin) <br>
Horizontal axis: Response variable <br>

_The histogram can be used to answer the following questions:_ <br>
1. What kind of population distribution do the data come from<br>
2. Where are the data located<br>
3. How spread out are the data<br>
4. Are the data symmetric or skewed<br> 
5. Are there outliers in the data<br>
The code to plot a histogram has been given below:


```python
# histogram from height
hist(auto_mobile$height, # histogram
 col="peachpuff", # column color
 border="black",
 prob = TRUE, # show densities instead of frequencies
 xlab = "Height",
 main = "Histogram for Height")

# histogram from weight
hist(auto_mobile$price, # histogram
 col="peachpuff", # column color
 border="black",
 prob = TRUE, # show densities instead of frequencies
 xlab = "price",
 main = "Histogram for price")

```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_26_0.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_26_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#EDA" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Density Plot
A density plot is a **smoothed, continuous version of a histogram** estimated from the data.<br>

The x-axis is the value of the variable just like in a histogram. The y-axis in a density plot is the probability density function for the kernel density estimation. However, one should carefully specify **this is a probability density and not a probability**. The difference is the probability density is the probability per unit on the x-axis. To convert to an actual probability, we need to find the area under the curve for a specific interval on the x-axis.<br>

Because this is a probability density and not a probability, the y-axis can take values greater than one. The only requirement of the density plot is that the total area under the curve integrates to one.<br>

Think of the y-axis on a density plot as a value only for relative comparisons between different categories.

The code to make a density plot is given below:


```python
Body_Style = auto_mobile$body.style
ggplot(auto_mobile, aes(x=Body_Style, col= Body_Style)) +
  geom_density()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_29_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

### Boxplot
When the data distribution is displayed in a standardized way using 5 summary – minimum, Q1 (First Quartile), median, Q3(third Quartile), and maximum, it is called a Box plot.<br>

It is also termed as box and whisker plot when the lines extending from the boxes indicate variability outside the upper and lower quartiles.<br>

Outliers can be plotted as unique points.
<br>

**Application of Boxplot:**
It is used to know:<br>
1. The outliers and its values<br>
2. Symmetry of Data<br>
3. Tight grouping of data<br>
4. Data skewness -if, in which direction and how<br>


```python
# box plot for length
boxplot(auto_mobile$length,
main = "Boxplot of length",
xlab = "Length",
col = "orange",
border = "brown",
horizontal = TRUE
)

# box plot for width
boxplot(auto_mobile$width,
main = "Boxplot of width",
xlab = "Width",
col = "blue",
border = "brown",
horizontal = TRUE)

# box plot for width
boxplot(auto_mobile$height,
main = "Boxplot of height",
xlab = "height",
col = "green",
border = "green",
horizontal = TRUE)


```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_32_0.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_32_1.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_32_2.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


### Table
Using the function given below, we can make a table to show all the statistics that are used in bivariate analysis. Please note that using df.describe() also gives us a table of similar structure but would not include values for some statistics like skewness, variance, etc.


```python
summary(auto_mobile)
```


           X           symboling       normalized.losses         make   
     Min.   :  0.0   Min.   :-2.0000   Min.   : 65.0     toyota    :32  
     1st Qu.: 51.5   1st Qu.: 0.0000   1st Qu.:101.0     nissan    :18  
     Median :103.0   Median : 1.0000   Median :122.0     mazda     :16  
     Mean   :102.6   Mean   : 0.8374   Mean   :121.9     honda     :13  
     3rd Qu.:153.5   3rd Qu.: 2.0000   3rd Qu.:137.0     mitsubishi:13  
     Max.   :204.0   Max.   : 3.0000   Max.   :256.0     subaru    :12  
                                                         (Other)   :99  
      fuel.type   aspiration  num.of.doors       body.style drive.wheels
     diesel: 19   std  :167   four:114     convertible: 6   4wd:  9     
     gas   :184   turbo: 36   two : 89     hardtop    : 8   fwd:118     
                                           hatchback  :70   rwd: 76     
                                           sedan      :94               
                                           wagon      :25               
                                                                        
                                                                        
     engine.location   wheel.base         length          width      
     front:200       Min.   : 86.60   Min.   :141.1   Min.   :60.30  
     rear :  3       1st Qu.: 94.50   1st Qu.:166.6   1st Qu.:64.10  
                     Median : 97.00   Median :173.2   Median :65.50  
                     Mean   : 98.78   Mean   :174.1   Mean   :65.92  
                     3rd Qu.:102.40   3rd Qu.:183.3   3rd Qu.:66.90  
                     Max.   :120.90   Max.   :208.1   Max.   :72.30  
                                                                     
         height       curb.weight   engine.type num.of.cylinders  engine.size   
     Min.   :47.80   Min.   :1488   dohc : 12   eight :  5       Min.   : 61.0  
     1st Qu.:52.00   1st Qu.:2145   dohcv:  1   five  : 11       1st Qu.: 97.0  
     Median :54.10   Median :2414   l    : 12   four  :157       Median :120.0  
     Mean   :53.73   Mean   :2558   ohc  :146   six   : 24       Mean   :127.1  
     3rd Qu.:55.50   3rd Qu.:2944   ohcf : 15   three :  1       3rd Qu.:143.0  
     Max.   :59.80   Max.   :4066   ohcv : 13   twelve:  1       Max.   :326.0  
                                    rotor:  4   two   :  4                      
      fuel.system      bore           stroke      compression.ratio
     mpfi   :93   Min.   :2.540   Min.   :2.070   Min.   : 7.00    
     2bbl   :66   1st Qu.:3.150   1st Qu.:3.110   1st Qu.: 8.60    
     idi    :19   Median :3.310   Median :3.290   Median : 9.00    
     1bbl   :11   Mean   :3.331   Mean   :3.254   Mean   :10.09    
     spdi   : 9   3rd Qu.:3.590   3rd Qu.:3.410   3rd Qu.: 9.40    
     4bbl   : 3   Max.   :3.940   Max.   :4.170   Max.   :23.00    
     (Other): 2   NA's   :4       NA's   :4                        
       horsepower         peak.rpm       city.mpg      highway.mpg  
     Min.   :   48.0   Min.   :4150   Min.   :13.00   Min.   :16.0  
     1st Qu.:   70.0   1st Qu.:4800   1st Qu.:19.00   1st Qu.:25.0  
     Median :   95.0   Median :5200   Median :24.00   Median :30.0  
     Mean   :  233.6   Mean   :5126   Mean   :25.17   Mean   :30.7  
     3rd Qu.:  120.5   3rd Qu.:5500   3rd Qu.:30.00   3rd Qu.:34.0  
     Max.   :13207.0   Max.   :6600   Max.   :49.00   Max.   :54.0  
                       NA's   :2                                    
         price      
     Min.   : 5118  
     1st Qu.: 7782  
     Median :10595  
     Mean   :13242  
     3rd Qu.:16500  
     Max.   :45400  
                    


<a class="list-group-item list-group-item-action" data-toggle="list" href="#EDA" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

# Bivariate analysis
Bivariate analysis is performed to **find the relationship between each variable in the dataset and the target variable of interest** (or) using 2 variables and finding relationship  between them. Ex:-Box plot, Violin plot.<br><br>**Bivariate Analysis** can be further classified in broad the category 
1. Numerical- Numerical
2. Categorical- Categorical 
3. Numerical- Categorical 

## Numerical- Numerical

In this the relationship between the Numerical variables is studied by plotting various plot such as scatter plot, violin plot

### Scatter plot
A scatter plot (aka scatter chart, scatter graph) uses **dots to represent values for two different numeric variables**. The position of each dot on the horizontal and vertical axis indicates values for an individual data point. Scatter plots are used to observe relationships between variables.<br><br>**When you should use a scatter plot**<br> Scatter plots’ primary uses are to observe and show relationships between two numeric variables. The dots in a scatter plot not only report the values of individual data points, but also patterns when the data are taken as a whole.


```python
# simple scatterplot
ggplot(auto_mobile, 
       aes(x = engine.size, 
           y = price)) +
  geom_point(color="cornflowerblue", 
             size = 2, 
             alpha=.8) +
  scale_y_continuous(label = scales::dollar) + 
  labs(x = "Years Since PhD",
       y = "",
       title = "Experience vs. Salary",
       subtitle = "9-month salary for 2008-2009")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_41_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

### Adding best fit lines
It is often useful to summarize the relationship displayed in the scatterplot, using a best fit line. Many types of lines are supported, including linear, polynomial, and nonparametric (loess). By default, 95% confidence limits for these lines are displayed.


```python
# scatterplot with linear fit line
ggplot(auto_mobile,
       aes(x = engine.size, 
           y = price)) +
  geom_point(color= "steelblue") +
  geom_smooth(method = "lm")
```

    `geom_smooth()` using formula 'y ~ x'
    
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_44_1.png)



```python
# scatterplot with quadratic line of best fit
ggplot(auto_mobile, 
       aes(x = engine.size, 
           y = horsepower)) +
  geom_point(color= "steelblue") +
  geom_smooth(method = "lm", 
              formula = y ~ poly(x, 2), 
              color = "indianred3")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_45_0.png)


Scatter plots can also be made using the scatter() function


```python
# Simple Scatterplot
attach(auto_mobile)
scatter <- plot(price, engine.size, main="Scatterplot Example",
   xlab="Price ", ylab="Engine Size ", pch=19, col="yellowgreen")  #Type = "p" can also be used in place of pch
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_47_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#EDA" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


## Line Plot
When one of the two variables represents time, a line plot can be an effective method of displaying relationship. For example, the code below displays the relationship between time (year) and life expectancy (lifeExp) in the United States between 1952 and 2007. The data comes from the gapminder dataset.


```python
# use this library for bellow line package dataset
# install.packages('gapminder')
```


```python
# data used from gapminder, any other relavent data can be used fot thr line plot
data(gapminder, package="gapminder")

# Select US cases
library(dplyr)
plotdata <- filter(gapminder, 
                   country == "United States")

# simple line plot
ggplot(plotdata, 
       aes(x = year, 
           y = lifeExp)) +
  geom_line() 
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_51_0.png)



```python
# line plot with points
# and improved labeling
ggplot(plotdata, 
       aes(x = year, 
           y = lifeExp)) +
  geom_line(size = 1.5, 
            color = "lightgrey") +
  geom_point(size = 3, 
             color = "steelblue") +
  labs(y = "Life Expectancy (years)", 
       x = "Year",
       title = "Life expectancy changes over time",
       subtitle = "United States (1952-2007)",
       caption = "Source: http://www.gapminder.org/data/")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_52_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


## Categorical-Numerical

# Bar Plot
Bar plot for categorical and Numerical variables.


```python
# plot mean price 
# calculate mean price for each body style
# price <= auto_mobile$price
library(dplyr)
plotdata <- auto_mobile %>%
  group_by(body.style) %>%
  summarize(mean_price = mean(price))

# plot mean salaries
ggplot(plotdata, 
       aes(x = body.style, 
           y = mean_price)) +
  geom_bar(stat = "identity")

```

    `summarise()` ungrouping output (override with `.groups` argument)
    
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_56_1.png)


We can make it more attractive with some options.


```python
# plot mean salaries in a more attractive fashion
library(scales)
ggplot(plotdata, 
       aes(x = factor(body.style,
                      labels = c("convertible",
                                 "hardtop",
                                 "hatchback",
                                 "sedan", "wagon")), 
                      y = mean_price)) +
  geom_bar(stat = "identity", 
           fill = "cornflowerblue") +
  geom_text(aes(label = dollar(mean_price)), 
            vjust = -0.25) +
  scale_y_continuous(breaks = seq(0, 130000, 20000), 
                     label = dollar) +
  labs(title = "Mean price by body style", 
       subtitle = "Five body styles in total",
       x = "",
       y = "")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_58_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

## Box Plot
Box Plot for bivariate analysis.


```python
# plot the distribution of salaries by rank using boxplots
ggplot(auto_mobile, aes(x = fuel.type, 
                     y = price)) +
  geom_boxplot(notch = TRUE, 
               fill = "lightskyblue", 
               alpha = .7) +
  labs(title = "price distribution by fuel type")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_61_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

## Violin Plot
A violin plot is a method of plotting numeric data. It is similar to a box plot, with the addition of a rotated kernel density plot on each side. Violin plots are similar to box plots, except that they also show the probability density of the data at different values, usually smoothed by a kernel density estimator.


```python
# plot the distribution of price 
# by body style using violin plots
ggplot(auto_mobile, 
       aes(x = body.style,
           y = price)) +
  geom_violin(scale = "count", aes(fill = body.style), draw_quantiles = c(0.25, 0.5, 0.75)) + geom_jitter(height = 0, width = 0.1)
  labs(title = "Salary distribution by rank")
```


    $title
    [1] "Salary distribution by rank"
    
    attr(,"class")
    [1] "labels"



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_64_1.png)


A useful variation is to superimpose boxplots on violin plots


```python
# plot the distribution using violin and boxplots
ggplot(auto_mobile, 
       aes(x = body.style, 
           y = price)) +
  geom_violin(fill = "lightblue") +
  geom_boxplot(width = .2, 
               fill = "lightgreen",
               outlier.color = "black",
               outlier.size = 2) + 
  labs(title = "Price distribution by body style")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_66_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

# T Test
The t test tells you how significant the differences between groups are; In other words it lets you know if those differences (measured in means/averages) could have happened by chance.<br><br>
T-test measures, if two samples are different from one another. One of these samples could be the population, however, T-test in place of a Z-test if the population’s standard deviation is unknown.
There are a lot of similar assumptions to the Z-test. The sample must be random and independently selected as well as drawn from the normal distribution. The values should also be numeric and continuous. The sample size does not necessarily have to be large.

<br>Interpretation depends on hypothesis, if P-Value is less than 0.05 then we must reject the null hypothesis
x̄1 is the mean of first data set<br>
x̄2 is the mean of second data set<br>
S12 is the standard deviation of first data set<br>
S22 is the standard deviation of second data set<br>
N1 is the number of elements in the first data set<br>
N2 is the number of elements in the second data set<br>


```python
# in the formula x~y, note that y should be data having exactly 2 levels
t.test(price, y = NULL,
       alternative = c("two.sided", "less", "greater"),
       mu = 0, paired = FALSE, var.equal = FALSE,
       conf.level = 0.95)

t.test(price~fuel.type, auto_mobile)
```


    
    	One Sample t-test
    
    data:  price
    t = 23.885, df = 202, p-value < 2.2e-16
    alternative hypothesis: true mean is not equal to 0
    95 percent confidence interval:
     12148.76 14335.06
    sample estimates:
    mean of x 
     13241.91 
    



    
    	Welch Two Sample t-test
    
    data:  price by fuel.type
    t = 1.6633, df = 21.87, p-value = 0.1105
    alternative hypothesis: true difference in means is not equal to 0
    95 percent confidence interval:
     -780.7396 7095.0714
    sample estimates:
    mean in group diesel    mean in group gas 
                16103.58             12946.41 
    


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

## Chi Square Test
The Chi-square test of independence works by comparing the observed frequencies (so the frequencies observed in your sample) to the expected frequencies if there was no relationship between the two categorical variables (so the expected frequencies if the null hypothesis was true)


```python
dat <- auto_mobile
dat$size <- ifelse(dat$length < median(dat$length),
  "small", "big")

table(dat$body.style, dat$size)
```


                 
                  big small
      convertible   2     4
      hardtop       5     3
      hatchback    23    47
      sedan        54    40
      wagon        19     6


The contingency table gives the observed number of cases in each subgroup.


```python
test <- chisq.test(table(dat$body.style, dat$size))
test
```

    Warning message in chisq.test(table(dat$body.style, dat$size)):
    "Chi-squared approximation may be incorrect"
    


    
    	Pearson's Chi-squared test
    
    data:  table(dat$body.style, dat$size)
    X-squared = 18.2, df = 4, p-value = 0.001128
    



```python
test$statistic
test$p.value
```


<strong>X-squared:</strong> 18.1999843592922



0.00112783261249972


From the output and from test$p.value we see that the p-value is less than the significance level of 0.05. Like any other statistical test, if the p-value is less than the significance level, we can reject the null hypothesis.
<br><br>In our context, rejecting the null hypothesis for the Chi-square test of independence means that there is a significant relationship between the body style and the length. Therefore, knowing the value of one variable helps to predict the value of the other variable.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


## Categorical-Categorical

## Stacked bar chart
The stacked plot can be used to analyse two categorical variables. In the example below, we have presented a stacked plot that shows the count of each of the fuel types for the different body styles.


```python
library(ggplot2)

# stacked bar chart
ggplot(auto_mobile, 
       aes(x = body.style, fill = fuel.type)) + 
  geom_bar(position = "stack", width=0.5)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_80_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

## Grouped bar chart
Grouped bar charts place bars for the second categorical variable side-by-side. To create a grouped bar plot use the position = "dodge" option. Shown below is a grouped bar chart for aspiration and body.style.


```python
library(ggplot2)

# grouped bar plot
ggplot(auto_mobile, 
       aes(x = body.style, 
           fill = aspiration)) + 
  geom_bar(position = "dodge", width = 0.5)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_83_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

## Grouped kernel density plot
One can compare groups on a numeric variable by superimposing kernel density plots in a single graph


```python
# plot the distribution of price 
# by fuel type using kernel density plots
ggplot(auto_mobile, 
       aes(x = price, 
           fill = fuel.type)) +
  geom_density(alpha = 0.4) +
  labs(title = "Price distribution by fuel type")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_86_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

## Ridgeline Plot
A ridgeline plot (also called a joyplot) displays the distribution of a quantitative variable for several groups. They’re similar to kernel density plots with vertical faceting but take up less room. Ridgeline plots are created with the ggridges package.

Using the auto_mobile dataset, let’s plot the distribution of price by make.


```python
# create ridgeline graph
library(ggplot2)
library(ggridges)

ggplot(auto_mobile, 
       aes(x = price, 
           y = make, 
           fill = make)) +
  geom_density_ridges() + 
  theme_ridges() +
  labs("Price by make") +
  theme(legend.position = "none")+
  scale_x_continuous(breaks = seq(0, 130000, 20000), 
                     label = dollar)
```

    Warning message:
    "package 'ggridges' was built under R version 3.6.3"
    Picking joint bandwidth of 1750
    
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_89_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


## Z-Test

A Z-test is a **type of hypothesis test**. Hypothesis testing is just a way for you to figure out if results from a test are valid or repeatable.

For example, if someone said they had found a new drug that cures cancer, you would want to be sure it was probably true. A hypothesis test will tell you if it is probably true, or probably not true. A Z-test, is used when data is approximately normally distributed.

<Br>Several different types of tests are used in statistics (i.e. f test, chi square test, t test). **A Z test is used if:**
1. Sample size is greater than 30. Otherwise, a t test can be used 
2. Data points should be independent from each other. In other words, one data point isn’t related or doesn’t affect another data point
3. Data should be normally distributed. However, for large sample sizes (over 30) this does not always matter
4. Data should be randomly selected from a population, where each item has an equal chance of being selected. Sample sizes should be equal if at all possible

Interpretation depends on hypothesis, if P-Value is less than 0.05 then the null hypothesis must be rejected.

The Test Statistic: When sample is taken from a normal distribution with known variance, then our test statistic is:


```python
library(BSDA)
options(warn=-1)

z.test(x=price, y = NULL, alternative = "two.sided", mu = 0, sigma.x = 0.5,
  sigma.y = 0.5, conf.level = 0.95)
```

    Warning message:
    "package 'BSDA' was built under R version 3.6.3"
    Loading required package: lattice
    
    
    Attaching package: 'BSDA'
    
    
    The following object is masked from 'package:datasets':
    
        Orange
    
    
    


    
    	One-sample z-Test
    
    data:  price
    z = 377336, p-value < 2.2e-16
    alternative hypothesis: true mean is not equal to 0
    95 percent confidence interval:
     13241.84 13241.98
    sample estimates:
    mean of x 
     13241.91 
    


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


# Multivariate Analysis

## Grouping
In grouping, the values of the first two variables are mapped to the x and y axes. Then additional variables are mapped to other visual characteristics such as color, shape, size, line type, and transparency. Grouping allows you to plot the data for multiple groups in a single graph.

Using the auto_mobile dataset, let’s display the relationship between make and price.


```python
# plot make vs. price (color represents fuel.type)
grouping1 <- ggplot(auto_mobile, aes(x = make, 
                     y = price, 
                     color=fuel.type)) +
  geom_point() +
  labs(title = "Price by make and fuel type") +
  scale_y_continuous(breaks = seq(0, 130000, 20000), 
                     label = dollar)
grouping1 + theme(axis.text.x = element_text(face = "bold", color = "black", 
                           size = 12, angle = 45),
          axis.text.y = element_text(face = "bold", color = "black", 
                           size = 12, angle = 45),
         panel.background = element_rect(fill = "lightgrey",
                                colour = "white",
                                size = 1, linetype = "solid")) 



```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_97_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


```python
# plot price vs. make 
# (color represents body style, shape represents fuel type)
grouping2 <- ggplot(auto_mobile, 
       aes(x = make, 
           y = price, 
           color = body.style, 
           shape = fuel.type)) +
  geom_point(size = 3, 
             alpha = .6) +
  labs(title = "Price by make, body style, and fuel type") +
  scale_y_continuous(breaks = seq(0, 130000, 20000), 
                     label = dollar)
grouping2 + theme(axis.text.x = element_text(face = "bold", color = "black", 
                           size = 12, angle = 45),
          axis.text.y = element_text(face = "bold", color = "black", 
                           size = 12, angle = 45))
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_99_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

## Faceting
Grouping allows you to plot multiple variables in a single graph, using visual characteristics such as color, shape, and size.

In faceting, a graph consists of several separate plots or small multiples, one for each level of a third variable, or combination of variables. It is easiest to understand this with an example.


```python
# plot salary histograms by rank
faceting1 <-ggplot(auto_mobile, aes(x = price)) +
  geom_histogram(fill = "cornflowerblue",
                 color = "white") +
  facet_wrap(~body.style, ncol = 1) +
  labs(title = "price histograms by body style") +
  scale_x_continuous(breaks = seq(0, 130000, 20000), 
                     label = dollar)
faceting1
```

    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_102_1.png)



```python
# plot salary histograms by rank and sex
faceting2 <- ggplot(auto_mobile, aes(x = price)) +
  geom_histogram(color = "white",
                 fill = "cornflowerblue") +
  facet_grid(fuel.type ~ body.style) +
  labs(title = "Price histograms by fuel type and body style",
       x = "Price ($)") +
  scale_x_continuous(breaks = seq(0, 130000, 20000), 
                     label = dollar)
faceting2 + theme(axis.text.x = element_text(face = "bold", color = "black", 
                           size = 12, angle = 45),
          axis.text.y = element_text(face = "bold", color = "black", 
                           size = 12, angle = 45))
```

    `stat_bin()` using `bins = 30`. Pick better value with `binwidth`.
    
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_103_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

## Correlation

**Pearson’s Correlation Coefficient**
Pearson’s correlation coefficient is the test statistics that measures the statistical relationship, or association, between two continuous variables.  It is known as the best method of measuring the association between variables of interest because it is based on the method of covariance.  It gives information about the magnitude of the association, or correlation, as well as the direction of the relationship.


```python
str(auto_mobile)
```

    'data.frame':	203 obs. of  27 variables:
     $ X                : int  0 1 2 3 4 5 6 7 8 9 ...
     $ symboling        : int  3 3 1 2 2 2 1 1 1 0 ...
     $ normalized.losses: int  122 122 122 164 164 122 158 122 158 122 ...
     $ make             : Factor w/ 22 levels "alfa-romero",..: 1 1 1 2 2 2 2 2 2 2 ...
     $ fuel.type        : Factor w/ 2 levels "diesel","gas": 2 2 2 2 2 2 2 2 2 2 ...
     $ aspiration       : Factor w/ 2 levels "std","turbo": 1 1 1 1 1 1 1 1 2 2 ...
     $ num.of.doors     : Factor w/ 2 levels "four","two": 2 2 2 1 1 2 1 1 1 2 ...
     $ body.style       : Factor w/ 5 levels "convertible",..: 1 1 3 4 4 4 4 5 4 3 ...
     $ drive.wheels     : Factor w/ 3 levels "4wd","fwd","rwd": 3 3 3 2 1 2 2 2 2 1 ...
     $ engine.location  : Factor w/ 2 levels "front","rear": 1 1 1 1 1 1 1 1 1 1 ...
     $ wheel.base       : num  88.6 88.6 94.5 99.8 99.4 ...
     $ length           : num  169 169 171 177 177 ...
     $ width            : num  64.1 64.1 65.5 66.2 66.4 66.3 71.4 71.4 71.4 67.9 ...
     $ height           : num  48.8 48.8 52.4 54.3 54.3 53.1 55.7 55.7 55.9 52 ...
     $ curb.weight      : int  2548 2548 2823 2337 2824 2507 2844 2954 3086 3053 ...
     $ engine.type      : Factor w/ 7 levels "dohc","dohcv",..: 1 1 6 4 4 4 4 4 4 4 ...
     $ num.of.cylinders : Factor w/ 7 levels "eight","five",..: 3 3 4 3 2 2 2 2 2 2 ...
     $ engine.size      : int  130 130 152 109 136 136 136 136 131 131 ...
     $ fuel.system      : Factor w/ 8 levels "1bbl","2bbl",..: 6 6 6 6 6 6 6 6 6 6 ...
     $ bore             : num  3.47 3.47 2.68 3.19 3.19 3.19 3.19 3.19 3.13 3.13 ...
     $ stroke           : num  2.68 2.68 3.47 3.4 3.4 3.4 3.4 3.4 3.4 3.4 ...
     $ compression.ratio: num  9 9 9 10 8 8.5 8.5 8.5 8.3 7 ...
     $ horsepower       : int  111 111 154 102 115 110 110 110 140 160 ...
     $ peak.rpm         : num  5000 5000 5000 5500 5500 5500 5500 5500 5500 5500 ...
     $ city.mpg         : int  21 21 19 24 18 19 19 19 17 16 ...
     $ highway.mpg      : int  27 27 26 30 22 25 25 25 20 22 ...
     $ price            : int  13495 16500 16500 13950 17450 15250 17710 18920 23875 13207 ...
    


```python
num_cols <- unlist(lapply(auto_mobile, is.numeric))         # Identify numeric columns
num_cols
```


<dl class=dl-inline><dt>X</dt><dd>TRUE</dd><dt>symboling</dt><dd>TRUE</dd><dt>normalized.losses</dt><dd>TRUE</dd><dt>make</dt><dd>FALSE</dd><dt>fuel.type</dt><dd>FALSE</dd><dt>aspiration</dt><dd>FALSE</dd><dt>num.of.doors</dt><dd>FALSE</dd><dt>body.style</dt><dd>FALSE</dd><dt>drive.wheels</dt><dd>FALSE</dd><dt>engine.location</dt><dd>FALSE</dd><dt>wheel.base</dt><dd>TRUE</dd><dt>length</dt><dd>TRUE</dd><dt>width</dt><dd>TRUE</dd><dt>height</dt><dd>TRUE</dd><dt>curb.weight</dt><dd>TRUE</dd><dt>engine.type</dt><dd>FALSE</dd><dt>num.of.cylinders</dt><dd>FALSE</dd><dt>engine.size</dt><dd>TRUE</dd><dt>fuel.system</dt><dd>FALSE</dd><dt>bore</dt><dd>TRUE</dd><dt>stroke</dt><dd>TRUE</dd><dt>compression.ratio</dt><dd>TRUE</dd><dt>horsepower</dt><dd>TRUE</dd><dt>peak.rpm</dt><dd>TRUE</dd><dt>city.mpg</dt><dd>TRUE</dd><dt>highway.mpg</dt><dd>TRUE</dd><dt>price</dt><dd>TRUE</dd></dl>




```python
data <- auto_mobile[ , num_cols]                        # Subset numeric columns of data 
```


```python
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
ggcorrplot(r,
           lab = TRUE,
           lab_size = 2,
           outline.color = "white",
           title = "Correlation of numeric variables")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_111_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

The ggcorrplot function has several options for customizing the output. For example

**hc.order** = TRUE reorders the variables, placing variables with similar correlation patterns together.<br>
**type** = "lower" plots the lower portion of the correlation matrix.<br>
**lab** = TRUE overlays the correlation coefficients (as text) on the plot.<br>


```python
ggcorrplot(r, 
           hc.order = TRUE, 
           type = "lower",
           lab = TRUE,
           lab_size = 3,
           outline.color = "white",
           title = "Correlation of numeric variables")
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_114_0.png)



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 


## Mosaic Plots
Mosaic charts can display the relationship between categorical variables using rectangles whose areas represent the proportion of cases for any given combination of levels. The color of the tiles can also indicate the degree relationship among the variables.


```python
# Creating a table
tbl <- xtabs(~aspiration + fuel.type + engine.location, auto_mobile)
ftable(tbl)
```


                         engine.location front rear
    aspiration fuel.type                           
    std        diesel                        6    0
               gas                         158    3
    turbo      diesel                       13    0
               gas                          23    0



```python
# Mosaic plot from the table
library(vcd)
options(warn=-1)

mosaic1 <- mosaic(tbl, 
                  shade = TRUE,
                  legend = TRUE,
                  main = "Mosaic chart example")
mosaic1
```

    Loading required package: grid
    
    
    Attaching package: 'vcd'
    
    
    The following object is masked from 'package:BSDA':
    
        Trucks
    
    
    


                               fuel.type diesel gas
    aspiration engine.location                     
    std        front                          6 158
               rear                           0   3
    turbo      front                         13  23
               rear                           0   0



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_118_2.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

## 3D Scatter Plot


```python
# 3D Scatterplot
library(scatterplot3d)

attach(auto_mobile)
scatter3d <- scatterplot3d(price,bore,stroke, pch=16, highlight.3d=TRUE,type="h", main="3D Scatterplot")

```

    The following objects are masked from auto_mobile (pos = 11):
    
        aspiration, body.style, bore, city.mpg, compression.ratio,
        curb.weight, drive.wheels, engine.location, engine.size,
        engine.type, fuel.system, fuel.type, height, highway.mpg,
        horsepower, length, make, normalized.losses, num.of.cylinders,
        num.of.doors, peak.rpm, price, stroke, symboling, wheel.base,
        width, X
    
    
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_121_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

## Bubble Plot
A bubble chart is basically just a scatterplot where the point size is proportional to the values of a third quantitative variable.


```python
# Creating a bubble plot
library(ggplot2)
ggplot(auto_mobile, 
       aes(x = length, y = width, size = height)) +
  geom_point()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_124_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

## Scatter Plot Matrix
A scatterplot matrix is a collection of scatterplots organized as a grid. It is similar to a correlation plot but instead of displaying correlations, displays the underlying data.

You can create a scatterplot matrix using the ggpairs function in the GGally package.


```python
library(GGally)
options(warn=-1)

# prepare data
data(msleep, package="ggplot2")
# library(dplyr)
df <- auto_mobile %>% 
  mutate(log_length = log(length),
         log_width = log(width)) %>%
  select(log_length, log_length, bore, stroke)
 

# create a scatterplot matrix
ggpairs(df)

```

    Registered S3 method overwritten by 'GGally':
      method from   
      +.gg   ggplot2
    
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_127_1.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 
<a id='10'></a>
## Chi Square Test(categorical data)
The Chi-square test of independence works by comparing the observed frequencies (so the frequencies observed in your sample) to the expected frequencies if there was no relationship between the two categorical variables (so the expected frequencies if the null hypothesis was true)


```python
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
```


<strong>X-squared:</strong> 116.984187249141



```python
test$p.value
```


2.35323747910589e-24


From the output and from test$p.value we see that the p-value is less than the significance level of 0.05. Like any other statistical test, if the p-value is less than the significance level, we can reject the null hypothesis.
<br><br>In this context, rejecting the null hypothesis for the Chi-square test of independence means that there is a significant relationship between the body style and the length. Therefore, knowing the value of one variable helps to predict the value of the other variable.

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

## Z Score
The __Z-score__ is the signed number of standard deviations by which the value of an observation or data point is above the mean value of what is being observed or measured<br>
While calculating the Z-score we re-scale and centre the data and look for data points which are too far from zero. These data points which are way too far from zero will be treated as the outliers. In most of the cases a __threshold of 3 or -3__ is used i.e. if the Z-score value is greater than or less than 3 or -3 respectively, that data point will be identified as outliers.


```python
library(outliers)
```


```python
# Getting the z-scores for each value in refunt_value
outlier_scores <- scores(auto_mobile$price)

# creating a logical vector the same length as outlier_scores
# that is "TRUE" if outlier_scores is greater than 3 or
# less than negative 3
is_outlier <- outlier_scores > 3 | outlier_scores < -3

# Adding a column with info whether the refund_value is an outlier
auto_mobile$is_outlier <- is_outlier

# creating a dataframe with only outliers
auto_mobile_outliers <- auto_mobile[outlier_scores > 3| outlier_scores < -3, ]
# take a peek
# it can be done for multiple columns an it hence outllier can be removed
head(auto_mobile_outliers)
```


<table>
<caption>A data.frame: 4 × 28</caption>
<thead>
	<tr><th></th><th scope=col>X</th><th scope=col>symboling</th><th scope=col>normalized.losses</th><th scope=col>make</th><th scope=col>fuel.type</th><th scope=col>aspiration</th><th scope=col>num.of.doors</th><th scope=col>body.style</th><th scope=col>drive.wheels</th><th scope=col>engine.location</th><th scope=col>...</th><th scope=col>fuel.system</th><th scope=col>bore</th><th scope=col>stroke</th><th scope=col>compression.ratio</th><th scope=col>horsepower</th><th scope=col>peak.rpm</th><th scope=col>city.mpg</th><th scope=col>highway.mpg</th><th scope=col>price</th><th scope=col>is_outlier</th></tr>
	<tr><th></th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;fct&gt;</th><th scope=col>...</th><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;int&gt;</th><th scope=col>&lt;lgl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>17</th><td> 16</td><td>0</td><td>122</td><td>bmw          </td><td>gas</td><td>std</td><td>two </td><td>sedan      </td><td>rwd</td><td>front</td><td>...</td><td>mpfi</td><td>3.62</td><td>3.39</td><td>8.0</td><td>182</td><td>5400</td><td>16</td><td>22</td><td>41315</td><td>TRUE</td></tr>
	<tr><th scope=row>72</th><td> 73</td><td>0</td><td>122</td><td>mercedes-benz</td><td>gas</td><td>std</td><td>four</td><td>sedan      </td><td>rwd</td><td>front</td><td>...</td><td>mpfi</td><td>3.80</td><td>3.35</td><td>8.0</td><td>184</td><td>4500</td><td>14</td><td>16</td><td>40960</td><td>TRUE</td></tr>
	<tr><th scope=row>73</th><td> 74</td><td>1</td><td>122</td><td>mercedes-benz</td><td>gas</td><td>std</td><td>two </td><td>hardtop    </td><td>rwd</td><td>front</td><td>...</td><td>mpfi</td><td>3.80</td><td>3.35</td><td>8.0</td><td>184</td><td>4500</td><td>14</td><td>16</td><td>45400</td><td>TRUE</td></tr>
	<tr><th scope=row>127</th><td>128</td><td>3</td><td>122</td><td>porsche      </td><td>gas</td><td>std</td><td>two </td><td>convertible</td><td>rwd</td><td>rear </td><td>...</td><td>mpfi</td><td>3.74</td><td>2.90</td><td>9.5</td><td>207</td><td>5900</td><td>17</td><td>25</td><td>37028</td><td>TRUE</td></tr>
</tbody>
</table>




```python
dim(auto_mobile_outliers)
dim(auto_mobile)
# hence this can be removed from the orignial data set
```


<ol class=list-inline><li>4</li><li>28</li></ol>




<ol class=list-inline><li>203</li><li>28</li></ol>



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 

## IQR Score
The **interquartile range (IQR)**, also called the widespread  or middle 50%, or technically H-spread, is a measure of statistical dispersion, being equal to the difference between 75th and 25th percentiles, or between upper and lower quartiles, **IQR = Q3 − Q1**<br>

In other words, the IQR is the first quartile subtracted from the third quartile; these quartiles can be clearly seen on a box plot on the data<br>

It is a measure of the dispersion like standard deviation or variance, but is much more robust against outliers<br>


```python
remove_outliers <- function(x, na.rm = TRUE, ...) {
  qnt <- quantile(x, probs=c(.25, .75), na.rm = na.rm, ...)
  H <- 1.5 * IQR(x, na.rm = na.rm)
  y <- x
  y[x < (qnt[1] - H)] <- NA
  y[x > (qnt[2] + H)] <- NA
  y
}
```


```python
set.seed(1)
x <- rnorm(100)
x <- c(-10, x, 10)

# Passing the dataframe columns to remove the outliers
y <- remove_outliers(x)
## png()
par(mfrow = c(1, 2))
boxplot(x)
boxplot(y)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_143_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 
<a id='20'></a>
# Removing outliers - quick & dirty



```python
# In order to have a couple of outliers in this dataset, we simply multiply the values in mtcars$disp that are higher than 420 by *2
mtcars$disp[which(mtcars$disp >420)] <- c(mtcars$disp[which(mtcars$disp >420)]*2)

# Looking at $disp column of the mtcars dataset with boxplot
# boxplot(mtcars$disp)
outlier = boxplot(mtcars$disp, plot = FALSE)$out

# Finding the rows containg outliers
mtcars[which(mtcars$disp %in% outlier),]
```


<table>
<caption>A data.frame: 3 × 11</caption>
<thead>
	<tr><th></th><th scope=col>mpg</th><th scope=col>cyl</th><th scope=col>disp</th><th scope=col>hp</th><th scope=col>drat</th><th scope=col>wt</th><th scope=col>qsec</th><th scope=col>vs</th><th scope=col>am</th><th scope=col>gear</th><th scope=col>carb</th></tr>
	<tr><th></th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><th scope=row>Cadillac Fleetwood</th><td>10.4</td><td>8</td><td>944</td><td>205</td><td>2.93</td><td>5.250</td><td>17.98</td><td>0</td><td>0</td><td>3</td><td>4</td></tr>
	<tr><th scope=row>Lincoln Continental</th><td>10.4</td><td>8</td><td>920</td><td>215</td><td>3.00</td><td>5.424</td><td>17.82</td><td>0</td><td>0</td><td>3</td><td>4</td></tr>
	<tr><th scope=row>Chrysler Imperial</th><td>14.7</td><td>8</td><td>880</td><td>230</td><td>3.23</td><td>5.345</td><td>17.42</td><td>0</td><td>0</td><td>3</td><td>4</td></tr>
</tbody>
</table>




```python
# Removing the rows containing the outliers, one possible option is:
mtcars <- mtcars[-which(mtcars$disp %in% outlier),]

# Checking for outliers with boxplot
boxplot(mtcars$disp)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/R/Images/EDA/output_147_0.png)


<a class="list-group-item list-group-item-action" data-toggle="list" href="#Notebook-Content" role="tab" aria-controls="settings">Go to top<span class="badge badge-primary badge-pill"></span></a> 
