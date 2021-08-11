# Bayesian Network

<div class="list-group" id="list-tab" role="tablist">
  <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Notebook Content</h3>
  <a class="list-group-item list-group-item-action" data-toggle="list" href="#Introduction" role="tab" aria-controls="profile">Introduction<span class="badge badge-primary badge-pill"></span></a><br>
  <a class="list-group-item list-group-item-action" data-toggle="list" href="#Bayesian-Time-Series" role="tab" aria-controls="messages">Bayesian Time Series<span class="badge badge-primary badge-pill"></span></a><br>
  <a class="list-group-item list-group-item-action"  data-toggle="list" href="#Bayesian-Belief-Network" role="tab" aria-controls="settings">Bayesian Belief Network<span class="badge badge-primary badge-pill"></span></a><br>
    </div>

# Introduction

Bayesian networks are a type of probabilistic graphical model that uses Bayesian inference for probability computations. Bayesian networks aim to model conditional dependence, and therefore causation, by representing conditional dependence by edges in a directed graph. Through these relationships, one can efficiently conduct inference on the random variables in the graph through the use of factors.

The aim of Bayesian Linear Regression is not to find the single “best” value of the model parameters, but rather to determine the posterior distribution for the model parameters. Not only is the response generated from a probability distribution, but the model parameters are assumed to come from a distribution as well. The posterior probability of the model parameters is conditional upon the training inputs and outputs:

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Bayesian/download.png)

Here, P(β|y, X) is the posterior probability distribution of the model parameters given the inputs and outputs. This is equal to the likelihood of the data, P(y|β, X), multiplied by the prior probability of the parameters and divided by a normalization constant. This is a simple expression of Bayes Theorem, the fundamental underpinning of Bayesian Inference:

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Bayesian/download(1).png)

Let’s take a moment to think about why we would we even want to use Bayesian techniques in the first place. Well, there are a couple of advantages in doing so and these are particularly attractive for time series analysis. One issue when working with time series models is over-fitting particularly when estimating models with large numbers of parameters over relatively short time periods. This is not such a problem in this particular case but certainly can be when looking at multiple variables which is quite common in economic forecasting. One solution to the over-fitting problem, is to take a Bayesian approach which allows us to impose certain priors on our variables.

Generally, we can write a Bayesian structural model like this:

Yt=μt+xtβ+St+et,et∼N(0,σ2e)<br>
μt+1=μt+νt,νt∼N(0,σ2ν)


<br>Here xt

denotes a set of regressors, St represents seasonality, and μt is the local level term. The local level term defines how the latent state evolves over time and is often referred to as the unobserved trend. This could, for example, represent an underlying growth in the brand value of a company or external factors that are hard to pinpoint, but it can also soak up short term fluctuations that should be controlled for with explicit terms. Note that the regressor coefficients, seasonality and trend are estimated simultaneously, which helps avoid strange coefficient estimates due to spurious relationships (similar in spirit to Granger causality, see 1). In addition, due to the Bayesian nature of the model, we can shrink the elements of β to promote sparsity or specify outside priors for the means in case we’re not able to get meaningful estimates from the historical data 


```python
#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import pydlm.plot.dlmPlot as dlmPlot
from pydlm import dlm, trend, seasonality
```


```python
data_file = pd.read_csv(r'./dataset/dataset/Bayesian_data.csv')
```


```python
# Unemployment rate dataset for US multiple location and multiple job profiles
data_file.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>iclaimsNSA</th>
      <th>michigan.unemployment</th>
      <th>idaho.unemployment</th>
      <th>pennsylvania.unemployment</th>
      <th>unemployment.filing</th>
      <th>new.jersey.unemployment</th>
      <th>department.of.unemployment</th>
      <th>illinois.unemployment</th>
      <th>rhode.island.unemployment</th>
      <th>unemployment.office</th>
      <th>filing.unemployment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.536</td>
      <td>1.488</td>
      <td>-0.561</td>
      <td>1.773</td>
      <td>0.909</td>
      <td>2.021</td>
      <td>1.640</td>
      <td>0.300</td>
      <td>1.750</td>
      <td>0.498</td>
      <td>0.073</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.882</td>
      <td>1.100</td>
      <td>-0.992</td>
      <td>0.900</td>
      <td>0.148</td>
      <td>1.280</td>
      <td>1.014</td>
      <td>0.180</td>
      <td>-0.011</td>
      <td>0.264</td>
      <td>0.584</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.077</td>
      <td>1.155</td>
      <td>-1.212</td>
      <td>1.477</td>
      <td>0.210</td>
      <td>1.080</td>
      <td>1.009</td>
      <td>0.119</td>
      <td>-0.028</td>
      <td>0.031</td>
      <td>0.448</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.135</td>
      <td>0.530</td>
      <td>-1.034</td>
      <td>1.244</td>
      <td>-0.308</td>
      <td>1.067</td>
      <td>0.734</td>
      <td>0.727</td>
      <td>-0.230</td>
      <td>-0.143</td>
      <td>-0.269</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.373</td>
      <td>0.698</td>
      <td>-1.195</td>
      <td>0.643</td>
      <td>0.570</td>
      <td>1.125</td>
      <td>0.502</td>
      <td>0.598</td>
      <td>0.625</td>
      <td>-0.219</td>
      <td>-1.006</td>
    </tr>
  </tbody>
</table>
</div>




```python
# data and feature extration 
data_file = open('dataset\dataset\Bayesian_data.csv', 'r')

variables = data_file.readline().strip().split(',')
data_map = {}

for var in variables:
    data_map[var] = []

for line in data_file:
    for i, data_piece in enumerate(line.strip().split(',')):
        data_map[variables[i]].append(float(data_piece))
```


```python
# Extract and store the data.
time_series = data_map[variables[0]]
features = [[data_map[variables[j]][i] for j in range(1, len(variables)) ]
            for i in range(len(time_series))]
```

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Bayesian-Network" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

# Bayesian Time Series


```python
# Plot the raw data
dlmPlot.plotData(range(len(time_series)),
                 time_series,
                 showDataPoint=False,
                 label='raw_data')
plt.legend(loc='best', shadow=True)
plt.xlabel("No. of observations")
plt.ylabel("iclaimsNSA")
plt.show()
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Bayesian/output_14_0.png)



```python
seasonal52
```




    <pydlm.modeler.seasonality.seasonality at 0x1e916dca240>




```python
# A linear trend
linear_trend = trend(degree=1, discount=0.95, name='linear_trend', w=10)
# A seasonality
seasonal52 = seasonality(period=52, discount=0.99, name='seasonal52', w=10)
# Build a simple dlm
simple_dlm = dlm(time_series) + linear_trend + seasonal52
```

**The default colors for the plots are:**

>> The Y-axis shows the data for iclaimsNSA and X-axis shows the number of observations.

* original data: ‘black’
* filtered results: ‘blue’
* one-step ahead prediction: ‘green’
* smoothed results: ‘red’


```python
# Fit the model
simple_dlm.fit()
# Plot the fitted results
simple_dlm.turnOff('data points')
simple_dlm.plot()
```

    Initializing models...
    Initialization finished.
    Starting forward filtering...
    Forward filtering completed.
    Starting backward smoothing...
    Backward smoothing completed.
    

    C:\Users\mohitkumar\Anaconda3\envs\lets_code\lib\site-packages\pydlm\plot\dlmPlot.py:519: MatplotlibDeprecationWarning: Passing non-integers as three-element position specification is deprecated since 3.3 and will be removed two minor releases later.
      plt.subplot(str(size[0]) + str(size[1]) + str(location))
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Bayesian/output_18_2.png)



```python
simple_dlm.getMSE()
```




    0.1730241490628275




```python
# Plot each component (attribute the time series to each component)
simple_dlm.turnOff('predict plot')
simple_dlm.turnOff('filtered plot')
simple_dlm.plot('linear_trend')
simple_dlm.plot('seasonal52')
```

    C:\Users\mohitkumar\Anaconda3\lib\site-packages\pydlm\plot\dlmPlot.py:519: MatplotlibDeprecationWarning: Passing non-integers as three-element position specification is deprecated since 3.3 and will be removed two minor releases later.
      plt.subplot(str(size[0]) + str(size[1]) + str(location))
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Bayesian/output_20_1.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Bayesian/output_20_2.png)


Most of the time series shape is attributed to the local linear trend and the strong seasonality pattern is easily seen. To further verify the performance, we use this simple model for long-term forecasting. In particular, we use the previous **351 week's**data to forecast the next **200 weeks** and the previous **251 week's** data to forecast the next **200 weeks**. We lay the predicted results on top of the real data


```python
# predictMean gives the means of all the predicted values 
# predictVar gives the variance of all the predicted values 
(predictMean, predictVar) = simple_dlm.predictN(N = 200, date = 350)
len(predictMean)
```




    200




```python
# Plot the prediction give the first 351 weeks and forcast the next 200 weeks.
simple_dlm.plotPredictN(date=350, N=200)
# Plot the prediction give the first 251 weeks and forcast the next 200 weeks.
simple_dlm.plotPredictN(date=250, N=200)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Bayesian/output_23_0.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Bayesian/output_23_1.png)


From the figure we see that after the crisis peak around 2008 - 2009 (Week 280), the simple model can accurately forecast the next 200 weeks (left figure) given the first 351 weeks. However, the model fails to capture the change near the peak if the forecasting start before Week 280 (right figure).

**Dynamic linear regression**
<br>Now, for bulding more sophiscated model with extra variables in the data file. The extra variables are stored in the variable `features` in the actual code. To build the dynamic linear regression model, we simply add a new component dynamic is the component for modeling dynamically changing predictors, which accepts features as its argument.


```python
# Build a dynamic regression model
from pydlm import dynamic
regressor10 = dynamic(features=features, discount=1.0, name='regressor10', w=10)
drm = dlm(time_series) + linear_trend + seasonal52 + regressor10
drm.fit()
```

    Initializing models...
    Initialization finished.
    Starting forward filtering...
    Forward filtering completed.
    Starting backward smoothing...
    Backward smoothing completed.
    


```python
# Plot the fitted results
drm.turnOff('data points')
drm.plot()
```

    C:\Users\mohitkumar\Anaconda3\lib\site-packages\pydlm\plot\dlmPlot.py:519: MatplotlibDeprecationWarning: Passing non-integers as three-element position specification is deprecated since 3.3 and will be removed two minor releases later.
      plt.subplot(str(size[0]) + str(size[1]) + str(location))
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Bayesian/output_27_1.png)



```python
# Plot each component (attribution)
drm.turnOff('predict plot')
drm.turnOff('filtered plot')
drm.plot('linear_trend')
drm.plot('seasonal52')
drm.plot('regressor10')
```

    C:\Users\mohitkumar\Anaconda3\lib\site-packages\pydlm\plot\dlmPlot.py:519: MatplotlibDeprecationWarning: Passing non-integers as three-element position specification is deprecated since 3.3 and will be removed two minor releases later.
      plt.subplot(str(size[0]) + str(size[1]) + str(location))
    


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Bayesian/output_28_1.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Bayesian/output_28_2.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Bayesian/output_28_3.png)



```python
# predictMean gives the means of all the predicted values 
# predictVar gives the variance of all the predicted values 
drm.predictN(date = 250, N = 105)
```




    ([0.6011708678776531,
      0.7326529240557496,
      1.1878974527521469,
      1.7086789538146478,
      1.924568285711955,
      2.520247235170519,
      2.076512144108852,
      2.5606109446528893,
      2.546174412852885,
      3.106923072976006,
      4.150032585874724,
      2.4314898134643013,
      2.0941215348140627,
      2.042767715693315,
      1.933603128890958,
      1.16784760337004,
      1.3431826819176103,
      1.4769171282635432,
      1.6163157267870316,
      1.2496676647244243,
      1.0665351633439526,
      1.0365503329561456,
      1.4994875100130458,
      1.3627056866830196,
      1.2188733127955147,
      1.0175268128162647,
      1.0529996515250943,
      1.012183581388268,
      0.9635447987623758,
      0.9901146275814694,
      0.8509881578899738,
      1.3091507507245728,
      1.1678808199038828,
      1.3769006689037393,
      1.1979743791085329,
      1.8805509894545946,
      2.031169875603436,
      1.388223214459362,
      0.9018779992823451,
      0.7619145602397543,
      0.9711665777767172,
      0.6775459791151578,
      0.7286322524602071,
      0.723169948159875,
      0.5718941573480005,
      0.7073348807411128,
      0.6045262403503372,
      0.2860803367026993,
      0.5893602886822632,
      0.5505842483346879,
      0.4516869231805192,
      0.5564319956339929,
      0.5477250823796646,
      0.9749736795315552,
      0.9469125916547079,
      0.9821018077262129,
      1.403890357740973,
      1.4816074938925217,
      1.1902182350591661,
      1.6120630625084476,
      1.5510732128708777,
      2.0984811801935144,
      2.6118430138361886,
      1.2617018557238098,
      1.1073458009093735,
      0.9168116175561047,
      0.7106556444872919,
      0.19597628048822796,
      0.47826975662556087,
      0.4436888031368096,
      0.48800872071584794,
      0.2566527479789654,
      -0.0027820734565824155,
      -0.02462497219397787,
      0.2577181470313715,
      0.374303594642171,
      0.29076718143995806,
      0.020995627883989537,
      0.03709951358820393,
      -0.03543036831099876,
      -0.1339611887811056,
      -0.04236214625994737,
      -0.2388035028798106,
      0.22700973787872744,
      0.08700083880972032,
      0.12026164975701611,
      0.3034382117495795,
      1.0450928745281047,
      0.857637745391451,
      0.25940782071031254,
      0.3143300219021238,
      0.30708177326144726,
      0.2510066453893274,
      -0.059250654121472804,
      -0.09873504046204128,
      -0.11505386332149554,
      -0.3206526158340873,
      -0.1793351540894076,
      -0.1933646727101718,
      -0.2942564852524855,
      0.030815328889734366,
      0.06341443318480483,
      -0.16605689686377514,
      -0.0600728877568661,
      0.056377808135783594],
     [0.047249891957386415,
      0.05058072042743811,
      0.05736183377703097,
      0.12372567152749075,
      0.0752789678115273,
      0.1114772296461354,
      0.11001112813084689,
      0.10926995593679992,
      0.15442545790838721,
      0.14421716473311322,
      0.21484029837405005,
      0.14401206812565234,
      0.1720674111452408,
      0.16506084243433586,
      0.14454921954595965,
      0.12757446990182222,
      0.15170359984834245,
      0.14332194209889354,
      0.16617647760103726,
      0.13645834826063272,
      0.14292776494810258,
      0.13437899977996007,
      0.15995050042215134,
      0.1402014973332863,
      0.1648527596152729,
      0.15446059904974013,
      0.16119985122767386,
      0.21703764436162432,
      0.17540486328142715,
      0.17573032248368395,
      0.16841752014275624,
      0.19382310853367468,
      0.18974299706045428,
      0.19703266720397575,
      0.1868148103564669,
      0.1946818165893606,
      0.20548879705721818,
      0.2132388042572901,
      0.20682118773479977,
      0.21714481562607305,
      0.2062664741266815,
      0.20488045256257867,
      0.22713539030825472,
      0.23317572589568644,
      0.2236378849997615,
      0.23253171065587813,
      0.2518330468312228,
      0.29783241135632765,
      0.2914428066960644,
      0.2973848952441347,
      0.2852300027249827,
      0.29589061597389926,
      0.3525273802882539,
      0.36898692696769486,
      0.3786087730438197,
      0.35990713457258294,
      0.46424345024063274,
      0.40653659492307986,
      0.397234062408659,
      0.42478919396517995,
      0.5769381741807508,
      0.5163697662840604,
      0.4736046175329168,
      0.451447577987066,
      0.4850225225501357,
      0.5075984126840579,
      0.5397007620492531,
      0.5506672787451447,
      0.5589050414545754,
      0.6055546067922292,
      0.6510135329364782,
      0.6011561203822571,
      0.6231286698851689,
      0.6450035057468025,
      0.679440847757978,
      0.6567842123351922,
      0.7021818920565787,
      0.7080341648627696,
      0.7194783152237182,
      0.7281746049165698,
      0.744211106593106,
      0.7519834833482628,
      0.7682695127382405,
      0.7921413220347095,
      0.7907128578689329,
      0.8046229055190595,
      0.8169270758394892,
      0.8091245373899518,
      0.8510260853494814,
      0.8488811383762729,
      0.8694851255963466,
      0.8544337320648285,
      0.8219495561616333,
      0.8386344740499039,
      0.8629027322872929,
      0.8789995782759683,
      0.8880763413793281,
      0.9411905993686975,
      0.931689017371146,
      0.9571084677755862,
      0.9825729897674682,
      0.9822850152276165,
      0.9988992865753983,
      1.031351802589605,
      1.0792072117845328])



This time, the shape of the time series is mostly attributed to the regressor and the linear trend looks more linear. If we do long-term forecasting again, i.e., use the previous 301 week's data to forecast the next 150 weeks and the previous 251 week's data to forecast the next 200 weeks


```python
# Plot the prediction give the first 300 weeks and forcast the next 150 weeks.
drm.plotPredictN(N=150, date=300)
# Plot the prediction give the first 250 weeks and forcast the next 200 weeks.
drm.plotPredictN(N=200, date=250)
```


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Bayesian/output_31_0.png)



![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Bayesian/output_31_1.png)


The results look much better compared to the simple model

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Bayesian-Network" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

# Bayesian Belief Network

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Bayesian/download(2).png)

#### Mathematical Definition of Belief Networks

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Bayesian/download(3).png)

## Implementing using evidence and inference model


```python
#Importing the libraries
from pybbn.graph.dag import Bbn
from pybbn.graph.edge import Edge, EdgeType
from pybbn.graph.jointree import EvidenceBuilder
from pybbn.graph.node import BbnNode
from pybbn.graph.variable import Variable
from pybbn.pptc.inferencecontroller import InferenceController
```


```python
# create the nodes
# The data provides conditional probabilties of independent events
season = BbnNode(Variable(0, 'season', ['winter', 'summer']), [0.5, 0.5])
atmos_pres = BbnNode(Variable(1, 'atmos_press', ['high', 'low']), [0.5, 0.5])
allergies = BbnNode(Variable(2, 'allergies', ['allergic', 'non_alergic']), [0.7, 0.3, 0.2, 0.8])
rain = BbnNode(Variable(3, 'rain', ['rainy', 'sunny']), [0.9, 0.1, 0.7, 0.3, 0.3, 0.7, 0.1, 0.9])
grass = BbnNode(Variable(4, 'grass', ['grass', 'no_grass']), [0.8, 0.2, 0.3, 0.7])
umbrellas = BbnNode(Variable(5, 'umbrellas', ['on', 'off']), [0.99, 0.01, 0.80, 0.20, 0.20, 0.80, 0.01, 0.99])
dog_bark = BbnNode(Variable(6, 'dog_bark', ['bark', 'not_bark']), [0.8, 0.2, 0.1, 0.9])
cat_mood = BbnNode(Variable(7, 'cat_mood', ['good', 'bad']), [0.05, 0.95, 0.95, 0.05])
cat_hide = BbnNode(Variable(8, 'cat_hide', ['hide', 'show']), [0.20, 0.80, 0.95, 0.05, 0.95, 0.05, 0.70, 0.30])
```


```python
#buildind model architechture with dependencies
bbn = Bbn() \
    .add_node(season) \
    .add_node(atmos_pres) \
    .add_node(allergies) \
    .add_node(rain) \
    .add_node(grass) \
    .add_node(umbrellas) \
    .add_node(dog_bark) \
    .add_node(cat_mood) \
    .add_node(cat_hide) \
    .add_edge(Edge(season, allergies, EdgeType.DIRECTED)) \
    .add_edge(Edge(season, umbrellas, EdgeType.DIRECTED)) \
    .add_edge(Edge(season, rain, EdgeType.DIRECTED)) \
    .add_edge(Edge(atmos_pres, rain, EdgeType.DIRECTED)) \
    .add_edge(Edge(rain, grass, EdgeType.DIRECTED)) \
    .add_edge(Edge(rain, umbrellas, EdgeType.DIRECTED)) \
    .add_edge(Edge(rain, dog_bark, EdgeType.DIRECTED)) \
    .add_edge(Edge(rain, cat_mood, EdgeType.DIRECTED)) \
    .add_edge(Edge(dog_bark, cat_hide, EdgeType.DIRECTED)) \
    .add_edge(Edge(cat_mood, cat_hide, EdgeType.DIRECTED))
```

### Bayesian Network Architecture

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Bayesian/download(4).png)

The below code snippet basically give evidence to the network which is the season is winter with 1.0 probability. According to this evidence, when we do the inference, we get following results.


```python
# convert the BBN to a join tree
join_tree = InferenceController.apply(bbn)
# insert an observation evidence
ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name('season')) \
    .with_evidence('winter', 1.0) \
    .build()
join_tree.set_observation(ev)
# print the marginal probabilities
for node in join_tree.get_bbn_nodes():
    potential = join_tree.get_bbn_potential(node)
    print(node)
    print(potential)
    print('--------------------->')
```

    0|season|winter,summer
    0=winter|1.00000
    0=summer|0.00000
    --------------------->
    2|allergies|allergic,non_alergic
    2=allergic|0.70000
    2=non_alergic|0.30000
    --------------------->
    3|rain|rainy,sunny
    3=rainy|0.80000
    3=sunny|0.20000
    --------------------->
    4|grass|grass,no_grass
    4=grass|0.70000
    4=no_grass|0.30000
    --------------------->
    1|atmos_press|high,low
    1=high|0.50000
    1=low|0.50000
    --------------------->
    5|umbrellas|on,off
    5=on|0.95200
    5=off|0.04800
    --------------------->
    6|dog_bark|bark,not_bark
    6=bark|0.66000
    6=not_bark|0.34000
    --------------------->
    7|cat_mood|good,bad
    7=good|0.23000
    7=bad|0.77000
    --------------------->
    8|cat_hide|hide,show
    8=hide|0.87150
    8=show|0.12850
    --------------------->
    

When we add further evidence, like the dog is not barking


```python
# convert the BBN to a join tree
join_tree = InferenceController.apply(bbn)
# insert an observation evidence
ev = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name('season')) \
    .with_evidence('winter', 1.0) \
    .build()
ev2 = EvidenceBuilder() \
    .with_node(join_tree.get_bbn_node_by_name('dog_bark')) \
    .with_evidence('not_bark', 0.2) \
    .build()
join_tree.set_observation(ev)
join_tree.set_observation(ev2)
# print the marginal probabilities
for node in join_tree.get_bbn_nodes():
    potential = join_tree.get_bbn_potential(node)
    print(node)
    print(potential)
    print('--------------------->')
```

    0|season|winter,summer
    0=winter|1.00000
    0=summer|0.00000
    --------------------->
    2|allergies|allergic,non_alergic
    2=allergic|0.70000
    2=non_alergic|0.30000
    --------------------->
    3|rain|rainy,sunny
    3=rainy|0.47059
    3=sunny|0.52941
    --------------------->
    4|grass|grass,no_grass
    4=grass|0.53529
    4=no_grass|0.46471
    --------------------->
    1|atmos_press|high,low
    1=high|0.39706
    1=low|0.60294
    --------------------->
    5|umbrellas|on,off
    5=on|0.88941
    5=off|0.11059
    --------------------->
    6|dog_bark|bark,not_bark
    6=bark|0.00000
    6=not_bark|1.00000
    --------------------->
    7|cat_mood|good,bad
    7=good|0.52647
    7=bad|0.47353
    --------------------->
    8|cat_hide|hide,show
    8=hide|0.83162
    8=show|0.16838
    --------------------->
    

##### Lots of probability values changed when we add the evidence related to the barking

## Implementing using Probability model by Monty hall problem


```python
#Import required packages
import math
from pomegranate import *
 
# Initially the door selected by the guest is completely random
guest =DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )
 
# The door containing the prize is also a random process
prize =DiscreteDistribution( { 'A': 1./3, 'B': 1./3, 'C': 1./3 } )
```

Let’s take a look at initializing a Bayesian network in the first manner by quickly implementing the Monty Hall problem. 


```python
#childnode = ConditionalProbabilityTable.from_samples(data, [parentnode],weights=None, pseudocount=0.0)
# The door Monty picks, depends on the choice of the guest and the prize door
monty = ConditionalProbabilityTable(
[[ 'A', 'A', 'A', 0.0 ],
[ 'A', 'A', 'B', 0.5 ],
[ 'A', 'A', 'C', 0.5 ],
[ 'A', 'B', 'A', 0.0 ],
[ 'A', 'B', 'B', 0.0 ],
[ 'A', 'B', 'C', 1.0 ],
[ 'A', 'C', 'A', 0.0 ],
[ 'A', 'C', 'B', 1.0 ],
[ 'A', 'C', 'C', 0.0 ],
[ 'B', 'A', 'A', 0.0 ],
[ 'B', 'A', 'B', 0.0 ],
[ 'B', 'A', 'C', 1.0 ],
[ 'B', 'B', 'A', 0.5 ],
[ 'B', 'B', 'B', 0.0 ],
[ 'B', 'B', 'C', 0.5 ],
[ 'B', 'C', 'A', 1.0 ],
[ 'B', 'C', 'B', 0.0 ],
[ 'B', 'C', 'C', 0.0 ],
[ 'C', 'A', 'A', 0.0 ],
[ 'C', 'A', 'B', 1.0 ],
[ 'C', 'A', 'C', 0.0 ],
[ 'C', 'B', 'A', 1.0 ],
[ 'C', 'B', 'B', 0.0 ],
[ 'C', 'B', 'C', 0.0 ],
[ 'C', 'C', 'A', 0.5 ],
[ 'C', 'C', 'B', 0.5 ],
[ 'C', 'C', 'C', 0.0 ]], [guest, prize] 
```


```python
#adding states to the base
d1 = State( guest, name="guest" )
d2 = State( prize, name="prize" )
d3 = State( monty, name="monty" )
```


```python
#Building the Bayesian Network
network = BayesianNetwork( "Solving the Monty Hall Problem With Bayesian Networks" )
network.add_states(d1, d2, d3)
network.add_edge(d1, d3)
network.add_edge(d2, d3)
network.bake()
```

Let’s understand the dependencies here, the door selected by the guest and the door containing the car are completely random processes. However, the door Monty chooses to open is dependent on both the doors; the door selected by the guest, and the door the prize is behind. Monty has to choose in such a way that the door does not contain the prize and it cannot be the one chosen by the guest.


```python
beliefs = network.predict_proba({ 'guest' : 'A' })
beliefs = map(str, beliefs)
print("n".join( "{}t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) ))
```

    guesttAnprizet{
        "class" :"Distribution",
        "dtype" :"str",
        "name" :"DiscreteDistribution",
        "parameters" :[
            {
                "A" :0.2000000000000002,
                "B" :0.2000000000000002,
                "C" :0.5999999999999996
            }
        ],
        "frozen" :false
    }nmontyt{
        "class" :"Distribution",
        "dtype" :"str",
        "name" :"DiscreteDistribution",
        "parameters" :[
            {
                "C" :0.30000000000000016,
                "B" :0.6999999999999997,
                "A" :0.0
            }
        ],
        "frozen" :false
    }
    

In the above code snippet, we’ve assumed that the guest picks door ‘A’. Given this information, the probability of the prize door being ‘A’, ‘B’, ‘C’ is equal (1/3) since it is a random process. However, the probability of Monty picking ‘A’ is obviously zero since the guest picked door ‘A’. And the other two doors have a 50% chance of being picked by Monty since we don’t know which is the prize door.


```python
beliefs = network.predict_proba({'guest' : 'A', 'monty' : 'B'})
print("n".join( "{}t{}".format( state.name, str(belief) ) for state, belief in zip( network.states, beliefs )))
```

    guesttAnprizet{
        "class" :"Distribution",
        "dtype" :"str",
        "name" :"DiscreteDistribution",
        "parameters" :[
            {
                "A" :0.14285714285714315,
                "B" :0.0,
                "C" :0.8571428571428568
            }
        ],
        "frozen" :false
    }nmontytB
    

In the above code snippet, we’ve provided two inputs to our Bayesian Network, this is where things get interesting. We’ve mentioned the following:

The guest picks door ‘A’
Monty picks door ‘B’
Notice the output, the probability of the car being behind door ‘C’ is approx. 66%. This proves that if the guest switches his choice, he has a higher probability of winning. Though this might seem confusing to some of you, it’s a known fact that:

Guests who decided to switch doors won about 2/3 of the time
Guests who refused to switch won about 1/3 of the time

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Bayesian-Network" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


```python

```
