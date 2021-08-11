# Recommender System

<div class="list-group" id="list-tab" role="tablist">
  <h3 class="list-group-item list-group-item-action active" data-toggle="list"  role="tab" aria-controls="home">Notebook Content</h3>
  <a class="list-group-item list-group-item-action" data-toggle="list" href="#Introduction" role="tab" aria-controls="profile">Introduction<span class="badge badge-primary badge-pill"></span></a><br>
  <a class="list-group-item list-group-item-action" data-toggle="list" href="#Content-Based-Filtering" role="tab" aria-controls="messages">Content based filtering<span class="badge badge-primary badge-pill"></span></a><br>
  <a class="list-group-item list-group-item-action"  data-toggle="list" href="#Collaborative-Filtering" role="tab" aria-controls="settings">Collaborative filtering<span class="badge badge-primary badge-pill"></span></a><br>
    <a class="list-group-item list-group-item-action"  data-toggle="list" href="#Hybrid-Recommender-System" role="tab" aria-controls="settings">Hybrid Recommender System<span class="badge badge-primary badge-pill"></span></a><br>    
    </div>

# Introduction


![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Recommender%20System/download.png)

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Recommender%20System/download(1).png)

# Content-Based Filtering

* This type of filter does not involve other users if not ourselves. Based on what we like, the algorithm will simply pick items with similar content to recommend us.

* Cosine Similarity is one of the metric that we can use when calculating similarity, between users or contents.

* Content-based recommenders treat suggestions as a user-specific
category problem and learn a classifier for the customer's preferences depending on product traits. This approach is based on
information retrieval because content associated with the user’s preferences is treated as query to the system and unrated items
are scored with similar items

## Fashion Recommender System 

* To illustrate basic content based filtering on the basis of product description
* It uses cosine similarity to understand the likeness of different fashion garments


```python
#Importing required libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel 
df = pd.read_csv("fashion-data.csv")
```

The data contains 500 product type and we are trying to recommend top 10 similar product to the purchased item


```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Active classic boxers - There's a reason why o...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Active sport boxer briefs - Skinning up Glory ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Active sport briefs - These superbreathable no...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Alpine guide pants - Skin in, climb ice, switc...</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Alpine wind jkt - On high ridges, steep ice an...</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Word embedding using TF-IDF
tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
tfidf_matrix = tf.fit_transform(df['description'])
```


```python
#Calculating cosine similarities among all the available product
cosine_similarities = linear_kernel(tfidf_matrix, tfidf_matrix) 
results = {}
for idx, row in df.iterrows():
    similar_indices = cosine_similarities[idx].argsort()[:-100:1]
    #print(similar_indices)
    similar_items = [(cosine_similarities[idx][i], df['id'][i]) for i in similar_indices] 
    results[row['id']] = similar_items[1:]
```


```python
#Function to reverse map and recommend the items
def item(id):  
    return df.loc[df['id'] == id]['description'].tolist()[0].split(' - ')[0]

# Just reads the results out of the dictionary.def 
def recommend(item_id, num):
    print("Recommending " + str(num) + " products similar to " + item(item_id) + "...")
    print("-------")
    recs = results[item_id][:num]
    for rec in recs:
        print("Recommended: " + item(rec[1]) + " (score:" + str(rec[0]) + ")")
```


```python
#Recommendations
recommend(item_id=40, num=10)
```

    Recommending 10 products similar to Fezzman shirt...
    -------
    Recommended: Traversing auguille d'entreves (score:0.0187570535671376)
    Recommended: Going big in b.c. poster (score:0.018833693944229037)
    Recommended: Surf brim (score:0.01887428023955814)
    Recommended: Wild steelhead, alaska poster (score:0.018880358356743576)
    Recommended: Flyfishing the athabasca poster (score:0.019003498956779136)
    Recommended: Symmetry w16 poster (score:0.019048299828287412)
    Recommended: Lead an examined life poster (score:0.019347722522867858)
    Recommended: Beach bucket (score:0.019516558727964754)
    Recommended: Ultra hw mountaineering socks (score:0.019928579689678955)
    Recommended: Wyoming climbing poster (score:0.019929167637094222)
    

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Recommender-System" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

### Understanding Count Vectorizer and Cosine Similarity and Pearson Correlation 


```python
text = ["London Paris London","Paris Paris London" , "Male Male Female", "Female Male Female"]
```


```python
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
count_matrix = cv.fit_transform(text)
```


```python
print(cv.get_feature_names())
print(count_matrix.toarray())
```

    ['female', 'london', 'male', 'paris']
    [[0 2 0 1]
     [0 1 0 2]
     [1 0 2 0]
     [2 0 1 0]]
    

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Recommender%20System/download(2).png)



```python
from sklearn.metrics.pairwise import cosine_similarity
similarity_scores = cosine_similarity(count_matrix)
print(similarity_scores)
```

    [[1.  0.8 0.  0. ]
     [0.8 1.  0.  0. ]
     [0.  0.  1.  0.8]
     [0.  0.  0.8 1. ]]
    


```python
#correlation
np.corrcoef(count_matrix.toarray())
```




    array([[ 1.        ,  0.63636364, -0.81818182, -0.81818182],
           [ 0.63636364,  1.        , -0.81818182, -0.81818182],
           [-0.81818182, -0.81818182,  1.        ,  0.63636364],
           [-0.81818182, -0.81818182,  0.63636364,  1.        ]])



<a class="list-group-item list-group-item-action" data-toggle="list" href="#Recommender-System" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

## Movie Recommender System

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Recommender%20System/download(3).png)


```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,pairwise_distances
```


```python
df = pd.read_csv("movie_dataset.csv")
```


```python
df.shape
```




    (4803, 24)




```python
df.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>...</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>cast</th>
      <th>crew</th>
      <th>director</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>237000000</td>
      <td>Action Adventure Fantasy Science Fiction</td>
      <td>http://www.avatarmovie.com/</td>
      <td>19995</td>
      <td>culture clash future space war space colony so...</td>
      <td>en</td>
      <td>Avatar</td>
      <td>In the 22nd century, a paraplegic Marine is di...</td>
      <td>150.437577</td>
      <td>...</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
      <td>Sam Worthington Zoe Saldana Sigourney Weaver S...</td>
      <td>[{'name': 'Stephen E. Rivkin', 'gender': 0, 'd...</td>
      <td>James Cameron</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>300000000</td>
      <td>Adventure Fantasy Action</td>
      <td>http://disney.go.com/disneypictures/pirates/</td>
      <td>285</td>
      <td>ocean drug abuse exotic island east india trad...</td>
      <td>en</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>Captain Barbossa, long believed to be dead, ha...</td>
      <td>139.082615</td>
      <td>...</td>
      <td>169.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>At the end of the world, the adventure begins.</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>6.9</td>
      <td>4500</td>
      <td>Johnny Depp Orlando Bloom Keira Knightley Stel...</td>
      <td>[{'name': 'Dariusz Wolski', 'gender': 2, 'depa...</td>
      <td>Gore Verbinski</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>245000000</td>
      <td>Action Adventure Crime</td>
      <td>http://www.sonypictures.com/movies/spectre/</td>
      <td>206647</td>
      <td>spy based on novel secret agent sequel mi6</td>
      <td>en</td>
      <td>Spectre</td>
      <td>A cryptic message from Bond’s past sends him o...</td>
      <td>107.376788</td>
      <td>...</td>
      <td>148.0</td>
      <td>[{"iso_639_1": "fr", "name": "Fran\u00e7ais"},...</td>
      <td>Released</td>
      <td>A Plan No One Escapes</td>
      <td>Spectre</td>
      <td>6.3</td>
      <td>4466</td>
      <td>Daniel Craig Christoph Waltz L\u00e9a Seydoux ...</td>
      <td>[{'name': 'Thomas Newman', 'gender': 2, 'depar...</td>
      <td>Sam Mendes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>250000000</td>
      <td>Action Crime Drama Thriller</td>
      <td>http://www.thedarkknightrises.com/</td>
      <td>49026</td>
      <td>dc comics crime fighter terrorist secret ident...</td>
      <td>en</td>
      <td>The Dark Knight Rises</td>
      <td>Following the death of District Attorney Harve...</td>
      <td>112.312950</td>
      <td>...</td>
      <td>165.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>The Legend Ends</td>
      <td>The Dark Knight Rises</td>
      <td>7.6</td>
      <td>9106</td>
      <td>Christian Bale Michael Caine Gary Oldman Anne ...</td>
      <td>[{'name': 'Hans Zimmer', 'gender': 2, 'departm...</td>
      <td>Christopher Nolan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>260000000</td>
      <td>Action Adventure Science Fiction</td>
      <td>http://movies.disney.com/john-carter</td>
      <td>49529</td>
      <td>based on novel mars medallion space travel pri...</td>
      <td>en</td>
      <td>John Carter</td>
      <td>John Carter is a war-weary, former military ca...</td>
      <td>43.926995</td>
      <td>...</td>
      <td>132.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>Lost in our world, found in another.</td>
      <td>John Carter</td>
      <td>6.1</td>
      <td>2124</td>
      <td>Taylor Kitsch Lynn Collins Samantha Morton Wil...</td>
      <td>[{'name': 'Andrew Stanton', 'gender': 2, 'depa...</td>
      <td>Andrew Stanton</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>



If you visualize the dataset, you will see that it has many extra info about a movie. We don’t need all of them. So, we choose keywords, cast, genres and director column to use as our feature set(the so called “content” of the movie).


```python
features = ['keywords','cast','genres','director']
```

Our next task is to create a function for combining the values of these columns into a single string.


```python
def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['genres']+" "+row['director']
```

Now, we need to call this function over each row of our dataframe. But, before doing that, we need to clean and preprocess the data for our use. We will fill all the NaN values with blank string in the dataframe.


```python
for feature in features:
    df[feature] = df[feature].fillna('') #filling all NaNs with blank string

df["combined_features"] = df.apply(combine_features,axis=1) 
#applying combined_features() method over each rows of dataframe and storing the combined string in "combined_features" column
```


```python
df.iloc[0].combined_features
```




    'culture clash future space war space colony society Sam Worthington Zoe Saldana Sigourney Weaver Stephen Lang Michelle Rodriguez Action Adventure Fantasy Science Fiction James Cameron'




```python
cv = CountVectorizer() #creating new CountVectorizer() object
count_matrix = cv.fit_transform(df["combined_features"]) #feeding combined strings(movie contents) to CountVectorizer() object
```

At this point, 60% work is done. Now, we need to obtain the cosine similarity / pearson similarity matrix from the count matrix.


```python
cosine_sim = cosine_similarity(count_matrix)
```


```python
len(cosine_sim[0])
```




    4803




```python
pearson_sim = np.corrcoef(count_matrix.todense())
```

    C:\Users\mohitkumar\AppData\Roaming\Python\Python36\site-packages\numpy\lib\function_base.py:2559: RuntimeWarning: invalid value encountered in true_divide
      c /= stddev[:, None]
    C:\Users\mohitkumar\AppData\Roaming\Python\Python36\site-packages\numpy\lib\function_base.py:2560: RuntimeWarning: invalid value encountered in true_divide
      c /= stddev[None, :]
    


```python
len(pearson_sim[0])
```




    4803




```python
cosine_sim[0]
```




    array([1.        , 0.10540926, 0.12038585, ..., 0.        , 0.        ,
           0.        ])




```python
pearson_sim[0]
```




    array([ 1.        ,  0.10381994,  0.11901652, ..., -0.00162313,
           -0.0010406 , -0.00124536])



Now, we will define two helper functions to get movie title from movie index and vice-versa.


```python
def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]
def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]
```

Our next step is to get the title of the movie that the user currently likes. Then we will find the index of that movie. After that, we will access the row corresponding to this movie in the similarity matrix. Thus, we will get the similarity scores of all other movies from the current movie. Then we will enumerate through all the similarity scores of that movie to make a tuple of movie index and similarity score. This will convert a row of similarity scores like this- [1 0.5 0.2 0.9] to this- [(0, 1) (1, 0.5) (2, 0.2) (3, 0.9)] . Here, each item is in this form- (movie index, similarity score).


```python
movie_user_likes = "Arn: The Knight Templar"
movie_index = get_index_from_title(movie_user_likes)
similar_movies_with_cosine = list(enumerate(cosine_sim[movie_index])) 
```


```python
movie_user_likes = "Arn: The Knight Templar"
movie_index = get_index_from_title(movie_user_likes)
similar_movies_with_pearson = list(enumerate(pearson_sim[movie_index])) 
#accessing the row corresponding to given movie to find all the similarity scores for that movie and then enumerating over it
```

Now comes the most vital point. We will sort the list similar_movies according to similarity scores in descending order. Since the most similar movie to a given movie will be itself, we will discard the first element after sorting the movies.


```python
sorted_similar_movies_with_cosine = sorted(similar_movies_with_cosine,key=lambda x:x[1],reverse=True)[1:]
```


```python
sorted_similar_movies_with_pearson = sorted(similar_movies_with_pearson,key=lambda x:x[1],reverse=True)[1:]
```

Now, we will run a loop to print first 11 entries from sorted_similar_movies_with_cosine list.


```python
i=0
print("Top 11 similar movies to "+movie_user_likes+" are:\n")
for element in sorted_similar_movies_with_cosine:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>11:
        break
```

    Top 11 similar movies to Arn: The Knight Templar are:
    
    Caravans
    Undisputed
    Me You and Five Bucks
    Amidst the Devil's Wings
    Love's Abiding Joy
    Restless
    Pirates of the Caribbean: Dead Man's Chest
    Thor
    The Claim
    The Twilight Saga: Breaking Dawn - Part 2
    Battleship
    Cinderella
    

Now, we will run a loop to print first 11 entries from sorted_similar_movies_with_pearson list.


```python
i=0
print("Top 11 similar movies to "+movie_user_likes+" are:\n")
for element in sorted_similar_movies_with_pearson:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>11:
        break
```

    Top 11 similar movies to Arn: The Knight Templar are:
    
    Caravans
    Undisputed
    Me You and Five Bucks
    Love's Abiding Joy
    Restless
    Pirates of the Caribbean: Dead Man's Chest
    Thor
    The Claim
    The Twilight Saga: Breaking Dawn - Part 2
    Battleship
    Ronin
    Cinderella
    

Different similarity measure algorithms can be used i.e. Pearson correlation, Euclidean distance, Cosine similarity and
Jaccard coefficient. The algorithms used behave differently in different context. Majority of the algorithms shows
the same result in finding the similarity between the item contents. The resulting values are scaled in the range of 0 to 1 for Euclidean distance, Cosine similarity and Jaccard coefficient, whereas the values for Pearson correlation are from -1 to 1. Value 1 in all the four algorithms represent completely similar and value 0 represents completely dissimilar. Value -1 in Pearson correlation represents the negative similarity between the entities.

##### In the above example we can observe that most of the recommendations are same for cosine and pearson similarity metrics except for few such as "Ronin" and "Amidst the Devil's Wings".

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Recommender-System" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

# Conclusion

Further accuracy and performance can be enhanced by applying pre-processing methods for the combined features to bring them on same scale and remove noise and unwanted data. 
For eg:
* Converting combined features to lowercase
* Removing stopword
* Applying methods like lemmatization/stemming
* Remove extra whitespaces

# Collaborative Filtering

* Collaborative filtering is a technique that can filter out items that a user might like on the basis of reactions by similar users.
* It works by searching a large group of people and finding a smaller set of users with tastes similar to a particular user. It looks at the items they like and combines them to create a ranked list of suggestions.

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Recommender%20System/download(4).png)


## KNN on movie lens dataset (Item-Item based filtering)

* To implement an item based collaborative filtering, KNN is a perfect go-to model and also a very good baseline for recommender system development.
* KNN is a non-parametric, lazy learning method. It uses a database in which the data points are separated into several clusters to make inference for new samples.
* KNN does not make any assumptions on the underlying data distribution but it relies on item feature similarity.
* KNN’s performance will suffer from curse of dimensionality if it uses “euclidean distance” in its objective function. Euclidean distance is unhelpful in high dimensions because all vectors are almost equidistant to the search query vector (target movie’s features). Instead, we will use cosine similarity for nearest neighbor search.


```python
import os
import pandas as pd
from sklearn.neighbors import NearestNeighbors
data_path = 'ml-25m/'
movies_filename = 'movies.csv'
ratings_filename = 'ratings.csv'
df_movies = pd.read_csv(
    os.path.join(data_path, movies_filename),
usecols=['movieId', 'title'],
    dtype={'movieId': 'int32', 'title': 'str'})
df_ratings = pd.read_csv(
    os.path.join(data_path, ratings_filename),
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
```


```python
from scipy.sparse import csr_matrix
# pivot ratings into movie features
df_ratings=df_ratings[:6250024]
df_movie_features = df_ratings.pivot(
    index='movieId',
    columns='userId',
    values='rating'
).fillna(0)
mat_movie_features = csr_matrix(df_movie_features.values)
```


```python
mat_movie_features
```




    <41883x40510 sparse matrix of type '<class 'numpy.float32'>'
    	with 6250024 stored elements in Compressed Sparse Row format>




```python
df_movie_features.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>userId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>40501</th>
      <th>40502</th>
      <th>40503</th>
      <th>40504</th>
      <th>40505</th>
      <th>40506</th>
      <th>40507</th>
      <th>40508</th>
      <th>40509</th>
      <th>40510</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>3.5</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>1.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40510 columns</p>
</div>




```python
num_users = len(df_ratings.userId.unique())
num_items = len(df_ratings.movieId.unique())
print('There are {} unique users and {} unique movies in this data set'.format(num_users, num_items))
```

    There are 40510 unique users and 41883 unique movies in this data set
    


```python
# get count
df_ratings_cnt_tmp = pd.DataFrame(df_ratings.groupby('rating').size(), columns=['count'])
df_ratings_cnt_tmp
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>rating</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.5</th>
      <td>96334</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>196353</td>
    </tr>
    <tr>
      <th>1.5</th>
      <td>102872</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>411018</td>
    </tr>
    <tr>
      <th>2.5</th>
      <td>319901</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>1220327</td>
    </tr>
    <tr>
      <th>3.5</th>
      <td>801904</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>1651701</td>
    </tr>
    <tr>
      <th>4.5</th>
      <td>550918</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>898696</td>
    </tr>
  </tbody>
</table>
</div>




```python
# there are a lot more counts in rating of zero
total_cnt = num_users * num_items
rating_zero_cnt = total_cnt - df_ratings.shape[0]

df_ratings_cnt = df_ratings_cnt_tmp.append(
    pd.DataFrame({'count': rating_zero_cnt}, index=[0.0]),
    verify_integrity=True,
).sort_index()
df_ratings_cnt
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>1690430306</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>96334</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>196353</td>
    </tr>
    <tr>
      <th>1.5</th>
      <td>102872</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>411018</td>
    </tr>
    <tr>
      <th>2.5</th>
      <td>319901</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>1220327</td>
    </tr>
    <tr>
      <th>3.5</th>
      <td>801904</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>1651701</td>
    </tr>
    <tr>
      <th>4.5</th>
      <td>550918</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>898696</td>
    </tr>
  </tbody>
</table>
</div>




```python
#log normalise to make it easier to interpret on a graph
import numpy as np
df_ratings_cnt['log_count'] = np.log(df_ratings_cnt['count'])
df_ratings_cnt
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>log_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0.0</th>
      <td>1690430306</td>
      <td>21.248249</td>
    </tr>
    <tr>
      <th>0.5</th>
      <td>96334</td>
      <td>11.475577</td>
    </tr>
    <tr>
      <th>1.0</th>
      <td>196353</td>
      <td>12.187669</td>
    </tr>
    <tr>
      <th>1.5</th>
      <td>102872</td>
      <td>11.541241</td>
    </tr>
    <tr>
      <th>2.0</th>
      <td>411018</td>
      <td>12.926392</td>
    </tr>
    <tr>
      <th>2.5</th>
      <td>319901</td>
      <td>12.675767</td>
    </tr>
    <tr>
      <th>3.0</th>
      <td>1220327</td>
      <td>14.014629</td>
    </tr>
    <tr>
      <th>3.5</th>
      <td>801904</td>
      <td>13.594744</td>
    </tr>
    <tr>
      <th>4.0</th>
      <td>1651701</td>
      <td>14.317316</td>
    </tr>
    <tr>
      <th>4.5</th>
      <td>550918</td>
      <td>13.219341</td>
    </tr>
    <tr>
      <th>5.0</th>
      <td>898696</td>
      <td>13.708700</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get rating frequency
#number of ratings each movie got.
df_movies_cnt = pd.DataFrame(df_ratings.groupby('movieId').size(), columns=['count'])
df_movies_cnt.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>movieId</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>14284</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5983</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2902</td>
    </tr>
    <tr>
      <th>4</th>
      <td>628</td>
    </tr>
    <tr>
      <th>5</th>
      <td>2931</td>
    </tr>
  </tbody>
</table>
</div>




```python
#now we need to take only movies that have been rated atleast 50 times to get some idea of the reactions of users towards it
popularity_thres = 50
popular_movies = list(set(df_movies_cnt.query('count >=@popularity_thres').index))
df_ratings_drop_movies = df_ratings[df_ratings.movieId.isin(popular_movies)]
print('shape of original ratings data: ', df_ratings.shape)
print('shape of ratings data after dropping unpopular movies: ', df_ratings_drop_movies.shape)
```

    shape of original ratings data:  (6250024, 3)
    shape of ratings data after dropping unpopular movies:  (6025798, 3)
    


```python
# get number of ratings given by every user
df_users_cnt = pd.DataFrame(df_ratings_drop_movies.groupby('userId').size(), columns=['count'])
df_users_cnt.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>62</td>
    </tr>
    <tr>
      <th>2</th>
      <td>183</td>
    </tr>
    <tr>
      <th>3</th>
      <td>652</td>
    </tr>
    <tr>
      <th>4</th>
      <td>238</td>
    </tr>
    <tr>
      <th>5</th>
      <td>101</td>
    </tr>
  </tbody>
</table>
</div>




```python
# filter data to come to an approximation of user likings.
ratings_thres = 50
active_users = list(set(df_users_cnt.query('count >= @ratings_thres').index))
df_ratings_drop_users = df_ratings_drop_movies[df_ratings_drop_movies.userId.isin(active_users)]
print('shape of original ratings data: ', df_ratings.shape)
print('shape of ratings data after dropping both unpopular movies and inactive users: ', df_ratings_drop_users.shape)
```

    shape of original ratings data:  (6250024, 3)
    shape of ratings data after dropping both unpopular movies and inactive users:  (5551213, 3)
    


```python
# pivot and create movie-user matrix
movie_user_mat = df_ratings_drop_users.pivot(index='movieId', columns='userId', values='rating').fillna(0)
#map movie titles to images
movie_to_idx = {
    movie: i for i, movie in 
    enumerate(list(df_movies.set_index('movieId').loc[movie_user_mat.index].title))
}
# transform matrix to scipy sparse matrix
movie_user_mat_sparse = csr_matrix(movie_user_mat.values)
```


```python
movie_user_mat_sparse
```




    <8076x25362 sparse matrix of type '<class 'numpy.float32'>'
    	with 5551213 stored elements in Compressed Sparse Row format>




```python
# define model
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
# fit
#model_knn.fit(movie_user_mat_sparse)
```


```python
from fuzzywuzzy import fuzz
def fuzzy_matching(mapper, fav_movie, verbose=True):
    """
    return the closest match via fuzzy ratio. 
    
    Parameters
    ----------    
    mapper: dict, map movie title name to index of the movie in data
    fav_movie: str, name of user input movie
    verbose: bool, print log if True
    Return
    ------
    index of the closest match
    """
    
    match_tuple = []
    # get match
    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
    # sort
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        print('Oops! No match is found')
        return
    if verbose:
        print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
    return match_tuple[0][1]
```

    C:\Users\mohitkumar\Anaconda3\lib\site-packages\fuzzywuzzy\fuzz.py:11: UserWarning: Using slow pure-python SequenceMatcher. Install python-Levenshtein to remove this warning
      warnings.warn('Using slow pure-python SequenceMatcher. Install python-Levenshtein to remove this warning')
    


```python
def make_recommendation(model_knn, data, mapper, fav_movie, n_recommendations):
    """
    return top n similar movie recommendations based on user's input movie


    Parameters
    ----------
    model_knn: sklearn model, knn model
    data: movie-user matrix
    mapper: dict, map movie title name to index of the movie in data
    fav_movie: str, name of user input movie
    n_recommendations: int, top n recommendations
    Return
    ------
    list of top n similar movie recommendations
    """
    
    # fit
    model_knn.fit(data)
    # get input movie index
    print('You have input movie:', fav_movie)
    idx = fuzzy_matching(mapper, fav_movie, verbose=True)
    
    print('Recommendation system start to make inference')
    print('......\n')
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=n_recommendations+1)
    
    raw_recommends = \
        sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]
    # get reverse mapper
    reverse_mapper = {v: k for k, v in mapper.items()}
    # print recommendations
    print('Recommendations for {}:'.format(fav_movie))
    for i, (idx, dist) in enumerate(raw_recommends):
        print('{0}: {1}, with distance of {2}'.format(i+1, reverse_mapper[idx], dist))
```


```python
my_favorite = 'Justice League'

make_recommendation(
    model_knn=model_knn,
    data=movie_user_mat_sparse,
    fav_movie=my_favorite,
    mapper=movie_to_idx,
    n_recommendations=10)
```

    You have input movie: Justice League
    Found possible matches in our database: ['Justice League (2017)']
    
    Recommendation system start to make inference
    ......
    
    Recommendations for Justice League:
    1: Venom (2018), with distance of 0.5993596315383911
    2: Suicide Squad (2016), with distance of 0.5991568565368652
    3: Captain Marvel (2018), with distance of 0.5978403091430664
    4: Untitled Spider-Man Reboot (2017), with distance of 0.585890531539917
    5: Ant-Man and the Wasp (2018), with distance of 0.583364725112915
    6: Black Panther (2017), with distance of 0.5739504098892212
    7: Aquaman (2018), with distance of 0.5706990957260132
    8: Wonder Woman (2017), with distance of 0.560198187828064
    9: Star Wars: The Last Jedi (2017), with distance of 0.558818519115448
    10: Batman v Superman: Dawn of Justice (2016), with distance of 0.5285896062850952
    

We can effectively identify there are two shortcomings in item based collaborative filtering:
* popularity bias: recommender is prone to recommender popular items
* item cold-start problem: recommender fails to recommend new or less-known items because items have either none or very little interactions

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Recommender-System" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

## SVD on movie lens dataset (User-User based filtering)

* Matrix factorization can be used to discover features underlying the interactions between two different kinds of entities.
* One advantage of employing matrix factorization for recommender systems is the fact that it can incorporate implicit feedback—information that’s not directly given but can be derived by analyzing user behavior—such as items frequently bought or viewed.
* Using this capability we can estimate if a user is going to like a movie that they never saw. And if that estimated rating is high, we can recommend that movie to the user, so as to provide a more personalized experience.
* SVD is a matrix factorisation technique, which reduces the number of features of a dataset by reducing the space dimension from N-dimension to K-dimension (where K<N). In the context of the recommender system, the SVD is used as a collaborative filtering technique. It uses a matrix structure where each row represents a user, and each column represents an item. The elements of this matrix are the ratings that are given to items by users.

Step 1. Load the Data into Pandas Dataframe


```python
import pandas as pd
import numpy as np
import os
data_path = 'ml-25m/'
movies_filename = 'movies.csv'
ratings_filename = 'ratings.csv'

df_movies = pd.read_csv(
    os.path.join(data_path, movies_filename),
#     movies_filename,
    usecols=['movieId', 'title'],
    dtype={'movieId': 'int32', 'title': 'str'})

df_ratings = pd.read_csv(
    os.path.join(data_path, ratings_filename),
#     ratings_filename,
    usecols=['userId', 'movieId', 'rating'],
    
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
```

We pivot the dataframe to have userId as rows and movieId as columns, filling the null values with 0.0.


```python
df_ratings=df_ratings[:3125012]
df_movie_features = df_ratings.pivot(
    index='userId',
    columns='movieId',
    values='rating'
).fillna(0)
```


```python
R = df_movie_features.values
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)
```

Here, we will be using the scipy library in Python to implement SVD.


```python
from scipy.sparse.linalg import svds
U, sigma, Vt = svds(R_demeaned, k = 50)
# that the Sigma$ returned is just the values instead of a diagonal matrix. 
# This is useful, but since I'm going to leverage matrix multiplication to get predictions 
# I'll convert it to the diagonal matrix form.
sigma = np.diag(sigma)
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
```


```python
preds_df = pd.DataFrame(all_user_predicted_ratings, columns = df_movie_features.columns)
preds_df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>movieId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>208791</th>
      <th>208793</th>
      <th>208795</th>
      <th>208800</th>
      <th>208939</th>
      <th>209049</th>
      <th>209053</th>
      <th>209055</th>
      <th>209103</th>
      <th>209163</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.737401</td>
      <td>0.004221</td>
      <td>-0.047771</td>
      <td>-0.040918</td>
      <td>-0.027591</td>
      <td>-0.308165</td>
      <td>-0.003310</td>
      <td>-0.027573</td>
      <td>-0.012109</td>
      <td>0.122300</td>
      <td>...</td>
      <td>0.001757</td>
      <td>-0.004602</td>
      <td>-0.001823</td>
      <td>-0.003284</td>
      <td>-0.002494</td>
      <td>-0.000516</td>
      <td>-0.003018</td>
      <td>-0.003018</td>
      <td>-0.001044</td>
      <td>0.001097</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.187437</td>
      <td>0.494850</td>
      <td>0.070980</td>
      <td>-0.008444</td>
      <td>-0.117596</td>
      <td>0.379037</td>
      <td>0.100695</td>
      <td>0.093709</td>
      <td>-0.112567</td>
      <td>0.474100</td>
      <td>...</td>
      <td>-0.002033</td>
      <td>-0.002635</td>
      <td>0.006150</td>
      <td>0.001208</td>
      <td>-0.010914</td>
      <td>0.001051</td>
      <td>-0.002895</td>
      <td>-0.002895</td>
      <td>-0.008633</td>
      <td>0.000661</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.587274</td>
      <td>0.591198</td>
      <td>-0.307573</td>
      <td>-0.097345</td>
      <td>-0.032214</td>
      <td>0.708733</td>
      <td>0.186531</td>
      <td>-0.141126</td>
      <td>-0.106800</td>
      <td>-0.524378</td>
      <td>...</td>
      <td>-0.010914</td>
      <td>-0.013666</td>
      <td>-0.023079</td>
      <td>-0.007592</td>
      <td>-0.001193</td>
      <td>-0.001763</td>
      <td>-0.015496</td>
      <td>-0.015496</td>
      <td>-0.005212</td>
      <td>-0.011583</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2.562247</td>
      <td>0.290040</td>
      <td>-0.118626</td>
      <td>-0.030613</td>
      <td>-0.076762</td>
      <td>0.152793</td>
      <td>0.067345</td>
      <td>-0.026693</td>
      <td>-0.003602</td>
      <td>0.301227</td>
      <td>...</td>
      <td>-0.004437</td>
      <td>-0.007541</td>
      <td>-0.003240</td>
      <td>-0.000544</td>
      <td>-0.006979</td>
      <td>-0.002707</td>
      <td>-0.003732</td>
      <td>-0.003732</td>
      <td>-0.009087</td>
      <td>0.000631</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.443969</td>
      <td>0.982385</td>
      <td>1.359526</td>
      <td>0.135161</td>
      <td>1.137983</td>
      <td>1.721156</td>
      <td>1.118443</td>
      <td>0.072305</td>
      <td>0.504911</td>
      <td>1.218184</td>
      <td>...</td>
      <td>-0.001786</td>
      <td>-0.000672</td>
      <td>-0.001863</td>
      <td>-0.002384</td>
      <td>-0.006494</td>
      <td>0.000157</td>
      <td>-0.000126</td>
      <td>-0.000126</td>
      <td>-0.000736</td>
      <td>0.000396</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32146 columns</p>
</div>



Here, we just make a function that uses our factorized matrices to recommend movies to a user, given a user_id.


```python
def recommend_movies(preds_df, userID, movies_df, original_ratings_df, num_recommendations=5):
    
    # Get and sort the user's predictions
    user_row_number = userID - 1 # UserID starts at 1, not 0
    sorted_user_predictions = preds_df.iloc[user_row_number].sort_values(ascending=False) # UserID starts at 1
#     print(preds_df.iloc[user_row_number])
#     print(sorted_user_predictions)
    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.userId == (userID)]
    user_full = (user_data.merge(movies_df, how = 'left', left_on = 'movieId', right_on = 'movieId').
                     sort_values(['rating'], ascending=False)
                 )
#     print(user_full)
#     print 'User {0} has already rated {1} movies.'.format(userID, user_full.shape[0])
#     print 'Recommending highest {0} predicted ratings movies not already rated.'.format(num_recommendations)
    #                left_on = 'movieId',
#                right_on = 'movieId').
# merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left').rename(columns = {user_row_number: 'Predictions'}).
    # Recommend the highest predicted rating movies that the user hasn't seen yet.
    recommendations = (movies_df[~movies_df['movieId'].isin(user_full['movieId'])]).merge(pd.DataFrame(sorted_user_predictions).reset_index(), how = 'left', left_on = 'movieId',
               right_on = 'movieId').rename(columns = {user_row_number: 'Predictions'}).sort_values('Predictions', ascending = False).iloc[:num_recommendations, :-1]
                      

    return user_full, recommendations
```


```python
already_rated, predictions = recommend_movies(preds_df, 3, df_movies, df_ratings, 10)
```


```python
already_rated.head(10)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>userId</th>
      <th>movieId</th>
      <th>rating</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>628</th>
      <td>3</td>
      <td>136449</td>
      <td>5.0</td>
      <td>Ghost in the Shell 2.0 (2008)</td>
    </tr>
    <tr>
      <th>20</th>
      <td>3</td>
      <td>745</td>
      <td>5.0</td>
      <td>Wallace &amp; Gromit: A Close Shave (1995)</td>
    </tr>
    <tr>
      <th>479</th>
      <td>3</td>
      <td>81591</td>
      <td>5.0</td>
      <td>Black Swan (2010)</td>
    </tr>
    <tr>
      <th>23</th>
      <td>3</td>
      <td>858</td>
      <td>5.0</td>
      <td>Godfather, The (1972)</td>
    </tr>
    <tr>
      <th>26</th>
      <td>3</td>
      <td>924</td>
      <td>5.0</td>
      <td>2001: A Space Odyssey (1968)</td>
    </tr>
    <tr>
      <th>350</th>
      <td>3</td>
      <td>48516</td>
      <td>5.0</td>
      <td>Departed, The (2006)</td>
    </tr>
    <tr>
      <th>29</th>
      <td>3</td>
      <td>1148</td>
      <td>5.0</td>
      <td>Wallace &amp; Gromit: The Wrong Trousers (1993)</td>
    </tr>
    <tr>
      <th>289</th>
      <td>3</td>
      <td>27728</td>
      <td>5.0</td>
      <td>Ghost in the Shell 2: Innocence (a.k.a. Innoce...</td>
    </tr>
    <tr>
      <th>37</th>
      <td>3</td>
      <td>1213</td>
      <td>5.0</td>
      <td>Goodfellas (1990)</td>
    </tr>
    <tr>
      <th>290</th>
      <td>3</td>
      <td>27773</td>
      <td>5.0</td>
      <td>Old Boy (2003)</td>
    </tr>
  </tbody>
</table>
</div>




```python
predictions
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>movieId</th>
      <th>title</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>24453</th>
      <td>122904</td>
      <td>Deadpool (2016)</td>
    </tr>
    <tr>
      <th>6879</th>
      <td>7254</td>
      <td>The Butterfly Effect (2004)</td>
    </tr>
    <tr>
      <th>7655</th>
      <td>8636</td>
      <td>Spider-Man 2 (2004)</td>
    </tr>
    <tr>
      <th>4744</th>
      <td>5010</td>
      <td>Black Hawk Down (2001)</td>
    </tr>
    <tr>
      <th>1134</th>
      <td>1193</td>
      <td>One Flew Over the Cuckoo's Nest (1975)</td>
    </tr>
    <tr>
      <th>581</th>
      <td>608</td>
      <td>Fargo (1996)</td>
    </tr>
    <tr>
      <th>4129</th>
      <td>4370</td>
      <td>A.I. Artificial Intelligence (2001)</td>
    </tr>
    <tr>
      <th>23140</th>
      <td>119145</td>
      <td>Kingsman: The Secret Service (2015)</td>
    </tr>
    <tr>
      <th>4038</th>
      <td>4270</td>
      <td>Mummy Returns, The (2001)</td>
    </tr>
    <tr>
      <th>7974</th>
      <td>8972</td>
      <td>National Treasure (2004)</td>
    </tr>
  </tbody>
</table>
</div>



One typical problem caused by the data sparsity is the cold start problem. As collaborative filtering methods recommend items based on users’ past preferences, new users will need to rate a sufficient number of items to enable the system to capture their preferences accurately, and thus provides reliable recommendations.

![png](https://github.com/Affineindia/ML-Best-Practices-Standardized-Codes/blob/master/Python/Images/Recommender%20System/download(5).png)

#### Precision = the number of items that I liked that were also recommended to me/ the number of items that were recommended
 

#### Recall = the number of items that I liked that were also recommended to me/ the number of items I liked

Reference: https://surprise.readthedocs.io/en/latest/FAQ.html#how-to-compute-precision-k-and-recall-k

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Recommender-System" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>

# Hybrid Recommender System

**Hybrid recommender system** is the one that combines multiple
recommendation techniques together to produce the output. If one compares hybrid
recommender systems with collaborative or content-based systems, the recommendation
accuracy is usually higher in hybrid systems. The reason is the lack of information about the
domain dependencies in collaborative filtering, and about the people’s preferences in
content-based system. The combination of both leads to common knowledge increase, which
contributes to better recommendations. The knowledge increase makes it especially
promising to explore new ways to extend underlying collaborative filtering algorithms with
content data and content-based algorithms with the user behavior data.


**Step 1** Importing the required libraries


```python
from lightfm import LightFM
from sklearn.metrics.pairwise import cosine_similarity
```

**Step 2** Define the required functions for creating a recommender system

Helper functions to build recommender systems using Matrix factorization using LightFM package.


```python
def create_interaction_matrix(df,user_col, item_col, rating_col, norm= False, threshold = None):
    '''
    Function to create an interaction matrix dataframe from transactional type interactions
    Required Input -
        - df = Pandas DataFrame containing user-item interactions
        - user_col = column name containing user's identifier
        - item_col = column name containing item's identifier
        - rating col = column name containing user feedback on interaction with a given item
        - norm (optional) = True if a normalization of ratings is needed
        - threshold (required if norm = True) = value above which the rating is favorable
    Expected output - 
        - Pandas dataframe with user-item interactions ready to be fed in a recommendation algorithm
    '''
    interactions = df.groupby([user_col, item_col])[rating_col] \
            .sum().unstack().reset_index(). \
            fillna(0).set_index(user_col)
    if norm:
        interactions = interactions.applymap(lambda x: 1 if x > threshold else 0)
    return interactions
```


```python
def create_user_dict(interactions):
    '''
    Function to create a user dictionary based on their index and number in interaction dataset
    Required Input - 
        interactions - dataset create by create_interaction_matrix
    Expected Output -
        user_dict - Dictionary type output containing interaction_index as key and user_id as value
    '''
    user_id = list(interactions.index)
    user_dict = {}
    counter = 0 
    for i in user_id:
        user_dict[i] = counter
        counter += 1
    return user_dict
```


```python
def create_item_dict(df,id_col,name_col):
    '''
    Function to create an item dictionary based on their item_id and item name
    Required Input - 
        - df = Pandas dataframe with Item information
        - id_col = Column name containing unique identifier for an item
        - name_col = Column name containing name of the item
    Expected Output -
        item_dict = Dictionary type output containing item_id as key and item_name as value
    '''
    item_dict ={}
    for i in range(df.shape[0]):
        item_dict[(df.loc[i,id_col])] = df.loc[i,name_col]
    return item_dict
```


```python
def runMF(interactions, n_components=30, loss='warp', k=15, epoch=30,n_jobs = 4):
    '''
    Function to run matrix-factorization algorithm
    Required Input -
        - interactions = dataset create by create_interaction_matrix
        - n_components = number of embeddings you want to create to define Item and user
        - loss = loss function other options are logistic, brp
        - epoch = number of epochs to run 
        - n_jobs = number of cores used for execution 
    Expected Output  -
        Model - Trained model
    '''
    x = sparse.csr_matrix(interactions.values)
    model = LightFM(no_components= n_components, loss=loss,k=k)
    model.fit(x,epochs=epoch,num_threads = n_jobs)
    return model
```


```python
def sample_recommendation_user(model, interactions, user_id, user_dict, 
                               item_dict,threshold = 0,nrec_items = 10, show = True):
    '''
    Function to produce user recommendations
    Required Input - 
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
        - user_id = user ID for which we need to generate recommendation
        - user_dict = Dictionary type input containing interaction_index as key and user_id as value
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - threshold = value above which the rating is favorable in new interaction matrix
        - nrec_items = Number of output recommendation needed
    Expected Output - 
        - Prints list of items the given user has already bought
        - Prints list of N recommended items  which user hopefully will be interested in
    '''
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(model.predict(user_x,np.arange(n_items)))
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    known_items = list(pd.Series(interactions.loc[user_id,:] \
                                 [interactions.loc[user_id,:] > threshold].index) \
                                 .sort_values(ascending=False))
    
    scores = [x for x in scores if x not in known_items]
    return_score_list = scores[0:nrec_items]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    scores = list(pd.Series(return_score_list).apply(lambda x: item_dict[x]))
    if show == True:
        print("Known Likes:")
        counter = 1
        for i in known_items:
            print(str(counter) + '- ' + i)
            counter+=1

        print("\n Recommended Items:")
        counter = 1
        for i in scores:
            print(str(counter) + '- ' + i)
            counter+=1
    return return_score_list
    
```


```python
def sample_recommendation_item(model,interactions,item_id,user_dict,item_dict,number_of_user):
    '''
    Funnction to produce a list of top N interested users for a given item
    Required Input -
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
        - item_id = item ID for which we need to generate recommended users
        - user_dict =  Dictionary type input containing interaction_index as key and user_id as value
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - number_of_user = Number of users needed as an output
    Expected Output -
        - user_list = List of recommended users 
    '''
    n_users, n_items = interactions.shape
    x = np.array(interactions.columns)
    scores = pd.Series(model.predict(np.arange(n_users), np.repeat(x.searchsorted(item_id),n_users)))
    user_list = list(interactions.index[scores.sort_values(ascending=False).head(number_of_user).index])
    return user_list 
```


```python
def create_item_emdedding_distance_matrix(model,interactions):
    '''
    Function to create item-item distance embedding matrix
    Required Input -
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
    Expected Output -
        - item_emdedding_distance_matrix = Pandas dataframe containing cosine distance matrix b/w items
    '''
    df_item_norm_sparse = sparse.csr_matrix(model.item_embeddings)
    similarities = cosine_similarity(df_item_norm_sparse)
    item_emdedding_distance_matrix = pd.DataFrame(similarities)
    item_emdedding_distance_matrix.columns = interactions.columns
    item_emdedding_distance_matrix.index = interactions.columns
    return item_emdedding_distance_matrix
```


```python
def item_item_recommendation(item_emdedding_distance_matrix, item_id, 
                             item_dict, n_items = 10, show = True):
    '''
    Function to create item-item recommendation
    Required Input - 
        - item_emdedding_distance_matrix = Pandas dataframe containing cosine distance matrix b/w items
        - item_id  = item ID for which we need to generate recommended items
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - n_items = Number of items needed as an output
    Expected Output -
        - recommended_items = List of recommended items
    '''
    recommended_items = list(pd.Series(item_emdedding_distance_matrix.loc[item_id,:]. \
                                  sort_values(ascending = False).head(n_items+1). \
                                  index[1:n_items+1]))
    if show == True:
        print("Item of interest :{0}".format(item_dict[item_id]))
        print("Item similar to the above item:")
        counter = 1
        for i in recommended_items:
            print(str(counter) + '- ' +  item_dict[i])
            counter+=1
    return recommended_items
```

**Step 4** Loading the data


```python
data_path = 'ml-25m/'
movies_filename = 'movies.csv'
ratings_filename = 'ratings.csv'
df_movies = pd.read_csv(
    os.path.join(data_path, movies_filename),
usecols=['movieId', 'title'],
    dtype={'movieId': 'int32', 'title': 'str'})
df_ratings = pd.read_csv(
    os.path.join(data_path, ratings_filename),
    usecols=['userId', 'movieId', 'rating'],
    dtype={'userId': 'int32', 'movieId': 'int32', 'rating': 'float32'})
```

**Step 5** Create interaction matrix


```python
df_ratings=df_ratings[:3125012] ##Size is reduced to fit the data in memory
interactions = create_interaction_matrix(df = df_ratings,
                                         user_col = 'userId',
                                         item_col = 'movieId',
                                         rating_col = 'rating',
                                         threshold = '3')
interactions.shape
```




    (20593, 32146)




```python
interactions.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>movieId</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>...</th>
      <th>208791</th>
      <th>208793</th>
      <th>208795</th>
      <th>208800</th>
      <th>208939</th>
      <th>209049</th>
      <th>209053</th>
      <th>209055</th>
      <th>209103</th>
      <th>209163</th>
    </tr>
    <tr>
      <th>userId</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.5</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 32146 columns</p>
</div>



**Step 6** Create User Dict


```python
user_dict = create_user_dict(interactions=interactions)
```

**Step 7** Create Item dict


```python
movies_dict = create_item_dict(df = df_movies,
                               id_col = 'movieId',
                               name_col = 'title')
```

**Step 8** Building Matrix Factorization model


```python
mf_model = runMF(interactions = interactions,
                 n_components = 30,
                 loss = 'warp',
                 k = 15,
                 epoch = 30,
                 n_jobs = 4)
```

**Step 9** User Recommender


```python
rec_list = sample_recommendation_user(model = mf_model, 
                                      interactions = interactions, 
                                      user_id = 11, 
                                      user_dict = user_dict,
                                      item_dict = movies_dict, 
                                      threshold = 4,
                                      nrec_items = 10)
```

    Known Likes:
    1- Simpsons Movie, The (2007)
    2- Silence of the Lambs, The (1991)
    3- Reality Bites (1994)
    
     Recommended Items:
    1- Shrek (2001)
    2- Fight Club (1999)
    3- American Beauty (1999)
    4- Forrest Gump (1994)
    5- Lord of the Rings: The Fellowship of the Ring, The (2001)
    6- American Pie (1999)
    7- Finding Nemo (2003)
    8- Monsters, Inc. (2001)
    9- Sixth Sense, The (1999)
    10- Shawshank Redemption, The (1994)
    

**Step 10** Item - User Recommender


```python
#list of top N interested users for a given item
sample_recommendation_item(model = mf_model,
                           interactions = interactions,
                           item_id = 1,
                           user_dict = user_dict,
                           item_dict = movies_dict,
                           number_of_user = 15)
```




    [111,
     1135,
     15441,
     16754,
     15578,
     15968,
     6092,
     4846,
     19035,
     15380,
     1144,
     10073,
     19592,
     3094,
     3920]



**Step 11**  Item - Item Recommender


```python
item_item_dist = create_item_emdedding_distance_matrix(model = mf_model,
                                                       interactions = interactions)
```


```python
rec_list = item_item_recommendation(item_emdedding_distance_matrix = item_item_dist,
                                    item_id = 695,
                                    item_dict = movies_dict,
                                    n_items = 10)
```

    Item of interest :True Crime (1996)
    Item similar to the above item:
    1- Guilty as Sin (1993)
    2- Playing God (1997)
    3- Switchback (1997)
    4- Extreme Measures (1996)
    5- Night Falls on Manhattan (1996)
    6- Palmetto (1998)
    7- Newton Boys, The (1998)
    8- Mortal Thoughts (1991)
    9- Truth or Consequences, N.M. (1997)
    10- Flesh and Bone (1993)
    

**Hybrid filtering technique** combines different recommendation techniques in order to gain better system optimization to avoid some limitations and problems of pure recommendation systems. The idea behind hybrid techniques is that a combination of algorithms will provide more accurate and effective recommendations than a single algorithm as the disadvantages of one algorithm can be overcome by another algorithm. Using multiple recommendation techniques can suppress the weaknesses of an individual technique in a combined model. The combination of approaches can be done in any of the following ways: separate implementation of algorithms and combining the result, utilizing some content-based filtering in collaborative approach, utilizing some collaborative filtering in content-based approach, creating a unified recommendation system that brings together both approaches.

**REFERENCE:** https://www.sciencedirect.com/science/article/pii/S1110866515000341

<a class="list-group-item list-group-item-action" data-toggle="list" href="#Recommender-System" role="tab" aria-controls="profile">Go to top<span class="badge badge-primary badge-pill"></span></a>


```python

```
