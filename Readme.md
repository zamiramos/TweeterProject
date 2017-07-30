
# Part 1 -  Collecting Data (Tweets from NY)

## Data - PreProcessing

Because we don't have property of gender in tweet structure, we needed to use the NLTK well known names to build our train dataset.

In order to extract as much as possible the correct gender, we filter all the names which can be use for female and for male.


```python
#import nltk
#nltk.download()

from nltk.corpus import names

def GetWellDefineGenderFromName(first_name):
    first_name = first_name.lower()
    first_name = first_name.title()
    matchMale = False
    matchFemale = False
    if first_name in names.words("male.txt"):
        matchMale = True
    if first_name in names.words("female.txt"):
        matchFemale = True
    if matchMale and matchFemale:
        return None
        
    if matchMale:        
        return 'male'
        
    if matchFemale:
        return 'female'        
        
    return None
```

## Fetch raw data from tweeter

We choose that our population will be people from NY.

In order to get a meaningful tweets (means no commercial tweets from real people),we did the following:

1. Search for the words 'The', 'I', 'she', 'and'.
2. Ensure that the user language is english.
3. Ensure that the user location is NY.
4. Ensure that the name gender is well define as explaind above. 


```python
#!conda install -c conda-forge tweepy=3.5.0
#!conda install -c malev gender_detector=0.1.0

import tweepy
from tweepy import Stream
from tweepy.streaming import StreamListener
import requests
import json

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

class StdOutListener(StreamListener):
    def on_data(self, data):            
        
        tweet = json.loads(data)
        
        if not tweet.get('user'):
            return True
        
        user = tweet['user']    
        
        if user['lang'] != 'en':
            return True
        
        if user['location'] is None:
            return True
        
        if 'New York' not in user['location'] and 'NY' not in user['location']:
            return True            
                
        gender = GetWellDefineGenderFromName(user['name'].encode('utf-8').decode().split()[0])
        
        if gender is None:            
            return True            
        
        # append the hourly tweet file
        with open('tweets.data', 'a+') as f:
            f.write(data)            
        
        return True
    
    def on_error(self, status):
        print('status: %s' % status)

streamListener = StdOutListener()
stream = Stream(auth, streamListener, timeout=30)


####
#call the below function when you want to fetch data
####

#Remove the comment below for running

'''
stream.filter(locations=[-74,40,-73,41], track=['The', 'I', 'she', 'and'])
'''
```




    "\nstream.filter(locations=[-74,40,-73,41], track=['The', 'I', 'she', 'and'])\n"



## Data Cleaning - Define fucntions

### Decoding text to Ascii

Most of the NL Algorithms works with ASCII.


```python
def UTFToAscii(string):
    return string.decode('ascii', 'ignore')
```

### Extract URLs, @user_reference, hashtags Count


```python
from urllib.parse import urlparse

def ExtractReference(string):
    reference_count = 0
    hashtag_count = 0
    urls_count = 0
    for i in string.split():
        s, n, p, pa, q, f = urlparse(i)
        if s and n:
            urls_count = urls_count + 1
        elif i[:1] == '@':
            reference_count = reference_count + 1
        elif i[:1] == '#':
            hashtag_count = hashtag_count + 1            
        else:
            pass
    return reference_count, hashtag_count, urls_count
```

### Remove URLs, hashtags, @user_reference


```python
import re

def RemoveNonWords(string):
    return re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", string)
```

### Handle Escaping Characters


```python
from html.parser import HTMLParser

def RemoveEscaping(string):
    html_parser = HTMLParser()
    return html_parser.unescape(string)
```

### Convert to lowercase


```python
def ToLowercase(string):
    return string.lower()
```

### Handle Apostrophe Lookup


```python
import csv

def ReplaceApostrophe(string):
    appostophes_dict = None

    with open('appostophes.csv', mode='r') as infile:
        reader = csv.reader(infile)
        appostophes_dict = {rows[0]:rows[1] for rows in reader}
    
    words = string.split()
    reformed = [appostophes_dict[word] if word in appostophes_dict else word for word in words]
    string = " ".join(reformed)
    return string
    
```

### Clean RT


```python
def RemoveRT(string):
    words = string.split()
    reformed = ['' if word in 'RT' else word for word in words]
    string = " ".join(reformed)
    return string
```

### Flat Lines


```python
def FlatLines(string):
    lines = string.splitlines()
    string = " ".join(lines)
    return string
```

### Filter To First Line


```python
def GetFirstLine(string):
    lines = string.splitlines()  
    return lines[0]
```

### Stemming


```python
from nltk.stem import PorterStemmer

def Stemming(string):
    ps = PorterStemmer()
    
    words = string.split()
    words = map(ps.stem, words)   
    
    string = " ".join(words)
    return string
```

## Create dataframe from Json tweets file


```python
import pandas as pd
import json

def IsEnglish(s):
    for i in s.split():
        if not i.isalpha():
            return False
    return True    

#read data from tweets.data

tweets_df = pd.DataFrame(columns=['id', 'text', 'reference_count', 'hashtag_count', 'urls_count' ,'name', 'gender'])
for line in open('tweets.data', 'rb'):        
    tweet_data = json.loads(line)
    tweet_fullname = tweet_data['user']['name'].encode().decode('utf-8')
    if IsEnglish(tweet_fullname) == False:
        continue

    tweet_text = tweet_data['text'].encode('utf-8')
    tweet_text = UTFToAscii(tweet_text)
    tweet_text = RemoveEscaping(tweet_text)    
    reference_count, hashtag_count, urls_count = ExtractReference(tweet_text)
    tweet_text = RemoveNonWords(tweet_text)
    tweet_text = RemoveRT(tweet_text)
    tweet_text = tweet_text.strip()
    tweet_text = ToLowercase(tweet_text)

    #filter all text that smaller then 2 words
    if len(tweet_text.split()) < 1:
        continue
    
    gender = GetWellDefineGenderFromName(tweet_fullname.split()[0])
    
    if gender is None:
        continue

    tweets_df.loc[len(tweets_df)]=[tweet_data['id_str'], tweet_text, reference_count, hashtag_count, urls_count, tweet_fullname, gender]

```

    C:\Program Files\Anaconda3\lib\site-packages\ipykernel\__main__.py:5: DeprecationWarning: The unescape method is deprecated and will be removed in 3.5, use html.unescape() instead.
    

## Write Dataframe as CSV File


```python
tweets_df.to_csv('tweets_df.csv')
```

## Data Exploration

First quick look in the data:


```python
tweets_df.head(10)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>text</th>
      <th>reference_count</th>
      <th>hashtag_count</th>
      <th>urls_count</th>
      <th>name</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>875088051977977856</td>
      <td>how to give your employees the recognition the...</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Guy Santeramo</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>875088067471831043</td>
      <td>that was lovely thank you for sharing i too lo...</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>Alexa Harrison</td>
      <td>female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>875088072786022400</td>
      <td>sling could a team made entirely of players wh...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>Jay Zampi</td>
      <td>male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>875088110736089089</td>
      <td>my fan theory is that your fan theory has noth...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Susana Polo</td>
      <td>female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>875088118990462976</td>
      <td>psst youre not blessed youre just lucky as hel...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>JEFF BRANDT</td>
      <td>male</td>
    </tr>
    <tr>
      <th>5</th>
      <td>875088123126009857</td>
      <td>drinking age should be lowered to 18</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Cassie</td>
      <td>female</td>
    </tr>
    <tr>
      <th>6</th>
      <td>875088132122836994</td>
      <td>in the wake of alexandria and san francisco up...</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>Ralph</td>
      <td>male</td>
    </tr>
    <tr>
      <th>7</th>
      <td>875088135889334273</td>
      <td>if you say so princess im sure hell love to se...</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Brad Gibson</td>
      <td>male</td>
    </tr>
    <tr>
      <th>8</th>
      <td>875088148770033665</td>
      <td>when she asks you what you bring to the table</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>Solomon Grundy</td>
      <td>male</td>
    </tr>
    <tr>
      <th>9</th>
      <td>875088173939818497</td>
      <td>finally biting the bullet and reserving hotel ...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>Meg Roy</td>
      <td>female</td>
    </tr>
  </tbody>
</table>
</div>



The texts are look good and also the names the and the gender.


```python
tweets_df.describe(include='all')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>text</th>
      <th>reference_count</th>
      <th>hashtag_count</th>
      <th>urls_count</th>
      <th>name</th>
      <th>gender</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>11889</td>
      <td>11889</td>
      <td>11889</td>
      <td>11889</td>
      <td>11889</td>
      <td>11889</td>
      <td>11889</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>11888</td>
      <td>11133</td>
      <td>11</td>
      <td>13</td>
      <td>4</td>
      <td>7941</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>879427176537489409</td>
      <td>rowling 20 years ago today a world that i had ...</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>Joe Dicandia</td>
      <td>male</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>2</td>
      <td>21</td>
      <td>6430</td>
      <td>9672</td>
      <td>6159</td>
      <td>39</td>
      <td>6530</td>
    </tr>
  </tbody>
</table>
</div>




```python
tweets_df['gender'] = tweets_df['gender'].astype('category')
tweets_df.gender.value_counts()
```




    male      6530
    female    5359
    Name: gender, dtype: int64



We have 4133 samples.

female: ~0.39
male:~0.61

Types:


```python
tweets_df.dtypes
```




    id                   object
    text                 object
    reference_count      object
    hashtag_count        object
    urls_count           object
    name                 object
    gender             category
    dtype: object



Should not be missing values:

### Male ,Female Ratio


```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

%matplotlib inline

plt.figure();

tweets_df.gender.value_counts().plot(kind='pie', colors=['blue','red'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1360d243a20>




![png](output_44_1.png)


## Tweet Reference, Hashtag, URL Count By Gender


Let's explore the information which we extract from the original tweet text.



```python
var = pd.crosstab(tweets_df['reference_count'], tweets_df['gender'])
var.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x135b46d3dd8>




![png](output_47_1.png)



```python
var = pd.crosstab(tweets_df['hashtag_count'], tweets_df['gender'])
var.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1360d1eb3c8>




![png](output_48_1.png)



```python
var = pd.crosstab(tweets_df['urls_count'], tweets_df['gender'])
var.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x136049aef28>




![png](output_49_1.png)


We can see from above that these features not give us an additional information. (the male/female ratio remains almost the same for each value).

## Backup Dataframe to Pickle file


```python
tweets_df.to_pickle('DataFramePickle')
```

# Part 2 - Train Classifier For Gender

## 2.1 Try 1 - DT,  Naive bayes,  SVM linear Classifiers based on TF-IDF features

### Algorithms Description

TF-IDF - In information retrieval, tf–idf, short for term frequency–inverse document frequency, is a numerical statistic that is intended to reflect how important a word is to a document in a collection or corpus.[1] It is often used as a weighting factor in information retrieval, text mining, and user modeling. The tf-idf value increases proportionally to the number of times a word appears in the document, but is often offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words appear more frequently in general. Nowadays, tf-idf is one of the most popular term-weighting schemes. For instance, 83% of text-based recommender systems in the domain of digital libraries use tf-idf
(https://en.wikipedia.org/wiki/Tf%E2%80%93idf)


DT - A decision tree is a decision support tool that uses a tree-like graph or model of decisions and their possible consequences, including chance event outcomes, resource costs, and utility. It is one way to display an algorithm.
(https://en.wikipedia.org/wiki/Decision_tree)

Naive bayes - It is a classification technique based on Bayes' Theorem with an assumption of independence among predictors. In simple terms, a Naive Bayes classifier assumes that the presence of a particular feature in a class is unrelated to the presence of any other feature.
(https://en.wikipedia.org/wiki/Naive_Bayes_classifier)

SVM linear - In machine learning, support vector machines (SVMs, also support vector networks[1]) are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in a probabilistic classification setting). An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category based on on which side of the gap they fall. (https://en.wikipedia.org/wiki/Support_vector_machine)

## Global Configuration


```python
import os

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
        
write_path = 'Models\\'

ensure_dir(write_path)
```

## Function: Top features for model


```python
def print_topFeatures( classifier,stri,numOfFeatures):
    """Prints features with the highest coefficient values, per class"""
    dfSignificantFeatures = pd.DataFrame()
    class_labels = classifier.classes_
    for i, class_label in enumerate(class_labels):
        if len(class_labels)>2:
            coef = classifier.coef_[i]
        else:
            coef = classifier.feature_log_prob_[i]

        top20 = np.argsort(coef)[-40:]
        toList = (" %s" % (",".join(feature_names[j] for j in top20)))
        list = toList.split(',')
        dfSignificantFeatures[class_label.upper()] = list[::-1]
        
    dfSignificantFeatures.to_csv(write_path +'_best40Features_allWords_'+str(numOfFeatures)+stri+'.csv', mode='a', sep=',')
```

## Function: Split the data to train and test sets


```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_test_set(texts, gender, RemoveStopWords, f, t, analyzer, binary):

    corpus = np.array(texts.values).tolist()
    target = np.array(gender.values)
    #tf-idf vectorizer is set to count word occurances and normalize. NOT TF-IDF
    if RemoveStopWords:
        vectorizer = TfidfVectorizer(stop_words='english',analyzer=analyzer,use_idf=False, ngram_range=(f,t),binary=binary)
    else: #only SW
        vectorizer = TfidfVectorizer(analyzer=analyzer, use_idf=False, ngram_range=(f,t),binary=binary,norm='l1')
    
    X = vectorizer.fit_transform(corpus)
    global feature_names
    feature_names = vectorizer.get_feature_names()

    return X, target, feature_names, vectorizer
```

## Function: Plots a confusion matrix


```python
import matplotlib.pyplot as plt

def plot_cnf_matrix(cms, classes, model_name, numOfFeatures):
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(cms), cmap=plt.cm.jet, interpolation='nearest')
    
    width, height = cms.shape
    
    try: 
        xrange 
    except NameError: 
        xrange = range
    
    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(cms[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')
    
    cb = fig.colorbar(res)
    plt.xticks(range(width), classes)
    plt.yticks(range(height), classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title("Confusion Matrix " + str(numOfFeatures) + ' ' + model_name)
    plt.show()
```

## Function: Trains the model k-fold times and returns total accuracy


```python
from sklearn.metrics import confusion_matrix

def train_model_kf_cv(model, kf, data, target, numFolds, model_name, ngram):
    cm = []
    error = []
    for train_indices, test_indices in kf:
        # Get the dataset; this is the way to access values in a pandas DataFrame
        train_X = data[train_indices, :]
        train_Y = target[train_indices]
        test_X = data[test_indices, :]
        test_Y = target[test_indices]
        # Train the model
        model.fit(train_X, train_Y)
        predictions = model.predict(test_X)
        # Evaluate the model
        ###fpr, tpr, _ = roc_curve(test_Y, predictions)
        classes = model.classes_                
        cm.append(confusion_matrix(test_Y, predictions, labels=classes))
        ###total += auc(fpr, tpr)
        error.append(model.score(test_X, test_Y))
    accuracy = np.mean(error)
    for i in range(0,9):
        cms = np.mean(cm, axis=0)
    plot_cnf_matrix(cms, classes, model_name,ngram)
    ###auc = total / numFolds
    ###print "AUC of {0}: {1}".format(Model.__name__, accuracy)
    return accuracy
```

## Function to balanced data (50% Male, 50% Female)


```python
from numpy import unique
from numpy import random 
import operator

def balanced_sample_maker(X, y, random_seed=None):
    uniq_levels = unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}    

    if not random_seed is None:
        random.seed(random_seed)
    
    min_level_index = None
    if uniq_counts[uniq_levels[0]] < uniq_counts[uniq_levels[1]]:
        min_level_index = 0
    else:
        min_level_index = 1
    
    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx

    # oversampling on observations of positive label    
    sample_size = uniq_counts[uniq_levels[min_level_index]]
    over_sample_idx = numpy.random.choice(groupby_levels[uniq_levels[1-min_level_index]], size=sample_size, replace=True).tolist()
    balanced_copy_idx = groupby_levels[uniq_levels[min_level_index]] + over_sample_idx
    random.shuffle(balanced_copy_idx)

    return X[balanced_copy_idx], y[balanced_copy_idx]
```

## Function: Plot Learning Curve


```python
import numpy as np
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, \
                        ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")    

    return plt
```

## Function: Classify our data

df - Data

RemoveStopWords - If true remove stopping words

Analyzer - Specify 'word'/'char'- the minimum token unit

ngrams - the size of n-gram


```python
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier
import pickle
import sklearn.naive_bayes as nb
import sklearn
from sklearn import tree
from sklearn import svm

def Classify(df, RemoveStopWords, analyzer, ngrams):
    a = {'Classifer': pd.Series(index=['DT', 'Naive bayes', 'SVM linear' ]),}
    scoreTable = pd.DataFrame(a)

    scoreTable.__delitem__('Classifer')

    dfReduced = df.copy(deep=True)
    texts, labels = balanced_sample_maker(dfReduced['text'],dfReduced['gender']) 

    #if we want to check for several x top featurs
    for ngram in ngrams:
        nfolds=5
        accuracies = []
        data, target, feat_names, vectorizer = extract_test_set(texts, labels, RemoveStopWords, 1,ngram, analyzer, False)
        pickle.dump(vectorizer, open("Vectorizer_" + str(ngram) + ".pickle", 'wb'))
        kf = KFold(data.shape[0], n_folds = nfolds, shuffle = True, random_state = 1)        

        dtree = DecisionTreeClassifier(random_state=0, max_depth=50)
        plot_learning_curve(dtree, "DT", data, target, cv=kf, n_jobs=4)
        acc = train_model_kf_cv(dtree, kf, data, target, nfolds, 'DT',ngram)
        accuracies.append(acc)
        #pickle.dump(dtree, open(write_path+"dtree"+str(numOfFeatures), 'wb'))

        bayes = nb.MultinomialNB()
        plot_learning_curve(bayes, "Bayes", data, target, cv=kf, n_jobs=4)
        acc = train_model_kf_cv(bayes, kf, data, target, nfolds, 'Naive bayes',ngram)
        accuracies.append(acc)
        #pickle.dump(bayes, open(write_path+"bayes"+str(numOfFeatures), 'wb'))

        #print_topFeatures(bayes,'bayes',numOfFeatures)

        clf = sklearn.svm.SVC(decision_function_shape='ovo',kernel='linear')
        plot_learning_curve(clf, "SVM", data, target, cv=kf, n_jobs=4)
        acc = train_model_kf_cv(clf, kf, data, target, nfolds, 'SVM linear',ngram)
        accuracies.append(acc)
        pickle.dump(clf, open("SVM_Model_" + str(ngram) + ".pickle", 'wb'))
        scoreTable[ngram] = accuracies
    return scoreTable
```

## Run classifiers with removing stopping words


```python
import pandas as pd

tweets_df = pd.read_pickle(r'DataFramePickle')
```


```python
scoreTable = Classify(tweets_df, True, 'word', [1, 3, 5, 7 ,9])
```


![png](output_74_0.png)



![png](output_74_1.png)



![png](output_74_2.png)



![png](output_74_3.png)



![png](output_74_4.png)



![png](output_74_5.png)



![png](output_74_6.png)



![png](output_74_7.png)



![png](output_74_8.png)



![png](output_74_9.png)



![png](output_74_10.png)



![png](output_74_11.png)



![png](output_74_12.png)



![png](output_74_13.png)



![png](output_74_14.png)



![png](output_74_15.png)



![png](output_74_16.png)



![png](output_74_17.png)



![png](output_74_18.png)



![png](output_74_19.png)



![png](output_74_20.png)



![png](output_74_21.png)



![png](output_74_22.png)



![png](output_74_23.png)



![png](output_74_24.png)



![png](output_74_25.png)



![png](output_74_26.png)



![png](output_74_27.png)



![png](output_74_28.png)



![png](output_74_29.png)


## Score Table


```python
scoreTable
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1</th>
      <th>3</th>
      <th>5</th>
      <th>7</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DT</th>
      <td>0.537135</td>
      <td>0.536668</td>
      <td>0.534148</td>
      <td>0.535549</td>
      <td>0.532470</td>
    </tr>
    <tr>
      <th>Naive bayes</th>
      <td>0.637898</td>
      <td>0.666355</td>
      <td>0.665889</td>
      <td>0.665609</td>
      <td>0.666262</td>
    </tr>
    <tr>
      <th>SVM linear</th>
      <td>0.626982</td>
      <td>0.684923</td>
      <td>0.693693</td>
      <td>0.697705</td>
      <td>0.699571</td>
    </tr>
  </tbody>
</table>
</div>



## Run classifiers without  to remove stopping words


```python
scoreTable = Classify(tweets_df, False, 'word', [5])
```


![png](output_78_0.png)



![png](output_78_1.png)



![png](output_78_2.png)



![png](output_78_3.png)



![png](output_78_4.png)



![png](output_78_5.png)


## Conclusions until now

1. SVM algorithm has given us the best results. 
2. We can get better results if we will continue to increase the dataset size.
3. The best result given when was we didn't restrict the maximum number of words represents as vector
4. When n-gram > 5 the results are not getting better. (let's continue with n-gram 5)
5. Without removing stopping words the results are not good(compare to with removing stopping words)

## 2.2 Try 2 - CNN using Keras package

## Algorithm Description

Keras - Keras is a high-level neural networks API, written in Python and capable of running on top of either TensorFlow, CNTK or Theano. It was developed with a focus on enabling fast experimentation. Being able to go from idea to result with the least possible delay is key to doing good research.

GloVe- GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

The inspiration of the strcture comes from and article of Aric Bartle, Jim Zheng - Gender Classification with Deep Learning. (https://cs224d.stanford.edu/reports/BartleAric.pdf)

The strcture we built is:

1. Sequence of 20 words
2. Embedding Layer of 20 vectors with GloVe dictionary 300 dimension.
3. Convolution Layer of 3 words with 300 dimension.
4. Max Pooling Layer in size of 3 with 3000 dimension.
5. Flattern Layer.
6. Dense to 2 classes (male, female).

The intuition behind is to let the machine find the combination of the three words that emphasise the difference between male tweet to female tweet.

## Function: Testing Theano working with GPU


```python
from theano import function, config, shared, sandbox
import theano.tensor as T
import numpy
import time
 
vlen = 10 * 30 * 768 # 10 x #cores x # threads per core
iters = 1000
 
rng = numpy.random.RandomState(22)
x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
f = function([], T.exp(x))
print(f.maker.fgraph.toposort())
t0 = time.time()
for i in range(iters):
 r = f()
t1 = time.time()
print("Looping %d times took %f seconds" % (iters, t1 - t0))
print("Result is %s" % (r,))
if numpy.any([isinstance(x.op, T.Elemwise) for x in f.maker.fgraph.toposort()]):
 print('Used the cpu')
else:
 print('Used the gpu')
```

    WARNING (theano.sandbox.cuda): The cuda backend is deprecated and will be removed in the next release (v0.10).  Please switch to the gpuarray backend. You can get more information about how to switch at this URL:
     https://github.com/Theano/Theano/wiki/Converting-to-the-new-gpu-back-end%28gpuarray%29
    
    Using gpu device 0: Quadro M1000M (CNMeM is disabled, cuDNN not available)
    

    [GpuElemwise{exp,no_inplace}(<CudaNdarrayType(float32, vector)>), HostFromGpu(GpuElemwise{exp,no_inplace}.0)]
    Looping 1000 times took 0.781334 seconds
    Result is [ 1.23178029  1.61879349  1.52278066 ...,  2.20771813  2.29967761
      1.62323296]
    Used the gpu
    

## Read dictionary of words (GloVe package)


```python
import os
import numpy
import numpy as np

GLOVE_DIR = '.\\Glove\\'

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.300d.txt'), encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
```

    Found 400000 word vectors.
    


```python
from keras.layers import Input, Embedding, LSTM, Dense, GRU, Activation, SimpleRNN
from keras.layers.wrappers import Bidirectional
from keras.models import Sequential
from keras.models import Model
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.core import Flatten
from keras.preprocessing import sequence
import numpy
from sklearn.cross_validation import KFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
from sklearn.metrics import roc_auc_score

EMBEDDING_DIM = 300

# fix random seed for reproducibility
numpy.random.seed(7)

def PlotLearningCurve(history):
    # list all data in history
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def TrainKeras(df):
    texts, labels = balanced_sample_maker(tweets_df['text'],tweets_df['gender'])
        
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))    

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)    

    labels_unique, labels_inverse = np.unique(labels, return_inverse=True)
    print('Found %s unique labels.' % len(labels_unique))
    
    labels_binary = keras.utils.to_categorical(labels_inverse)
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels_binary.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels_binary = labels_binary[indices]
    nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-nb_validation_samples]
    y_train = labels_binary[:-nb_validation_samples]
    x_test = data[-nb_validation_samples:]
    y_test = labels_binary[-nb_validation_samples:]

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():        
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(word_index) + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=False)

    main_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32', name='main_input')
    embedding_out = embedding_layer(main_input)    
    word_context_out1 = Conv1D(EMBEDDING_DIM, 3, activation='relu')(embedding_out)    
    max_pooling_out = MaxPooling1D(5)(word_context_out1)
    
    flat_out = Flatten()(max_pooling_out)    
    main_output = Dense(len(labels_unique), activation='softmax')(flat_out)
    model = Model(inputs=[main_input], outputs=[main_output])
    
    model.compile(loss='categorical_crossentropy',
          optimizer='rmsprop',
          metrics=['acc'])

    print(model.summary())
    
    # happy learning!
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test),
      epochs=10, batch_size=128)
    
    PlotLearningCurve(history)

    scores = model.evaluate(x_test, y_test, verbose=0)    

    print("Accuracy: %.2f%%" % (scores[1]*100))
    
            
```

# Count the number of unique words


```python
words = set()

for index in tweets_df.index:
    sequences = tweets_df.iloc[index]['text'].split()
    for word in sequences:
        words.add(word)

len(words)
```




    21593




```python
MAX_NB_WORDS = 5000
VALIDATION_SPLIT = 0.2
MAX_SEQUENCE_LENGTH = 20

TrainKeras(tweets_df)
```

    Found 18080 unique tokens.
    Found 2 unique labels.
    Shape of data tensor: (10718, 20)
    Shape of label tensor: (10718, 2)
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    main_input (InputLayer)      (None, 20)                0         
    _________________________________________________________________
    embedding_76 (Embedding)     (None, 20, 300)           5424300   
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 18, 300)           270300    
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 3, 300)            0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 900)               0         
    _________________________________________________________________
    dense_83 (Dense)             (None, 2)                 1802      
    =================================================================
    Total params: 5,696,402
    Trainable params: 272,102
    Non-trainable params: 5,424,300
    _________________________________________________________________
    None
    Train on 8575 samples, validate on 2143 samples
    Epoch 1/10
    8575/8575 [==============================] - 4s - loss: 0.6979 - acc: 0.5418 - val_loss: 0.6791 - val_acc: 0.5497
    Epoch 2/10
    8575/8575 [==============================] - 3s - loss: 0.6016 - acc: 0.6769 - val_loss: 0.6695 - val_acc: 0.5973
    Epoch 3/10
    8575/8575 [==============================] - 3s - loss: 0.5095 - acc: 0.7532 - val_loss: 0.6360 - val_acc: 0.6477
    Epoch 4/10
    8575/8575 [==============================] - 3s - loss: 0.4184 - acc: 0.8222 - val_loss: 0.6509 - val_acc: 0.6659
    Epoch 5/10
    8575/8575 [==============================] - 3s - loss: 0.3538 - acc: 0.8589 - val_loss: 0.7360 - val_acc: 0.6234
    Epoch 6/10
    8575/8575 [==============================] - 3s - loss: 0.3031 - acc: 0.8859 - val_loss: 0.6648 - val_acc: 0.6701
    Epoch 7/10
    8575/8575 [==============================] - 3s - loss: 0.2673 - acc: 0.9059 - val_loss: 0.6876 - val_acc: 0.6855
    Epoch 8/10
    8575/8575 [==============================] - 3s - loss: 0.2438 - acc: 0.9155 - val_loss: 0.7306 - val_acc: 0.6668
    Epoch 9/10
    8575/8575 [==============================] - 3s - loss: 0.2236 - acc: 0.9209 - val_loss: 0.7381 - val_acc: 0.6780
    Epoch 10/10
    8575/8575 [==============================] - 3s - loss: 0.2181 - acc: 0.9245 - val_loss: 0.7657 - val_acc: 0.6972
    dict_keys(['val_loss', 'val_acc', 'loss', 'acc'])
    


![png](output_90_1.png)



![png](output_90_2.png)


    Accuracy: 69.72%
    

## Conclusions until now

1. 5000 maximum number of words choosen because it gives us the best results.
2. The sequence length is very a dominant parameter, 20 gives the best result.
3. The learning is super fast compare to SVM classifier.
4. I recognize in deep learning a lot of potential, but it gives me the impression that it is more sensitive in manner of configuration and data, the conventional algorithm SVM seems to be more stable.

We choose to continue with n-gram SVM.

# Part 3 - Sequence Generation With Keras

## Word Level Generation Based On LSTM (Many To One Architecture)


```python
from keras.layers import Input, Embedding, LSTM, Dense, GRU, Activation, SimpleRNN, Dropout, RepeatVector, TimeDistributed
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Flatten
import numpy as np
from keras.optimizers import RMSprop
import random

def GetUniqueWords(words):
    words_set = set()
        
    for word in words:                
        words_set.add(word)

    return words_set

def TrainManyToOneGenerationModel(tweets_texts):        
    words = text_to_word_sequence(tweets_texts, lower=False, split=" ")
    unique_words = GetUniqueWords(words)
    number_of_words = len(unique_words)
    
    print("number of unique words:" + str(number_of_words))
    print("number of words:" + str(len(words)))
    
    words_indices = dict((c, i) for i, c in enumerate(unique_words))
    indices_words = dict((i, c) for i, c in enumerate(unique_words))
    idx = [words_indices[w] for w in words]
    
    #context of 3 words
    cs = 5
    
    #build the input
    c1_dat = [idx[i] for i in range(0, len(idx)-1-cs, 1)]
    c2_dat = [idx[i+1] for i in range(0, len(idx)-1-cs, 1)]
    c3_dat = [idx[i+2] for i in range(0, len(idx)-1-cs, 1)] 
    c4_dat = [idx[i+3] for i in range(0, len(idx)-1-cs, 1)] 
    c5_dat = [idx[i+4] for i in range(0, len(idx)-1-cs, 1)]    
    
    #convert to numpy array
    x1 = np.array(c1_dat)
    x2 = np.array(c2_dat)
    x3 = np.array(c3_dat)
    x4 = np.array(c4_dat)
    x5 = np.array(c5_dat)
    
    #Putting it all in one matrix:
    input_ = np.stack([x1,x2,x3,x4,x5],axis=1)
    
    #Putting it all in one matrix:
    #input_ = np.stack([x1,x2,x3,x4,x5],axis=1)
    
    output_ = input_[1:,0:1]
    input_ = input_[:-1,:]
    
    print(input_)
    
    print(input_.shape)
    
    print(output_)
        
    n_fac = 42
    n_hidden = 256
    
    model=Sequential([
        Embedding(number_of_words, n_fac, input_length=cs),
        LSTM(n_hidden, return_sequences=False, activation='relu'),        
        Dense(number_of_words, activation='softmax'),
    ])    
    
    print(model.summary())    
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop' ,metrics=["accuracy"])
    model.fit(input_, y=output_, batch_size=1200, epochs=20, verbose=1)
    return model, words_indices, indices_words, idx


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature    
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def GenerateNextWordManyToOne(text, model, words_indices, indices_words, idx, diversity):
    tmp = text_to_word_sequence(text, lower=False, split=" ")
    idxs = [words_indices[w] for w in tmp]
    arr = np.array(idxs)[np.newaxis,:]    
    pred = model.predict(arr)    
    return sample(pred[0], diversity)

def GenerateManyToOneSentences(model, words_indices, indices_words, idx, sample_length, diversity):
    generated_text = ""
    
    cs = 5
    max_len = sample_length    
    
    #random start vector
    newtweet_indices = words_indices['NEWTWEET']    
    start_index_arr = [i for i, j in enumerate(idx) if j == newtweet_indices]    
        
    start_index = random.randint(0, len(start_index_arr)-5)
        
    real_sentence = []
    for index in range(start_index_arr[start_index]- (cs-1), start_index_arr[start_index]-(cs-1) + max_len):
        real_sentence.append(indices_words[idx[index]])                
    
    generated_text = ""
    last_generated_index = None
    
    for i in range(0, max_len - cs):
        #get next 5 words
        next_cs_words = ""
        for i in range(i, i + cs):
            next_cs_words += real_sentence[i] + " "
        
        next_index = GenerateNextWordManyToOne(next_cs_words, model, words_indices, indices_words, idx, diversity)
        
        if (last_generated_index == None):
            generated_text += indices_words[next_index]
            
        if (last_generated_index != None and last_generated_index != next_index):
            generated_text += " " + indices_words[next_index]
        
        last_generated_index = next_index
    #seperate the tweets using the seperator token we've inserted
    generated_tweets = generated_text.split(' NEWTWEET ')
                
    return [tweet for tweet in generated_tweets if len(tweet.split()) > 3]

def GenerateTweets(model, words_indices, indices_words, idx, number_of_tweets):
    number_of_generated_tweets = 0
    output_generated_tweets = []
    
    sample_length = 40
    diversity = 0.3
    
    while (number_of_generated_tweets < number_of_tweets):
        generated_tweets = GenerateManyToOneSentences(model, words_indices, indices_words, idx, sample_length, diversity)
        output_generated_tweets += generated_tweets
        number_of_generated_tweets += len(generated_tweets)
    
    return output_generated_tweets
```

## Split Data By Gender


```python
import pandas as pd

tweets_df = pd.read_pickle(r'DataFramePickle')

def SplitDataByGender(df):
    male_tweets = df[df['gender']=='male']
    female_tweets = df[df['gender']=='female']
    return male_tweets, female_tweets

male_tweets, female_tweets = SplitDataByGender(tweets_df)
```

## 3.1 Build Sequence Generation Model for Male

### Preprocessing  - Join to one text


```python
#join tweets to one string of text and adding a seperator token
male_tweets_text = ' NEWTWEET '.join(male_tweets['text'])
```


```python
male_generation_model, male_words_indices, male_indices_words, male_idx = TrainManyToOneGenerationModel(male_tweets_text)
```

    number of unique words:15268
    number of words:101635
    [[14270 10778  4632  2094  9037]
     [10778  4632  2094  9037   561]
     [ 4632  2094  9037   561 11833]
     ..., 
     [ 9601     7 11456  8520  4696]
     [    7 11456  8520  4696  6093]
     [11456  8520  4696  6093  6512]]
    (101628, 5)
    [[10778]
     [ 4632]
     [ 2094]
     ..., 
     [    7]
     [11456]
     [ 8520]]
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_77 (Embedding)     (None, 5, 42)             641256    
    _________________________________________________________________
    lstm_24 (LSTM)               (None, 256)               306176    
    _________________________________________________________________
    dense_84 (Dense)             (None, 15268)             3923876   
    =================================================================
    Total params: 4,871,308
    Trainable params: 4,871,308
    Non-trainable params: 0
    _________________________________________________________________
    None
    Epoch 1/20
    101628/101628 [==============================] - 121s - loss: 7.4771 - acc: 0.0578         
    Epoch 2/20
    101628/101628 [==============================] - 117s - loss: 7.1346 - acc: 0.0642     
    Epoch 3/20
    101628/101628 [==============================] - 120s - loss: 7.0873 - acc: 0.0642     
    Epoch 4/20
    101628/101628 [==============================] - 120s - loss: 7.0536 - acc: 0.0642     
    Epoch 5/20
    101628/101628 [==============================] - 119s - loss: 7.0287 - acc: 0.0642     
    Epoch 6/20
    101628/101628 [==============================] - 118s - loss: 7.0082 - acc: 0.0642     
    Epoch 7/20
    101628/101628 [==============================] - 117s - loss: 6.9836 - acc: 0.0666     
    Epoch 8/20
    101628/101628 [==============================] - 118s - loss: 6.8679 - acc: 0.1059     
    Epoch 9/20
    101628/101628 [==============================] - 117s - loss: 6.5566 - acc: 0.1099     
    Epoch 10/20
    101628/101628 [==============================] - 119s - loss: 6.1734 - acc: 0.1256     
    Epoch 11/20
    101628/101628 [==============================] - 118s - loss: 5.8428 - acc: 0.1474     
    Epoch 12/20
    101628/101628 [==============================] - 118s - loss: 5.5914 - acc: 0.1829     
    Epoch 13/20
    101628/101628 [==============================] - 118s - loss: 5.3591 - acc: 0.2162     
    Epoch 14/20
    101628/101628 [==============================] - 120s - loss: 5.1255 - acc: 0.2531     
    Epoch 15/20
    101628/101628 [==============================] - 120s - loss: 4.9214 - acc: 0.2868     
    Epoch 16/20
    101628/101628 [==============================] - 119s - loss: 4.7320 - acc: 0.3186     
    Epoch 17/20
    101628/101628 [==============================] - 119s - loss: 4.5471 - acc: 0.3504     
    Epoch 18/20
    101628/101628 [==============================] - 125s - loss: 4.3528 - acc: 0.3867     
    Epoch 19/20
    101628/101628 [==============================] - 127s - loss: 4.1915 - acc: 0.4145     
    Epoch 20/20
    101628/101628 [==============================] - 130s - loss: 4.0184 - acc: 0.4469     
    

### Save Model


```python
male_generation_model.save_weights("manyToOneMaleGenerationModel.h5")
```

### Generate 30% new tweets for Male


```python
generated_male_tweets = GenerateTweets(male_generation_model, male_words_indices, male_indices_words, male_idx,len(male_tweets['text'])*0.3)
```

## let's look at the first 10 tweets


```python
print(generated_male_tweets[30:40])
```

    ['but she take far love a most support of support the better there', 'taking tell right but i really tell next definitely', 'for travel be me trump where he', 'and great 2 morning by', 'wonder went the big isnt ass', '1 same agree msm back russia man black piece long make tweet pride', 'new congress still have enables 1 the for were the stories 6 to shooting justice with paid', 'i got isnt wait is while rights game that these by', 'i dont youre dont have several with followers that seen', 'gun youre still shit a their every being']
    

## 3.2 Build Sequence Generation Model for Female

### Preprocessing  - Join to one text


```python
#join tweets to one tweet
female_tweets_text = ' NEWTWEET '.join(female_tweets['text'])
```


```python
female_generation_model, female_words_indices, female_indices_words, female_idx = TrainManyToOneGenerationModel(female_tweets_text)
```

    number of unique words:12758
    number of words:82985
    [[ 1495  1523  3064  4446  1134]
     [ 1523  3064  4446  1134  5111]
     [ 3064  4446  1134  5111  4679]
     ..., 
     [ 2797  3743  4430  5493  3583]
     [ 3743  4430  5493  3583 10148]
     [ 4430  5493  3583 10148  3310]]
    (82978, 5)
    [[1523]
     [3064]
     [4446]
     ..., 
     [3743]
     [4430]
     [5493]]
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_78 (Embedding)     (None, 5, 42)             535836    
    _________________________________________________________________
    lstm_25 (LSTM)               (None, 256)               306176    
    _________________________________________________________________
    dense_85 (Dense)             (None, 12758)             3278806   
    =================================================================
    Total params: 4,120,818
    Trainable params: 4,120,818
    Non-trainable params: 0
    _________________________________________________________________
    None
    Epoch 1/20
    82978/82978 [==============================] - 94s - loss: 7.4153 - acc: 0.0572          
    Epoch 2/20
    82978/82978 [==============================] - 91s - loss: 7.0450 - acc: 0.0646     
    Epoch 3/20
    82978/82978 [==============================] - 91s - loss: 7.0057 - acc: 0.0646     
    Epoch 4/20
    82978/82978 [==============================] - 95s - loss: 6.9749 - acc: 0.0646     
    Epoch 5/20
    82978/82978 [==============================] - 94s - loss: 6.9551 - acc: 0.0646     
    Epoch 6/20
    82978/82978 [==============================] - 95s - loss: 6.9306 - acc: 0.0646     
    Epoch 7/20
    82978/82978 [==============================] - 93s - loss: 6.9173 - acc: 0.0646     
    Epoch 8/20
    82978/82978 [==============================] - 93s - loss: 6.8905 - acc: 0.0646     
    Epoch 9/20
    82978/82978 [==============================] - 94s - loss: 6.8235 - acc: 0.0652     
    Epoch 10/20
    82978/82978 [==============================] - 94s - loss: 6.6514 - acc: 0.0901     
    Epoch 11/20
    82978/82978 [==============================] - 92s - loss: 6.3880 - acc: 0.1042     
    Epoch 12/20
    82978/82978 [==============================] - 94s - loss: 6.1471 - acc: 0.1111     
    Epoch 13/20
    82978/82978 [==============================] - 94s - loss: 5.9263 - acc: 0.1311     
    Epoch 14/20
    82978/82978 [==============================] - 84s - loss: 5.7040 - acc: 0.1653     
    Epoch 15/20
    82978/82978 [==============================] - 83s - loss: 5.4746 - acc: 0.1961     
    Epoch 16/20
    82978/82978 [==============================] - 83s - loss: 5.2800 - acc: 0.2193     
    Epoch 17/20
    82978/82978 [==============================] - 83s - loss: 5.0961 - acc: 0.2480     
    Epoch 18/20
    82978/82978 [==============================] - 83s - loss: 4.9271 - acc: 0.2709     
    Epoch 19/20
    82978/82978 [==============================] - 90s - loss: 4.7437 - acc: 0.2979     
    Epoch 20/20
    82978/82978 [==============================] - 87s - loss: 4.5816 - acc: 0.3257     
    

### Save Model


```python
female_generation_model.save_weights("manyToOneFemaleGenerationModel.h5")
```

### Generate 30% new tweets for Female


```python
generated_female_tweets = GenerateTweets(female_generation_model, female_words_indices, female_indices_words, female_idx,len(female_tweets['text'])*0.3)
```

## Let's look at 10 tweets


```python
print(generated_female_tweets[30:40])
```

    ['i just please wont say u', 'up feel its is only im', 'never for the best ago', 'my explains care us with these check too and were', 'the great call them to hes is told out on the story u them to after and almost to had', 'even how i she', 'who an two york and', 'dead this be win the help who he really to can you easy ever shit here', 'a new cuba for week', 'were be my a from as a an']
    

## Build Generated Tweets Dataframe


```python
generated_df_male = pd.DataFrame({'text': generated_male_tweets, 'gender': 'male'})
generated_df_female = pd.DataFrame({'text': generated_female_tweets, 'gender': 'female'})
generated_df = pd.concat([generated_df_male, generated_df_female], join="inner").reset_index()
generated_df.describe(include='all')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>gender</th>
      <th>text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3569.000000</td>
      <td>3569</td>
      <td>3569</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>2</td>
      <td>3569</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>male</td>
      <td>on an sessions that the travel man htt</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>1960</td>
      <td>1</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>900.379938</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>529.881361</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>446.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>892.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1338.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1959.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
generated_df['gender'] = generated_df['gender'].astype('category')
generated_df.gender.value_counts()
```




    male      1960
    female    1609
    Name: gender, dtype: int64



### Load SVM Model With Pickle And Predict


```python
def encode_data_target(texts, gender, vectorizer):
    corpus = np.array(texts.values).tolist()
    target = np.array(gender.values)
    
    data = vectorizer.transform(corpus)
    
    return data, target
    

def PredictGender(df):
    model_name = 'SVM 5-gram'
    ngram = 5
    
    model = pickle.load(open( "SVM_Model_" + str(ngram)+ ".pickle", "rb"))
    vectorizer = pickle.load(open( "Vectorizer_" + str(ngram)+ ".pickle", "rb"))
    
    dfReduced = df.copy(deep=True)
    
    texts, labels = balanced_sample_maker(dfReduced['text'],dfReduced['gender']) 
    data, target = encode_data_target(texts, labels, vectorizer) 
    
    print(labels.value_counts())
    
    cm = []
    error = []
    
    predictions = model.predict(data)
        
    classes = model.classes_                
    cm.append(confusion_matrix(target, predictions, labels=classes))
        
    error.append(model.score(data, target))
    accuracy = np.mean(error)
    for i in range(0,9):
        cms = np.mean(cm, axis=0)
    plot_cnf_matrix(cms, classes, model_name ,ngram)
    return accuracy

print("Accuracy:" + str(PredictGender(generated_df)))
```

    male      1609
    female    1609
    Name: gender, dtype: int64
    


![png](output_121_1.png)


    Accuracy:0.682411435674
    

## Conclusion
#Classification (Part 1 & 2)

1. The best accuracy we got was 69.7% which is not that far from the state of the art considering the relatively small dataset we've used. This article http://dl.acm.org/citation.cfm?id=2145568 reaches accuracy of 76% percent for classification based on tweets text alone with a corpus of ~4,000,000 tweets.
2. We can see that the more tweets we use to train the model, the more accurate it gets. We assume that if our dataset was bigger and contained millions of tweets we would have reached higher accuracy. 
3. The best results in terms of accuracy were recieved using CNN.

#Text generation (Part 3 & 4)

1. The generated tweets were classified correctly in an accuracy rate of 68%. Compared to the accuracy rate of the original tweets, The difference is not significant(~0.02) in terms of classification.
2. From the confusion matrix we can see that the male texts are more difficult to classify. With female texts we got ~0.86 accuracy (~1396 from ~1609).
2. The generated tweets are not perfect, but some of them are quite good.
