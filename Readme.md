
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
        
        gender = GetWellDefineGenderFromName(user['name'].encode('utf-8').split()[0])
        
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

stream.filter(locations=[-74,40,-73,41], track=['The', 'I', 'she', 'and'])
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-4-78819c66e5ab> in <module>()
          8 import json
          9 
    ---> 10 auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
         11 auth.set_access_token(access_token, access_token_secret)
         12 
    

    NameError: name 'consumer_key' is not defined


## Data Cleaning - Define fucntions

### Decoding text to Ascii

Most of the NL Algorithms works with ASCII.


```python
def UTFToAscii(string):
    return string.decode('ascii', 'ignore')
```

### Extract URLs, @user_reference, hashtags Count


```python
def ExtractReference(string):
    reference_count = 0
    hashtag_count = 0
    urls_count = 0
    for i in string.split():
        s, n, p, pa, q, f = urlparse.urlparse(i)
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
import urlparse

def RemoveNonWords(string):
    return re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", "", string)
```

### Handle Escaping Characters


```python
import HTMLParser

def RemoveEscaping(string):
    html_parser = HTMLParser.HTMLParser()
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
for line in open('tweets.data', 'r'):    
    tweet_data = json.loads(line)
    tweet_fullname = tweet_data['user']['name'].encode('utf-8')
    if IsEnglish(tweet_fullname) == False:
        continue

    if len(tweet_fullname.split()) < 2:
        continue

    tweet_text = tweet_data['text'].encode('utf-8')
    tweet_text = UTFToAscii(tweet_text)
    tweet_text = RemoveEscaping(tweet_text)
    tweet_text = FlatLines(tweet_text)
    reference_count, hashtag_count, urls_count = ExtractReference(tweet_text)
    tweet_text = RemoveNonWords(tweet_text)    
    tweet_text = RemoveRT(tweet_text)
    tweet_text = ToLowercase(tweet_text)
    tweet_text = ReplaceApostrophe(tweet_text)

    #filter all text that smaller then 2 words
    if len(tweet_text.split()) < 3:
        continue
    
    #filter all names with no legal capital letter
    if  len(re.findall(r'[A-Z]',tweet_fullname)) != len(tweet_fullname.split()):            
        continue
    
    gender = GetWellDefineGenderFromName(tweet_fullname.split()[0])
    
    if gender is None:
        continue

    tweets_df.loc[len(tweets_df)]=[tweet_data['id_str'], tweet_text, reference_count, hashtag_count, urls_count, tweet_fullname, gender]

```

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
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>Guy Santeramo</td>
      <td>male</td>
    </tr>
    <tr>
      <th>1</th>
      <td>875088067471831043</td>
      <td>that was lovely thank you for sharing i too lo...</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Alexa Harrison</td>
      <td>female</td>
    </tr>
    <tr>
      <th>2</th>
      <td>875088072786022400</td>
      <td>sling could a team made entirely of players wh...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Jay Zampi</td>
      <td>male</td>
    </tr>
    <tr>
      <th>3</th>
      <td>875088110736089089</td>
      <td>my fan theory is that your fan theory has noth...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Susana Polo</td>
      <td>female</td>
    </tr>
    <tr>
      <th>4</th>
      <td>875088135889334273</td>
      <td>if you say so princess im sure hell love to se...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Brad Gibson</td>
      <td>male</td>
    </tr>
    <tr>
      <th>5</th>
      <td>875088148770033665</td>
      <td>when she asks you what you bring to the table</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Solomon Grundy</td>
      <td>male</td>
    </tr>
    <tr>
      <th>6</th>
      <td>875088173939818497</td>
      <td>finally biting the bullet and reserving hotel ...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>Meg Roy</td>
      <td>female</td>
    </tr>
    <tr>
      <th>7</th>
      <td>875088203258109952</td>
      <td>im speaking at digipub summit in new york this...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>Adam Smith</td>
      <td>male</td>
    </tr>
    <tr>
      <th>8</th>
      <td>875088215715217410</td>
      <td>this couldnt be easier i just called my rep to...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>Krystyna Hutchinson</td>
      <td>female</td>
    </tr>
    <tr>
      <th>9</th>
      <td>875088231964041218</td>
      <td>the gun contagion in america continues to thre...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>Michael Corley</td>
      <td>male</td>
    </tr>
  </tbody>
</table>
</div>



The texts are look good and also the names the and the gender.


```python
tweets_df.describe(include='all')
```




<div>
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
      <td>4133</td>
      <td>4133</td>
      <td>4133.000000</td>
      <td>4133.000000</td>
      <td>4133.000000</td>
      <td>4133</td>
      <td>4133</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>4133</td>
      <td>3986</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3361</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>875432697602113537</td>
      <td>i just want a country where affording health c...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Joe Dicandia</td>
      <td>male</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>1</td>
      <td>8</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>14</td>
      <td>2490</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.032664</td>
      <td>0.345996</td>
      <td>0.533995</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.987407</td>
      <td>0.966179</td>
      <td>0.571270</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.000000</td>
      <td>11.000000</td>
      <td>3.000000</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
tweets_df['gender'] = tweets_df['gender'].astype('category')
tweets_df.gender.value_counts()
```




    male      2490
    female    1643
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
    reference_count     float64
    hashtag_count       float64
    urls_count          float64
    name                 object
    gender             category
    dtype: object



Should not be missing values:


```python
tweets_df.isnull().sum()
```




    id                 0
    text               0
    reference_count    0
    hashtag_count      0
    urls_count         0
    name               0
    gender             0
    dtype: int64



### Male ,Female Ratio


```python
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

%matplotlib inline

plt.figure();

tweets_df.gender.value_counts().plot(kind='pie', colors=['blue','red'])
```




    <matplotlib.axes._subplots.AxesSubplot at 0xeeb4550>




![png](output_41_1.png)


## Tweet Reference, Hashtag, URL Count By Gender


Let's explore the information which we extract from the original tweet text.



```python
var = pd.crosstab(tweets_df['reference_count'], tweets_df['gender'])
var.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xe11a9e8>




![png](output_44_1.png)



```python
var = pd.crosstab(tweets_df['hashtag_count'], tweets_df['gender'])
var.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xe028898>




![png](output_45_1.png)



```python
var = pd.crosstab(tweets_df['urls_count'], tweets_df['gender'])
var.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)
```




    <matplotlib.axes._subplots.AxesSubplot at 0xe446940>




![png](output_46_1.png)


We can see from above that these features not give us an additional information. (the male/female ratio remains almost the same for each value).

## Backup Dataframe to Pickle file


```python
tweets_df.to_pickle('DataFramePickle')
```

# Part 2 - Part 2 - Train Classifier For Gender

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

## This function returns the top features for every model


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

## This function returns the train and test sets


```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_test_set(df, numOfFeatures, RemoveStopWords, f, t, analyzer):

    corpus = np.array(df.text.values).tolist()
    target = np.array(df['gender'].values)
    #tf-idf vectorizer is set to count word occurances and normalize. NOT TF-IDF
    if RemoveStopWords:
        vectorizer = TfidfVectorizer(stop_words='english',analyzer=analyzer,use_idf=False, ngram_range=(f,t),binary=False)
    else: #only SW
        vectorizer = TfidfVectorizer(max_features=numOfFeatures,analyzer=analyzer, use_idf=False, ngram_range=(f,t),binary=False,norm='l1')
    
    X = vectorizer.fit_transform(corpus)
    global feature_names
    feature_names = vectorizer.get_feature_names()

    return X, target, feature_names#
```

## This function plots a confusion matrix


```python
def plot_cnf_matrix(cms, classes, model_name, numOfFeatures):
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(cms), cmap=plt.cm.jet, interpolation='nearest')
    
    width, height = cms.shape
    
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
```

## This function helps us to classify our data 


```python
from sklearn.cross_validation import KFold
from sklearn.tree import DecisionTreeClassifier
import pickle
import sklearn.naive_bayes as nb
import sklearn

def Classify(df,RemoveStopWords,analyzer,n):
    a = {'Classifer': pd.Series(index=['DT', 'Naive bayes', 'SVM linear' ]),}
    scoreTable = pd.DataFrame(a)

    scoreTable.__delitem__('Classifer')

    tests = ['allLangs'] 
    nums = [100,200,400,600,900]

    for test in tests:
        dfReduced = df.copy(deep=True)

        if test is not "allLangs":
            langs = str.split(test, '_')
            dfReduced = dfReduced[dfReduced['gender'].isin(langs)]
        
        #if we want to check for several x top featurs
        for numOfFeatures in nums:
            nfolds=5
            accuracies = []
            data, target, feat_names = extract_test_set(dfReduced,numOfFeatures,RemoveStopWords, n,n, analyzer)
            kf = KFold(data.shape[0], n_folds = nfolds, shuffle = True, random_state = 1)
            
            test = test+" top "+str(numOfFeatures)+" features"
            
            dtree = DecisionTreeClassifier(random_state=0, max_depth=50)
            acc = train_model_kf_cv(dtree, kf, data, target, nfolds, 'DT',numOfFeatures)
            accuracies.append(acc)
            pickle.dump(dtree, open(write_path+"dtree"+str(numOfFeatures), 'wb'))

            bayes = nb.MultinomialNB()
            acc = train_model_kf_cv(bayes, kf, data, target, nfolds, 'Naive bayes',numOfFeatures)
            accuracies.append(acc)
            pickle.dump(bayes, open(write_path+"bayes"+str(numOfFeatures), 'wb'))

            print_topFeatures(bayes,'bayes',numOfFeatures)

            clf = sklearn.svm.SVC(decision_function_shape='ovo',kernel='linear')
            acc = train_model_kf_cv(clf, kf, data, target, nfolds, 'SVM linear',numOfFeatures)
            accuracies.append(acc)
            pickle.dump(clf, open(write_path+"svm"+str(numOfFeatures), 'wb'))
            scoreTable[numOfFeatures] = accuracies
    return scoreTable
    #writer = pd.ExcelWriter(write_path  + '_Results.xlsx')
    #scoreTable.to_excel(writer,'Score table')
    #writer.save()
```

## This function trains the model k-fold times and returns total accuracy


```python
from sklearn.metrics import confusion_matrix

def train_model_kf_cv(model, kf, data, target, numFolds, model_name,numFeatures):
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
    plot_cnf_matrix(cms, classes, model_name,numFeatures)
    ###auc = total / numFolds
    ###print "AUC of {0}: {1}".format(Model.__name__, accuracy)
    return accuracy
```

## Read DataFrame and run Classifier


```python
tweets_df = pd.read_pickle(r'DataFramePickle')

scoreTable = Classify(tweets_df, False, 'word', 1)
```


![png](output_64_0.png)



![png](output_64_1.png)



![png](output_64_2.png)



![png](output_64_3.png)



![png](output_64_4.png)



![png](output_64_5.png)



![png](output_64_6.png)



![png](output_64_7.png)



![png](output_64_8.png)



![png](output_64_9.png)



![png](output_64_10.png)



![png](output_64_11.png)



![png](output_64_12.png)



![png](output_64_13.png)



![png](output_64_14.png)


## Score Table


```python
scoreTable
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>100</th>
      <th>200</th>
      <th>400</th>
      <th>600</th>
      <th>900</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>DT</th>
      <td>0.531329</td>
      <td>0.553107</td>
      <td>0.561325</td>
      <td>0.553827</td>
      <td>0.557702</td>
    </tr>
    <tr>
      <th>Naive bayes</th>
      <td>0.602465</td>
      <td>0.602223</td>
      <td>0.601739</td>
      <td>0.602707</td>
      <td>0.602948</td>
    </tr>
    <tr>
      <th>SVM linear</th>
      <td>0.602465</td>
      <td>0.602223</td>
      <td>0.602223</td>
      <td>0.602464</td>
      <td>0.602464</td>
    </tr>
  </tbody>
</table>
</div>




```python

```
