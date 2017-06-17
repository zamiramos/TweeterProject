
# coding: utf-8

# In[1]:
from nltk.corpus import names
import gender_guesser.detector as genderDetect
import csv
import re
import urlparse
import HTMLParser
import pandas as pd
import numpy as np
import sklearn
import sklearn.naive_bayes as nb
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

import json
import pickle
import os
import nltk
from nltk.corpus import stopwords
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import xlsxwriter
import matplotlib.pyplot as plt


# paths to use

# In[2]:

write_path = r'D:\BG\H\Data Scientist\Project\DATA\Results\Test\\'
read_directory = r'C:\Users\Yohay_Nahari\Desktop\Google Drive\קוובנגה\פרויקט גמר\פוסטר\קוד סופי\concat\concat10\\'


# This function returns the top features for every model

# In[3]:

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


# This function returns the train and test sets

# In[4]:

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


# This function plots a confusion matrix to a .png file

# In[7]:

def plot_cnf_matrix(cms, classes, model_name,numOfFeatures):
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
    plt.savefig(write_path + '_confusion_matrix_'+str(numOfFeatures)+'_' + model_name +'.png', format='png')


# This function helps us to classify our data 

# In[14]:

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
            
    writer = pd.ExcelWriter(write_path  + '_Results.xlsx')
    scoreTable.to_excel(writer,'Score table')
    writer.save()


# This function trains the model k-fold times and returns total accuracy

# In[13]:

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

# main

tweets_df=pd.read_pickle(r'DataFramePickle')

Classify(tweets_df,False, 'word', 1)

