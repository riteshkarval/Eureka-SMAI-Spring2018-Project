#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
import numpy as np


# In[2]:


with open('../dataset/reviews.pkl', 'rb') as handle:
    data = pickle.load(handle)


# In[3]:


titles = list(data.keys())


# In[4]:


reviews = []
ratings = []
vocab = []
lengths = []
for each_title in titles:
    r_pair = data[each_title]
    for each_pair in r_pair:
        reviews.append(each_pair[0])
        if int(each_pair[1]) > 6:
            ratings.append(1)
        else:
            ratings.append(0)
        vocab += each_pair[0].split()
        lengths.append(len(each_pair[0].split()))


# In[5]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[6]:


vectorizer = TfidfVectorizer(min_df = 5,
                             max_df = 0.8,
                             sublinear_tf = True,
                             use_idf = True)


# In[7]:


train_vectors = vectorizer.fit_transform(reviews)


# In[8]:


print(train_vectors.shape)


# In[9]:


import time
from sklearn import svm
from sklearn.metrics import classification_report


# In[10]:


clf = svm.SVC(gamma='scale')


# In[11]:


clf.fit(train_vectors[:1000],ratings[:1000])


# In[14]:


print(clf.predict(train_vectors[1000:1100]))

