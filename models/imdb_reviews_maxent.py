#!/usr/bin/env python
# coding: utf-8

# In[34]:


from nltk.classify import MaxentClassifier
import pickle


# In[35]:


def word_feats(words):
 return dict([(word, True) for word in words.split()])


# In[36]:


with open('reviews.pkl', 'rb') as handle:
    data = pickle.load(handle)


# In[37]:


titles = list(data.keys())


# In[38]:


reviews = {'pos':[],
           'neg':[]}
for each_title in titles:
    r_pair = data[each_title]
    for each_pair in r_pair:
        if int(each_pair[1]) > 7:
            reviews['pos'].append(each_pair[0])
        else:
            reviews['neg'].append(each_pair[0])


# In[39]:


len(reviews['pos'])


# In[40]:


negfeats = [(word_feats(review), 'neg') for review in reviews['neg']]
posfeats = [(word_feats(review), 'pos') for review in reviews['pos']]


# In[41]:


negcutoff = int(len(negfeats)*3/4)
poscutoff = int(len(posfeats)*3/4)


# In[42]:


trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]


# In[43]:


classifier = MaxentClassifier.train(trainfeats, algorithm='gis', max_iter=10, min_lldelta=0.5)


# In[44]:


print(classifier.show_most_informative_features(10))


# In[45]:


print(classifier.labels())


# In[46]:


from nltk.classify.util import accuracy


# In[47]:


print(accuracy(classifier, testfeats))


# In[48]:


probs = classifier.prob_classify(trainfeats[0][0])
print(probs.max())

