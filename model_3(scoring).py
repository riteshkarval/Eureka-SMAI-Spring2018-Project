import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def split(data):
    data = data.sample(frac=1).reset_index(drop=True)
    l=int(len(data)*0.75)
    return data[:l],data[l:]

dataset=pd.read_csv("preprocessed_dataset.csv")
dataset_train,dataset_test=split(dataset)


vectorizer = CountVectorizer(max_features=1500, min_df=10, max_df=0.7)


X_pos = vectorizer.fit_transform(dataset_train['Comments'][dataset_train['Rating']>7])
x_pos=X_pos.toarray()
pos=np.sum(x_pos, axis=0)
pos=np.sum(pos)
freq_pos={}
for i in range(len(vectorizer.get_feature_names())):
     freq_pos[vectorizer.get_feature_names()[i]]=x_pos[:,i].sum(axis=0)


X_neg = vectorizer.fit_transform(dataset_train['Comments'][dataset_train['Rating']<=7])
x_neg=X_neg.toarray()
neg=np.sum(x_neg,axis=0)
neg=np.sum(neg)
freq_neg={}
for i in range(len(vectorizer.get_feature_names())):
     freq_neg[vectorizer.get_feature_names()[i]]=x_neg[:,i].sum(axis=0)


X_test=dataset_test['Comments']
y_test=dataset_test['class']
y_pred=[]


for i in range(len(X_test)):
    text=" ".join(X_test.iloc[i].split())
    score=0
    for word in text.split(" "):
        if word in list(freq_pos.keys()):
            if word in list(freq_neg.keys()):
                p=freq_pos[word]
                n=freq_neg[word]
            else:
                p = freq_pos[word]
                n = 0
        elif word in list(freq_neg.keys()):
            p=0
            n=freq_neg[word]
        else:
            p=0
            n=0
        score_word=np.log(((p+1)/pos)*(neg/(n+1)))
        score=score+score_word
    if score>0:
        y_pred.append(1)
    else:
        y_pred.append(0)
print(y_pred)
print(np.array(y_test))



print("************classification report for all callsifiers used**********")
print("**************** Scoring ************************")
print(np.unique(y_test))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))

