import nltk, nltk.classify.util, nltk.metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import svm


def get_data():
    dataset = pd.read_csv("preprocessed_dataset.csv")
    return dataset


def split(data):
    data = data.sample(frac=1).reset_index(drop=True)
    l=int(len(data)*0.65)
    return data[:l],data[l:]


def model_1(dataset_train,dataset_test):
    X_train,X_test,y_train,y_test=dataset_train['Comments'],dataset_test['Comments'],dataset_train['class'],dataset_test['class']
    X=pd.concat([X_train,X_test],axis=0)
    vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df=0.7)
    X = vectorizer.fit_transform(X).toarray()
    tfidfconverter = TfidfTransformer()
    X = tfidfconverter.fit_transform(X).toarray()
    X_train = X[:len(X_train)]
    X_test = X[len(X_train):]
    classifier1 = svm.SVC(kernel='linear')  # Linear Kernel
    classifier1.fit(X_train, y_train)
    y_pred = classifier1.predict(X_test)
    """
    print(y_pred)
    print("************classification report for all callsifiers used**********")
    print("**************** SVM ************************")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))"""
    return y_pred,y_test


def model_2(dataset_train,dataset_test):

    def word_feats(words):
        return dict([(word, True) for word in words])


    def classification(negfeats, posfeats, pospercent, negpercent):
        negcutoff = int(len(negfeats) * negpercent)
        poscutoff = int(len(posfeats) * pospercent)
        trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
        algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
        classifier = nltk.MaxentClassifier.train(trainfeats, algorithm, max_iter=12)
        return classifier


    negids = dataset_train['Comments'][dataset_train['class'] == 0]
    posids = dataset_train['Comments'][dataset_train['class'] == 1]
    negfeats = [(word_feats(str(f).split(" ")), 0) for f in negids]
    posfeats = [(word_feats(str(f).split(" ")), 1) for f in posids]
    classifier = classification(negfeats, posfeats, 1, 1)
    X_test = dataset_test['Comments']
    y_test = dataset_test['class']
    y_pred = []
    reviews = X_test
    for to_review in reviews:
        to_review_words = to_review.split(" ")
        wordfeats = word_feats(to_review_words)
        probs = classifier.prob_classify(wordfeats)
        y_pred.append(probs.max())
    """
    print(y_pred)
    print("************classification report for all callsifiers used**********")
    print("**************** Maximum Entropy ************************")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))"""
    return y_pred,y_test

def model_3(dataset_train,dataset_test):

    vectorizer = CountVectorizer(max_features=1500, min_df=10, max_df=0.7)
    X_pos = vectorizer.fit_transform(dataset_train['Comments'][dataset_train['Rating'] > 7])
    x_pos = X_pos.toarray()
    pos = np.sum(x_pos, axis=0)
    pos = np.sum(pos)
    freq_pos = {}
    for i in range(len(vectorizer.get_feature_names())):
        freq_pos[vectorizer.get_feature_names()[i]] = x_pos[:, i].sum(axis=0)

    X_neg = vectorizer.fit_transform(dataset_train['Comments'][dataset_train['Rating'] <= 7])
    x_neg = X_neg.toarray()
    neg = np.sum(x_neg, axis=0)
    neg = np.sum(neg)
    freq_neg = {}
    for i in range(len(vectorizer.get_feature_names())):
        freq_neg[vectorizer.get_feature_names()[i]] = x_neg[:, i].sum(axis=0)

    X_test = dataset_test['Comments']
    y_test = dataset_test['class']
    y_pred = []

    for i in range(len(X_test)):
        text = " ".join(X_test.iloc[i].split())
        score = 0
        for word in text.split(" "):
            if word in list(freq_pos.keys()):
                if word in list(freq_neg.keys()):
                    p = freq_pos[word]
                    n = freq_neg[word]
                else:
                    p = freq_pos[word]
                    n = 0
            elif word in list(freq_neg.keys()):
                p = 0
                n = freq_neg[word]
            else:
                p = 0
                n = 0
            score_word = np.log(((p + 1) / pos) * (neg / (n + 1)))
            score = score + score_word
        if score > 0:
            y_pred.append(1)
        else:
            y_pred.append(0)
    """
    print(y_pred)
    print("************classification report for all callsifiers used**********")
    print("**************** Scoring ************************")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    print(accuracy_score(y_test, y_pred))"""
    return y_pred,y_test



def naive(y_1,y_2,y_3):
    y=[]
    for i in range(len(y_1)):
        votes = [0, 0]
        if(y_1[i]==0):
            votes[0]+=1
        else:
            votes[1]+=1
        if (y_2[i] == 0):
            votes[0] += 1
        else:
            votes[1] += 1
        if (y_3[i] == 0):
            votes[0] += 1
        else:
            votes[1] += 1
        if(votes[0]>votes[1]):
            y.append(0)
        else:
            y.append(1)
    return y
def main():
    dataset=get_data()
    dataset_train,dataset_test=split(dataset)
    y_pred_1,y_test_1=model_1(dataset_train,dataset_test)
    y_pred_2,y_test_2=model_2(dataset_train, dataset_test)
    y_pred_3, y_test_3=model_3(dataset_train,dataset_test)
    y=naive(y_pred_1,y_pred_2,y_pred_3)
    print("**************** Naive Voting ************************")
    print(confusion_matrix(y_test_1, y))
    print(classification_report(y_test_1, y))
    print(accuracy_score(y_test_1, y))


main()
