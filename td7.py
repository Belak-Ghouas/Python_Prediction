#!/usr/bin/env python
#encoding=utf8

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import pairwise
from collections import defaultdict 
import csv
import random
import math
import numpy as np
from sklearn.model_selection import KFold

from sklearn.svm import SVC


def split_lines(input, seed, output1, output2):
    """Distributes the lines of 'input' to 'output1' and 'output2' pseudo-randomly.

  Args:
    input: a string, the name of the input file.
    seed: an integer, the seed of the pseudo-random generator used. The split
        should be different with different seeds. Conversely, using the same
        seed and the same input should yield exactly the same outputs.
    output1: a string, the name of the first output file.
    output2: a string, the name of the second output file.
  """
    random.seed(seed)
    out1 = open(output1, 'w')
    out2 = open(output2, 'w')
    for line in open(input, 'r').readlines():
        if random.randint(0, 1):
            out1.write(line)
        else:
            out2.write(line)



def read_data(filename):
    X=[]
    Y=[]
    maliste=[]
   
    for line in open(filename, 'r').readlines():
        tmp=[]
        maliste=line.split(',')
        for s in maliste[2:]:
            tmp.append(float(s))
        if (maliste[1]=='M'):
            Y.append(True)
        else:
            Y.append(False)
        X.append(tmp)
    return (X,Y)
    
def simple_distance(data1, data2):
    distance=0
    for i in range(len(data1)):
         distance+=(data1[i]-data2[i])**2
    return math.sqrt(distance)

def k_nearest_neighbors(x, points, dist_function, k):
    resultdist=[]
    resultpos=[]
    for i in range (len(points)):
        resultdist.append((dist_function(x,points[i]),i))
    resultdist.sort(key=lambda x: x[0])
    for j in range(k):
        resultpos.append((resultdist[j])[1])

    return resultpos


def is_cancerous_knn(x, train_x, train_y, dist_function, k):
    cancer=k_nearest_neighbors(x,train_x,dist_function,k)
    true=0;
    false=0;
    for i in cancer:
        if(train_y[i]==True):
            true+=1
        else:
            false+=1

    return(false-true<=0)



def eval_cancer_classifier(test_x, test_y, classifier):
    prediction=[]
    trouver=0
    nontrouver=0
    for test in test_x:
        prediction.append(classifier(test))
    for i in range (len (prediction)):
        if (prediction[i]==test_y[i]):
            trouver+=1
        else:
            nontrouver+=1
    return(float(nontrouver)/float(len(prediction)))

split_lines("wdbc.data",50,"train","test")
train_x , train_y =read_data("train")
test_x , test_y =read_data("test")

def eval_cancer_untrained_classifier(test_x, test_y, X_train, y_train, classifier):
    predict_y = []
    correct = 0
    wrong = 0

    length = len(test_y)

    for x in test_x:
        predict_y.append(classifier(X_train, y_train ,x))

    for i in range(length):
        if(predict_y[i] != test_y[i]):
            wrong += 1

    return ((float(wrong)/float(length)))


def cross_validation(train_x, train_y, untrained_classifier):

    X = np.array(train_x)
    Y = np.array(train_y)
    kf = KFold(n_splits=5)
    listeTrainx=[]
    listeTrainY=[]
    score=[]
    result=[]
    success=0
    error=0
    length=0

    for train_index, test_index  in kf.split(X):
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], Y[train_index], Y[test_index]
        length=len(X_test)
        for x in X_test:
            result.append(untrained_classifier(  X_train,  y_train, x))
        for i in range (length):
            if result[i]==y_test[i]:
                success+=1
            else:
                error+=1
        score.append((float(error)/float(length)))
        length=0
        result=[]
        success=0
        error=0

    return(sum(score)/5)
        

#print( cross_validation(train_x,train_y, lambda train_x, train_y, x: is_cancerous_knn(x, train_x, train_y, dist_function=simple_distance, k=7)))

def sampled_range(mini, maxi, num):
    if not num:
         return []
    lmini = math.log(mini)
    lmaxi = math.log(maxi)
    ldelta = (lmaxi - lmini) / (num - 1)
    out = [x for x in set([int(math.exp(lmini + i * ldelta)) for i in range(num)])]
    out.sort()
    return out

def find_best_k(train_x, train_y, untrained_classifier_for_k):
    k_list = sampled_range(3, 50, 10)
    best_score = 1
    best_k = 0

    for k in k_list:
        score = cross_validation(train_x, train_y, lambda train_x, train_y, x: is_cancerous_knn(x, train_x,train_y, dist_function=simple_distance, k=k))
        if score < best_score:
            best_score = score
            best_k = k

    return best_k
#print(find_best_k(train_x, train_y, None))   #k=7   64% erreur


def svm_classify(train_x, train_y, X):

    clf = SVC(C=find_best_k(train_x, train_y, None), gamma='auto')

    x = np.array(train_x)
    y = np.array(train_y)
    clf.fit(x, y)

    return clf.predict(X)

#X = np.array(test_x)
#print(svm_classify(train_x, train_y, X)) 
