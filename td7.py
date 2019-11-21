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

def cross_validation(train_x, train_y, untrained_classifier):

    X = np.array(train_x)
    Y = np.array(train_y)
    kf = KFold(n_splits=5)
    listeTrainx=[]
    listeTrainY=[]

    for train_X , train_Y  in kf.split(X,Y):
        listeTrainx.append(train_X)
        listeTrainY.append(train_Y)
