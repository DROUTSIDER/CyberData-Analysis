## Load data from https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-51/
import random
import pandas as pd
import collections
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import datetime
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn import  tree
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")
#create dataframe
columns=['StartTime','Duration', 'Protocol', 'Source','Direction','Dest', 'Flag','Tos','Packet' ,'Bytes','Flows','Label']
lst=[]
with open('capture20110818.pcap.netflow.labeled') as fp:  
    for cnt, line in enumerate(fp):
        k=[]
        if cnt!=0:
            dat=line.split("\t")
            if len(dat)>=13:
                for d in dat:
                    d.strip()
                    if len(d)==0:
                        k = dat.remove(d)
            if k:
                lst.append(k)
            else: lst.append(dat)
dataset=pd.DataFrame(lst, columns=columns)

## Pre-Proccesssing (Remove background, type convertion)
#remove background
dataset=dataset.loc[dataset.Label!='Background\n']
#replace Nan with 0
dataset=dataset.fillna(0)
#type convertion
dataset.Tos=dataset.Tos.astype(int)
dataset['Packet']=dataset['Packet'].astype(int)
dataset['Bytes']=dataset['Bytes'].astype(int)
dataset.Flows=dataset.Flows.astype(int)
#Drop previous NaN samples
dataset=dataset.loc[dataset.Label!=0]
dataset.StartTime=pd.to_datetime(dataset.StartTime)
dataset=dataset.set_index(dataset.StartTime)

## split IPs and ports to create new features
def ip(data):
    return data.split(':')[0]
def port(data):
    if len(data.split(':'))>1:
        return data.split(':')[1]
    else:
        return(' ')

dataset['SourceIP']=dataset['Source'].apply(lambda x: ip(x))
dataset['SourcePort']=dataset['Source'].apply(lambda y: port(y))
dataset['DestIP']=dataset['Dest'].apply(lambda x: ip(x))
dataset['DestPort']=dataset['Dest'].apply(lambda y: port(y))

## Implemente BClus detection method (aggregating netflow)

begin=dataset.index[0]
end=dataset.index[0]
new_dataset=pd.DataFrame()
while begin in dataset.index:
    #take two minutes time window
    end=begin+datetime.timedelta(minutes=2)
    window =dataset.loc[ (dataset.index>=begin) & (dataset.index<=end)]
    remain=dataset.loc[dataset.index> end]
    #create inner time window inside two-minute time window
    begin1=begin
    for i in range(0,2):
        end1=begin1+datetime.timedelta(minutes=1)
        window1 =window.loc[ (window.index>=begin1) & (window.index<=end1)]
        #aggregate
        group = window1.groupby('SourceIP')
        agg = group.aggregate({'Packet': np.sum,'Bytes':np.sum,'Flows':np.sum,'Tos':np.sum})
        agg['Destinations'] = window1.groupby('SourceIP').Dest.nunique()
        agg['SourcePorts'] = window1.groupby('SourceIP').SourcePort.nunique()
        agg['DestPorts'] = window1.groupby('SourceIP').DestPort.nunique()
        new_dataset=new_dataset.append(agg, ignore_index=False)
        begin1=end1
    if len(remain)==0:
        break
    else:
        begin=remain.index[0]
new_dataset=new_dataset.reset_index()

## Label new dataset (1 for infected host, 0 else)

#infected hosts from the website of dataset
def label(data):
    infected={'147.32.84.165','147.32.84.191','147.32.84.192','147.32.84.193','147.32.84.204',
             '147.32.84.205','147.32.84.206','147.32.84.207','147.32.84.208','147.32.84.209'}
    if data in infected:
        return 1
    else:
        return 0

new_dataset['Label']=new_dataset['SourceIP'].apply(lambda y: label(y))

## Classification using DecisionTree on Packet Level 

from sklearn import  tree

ips=new_dataset.SourceIP
final_dataset_packet=new_dataset.drop('SourceIP',axis=1)

TN=[]
FP=[]
FN=[]
TP=[]
Precision=[]
Recall=[]
Accuracy=[]
for i in range(0,10):
    classifier=tree.DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(final_dataset_packet, final_dataset_packet['Label'],test_size=0.2)
    X_train=X_train.drop('Label',axis=1)
    X_test=X_test.drop('Label',axis=1)

    smt=SMOTE(random_state=42, ratio=float(0.5))
    new_X_train, new_y_train=smt.fit_sample(X_train,y_train)
    classifier.fit(new_X_train, new_y_train)
    predicts=classifier.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_pred=predicts,y_true=y_test).ravel()
    precision=float(tp)/(tp+fp)
    recall=float(tp)/(tp+fn)
    accuracy=float(tp+tn)/(tp+fn+tn+fp)
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    TP.append(tp)
    Precision.append(precision)
    Recall.append(recall)
    Accuracy.append(accuracy)

print('True Negative',np.mean(TN))
print('False Negative',np.mean(FN))
print('-------------')
print('True Positive',np.mean(TP))
print('False Positive',np.mean(FP))
print('-------------')
print( 'Precision :', np.mean(Precision))
print ('Recall : ',np.mean(Recall))
print ('Accuracy : ',np.mean(Accuracy))
print('-------------')
print('-------------')

## Classification using DecisionTree  on Host Level

#Group by SourceIP
new_dataset2=new_dataset.groupby('SourceIP')
new_dataset2=new_dataset2.sum()
new_dataset2=new_dataset2.reset_index()

#Assign labels
def label(data):
    infected={'147.32.84.165','147.32.84.191','147.32.84.192','147.32.84.193','147.32.84.204',
             '147.32.84.205','147.32.84.206','147.32.84.207','147.32.84.208','147.32.84.209'}
    if data in infected:
        return 1
    else:
        return 0

new_dataset2['Label']=new_dataset2['SourceIP'].apply(lambda y: label(y))
final_dataset_host=new_dataset2.drop('SourceIP',axis=1)
TN=[]
FP=[]
FN=[]
TP=[]
Precision=[]
Recall=[]
Accuracy=[]
for i in range(0,10):
    classifier=tree.DecisionTreeClassifier()
    X_train, X_test, y_train, y_test = train_test_split(final_dataset_host, final_dataset_host['Label'],test_size=0.2)
    X_train=X_train.drop('Label',axis=1)
    X_test=X_test.drop('Label',axis=1)

    smt=SMOTE(ratio=float(0.5))
    new_X_train, new_y_train=smt.fit_sample(X_train,y_train)
    classifier.fit(new_X_train, new_y_train)
    predicts=classifier.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(predicts,y_test,labels=[0,1]).ravel()
    precision=float(tp)/(tp+fp)
    recall=float(tp)/(tp+fn)
    accuracy=float(tp+tn)/(tp+fn+tn+fp)
    TN.append(tn)
    FP.append(fp)
    FN.append(fn)
    TP.append(tp)
    Precision.append(precision)
    Recall.append(recall)
    Accuracy.append(accuracy)

print('True Negative',np.mean(TN))
print('False Negative',np.mean(FN))
print('-------------')
print('True Positive',np.mean(TP))
print('False Positive',np.mean(FP))
print('-------------')
print( 'Precision :', np.mean(Precision))
print ('Recall : ',np.mean(Recall))
print ('Accuracy : ',np.mean(Accuracy))