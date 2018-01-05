from __future__ import division

import os
import numpy
import load_data
import representation
from sklearn import svm

from scipy.spatial.distance import correlation
from scipy.spatial.distance import cosine
from scipy.spatial.distance import euclidean

#### input paths
trainFilePath = '../Data/train-data.txt'
devFilePath = '../Data/dev-data.txt'
testFilePath = '../Data/hlp_workshop_test_set_to_release.txt'

#### loading data
trainClassID, trainTweetTxt = load_data.get_train(trainFilePath)
devClassID, devTweetTxt = load_data.get_dev(devFilePath)
testTweedID, testTweetTxt = load_data.get_test(testFilePath)

#### representing as a matrix
j = 1
trainDTMatirix, devDTMatirix, testDTMatirix = representation.get_dtm(trainTweetTxt, devTweetTxt, testTweetTxt,  j)

trainDTMatirix = numpy.concatenate((trainDTMatirix.todense(), devDTMatirix.todense()), axis=0)
trainClassID = trainClassID + devClassID
#### for class embedding representation
male = [0] * trainDTMatirix.shape[1]
female = [0] * trainDTMatirix.shape[1]
male = numpy.array(male)
female = numpy.array(female)
for i in range (0,trainDTMatirix.shape[0]):
    if trainClassID[i] == 0:
        male = male + (trainDTMatirix[i,:].tolist()[0])
    elif trainClassID[i] == 1:
        female = female + (trainDTMatirix[i,:].tolist()[0])
    else:
        pass
male = male.tolist()
female = female.tolist()

featMatrix = []
for i in range (0, trainDTMatirix.shape[0]):
    tempfeat = []
    tempfeat.append(correlation(male,trainDTMatirix[i,:].tolist()[0]))
    tempfeat.append(cosine(male,trainDTMatirix[i,:].tolist()[0]))
    tempfeat.append(euclidean(male,trainDTMatirix[i,:].tolist()[0]))
    tempfeat.append(correlation(female,trainDTMatirix[i,:].tolist()[0]))
    tempfeat.append(cosine(female,trainDTMatirix[i,:].tolist()[0]))
    tempfeat.append(euclidean(female,trainDTMatirix[i,:].tolist()[0]))
    featMatrix.append(tempfeat)
featMatrix = numpy.matrix(featMatrix)
featMatrix = numpy.nan_to_num(featMatrix)
trainDTMatirix = featMatrix

#### separating instances belongs to class 0 and 1
trainzero = []
trainone = []
for i in range (0, trainDTMatirix.shape[0]):
    if trainClassID[i] == 0:
        trainzero.append(trainDTMatirix[i,:].tolist()[0])
    elif trainClassID[i] == 1:
        trainone.append(trainDTMatirix[i,:].tolist()[0])
    else:
        pass

#### for getting densed numpy matrix
trainzero = numpy.matrix(trainzero)
trainone = numpy.matrix(trainone)

clf1 = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf1.fit(trainone)

y_pred_train1 = clf1.predict(trainone)
print 'error against train positive: ', y_pred_train1[y_pred_train1 == -1].size

y_pred_train2 = clf1.predict(trainzero)
print 'error against train Negative: ', y_pred_train2[y_pred_train2 == 1].size

print '###### model and performace for negative class #############'
clf2 = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf2.fit(trainzero)


y_pred_train3 = clf2.predict(trainzero)
print 'error against train positive: ', y_pred_train3[y_pred_train3 == -1].size

y_pred_train4 = clf2.predict(trainone)

print 'error against train Negative: ', y_pred_train4[y_pred_train4 == 1].size

testfeatMatrix = []
for i in range (0, testDTMatirix.shape[0]):
    tempfeat = []
    tempfeat.append(correlation(male,testDTMatirix[i,:].toarray().tolist()[0]))
    tempfeat.append(cosine(male, testDTMatirix[i,:].toarray().tolist()[0]))
    tempfeat.append(euclidean(male,testDTMatirix[i,:].toarray().tolist()[0]))
    tempfeat.append(correlation(female,testDTMatirix[i,:].toarray().tolist()[0]))
    tempfeat.append(cosine(female,testDTMatirix[i,:].toarray().tolist()[0]))
    tempfeat.append(euclidean(female,testDTMatirix[i,:].toarray().tolist()[0]))
    testfeatMatrix.append(tempfeat)
testfeatMatrix = numpy.matrix(testfeatMatrix)
testfeatMatrix = numpy.nan_to_num(testfeatMatrix)
testDTMatirix = testfeatMatrix

y_pred = clf2.predict(testDTMatirix)
writeFile = open('Task_1_run1.tsv', 'w')
for thing1, thing2 in zip(testTweedID, y_pred):
    if thing2 == 1:
        writeFile.write(thing1+'\t'+str(0)+'\n')
    else:
        writeFile.write(thing1+'\t'+str(1)+'\n')
writeFile.close()

