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

##    trainDTMatirix, devDTMatirix, testDTMatirix = representation.get_dtm(trainTweetTxt, devTweetTxt, testTweetTxt,  j)
trainDTMatirix, devDTMatirix, testDTMatirix = representation.get_tdidf(trainTweetTxt, devTweetTxt, testTweetTxt,  j)

male = [0] * trainDTMatirix.shape[1]
female = [0] * trainDTMatirix.shape[1]
male = numpy.array(male)
female = numpy.array(female)
for i in range (0,trainDTMatirix.shape[0]):
    if trainClassID[i] == 0:
        male = male + (trainDTMatirix[i,:].toarray().tolist()[0])
    elif trainClassID[i] == 1:
        female = female + (trainDTMatirix[i,:].toarray().tolist()[0])
    else:
        pass
male = male.tolist()
female = female.tolist()

featMatrix = []
for i in range (0, trainDTMatirix.shape[0]):
    tempfeat = []
    tempfeat.append(correlation(male,trainDTMatirix[i,:].toarray().tolist()[0]))
    tempfeat.append(cosine(male,trainDTMatirix[i,:].toarray().tolist()[0]))
    tempfeat.append(euclidean(male,trainDTMatirix[i,:].toarray().tolist()[0]))
    tempfeat.append(correlation(male,trainDTMatirix[i,:].toarray().tolist()[0]))
    tempfeat.append(cosine(male,trainDTMatirix[i,:].toarray().tolist()[0]))
    tempfeat.append(euclidean(male,trainDTMatirix[i,:].toarray().tolist()[0]))
    featMatrix.append(tempfeat)
featMatrix = numpy.matrix(featMatrix)
featMatrix = numpy.nan_to_num(featMatrix)
trainDTMatirix = featMatrix

devfeatMatrix = []
for i in range (0, devDTMatirix.shape[0]):
    tempfeat = []
    tempfeat.append(correlation(male,devDTMatirix[i,:].toarray().tolist()[0]))
    tempfeat.append(cosine(male,devDTMatirix[i,:].toarray().tolist()[0]))
    tempfeat.append(euclidean(male,devDTMatirix[i,:].toarray().tolist()[0]))
    tempfeat.append(correlation(male,devDTMatirix[i,:].toarray().tolist()[0]))
    tempfeat.append(cosine(male,devDTMatirix[i,:].toarray().tolist()[0]))
    tempfeat.append(euclidean(male,devDTMatirix[i,:].toarray().tolist()[0]))
    devfeatMatrix.append(tempfeat)
devfeatMatrix = numpy.matrix(devfeatMatrix)
devfeatMatrix = numpy.nan_to_num(devfeatMatrix)
devDTMatirix = devfeatMatrix

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

devzero = []
devone = []
for i in range (0,devDTMatirix.shape[0]):
    if devClassID[i] == 0:
        devzero.append(devDTMatirix[i,:].tolist()[0])
    elif devClassID[i] == 1:
        devone.append(devDTMatirix[i,:].tolist()[0])
    else:
        pass

#### for getting densed numpy matrix
trainzero = numpy.matrix(trainzero)
trainone = numpy.matrix(trainone)
devzero = numpy.matrix(devzero)
devone = numpy.matrix(devone)

#### developing classification model
print '###### model and performace for positive class #############'
clf1 = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf1.fit(trainone)

y_pred_test1 = clf1.predict(devone)
y_pred_train1 = clf1.predict(trainone)
print 'error against train positive: ', y_pred_train1[y_pred_train1 == -1].size
print 'error against dev positive: ', y_pred_test1[y_pred_test1 == -1].size

y_pred_test2 = clf1.predict(devzero)
y_pred_train2 = clf1.predict(trainzero)
print 'error against train Negative: ', y_pred_train2[y_pred_train2 == 1].size
print 'error against dev Negative: ', y_pred_test2[y_pred_test2 == 1].size

print 'train Accuracy : ', (trainDTMatirix.shape[0] - (y_pred_train1[y_pred_train1 == -1].size + y_pred_train2[y_pred_train2 == 1].size))/trainDTMatirix.shape[0]
print 'dev Accuracy : ', (devDTMatirix.shape[0] - (y_pred_test1[y_pred_test1 == -1].size + y_pred_test2[y_pred_test2 == 1].size))/devDTMatirix.shape[0]

print '###### model and performace for negative class #############'
clf2 = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
clf2.fit(trainzero)

y_pred_test3 = clf2.predict(devzero)
y_pred_train3 = clf2.predict(trainzero)
print 'error against train positive: ', y_pred_train3[y_pred_train3 == -1].size
print 'error against dev positive: ', y_pred_test3[y_pred_test3 == -1].size

y_pred_test4 = clf2.predict(devone)
y_pred_train4 = clf2.predict(trainone)

print 'error against train Negative: ', y_pred_train4[y_pred_train4 == 1].size
print 'error against dev Negative: ', y_pred_test4[y_pred_test4 == 1].size

print 'train Accuracy : ', (trainDTMatirix.shape[0] - (y_pred_train3[y_pred_train3 == -1].size + y_pred_train4[y_pred_train4 == 1].size))/trainDTMatirix.shape[0]
print 'dev Accuracy : ', (devDTMatirix.shape[0] - (y_pred_test3[y_pred_test3 == -1].size + y_pred_test4[y_pred_test4 == 1].size))/devDTMatirix.shape[0]


y_pred = clf2.predict(testDTMatirix.todense())
writeFile = open('Task_1_run4.tsv', 'w')
for thing1, thing2 in zip(testTweedID, y_pred):
    if thing2 == 1:
        writeFile.write(thing1+'\t'+str(1)+'\n')
    else:
        writeFile.write(thing1+'\t'+str(0)+'\n')
writeFile.close()

