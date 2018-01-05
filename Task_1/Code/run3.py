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


trainDTMatirix, devDTMatirix, testDTMatirix = representation.get_dtm(trainTweetTxt, devTweetTxt, testTweetTxt,  1)
##trainDTMatirix, devDTMatirix, testDTMatirix = representation.get_tdidf(trainTweetTxt, devTweetTxt, testTweetTxt,  1)
trainDTMatirix = trainDTMatirix.todense()
devDTMatirix = devDTMatirix.todense()

trainDTMatirix = numpy.concatenate((trainDTMatirix, devDTMatirix), axis=0)
trainClassID = trainClassID + devClassID

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



y_pred = clf2.predict(testDTMatirix.todense())
writeFile = open('Task_1_run3.tsv', 'w')
for thing1, thing2 in zip(testTweedID, y_pred):
    if thing2 == 1:
        writeFile.write(thing1+'\t'+str(0)+'\n')
    else:
        writeFile.write(thing1+'\t'+str(1)+'\n')
writeFile.close()

