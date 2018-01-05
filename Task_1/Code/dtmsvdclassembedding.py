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

for j in range (4,17):
    trainDTMatirix, devDTMatirix, testDTMatirix = representation.get_dtm(trainTweetTxt, devTweetTxt, testTweetTxt,  j)
##    trainDTMatirix, devDTMatirix, testDTMatirix = representation.get_tdidf(trainTweetTxt, devTweetTxt, testTweetTxt,  j)

    ## for distributed representation
    Matirix = numpy.concatenate((trainDTMatirix.todense(), devDTMatirix.todense()))
    U, s, V = numpy.linalg.svd(Matirix, full_matrices=True)
    Matirix = U.transpose()
    trainDTMatirix = Matirix[0:trainDTMatirix.shape[0], :]
    devDTMatirix = Matirix[trainDTMatirix.shape[0]:Matirix.shape[0], :]
    ##
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
        tempfeat.append(correlation(male,trainDTMatirix[i,:].tolist()[0]))
        tempfeat.append(cosine(male,trainDTMatirix[i,:].tolist()[0]))
        tempfeat.append(euclidean(male,trainDTMatirix[i,:].tolist()[0]))
        featMatrix.append(tempfeat)
    featMatrix = numpy.matrix(featMatrix)
    featMatrix = numpy.nan_to_num(featMatrix)
    trainDTMatirix = featMatrix

    devfeatMatrix = []
    for i in range (0, devDTMatirix.shape[0]):
        tempfeat = []
        tempfeat.append(correlation(male,devDTMatirix[i,:].tolist()[0]))
        tempfeat.append(cosine(male,devDTMatirix[i,:].tolist()[0]))
        tempfeat.append(euclidean(male,devDTMatirix[i,:].tolist()[0]))
        tempfeat.append(correlation(male,devDTMatirix[i,:].tolist()[0]))
        tempfeat.append(cosine(male,devDTMatirix[i,:].tolist()[0]))
        tempfeat.append(euclidean(male,devDTMatirix[i,:].tolist()[0]))
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
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(trainone)

    y_pred_test1 = clf.predict(devone)
    y_pred_train1 = clf.predict(trainone)
    print 'error against train positive: ', y_pred_train1[y_pred_train1 == -1].size
    print 'error against dev positive: ', y_pred_test1[y_pred_test1 == -1].size

    y_pred_test2 = clf.predict(devzero)
    y_pred_train2 = clf.predict(trainzero)
    print 'error against train Negative: ', y_pred_train2[y_pred_train2 == 1].size
    print 'error against dev Negative: ', y_pred_test2[y_pred_test2 == 1].size

    print 'train Accuracy : ', (trainDTMatirix.shape[0] - (y_pred_train1[y_pred_train1 == -1].size + y_pred_train2[y_pred_train2 == 1].size))/trainDTMatirix.shape[0]
    print 'dev Accuracy : ', (devDTMatirix.shape[0] - (y_pred_test1[y_pred_test1 == -1].size + y_pred_test2[y_pred_test2 == 1].size))/devDTMatirix.shape[0]

    print '###### model and performace for negative class #############'
    clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
    clf.fit(trainzero)

    y_pred_test3 = clf.predict(devzero)
    y_pred_train3 = clf.predict(trainzero)
    print 'error against train positive: ', y_pred_train3[y_pred_train3 == -1].size
    print 'error against dev positive: ', y_pred_test3[y_pred_test3 == -1].size

    y_pred_test4 = clf.predict(devone)
    y_pred_train4 = clf.predict(trainone)

    print 'error against train Negative: ', y_pred_train4[y_pred_train4 == 1].size
    print 'error against dev Negative: ', y_pred_test4[y_pred_test4 == 1].size

    print 'train Accuracy : ', (trainDTMatirix.shape[0] - (y_pred_train3[y_pred_train3 == -1].size + y_pred_train4[y_pred_train4 == 1].size))/trainDTMatirix.shape[0]
    print 'dev Accuracy : ', (devDTMatirix.shape[0] - (y_pred_test3[y_pred_test3 == -1].size + y_pred_test4[y_pred_test4 == 1].size))/devDTMatirix.shape[0]


    ##y_pred = clf.predict(testTweedID.todense())
    ##writeFile = open('one.tsv', 'w')
    ##for thing1, thing2 in zip(tweetIDtest, y_pred):
    ##    if thing2 == 1:
    ##        writeFile.write(thing1+'\t'+str(1)+'\n')
    ##    else:
    ##        writeFile.write(thing1+'\t'+str(0)+'\n')
    ##writeFile.close()

