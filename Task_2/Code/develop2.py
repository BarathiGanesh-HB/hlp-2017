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
testFilePath = '../Data/task_2_test_set_to_release.txt'

#### loading data
trainClassID, trainTweetTxt = load_data.get_train(trainFilePath)
devClassID, devTweetTxt = load_data.get_dev(devFilePath)
testTweedID, testTweetTxt = load_data.get_test(testFilePath)

trainClassID = trainClassID + devClassID
#### representing as a matrix
mini_df = 1
for k in range (1, 2):
    trainDTMatirix, devDTMatirix, testDTMatirix = representation.get_dtm(trainTweetTxt, devTweetTxt, testTweetTxt,  k)
##    trainDTMatirix, devDTMatirix, testDTMatirix = representation.get_tdidf(trainTweetTxt, devTweetTxt, testTweetTxt,  k)

    trainDTMatirix = trainDTMatirix.todense()
    devDTMatirix = devDTMatirix.todense()
    testDTMatirix = testDTMatirix.todense()

    trainDTMatirix = numpy.concatenate((trainDTMatirix, devDTMatirix))
    

##    Matirix = numpy.concatenate((trainDTMatirix, devDTMatirix))
##    U, s, V = numpy.linalg.svd(Matirix, full_matrices=True)
##    Matirix = U.transpose()
##    trainDTMatirix = Matirix[0:trainDTMatirix.shape[0], :]
##    devDTMatirix = Matirix[trainDTMatirix.shape[0]:Matirix.shape[0], :]
    
    #### separating classes
    class1train = []
    class1 = []
    class2train = []
    class2 = []
    class3train = []
    class3 = []
    for i in range (0, trainDTMatirix.shape[0]):
        if trainClassID[i] == 1:
            class1train.append(trainDTMatirix[i,:].tolist()[0])
            class1.append(trainClassID[i])
        elif trainClassID[i] == 2:
            class2train.append(trainDTMatirix[i,:].tolist()[0])
            class2.append(trainClassID[i])
        else:
            class3train.append(trainDTMatirix[i,:].tolist()[0])
            class3.append(trainClassID[i])

    class1train = numpy.matrix(class1train)
    class2train = numpy.matrix(class2train)
    class3train = numpy.matrix(class3train)

    dev1train = []
    dev1 = []
    dev2train = []
    dev2 = []
    dev3train = []
    dev3 = []
    for i in range (0, devDTMatirix.shape[0]):
        if devClassID[i] == 1:
            dev1train.append(devDTMatirix[i,:].tolist()[0])
            dev1.append(devClassID[i])
        elif devClassID[i] == 2:
            dev2train.append(devDTMatirix[i,:].tolist()[0])
            dev2.append(devClassID[i])
        else:
            dev3train.append(devDTMatirix[i,:].tolist()[0])
            dev3.append(devClassID[i])

    dev1train = numpy.matrix(dev1train)
    dev2train = numpy.matrix(dev2train)
    dev3train = numpy.matrix(dev3train)
    
    #### finding class embeddings
    class1vec = [0] * trainDTMatirix.shape[1]
    class2vec = [0] * trainDTMatirix.shape[1]
    class3vec = [0] * trainDTMatirix.shape[1]
    class1vec = numpy.array(class1vec)
    class2vec = numpy.array(class2vec)
    class3vec = numpy.array(class3vec)
    
    for i in range (0,trainDTMatirix.shape[0]):
        if trainClassID[i] == 1:
            class1vec = class1vec + (trainDTMatirix[i,:].tolist()[0])
        elif trainClassID[i] == 2:
            class2vec = class2vec + (trainDTMatirix[i,:].tolist()[0])
        else:
            class3vec = class3vec + (trainDTMatirix[i,:].tolist()[0])
    class1vec = class1vec.tolist()
    class2vec = class2vec.tolist()
    class3vec = class3vec.tolist()

    def get_feat(trainDTMatirix, male, female):
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
        return trainDTMatirix

    
    matrix12 = numpy.concatenate((class1train , class2train), axis=0)
    matrix13 = numpy.concatenate((class1train , class3train), axis=0)
    matrix23 = numpy.concatenate((class2train , class3train), axis=0)
    class12 = class1 + class2
    class13 = class1 + class3
    class23 = class2 + class3

    feat12 = get_feat(matrix12, class1vec, class2vec)
    feat13 = get_feat(matrix13, class1vec, class3vec)
    feat23 = get_feat(matrix23, class2vec, class3vec)

    
    matrixdev12 = numpy.concatenate((dev1train , dev2train), axis=0)
    matrixdev13 = numpy.concatenate((dev1train , dev3train), axis=0)
    matrixdev23 = numpy.concatenate((dev2train , dev3train), axis=0)
    dev12 = dev1 + dev2
    dev13 = dev1 + dev3
    dev23 = dev2 + dev3
    
    #### classification
    from sklearn import svm
    clf12 = svm.SVC()
    clf12.fit(feat12, class12)
    clf13 = svm.SVC()
    clf13.fit(feat13, class13)
    clf23 = svm.SVC()
    clf23.fit(feat23, class23)

##    from sklearn.ensemble import RandomForestClassifier
##    for l in range (10,150, 10):
##        print 'estimator', l
##        clf12 = RandomForestClassifier(n_estimators = l)
##        clf12.fit(feat12, class12)
##        clf13 = RandomForestClassifier(n_estimators = l)
##        clf13.fit(feat13, class13)
##        clf23 = RandomForestClassifier(n_estimators = l)
##        clf23.fit(feat23, class23)
    devDTMatirix = testDTMatirix
    prediction = []
    for i in range (0, devDTMatirix.shape[0]):
        test = devDTMatirix[i, :]
        testfeat = get_feat(test, class1vec, class2vec)
        testClass = clf12.predict(testfeat)[0]
        if testClass == 1:
            testfeat = get_feat(test, class1vec, class3vec)
            testClass = clf13.predict(testfeat)[0]
        else:
            testfeat = get_feat(test, class2vec, class3vec)
            testClass = clf23.predict(testfeat)[0]
        prediction.append(testClass)

    writeFile = open('task_2run1.tsv', 'w')
    for thing1, thing2 in zip(testTweedID, prediction):
        writeFile.write(thing1+'\t'+thing2+'\n')
    writeFile.close()
    
