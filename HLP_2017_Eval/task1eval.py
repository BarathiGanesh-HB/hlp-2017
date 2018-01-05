#goldstandardfile = './task1a/subsampl1.txt'
goldstandardfile = 'hlp_workshop_testset_labels_task1.txt'
#input_directory = './task1/Northeastern NLP Task 1/'
input_directory = './task1/'

from os import listdir
from string import strip

goldstandard_dict = {}
infile = open(goldstandardfile)
for line in infile:
    items = strip(line).split('\t')
    id_ = items[0]
    class_ = items[-1]
    goldstandard_dict[id_] = class_

sublist = listdir(input_directory)
for file_ in sublist:
    tp = 0.
    fp = 0.
    fn = 0.
    tn = 0.
    #prediction_dict = {}
    infile = open(input_directory+file_)
    for line in infile:
        items = strip(line).split('\t')
        id_ = items[0]
        class_ = items[-1]
        if str(class_) == '1' and goldstandard_dict[id_]=='1':
            tp+=1.
        if str(class_) == '1' and goldstandard_dict[id_] == '0':
            fp+=1.
        if str(class_) == '0' and goldstandard_dict[id_] == '1':
            fn+=1.
        if str(class_) == '0' and goldstandard_dict[id_] == '0':
            tn+=1.
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    f = (2*p*r)/(p+r)

    print 'Results for submission: ' + file_
    print 'ADR Precision ' + str(p)
    print 'ADR Recall ' + str(r)
    print 'ADR F-score: ' + str(f)+'\n'