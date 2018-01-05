goldstandardfile = './goldstandard_task2_new.txt'
input_directory = './task2/allsubmissions/csv/'

from os import listdir
from string import strip

goldstandard_dict = {}
infile = open(goldstandardfile)
for line in infile:
    items = strip(line).split('\t')
    id_ = items[0]#+'-'+items[1]
    class_ = items[-1]
    goldstandard_dict[id_] = class_

sublist = listdir(input_directory)
for file_ in sublist:
  if not file_ == '.DS_Store':
    tp = 0.
    fp = 0.
    fn = 0.
    tn = 0.
    #prediction_dict = {}
    infile = open(input_directory+file_)
    print 'Results for submission: ' + file_
    for line in infile:
        #print line.encode('utf-8')
        items = strip(line).split(',')#split('\t')
        id_ = items[0]#+'-'+items[1]
        class_ = items[-1]
        if (str(class_) == '1' and goldstandard_dict[id_]=='1') or (str(class_) == '2' and goldstandard_dict[id_]=='2'):
            tp+=1.
        if (str(class_) == '1' and not goldstandard_dict[id_] == '1') or (str(class_) == '2' and not goldstandard_dict[id_] == '2'):
            fp+=1.
        if str(class_) == '3' and not goldstandard_dict[id_] == '3' or str(class_) == '2' and goldstandard_dict[id_] == '1' or str(class_) == '1' and goldstandard_dict[id_] == '2':
            fn+=1.
        if str(class_) == '3' and goldstandard_dict[id_] == '3' :
            tn+=1.
    p = tp/(tp+fp)
    r = tp/(tp+fn)
    f = (2*p*r)/(p+r)


    print 'Micro-averaged Precision for classes 1 and 2: ' + str(p)
    print 'Micro-averaged Recall for classes 1 and 2: ' + str(r)
    print 'Micro-averaged F-score for classes 1 and 2: ' + str(f) + '\n'
