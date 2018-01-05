def get_train(filePath):
    #### creating train documents
    tweedID = []
    userID = []
    temp = []
    classID = []
    tweetTxt = []
    readFile = open(filePath, 'r+')
    for line in readFile:
        line = line.replace('\n', '')
        items = line.split('\t')
        tweedID.append(items[0])
        userID.append(items[1])
        temp.append(items[2])
        classID.append(int(items[3]))
        tweetTxt.append(items[4])
    readFile.close()
    return classID, tweetTxt

def get_dev(filePath):
    #### creating dev documents
    tweedID = []
    userID = []
    temp = []
    classID = []
    tweetTxt = []
    readFile = open(filePath, 'r+')
    for line in readFile:
        line = line.replace('\n', '')
        items = line.split('\t')
        tweedID.append(items[0])
        userID.append(items[1])
        temp.append(items[2])
        classID.append(int(items[3]))
        tweetTxt.append(items[4])
    readFile.close()
    return classID, tweetTxt

def get_test(filePath):
    #### creating test documents
    tweedID = []
    tweetTxt = []
    readFile = open(filePath, 'r+')
    for line in readFile:
        line = line.replace('\n', '')
        items = line.split('\t')
        tweedID.append(items[0])
        tweetTxt.append(items[1])
    readFile.close()
    return tweedID, tweetTxt
