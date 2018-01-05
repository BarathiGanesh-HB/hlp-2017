from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

File = open('/home/barathiganeshhb/Desktop/2', 'r+')
text = File.read()
text = text.replace('\n', '')
File.close()

## function for creating document - term matrix
def get_dtm(trainTweetTxt, devTweetTxt, testTweetTxt,  mini_df):
    data_samples = trainTweetTxt + devTweetTxt + testTweetTxt + [text]
    print 'term document matrix'
    print 'minimum document frequency', mini_df
    tf_vectorizer = CountVectorizer(min_df = mini_df)
    tf_vectorizer.fit(data_samples)
    trainDTMatirix = tf_vectorizer.transform(trainTweetTxt)
    devDTMatirix = tf_vectorizer.transform(devTweetTxt)
    testDTMatirix = tf_vectorizer.transform(testTweetTxt)
    return trainDTMatirix, devDTMatirix, testDTMatirix

## function for creating term frequency - inverse document frequency matrix
def get_tdidf(trainTweetTxt, devTweetTxt, testTweetTxt,  mini_df):
    data_samples = trainTweetTxt + devTweetTxt + testTweetTxt + [text]
    print 'term frequency - inverse document frequency matrix'
    print 'minimum document frequency', mini_df
    tfidf_vectorizer = TfidfVectorizer(min_df = mini_df)
    tfidf_vectorizer.fit(data_samples)
    trainDTMatirix = tfidf_vectorizer.transform(trainTweetTxt)
    devDTMatirix = tfidf_vectorizer.transform(devTweetTxt)
    testDTMatirix = tfidf_vectorizer.transform(testTweetTxt)
    return trainDTMatirix, devDTMatirix, testDTMatirix
