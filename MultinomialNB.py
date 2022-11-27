from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pickle

mnb = MultinomialNB()
filename_bow = './models/mnb_bow.sav'
filename_tfidf = './models/mnb_tfidf.sav'


def multi_bow(cv_train_reviews, train_sentiments, test_sentiments, cv_test_reviews):
    mnb_bow = mnb.fit(cv_train_reviews, train_sentiments)
    print(mnb_bow)

    mnb_bow_predict = mnb_bow.predict(cv_test_reviews)
    print(mnb_bow_predict)

    mnb_bow_score = accuracy_score(test_sentiments, mnb_bow_predict)
    print("mnb_bow_score :", mnb_bow_score)

    mnb_bow_report = classification_report(test_sentiments, mnb_bow_predict, target_names=['Positive', 'Negative'])
    print(mnb_bow_report)
    pickle.dump(mnb_bow, open(filename_bow, 'wb'))


def multi_tfidf(tv_train_reviews, train_sentiments, test_sentiments, tv_test_reviews):
    mnb_tfidf = mnb.fit(tv_train_reviews, train_sentiments)
    print(mnb_tfidf)

    mnb_tfidf_predict = mnb_tfidf.predict(tv_test_reviews)
    print(mnb_tfidf_predict)

    mnb_tfidf_score = accuracy_score(test_sentiments, mnb_tfidf_predict)
    print("mnb_tfidf_score :", mnb_tfidf_score)

    mnb_tfidf_report = classification_report(test_sentiments, mnb_tfidf_predict, target_names=['Positive', 'Negative'])
    print(mnb_tfidf_report)
    pickle.dump(mnb_tfidf, open(filename_tfidf, 'wb'))
