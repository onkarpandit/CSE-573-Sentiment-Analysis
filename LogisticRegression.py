from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import pickle

lr = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
filename_bow = './models/Lr_bow.sav'
filename_tfidf = './models/Lr_tfidf.sav'


def logistic_bow(cv_train_reviews, train_sentiments, test_sentiments, cv_test_reviews, cv_fit, tv_fit):
    lr_bow = lr.fit(cv_train_reviews, train_sentiments)
    print(lr_bow)

    lr_bow_predict = lr_bow.predict(cv_test_reviews)
    print(lr_bow_predict)

    lr_bow_score = accuracy_score(test_sentiments, lr_bow_predict)
    print("lr_bow_score :", lr_bow_score)

    lr_bow_report = classification_report(test_sentiments, lr_bow_predict, target_names=['Positive', 'Negative'])
    print(lr_bow_report)
    pickle.dump((lr_bow, cv_fit, tv_fit), open(filename_bow, 'wb'))


def logistic_tfidf(tv_train_reviews, train_sentiments, test_sentiments, tv_test_reviews):
    # Fitting the model for tfidf features
    lr_tfidf = lr.fit(tv_train_reviews, train_sentiments)
    print(lr_tfidf)

    # Predicting the model for tfidf features
    lr_tfidf_predict = lr_tfidf.predict(tv_test_reviews)
    print(lr_tfidf_predict)

    lr_tfidf_score = accuracy_score(test_sentiments, lr_tfidf_predict)
    print("lr_tfidf_score :", lr_tfidf_score)

    lr_tfidf_report = classification_report(test_sentiments, lr_tfidf_predict, target_names=['Positive', 'Negative'])
    print(lr_tfidf_report)
    pickle.dump(lr_tfidf, open(filename_tfidf, 'wb'))

