from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
import pickle

svm = SGDClassifier(loss='hinge', max_iter=500, random_state=42)
filename_bow = './models/SVC_bow.sav'
filename_tfidf = './models/SVC_tfidf.sav'


def support_bow(cv_train_reviews, train_sentiments, test_sentiments, cv_test_reviews):
    svm_bow = svm.fit(cv_train_reviews, train_sentiments)
    print(svm_bow)

    svm_bow_predict = svm_bow.predict(cv_test_reviews)
    print(svm_bow_predict)

    svm_bow_score = accuracy_score(test_sentiments, svm_bow_predict)
    print("svm_bow_score :", svm_bow_score)

    svm_bow_report = classification_report(test_sentiments, svm_bow_predict, target_names=['Positive', 'Negative'])
    print(svm_bow_report)
    pickle.dump(svm_bow, open(filename_bow, 'wb'))


def support_tfidf(tv_train_reviews, train_sentiments, test_sentiments, tv_test_reviews):
    svm_tfidf = svm.fit(tv_train_reviews, train_sentiments)
    print(svm_tfidf)

    svm_tfidf_predict = svm_tfidf.predict(tv_test_reviews)
    print(svm_tfidf_predict)

    svm_tfidf_score = accuracy_score(test_sentiments, svm_tfidf_predict)
    print("svm_tfidf_score :", svm_tfidf_score)

    svm_tfidf_report = classification_report(test_sentiments, svm_tfidf_predict, target_names=['Positive', 'Negative'])
    print(svm_tfidf_report)
    pickle.dump(svm_tfidf, open(filename_tfidf, 'wb'))