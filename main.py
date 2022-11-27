import pandas as pd
from DataPreprocessing import denoise_text, remove_special_characters, simple_stemmer, remove_stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from LogisticRegression import logistic_bow, logistic_tfidf
from SVC import support_bow, support_tfidf
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from MultinomialNB import multi_tfidf, multi_bow
import pickle

lr_filename_bow = './models/Lr_bow.sav'
lr_filename_tfidf = './models/Lr_tfidf.sav'
svc_filename_bow = './models/SVC_bow.sav'
svc_filename_tfidf = './models/SVC_tfidf.sav'
mnb_filename_bow = './models/mnb_bow'
mnb_filename_tfidf = './models/mnb_tfidf'

# For bag-of-words
cv = CountVectorizer(min_df=0, max_df=1, binary=False, ngram_range=(1, 3))
tv = TfidfVectorizer(min_df=0, max_df=1, use_idf=True, ngram_range=(1, 3))
imdb_data = pd.read_csv('./input/IMDB Dataset.csv')


def linear_models_sentiment_analysis(sentence):
    sentence = denoise_text(sentence)
    sentence = remove_special_characters(sentence)
    sentence = simple_stemmer(sentence)
    sentence = remove_stopwords(sentence)

    with open(lr_filename_bow, 'rb') as f:
        lr_bow, cv_fit, tv_fit = pickle.load(f)
    lr_tfidf = pickle.load(open(lr_filename_tfidf, 'rb'))
    svc_bow = pickle.load(open(svc_filename_bow, 'rb'))
    svc_tfidf = pickle.load(open(svc_filename_tfidf, 'rb'))
    mnb_bow = pickle.load(open(mnb_filename_bow, 'rb'))
    mnb_tfidf = pickle.load(open(mnb_filename_tfidf, 'rb'))

    transform_bow = cv_fit.transform([sentence])
    transform_tfidf = tv_fit.transform([sentence])

    prediction_lr_bow = int(lr_bow.predict(transform_bow)[0])
    prediction_lr_tfidf = int(lr_tfidf.predict(transform_tfidf)[0])
    prediction_svc_bow = int(svc_bow.predict(transform_bow)[0])
    prediction_svc_tfidf = int(svc_tfidf.predict(transform_tfidf)[0])
    prediction_mnb_bow = int(mnb_bow.predict(transform_bow)[0])
    prediction_mnb_tfidf = int(mnb_tfidf.predict(transform_tfidf)[0])

    prediction = {
        "prediction_lr_bow": prediction_lr_bow,
        "prediction_lr_tfidf": prediction_lr_tfidf,
        "prediction_svc_bow": prediction_svc_bow,
        "prediction_svc_tfidf": prediction_svc_tfidf,
        "prediction_mnb_bow": prediction_mnb_bow,
        "prediction_mnb_tfidf": prediction_mnb_tfidf
    }
    print(prediction)
    return prediction


def model_training():

    # split the dataset
    # train dataset
    train_reviews = imdb_data.review[:40000]
    train_sentiments = imdb_data.sentiment[:40000]
    # test dataset
    test_reviews = imdb_data.review[40000:]
    test_sentiments = imdb_data.sentiment[40000:]

    imdb_data['review'] = imdb_data['review'].apply(denoise_text)
    imdb_data['review'] = imdb_data['review'].apply(remove_special_characters)
    imdb_data['review'] = imdb_data['review'].apply(simple_stemmer)
    imdb_data['review'] = imdb_data['review'].apply(remove_stopwords)

    norm_train_reviews = imdb_data.review[:40000]
    norm_test_reviews = imdb_data.review[40000:]

    cv_train_reviews = cv.fit_transform(norm_train_reviews)
    cv_test_reviews = cv.transform(norm_test_reviews)

    tv_train_reviews = tv.fit_transform(norm_train_reviews)
    tv_test_reviews = tv.transform(norm_test_reviews)

    # labeling the sentient data
    lb = LabelBinarizer()
    # transformed sentiment data
    sentiment_data = lb.fit_transform(imdb_data['sentiment'])

    train_sentiments = sentiment_data[:40000]
    test_sentiments = sentiment_data[40000:]

    logistic_bow(cv_train_reviews, train_sentiments, test_sentiments, cv_test_reviews, cv, tv)
    logistic_tfidf(tv_train_reviews, train_sentiments, test_sentiments, tv_test_reviews)

    support_bow(cv_train_reviews, train_sentiments, test_sentiments, cv_test_reviews)
    support_tfidf(tv_train_reviews, train_sentiments, test_sentiments, tv_test_reviews)

    multi_bow(cv_train_reviews, train_sentiments, test_sentiments, cv_test_reviews)
    multi_tfidf(tv_train_reviews, train_sentiments, test_sentiments, tv_test_reviews)

    # word cloud for positive review words
    plt.figure(figsize=(10, 10))
    positive_text = norm_train_reviews[1]
    WC = WordCloud(width=1000, height=500, max_words=500, min_font_size=5)
    positive_words = WC.generate(positive_text)
    plt.imshow(positive_words, interpolation='bilinear')
    plt.show

    # word cloud for negative review words
    plt.figure(figsize=(10, 10))
    negative_text = norm_train_reviews[8]
    WC = WordCloud(width=1000, height=500, max_words=500, min_font_size=5)
    negative_words = WC.generate(negative_text)
    plt.imshow(negative_words, interpolation='bilinear')
    plt.show


model_training()

