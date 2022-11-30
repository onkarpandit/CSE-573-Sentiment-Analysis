import random

import matplotlib.pyplot as plt
import tensorflow as tf
from numpy.random import seed
from sklearn.manifold import TSNE

from HAHNN import HAHNN
from constants import *
from dataLoadUtilities import *

tf.compat.v1.disable_eager_execution()

nltk.download('punkt')
nltk.download('stopwords')

os.environ['PYTHONHASHSEED'] = str(1024)
tf.random.set_seed(1024)
seed(1024)
np.random.seed(1024)
random.seed(1024)


def plot_term_embd(_embds, titles, _title='tsne.png'):
    plt.figure(figsize=(18, 18))
    for pos, title in enumerate(titles):
        x_co, y_co = _embds[pos, :]
        plt.scatter(x_co, y_co)
        plt.annotate(title,
                     xy=(x_co, y_co),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.savefig(_title)


if __name__ == '__main__':
    YELP_DATA_PATH = '../yelp_reviews_sampling.json'
    IMDB_DATA_PATH = '../imdb_reviews.csv'
    SAVED_MODEL_DIR = '../saved_models'
    SAVED_MODEL_FILENAME = 'model.tf'

    if dataset == 'yelp':
        if not os.path.isfile("../yelpTraining.npy"):
            (x_training, y_training) = get_movie_reviews_yelp(json_location=YELP_DATA_PATH, size=400000)
            X_arr = np.array(x_training)
            np.save("../yelpTraining.npy", X_arr)

            Y_arr = np.array(y_training)
            np.save("../yelpLabels.npy", Y_arr)
        else:
            (x_training, y_training) = np.load("../yelpTraining.npy", allow_pickle=True), np.load("../yelpLabels.npy", allow_pickle=True)
    else:
        if not os.path.isfile("../trainingData.npy"):
            (x_training, y_training) = get_movie_reviews_imdb(csv_location=IMDB_DATA_PATH, size=49000)

            X_arr = np.array(x_training)
            np.save("../trainingData.npy", X_arr)

            Y_arr = np.array(y_training)
            np.save("../labels.npy", Y_arr)
        else:
            (x_training, y_training) = np.load("../trainingData.npy", allow_pickle=True), np.load("../labels.npy", allow_pickle=True)

    _max = 200
    vect_size = 200

    f_title = '../fasttext_model.txt'
    _fasttext = gensim.models.FastText.load(f_title)
    terms = []
    _embd = np.array([])
    i = 0
    for x in _fasttext.wv.vocab:
        if i == _max: break

        terms.append(x)
        _embd = np.append(_embd, _fasttext[x])
        i += 1

    _embd = _embd.reshape(_max, vect_size)
    _t = TSNE(n_components=2)
    generated_embds = _t.fit_transform(_embd)

    model_to_train = HAHNN()
    model_to_train.process_final_model(x_training, y_training, batch_size=64, epochs=8, embd_location=True, tokenizer_dir=SAVED_MODEL_DIR,
                                       base_name=SAVED_MODEL_FILENAME)
