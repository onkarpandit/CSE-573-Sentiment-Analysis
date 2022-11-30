import pickle
import random
from textwrap import wrap

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from numpy.random import seed
from sklearn.metrics import confusion_matrix

from AttentionLayer import Attention as Att
from dataLoadUtilities import *

tf.compat.v1.disable_eager_execution()

nltk.download('punkt')
nltk.download('stopwords')

os.environ['PYTHONHASHSEED'] = str(1024)
tf.random.set_seed(1024)
seed(1024)
np.random.seed(1024)
random.seed(1024)

graph = tf.compat.v1.get_default_graph()


class TestModel:
    def __init__(self):
        self.model = None
        self.LARGEST_SEN_SIZE = 0
        self.LARGEST_SEN_ENUM = 0
        self.LENGTH_OF_VOCAB = 0
        self.term_embd = None
        self.model = None
        self.term_att_final = None
        self.term_splitter = None
        self.no_of_labels = 2

    def retrieve_term_splitter_f_title(self, title):
        return title + '.tokenizer'

    def make_opposite_term_position(self):
        self.reverse_word_index = {value: position for position, value in self.term_splitter.word_index.items()}

    def enc_content(self, content):
        out = np.zeros((len(content), self.LARGEST_SEN_ENUM, self.LARGEST_SEN_SIZE))
        for position, c in enumerate(content):
            single_out = np.array(pad_sequences(
                self.term_splitter.texts_to_sequences(c),
                maxlen=self.LARGEST_SEN_SIZE))[:self.LARGEST_SEN_ENUM]
            out[position][-len(single_out):] = single_out
        return out

    def enc_in(self, _in):
        _in = np.array(_in)
        if not _in.shape:
            _in = np.expand_dims(_in, 0)
        content = np.array([scale_text(t) for t in _in])
        return self.enc_content(content)

    def predict(self, _in):
        enc_in = self.enc_content(_in)
        return self.model.predict(enc_in)

    def retrieve_word_weights(self):
        return self.retrieve_layer('word_attention').get_weights()[0]

    def get_attention_weights(self, _in_txt):
        scaled_txt = scale_text(_in_txt)
        enc_txt = self.enc_in(_in_txt)[0]
        word_attention = enc_txt
        word_attention = word_attention.astype(float)

        enc_txt_tight = enc_txt[-len(scaled_txt):]
        enc_txt_tight = [list(filter(lambda x: x > 0, sentence)) for sentence in enc_txt_tight]
        refined_text = [[self.reverse_word_index[int(pos)]
                                for pos in x] for x in enc_txt_tight]
        word_attention_tight = word_attention[-len(scaled_txt):]
        word_attention_tight = word_attention_tight / np.expand_dims(np.sum(word_attention_tight, -1), -1)
        word_attention_tight = np.array([att[-len(s):]
                                     for att, s in zip(word_attention_tight, enc_txt_tight)])
        _attention_weights = []
        for pos, _in_txt in enumerate(refined_text):
            _attention_weights.append(list(zip(_in_txt, word_attention_tight[pos])))

        sen_enc_ = Model(inputs=self.model.input,
                                             outputs=self.model.get_layer('dense_transform_sentence').output)
        sen_encs_ = np.squeeze(
            sen_enc_.predict(np.expand_dims(enc_txt, 0)), 0)
        sen_background = self.retrieve_layer('sentence_attention').get_weights()[0]
        sen_att = np.exp(np.squeeze(np.dot(sen_encs_, sen_background), -1))
        sen_att = sen_att.astype(float)
        sen_att_tight = sen_att[-len(scaled_txt):]
        sen_att_tight = sen_att_tight / np.expand_dims(np.sum(sen_att_tight, -1), -1)
        final_weights = list(zip(_attention_weights, sen_att_tight))
        return final_weights

    def retrieve_layer(self, layer_name):
        return self.model.get_layer(layer_name)

    def retrieve_term_splitter_location(self, base_folder, title):
        return os.path.join(
            base_folder, self.retrieve_term_splitter_f_title(title))

    def retrieve_params_from_saved_model(self, m_folder, f_title):
        with CustomObjectScope({'Attention': Att}):
            print(os.path.join(m_folder, f_title))
            self.model = load_model(os.path.join(m_folder, f_title))
            self.term_att_final = self.retrieve_layer('time_distributed').layer
            term_splitter_location = self.retrieve_term_splitter_location(m_folder, f_title)
            _term_splitter = pickle.load(open(term_splitter_location, "rb"))
            self.term_splitter = _term_splitter['tokenizer']
            self.LARGEST_SEN_ENUM = _term_splitter['maxSentenceCount']
            self.LARGEST_SEN_SIZE = _term_splitter['maxSentenceLength']
            self.LENGTH_OF_VOCAB = _term_splitter['vocabularySize']
            self.make_opposite_term_position()


def build_weight_matrix(activation_maps):
    _words_weights = []
    _words = []
    _sentences_weights = []
    _sentences = []
    mx_len = 0
    for i in range(len(activation_maps)):
        mx_len = max(mx_len, len(activation_maps[i][0]))

    mx_len /= 2

    row_len = 13

    for i in range(len(activation_maps)):
        l = len(activation_maps[i][0])

        _sentences_weights.append([activation_maps[i][1]])

        _sentence = ""
        for pr in activation_maps[i][0]:
            _sentence = _sentence + pr[0] + " "
        _sentences.append(["\n".join(wrap(_sentence, 150))])

        for j in range(0, l, row_len):
            arr = [] * row_len
            labels = [] * row_len
            if j + row_len <= l:
                for k in range(j, j + row_len):
                    arr.append(activation_maps[i][0][k][1])
                    labels.append(activation_maps[i][0][k][0])
            else:
                for k in range(j, l):
                    arr.append(activation_maps[i][0][k][1])
                    labels.append(activation_maps[i][0][k][0])
                for k in range(l, j + row_len):
                    arr.append(0)
                    labels.append("")
            _words_weights.append(np.array(arr))
            _words.append(np.array(labels))

    _words_weights = np.array(_words_weights)
    _words = np.array(_words)
    _sentences = np.array(_sentences)
    _sentences_weights = np.array(_sentences_weights)
    return _words_weights, _words, _sentences_weights, _sentences


def count_word_frequencies(model, data):
    dic = {}
    with graph.as_default():
        for text in data:
            activation_maps = model.get_attention_weights(text, websafe=True)
            for i in range(len(activation_maps)):
                for pr in activation_maps[i][0]:
                    if pr[0] not in dic:
                        dic[pr[0]] = [pr[1]]
                    else:
                        dic[pr[0]].append(pr[1])
    return 1


def predict(x_test):
    model = TestModel()
    model.retrieve_params_from_saved_model('../saved_models', './model.tf')
    y_test = list()
    with graph.as_default():
        for i in tqdm(range(len(x_test))):
            text = x_test[i]
            ntext = scale_text(text)
            preds = model.predict([ntext])[0]
            prediction = np.argmax(preds).astype(float)
            y_test.append('neg' if prediction == 0 else 'pos')
    return y_test


def generate_att_weights_matrix(model, text, base_folder, index):
    with graph.as_default():
        attention_weights = model.get_attention_weights(text)
        _words_weights, _words, _sentences_weights, _sentences = build_weight_matrix(attention_weights)

        fig, ax = plt.subplots(figsize=(23, 10))
        sns.set(font_scale=1.2)
        ax = sns.heatmap(_words_weights, annot=_words, fmt="", cmap="Reds", linewidths=.5, ax=ax)
        plt.tight_layout()
        plt.savefig(f"../{base_folder}/word{index}.png", pad_inches=0)

        fig, ax = plt.subplots(figsize=(23, 10))
        ax = sns.heatmap(_sentences_weights, annot=_sentences, fmt="", cmap="Reds", linewidths=.5, ax=ax)
        plt.tight_layout()
        plt.savefig(f"../{base_folder}/sentence{index}.png", pad_inches=0)


def retrieve_test_data():
    df = pd.read_csv("../imdb_reviews.csv")
    y_truth = df['sentiment'].to_list()
    y_truth = [0 if x == 'neg' else 1 for x in y_truth]
    x_test = df['text'].to_list()
    return x_test, y_truth


def retrieve_pre_tested_data(x_test, y_truth):
    y_test = np.load("predicted.npy", allow_pickle=True)
    y_test = [0 if x == 'neg' else 1 for x in y_test]
    false_negative = []
    false_positive = []
    true_positive = []
    true_negative = []
    for i in range(len(y_test)):
        if y_truth[i] == 1 and y_test[i] == 0:
            false_negative.append(x_test[i])
        elif y_truth[i] == 0 and y_test[i] == 1:
            false_positive.append(x_test[i])
        elif y_truth[i] == 1 and y_test[i] == 1:
            true_positive.append(x_test[i])
        else:
            true_negative.append(x_test[i])
    return y_test, true_positive, true_negative, false_positive, false_negative


def generate_confusion_matrix(y_truth, y_test):
    return confusion_matrix(y_truth, y_test)


def generate_att_weights_matrix_category_wise(true_positive, true_negative, false_positive, false_negative):
    model = TestModel()
    model.retrieve_params_from_saved_model('../saved_models', './model.tf')

    for i in tqdm(range(31, 50)):
        generate_att_weights_matrix(model, false_positive[i], "false_positive", i)
        generate_att_weights_matrix(model, false_negative[i], "false_negative", i)
        generate_att_weights_matrix(model, true_positive[i], "true_positive", i)
        generate_att_weights_matrix(model, true_negative[i], "true_negative", i)


def main():
    x_test, y_truth = retrieve_test_data()
    y_test = predict(x_test[0:1])
    print(f"Input: {x_test[0:1]}")
    print(f"Output: {y_test}")


if __name__ == '__main__':
    main()
