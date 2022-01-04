import csv
import os
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import string
from gensim.models import KeyedVectors
import numpy as np


def preprocess_arr():
    tsv_file = open(os.path.join(os.getcwd(), '..', 'dataset.tsv'))
    read_tsv = csv.reader(tsv_file, delimiter='\t')
    stop_words = stopwords.words('english')
    lis = []

    for text in read_tsv:
        text[1] = text[1].lower()
        text[1] = "".join([char for char in text[1] if char not in string.punctuation])
        text[1] = word_tokenize(text[1])

        # comment the next line to include stopwords
        #text[1] = [word for word in text[1] if word not in stop_words]

        text[2] = text[2].lower()
        text[2] = "".join([char for char in text[2] if char not in string.punctuation])
        text[2] = word_tokenize(text[2])

        # comment the next line to include stopwords
        #text[2] = [word for word in text[2] if word not in stop_words]

        lis.append(text)

    return lis


def dummy_fun(doc):
    return doc


def vector_space_model(preprocessed_list: list):
    tfidf = TfidfVectorizer(analyzer='word',
                            tokenizer=dummy_fun,
                            preprocessor=dummy_fun,
                            token_pattern=None)
    vectors = tfidf.fit_transform([preprocessed_list[1], preprocessed_list[2]])
    return float(preprocessed_list[0]), cosine_similarity(vectors[0:1], vectors)[0][1]


def short_text_rep_wvm(wv: KeyedVectors, preprocessed_list: list):
    sen_1_words = [w for w in preprocessed_list[1] if w in wv]
    sen_2_words = [w for w in preprocessed_list[2] if w in wv]
    dic_mean = {}
    bow = set()

    for word in sen_1_words:
        bow.add(word)

    for word in sen_2_words:
        bow.add(word)

    for word in bow:
        found = wv.get_vector(word, norm=True)
        if word in sen_1_words and word in sen_2_words:
            dic_mean[word] = (np.mean(found), np.mean(found))
        if word in sen_1_words and word not in sen_2_words:
            dic_mean[word] = (np.mean(found), 0)
        if word not in sen_1_words and word in sen_2_words:
            dic_mean[word] = (0, np.mean(found))

    vec_short_mean_sen1 = []
    vec_short_mean_sen2 = []

    for word, tup in dic_mean.items():
        vec_short_mean_sen1.append(tup[0])
        vec_short_mean_sen2.append(tup[1])

    return cosine_similarity([vec_short_mean_sen1, vec_short_mean_sen2])[0][1]


def short_text_widf(wv: KeyedVectors, preprocessed_list: list):
    sen_1_words = [w for w in preprocessed_list[1] if w in wv]
    sen_2_words = [w for w in preprocessed_list[2] if w in wv]
    dic_mean = {}
    bow = set()
    tfidf = TfidfVectorizer(analyzer='word',
                            tokenizer=dummy_fun,
                            preprocessor=dummy_fun,
                            token_pattern=None)
    vectors = tfidf.fit_transform([preprocessed_list[1], preprocessed_list[2]])
    idf = tfidf.idf_
    idf_dic = dict(zip(tfidf.get_feature_names_out(), idf))

    for word in sen_1_words:
        bow.add(word)

    for word in sen_2_words:
        bow.add(word)

    for word in bow:
        filled = np.zeros(len(wv[word])).fill(idf_dic[word])
        found = wv.get_vector(word, norm=True)
        if word in sen_1_words and word in sen_2_words:
            dic_mean[word] = (np.average(found, weights=filled), np.average(found, weights=filled))
        if word in sen_1_words and word not in sen_2_words:
            dic_mean[word] = (np.average(found, weights=filled), 0)
        if word not in sen_1_words and word in sen_2_words:
            dic_mean[word] = (0, np.average(found, weights=filled))

    vec_short_mean_sen1 = []
    vec_short_mean_sen2 = []

    for word, tup in dic_mean.items():
        vec_short_mean_sen1.append(tup[0])
        vec_short_mean_sen2.append(tup[1])

    return cosine_similarity([vec_short_mean_sen1, vec_short_mean_sen2])[0][1]


def printable_res(corrlist: list):
    lis_str = [f"For TfIdfVectorizer Method the correlation was {'{:.2f}'.format(corrlist[0])}",
               f"For Mean Averaging Method the correlation was {'{:.2f}'.format(corrlist[1])}",
               f"For Weighted Averaging Method the correlation was {'{:.2f}'.format(corrlist[2])}"]

    # change the name of the file to save different files
    save(os.path.join(os.getcwd(), '..', 'out'), lis_str, 'correlations_with_stopwords.txt')


def save(path, data: list, name: str):
    file_path = os.path.join(path, name)
    textfile = open(file_path, 'w')
    for element in data:
        textfile.write(element + '\n')
    textfile.close()


if __name__ == "__main__":
    word_vec = KeyedVectors.load_word2vec_format(os.path.join(os.getcwd(), '..', 'wiki-news-300d-1M-subword.vec'))
    ru = preprocess_arr()
    vec_sm_cos_sim = []
    vec_sm_gt = []
    pearson_corr = []
    for row in ru:
        cos_sim = vector_space_model(row)
        vec_sm_cos_sim.append(cos_sim[1])
        vec_sm_gt.append(cos_sim[0])

    pearson_corr.append(pearsonr(vec_sm_cos_sim, vec_sm_gt)[0])

    vec_lm_mean = []
    for row in ru:
        vec_lm_mean.append(short_text_rep_wvm(word_vec, row))

    pearson_corr.append(pearsonr(vec_lm_mean, vec_sm_gt)[0])

    vec_lm_wm = []
    for row in ru:
        vec_lm_wm.append(short_text_widf(word_vec, row))

    pearson_corr.append(pearsonr(vec_lm_wm, vec_sm_gt)[0])
    printable_res(pearson_corr)

