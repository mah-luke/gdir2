import glob
import os
import pandas as pd
import gensim
from gensim.parsing.preprocessing import *
import gzip
from nltk.corpus import stopwords
import string
from gensim.models import KeyedVectors


def load_data(path: str) -> pd.DataFrame:
    all_files = glob.glob(os.path.join(path, '*.gz'))
    data = pd.concat(pd.read_json(f, 'r') for f in (gzip.open(a, 'r') for a in all_files))
    # with gzip.open(all_files[0], 'r') as f:
    #     data = pd.read_json(f, 'r')
    return data


def preprocess_data_and_train(data: pd.DataFrame):
    FILTERS = [strip_tags,
               strip_punctuation, strip_multiple_whitespaces, strip_numeric, strip_short]
    texts_tokenized = [preprocess_string(text.lower(), FILTERS) for text in data.text.astype(str)]
    stop_words = stopwords.words('german')
    texts_tokenized = [text for text in texts_tokenized if text not in stop_words]

    german_model = gensim.models.Word2Vec(sentences=texts_tokenized, vector_size=100, window=5,
                                          min_count=1, epochs=10, workers=2)
    german_model.build_vocab(texts_tokenized, progress_per=1000)
    german_model.train(texts_tokenized, total_examples=german_model.corpus_count, epochs=german_model.epochs)
    german_model.wv.save_word2vec_format('../german-tweet-model.vec')

    return german_model.wv


def printable_ms(dic_w: dict):
    lis = []
    counter = 1
    for word, value in dic_w.items():
        for tup in value:
            score = '{:.2f}'.format(tup[1])
            output = f'word{counter}: "{word}" found "{tup[0]}" -> score: {score}'
            lis.append(output)

        counter += 1

    save(os.path.join(os.getcwd(), '..', 'out'), lis, 'ms_scores_part3.txt')


def save(path, data: list, name: str):
    file_path = os.path.join(path, name)
    textfile = open(file_path, 'w')
    for element in data:
        textfile.write(element + '\n')
    textfile.close()


if __name__ == "__main__":
    df = load_data('../german-tweet-sample-2019-08')
    preprocess_data_and_train(df)
    model = preprocess_data_and_train(df)

    # uncomment all lines above and comment only the next line to retrain model
    # model = KeyedVectors.load_word2vec_format(os.path.join(os.getcwd(), '..', 'german-tweet-model.vec'))
    dic1 = {'spiel': model.most_similar('spiel', topn=3),
            'polizei': model.most_similar('polizei', topn=3),
            'berlin': model.most_similar('berlin', topn=3)}
    printable_ms(dic1)
    print('Done! ---------------')

