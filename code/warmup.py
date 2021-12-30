from gensim.models import KeyedVectors
import os


def pair_res(wv: KeyedVectors):
    dic1 = {('cat', 'dog'): wv.similarity('cat', 'dog'),
            ('cat', 'Vienna'): wv.similarity('cat', 'Vienna'),
            ('Vienna', 'Austria'): wv.similarity('Vienna', 'Austria'),
            ('Austria', 'dog'): wv.similarity('Austria', 'dog')}

    return dic1


def ms_res(wv: KeyedVectors):
    dic1 = {'Vienna': wv.most_similar('Vienna', topn=3),
            'Austria': wv.most_similar('Austria', topn=3),
            'cat': wv.most_similar('cat', topn=3)}

    return dic1


def printable_pair(dic_w: dict):
    lis = []
    counter = 1
    for (w1, w2), value in dic_w.items():
        score = '{:.2f}'.format(value)
        output = f'pair {counter}: ("{w1}", "{w2}") -> similarity: {score}'
        lis.append(output)
        counter += 1

    save(os.path.join(os.getcwd(), '..', 'out'), lis, 'pair_similarities.txt')


def printable_ms(dic_w: dict):
    lis = []
    counter = 1
    for word, value in dic_w.items():
        for tup in value:
            score = '{:.2f}'.format(tup[1])
            output = f'word{counter}: "{word}" found "{tup[0]}" -> score: {score}'
            lis.append(output)

        counter += 1

    save(os.path.join(os.getcwd(), '..', 'out'), lis, 'ms_scores.txt')


def save(path, data: list, name: str):
    file_path = os.path.join(path, name)
    textfile = open(file_path, 'w')
    for element in data:
        textfile.write(element + '\n')
    textfile.close()


if __name__ == "__main__":
    word_vec = KeyedVectors.load_word2vec_format(os.path.join(os.getcwd(), '..', 'wiki-news-300d-1M-subword.vec'))
    dic = pair_res(word_vec)
    printable_pair(dic)
    dc = ms_res(word_vec)
    printable_ms(dc)

