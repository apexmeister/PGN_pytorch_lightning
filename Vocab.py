import collections
import pandas as pd
from tqdm import tqdm

SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '<pad>'
UNKNOWN_TOKEN = '<unk>'  # OOV token
START_DECODING = '<bos>'  # decoding start
STOP_DECODING = '<eos>'  # decoding end


class Vocab(object):
    def __init__(self, vocab_file=None, vocab_size=50000):
        '''
            #build Vocab class with vocab file
            vocab_file content(word count):
                to 5751035
                a 5100555
                and 4892247
        '''
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0  # total words numbers in vocab class

        # <pad>, <unk>, <bos> and <eos> ids：0，1，2，3
        for w in [PAD_TOKEN, UNKNOWN_TOKEN, START_DECODING, STOP_DECODING]:
            self._word_to_id[w] = self._count
            self._id_to_word[self._count] = w
            self._count += 1


        # load vocab file
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                parts = line.split("\t")

                # checking
                if len(parts) != 2:
                    print('Warning: incorrectly formatted line in vocabulary file: {}'.format(line))
                    continue

                w = parts[0]
                # checking
                if w in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                    raise Exception(
                        '<s>, </s>, [UNK], [PAD], [START] and [STOP] shouldn\'t be in the vocab file, but %s is' % w)
                # checking
                if w in self._word_to_id:
                    raise Exception('Duplicated word in vocabulary file: %s' % w)

                # writing in
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                if vocab_size != 0 and self._count >= vocab_size:
                    print("max_size of vocab was specified as {}; we now have {} words. Stopping reading.".format(
                        vocab_size, self._count))
                    break

        print("Finished constructing vocabulary of {} total words. Last word added: {}".format(self._count,
                                                                                               self._id_to_word[
                                                                                                    self._count - 1]))
        print("Vocab:")
        for idx in [0, 1, 2, 3, 4, -1, self._count -3, self._count -2, self._count-1]:
            if idx == -1:
                print("    ......")
                continue
            print("   ", idx, self._id_to_word[idx])

    def word2id(self, word):
        '''get a word's id'''
        if word not in self._word_to_id:
            return self._word_to_id[UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id2word(self, word_id):
        '''decoder word with word id'''
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]

    def get_vocab_size(self):
        '''get vocab size'''
        return self._count


def article2ids(article_words, vocab):
    '''
    args：
        article(list[str])；
        class Vocab；

    return:
        ids：idx(list[int])；
        oovs：oovs(list[int])。
    '''
    article_ids = []
    article_oovs = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in article_words:
        i = vocab.word2id(w)
        if i == unk_id:  # i is oov?
            if w not in article_oovs:  # add i into oovs
                article_oovs.append(w)
            oov_num = article_oovs.index(w)  # give the first oov in article 0 id, and the second 1
            article_ids.append(
                vocab.get_vocab_size() + oov_num)  # to extend standard vocabulary size from 50000 to 50001 with oov
        else:
            article_ids.append(i)

    return article_ids, article_oovs

def abstract2ids(abstract_words, vocab, article_oovs):
    '''
    :param abstract_words: list[str]
    :param vocab: class Vocab
    :param article_oovs: list[int]
    :return: abstract_ids: list[int]
    '''

    abstract_ids = []
    unk_id = vocab.word2id(UNKNOWN_TOKEN)
    for w in abstract_words:
        i = vocab.word2id(w)
        if i == unk_id:
            if w in article_oovs:
                vocab_idx = vocab.get_vocab_size() + article_oovs.index(w)
                abstract_ids.append(vocab_idx)
            else:
                abstract_ids.append(unk_id)

        else:
            abstract_ids.append(i)
    return abstract_ids


def makeVocabdict(data_dir, vocab_dir, vocab_size):
    vocab_counter = collections.Counter()
    # TODO create vocab file assume load csv file
    df = pd.read_csv(data_dir)
    articles = df['articles'].tolist()
    titles = df['titles'].tolist()

    for art, tit, in tqdm(zip(articles, titles)):
        try:
            ''' #for CNN/DM 
            art_tokens = art.lower().strip().split(' ')
            tit_tokens = tit.lower().strip().split(' ')
            '''
            #for SAMsum
            art_ = art.strip().split('\n')
            art_ = ' '.join(art_)
            art_tokens = art_.strip().split(' ')
            tit_tokens = tit.strip().split(' ')

            # tit_tokens = [t for t in tit_tokens if
            #               t not in [SENTENCE_START, SENTENCE_END]]  # remove special characters
            tokens = art_tokens + tit_tokens
            tokens = [t.strip() for t in tokens]
            tokens = [t for t in tokens if t != ""]

            for token in tokens:
                if token in vocab_counter:
                    continue
                else:
                    vocab_counter[token] += 1
            # vocab_counter.update(tokens)
        except:
            print(art)
            print(tit)

    with open(vocab_dir, 'w', encoding='utf-8') as writer:
        for word, count in vocab_counter.most_common(vocab_size):
            if word in [SENTENCE_START, SENTENCE_END, UNKNOWN_TOKEN, PAD_TOKEN, START_DECODING, STOP_DECODING]:
                continue
            else:
                writer.write(word + '\t' + str(count) + '\n')

    print("Finished writing vocab file")


if __name__ == '__main__':
    data_path = "train.csv"
    vocab_path = "vocab"
    makeVocabdict(data_path, vocab_path, 60000)
    vocab = Vocab(vocab_path, 50000)

    print(vocab.get_vocab_size())
    #
