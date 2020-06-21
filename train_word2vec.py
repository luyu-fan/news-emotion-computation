'''
@File: train_word2vec.py
@Author: Luyufan
@Date: 2020/6/14
@Desc: 利用gensim对爬取的所有新闻训练词向量（word2vec),考虑到翻译到中文与英文不同的准确性，采用翻译后的英文语料训练。
'''

from gensim.models import word2vec
import config
import multiprocessing
import stream_sentence_generator

class W2VModel(object):

    def __init__(self,model_path):
        self.__model_path = model_path
        self.__w2v_model = None

    def train_model(self,sentences,size,window,min_count,iter,sg,hs,workers):
        print("Staring training word2vec...")
        self.__w2v_model = word2vec.Word2Vec(sentences,
                                       size=size,
                                       window=window,
                                       min_count=min_count,
                                       iter=iter,
                                       sg=sg,
                                       hs=hs,
                                       workers=workers
                                       )
        print("Training finished,saving......")
        self.__w2v_model.save(self.__model_path)

    def load_model(self,model_path = None):
        if model_path is None:
            model_path = self.__model_path
        self.__w2v_model = word2vec.Word2Vec.load(model_path)

    def is_in_vocab(self,word):
        if word in self.__w2v_model.wv.vocab:
            return True
        else:
            return False

    def calc_similarity(self,word1,word2):
        return self.__w2v_model.wv.similarity(word1, word2)

    def get_word_embedding(self,word):
        return self.__w2v_model[word]

    def get_vocab_size(self):
        return len(self.__w2v_model.wv.vocab)

if __name__ == "__main__":

    # training word2ved based on gensim lib.
    sentences = stream_sentence_generator.StreamingSentenceGenenrator(config.WORD2VEC_CORPUS_DATA_ROOT)
    word2vec_model = W2VModel(config.WORD2VEC_MODEL_PATH)
    word2vec_model.train_model(sentences,
                            size=config.W2V_EMBEDDING_SIZE,
                            window=config.W2V_WINDOW,
                            min_count=config.W2V_MIN_COUNT,
                            iter=config.W2V_EPOCH,
                            sg=config.W2V_SKIP_GRAM,
                            hs=config.W2V_HS_LOSS,
                            workers=multiprocessing.cpu_count()
                               )
    # test
    word2vec_model = W2VModel(config.WORD2VEC_MODEL_PATH)
    word2vec_model.load_model()
    print(word2vec_model.get_vocab_size())
    print(word2vec_model.calc_similarity('worst', 'best'))
    print(word2vec_model.calc_similarity('beijing', 'shanghai'))
    print(word2vec_model.calc_similarity('china', 'usa'))
    print(word2vec_model.get_word_embedding("china"))

