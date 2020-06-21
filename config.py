'''
@File: config.py
@Author: Luyufan
@Date: 2020/6/14
@Desc: 基础配置
'''

"""1. word2vec phrase"""
# word2vec training
WORD2VEC_CORPUS_DATA_ROOT = "./data/word2vec_corpus/"
WORD2VEC_MODEL_PATH = './models/word2vec.model'
WORD2VEC_STOPWORDS = "./data/word2vec_corpus/english_stopwords.txt"

W2V_EMBEDDING_SIZE = 64
W2V_WINDOW = 5
W2V_MIN_COUNT = 10
W2V_EPOCH = 200
W2V_SKIP_GRAM = 1
W2V_HS_LOSS = 1

"""2. processed data phrase"""
PROCESSED_DATA_PATH = "./data/processed_data/processed_data.csv"


"""3. regression phrase"""
PAD_EMBEDDING = True
PAD_VALUE = 0
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 800    # 800 words for each news
LR = 0.002
GAMMA = 0.98
EPOCH = 360
WD = 1e-4
REGRESSION_MODEL_ROOT = './models/'
SUM_WRITER_ROOT = "./viz_loss/"

"""4. scoring phrase"""
TEST_REG_MODEL = './models/regression_356_model.pth'
UNSCORED_FILE_FOLDER = "./data/unlabel_corpus"


