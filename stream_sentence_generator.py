'''
@File: stream_sentence_generator.py
@Author: Luyufan
@Date: 2020/6/14
@Desc: 流式sentence (每一篇文章词列表的列表)生成器，将所有的excel文件中的新闻内容变成sentence可迭代对象
'''

import os,math
from warnings import warn
import xlrd
import utils.common_utils as utils

class StreamingSentenceGenenrator(object):

    def __init__(self,data_root,need_label = False,content_attribute = "english_content",label_attribute = "attitude_score"):
        """
        initialize the StreamingSentenceGenenrator
        :param data_root: raw corpus data
        :param need_label: should return label
        :param content_attribute: content
        :param label_attribute: label
        """
        self.data_root = data_root
        self.need_label = need_label
        self.content_attribute = content_attribute
        self.label_attribute = label_attribute

    def __iter__(self):
        """
        generator for corpus
        :return:
        """
        for root,dr,fs in os.walk(self.data_root):
            for file in fs:
                if file.endswith(".xlsx"):
                    file = os.path.join(root,file)
                    print("==> Processing",file)
                    try:
                        df = utils.load_data_from_excel(file)
                        self.corpus_size = df.shape[0]
                    except (OSError,xlrd.XLRDError) as e:
                        warn("==> Warning: can not read file: %s" % file)
                        continue
                    else:
                        for index in range(self.corpus_size):
                            content_label = None
                            try:
                                english_content = df.loc[index][self.content_attribute]
                                if self.need_label:
                                    content_label = df.loc[index][self.label_attribute]
                                    content_label = float(content_label) + 0.0
                                    if math.isnan(content_label):
                                        continue
                            except (AttributeError, KeyError):
                                    warn("==> Warning: file %s has not contained some attributes" % file)
                                    continue
                            else:
                                clean_word_list = utils.remove_stopwords(english_content)
                                if len(clean_word_list) > 0:
                                    yield (clean_word_list,content_label) if self.need_label else clean_word_list
                                else:
                                    continue

if __name__ == "__main__":

    # # testing my_corpus
    my_corpus = StreamingSentenceGenenrator("../data/word2vec_corpus/",True)
    for i,item in enumerate(my_corpus):
        print(item)


