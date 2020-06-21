'''
@File: data_generator.py
@Author: Luyufan
@Date: 2020/6/14
@Desc: 利用原始文本得到适合训练的索引文件
'''

import stream_sentence_generator
import config
from train_word2vec import W2VModel


class DATAGenerator(object):

    def __init__(self,sentences,word2vec_model,save_path):
        self.__w2v_model = word2vec_model
        self.__sentences = sentences
        self.__save_path = save_path

    def generate_dataset_table(self):
        """

        :return:
        """
        file = open(self.__save_path, "w+", encoding="utf-8")
        file.write("word_list|target_score\n")
        word_length_list = []
        for words_list,label in self.__sentences:
            valid_words_list = list(filter(self.__w2v_model.is_in_vocab,words_list))
            word_length_list.append(len(valid_words_list))
            saved_str = str(valid_words_list)+"|"+str(label) + "\n"
            try:
                file.write(saved_str)
                file.flush()
            except IOError as e:
                file.close()
                print("==> can not save for some reasons:",e)
                raise e
        file.close()
        self.display(word_length_list)

    def display(self,length_list):
        length_list = sorted(length_list)
        import matplotlib.pyplot as plt
        plt.figure(dpi=120)
        plt.scatter(list(range(len(length_list))), length_list)
        plt.xticks(rotation=45)
        plt.title("Statistic")
        plt.xlabel("News Index")
        plt.ylabel("Words List Length")
        plt.show()

if __name__ == "__main__":

    stream_sentence_generator = stream_sentence_generator.StreamingSentenceGenenrator(config.WORD2VEC_CORPUS_DATA_ROOT, need_label=True)
    w2v_model = W2VModel(config.WORD2VEC_MODEL_PATH)
    w2v_model.load_model()

    train_data_generator = DATAGenerator(stream_sentence_generator, w2v_model, config.PROCESSED_DATA_PATH)
    train_data_generator.generate_dataset_table()
