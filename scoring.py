'''
@File: scoring.py
@Author: Luyufan
@Date: 2020/6/18
@Desc: 利用训练好的模型对没有标注的文件进行标注
'''
import os,torch,xlrd,warnings
from train_word2vec import W2VModel
import config
import utils.common_utils as utils
from train_regression import ProcessedDataSet,RegressionNetwork_TextCNN

device = torch.device("cpu")

if  __name__ == "__main__":

    # models
    w2v_model = W2VModel(config.WORD2VEC_MODEL_PATH)
    w2v_model.load_model()
    regression_model = torch.load(config.TEST_REG_MODEL).to(device)
    regression_model.eval()

    # valid model
    data_processor = ProcessedDataSet(config.PROCESSED_DATA_PATH,
                                      w2v_model,
                                      False,
                                      config.MAX_SEQ_LENGTH,
                                      config.W2V_EMBEDDING_SIZE,
                                      config.PAD_EMBEDDING,
                                      config.PAD_VALUE)

    print("Validation ... ...")
    error_list = []
    for i,data in enumerate(data_processor):
        embeddings,label = data
        features = torch.unsqueeze(torch.tensor(embeddings),0)
        score = regression_model(features).item()
        err = score - label
        error_list.append(err)
    utils.plot_2d_scatter(list(range(len(error_list))), error_list, title="Valid Error", xlabel="item index", ylabel="error")


    # scoring unlabeled data
    for root, dr, fs in os.walk(config.UNSCORED_FILE_FOLDER):
        for file in fs:
            if file.endswith(".xlsx"):
                file = os.path.join(root, file)
                print("==> Scoring", file)
                try:
                    df = utils.load_data_from_excel(file)
                    df["attitude_score"] = None
                    corpus_size = df.shape[0]
                except (OSError, xlrd.XLRDError) as e:
                    warnings.warn("==> Warning: can not read file: %s" % file)
                    continue
                else:
                    for index in range(corpus_size):
                        content_label = None
                        try:
                            english_content = df.loc[index,"english_content"]
                        except (AttributeError, KeyError):
                            warnings.warn("==> Warning: file %s has not contained some attributes" % file)
                            continue
                        else:
                            clean_word_list = utils.remove_stopwords(english_content)
                            if len(clean_word_list) > 0:
                                words_embeddings = data_processor.get_fed_embeddings(clean_word_list)
                                features = torch.unsqueeze(torch.tensor(words_embeddings), 0)
                                score = regression_model(features).item()
                                df.loc[index,"attitude_score"] = round(score,4)
                df.to_excel(file)