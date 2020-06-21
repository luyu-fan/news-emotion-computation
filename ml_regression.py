'''
@File: ml_regression.py
@Author: Luyufan
@Date: 2020/6/17
@Desc: 神经网络一直训练不好，参数调了很多都不起作用，尝试采用现有的那些机器学习包来做回归进行对比
       这种方法无论是否进行了降维都不如神经网络，如果没有做降维速度非常慢，但如果做了降维，效果并
       没有得到提升，而且无法做到在线降维，考虑到标注的在线性，并不能用于实际任务。
'''
import numpy as np
import matplotlib.pyplot as plt

from train_word2vec import W2VModel
from train_regression import ProcessedDataSet
from utils.common_utils import plot_2d_scatter,plot_3d_scatter
import config

def get_dataset(data_iter,should_decomposition = False,use_PCA = False):
    features = [0 for _ in range(len(data_iter))]
    labels = [0 for _ in range(len(data_iter))]
    for i, data in enumerate(data_iter):
        embedding_feature, label = data
        embedding_feature = np.array(embedding_feature)
        features[i] = embedding_feature.reshape(1, -1)[0]
        labels[i] = label
    features = np.array(features)
    if should_decomposition:
        if use_PCA:
            print("PCA decomposition:")
            from sklearn.decomposition import PCA
            pca_dec = PCA(n_components=2)
            decom_features = pca_dec.fit_transform(features)
            features = decom_features
        else:
            print("t-SNE decomposition:")
            from sklearn.manifold import TSNE
            tsne_dec = TSNE(n_components=2,learning_rate=400)
            decom_features = tsne_dec.fit_transform(features)
            features = decom_features
    labels = np.array(labels)
    return features,labels

def fit_model(train_features,train_labels,valid_features,valid_labels,model,desc):
    print("--------------------------Model:%s--------------------------" % desc)
    model.fit(train_features,train_labels)
    model_score = model.score(valid_features,valid_labels)
    result = model.predict(valid_features)
    plot_2d_scatter(np.arange(len(result)), valid_labels - result,
                    title=f"Model:{desc}---score:{model_score}",xlabel="item index",ylabel="distance error")

if __name__ == "__main__":
    # 模型准备
    # get train set and test set
    w2v_model = W2VModel(config.WORD2VEC_MODEL_PATH)
    w2v_model.load_model()
    train_dataset = ProcessedDataSet(config.PROCESSED_DATA_PATH,
                                     w2v_model, train=True,
                                     max_seq=config.MAX_SEQ_LENGTH,
                                     embedding_size=config.W2V_EMBEDDING_SIZE,
                                     pad=config.PAD_EMBEDDING,
                                     pad_value=config.PAD_VALUE)
    valid_dataset = ProcessedDataSet(config.PROCESSED_DATA_PATH,
                                     w2v_model,
                                     train=False,
                                     max_seq=config.MAX_SEQ_LENGTH,
                                     embedding_size=config.W2V_EMBEDDING_SIZE,
                                     pad=config.PAD_EMBEDDING,
                                     pad_value=config.PAD_VALUE)

    # 预处理
    train_feature, train_label = get_dataset(train_dataset, should_decomposition=True, use_PCA=True)
    valid_feature, valid_label = get_dataset(valid_dataset, should_decomposition=True, use_PCA=True)
    np.save("t_features.npy", train_feature)
    np.save("t_labels.npy", train_label)
    np.save("v_features.npy", valid_feature)
    np.save("v_labels.npy", valid_label)
    train_feature = np.load("t_features.npy")
    train_label = np.load("t_labels.npy")
    valid_feature = np.load("v_features.npy")
    valid_label = np.load("v_labels.npy")
    print(train_feature.shape, train_label.shape)
    print(valid_feature.shape, valid_label.shape)
    plot_3d_scatter(train_feature[:, 0], train_feature[:, 1], train_label,
                    title= "Train Decomposition",xlabel="fea1",ylabel="fea2",zlabel="attitude score")
    plot_3d_scatter(valid_feature[:, 0], valid_feature[:, 1], valid_label,
                    title="Valid Decomposition", xlabel="fea1", ylabel="fea2", zlabel="attitude score")

    # 方法选择
    # 1.决策树回归
    from sklearn import tree
    model_decision_tree_regression = tree.DecisionTreeRegressor()
    fit_model(train_feature,train_label,valid_feature,valid_label,model_decision_tree_regression,"DecisionTreeRegressor")

    # # 2.线性回归
    from sklearn.linear_model import LinearRegression
    model_linear_regression = LinearRegression()
    fit_model(train_feature,train_label,valid_feature,valid_label,model_linear_regression,"LinearRegression")

    # # 3.SVM回归
    from sklearn import svm
    model_svm = svm.SVR()
    fit_model(train_feature,train_label,valid_feature,valid_label,model_svm,"SVR")

    # # 4.kNN回归
    from sklearn import neighbors
    model_k_neighbor = neighbors.KNeighborsRegressor()
    fit_model(train_feature,train_label,valid_feature,valid_label,model_k_neighbor,"KNeighborsRegressor")

    # # 5.随机森林回归
    from sklearn import ensemble
    model_random_forest_regressor = ensemble.RandomForestRegressor(n_estimators=20)  # 使用20个决策树
    fit_model(train_feature,train_label,valid_feature,valid_label,model_random_forest_regressor,"RandomForestRegressor20")

    # # 6.Adaboost回归
    from sklearn import ensemble
    model_adaboost_regressor = ensemble.AdaBoostRegressor(n_estimators=50)  # 这里使用50个决策树
    fit_model(train_feature,train_label,valid_feature,valid_label,model_adaboost_regressor,"AdaBoostRegressor50")

    # # 7.GBRT回归
    from sklearn import ensemble
    model_gradient_boosting_regressor = ensemble.GradientBoostingRegressor(n_estimators=100)  # 这里使用100个决策树
    fit_model(train_feature,train_label,valid_feature,valid_label,model_gradient_boosting_regressor,"GradientBoostingRegressor100")

    # # 8.Bagging回归
    from sklearn import ensemble
    model_bagging_regressor = ensemble.BaggingRegressor()
    fit_model(train_feature,train_label,valid_feature,valid_label,model_bagging_regressor,"BaggingRegressor")

    # # 9.ExtraTree极端随机数回归
    from sklearn.tree import ExtraTreeRegressor
    model_extra_tree_regressor = ExtraTreeRegressor()
    fit_model(train_feature,train_label,valid_feature,valid_label,model_extra_tree_regressor,"ExtraTreeRegressor")
