'''
@File: train_regression.py
@Author: Luyufan
@Date: 2020/6/16
@Desc: 训练网络回归计算情感值
'''
import pandas as pd,numpy as np,os
import torch as torch
import torch.nn as nn
import torch.optim as optim
import config
import torch.utils.tensorboard as tensorboard
from torch.utils.data import DataLoader,Dataset
from train_word2vec import W2VModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProcessedDataSet(Dataset):

    def __init__(self,processed_data_path,w2v_model,train,max_seq,embedding_size,pad = False, pad_value = 0):
        super(Dataset,self).__init__()
        self.__train = train
        self.__all_data = None
        self.__w2v_model = w2v_model
        self.__pad = pad
        self.__pad_value = pad_value
        self.__max_seq = max_seq
        self.__embedding_size = embedding_size
        try:
            self.__all_data = pd.read_csv(processed_data_path,sep="|")
        except (OSError,IOError) as e:
            print("==> can not load data file", processed_data_path)
        else:
            # split the dataset
            self.__total_index = range(self.__all_data.shape[0])
            self.__test_index = [x for x in self.__total_index if x % 4 == 0]
            self.__train_index = [x for x in self.__total_index if x % 4 != 0]

    def __len__(self):
        if self.__train:
            return len(self.__train_index)
        else:
            return len(self.__test_index)

    def __getitem__(self, index):
        if self.__train:
            tmp_index = self.__train_index
        else:
            tmp_index = self.__test_index
        words_list = eval(self.__all_data.iloc[tmp_index[index]][0])
        label = int(self.__all_data.iloc[tmp_index[index]][1])
        return self.get_fed_embeddings(words_list),label

    def get_fed_embeddings(self,words_list):
        words_list = list(filter(self.__w2v_model.is_in_vocab, words_list))
        valid_length = min(self.__max_seq, len(words_list))
        embeddings = [0 for i in range(valid_length)]
        for i in range(valid_length):
            embeddings[i] = self.__w2v_model.get_word_embedding(words_list[i])
        if self.__pad:
            pad_embeddings = np.ones(shape=(self.__max_seq, self.__embedding_size), dtype=np.float32)
            pad_embeddings *= self.__pad_value
            pad_embeddings[:valid_length] = np.array(embeddings)
            return pad_embeddings
        else:
            return np.array(embeddings),

class RegressionNetwork_RNN(nn.Module):
    """basic rnn regression"""
    def __init__(self,net,layers =2,input_size = 64, hidden_size = 64):
        super(RegressionNetwork_RNN, self).__init__()
        assert layers > 0
        assert issubclass(net,nn.RNNBase)

        self.__rnn_net = net(input_size=input_size,hidden_size=hidden_size,num_layers=layers,bias=False,batch_first=True)
        self.__dropout = nn.Dropout()
        self.__fc = nn.Linear(in_features=hidden_size,out_features=1,bias=False)

    def forward(self,embeddings):
        x,_ = self.__rnn_net(embeddings)
        # output shape:
        # batch * seq_length * hidden_size
        x = x[:,-1,:]   # only last layer
        regression_value = self.__fc(self.__dropout(x))
        return regression_value

class RegressionNetwork_TextCNN(nn.Module):
    """TextCNN for regression"""
    def __init__(self,embedding_size = 64,kernel_size_list = None,channels = None):
        super(RegressionNetwork_TextCNN,self).__init__()
        assert isinstance(kernel_size_list,(list,tuple))
        assert isinstance(channels,(list,tuple))
        assert len(kernel_size_list) == len(channels)

        self.__channels = channels

        self.__convs = nn.ModuleList(
            [nn.Conv2d(in_channels=1,
                       out_channels=out_c,
                       kernel_size=(k,embedding_size),
                       padding=((k-1)//2,0),
                       bias=False,
                       stride=(1,1)) for k,out_c in zip(kernel_size_list,channels)
             ]
        )

        self.__pool = nn.AdaptiveMaxPool2d((1,1))
        self.__tanh = nn.Tanh()
        self.__dropout = nn.Dropout()
        self.__pre_fc = nn.Linear(sum(channels),out_features=64,bias=False)
        self.__fc = nn.Linear(64,out_features=1, bias=False)

    def forward(self, embeddings):
        batch = embeddings.size()[0]
        embeddings = torch.unsqueeze(embeddings,dim=1)
        convd_features = [conv(embeddings) for conv in self.__convs]
        act_features = [self.__tanh(features) for features in convd_features]
        pooled_features =[self.__pool(features) for features in act_features]  # batch * channels * 1 * 1
        pooled_features = [features.view(batch,channel) for features,channel in zip(pooled_features,self.__channels)]
        feature_vector = torch.cat(pooled_features,dim=1)
        feature_vector = self.__tanh(self.__pre_fc(self.__dropout(feature_vector)))
        reg_value = self.__fc(feature_vector)
        return reg_value

def train():

    # prepare
    w2v_model = W2VModel(config.WORD2VEC_MODEL_PATH)
    w2v_model.load_model()

    bacth_size = config.BATCH_SIZE if config.PAD_EMBEDDING else 1
    train_dataset = ProcessedDataSet(config.PROCESSED_DATA_PATH,
                                     w2v_model,train=True,
                                     max_seq=config.MAX_SEQ_LENGTH,
                                     embedding_size=config.W2V_EMBEDDING_SIZE,
                                     pad=config.PAD_EMBEDDING,
                                     pad_value = config.PAD_VALUE)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=bacth_size,
                                  shuffle=True,
                                  num_workers=0,
                                  drop_last=False)
    valid_dataset = ProcessedDataSet(config.PROCESSED_DATA_PATH,
                                  w2v_model,
                                  max_seq=config.MAX_SEQ_LENGTH,
                                  embedding_size=config.W2V_EMBEDDING_SIZE,
                                  pad=config.PAD_EMBEDDING,
                                  train=False,
                                  pad_value = config.PAD_VALUE
                                     )
    valid_dataloader = DataLoader(valid_dataset,
                                  batch_size=bacth_size,
                                  shuffle=False,
                                  num_workers=0,
                                  drop_last=False)

    # define model

    # region Using RNN Model
    # rnn_net = nn.LSTM
    # regression_net = RegressionNetwork_RNN(rnn_net).to(device)
    # endregion

    # region Using CNN Model
    regression_net = RegressionNetwork_TextCNN(embedding_size=64,kernel_size_list=(3,5,),channels=(128,64,)).to(device)
    # endregion

    # criterion
    mse_criterion = nn.MSELoss()

    # optimizer
    optimizer = optim.SGD(regression_net.parameters(),lr=config.LR,weight_decay=config.WD)

    # lr_scheduler
    lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer,gamma=config.GAMMA)

    # summary
    summary = tensorboard.SummaryWriter(log_dir=config.SUM_WRITER_ROOT,comment="Regression Loss Visualization",flush_secs=6)

    # running
    global_step = 0
    print("-" * 120 +"\n"+"Start training:")
    print(regression_net)
    for epoch in range(config.EPOCH):
        # region Training
        total_train_loss = 0
        tmp_step = 0
        regression_net.train()
        for i,data in enumerate(train_dataloader):
            optimizer.zero_grad()
            global_step += 1
            tmp_step += 1

            embeddings,target_score = data
            embeddings = embeddings.to(device)
            target_score = target_score.to(device)

            predict_score = regression_net(embeddings)
            loss = mse_criterion(predict_score.float().view(-1,1),target_score.float().view(-1,1))
            total_train_loss += loss

            loss.backward()
            optimizer.step()

            if tmp_step % 10 == 0:
                print("Epoch: %d, Global step:%d,Training Average MSE Loss:%.4f" % (epoch,global_step,total_train_loss.item() / tmp_step))
                summary.add_scalar(tag="training average loss",scalar_value=total_train_loss.item() / tmp_step,global_step=global_step)

        lr_scheduler.step(epoch=epoch)
        torch.save(regression_net, os.path.join(config.REGRESSION_MODEL_ROOT, "regression_" + str(epoch) + "_model.pth"))
        # endregion

        # region Validation
        total_valid_loss = 0
        tmp_step = 0
        regression_net.eval()
        with torch.no_grad():
            for i, data in enumerate(valid_dataloader):
                tmp_step += 1
                embeddings, target_score = data
                embeddings = embeddings.to(device)
                target_score = target_score.to(device)
                predict_score = regression_net(embeddings)
                predict_score, target_score = predict_score.float().view(-1, 1), target_score.float().view(-1, 1)
                loss = mse_criterion(predict_score,target_score)
                total_valid_loss += loss
        print("Epoch: %d, Global step:%d,Validation Average MSE Loss:%.4f" % (epoch,global_step, total_valid_loss.item() / tmp_step))
        summary.add_scalar(tag="validation average loss", scalar_value=total_valid_loss.item() / tmp_step, global_step=global_step)
        # endregion

if __name__ == "__main__":
    train()