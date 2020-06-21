'''
@File: common_utils.py
@Author: Luyufan
@Date: 2020/6/14
@Desc: 通用支持方法
'''

import re
import pandas as pd
import matplotlib.pyplot as plt
import config

# region pre-execute
special_characters = '[·’!"\#$%&\'()＃！（）*+,-./:;<=>?\@，：?￥★、…．•«»■\\n＞【】［］《》？“”‘’\[\\]^_`{|}~]+'

stop_words = []
try:
    with open(config.WORD2VEC_STOPWORDS, mode="r", encoding="utf-8") as f:
        for stop_word in f:
            stop_word = stop_word[:-1]   # trim the \n
            stop_words.append(stop_word)
        f.close()
except IOError:
    print("Warning: can not load english_stopwords.txt")

#endregion

def remove_special_characters(content):
    if not isinstance(content,str):
        return " "
    clean_content = re.sub(special_characters," ",content)
    return clean_content

def filter_words_list(word):
    if word in stop_words or len(word) == 0:
        return False
    else:
        try:
            _ = int(word)
        except (ValueError,TypeError) as e:
            return True
        else:
            return False

def load_data_from_excel(path):
    df = pd.read_excel(path, sheet_name=0, header=0)
    return df

def remove_stopwords(content):
    content = remove_special_characters(content)
    content = content.lower()
    word_list = content.split(" ")
    clean_word_list = list(filter(filter_words_list, word_list))
    return clean_word_list

def plot_2d_scatter(x, y, title, xlabel, ylabel):
    plt.figure(dpi=200)
    plt.title(title)
    plt.scatter(x,y)
    plt.xlabel(xlabel)
    plt.xlabel(ylabel)
    plt.show()

def plot_3d_scatter(x,y,z,title,xlabel,ylabel,zlabel):
    # plot
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(dpi=300)
    plt.title(title)
    ax = Axes3D(fig)
    ax.scatter(x, y, z)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    plt.show()