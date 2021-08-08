# 首先加载必用的库
from mainWindow import *
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal, QFile
import time
import sys
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import codecs
import json
import chardet
import jieba
from collections import Counter
from xpinyin import Pinyin
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib
import torch
import pickle
import argparse
import spacy
from supar import Parser
from acgcn_bert import ACGCN_BERT
from acgcn import ACGCN
from data_utils import build_embedding_matrix, Tokenizer, Bert_Tokenizer
from generate_acg import dependency_adj_matrix_biaffine, SRD, get_weight_matrix, WhitespaceTokenizer
# pyuic5 -o 目标文件名.py 源文件名.ui
# pyrcc5 -o 目标文件名.py 源文件名.qrc

matplotlib.use("Qt5Agg")  # 声明使用QT5
# 全局变量定义
neg = pos = sum = 0
max_tokens = 0
text_lists = []

text_lists_net = []
hotelId = 0
url = city = ''

# 使用word2vec向量化中文语句
# 使用Gensim加载预先训练好的模型 约260k
# cn_model = KeyedVectors.load_word2vec_format(
#     "./sgns.zhihu.bigram", binary=False)
# text_result = []
parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='acgcn', type=str)
parser.add_argument('--dataset', default='rest14', type=str, help='twitter, rest14, lap14, rest15, rest16')
parser.add_argument('--optimizer', default='adam', type=str)
parser.add_argument('--initializer', default='xavier_uniform_', type=str)
parser.add_argument('--learning_rate', default=0.001, type=float)
parser.add_argument('--l2reg', default=0.00001, type=float)
parser.add_argument('--num_epoch', default=100, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--log_step', default=5, type=int)
parser.add_argument('--embed_type', default='glove', type=str)
parser.add_argument('--embed_dim', default=300, type=int)
parser.add_argument('--hidden_dim', default=768, type=int)
parser.add_argument('--pos_dim', default=300, type=int)
parser.add_argument('--num_attention_heads', default=6, type=int)
parser.add_argument('--attention_probs_dropout_prob', default=0.1, type=float)
parser.add_argument('--polarities_dim', default=3, type=int)
parser.add_argument('--bert_model_dir', default='./bert-base-uncased', type=str)
parser.add_argument('--save', default=False, type=bool)
parser.add_argument('--highway', default=True, type=bool)
parser.add_argument('--layernorm', default=False, type=bool)
parser.add_argument('--seed', default=776, type=int)
parser.add_argument('--device', default=None, type=str)
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of bilstm or highway or elmo.')
opt = parser.parse_args()
opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = Parser.load('biaffine-dep-bert-en')
nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

# print("loading {0} tokenizer...".format(opt.dataset))
# with open('./middle_var/rest14_word2idx.pkl', 'rb') as f:
#     word2idx = pickle.load(f)
#     tokenizer = Tokenizer(word2idx=word2idx)
# embedding_matrix = build_embedding_matrix(tokenizer.word2idx, opt.embed_dim, opt.dataset)
tokenizer = Bert_Tokenizer()

def get_one_data_bert(text, aspect, tokenizer):
    words = text.split()
    aspect_index = words.index(aspect)
    text_left = ' '.join(words[:aspect_index])
    text_indices, text_trans_indices = tokenizer.text_to_sequence(text, True)
    _, aspect_trans_indices = tokenizer.text_to_sequence(aspect, True)
    _, left_trans_indices = tokenizer.text_to_sequence(text_left, True)
    _, trans_matrix = dependency_adj_matrix_biaffine(text)
    words_list = text.split()
    relative_dis = []
    for item in aspect.split():
        relative_dis.append(np.array(SRD(trans_matrix, words_list.index(item), words_list)))
    matrix = get_weight_matrix(words_list, aspect.split(), relative_dis)
    inputs = (torch.tensor([text_indices]), [text_trans_indices], [aspect_trans_indices], [left_trans_indices],
              torch.tensor([matrix]))
    return inputs, words, aspect.split()

def get_one_data(text, aspect, tokenizer):
    words = text.split()
    aspect_index = words.index(aspect)
    text_left = ' '.join(words[:aspect_index])
    text_indices = tokenizer.text_to_sequence(text)
    aspect_indices = tokenizer.text_to_sequence(aspect)
    left_indices = tokenizer.text_to_sequence(text_left)
    tokens = nlp(text)
    _, trans_matrix = dependency_adj_matrix_biaffine(text)
    words_list = text.split()
    relative_dis = []
    for item in aspect.split():
        relative_dis.append(np.array(SRD(trans_matrix, words_list.index(item), words_list)))
    matrix = get_weight_matrix(words_list, aspect.split(), relative_dis)
    
    pos = []
    pos_mapping = {'adj': 1, 'adv': 2, 'verb':3, 'others': 4}
    for token in tokens:
        if token.pos_ == 'ADJ':
            pos.append('adj')
        elif token.pos_ == 'ADV':
            pos.append('adv')
        elif token.pos_ == 'VERB':
            pos.append('verb')
        else:
            pos.append('others')
    pos_tag = [pos_mapping[item] for item in pos]
    inputs = (torch.tensor([text_indices]), torch.tensor([aspect_indices]), torch.tensor([left_indices]),
              torch.tensor([matrix]), torch.tensor([pos_tag]))
    return inputs, words, aspect.split()

class MyFigure(FigureCanvas):
    def __init__(self, width=5, height=4, dpi=100):
        # 第一步：创建一个创建Figure
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        # 第二步：在父类中激活Figure窗口
        super(MyFigure, self).__init__(self.fig)  # 此句必不可少，否则不能显示图形
        # 第三步：创建一个子图，用于绘制图形用，111表示子图编号，如matlab的subplot(1,1,1)
        self.axes = self.fig.add_subplot(111)

def print_args(model):
    n_trainable_params, n_nontrainable_params = 0, 0
    for n, p in model.named_parameters():
        print(n, p.size())

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)
        # Form
        self.F = MyFigure(width=3, height=2, dpi=100)
        self.model = ACGCN_BERT(opt)
        print_args(self.model)
        self.TrainButton.clicked.connect(self.train)  # 训练按钮
        self.analyse.clicked.connect(self.Analyse)  # 分析

        # RadioButtonGroup
        self.buttonGroup = QtWidgets.QButtonGroup(self)
        self.buttonGroup.addButton(self.radio_openfile)
        self.buttonGroup.addButton(self.radio_input)

        self.radio_openfile.toggled.connect(self.opTest)
        self.radio_input.toggled.connect(self.inTest)

        # 初始化隐藏其他元素
        self.frame_openfile.setVisible(False)
        self.frame_opResult.setVisible(False)
        self.frame_net.setVisible(False)
        self.frame_netResult.setVisible(False)

        # 读取文件
        self.open_txt.clicked.connect(self.open)
        self.tabWidget.setCurrentIndex(0)

        # 网络爬虫相关
        self.crawlButton.clicked.connect(self.Crawl)

        # 词频统计
        self.button_count.clicked.connect(self.WordCount)
        self.button_clearBuffer.clicked.connect(self.clearBuffer)

    def inTest(self):
        self.frame_input_text.setVisible(True)
        self.frame_input_aspect.setVisible(True)
        self.frame_inputResult.setVisible(True)

        self.frame_openfile.setVisible(False)
        self.frame_opResult.setVisible(False)

        self.frame_net.setVisible(False)
        self.frame_netResult.setVisible(False)

    def opTest(self):
        self.frame_input_text.setVisible(False)
        self.frame_inputResult.setVisible(False)

        self.frame_input_aspect.setVisible(False)
        self.frame_inputResult.setVisible(False)

        self.frame_openfile.setVisible(True)
        self.frame_opResult.setVisible(True)

        self.frame_net.setVisible(False)
        self.frame_netResult.setVisible(False)

    def open(self):
        global sum, pos, neg, neu
        sum = pos = neg = neu = 0
        if self.sender() == self.open_txt:
            file = QFileDialog.getOpenFileName(
                self, "打开文件", "./", "txt文件(*.txt)")

            if file[0]:
                f = QFile(file[0])
                f = open(file[0], "r", encoding='utf-8')
                with f:
                    data = f.read()
                    text_lists = data.split('\n')

                f.close()
            self.OutputBox.append("---------txt文件内容----------")

            def predict_sentiment(text, aspect):
                global sum, pos, neg, neu
                inputs, context, aspect = get_one_data_bert(text, aspect, tokenizer)
                inputs = [item.to(opt.device) if type(item)!=list else item for item in inputs]

                outputs = self.model(inputs)
                predict = torch.argmax(outputs, -1).cpu().numpy()
                # 预测 
                sen = ''
                sum += 1
                if predict[0] == 2:
                    pos += 1
                    sen = '积极'
                elif predict[0] == 0:
                    neg += 1
                    sen = '消极'
                else:
                    neu += 1
                    sen = '中性'
                return sen

            for i in range(0, len(text_lists), 2):
                self.OutputBox.append('-'+text_lists[i])
                sen = predict_sentiment(text_lists[i], text_lists[i+1])
                self.OutputBox.append('-'+sen)
            self.sum_show.setText(str(len(text_lists) // 2))
            self.pos_show.setText(str(pos))
            self.neg_show.setText(str(neg))
            self.aver_show.setText(str(neu))


    def train(self): 
        self.model.load_state_dict(torch.load('./state_dict/acgcn_bert_rest14.pkl', map_location=torch.device('cpu')))
        # 加载已训练的模型
        self.OutputBox.append("Complate")
        self.progressBar.setValue(100)
        self.TrainButton.setText("模型加载完成")
        self.TrainButton.setEnabled(False)

        # 训练结束后将三个模态框设为启用状态
        self.frame_input_text.setEnabled(True)
        self.frame_input_aspect.setEnabled(True)
        self.frame_net.setEnabled(True)
        self.frame_openfile.setEnabled(True)

        # PyQt Bug:自动设置为不可选
        self.open_txt.setEnabled(True)
        self.analyse.setEnabled(True)
        
    def Analyse(self):
        self.OutputBox.append('输入：'+self.input_text.text())
        text = self.input_text.text()
        aspect = self.input_aspect.text()
        inputs, context, aspect = get_one_data_bert(text, aspect, tokenizer)
        inputs = [item.to(opt.device) if type(item)!=list else item for item in inputs]

        outputs = self.model(inputs)
        predict = torch.argmax(outputs, -1).cpu().numpy()
        if predict[0] == 2:
            self.OutputBox.append('Positive')
            self.judge.setText("正面评价")
            self.graphicsView.setStyleSheet(
                "background:url(:/img/Excellent.png) no-repeat; border:n")
        elif predict[0] == 1:
            self.OutputBox.append('Neutral')
            self.judge.setText("折中的评价")
            self.graphicsView.setStyleSheet(
                "background:url(:/img/Average.png) no-repeat; border:n")
        else:
            self.OutputBox.append('Negative')
            self.judge.setText("负面评价")
            self.graphicsView.setStyleSheet(
                "background:url(:/img/Poor.png) no-repeat; border:n")
        

    def Crawl(self):
        global city, hotelId, text_lists_net
        text_lists_net = []
        city = self.citybox.currentText()
        hotelId = self.input_id.text()

        def crawlFunction(index):
            global url
            if index == 1:
                url = "http://touch.qunar.com/api/hotel/hoteldetail/comment?seq=" + \
                    city + "_city_" + hotelId
                crawExe(1)
            else:
                crawlFunction(1)
                for i in range(2, index+1):
                    url = "http://touch.qunar.com/api/hotel/hoteldetail/comment?seq=" + \
                        city + "_city_" + hotelId + \
                        "&commentType=0&commentPage="+str(i)
                    crawExe(i)

        def crawExe(index):

            try:
                self.OutputBox.append(
                    "-------正在抓取第" + str(index) + "页---------")
                self.OutputBox.append(url)
                print("正在抓取第" + str(index) + "页")
                print(url)

                data = (json.loads(requests.get(url).text))[
                    'data']['commentData']

                self.label_hotelname.setText(data['hotelName'])
                self.label_total.setText(str(data['allTotal']))
                self.label_good.setText(str(data['goodTotal']))
                self.label_mid.setText(str(data['mediumTotal']))
                self.label_bad.setText(str(data['badTotal']))
                self.label_aver.setText(str(data['score'])+'/5')

                self.OutputBox.append('-------------评论内容--------------')
                for each in data['comments']:
                    text_lists_net.append(each['content'])
                    self.OutputBox.append(each['content'])
            except:
                pass

        city = Pinyin().get_pinyin(self.citybox.currentText(), '')
        crawlFunction(self.spinBox.value())
        try:
            with open('qunar_hotel_info.csv', "r+", encoding='utf-8') as f:
                f.seek(0)
                f.truncate()  # 清空文件
        except:
            pass
        pd.DataFrame({'content': text_lists_net}).to_csv(
            './qunar_hotel_info.csv', index=True)
        self.button_count.setEnabled(True)
        self.button_clearBuffer.setEnabled(True)

    def WordCount(self):
        self.tabWidget.setCurrentIndex(1)
        with codecs.open('qunar_hotel_info.csv', 'r', encoding='utf8') as f:
            txt = re.sub("[^\u4E00-\u9FA5]", "", f.read())

        stop_list=['酒店','我们','就是','因为','虽然','比较','还有','入住','房间','时候','北京','非常']
        
        # 去停用词
        for text in stop_list:
            txt=txt.replace(text,'')

        seg_list = jieba.cut(txt)

        wl = jieba.cut(txt)
        wl2 = " ".join(wl)
        c = Counter()
        for x in seg_list:
            if len(x) > 1 and x != '\r\n':
                c[x] += 1

        for (k, v) in c.most_common(30):
            print('%s%s %s  %d' % (' '*(5-len(k)), k, '*'*int(v/3), v))
            self.OutputBox_count.append('%s%s %s  %d' % (
                ' '*(5-len(k)), k, '*'*int(v/3), v))

        wc = WordCloud(background_color="rgb(250, 250, 250)",
                       width=350,
                       height=180,
                       max_words=40,
                       max_font_size=60,
                       min_font_size=10,
                       font_path="./design/font.ttf",
                       random_state=40)
        wc.generate(wl2)
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.savefig('./img/wordCloud.jpg')
        self.label_cloud.setPixmap(QtGui.QPixmap('./img/wordcloud.jpg'))

    def clearBuffer(self):
        with open('qunar_hotel_info.csv', "r+", encoding='utf-8') as f:
            f.seek(0)
            f.truncate()  # 清空文件
        self.OutputBox_count.setText('')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()
    myWin.show()
    sys.exit(app.exec_())
