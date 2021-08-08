# -*- coding: utf-8 -*-

import os
import math
import argparse
import random
from importlib_metadata import FastPath
import numpy as np
import torch
import torch.nn as nn
import sys
from bucket_iterator import BucketIterator
from sklearn import metrics
from data_utils import ABSADatesetReader
from models.acgcn import ACGCN
from models.acgcn_bert import ACGCN_BERT
import pickle
from data_utils import Tokenizer
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import spacy 
nlp = spacy.load("en_core_web_sm")

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        
        if self.opt.dataset == 'lap14':
            if self.opt.embed_type == 'glove':
                model_path = './state_dict/acgcn_lap14.pkl'
                seed = 2
            else:
                model_path = './state_dict/acgcn_bert_lap14.pkl'
        elif self.opt.dataset == 'rest14':
            if self.opt.embed_type == 'glove':
                model_path = './state_dict/acgcn_rest14.pkl'
            else:
                model_path = './state_dict/acgcn_bert_rest14.pkl'
        elif self.opt.dataset == 'rest15':
            if self.opt.embed_type == 'glove':
                model_path = './state_dict/acgcn_rest15.pkl'
            else:
                model_path = './state_dict/acgcn_bert_rest15.pkl'
        elif self.opt.dataset == 'rest16':
            if self.opt.embed_type == 'glove':
                model_path = './state_dict/acgcn_rest16.pkl'
                seed = 2
            else:
                model_path = './state_dict/acgcn_bert_rest16.pkl'
        elif self.opt.dataset == 'twitter':
            if self.opt.embed_type == 'glove':
                model_path = './state_dict/acgcn_twitter.pkl'
            else:
                model_path = './state_dict/acgcn_bert_twitter.pkl'
        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_type=opt.embed_type, embed_dim=opt.embed_dim)
        
        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=opt.batch_size, embed_type=opt.embed_type, shuffle=False)
        
        print('loading model {0} ...'.format(opt.model_name))
        if opt.embed_type == 'glove':
            self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(opt.device)
        else:
            self.model = opt.model_class(opt).to(opt.device)
        self.model.load_state_dict(torch.load(model_path))
        # self.set_seed(seed)
        # self._print_args()
        self.global_f1 = 0.

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # fout = open('./badcase/lap14_badcase.txt', 'w', encoding='utf-8')
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) if type(t_sample_batched[col])!=list else t_sample_batched[col] for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(opt.device)
                context = t_sample_batched['context']
                aspect = t_sample_batched['aspect']
                t_outputs = self.model(t_inputs)
                predict = torch.argmax(t_outputs, -1).cpu().numpy()
                # for i in range(len(predict)):
                #     if predict[i] != t_targets[i].item():
                #         if predict[i] == 0:
                #             pre = '消极'
                #         elif predict[i] == 1:
                #             pre = '中性'
                #         else:
                #             pre = '积极'
                #         fout.write(pre + '\t')
                #         if t_targets[i] == 0:
                #             tar = '消极'
                #         elif t_targets[i] == 1:
                #             tar = '中性'
                #         else:
                #             tar = '积极'
                #         fout.write(tar + '\t')
                #         fout.write(aspect[i] + '\t')
                #         fout.write(context[i] + '\n')

                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_test_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def run(self):
        test_acc, test_f1 = self._evaluate_acc_f1()   
        print("Accuracy for {0} is".format(self.opt.dataset), test_acc)
        print("F1 score for {0} is".format(self.opt.dataset), test_f1)

def get_one_data(fname, tokenizer):
    fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        polarity = lines[i + 2].strip()
        context = text_left + " " + aspect + " " + text_right
        text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
        aspect_indices = tokenizer.text_to_sequence(aspect)
        left_indices = tokenizer.text_to_sequence(text_left)
        polarity = int(polarity)+1
        context = context.strip()
        tokens = nlp(context)
        words = context.split()
        matrix = np.zeros((len(words), len(words))).astype('float32')
        assert len(words) == len(list(tokens))
        for token in tokens:
            matrix[token.i][token.i] = 1
            for child in token.children:
                matrix[token.i][child.i] = 1
                matrix[child.i][token.i] = 1
        
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
        return inputs, torch.tensor([polarity]), words, aspect.split()


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='acgcn', type=str)
    parser.add_argument('--dataset', default='rest16', type=str, help='twitter, rest14, lap14, rest15, rest16')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_type', default='glove', type=str)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--pos_dim', default=300, type=int)
    parser.add_argument('--num_attention_heads', default=6, type=int)
    parser.add_argument('--attention_probs_dropout_prob', default=0.1, type=float)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--bert_model_dir', default='./bert-base-uncased', type=str)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--highway', default=False, type=bool)
    parser.add_argument('--layernorm', default=False, type=bool)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--device', default=None, type=str)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of bilstm or highway or elmo.')
    opt = parser.parse_args()
    model_classes = {
        'acgcn': ACGCN,
        'acgcn_bert': ACGCN_BERT
    }
    input_colses = {
        'acgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph', 'pos_tag'],
        'acgcn_bert':['text_indices', 'text_trans_indices', 'aspect_trans_indices', 'left_trans_indices', 'dependency_graph']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("Current Device is cuda")
    else:
        print("Current Device is cpu")

    test_one = False

    if test_one == False:
        ins = Instructor(opt)
        ins.run()
    
    else:
        input_file = './one_data.txt'
        dataset = 'rest14'
        embedding_matrix_file_name = '300_rest14_embedding_matrix.pkl'

        with open(dataset+'_word2idx.pkl', 'rb') as f:
            word2idx = pickle.load(f)
            tokenizer = Tokenizer(word2idx=word2idx)
            
        if os.path.exists(embedding_matrix_file_name):
            print('loading embedding_matrix:', embedding_matrix_file_name)
            embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
        
        model = opt.model_class(embedding_matrix, opt).to(opt.device)
        model.load_state_dict(torch.load('./state_dict/asgcn_rest14_pos_test.pkl'))
        inputs, target, context, aspect = get_one_data(input_file, tokenizer)
        inputs = [item.to(opt.device) for item in inputs]
        outputs, att_weight = model(inputs)
        predict = torch.argmax(outputs, -1).cpu().numpy()
        pre = ''
        tar = ''
        for i in range(len(predict)):
            if predict[i] != target[i].item():
                if predict[i] == 0:
                    pre = '消极'
                elif predict[i] == 1:
                    pre = '中性'
                else:
                    pre = '积极'
                if target[i] == 0:
                    tar = '消极'
                elif target[i] == 1:
                    tar = '中性'
                else:
                    tar = '积极'
        print(pre, tar)
        d = att_weight.cpu().detach().numpy()
        col = ['aspect']     #需要显示的词
        index = context  #需要显示的词
        df = pd.DataFrame(d.transpose(), columns=col, index=index )

        fig = plt.figure()

        ax = fig.add_subplot(111)

        cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
        #cax = ax.matshow(df)
        fig.colorbar(cax)

        tick_spacing = 1
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

        # fontdict = {'rotation': 'vertical'}    #设置文字旋转
        fontdict = {'rotation': 90}       #或者这样设置文字旋转
        #ax.set_xticklabels([''] + list(df.columns), rotation=90)  #或者直接设置到这里
        ax.set_xticklabels([''] + list(df.columns), fontdict=fontdict)
        ax.set_yticklabels([''] + list(df.index))
        plt.savefig("att_weight.png")
        plt.show()