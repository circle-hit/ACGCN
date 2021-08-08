# -*- coding: utf-8 -*-

import numpy as np
import pickle
from numpy.lib.function_base import average
from supar import Parser
from tqdm import trange
import spacy
from spacy.tokens import Doc

parser = Parser.load('biaffine-dep-bert-en')

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

def dependency_adj_matrix_spacy(text):
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1
    
    trans_matrix = [[float('inf') for _ in range(len(words))] for _ in range(len(words))]
    for i in range(len(words)):
        for j in range(len(words)):
            if matrix[i][j]:
                trans_matrix[i][j] = 1

    return matrix, trans_matrix

def dependency_adj_matrix_biaffine(text):
    words = text.split()
    dataset = parser.predict([words], prob=True, verbose=False)
    arcs = dataset.arcs[0]
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(arcs)
    for i in range(len(words)):
        matrix[i][i] = 1
        if arcs[i] == 0:
            continue
        matrix[i][arcs[i]-1] = 1
        matrix[arcs[i]-1][i] = 1
    
    trans_matrix = [[float('inf') for _ in range(len(words))] for _ in range(len(words))]
    for i in range(len(words)):
        for j in range(len(words)):
            if matrix[i][j]:
                trans_matrix[i][j] = 1

    return matrix, trans_matrix

def SRD(data_matrix, start_node, words_list):
    '''''
    Dijkstra求解最短路径算法
    输入: 原始数据矩阵，起始顶点
    输出: 起始顶点到其他顶点的最短距离
    '''
    vex_num = len(data_matrix)
    flag_list = ['False'] * vex_num
    prev=[0] * vex_num
    dist=['0'] * vex_num
    for i in range(vex_num):
        flag_list[i] = False
        prev[i] = 0
        dist[i] = data_matrix[start_node][i]

    flag_list[start_node] = False
    dist[start_node] = 0
    k = 0
    for i in range(1, vex_num):
        min_value = 99999
        for j in range(vex_num):
            if flag_list[j] == False and dist[j] != float('inf'):
                min_value = dist[j]
                k = j
        flag_list[k] = True
        for j in range(vex_num):
            if data_matrix[k][j] == float('inf'):
                temp = float('inf')
            else:
                temp = min_value + data_matrix[k][j]
            if flag_list[j] == False and temp != float('inf') and temp < dist[j]:
                dist[j] = temp
                prev[j] = k
    return dist

def get_mask_matrix(words_list, aspect_list, relative_dis, thr=2):
    '''''
    Generate Mask ACG matrix for ablation study
    '''
    matrix = np.zeros((len(words_list), len(words_list))).astype('float32')
    assert len(aspect_list) == len(relative_dis)
    for i in range(len(aspect_list)):
        idx = words_list.index(aspect_list[i])
        for j in range(len(words_list)):
            matrix[idx][idx] = 1
            if relative_dis[i][j] <= thr:
                matrix[idx][j] = 1
                matrix[j][idx] = 1
    return matrix

def get_weight_matrix(words_list, aspect_list, relative_dis, thr=2):
    '''''
    Generate ACG adjacent matrix
    '''
    matrix = np.zeros((len(words_list), len(words_list))).astype('float32')
    assert len(aspect_list) == len(relative_dis)
    for i in range(len(aspect_list)):
        idx = words_list.index(aspect_list[i])
        matrix[idx][idx] = 1
        for j in range(len(words_list)):
            matrix[idx][j] = 1 - ((relative_dis[i][j] - thr) / len(words_list))
            matrix[j][idx] = 1 - ((relative_dis[i][j] - thr) / len(words_list))
    return matrix

def get_fc_matrix(words_list):
    '''''
    Generate fully-connected adjacent matrix
    '''
    matrix = np.ones((len(words_list), len(words_list))).astype('float32')
    return matrix

def get_cen_matrix(words_list, aspect_list, relative_dis):
    '''''
    Diectly link words to the aspect for ablation study
    '''
    matrix = np.zeros((len(words_list), len(words_list))).astype('float32')
    assert len(aspect_list) == len(relative_dis)
    for i in range(len(aspect_list)):
        idx = words_list.index(aspect_list[i])
        matrix[idx][idx] = 1
        for j in range(len(words_list)):
            matrix[idx][j] = 1 
            matrix[j][idx] = 1
    return matrix

def process(filename, type='weight', tool='biaffine'):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    if tool == 'biaffine':
        print("-"*20 + "Using Toolkit Biaffine" + "-"*20)
        fout_weight = open(filename+'_biaffine_' + type + '.graph', 'wb')
        for i in trange(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            adj_matrix, trans_matrix = dependency_adj_matrix_biaffine(text_left+' '+aspect+' '+text_right)
            words_list = (text_left+' '+aspect+' '+text_right).split()
            relative_dis = []
            for item in aspect.split():
                relative_dis.append(np.array(SRD(trans_matrix, words_list.index(item), words_list)))

            if type == 'weight':
                matrix = get_weight_matrix(words_list, aspect.split(), relative_dis)
            elif type == 'mask':
                matrix = get_mask_matrix(words_list, aspect.split(), relative_dis)
            elif type == 'cen':
                matrix = get_cen_matrix(words_list, aspect.split(), relative_dis)
            elif type == 'dep':
                matrix = adj_matrix
            else:
                matrix = get_fc_matrix(words_list)
            idx2graph[i] = matrix
    else:
        print("-"*20 + "Using Toolkit Spacy" + "-"*20)
        fout_weight = open(filename+'_spacy_' + type + '.graph', 'wb')
        for i in trange(0, len(lines), 3):
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            adj_matrix, trans_matrix = dependency_adj_matrix_spacy(text_left+' '+aspect+' '+text_right)
            words_list = (text_left+' '+aspect+' '+text_right).split()
            relative_dis = []
            for item in aspect.split():
                relative_dis.append(np.array(SRD(trans_matrix, words_list.index(item), words_list)))

            if type == 'weight':
                matrix = get_weight_matrix(words_list, aspect.split(), relative_dis)
            elif type == 'mask':
                matrix = get_mask_matrix(words_list, aspect.split(), relative_dis)
            elif type == 'dep':
                matrix = adj_matrix
            else:
                matrix = get_fc_matrix(words_list)
            idx2graph[i] = matrix

    pickle.dump(idx2graph, fout_weight)     
    fout_weight.close()

if __name__ == '__main__':
    process('./datasets/acl-14-short-data/train.raw')
    process('./datasets/acl-14-short-data/test.raw')
    process('./datasets/semeval14/restaurant_train.raw')
    process('./datasets/semeval14/restaurant_test.raw')
    process('./datasets/semeval14/laptop_train.raw')
    process('./datasets/semeval14/laptop_test.raw')
    process('./datasets/semeval15/restaurant_train.raw')
    process('./datasets/semeval15/restaurant_test.raw')
    process('./datasets/semeval16/restaurant_train.raw')
    process('./datasets/semeval16/restaurant_test.raw')