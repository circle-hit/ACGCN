# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
import spacy 
from spacy.tokens import Doc
from transformers import BertTokenizer

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load("en_core_web_sm")
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

def load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = './middle_var/{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        fname = './glove/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix

class Bert_Tokenizer(object):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def text_to_sequence(self, text, tran=False):
        text = text.lower()
        words = text.split()
        if tran==False:
            unknownidx = 1
            sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
            if len(sequence) == 0:
                sequence = [0]
            return sequence
        else:
            trans=[]
            realwords=[]
            for word in words:
                wordpieces=self.tokenizer._tokenize(word)
                tmplen=len(realwords)
                realwords.extend(wordpieces)
                trans.append([tmplen,len(realwords)])
            sequence = [self.tokenizer._convert_token_to_id('[CLS]')]+[self.tokenizer._convert_token_to_id(w) for w in realwords]+[self.tokenizer._convert_token_to_id('[SEP]')]
            if len(sequence) == 0:
                sequence = [0]
            return sequence,trans

class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<oov>'] = self.idx
            self.idx2word[self.idx] = '<oov>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence

class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)



class ABSADatesetReader:
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    @staticmethod
    def __read_data__(fname, tokenizer, type='glove'):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname+'_biaffine_weight.graph', 'rb')
        idx2graph = pickle.load(fin)
        fin.close()

        all_data = []
        for i in range(0, len(lines), 3): 
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()
            context = text_left + " " + aspect + " " + text_right
            context = context.strip()
            doc = nlp(context)
            if type != 'glove':
            # For BERT based model 
                text_indices, text_trans_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right, True)
                aspect_indices, aspect_trans_indices = tokenizer.text_to_sequence(aspect, True)
                left_indices, left_trans_indices = tokenizer.text_to_sequence(text_left, True)
                polarity = int(polarity)+1
                dependency_graph = idx2graph[i]
                pos = []
                pos_mapping = {'adj': 1, 'adv': 2, 'verb':3, 'others': 4}
                for token in doc:
                    if token.pos_ == 'ADJ':
                        pos.append('adj')
                    elif token.pos_ == 'ADV':
                        pos.append('adv')
                    elif token.pos_ == 'VERB':
                        pos.append('verb')
                    else:
                        pos.append('others')
                pos_tag = [pos_mapping[item] for item in pos]
                data = {
                'context': context,
                'aspect': aspect,
                'text_indices': text_indices,
                'aspect_indices': aspect_indices,
                'left_indices': left_indices,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
                'pos_tag': pos_tag,
                'text_trans_indices': text_trans_indices,
                'aspect_trans_indices': aspect_trans_indices,
                'left_trans_indices': left_trans_indices
            }
            else:
                text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
                aspect_indices = tokenizer.text_to_sequence(aspect)
                left_indices = tokenizer.text_to_sequence(text_left)
                polarity = int(polarity)+1
                dependency_graph = idx2graph[i]
                pos = []
                pos_mapping = {'adj': 1, 'adv': 2, 'verb':3, 'others': 4}
                for token in doc:
                    if token.pos_ == 'ADJ':
                        pos.append('adj')
                    elif token.pos_ == 'ADV':
                        pos.append('adv')
                    elif token.pos_ == 'VERB':
                        pos.append('verb')
                    else:
                        pos.append('others')
                pos_tag = [pos_mapping[item] for item in pos]
                if len(dependency_graph[0]) != len(text_indices):
                    print(text_left + " " + aspect + " " + text_right)
                    print(dependency_graph)
                data = {
                'context': context,
                'aspect': aspect,
                'text_indices': text_indices,
                'aspect_indices': aspect_indices,
                'left_indices': left_indices,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
                'pos_tag': pos_tag,
            }
            all_data.append(data)
        return all_data

    def __init__(self, dataset='twitter', embed_type='glove', embed_dim=300):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
            'rest15': {
                'train': './datasets/semeval15/restaurant_train.raw',
                'test': './datasets/semeval15/restaurant_test.raw'
            },
            'rest16': {
                'train': './datasets/semeval16/restaurant_train.raw',
                'test': './datasets/semeval16/restaurant_test.raw'
            },
        }
        if embed_type == 'glove':
            text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
            if os.path.exists('./middle_var/'+dataset+'_word2idx.pkl'):
                print("loading {0} tokenizer...".format(dataset))
                with open('./middle_var/'+dataset+'_word2idx.pkl', 'rb') as f:
                     word2idx = pickle.load(f)
                     tokenizer = Tokenizer(word2idx=word2idx)
            else:
                tokenizer = Tokenizer()
                tokenizer.fit_on_text(text)
                with open(dataset+'_word2idx.pkl', 'wb') as f:
                     pickle.dump(tokenizer.word2idx, f)
            self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        else:
            print('Using BERT')
            tokenizer = Bert_Tokenizer()
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer, embed_type))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer, embed_type))
    
