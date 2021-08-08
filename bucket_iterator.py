# -*- coding: utf-8 -*-

import math
import random
import torch
import numpy

class BucketIterator(object):
    def __init__(self, data, batch_size, embed_type, sort_key='text_indices', shuffle=True, sort=True):
        self.embed_type = embed_type
        self.shuffle = shuffle
        self.sort = sort
        self.sort_key = sort_key
        self.batches = self.sort_and_pad(data, batch_size)
        self.batch_len = len(self.batches)

    def sort_and_pad(self, data, batch_size):
        num_batch = int(math.ceil(len(data) / batch_size))
        if self.sort:
            sorted_data = sorted(data, key=lambda x: len(x[self.sort_key]))
        else:
            sorted_data = data
        batches = []
        for i in range(num_batch):
            batches.append(self.pad_data(sorted_data[i*batch_size : (i+1)*batch_size], self.embed_type))
        return batches

    def pad_data(self, batch_data, type='glove'):
        batch_context = []
        batch_aspect = []
        batch_text_indices = []
        batch_aspect_indices = []
        batch_left_indices = []
        batch_polarity = []
        batch_dependency_graph = []
        batch_pos_tag = []
        batch_text_trans_indices = []
        batch_aspect_trans_indices = []
        batch_left_trans_indices = []
        if type == 'glove':
            max_len = max([len(t[self.sort_key]) for t in batch_data])
            for item in batch_data:
                context, aspect, text_indices, aspect_indices, left_indices, polarity, dependency_graph, pos_tag = \
                    item['context'], item['aspect'], item['text_indices'], item['aspect_indices'], item['left_indices'],\
                    item['polarity'], item['dependency_graph'], item['pos_tag']
                text_padding = [0] * (max_len - len(text_indices))
                aspect_padding = [0] * (max_len - len(aspect_indices))
                left_padding = [0] * (max_len - len(left_indices))
                pos_tag_padding = [0] * (max_len - len(pos_tag))
                batch_context.append(context)
                batch_aspect.append(aspect)
                batch_text_indices.append(text_indices + text_padding)
                batch_aspect_indices.append(aspect_indices + aspect_padding)
                batch_left_indices.append(left_indices + left_padding)
                batch_polarity.append(polarity)
                batch_dependency_graph.append(numpy.pad(dependency_graph, \
                    ((0,max_len-len(dependency_graph[0])),(0,max_len-len(dependency_graph[0]))), 'constant'))
                batch_pos_tag.append(pos_tag + pos_tag_padding)

            return { \
                    'context': batch_context,
                    'aspect': batch_aspect,
                    'text_indices': torch.tensor(batch_text_indices), \
                    'aspect_indices': torch.tensor(batch_aspect_indices), \
                    'left_indices': torch.tensor(batch_left_indices), \
                    'polarity': torch.tensor(batch_polarity), \
                    'dependency_graph': torch.tensor(batch_dependency_graph), \
                    'pos_tag': torch.tensor(batch_pos_tag),
                }

        else:
            max_len = max([len(t[self.sort_key]) for t in batch_data])
            max_len1 = max([len(t['text_trans_indices']) for t in batch_data])
            for item in batch_data:
                 
                context, aspect, text_indices, aspect_indices, left_indices, polarity, dependency_graph, pos_tag, text_trans_indices, aspect_trans_indices, left_trans_indices = \
                    item['context'], item['aspect'], item['text_indices'], item['aspect_indices'], item['left_indices'],\
                    item['polarity'], item['dependency_graph'], item['pos_tag'], item['text_trans_indices'], item['aspect_trans_indices'],\
                    item['left_trans_indices']
                
                text_padding = [0] * (max_len - len(text_indices))
                aspect_padding = [0] * (max_len - len(aspect_indices))
                left_padding = [0] * (max_len - len(left_indices))
                pos_tag_padding = [0] * (max_len - len(pos_tag))
                batch_context.append(context)
                batch_aspect.append(aspect)
                batch_text_indices.append(text_indices + text_padding)
                batch_aspect_indices.append(aspect_indices + aspect_padding)
                batch_left_indices.append(left_indices + left_padding)
                batch_polarity.append(polarity)
                batch_dependency_graph.append(numpy.pad(dependency_graph, \
                    ((0,max_len1-len(dependency_graph[0])),(0,max_len1-len(dependency_graph[0]))), 'constant'))
                batch_pos_tag.append(pos_tag + pos_tag_padding)
                batch_text_trans_indices.append(text_trans_indices)
                batch_aspect_trans_indices.append(aspect_trans_indices)
                batch_left_trans_indices.append(left_trans_indices)

            return { \
                    'context': batch_context,
                    'aspect': batch_aspect,
                    'text_indices': torch.tensor(batch_text_indices), \
                    'aspect_indices': torch.tensor(batch_aspect_indices), \
                    'left_indices': torch.tensor(batch_left_indices), \
                    'polarity': torch.tensor(batch_polarity), \
                    'dependency_graph': torch.tensor(batch_dependency_graph), \
                    'pos_tag': torch.tensor(batch_pos_tag),
                    'text_trans_indices': batch_text_trans_indices,
                    'aspect_trans_indices': batch_aspect_trans_indices,
                    'left_trans_indices': batch_left_trans_indices
                }

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.batches)
        for idx in range(self.batch_len):
            yield self.batches[idx]