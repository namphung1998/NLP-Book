from argparse import Namespace
from collections import Counter
import json
import os
import re
import string

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class Vocabulary(object):
    def __init__(self, token_to_idx=None, add_unk=True, unk_token='<UNK>'):
        """
        @param token_to_idx (dict): map of tokens to indices
        @param add_unk (bool)
        @param unk_token (str)
        """

        if token_to_idx is None:
            token_to_idx = {}

        self._token_to_idx = token_to_idx
        self._idx_to_token = {i: t for t, i in self._token_to_idx.items()}
        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)

    def to_serializable(self):
        """
        Returns a dictionary that can be serialized
        """
        return {'token_to_idx': self._token_to_idx,
                'add_unk': self._add_unk,
                'unk_token': self._unk_token}

    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):
        """
        Update mapping dictionaries

        @param token (str): the item to be added to this Vocabulary

        @returns index (int): the index of the new token
        """
        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index

    def add_many(self, tokens):
        """
        Adds a list of tokens to this Vocabulary

        @param tokens (List[str])

        @returns indices
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """
        Retrieves the index of a given token, or the unk index
        @param token (str)

        @returns index (int)
        """
        if self.unk_index > 0:
            return self._token_to_idx.get(token, self.unk_index)
        return self._token_to_idx[token]

    def lookup_index(self, i):
        if i not in self._idx_to_token:
            raise KeyError(f'The index {i} is not in this Vocabulary.')
        return self._idx_to_token[i]

    def __str__(self):
        return f'Vocabulary(size={len(self)})'

    def __len__(self):
        return len(self._token_to_idx)


class ReviewVectorizer(object):
    def __init__(self, review_vocab, rating_vocab):
        """

        """
