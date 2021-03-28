# import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from gensim.models import Word2Vec


def load_training_data(path='training_label.txt'):
    if 'training_label' in path:
        with open(path, 'r') as f:
            lines = f.readlines()
            lines = [line.strip('\n').split(' ') for line in lines]
        x = [line[2:] for line in lines]
        y = [line[0] for line in lines]
        return x, y
    else:
        with open(path, 'r') as f:
            lines = f.readlines()
            x = [line.strip('\n').split(' ') for line in lines]
        return x


def load_testing_data(path='testing_data'):
    with open(path, 'r') as f:
        lines = f.readlines()
        X = ["".join(line.strip('\n').split(",")[1:]).strip()
             for line in lines[1:]]
        X = [sen.split(' ') for sen in X]
    return X


class Preprocess():
    """Class for preprocessing, including word embedding, sentence to vector"""
    def __init__(self, sentences, sen_len, w2v_path="./w2v.model"):
        self.w2v_path = w2v_path
        self.sentences = sentences
        self.sen_len = sen_len
        self.idx2word = []
        self.word2idx = {}
        self.embedding_matrix = []

    def get_w2v_model(self):
        """Load the embedding model"""
        self.embedding = Word2Vec.load(self.w2v_path)
        self.embedding_dim = self.embedding.vector_size

    def add_embedding(self, word):
        """Add words into embedding dictionary"""
        vector = torch.empty(1, self.embedding_dim)
        nn.init.uniform_(vector)
        self.word2idx[word] = len(self.word2idx)
        self.idx2word.append(word)
        self.embedding_matrix = torch.cat([self.embedding_matrix, vector], 0)

    def make_embedding(self, load=True):
        print("Get embedding ...")
        if load:
            print("loading word to vec model ...")
            self.get_w2v_model()
        else:
            raise NotImplementedError

        for i, word in enumerate(self.embedding.wv.vocab):
            print('get words #{}'.format(i+1), end='\r')
            self.word2idx[word] = len(self.word2idx)
            self.idx2word.append(word)
            self.embedding_matrix.append(self.embedding[word])

        print()
        self.embedding_matrix = torch.tensor(self.embedding_matrix)
        # Add "<PAD>" and "<UNK>"
        self.add_embedding("<PAD>")
        self.add_embedding("<UNK>")
        print("total words: {}".format(len(self.embedding_matrix)))
        return self.embedding_matrix

    def pad_sequence(self, sentence):
        """Padding sequence to same length"""
        if len(sentence) > self.sen_len:
            sentence = sentence[:self.sen_len]
        else:
            pad_len = self.sen_len - len(sentence)
            for _ in range(pad_len):
                sentence.append(self.word2idx["<PAD>"])
        assert len(sentence) == self.sen_len
        return sentence

    def sentence_word2idx(self):
        sentence_list = []
        for i, sen in enumerate(self.sentences):
            print('sentence count #{}'.format(i+1), end='\r')
            sentence_idx = []
            for word in sen:
                if (word in self.word2idx.keys()):
                    sentence_idx.append(self.word2idx[word])
                else:
                    sentence_idx.append(self.word2idx["<UNK>"])

            sentence_idx = self.pad_sequence(sentence_idx)
            sentence_list.append(sentence_idx)
        return torch.LongTensor(sentence_list)

    def labels_to_tensor(self, y):
        y = [int(label) for label in y]
        return torch.LongTensor(y)


class TwitterDataset(Dataset):
    """
    Expected data shape like:(data_num, data_len)
    Data can be a list of numpy array or a list of lists
    input data shape : (data_num, seq_len, feature_dim)

    __len__ will return the number of data
    """
    def __init__(self, X, y):
        self.data = X
        self.label = y

    def __getitem__(self, idx):
        if self.label is None:
            return self.data[idx]

        return self.data[idx], self.label[idx]

    def __len__(self):
        return len(self.data)


def get_train_valid_dataloader(
        train_fpath,
        train_nolabel_fpath,
        w2v_path,
        sen_len=20,
        batch_size=64,
        valid_split=0.1,
        num_workers=16,
        rand_seed=20,
        return_embedding=True):
    assert 0 <= valid_split < 1, "Invalid size for validation set"

    # Load text data
    train_x, y = load_training_data(train_fpath)
    train_x_no_label = load_training_data(train_nolabel_fpath)

    # Preprocessing to input data
    preprocess = Preprocess(train_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    train_x = preprocess.sentence_word2idx()
    y = preprocess.labels_to_tensor(y)

    # Shuffle the train_x and y
    num_x = train_x.shape[0]
    rand_indices = np.arange(num_x)
    np.random.seed(seed=rand_seed)
    np.random.shuffle(rand_indices)
    num_train = int(num_x * (1 - valid_split))

    # Split train and valid set
    X_train = train_x[rand_indices[:num_train]]
    X_val = train_x[rand_indices[num_train:]]
    y_train = y[rand_indices[:num_train]]
    y_val = y[rand_indices[num_train:]]

    # Create dataset
    train_dataset = TwitterDataset(X=X_train, y=y_train)
    val_dataset = TwitterDataset(X=X_val, y=y_val)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=num_workers)

    valid_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers)

    return train_loader, valid_loader, embedding


def get_test_dataloader(
        test_fpath,
        w2v_path,
        sen_len=20,
        batch_size=128,
        num_workers=16,
        return_embedding=True):
    # Load text data
    test_x = load_testing_data(test_fpath)

    # Preprocessing to input data
    preprocess = Preprocess(test_x, sen_len, w2v_path=w2v_path)
    embedding = preprocess.make_embedding(load=True)
    test_x = preprocess.sentence_word2idx()

    # Create dataset
    test_dataset = TwitterDataset(X=test_x, y=None)

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers)

    return test_loader, embedding
