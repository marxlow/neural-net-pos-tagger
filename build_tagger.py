# python3.5 build_tagger.py <train_file_absolute_path> <model_file_absolute_path>

import os
import math
import sys
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

class TextDataSet(Dataset):
    def __init__(self, corpus):
        # Corpus is a list of sentences & labels.
        sentences = [ ]
        labels = [ ]

        # Get the Max length of the sentence, we need this to pad all shorter sentences
        for sentence, label in corpus:
            sentences.append(" ".join(sentence))
            labels.append(" ".join(label))

        # Save padded sentences and labels
        self.sentences = sentences
        self.labels = labels
        self.corpus = corpus

    def __getitem__(self, index):
        return self.sentences[index], self.labels[index]
        
    def __len__(self):
        return len(self.corpus)

class PosTagModel(nn.Module):
    def __init__(self, word_embedding_dim, word_vocab_size, hidden_dim, tagset_size, word_padding_index, char_embedding_dim, char_vocab_size, convo_layer, window, hidden_layers):
        super(PosTagModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.convo_layer = convo_layer
        self.hidden_layers = hidden_layers
        # CNN for Character Sequence
        self.char_embeddings = nn.Embedding(char_vocab_size, char_embedding_dim).to(device)
        self.cnn = nn.Conv1d(char_embedding_dim, convo_layer, window, padding=1).to(device)
        # RNN with LSTM
        self.word_embeddings = nn.Embedding(word_vocab_size, word_embedding_dim, padding_idx=word_padding_index).to(device)
        self.lstm = nn.LSTM(word_embedding_dim + convo_layer, hidden_dim, num_layers=hidden_layers, dropout=0.5, bidirectional=True, batch_first=True).to(device)
        self.hidden_to_tag = nn.Linear(hidden_dim * 2, tagset_size).to(device)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.hidden_layers * 2, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.hidden_layers * 2, batch_size, self.hidden_dim).to(device))

    def forward(self, word_indexes, char_indexes):
        # CNN 
        char_embeds = self.char_embeddings(char_indexes)
        batch_size = char_embeds.shape[0]
        num_words = char_embeds.shape[1]
        num_chars = char_embeds.shape[2]
        char_dim = char_embeds.shape[3]
        char_embeds = char_embeds.view(batch_size * num_words, char_dim, num_chars)
        char_cnn_out = self.cnn(char_embeds) # (batch_size * num_words) * convo_layer * num_char
        char_cnn_out, _ = torch.max(char_cnn_out, 2) # (batch_size * num_words) * convo_layer
        char_cnn_out = char_cnn_out.view(batch_size, num_words, self.convo_layer) # batch_size * num_words * convo_layer

        # LSTM
        embeds = self.word_embeddings(word_indexes) # batch_size * max_sent_len * word_dim
        embeds = torch.cat((embeds, char_cnn_out), 2).to(device) # batch_size * max_sent_len * (word_dim + convo_layer)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.contiguous()
        tag_space = self.hidden_to_tag(lstm_out.view(-1, lstm_out.shape[2]))
        tag_scores = F.log_softmax(tag_space, dim=1).to(device) # (batch_size * max_sent_len) * num_pos_tag
        return tag_scores

    def prepare_batch_sequence(self, train_data):
        # Get max length of sentence and words
        sentence_lengths = []
        word_lengths = []
        sentences, labels = train_data
        for sentence in sentences:
            sentence_lengths.append(len(sentence.split()))
            for word in sentence.split():
                word_lengths.append(len(word))
        max_sentence_length = max(sentence_lengths)
        max_word_length = max(word_lengths)


        pad = "<PAD>"
        # Perform padding on sentences of shorter length
        char_tensors = []
        word_tensors = []
        label_tensors = []
        for i in range(len(sentence_lengths)):
            padded_sentence = sentences[i].split(" ")
            padded_label = labels[i].split(" ")
            # Add padding to remaining lengths
            for j in range(max_sentence_length - sentence_lengths[i]):
                padded_sentence.append(pad)
                padded_label.append(pad)

            # Perform padding on words of shorter length. Index = 0
            char_indexes = []
            sentence_char_indexes = []
            for word in padded_sentence:
                if (word == pad):
                    char_indexes = np.zeros(max_word_length)
                else:
                    char_indexes = ([self.char_to_index[char] for char in word])
                    for j in range(max_word_length - len(word)):
                        char_indexes.append(0)
                sentence_char_indexes.append(torch.tensor(char_indexes, dtype=torch.long).to(device))
            
            # Indexes should be of size max_sentence_len * max_word_len
            sentence_char_indexes = torch.stack(sentence_char_indexes).to(device)
            char_tensors.append(sentence_char_indexes)
            # Indexes should be of size: batch_size * max_sentence_length
            word_indexes = ([self.word_to_index[word] for word in padded_sentence])
            word_tensors.append(torch.tensor(word_indexes, dtype=torch.long).to(device))
            label_indexes = ([self.tag_to_index[tag] for tag in padded_label])
            label_tensors.append(torch.tensor(label_indexes, dtype=torch.long).to(device))

        char_tensors = torch.stack(char_tensors).to(device) # batch_size * max_word_len * max_char_len
        word_tensors = torch.stack(word_tensors).to(device)
        label_tensors = torch.stack(label_tensors).to(device)
        return char_tensors, word_tensors, label_tensors

    def prepare_sequence(self, list_of_element, element_to_idx):
        indexes = [element_to_idx[element] for element in list_of_element]
        return torch.tensor(indexes, dtype=torch.long).to(device)

class DataParser:
    def __init__(self, train_file):
        # Declare unknown, pad
        self.unknown_element = "UNKNOWN"
        self.pad = "<PAD>"

        # Parse each line in train_file to a list of sentences
        with open(train_file) as f:
            lines = f.readlines()

        # Go through each sentence in the corpus and collect a list of words and tags
        training_data = []
        train_word_list = []
        train_tag_list = []
        for line in lines:
            word_tag_list = line.split()
            words = []
            tags = []
            for word_tag in word_tag_list:
                parsed_word_tag = word_tag.split("/")
                words.append(parsed_word_tag[0])
                train_word_list.append(parsed_word_tag[0])
                tags.append(parsed_word_tag[len(parsed_word_tag) - 1])
                train_tag_list.append(parsed_word_tag[len(parsed_word_tag) - 1])
            training_data.append((words, tags))

        # Get unique words and tags from corpus
        unique_words = set(train_word_list)
        unique_words.add(self.unknown_element) # Add unknown words into dictionary 
        unique_words.add(self.pad)
        unique_tags = set(train_tag_list)
        unique_tags.add(self.pad)

        # Character parse
        train_text = open(train_file, "r").read()
        unique_chars = list(set(train_text.replace(" ", "")))
        unique_chars.append(self.unknown_element)
        unique_chars.insert(0, self.pad)

        
        # Store corpus information
        self.training_data = training_data
        self.char_to_index = {char: i for i, char in enumerate(unique_chars)}
        self.word_to_index = {word: i for i, word in enumerate(unique_words)}
        self.tag_to_index = {tag: i for i, tag in enumerate(unique_tags)}
        self.index_to_tag = dict(map(reversed, self.tag_to_index.items()))
        self.word_padding_index = self.word_to_index["<PAD>"]
        self.tag_padding_index = self.tag_to_index["<PAD>"]
        self.char_vocab_size = len(self.char_to_index)
        self.vocab_size = len(self.word_to_index)
        self.tagset_size = len(self.tag_to_index)
        print("[Parse] Completed reading train file:", train_file)


def train_model(train_file, model_file):
    start_time = datetime.datetime.now()
    # Hyper parameters CNN
    char_embedding_dim = 10
    window = 3
    convo_layer = 32
    # Hyper parameters RNN
    word_embedding_dim = 200
    hidden_dim = 200
    hidden_layers = 2
    # Hyper parameters Model
    max_epoch = 5
    batch_size = 16
    learning_rate = 0.75
    momentum = 0.9

    # 1. Parse data
    parser = DataParser(train_file)
    
    # 2.Initialize Model
    model = PosTagModel(word_embedding_dim, parser.vocab_size, hidden_dim, parser.tagset_size, parser.word_padding_index, char_embedding_dim, parser.char_vocab_size, convo_layer, window, hidden_layers)
    if torch.cuda.is_available():
        model.cuda() 
    model.char_to_index = parser.char_to_index
    model.word_to_index = parser.word_to_index
    model.tag_to_index = parser.tag_to_index
    model.index_to_tag = parser.index_to_tag
    model.unknown_element = parser.unknown_element

    # 3. Prepare optimizer and loss function
    loss_function = nn.CrossEntropyLoss(ignore_index=parser.tag_padding_index).to(device)
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    optimizer = optim.Adam(model.parameters())

    # 4. Prepare for batch processing
    train_data_set = TextDataSet(parser.training_data)
    train_data_loader = DataLoader(
        train_data_set,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 8,
    )

    # 5. Run model
    for epoch in range(max_epoch):
        print("~~~~~~ Running epoch: ", epoch)
        # for iter, train_data in enumerate(tqdm(train_data_loader)): # Uncomment to see progress
        for iter, train_data in enumerate(train_data_loader):
            # Get sentences and labels for batch
            batch_size = len(train_data[0])

            # Clear accumulated gradient
            model.zero_grad()

            # Clear hidden state
            model.hidden = model.init_hidden(batch_size)

            # Prepare tensors
            char_indexes, word_indexes, targets = model.prepare_batch_sequence(train_data)
            targets = targets.view(-1) # Flatten to (batch_size * max_sent_len)

            # Run Model & calculate loss
            tag_scores = model(word_indexes, char_indexes)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
    
    # 6. Save model
    model.hidden = model.init_hidden(1)
    torch.save(model, model_file)

    end_time = datetime.datetime.now()
    print('Time:', end_time - start_time)
    print('Finished...')
		
if __name__ == "__main__":
    # make no changes here
    train_file = sys.argv[1]
    model_file = sys.argv[2]
    train_model(train_file, model_file)
