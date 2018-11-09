# python3.5 run_tagger.py <test_file_absolute_path> <model_file_absolute_path> <output_file_absolute_path>
import os
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:0")

class PosTagModel(nn.Module):
    def __init__(self, word_embedding_dim, word_vocab_size, hidden_dim, tagset_size, word_padding_index):
        super(PosTagModel, self).__init__()
        self.hidden_dim = hidden_dim

        # RNN with LSTM
        self.word_embeddings = nn.Embedding(word_vocab_size, word_embedding_dim, padding_idx=word_padding_index).to(device)
        self.lstm = nn.LSTM(word_embedding_dim, hidden_dim, num_layers=4, dropout=0.5, bidirectional=True, batch_first=True).to(device)
        self.hidden_to_tag = nn.Linear(hidden_dim * 2, tagset_size).to(device)

    def init_hidden(self, batch_size):
        return (torch.zeros(self.hidden_layers * 2, batch_size, self.hidden_dim).to(device),
                torch.zeros(self.hidden_layers * 2, batch_size, self.hidden_dim).to(device))

    def forward(self, char_indexes, word_indexes):
        # CNN -- old (With padding)
        # char_vecs = []
        # for word in sentence.split():
        #     indexes = []
        #     for char in word:
        #         if char in self.char_to_index:
        #             indexes.append(self.char_to_index[char])
        #         else:
        #             indexes.append(self.char_to_index[self.unknown_element])
        #     word_char_indexes = torch.tensor(indexes, dtype=torch.long).to(device)
        #     char_embeds = self.char_embeddings(word_char_indexes)
        #     char_embeds = torch.unsqueeze(char_embeds, 2) # Add additional dimensions
        #     char_cnn_out = self.cnn(char_embeds)
        #     char_cnn_out = torch.squeeze(char_cnn_out)
        #     if (char_cnn_out.size(0) != 32):
        #         char_cnn_out, _ = torch.max(char_cnn_out, 0)
        #     char_vecs.append(char_cnn_out)
        # char_vecs = torch.stack(char_vecs)

        # CNN
        char_embeds = self.char_embeddings(char_indexes)
        batch_size = 1
        num_words = char_embeds.shape[0]
        num_chars = char_embeds.shape[1]
        char_dim = char_embeds.shape[2]
        char_embeds = char_embeds.view(batch_size * num_words, char_dim, num_chars)
        char_cnn_out = self.cnn(char_embeds) # (1 * num_words) * convo_layer * num_char
        char_cnn_out, _ = torch.max(char_cnn_out, 2) # (1 * num_words) * convo_layer
        char_cnn_out = char_cnn_out.view(batch_size, num_words, self.convo_layer) # 1 * num_words * convo_layer

        # LSTM
        embeds = self.word_embeddings(word_indexes) # 1 x max_sent_len * word_dim
        embeds = torch.unsqueeze(embeds, 0)
        embeds = torch.cat((embeds, char_cnn_out), 2) # 1 x max_sent_len * (word_dim + convo_layer)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out = lstm_out.contiguous()
        tag_space = self.hidden_to_tag(lstm_out.view(-1, lstm_out.shape[2]))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def prepare_sequence(self, sentence):
        char_indexes =  []
        word_indexes = []
        word_lengths = []

        # For RNN
        for word in sentence:
            if word in self.word_to_index:
                word_indexes.append(self.word_to_index[word])
            else:
                word_indexes.append(self.word_to_index[self.unknown_element])
            word_lengths.append(len(word))

        # For CNN (With padding)
        max_word_len = max(word_lengths)
        for word in sentence:
            char_index = []
            word_len = len(word)
            for char in word:
                if char in self.char_to_index:
                    char_index.append(self.char_to_index[char])
                else:
                    char_index.append(self.char_to_index[self.unknown_element])
            # Pad characters
            for j in range(max_word_len - word_len):
                char_index.append(self.char_to_index["<PAD>"])
            char_index = torch.tensor(char_index, dtype=torch.long).to(device)
            char_indexes.append(char_index)

        char_indexes = torch.stack(char_indexes) # dim: word_len * char_len
        word_indexes = torch.tensor(word_indexes, dtype=torch.long).to(device) # dim: word_len
        return char_indexes, word_indexes

def tag_sentence(test_file, model_file, out_file):
    # Load model
    model = torch.load(model_file)
    if torch.cuda.is_available():
        model.cuda()
    
    # Read in test data and store in sentences array
    with open(test_file, "r") as ins:
        test_sentences = []
        for line in ins:
            test_sentences.append(line)

    # Run model against test
    sentences_out = []
    for sentence in tqdm(test_sentences):
        with torch.no_grad():
            words_in_sentence = sentence.split()
            char_indexes, word_indexes = model.prepare_sequence(words_in_sentence)
            tag_scores = model(char_indexes, word_indexes)

            # Transform indexes into POS tags
            word_with_tag_sentence = []
            for i in range(0, len(words_in_sentence)):
                word = words_in_sentence[i]
                _, tag_index = tag_scores[i].max(0)
                tag = model.index_to_tag[tag_index.item()]
                word_with_tag = word + "/" + tag
                word_with_tag_sentence.append(word_with_tag)
            sentences_out.append(word_with_tag_sentence)

    # Write to sents.out
    with open(out_file, "w") as data:
        for iter, sentence in enumerate(sentences_out):
            sentence = " ".join(sentence)
            data.write(sentence)
            data.write("\n")

    print('Finished...')

if __name__ == "__main__":
    # make no changes here
    test_file = sys.argv[1]
    model_file = sys.argv[2]
    out_file = sys.argv[3]
    tag_sentence(test_file, model_file, out_file)