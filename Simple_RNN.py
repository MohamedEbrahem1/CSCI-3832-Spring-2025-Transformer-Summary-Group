import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
from torch import optim

# Implemented by using class notes, pytorch.org tutorials, and modifying them but also using DeepSeek for ideas here and there


class LSTM(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        self.classification_hidden_layer = nn.Linear(hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()

    def forward(self, inputs, input_lengths):

        embedding = self.embedding(inputs)

        packed_input = pack_padded_sequence(
            embedding, input_lengths, batch_first=True, enforce_sorted=False
        )

        output, (hn, _) = self.lstm(
            packed_input
        )  # What is the difference between output and hn?

        output, output_lengths = pad_packed_sequence(output, batch_first=True)

        # print(output.shape) # Shape: batch_size x sequence_length x hidden_dim

        h1 = self.classification_hidden_layer(output)

        h1 = self.relu(h1)

        final_output = self.output_layer(h1)

        return final_output
