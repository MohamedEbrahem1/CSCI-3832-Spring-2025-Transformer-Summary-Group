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


def predict(model, valid_dataloader):

    sigmoid = nn.Sigmoid()

    total_correct = 0
    total_examples = len(valid_dataloader.dataset)

    for (x, x_lengths), y in valid_dataloader:

        output = sigmoid(model(x, x_lengths))

        for i in range(output.shape[0]):
            if (output[i] < 0.5 and y[i] == 0) or (output[i] >= 0.5 and y[i] == 1):
                total_correct += 1

    accuracy = total_correct / total_examples
    print("accuracy: {}".format(accuracy))
    return accuracy

def train_lstm_classification(model, train_dataset, valid_dataset, epochs=10, batch_size=32, learning_rate=.001, print_frequency=25):

    criteria = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    epochs = epochs
    batch_size = batch_size
    print_frequency = print_frequency

    #We'll create an instance of a torch dataloader to collate our data. This class handles batching and shuffling (should be done each epoch)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=128, shuffle=False)

    print('Total train batches: {}'.format(train_dataset.__len__() / batch_size))

    best_accuracy = 0.0
    best_model_sd = None

    for i in range(epochs):
        print('### Epoch: ' + str(i+1) + ' ###')
    
        model.train()

        avg_loss = 0

        for step, data in enumerate(train_dataloader):

            (x, x_lengths), y = data	# Our dataset is returning the input example x and also the lengths of the examples, so we'll unpack that here

            optimizer.zero_grad()

            model_output = model(x, x_lengths)

            loss = criteria(model_output.squeeze(1), y.float())

            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if step % print_frequency == (print_frequency - 1):
                print('epoch: {} batch: {} loss: {}'.format(
                    i,
                    step,
                    avg_loss / print_frequency
                ))
                avg_loss = 0

        print('Evaluating...')
        model.eval()
        with torch.no_grad():
            acc = predict(model, valid_dataloader)
            if acc > best_accuracy:
                best_model_sd = copy.deepcopy(model.state_dict())
                best_accuracy = acc

    return model.state_dict(), best_model_sd

#model = LSTMModel(embedding_matrix, lstm_hidden_size=50, num_lstm_layers=2, bidirectional=True)
#model, best_model = train_lstm_classification(model, train_dataset, valid_dataset, batch_size=128, epochs=1)