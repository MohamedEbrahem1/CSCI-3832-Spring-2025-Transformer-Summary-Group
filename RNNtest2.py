import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch import nn
from torch import optim
import numpy as np
import copy
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tqdm import tqdm

# Load and prepare GloVe embeddings
glove_file = 'glove.6B.50d.txt'
embeddings_dict = {}
with open(glove_file, 'r', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_dict[word] = vector

# Create vocabulary mapping
vocab_size = len(embeddings_dict) + 2  # +2 for <pad> and <unk>
embedding_dim = 50
word2id = {'<pad>': 0, '<unk>': 1}
embedding_matrix = np.zeros((vocab_size, embedding_dim))

# Initialize <unk> embeddings
embedding_matrix[1] = np.random.uniform(-0.1, 0.1, embedding_dim)

# Fill embeddings matrix
current_idx = 2
for word in embeddings_dict:
    if current_idx >= vocab_size:
        break
    word2id[word] = current_idx
    embedding_matrix[current_idx] = embeddings_dict[word]
    current_idx += 1

print(f'Created embedding matrix with {current_idx} words')

class SummarizationDataset(Dataset):
    def __init__(self, texts, summaries, word2id, max_length):
        self.texts = texts
        self.summaries = summaries
        self.word2id = word2id
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        summary = self.summaries[idx]

        # Tokenize text
        text_tokens = word_tokenize(text.lower())
        text_ids = [self.word2id.get(tok, 1) for tok in text_tokens[:self.max_length]]
        text_padded = text_ids + [0]*(self.max_length - len(text_ids))
        text_len = min(len(text_ids), self.max_length)

        # Tokenize summary
        summary_tokens = word_tokenize(summary.lower())
        summary_ids = [self.word2id.get(tok, 1) for tok in summary_tokens[:self.max_length]]
        summary_padded = summary_ids + [0]*(self.max_length - len(summary_ids))
        summary_len = min(len(summary_ids), self.max_length)

        return (torch.tensor(text_padded), torch.tensor(text_len)), (torch.tensor(summary_padded), torch.tensor(summary_len))

class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.embedding.weight.requires_grad = False  # Freeze embeddings

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, inputs, input_lengths):
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(embedded, input_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, (hn, cn) = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        logits = self.fc(output)
        return logits

def train_model(model, train_dataset, valid_dataset, epochs=5, batch_size=32, lr=0.001):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_loss = float('inf')
    best_weights = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for (texts, text_lens), (summaries, _) in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            optimizer.zero_grad()
            outputs = model(texts, text_lens)
            loss = criterion(outputs.view(-1, vocab_size), summaries.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for (texts, text_lens), (summaries, _) in valid_loader:
                outputs = model(texts, text_lens)
                val_loss += criterion(outputs.view(-1, vocab_size), summaries.view(-1)).item()
        avg_val_loss = val_loss / len(valid_loader)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_weights = copy.deepcopy(model.state_dict())
        print(f"Validation Loss: {avg_val_loss:.4f}\n")

    model.load_state_dict(best_weights)
    return model

def generate_summary(model, text, word2id, id2word, max_length=50):
    model.eval()
    tokens = word_tokenize(text.lower())[:max_length]
    input_ids = [word2id.get(tok, 1) for tok in tokens] + [0]*(max_length - len(tokens))
    input_tensor = torch.tensor([input_ids])
    length_tensor = torch.tensor([min(len(tokens), max_length)])

    with torch.no_grad():
        outputs = model(input_tensor, length_tensor)
        predicted = outputs.argmax(-1).squeeze().tolist()

    summary = []
    for id in predicted[:length_tensor.item()]:
        if id == 0:
            break
        summary.append(id2word.get(id, '<unk>'))
    return ' '.join(summary)

# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv('Reviews.csv')  # Ensure 'Text' and 'Summary' columns exist
    texts = df['Text'].tolist()[:1000]  # Sample data
    summaries = df['Summary'].tolist()[:1000]

    # Prepare datasets
    max_seq_length = 30
    train_size = int(0.8 * len(texts))
    train_data = SummarizationDataset(texts[:train_size], summaries[:train_size], word2id, max_seq_length)
    valid_data = SummarizationDataset(texts[train_size:], summaries[train_size:], word2id, max_seq_length)

    # Initialize model
    model = LSTM(vocab_size, 50, 128)
    model = train_model(model, train_data, valid_data, epochs=5, batch_size=32)

    # Generate example summaries
    id2word = {v:k for k,v in word2id.items()}
    test_texts = [
        "This product works amazingly well! It's easy to use and very effective.",
        "Not satisfied with the quality. It broke after just a few days of use."
    ]

    for text in test_texts:
        summary = generate_summary(model, text, word2id, id2word, max_seq_length)
        print(f"Original: {text}\nSummary: {summary}\n{'-'*50}")