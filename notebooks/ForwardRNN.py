import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from nltk.tokenize import word_tokenize
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import tqdm
import csv

# This was created by me (Tian) with help from DeepSeek, I made most of the LSTM Architecture and the Training Loop
# and Deepseek helped me refine and debug those parts and write the summarization and collate functions.
# Will need help polishing and debugging and also writting and documenting

# Load limited GloVe embeddings for performance and testing and preprocess words
GLOVE_LIMIT = 150000  # Reduced vocabulary size
glove_file = "glove.6B.50d.txt"
word2id = {"<pad>": 0, "<unk>": 1}
embedding_matrix = []

with open(glove_file, "r", encoding="utf-8") as f:
    lines = []
    for i, line in enumerate(f):
        if i >= GLOVE_LIMIT:
            break
        lines.append(line)

    embedding_matrix = np.zeros((len(lines) + 2, 50), dtype="float32")
    for idx, line in enumerate(tqdm(lines, desc="Loading GloVe"), start=2):
        parts = line.split()
        word2id[parts[0]] = idx
        embedding_matrix[idx] = np.array(parts[1:], dtype="float32")

embedding_matrix[1] = np.random.uniform(-0.1, 0.1, 50).astype("float32")

# Text Dataset class was made by using the Movie Dataset class from HW3 as a basis,
# DeepSeek helped me write the collate function for optimization and speed when
# processing data. The rest of the structure was coded by me with DeepSeek debugging


class TextDataset(Dataset):
    def __init__(self, csv_file, word2id, max_length=64, sample_size=5000):
        self.df = pd.read_csv(csv_file, usecols=["Text"]).sample(sample_size)
        self.word2id = word2id
        self.max_length = max_length
        self.data = self._preprocess()

    def _preprocess(self):
        processed = []
        for text in tqdm(self.df["Text"], desc="Tokenizing"):
            tokens = word_tokenize(text.lower())[: self.max_length]
            ids = [self.word2id.get(tok, 1) for tok in tokens]
            processed.append((ids, len(ids)))
        return processed

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# This function was made by DeepSeek for optimization and Performance when processing
def collate_fn(batch):
    inputs, lengths = zip(*batch)
    sorted_idx = np.argsort(lengths)[::-1]
    sorted_inputs = [inputs[i] for i in sorted_idx]
    sorted_lengths = [lengths[i] for i in sorted_idx]

    max_len = max(sorted_lengths)
    padded = torch.zeros((len(sorted_inputs), max_len), dtype=torch.long)
    for i, seq in enumerate(sorted_inputs):
        padded[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    return padded, torch.tensor(sorted_lengths, dtype=torch.long)


# Simple forward LSTM architecture made by me using HW3 as a basis
class SummarizationLSTM(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            torch.tensor(embedding_matrix), padding_idx=0, freeze=False
        )
        self.lstm = nn.LSTM(50, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, len(word2id))

    def forward(self, inputs, lengths):
        embedded = self.embedding(inputs)
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(packed)

        # Unpack and pad to original sequence length
        output_unpacked, _ = pad_packed_sequence(
            output,
            batch_first=True,
            total_length=inputs.size(1),  # Match input sequence length
        )
        return self.fc(output_unpacked)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training model made using class RNN Seq labelling example and HW3
def train_model(csv_file, batch_size=16, num_epochs=10):
    dataset = TextDataset(csv_file, word2id)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=0
    )

    #
    model = SummarizationLSTM(embedding_matrix).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for i, (inputs, lengths) in enumerate(progress):
            inputs, lengths = inputs.to(device), lengths.to(device)

            outputs = model(inputs, lengths)

            # Proper dimension handling
            loss = criterion(
                outputs.view(-1, outputs.size(-1)),  # (batch*seq_len, vocab_size)
                inputs.view(-1),  # (batch*seq_len)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix({"loss": f"{total_loss/(i+1):.4f}"})

    return model


# This function was mainly written by DeepSeek with me debugging and modifying chunks here and there
# function predicts the most likely suquential token, then writes the summary by mapping the token
# its word


def generate_summary(model, text, word2id, max_length=20):
    model.eval()
    tokens = word_tokenize(text.lower())[:64]
    input_ids = [word2id.get(tok, 1) for tok in tokens]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    length_tensor = torch.tensor([len(input_ids)], dtype=torch.long).to(device)

    with torch.no_grad():
        embedded = model.embedding(input_tensor)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, length_tensor, batch_first=True
        )
        output, _ = model.lstm(packed)
        logits = model.fc(output.data)  # Use output.data for unpacked sequence
        preds = logits.argmax(-1).cpu().numpy()

    id2word = {v: k for k, v in word2id.items()}
    return " ".join([id2word.get(idx, "<unk>") for idx in preds if idx != 0])


def process_reviews(csv_file_path, model, word2id):
    # Open the CSV file
    with open(csv_file_path, mode="r", encoding="utf-8") as file:
        csv_reader = csv.DictReader(file)

        # Process each row
        for row in csv_reader:
            # Get the review text
            review_text = row["Review"]

            # Generate summary for this review
            summary = generate_summary(model, review_text, word2id)

            # Print or store the results as needed
            print(f"Original Review: {review_text}")
            print(f"Generated Summary: {summary}\n")
            print("-" * 50)  # Separator for readability


if __name__ == "__main__":
    data_path = "../data/"
    model = train_model(data_path + "Reviews.csv")

    process_reviews(data_path + "test_data/" + "item1.csv", model, word2id)
    process_reviews(data_path + "test_data/" + "item2.csv", model, word2id)
    process_reviews(data_path + "test_data/" + "item3.csv", model, word2id)
