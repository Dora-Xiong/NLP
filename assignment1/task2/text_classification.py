import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import jieba
import numpy as np
import os
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt

# File paths
train_file = "train.txt"
dev_file = "dev.txt"
test_file = "test.txt"

# Data Preparation
class TextDataset(Dataset):
    def __init__(self, filepath, vocab=None, is_train=True):
        self.sentences = []
        self.labels = []
        self.vocab = vocab
        self.is_train = is_train
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split("\t")
                assert len(parts) == 2 
                sentence, label = parts
                words = list(jieba.cut(sentence, cut_all=False))
                self.sentences.append(words)
                self.labels.append(int(label))
        if is_train:
            self.vocab = self.build_vocab(self.sentences)
        self.indexed_sentences = [self.sentence_to_index(s) for s in self.sentences]

    def build_vocab(self, sentences):
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for sentence in sentences:
            for word in sentence:
                if word not in vocab:
                    vocab[word] = len(vocab)
        return vocab

    def sentence_to_index(self, sentence):
        return [self.vocab.get(word, self.vocab["<UNK>"]) for word in sentence]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return torch.tensor(self.indexed_sentences[index]), torch.tensor(self.labels[index])
    
# padding
def collate_fn(batch):
    sentences, labels = zip(*batch)
    # 对句子进行 padding
    sentences = pad_sequence(sentences, batch_first=True, padding_value=0)  # 使用 <PAD> 的索引 0 进行补齐
    labels = torch.tensor(labels)
    return sentences, labels


# CLassifier
class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, kernel_sizes=(3, 4, 5), num_channels=100, dropout=0.5):
        super(CNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv2d(1, num_channels, (k, embed_size)) for k in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_channels, num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)  # Add channel dimension
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs]  # Apply convolution and squeeze
        x = [F.max_pool1d(item, item.size(2)).squeeze(2) for item in x]  # Apply max pooling
        x = torch.cat(x, 1)  # Concatenate along channel dimension
        x = self.dropout(x)
        x = self.fc(x)
        return self.sigmoid(x)

# Training and Evaluation
def train_and_evaluate(train_file, dev_file, test_file, epochs=20, batch_size=64, embed_size=128, lr=0.001):
    train_data = TextDataset(train_file)
    dev_data = TextDataset(dev_file, vocab=train_data.vocab, is_train=False)
    test_data = TextDataset(test_file, vocab=train_data.vocab, is_train=False)
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    model = CNNClassifier(vocab_size=len(train_data.vocab), embed_size=embed_size, num_classes=len(set(train_data.labels)))
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping setup
    best_dev_acc = 0
    patience, trials = 5, 0
    
    x_s = []
    training_losses = []
    dev_accs = []
    
    for epoch in range(epochs):
        model.train()
        training_loss = []
        for sentences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sentences)
            loss = criterion(outputs, labels)
            training_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            
        x_s.append(epoch)
        training_losses.append(np.mean(training_loss))

        # Evaluate on dev set
        dev_acc = evaluate(model, dev_loader)
        dev_accs.append(dev_acc)
        print(f"Epoch {epoch+1}, Dev Accuracy: {dev_acc:.4f}")

        # Early stopping
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            trials = 0
        else:
            trials += 1
            if trials >= patience:
                torch.save(model.state_dict(), "best_model.pth")
                plt.plot(x_s, training_losses, label='Training Loss')
                plt.plot(x_s, dev_accs, label='Dev Accuracy')
                plt.legend()
                plt.show()
                print("Early stopping...")
                break

    # Test set evaluation
    test_acc = evaluate(model, test_loader)
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc

# Evaluation function
def evaluate(model, data_loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for sentences, labels in data_loader:
            outputs = model(sentences)
            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    return accuracy_score(targets, preds)

# Run the training
train_and_evaluate(train_file, dev_file, test_file)


