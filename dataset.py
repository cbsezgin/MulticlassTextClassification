import torch


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, tokens, labels, embeddings):
        self.tokens = tokens
        self.labels = labels
        self.embeddings = embeddings

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        return self.labels[idx], self.embeddings[self.tokens[idx], :]

