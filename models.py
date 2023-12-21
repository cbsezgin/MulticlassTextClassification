import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score


class RNNNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNNetwork, self).__init__()

        self.rnn = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_data):
        _, hidden = self.rnn(input_data)
        output = self.linear(hidden)
        return output


class LSTMNetwork(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMNetwork, self).__init__()

        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.linear = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, input_data):
        _, (hidden, _) = self.lstm(input_data)
        output = self.linear(hidden[-1])
        return output

def train(train_loader,valid_loader, model, criterion, optimizer, device, epochs, model_path):
    best_loss = 1e8
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        valid_loss = []
        train_loss = []

        model.train()

        for batch_labels, batch_data in tqdm(train_loader):
            batch_labels = batch_labels.to(device).type(torch.LongTensor)
            batch_data = batch_data.to(device)

            batch_output = model(batch_data)
            batch_output = torch.squeeze(batch_output)

            loss = criterion(batch_output, batch_labels)
            train_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()

        for batch_labels, batch_data in tqdm(valid_loader):
            batch_labels = batch_labels.to(device).type(torch.LongTensor)
            batch_data = batch_data.to(device)

            batch_output = model(batch_data)
            batch_output = torch.squeeze(batch_output)

            loss = criterion(batch_output, batch_labels)
            valid_loss.append(loss.item())

        t_loss = np.mean(train_loss)
        v_loss = np.mean(valid_loss)
        print(f"Train Loss: {t_loss:.4f}, Val Loss: {v_loss:.4f}")

        if v_loss<best_loss:
            best_loss = v_loss
            torch.save(model.state_dict(), model_path)


def test(test_loader, model, criterion, device):
    model.eval()
    test_loss = []
    test_accuracy = []
    for batch_labels, batch_data in tqdm(test_loader):
        batch_labels = batch_labels.to(device).type(torch.LongTensor)
        batch_data = batch_data.to(device)

        batch_output = model(batch_data)
        batch_output = torch.squeeze(batch_output)

        loss = criterion(batch_output, batch_labels)
        test_loss.append(loss.item())
        batch_preds = torch.argmax(batch_output, axis=1)

        if torch.cuda.is_available():
            batch_labels = batch_labels.cpu()
            batch_preds = batch_preds.cpu()

        test_accuracy.append(accuracy_score(batch_labels.detach().numpy(), batch_preds.detach().numpy()))

    test_loss = np.mean(test_loss)
    test_accuracy = np.mean(test_accuracy)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy:  {test_accuracy:.4f}")

