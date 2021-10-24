import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
from torch.nn import GRU, GRUCell
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm



"""Batch Normalization + création des Dataloaders destinés à la cellule GRU + entraînement GRU"""


#device = torch.device("cuda")
device = torch.device("cpu")
print("ok", device)

from sklearn.metrics import f1_score


def f1_score_func(preds, labels):
    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')


def inference(net, X):
    probs = net.forward(X)[-1]
    labels = np.argmax(probs, 1)
    return labels, probs


def softmax(Z):
    E = np.exp(Z - Z.max(1, keepdims=True))
    return E / E.sum(1, keepdims=True)


means = np.zeros(768)

for i in os.listdir("AllFiles"):
    f = torch.load("AllFiles/" + i)
    for n in range(len(f)):
        A = f[n].detach().numpy()
        for k in range(len(A[0])):
            means[k] += A[0][k] / 151
            # print(means[k])

variances = np.zeros(768)

for i in os.listdir("AllFiles"):
    f = torch.load("AllFiles/" + i)
    for n in range(len(f)):
        A = f[n].detach().numpy()
        for k in range(len(A[0])):
            variances[k] += (A[0][k] - means[k]) ** 2

var = variances / 151
sigma = np.sqrt(var)

#print(means)
#print(sigma)

tab = []
for i in os.listdir("Train/TrainData"):
    a = torch.load("Train/TrainData/" + i)
    bit = []
    for n in range(167):
        A = a[n].detach().numpy()
        bit.append((A[0][:] - means) / np.sqrt(sigma ** 2 + 1e-5))
    bit2 = np.stack(bit)
    tab.append(bit2)
Xtrain = np.stack(tab)

taB = []
for i in os.listdir("Test/TestData"):
    a = torch.load("Test/TestData/" + i)
    bit = []
    for n in range(167):
        A = a[n].detach().numpy()
        # print("A", A)
        bit.append((A[0][:] - means) / np.sqrt(sigma ** 2 + 1e-5))
    bit2 = np.stack(bit)
    taB.append(bit2)
Xtest = np.stack(taB)

Tab = []
for i in os.listdir("Validation/ValidationData"):
    a = torch.load("Validation/ValidationData/" + i)
    bit = []
    for n in range(167):
        A = a[n].detach()
        # print("A", A)
        bit.append((A[0][:] - means) / np.sqrt(sigma ** 2 + 1e-5))
    bit2 = np.stack(bit)
    Tab.append(bit2)
Xval = np.stack(Tab)

print(Xtrain.shape, Xtest.shape, Xval.shape)

tab = []
for i in os.listdir("Train/TrainClasses"):
    f = open("Train/TrainClasses/" + i)
    f1 = f.read()
    F = f1.splitlines()
    bit = []
    for n in range(len(F)):
        vec = np.zeros(10)
        vec[int(F[n])] += 1
        bit.append(vec)
    tab.append(bit)
Ytrain = np.stack(tab)

taB = []
for i in os.listdir("Test/TestClasses"):
    f = open("Test/TestClasses/" + i)
    f1 = f.read()
    F = f1.splitlines()
    bit = []
    for n in range(len(F)):
        vec = np.zeros(10)
        vec[int(F[n])] += 1
        bit.append(vec)
    taB.append(bit)
Ytest = np.stack(taB)

Tab = []
for i in os.listdir("Validation/ValidationClasses"):
    f = open("Validation/ValidationClasses/" + i)
    f1 = f.read()
    F = f1.splitlines()
    bit = []
    for n in range(len(F)):
        vec = np.zeros(10)
        vec[int(F[n])] += 1
        bit.append(vec)
    Tab.append(bit)
Yval = np.stack(Tab)

print(Ytrain.shape, Ytest.shape, Yval.shape)

xtrain = torch.from_numpy(Xtrain)
xtest = torch.from_numpy(Xtest)
xval = torch.from_numpy(Xval)
ytrain = torch.from_numpy(Ytrain)
ytest = torch.from_numpy(Ytest)
yval = torch.from_numpy(Yval)

#print(xtrain)

train_data = TensorDataset(xtrain, ytrain)
test_data = TensorDataset(xtest, ytest)
val_data = TensorDataset(xval, yval)

batch_size = 20

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_data, drop_last=True)
val_loader = DataLoader(val_data, drop_last=True)

torch.save(train_loader, "train_loader.pt")
torch.save(test_loader, "test_loader.pt")
torch.save(val_loader, "val_loader.pt")


class GRUNet(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, drop_prob=0.2):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, batch_first=True, dropout=drop_prob)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, h):
        out, h = self.gru(x, h)
        return out, h

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


def train(train_loader, learn_rate, hidden_dim=10, EPOCHS=100, model_type="GRU"):
    input_dim = 768
    output_dim = 10
    n_layers = 1

    model = GRUNet(input_dim, hidden_dim, output_dim, n_layers)
    model.to(device)

    # Defining loss function and optimizer
    LossFunction = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

    """TRAINING"""
    for epoch in range(1, EPOCHS + 1):
        h = model.init_hidden(batch_size)
        avg_loss = 0
        count = 0
        n = 10000
        while count <= n:
            for x, label in train_loader:
                count += 1
                h = h.data
                model.zero_grad()
                out, h = model.forward(x.to(device).float(), h)
                loss = LossFunction(out, label.to(device).float())
                loss.backward()
                optimizer.step()
                avg_loss += loss.item()
                if count % n == 0:
                    print("top")
                    tab = []
                    l = label.to(device)
                    for q in range(len(l)):
                        result = []
                        ech1 = l[q].detach().numpy()
                        outch1 = out[q].detach().numpy()
                        for r in range(len(ech1)):
                            ver1 = np.argmax(ech1[r])
                            ver2 = np.argmax(outch1[r])
                            test = int(ver1 == ver2)
                            result.append(test)
                        tab.append(np.mean(result))
                    MEAN = np.mean(tab) * 100
                    print("accuracy", MEAN)
                    print("Epoch {}......Step: {}/{}....... Average Loss for Epoch: {}".format(epoch, count,
                                                                                               len(train_loader),
                                                                                               avg_loss / count))
    return model


lr = 0.00001
gru_model = train(train_loader, lr, model_type="GRU")





