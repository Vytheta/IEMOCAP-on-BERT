import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


"""Création des Dataloaders destinés à la cellule GRU"""


from sklearn.metrics import f1_score

def f1_score_func(preds, labels):

    preds_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, preds_flat, average='weighted')

def inference(net, X):

    probs = net.forward(X)[-1]
    labels = np.argmax(probs, 1)
    return labels, probs




means = np.zeros(768)

for i in os.listdir("AllFiles"):
    f = torch.load("AllFiles/" + i)
    for n in range(len(f)):
        A = f[n].detach().numpy()
        for k in range(len(A[0])):
            means[k] += A[0][k] / 151

vars = np.zeros(768)

for i in os.listdir("AllFiles"):
    f = torch.load("AllFiles/" + i)
    for n in range(len(f)):
        A = f[n].detach().numpy()
        for k in range(len(A[0])):
            vars[k] += (A[0][k] - means[k]) ** 2
var = vars / 151
sigma = np.sqrt(var)



tab = []
for i in os.listdir("Train/TrainData"):
    a = torch.load("Train/TrainData/" + i)
    bit = []
    for n in range(167):
        A = a[n].detach().numpy()
        #bit.append(A[0])
        bit.append((A[0][:] - means) / np.sqrt(sigma ** 2 + 1e-5))
    bit2 = np.stack(bit)
    tab.append(bit2)
Xtrain = np.stack(tab)

Tab = []
for i in os.listdir("Test/TestData"):
    a = torch.load("Test/TestData/" + i)
    bit = []
    for n in range(167):
        A = a[n].detach().numpy()
        #print("A", A)
        #bit.append(A[0])
        bit.append((A[0][:] - means) / np.sqrt(sigma ** 2 + 1e-15))
    bit2 = np.stack(bit)
    Tab.append(bit2)
Xtest = np.stack(Tab)

taB = []
for i in os.listdir("Validation/ValidationData"):
    a = torch.load("Validation/ValidationData/" + i)
    bit = []
    for n in range(167):
        A = a[n].detach().numpy()
        #print("A", A)
        #bit.append(A[0])
        bit.append((A[0][:] - means) / np.sqrt(sigma ** 2 + 1e-15))
    bit2 = np.stack(bit)
    taB.append(bit2)
Xval = np.stack(taB)

print(Xtrain.shape, Xtest.shape, Xval.shape)



tab = []
for i in os.listdir("Train/TrainClasses"):
    f = open("Train/TrainClasses/" + i)
    f1 = f.read()
    F = f1.splitlines()
    bit = []
    for n in range(len(F)):
        #bit.append(int(F[n]))
        vec = np.zeros(10)
        vec[int(F[n])] += 1
        bit.append(vec)
    tab.append(bit)
Ytrain = np.stack(tab)

tab = []
for i in os.listdir("Test/TestClasses"):
    f = open("Test/TestClasses/" + i)
    f1 = f.read()
    F = f1.splitlines()
    bit = []
    for n in range(len(F)):
        #bit.append(int(F[n]))
        vec = np.zeros(10)
        vec[int(F[n])] += 1
        bit.append(vec)
    tab.append(bit)
Ytest = np.stack(tab)

tab = []
for i in os.listdir("Validation/ValidationClasses"):
    f = open("Validation/ValidationClasses/" + i)
    f1 = f.read()
    F = f1.splitlines()
    bit = []
    for n in range(len(F)):
        #bit.append(int(F[n]))
        vec = np.zeros(10)
        vec[int(F[n])] += 1
        bit.append(vec)
    tab.append(bit)
Yval = np.stack(tab)

print(Ytrain.shape, Ytest.shape, Yval.shape)

#print(Xtrain)

train_data = TensorDataset(torch.from_numpy(Xtrain), torch.from_numpy(Ytrain))
test_data = TensorDataset(torch.from_numpy(Xtest), torch.from_numpy(Ytest))
val_data = TensorDataset(torch.from_numpy(Xval), torch.from_numpy(Yval))

batch_size = 10

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size, drop_last=True)
test_loader = DataLoader(test_data, batch_size=batch_size, drop_last=True)
val_loader = DataLoader(val_data, batch_size=batch_size, drop_last=True)

torch.save(train_loader, "train_loader.pt")
torch.save(test_loader, "test_loader.pt")
torch.save(val_loader, "val_loader.pt")







