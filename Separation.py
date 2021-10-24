import numpy as np
import torch
import os
from random import shuffle


"""Création des piles train, validation et test"""


tab1 = []
tab2 = []

for i in os.listdir("AllFiles"):
    tab1.append(str(i))

T = np.array(tab1)
print(T)
shuffle(T)
print("T1", T)

for k in range(111):
    t = torch.load("AllFiles/" + T[k])
    torch.save(t, "Train/TrainData/" + T[k])

#for k in range(111):
#    f = open("Train/TrainIndices/" + T[k][:-3], "w")
#    g = open("AllIndices/" + T[k][:-3])
#    g1 = g.read()
#    G = g1.splitlines()
#    for n in range(len(G)):
#        print(G[n], file=f)
#    f.close()
#    g.close()

for k in range(111):
    f = open("Train/TrainClasses/" + T[k][:-3], "w")
    g = open("AllClasses/" + T[k][:-3])
    g1 = g.read()
    G = g1.splitlines()
    for n in range(len(G)):
        print(G[n], file=f)
    f.close()
    g.close()


for k in range(111, 131):
    t = torch.load("AllFiles/" + T[k])
    torch.save(t, "Validation/ValidationData/" + T[k])

#for k in range(111, 131):
#    f = open("Validation/ValidationIndices/" + T[k][:-3], "w")
#    g = open("AllIndices/" + T[k][:-3])
#    g1 = g.read()
#    G = g1.splitlines()
#    for n in range(len(G)):
#        print(G[n], file=f)
#    f.close()
#    g.close()

for k in range(111, 131):
    f = open("Validation/ValidationClasses/" + T[k][:-3], "w")
    g = open("AllClasses/" + T[k][:-3])
    g1 = g.read()
    G = g1.splitlines()
    for n in range(len(G)):
        print(G[n], file=f)
    f.close()
    g.close()


for k in range(131, 151):
    t = torch.load("AllFiles/" + T[k])
    torch.save(t, "Test/TestData/" + T[k])


#for k in range(131, 151):
#    f = open("Test/TestIndices/" + T[k][:-3], "w")
#    g = open("AllIndices/" + T[k][:-3])
#    g1 = g.read()
#    G = g1.splitlines()
#    for n in range(len(G)):
#        print(G[n], file=f)
#    f.close()
#    g.close()

for k in range(131, 151):
    f = open("Test/TestClasses/" + T[k][:-3], "w")
    g = open("AllClasses/" + T[k][:-3])
    g1 = g.read()
    G = g1.splitlines()
    for n in range(len(G)):
        print(G[n], file=f)
    f.close()
    g.close()
