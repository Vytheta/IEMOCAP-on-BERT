import torch
import pandas as pd
import numpy as np
import csv
import os


for n in range(5):

    for i in os.listdir("ImproActorLabel/session" + str(n + 1) + "/F"):
        alpha = open("ImproActorLabel/session" + str(n + 1) + "/F/" + i)
        beta = alpha.read()
        f = open("AllIndices/" + "IF" + str(n + 1) + i[5:], "w")
        print(beta, file=f)
        f.close()

    for i in os.listdir("ImproActorLabel/session" + str(n + 1) + "/M"):
        alpha = open("ImproActorLabel/session" + str(n + 1) + "/M/" + i)
        beta = alpha.read()
        f = open("AllIndices/" + "IM" + str(n + 1) + i[5:], "w")
        print(beta, file=f)
        f.close()

    for i in os.listdir("ScriptActorLabel/session" + str(n + 1) + "/F"):
        alpha = open("ScriptActorLabel/session" + str(n + 1) + "/F/" + i)
        beta = alpha.read()
        f = open("AllIndices/" + "SF" + str(n + 1) + i[6:], "w")
        print(beta, file=f)
        f.close()

    for i in os.listdir("ScriptActorLabel/session" + str(n + 1) + "/M"):
        alpha = open("ScriptActorLabel/session" + str(n + 1) + "/M/" + i)
        beta = alpha.read()
        f = open("AllIndices/" + "SM" + str(n + 1) + i[6:], "w")
        print(beta, file=f)
        f.close()