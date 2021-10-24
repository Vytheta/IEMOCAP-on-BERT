import numpy as np
import torch
import pickle
import os
from transformers import BertModel, BertConfig
from torch.utils.data import DataLoader, SequentialSampler


"""Création des Dataloaders destinés à BERT"""


for n in range(5):

    for i in os.listdir("TensorsImpro/session" + str(n + 1) + "/F"):
        F = torch.load("TensorsImpro/session" + str(n + 1) + "/F/" + i)
        dataloader = DataLoader(F, sampler=SequentialSampler(F))
        torch.save(dataloader, "DataLoadersImpro/session" + str(n + 1) + "/F/" +
                   str(n + 1) + 'dataloaderFImpro' + i[16:-3] + '.pt')

    for i in os.listdir("TensorsImpro/session" + str(n + 1) + "/M"):
        F = torch.load("TensorsImpro/session" + str(n + 1) + "/M/" + i)
        dataloader = DataLoader(F, sampler=SequentialSampler(F))
        torch.save(dataloader, "DataLoadersImpro/session" + str(n + 1) + "/M/" +
                   str(n + 1) + 'dataloaderMImpro' + i[16:-3] + '.pt')

    for i in os.listdir("TensorsScript/session" + str(n + 1) + "/F"):
        F = torch.load("TensorsScript/session" + str(n + 1) + "/F/" + i)
        dataloader = DataLoader(F, sampler=SequentialSampler(F))
        torch.save(dataloader, "DataLoadersScript/session" + str(n + 1) + "/F/" +
                   str(n + 1) + 'dataloaderFScript' + i[17:-3] + '.pt')

    for i in os.listdir("TensorsScript/session" + str(n + 1) + "/M"):
        F = torch.load("TensorsScript/session" + str(n + 1) + "/M/" + i)
        dataloader = DataLoader(F, sampler=SequentialSampler(F))
        torch.save(dataloader, "DataLoadersScript/session" + str(n + 1) + "/M/" +
                   str(n + 1) + 'dataloaderMScript' + i[17:-3] + '.pt')