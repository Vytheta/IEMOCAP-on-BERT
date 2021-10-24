import torch
import pandas as pd
import numpy as np
import csv
import os


"Création des Datasets destinés à BERT"


from transformers import BertTokenizer
from torch.utils.data import TensorDataset

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)

for n in range(5):

    for i in os.listdir("FullImpro/session" + str(n + 1) + "/F"):
        f1 = open("FullImpro/session" + str(n + 1) + "/F/" + i)
        f = f1.read()
        f1.close()
        h = open("csvImproSentences/session" + str(n + 1) + "/F/" + i, "w")
        F = f.splitlines()
        sentences = []
        for j in range(len(F)):
            sentence = F[j].split(';')[1]
            sentences.append(sentence)
        tab = np.array(sentences)
        with open("CSVfiles/" + str(n + 1) + 'dfDataImproF' + i + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for r in range(len(F)):
                writer.writerow([r, tab[r]])
            for r in range(len(F), 167):
                writer.writerow([r, " "])
        dfData = pd.read_csv("CSVfiles/" + str(n + 1) + 'dfDataImproF' + i + '.csv', names=["turn", "speech"])
        dfData.set_index('turn', inplace=True)
        print(dfData, file=h)
        h.close()
        encoded_data = tokenizer.batch_encode_plus(dfData.speech.values, add_special_tokens=True,
                                                   return_attention_mask=True, truncation=True, padding='longest',
                                                   return_tensors='pt')

        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        dst_Data = TensorDataset(input_ids, attention_masks)
        torch.save(dst_Data, "TensorsImpro/session" + str(n + 1) + "/F/" +
                   str(n + 1) + 'ImproSentencesF' + i[5:] + '.pt')

    for i in os.listdir("FullImpro/session" + str(n + 1) + "/M"):
        f1 = open("FullImpro/session" + str(n + 1) + "/M/" + i)
        f = f1.read()
        f1.close()
        h = open("csvImproSentences/session" + str(n + 1) + "/M/" + i, "w")
        F = f.splitlines()
        sentences = []
        for j in range(len(F)):
            sentence = F[j].split(';')[1]
            sentences.append(sentence)
        tab = np.array(sentences)
        with open("CSVfiles/" + str(n + 1) + 'dfDataImproM' + i + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for r in range(len(F)):
                writer.writerow([r, tab[r]])
            for r in range(len(F), 167):
                writer.writerow([r, " "])
        dfData = pd.read_csv("CSVfiles/" + str(n + 1) + 'dfDataImproM' + i + '.csv', names=["turn", "speech"])
        dfData.set_index('turn', inplace=True)
        print(dfData, file=h)
        h.close()
        encoded_data = tokenizer.batch_encode_plus(dfData.speech.values, add_special_tokens=True,
                                                   return_attention_mask=True, truncation=True, padding='longest',
                                                   return_tensors='pt')
        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        dst_Data = TensorDataset(input_ids, attention_masks)
        torch.save(dst_Data, "TensorsImpro/session" + str(n + 1) + "/M/" +
                   str(n + 1) + 'ImproSentencesM' + i[5:] + '.pt')

    for i in os.listdir("FullScript/session" + str(n + 1) + "/F"):
        f1 = open("FullScript/session" + str(n + 1) + "/F/" + i)
        f = f1.read()
        f1.close()
        h = open("csvScriptSentences/session" + str(n + 1) + "/F/" + i, "w")
        F = f.splitlines()
        sentences = []
        for j in range(len(F)):
            sentence = F[j].split(';')[1]
            sentences.append(sentence)
        tab = np.array(sentences)
        with open("CSVfiles/" + str(n + 1) + 'dfDataScriptF' + i + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for r in range(len(F)):
                writer.writerow([r, tab[r]])
            for r in range(len(F), 167):
                writer.writerow([r, " "])
        dfData = pd.read_csv("CSVfiles/" + str(n + 1) + 'dfDataScriptF' + i + '.csv', names=["turn", "speech"])
        dfData.set_index('turn', inplace=True)
        print(dfData, file=h)
        h.close()
        encoded_data = tokenizer.batch_encode_plus(dfData.speech.values, add_special_tokens=True,
                                                   return_attention_mask=True, truncation=True, padding='longest',
                                                   return_tensors='pt')
        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        dst_Data = TensorDataset(input_ids, attention_masks)
        torch.save(dst_Data, "TensorsScript/session" + str(n + 1) + "/F/" +
                   str(n + 1) + 'ScriptSentencesF' + i[6:] + '.pt')

    for i in os.listdir("FullScript/session" + str(n + 1) + "/M"):
        f1 = open("FullScript/session" + str(n + 1) + "/M/" + i)
        f = f1.read()
        f1.close()
        h = open("csvScriptSentences/session" + str(n + 1) + "/M/" + i, "w")
        F = f.splitlines()
        sentences = []
        for j in range(len(F)):
            sentence = F[j].split(';')[1]
            sentences.append(sentence)
        tab = np.array(sentences)
        with open("CSVfiles/" + str(n + 1) + 'dfDataScriptM' + i + '.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            for r in range(len(F)):
                writer.writerow([r, tab[r]])
            for r in range(len(F), 167):
                writer.writerow([r, " "])
        dfData = pd.read_csv("CSVfiles/" + str(n + 1) + 'dfDataScriptM' + i + '.csv', names=["turn", "speech"])
        dfData.set_index('turn', inplace=True)
        print(dfData, file=h)
        h.close()
        encoded_data = tokenizer.batch_encode_plus(dfData.speech.values, add_special_tokens=True,
                                                   return_attention_mask=True, truncation=True, padding='longest',
                                                   return_tensors='pt')
        input_ids = encoded_data['input_ids']
        attention_masks = encoded_data['attention_mask']
        dst_Data = TensorDataset(input_ids, attention_masks)
        torch.save(dst_Data, "TensorsScript/session" + str(n + 1) + "/M/" +
                   str(n + 1) + 'ScriptSentencesM' + i[6:] + '.pt')
