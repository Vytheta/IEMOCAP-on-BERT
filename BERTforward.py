import torch
import os
from tqdm import tqdm
from transformers import BertTokenizer
from transformers import BertModel


"""Passage des premiers Dataloaders à travers BERT, obtention des vecteurs d'attention (classe)"""


tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
model = BertModel.from_pretrained('bert-base-uncased')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


for n in range(5):

    for i in os.listdir("DataLoadersImpro/session" + str(n + 1) + "/F"):
        F = torch.load("DataLoadersImpro/session" + str(n + 1) + "/F/" + i)
        f = open("FinalImproVectors/session" + str(n + 1) + "/F/" + "impro" + i[17:-3], "w")
        outputs = []
        for k in tqdm(F):
            k = tuple(b.to(device) for b in k)
            inputs = {'input_ids': k[0], 'attention_mask': k[1]}
            with torch.no_grad():
                output1 = model.forward(**inputs, return_dict=True)
                output = output1.pooler_output
                outputs.append(output)
                #print(outputs, file=f)
        torch.save(outputs, "FinalImproData/session" + str(n + 1) + "/F/IF" + str(n + 1) + i[17:-3] + ".pt")
        f.close()

    for i in os.listdir("DataLoadersImpro/session" + str(n + 1) + "/M"):
        F = torch.load("DataLoadersImpro/session" + str(n + 1) + "/M/" + i)
        f = open("FinalImproVectors/session" + str(n + 1) + "/M/" + "impro" + i[17:-3], "w")
        outputs = []
        for k in tqdm(F):
            k = tuple(b.to(device) for b in k)
            inputs = {'input_ids': k[0], 'attention_mask': k[1]}
            with torch.no_grad():
                output1 = model.forward(**inputs, return_dict=True)
                output = output1.pooler_output
                outputs.append(output)
                #print(outputs, file=f)
        torch.save(outputs, "FinalImproData/session" + str(n + 1) + "/M/IM" + str(n + 1) + i[17:-3] + ".pt")
        f.close()

    for i in os.listdir("DataLoadersScript/session" + str(n + 1) + "/F"):
        F = torch.load("DataLoadersScript/session" + str(n + 1) + "/F/" + i)
        f = open("FinalScriptVectors/session" + str(n + 1) + "/F/" + "script" + i[18:-3], "w")
        outputs = []
        for k in tqdm(F):
            k = tuple(b.to(device) for b in k)
            inputs = {'input_ids': k[0], 'attention_mask': k[1]}
            with torch.no_grad():
                output1 = model.forward(**inputs, return_dict=True)
                output = output1.pooler_output
                outputs.append(output)
                #print(outputs, file=f)
        torch.save(outputs, "FinalScriptData/session" + str(n + 1) + "/F/SF" + str(n + 1) + i[18:-3] + ".pt")
        f.close()

    for i in os.listdir("DataLoadersScript/session" + str(n + 1) + "/M"):
        F = torch.load("DataLoadersScript/session" + str(n + 1) + "/M/" + i)
        f = open("FinalScriptVectors/session" + str(n + 1) + "/M/" + "script" + i[18:-3], "w")
        outputs = []
        for k in tqdm(F):
            k = tuple(b.to(device) for b in k)
            inputs = {'input_ids': k[0], 'attention_mask': k[1]}
            with torch.no_grad():
                output1 = model.forward(**inputs, return_dict=True)
                output = output1.pooler_output
                outputs.append(output)
                #print(outputs, file=f)
        torch.save(outputs, "FinalScriptData/session" + str(n + 1) + "/M/SM" + str(n + 1) + i[18:-3] + ".pt")
        f.close()