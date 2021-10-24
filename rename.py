import os


"""Simple transfert"""


for n in range(5):

    for i in os.listdir("FinalImproData/session" + str(n + 1) + "/F"):
        os.rename("FinalImproData/session" + str(n + 1) + "/F/" + i, "AllFiles/" + i)

    for i in os.listdir("FinalImproData/session" + str(n + 1) + "/M"):
        os.rename("FinalImproData/session" + str(n + 1) + "/M/" + i, "AllFiles/" + i)

    for i in os.listdir("FinalScriptData/session" + str(n + 1) + "/F"):
        os.rename("FinalScriptData/session" + str(n + 1) + "/F/" + i, "AllFiles/" + i)

    for i in os.listdir("FinalScriptData/session" + str(n + 1) + "/M"):
        os.rename("FinalScriptData/session" + str(n + 1) + "/M/" + i, "AllFiles/" + i)


