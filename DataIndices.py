import os

for n in range(5):

    for i in os.listdir("FullImpro/session" + str(n + 1) + "/F"):
        f1 = open("FullImpro/session" + str(n + 1) + "/F/" + i)
        f = f1.read()
        f1.close()
        h = open("ImproActorLabel/session" + str(n + 1) + "/F/" + i, "w")
        F = f.splitlines()
        for j in range(len(F)):
            if F[j][0] == "F":
                print(1, file=h)
            else:
                print(2, file=h)
        for j in range(len(F), 167):
            print(0, file=h)
        h.close()
    for i in os.listdir("FullImpro/session" + str(n + 1) + "/M"):
        f1 = open("FullImpro/session" + str(n + 1) + "/M/" + i)
        f = f1.read()
        f1.close()
        h = open("ImproActorLabel/session" + str(n + 1) + "/M/" + i, "w")
        F = f.splitlines()
        for j in range(len(F)):
            if F[j][0] == "F":
                print(1, file=h)
            else:
                print(2, file=h)
        for j in range(len(F), 167):
            print(0, file=h)
        h.close()
    for i in os.listdir("FullScript/session" + str(n + 1) + "/F"):
        f1 = open("FullScript/session" + str(n + 1) + "/F/" + i)
        f = f1.read()
        f1.close()
        h = open("ScriptActorLabel/session" + str(n + 1) + "/F/" + i, "w")
        F = f.splitlines()
        for j in range(len(F)):
            if F[j][0] == "F":
                print(1, file=h)
            else:
                print(2, file=h)
        for j in range(len(F), 167):
            print(0, file=h)
        h.close()
    for i in os.listdir("FullScript/session" + str(n + 1) + "/M"):
        f1 = open("FullScript/session" + str(n + 1) + "/M/" + i)
        f = f1.read()
        f1.close()
        h = open("ScriptActorLabel/session" + str(n + 1) + "/M/" + i, "w")
        F = f.splitlines()
        for j in range(len(F)):
            if F[j][0] == "F":
                print(1, file=h)
            else:
                print(2, file=h)
        for j in range(len(F), 167):
            print(0, file=h)
        h.close()
