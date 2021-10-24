import os
import pickle
import numpy as np


"""Association classe/numéro (vector parameters)"""


label_dict = {'xxx': 0, 'neu': 1, 'exc': 2, 'hap': 3, 'sur': 4, 'dis': 5, 'ang': 6, 'fru': 7, 'fea': 8, 'sad': 9}
print(label_dict)
file = open('dict.pkl', 'wb')
pickle.dump(label_dict, file)
file.close()

tab = []
for n in range(5):

    for i in os.listdir("Classes"):
        H = open("Classes/" + i)
        gamma = open("AllClasses/" + i, "w")
        h1 = H.read()
        h = h1.splitlines()
        tab.append(len(h))
        alpha = 0
        for k in range(len(h)):
            if h[k] == 'xxx':
                alpha = 0
            elif h[k] == 'neu':
                alpha = 1
            elif h[k] == 'exc':
                alpha = 2
            elif h[k] == 'hap':
                alpha = 3
            elif h[k] == 'sur':
                alpha = 4
            elif h[k] == 'dis':
                alpha = 5
            elif h[k] == 'ang':
                alpha = 6
            elif h[k] == 'fru':
                alpha = 7
            elif h[k] == 'fea':
                alpha = 8
            elif h[k] == 'sad':
                alpha = 9
            print(str(alpha), file=gamma)
        for k in range(len(h), 167):
            print(str(0), file=gamma)

Tab = np.array(tab)
mx = np.max(Tab)
print(mx)


