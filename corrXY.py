import numpy as np
import matplotlib.pyplot as plt
import random

# Data representation

####################################################################

Data = open('data.csv').readlines()

D = []
for i in range(len(Data)):
    Data[i] = Data[i].strip()
    D.append( [ float(dat) for dat in  Data[i].split(",")] )
    
time  = np.array(D).T[0]
X = np.array(D).T[1]
Y = np.array(D).T[2]

##################################################################

E = lambda vec : sum(vec)/len(vec)
STD = lambda vec : np.sqrt( E(vec*vec) - E(vec)**2 )

CorrXY = (E(X*Y) - E(X)*E(Y) ) / ( STD(X) * STD(Y)  )

print (CorrXY)

plt.scatter(X,Y, s = 1, color = "orange")
plt.grid()
plt.xlabel("X [m]", fontsize = 16)
plt.ylabel("Y [m]", fontsize = 16)
plt.title("X vs Y",fontsize = 18)
plt.show()

