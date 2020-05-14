import numpy as np
import matplotlib.pyplot as plt

#########################################################

Data = open('data.csv').readlines()

D = []
for i in range(len(Data)):
    Data[i] = Data[i].strip()
    D.append( [ float(dat) for dat in  Data[i].split(",")] )
    
time  = np.array(D).T[0]
Xtime = np.array(D).T[1]
Ytime = np.array(D).T[2]

##########################################################

# Parametros optimos

#    [   A0 ,   A1 ,  w1 ,  f1 ,  A2  ,  w2 ,  f2 ]
PX = [-0.059, 1.950, 7.95, 0.10, 1.326, 0.48, 3.14]
PY = [-0.049, 1.834, 7.95, 3.28, 1.118, 0.48, 3.14]

Model = lambda P,t: P[0] + P[1]*np.cos(2*np.pi*P[2]*t+P[3]) + P[4]*np.cos(2*np.pi*P[5]*t+P[6])

χ2X = lambda P : sum([ (Model(P,t) - Xtime[i])**2 / np.abs(Model(P,t)) for i,t in enumerate(time) ])/N
χ2Y = lambda P : sum([ (Model(P,t) - Ytime[i])**2 / np.abs(Model(P,t)) for i,t in enumerate(time) ])/N

fig, axes = plt.subplots(ncols = 1, nrows = 2, sharex = True)

def Graf(Since,Until,Title):
    
    axes[0].plot(time,Model(PX,time), color = "gray", lw = 1.7)
    axes[0].plot(time,Xtime, ls = "",marker= "o", ms= 1,color = "orange", alpha = 0.5)
    axes[0].set_xlim(Since,Until)

    axes[1].plot(time,Model(PY,time), color = "gray", lw = 1.7)
    axes[1].plot(time,Ytime, ls = "",marker= "o", ms= 1, color = "orange", alpha = 0.5)
    axes[1].set_xlim(Since,Until)

    axes[0].set_ylabel("X [m]", fontsize = 14)
    axes[1].set_ylabel("Y [m]", fontsize = 14)
    axes[1].set_xlabel("Tiempo [s]", fontsize = 14)

    axes[0].grid(True) ;  axes[1].grid(True)

    axes[0].set_title(Title, fontsize = 16)
    plt.show()


Graf(3.5,3.9,"Tiempo vs Posiciones : Escala menor")
Graf(0,5,"Tiempo vs Posiciones : Escala mayor")
