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
Xtime = np.array(D).T[1]
Ytime = np.array(D).T[2]

N = len(time)
sample_rate = N/time[-1]
frequency = np.linspace (0, sample_rate/2, int (N/2))
Xfreq, Yfreq  = np.fft.fft(Xtime), np.fft.fft(Ytime)

##################################################################

def Get_Params(Freq_Array):
    N = len(Freq_Array)
    f =  2/N * (Freq_Array[0:np.int (N/2)]
    A = np.abs(f)

    # Indices donde estan las 2 frecuencias de mayor amplitud
    ind = [np.where(np.abs(Amplitudes)  == a )[0][0] for a in sorted(ModAmp)[:-3:-1]]
    Ar = r.real[ind] ; Ai = r.imag[ind]

    Amps = A[ind]
    Phases = np.arctan2(Ai,Ar)
    Ws = frequency[ind]

    return [Amps[0],Ws[0],Phases[0],Amps[1],Ws[1],Phases[1]]
    
XParams = [sum([Xtime[i] for i in range(N)])/N] +  Get_Params(XFreq)
YParams = [sum([Ytime[i] for i in range(N)])/N] +  Get_Params(YFreq)


# El modelo es el siguiente:  # Modelo ~ A + A1*cos(2π*w1*t+p1) + A2*cos(2π*w2*t+p2)

Model = lambda P,t: P[0] + P[1]*np.cos(2*np.pi*P[2]*t+P[3]) + P[4]*np.cos(2*np.pi*P[5]*t+P[6])
χ2 = lambda P : sum([ (Model(P,t) - Xtime[i])**2 / np.abs(Model(P,t)) for i,t in enumerate(time) ])/N

# Algoritmo de exploracion

P0 = XParams   # Cambiar por YParams para minimizar Y

# Mod es una lista con todas las posibles variacioens de los parametros
                
Mod = []

Mod.append (np.linspace(P0[0]-0.4,P0[0]+0.4,100))             # M[0]

Mod.append (np.linspace(P0[1]-0.4,P0[1]+0.4,100))             # M[1]
Mod.append (np.linspace(0.75*Ws[0] , 1.25*Ws[0] , 100))       # M[2]
Mod.append (np.linspace(0,2*np.pi,200))                       # M[3]

Mod.append (np.linspace(P0[4]-0.4,P0[4]+0.4,100))             # M[4]
Mod.append (np.linspace(0.75*Ws[1] , 1.25*Ws[1] , 100))       # M[5]
Mod.append (np.linspace(0,2*np.pi,200))                       # M[6]

# Minimizacion
           
Loss = []
eras = 10
epochs = 15
Best = np.ones(7)

for E in range(eras):
    
    P0 = XParams
    Losslist = [χ2(P0)]
    
    for e in range(epochs):

        print ("Era: ",E,", Epoch: ",e,", χ2: ",round(χ2(P0),5))
    
        List = np.arange(0,7)
        random.shuffle(List)
        
        for i in List:

            P0[i] = Mod[i]
            P0[i] = Mod[i][np.where(χ2(P0) ==  min(χ2(P0)))[0][0]] 
        
        Losslist.append(χ2(P0))

        if len(Losslist) != len(set(Losslist)): break

    Loss.append(Losslist)
    if χ2(P0) < χ2(Best): Best = P0
        
    print ("")
    print ("P0 ->", list(map(lambda x: round(x,4),P0)))
    print ("Best ->",list(map(lambda x: round(x,4),Best)),"with χ2: ",round(χ2(Best),5)    )
    print ("")

# Graficacion del proceso de exploracion

fig = plt.figure(figsize = [15,7])

for l in Loss: plt.plot(l)
plt.xlabel("Epoch",fontsize= 16)
plt.ylabel("$\chi^2$",fontsize = 16)
plt.title("Expanded space minimization exploration", fontsize = 18) 
plt.grid()
plt.show()
    
    







