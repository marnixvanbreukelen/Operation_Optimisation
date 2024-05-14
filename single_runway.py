import numpy as np
import pandas as pd
from gurobipy import *
import csv
import matplotlib.pyplot as plt

###SPECIFY DATA HERE###
data_number = 2



###Opening data file and extracting data
data_save = []
with open(f'data/airland{data_number}.txt') as file:
    reader = csv.reader(file, delimiter=' ')
    i = 0
    for row in reader:
        row.pop(0)
        row.pop(-1)
        data_temp = [float(i) for i in row]
        # print(data_temp)
        #get the number of planes
        if i == 0:
            number_of_planes = data_temp[0]
        #get the data per plane
        if i != 0:
            data_save = data_save+data_temp
            # print('data_save',data_save)
        i += 1

#split the total data to lists per plane, with first 6 datapoints and then the separation times
chunk_size = int(number_of_planes+6)
data = [data_save[i:i + chunk_size] for i in range(0, len(data_save), chunk_size)]
print('final data matrix',data)

### Defining all initial settings ###

#number of planes
P = int(number_of_planes)

#earliest landing time
E_i = [el[1] for el in data]
print(E_i)

#target landing time
T_i = [el[2] for el in data]
print(T_i)

#latest landing time
L_i = [el[3] for el in data]
print(L_i)

#seperation requirement
S_ij = [el[6:] for el in data]
print(S_ij)

#penalty cost too early
g_i = [el[4] for el in data]
print(g_i)

#penalty cost too late
h_i = [el[5] for el in data]
print(h_i)

### Define sets ###
W = []
V = []
U = []

#define set W for which plane i lands definetely before plane j (separation satisfied automaticly)
for i in range(P):
    for j in range(P):
        if i != j:
            if L_i[i] < E_i[j] and L_i[i] + S_ij[i][j] <= E_i[j]:
                W.append((i,j))

#define set V for which plane i lands definetely before plane j (separation NOT satisfied automaticly)
for i in range(P):
    for j in range(P):
        if i != j:
            if L_i[i] < E_i[j] and L_i[i] + S_ij[i][j] > E_i[j]:
                V.append((i,j))

#define set U for which we are uncertain which plane lands first
for i in range(P):
    for j in range(P):
        if i != j:
            if E_i[j] <= E_i[i] <= L_i[j]:
                U.append((i,j))
            elif E_i[j] <= L_i[i] <= L_i[j]:
                U.append((i,j))
            elif E_i[i] <= E_i[j] <= L_i[i]:
                U.append((i,j))
            elif E_i[i] <= L_i[j] <= L_i[i]:
                U.append((i,j))
print(U)
print(V)
print(W)

### Defining optimization model ###
model = Model()

### Decision variables ###
#landing time of plane i
x = {}
#how soon plane i lands before T_i
alpha = {}
#how soon plane i lands after T_i
beta = {}
#if plane i lands before j: d = 1, 0 otherwise
d = {}


for i in range(P):
    x[i] = model.addVar(lb=0,
                        vtype=GRB.CONTINUOUS,
                        name='x[%s]' % (i))
for i in range(P):
    alpha[i] = model.addVar(lb=0,
                        vtype=GRB.CONTINUOUS,
                        name='alpha[%s]' % (i))
for i in range(P):
    beta[i] = model.addVar(lb=0,
                        vtype=GRB.CONTINUOUS,
                        name='beta[%s]' % (i))
for i in range(P):
    for j in range(P):
        if j != i: #todo check if this creates the decision variable properly
            d[i,j] = model.addVar(lb=0, ub=1,
                                vtype=GRB.BINARY,
                                name='d[%s, %s]' % (i,j))
model.update()

### Constraints ###



model.update()

### Defining objective ###

obj = LinExpr()






model.setObjective(obj, GRB.MINIMIZE)
model.update()
model.write('model.lp')
model.optimize()



### Optional post-processing ###
