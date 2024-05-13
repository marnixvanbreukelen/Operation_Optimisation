import numpy as np
import pandas as pd
from gurobipy import *
import matplotlib.pyplot as plt

# df = pd.read_csv('data/airland1.txt')
# print(df)

text_file = open("data/airland1.txt", "r")
lines = text_file.read().split('\n')
print(lines)

### Defining all initial settings ###
do = 1 #placeholder, should be fixed, read from file

#number of planes
P = do

#earliest landing time
E_i = do

#latest landing time
L_i = do

#target landing time
T_i = do

#seperation requirement
S_ij = do

#penalty cost too early
g_i = do

#penalty cost too late
h_i = do

### Define sets ###


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


for i in range(len(P)):
    x[i] = model.addVar(lb=0,
                        vtype=GRB.CONTINUOUS,
                        name='x[%s]' % (i))
for i in range(len(P)):
    alpha[i] = model.addVar(lb=0,
                        vtype=GRB.CONTINUOUS,
                        name='alpha[%s]' % (i))
for i in range(len(P)):
    beta[i] = model.addVar(lb=0,
                        vtype=GRB.CONTINUOUS,
                        name='beta[%s]' % (i))
for i in range(len(P)):
    for j in range(len(P)):
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
model.write('VRP_example.lp')
model.optimize()



### Optional post-processing ###
