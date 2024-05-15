from gurobipy import *
import csv
from single_runway import read_data
import numpy as np

def optimize_multiple_runway(data_number, R):
    #read data
    P, E_i, T_i, L_i, S_ij, g_i, h_i = read_data(data_number)
    s_ij = np.zeros((P,P)) #todo check if zeros is the way to approach:: the result is good
    ### Define sets ###
    W = []
    V = []
    U = []

    #define set W for which plane i lands definetely before plane j (separation satisfied automaticly)
    for i in range(P):
        for j in range(P):
            if i != j:
                if L_i[i] < E_i[j] and L_i[i] + max(S_ij[i][j], s_ij[i][j]) <= E_i[j]:
                    W.append((i,j))

    #define set V for which plane i lands definetely before plane j (separation NOT satisfied automaticly)
    for i in range(P):
        for j in range(P):
            if i != j:
                if L_i[i] < E_i[j] and L_i[i] + max(S_ij[i][j], s_ij[i][j]) > E_i[j]:
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
    #if i and j land on the same runway: z = 1, 0 otherwise
    z = {}
    #if i land on r: y = 1, 0 otherwise
    y ={}


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
            if j != i:
                d[i,j] = model.addVar(lb=0, ub=1,
                                    vtype=GRB.BINARY,
                                    name='d[%s,%s]' % (i,j))
    for i in range(P):
        for j in range(P):
            if j != i:
                z[i,j] = model.addVar(lb=0, ub=1,
                                      vtype=GRB.BINARY,
                                      name='z[%s,%s]' % (i,j))
    for i in range(P):
        for r in range(R):
            y[i,r] = model.addVar(lb=0, ub=1,
                                  vtype=GRB.BINARY,
                                  name='y[%s,%s]' % (i,r))

    model.update()

    ### Constraints ###
    #constraint 1
    for i in range(P):
        model.addLConstr(x[i], GRB.GREATER_EQUAL, E_i[i],
                         name='1a')
        model.addLConstr(x[i], GRB.LESS_EQUAL, L_i[i],
                         name='1b')

    #constraint 2
    for i in range(P):
        for j in range(P):
            if j > i:
                model.addLConstr(d[i,j]+d[j,i], GRB.EQUAL, 1,
                                 name='2')

    #constraint 6
    for (i,j) in W + V:
        model.addLConstr(d[i,j], GRB.EQUAL, 1,
                         name='6')

    #constraint 14-18
    for i in range(P):
        model.addLConstr(alpha[i], GRB.GREATER_EQUAL, T_i[i]-x[i],
                         name='14')
        model.addLConstr(alpha[i], GRB.GREATER_EQUAL, 0,
                         name='15a')
        model.addLConstr(alpha[i], GRB.LESS_EQUAL, T_i[i] - E_i[i],
                         name='15b')
        model.addLConstr(beta[i], GRB.GREATER_EQUAL, x[i] - T_i[i],
                         name='16')
        model.addLConstr(beta[i], GRB.GREATER_EQUAL, 0,
                         name='17a')
        model.addLConstr(beta[i], GRB.LESS_EQUAL, L_i[i] - T_i[i],
                         name='17b')
        model.addLConstr(x[i], GRB.EQUAL, T_i[i]-alpha[i]+beta[i],
                         name='18')

    # constraint 28
    for i in range(P):
        model.addLConstr(quicksum(y[i,r] for r in range(R)), GRB.EQUAL, 1,
                         name='28')

    # constraint 29
    for i in range(P):
        for j in range(P):
            if j > i:
                model.addLConstr(z[i, j], GRB.EQUAL, z[j, i],
                                 name='29')

    # constraint 30
    for i in range(P):
        for j in range(P):
            if j > i:
                for r in range(R):
                    model.addLConstr(z[i, j], GRB.GREATER_EQUAL, y[i, r]+y[j,r]-1,
                                     name='30')

    # constraint 31 (derived from 7)
    for (i, j) in V:
        model.addLConstr(x[j], GRB.GREATER_EQUAL, x[i] + S_ij[i][j]*z[i,j]+s_ij[i][j]*(1-z[i,j]),
                         name='31')

    # constraint 33 (derived from 12, which is derived from 8) (M = Li+max(S,s)-Ej)
    for (i, j) in U:
        model.addLConstr(x[j], GRB.GREATER_EQUAL, x[i] + S_ij[i][j]*z[i,j]+s_ij[i][j]*(1-z[i,j]) - (L_i[i] + max(S_ij[i][j], s_ij[i][j]) - E_i[j]) * d[j, i],
                         name='33')
    model.update()

    ### Defining objective ###

    obj = LinExpr()

    for i in range(P):
        obj += g_i[i]*alpha[i]+h_i[i]*beta[i]


    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    model.write(f'model{data_number}.lp')

    #OPTIMIZE MODEL
    model.optimize()
    return
#RUN MODEL
###SPECIFY DATA HERE###
data_number = 8
R = 2
optimize_multiple_runway(data_number,R)
