import numpy as np
import pandas as pd
from gurobipy import *
import csv
import matplotlib.pyplot as plt

###Opening data file and extracting data
def read_data(data_number):
    data_save = []
    with open(f'data/airland{data_number}.txt') as file:
        reader = csv.reader(file, delimiter=' ')
        i = 0
        for row in reader:
            row.pop(0)
            if row[-1] == "":
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
    # print('final data matrix',data)

    ### Defining all initial settings ###

    #number of planes
    P = int(number_of_planes)

    #Appearance time
    A_i = [el[0] for el in data]
    # print(A_i)

    #earliest landing time
    E_i = [el[1] for el in data]
    # print(E_i)

    #target landing time
    T_i = [el[2] for el in data]
    # print(T_i)

    #latest landing time
    L_i = [el[3] for el in data]
    # print(L_i)

    #seperation requirement
    S_ij = [el[6:] for el in data]
    # print(S_ij)

    #penalty cost too early
    g_i = [el[4] for el in data]
    # print(g_i)

    #penalty cost too late
    h_i = [el[5] for el in data]
    # print(h_i)
    print("Data loaded succesfully")
    return P, A_i, E_i, T_i, L_i, S_ij, g_i, h_i



def optimize_single_runway(P, E_i, T_i, L_i, S_ij, g_i, h_i, data_number):
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
    # print(U)
    # print(V)
    # print(W)

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
            if j != i:
                d[i,j] = model.addVar(lb=0, ub=1,
                                    vtype=GRB.BINARY,
                                    name='d[%s,%s]' % (i,j))
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

    #constraint 7
    for (i,j) in V:
        model.addLConstr(x[j], GRB.GREATER_EQUAL, x[i]+S_ij[i][j],
                         name='7')

    #constraint 12 (M = L+S-E)
    for (i,j) in U:
        model.addLConstr(x[j], GRB.GREATER_EQUAL, x[i]+S_ij[i][j]*d[i,j]-(L_i[i]-E_i[j])*d[j,i],
                         name='12')

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
    model.update()

    ### Defining objective ###

    obj = LinExpr()

    for i in range(P):
        obj += g_i[i]*alpha[i]+h_i[i]*beta[i]


    model.setObjective(obj, GRB.MINIMIZE)
    model.update()
    # model.write(f'model_files/model{data_number}.lp')
    # Set the time limit to 60 seconds
    model.setParam('TimeLimit', 1)

    #OPTIMIZE MODEL
    model.optimize()
    #get solution: is a list of all decision variables ordered x, alpha, beta ,d
    solution = []
    for v in model.getVars():
        try:
            solution.append(v.x)
        except AttributeError:
            solution = 100*np.ones(4*P)

    # Function to parse Gurobi variable names and values
    def parse_var(var):
        name = var.varName
        value = var.x
        return name, value

    # Initialize the dictionary
    var_dict = {}

    # Populate the dictionary
    for var in model.getVars():
        name, value = parse_var(var)
        if '[' in name:
            base_name, indices = name.split('[')
            indices = indices.rstrip(']').split(',')
            indices = tuple(map(int, indices))  # Convert indices to tuple of integers
            if len(indices) == 1:
                indices = indices[0]  # Unwrap single-element tuple
            if base_name not in var_dict:
                var_dict[base_name] = {}
            var_dict[base_name][indices] = value
        else:
            var_dict[name] = value

    # Nested dictionary for variables like y
    def nested_dict():
        from collections import defaultdict
        return defaultdict(nested_dict)

    nested_var_dict = nested_dict()

    for key, value in var_dict.items():
        if isinstance(value, dict):
            for indices, val in value.items():
                if isinstance(indices, tuple):
                    if len(indices) == 2:
                        nested_var_dict[key][indices[0]][indices[1]] = val
                    else:
                        nested_var_dict[key][indices] = val
                else:
                    nested_var_dict[key][indices] = val
        else:
            nested_var_dict[key] = value

    # Convert nested defaultdict to regular dict
    import json
    final_var_dict = json.loads(json.dumps(nested_var_dict))

    return solution, final_var_dict

#RUN MODEL
###SPECIFY DATA HERE###
# data_number = 3
# # read data
# P, A_i, E_i, T_i, L_i, S_ij, g_i, h_i = read_data(data_number)
# print(optimize_single_runway(P, E_i, T_i, L_i, S_ij, g_i, h_i, data_number))


# data_number = 1
# # read data
# P, A_i, E_i, T_i, L_i, S_ij, g_i, h_i = read_data(data_number)
# # print(optimize_single_runway(P, E_i, T_i, L_i, S_ij, g_i, h_i))
# solution, final_var_dict = optimize_single_runway(P, E_i, T_i, L_i, S_ij, g_i, h_i, data_number)
# print(solution, final_var_dict)
# ### Optional post-processing ###
