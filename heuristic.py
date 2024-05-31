#heuristic implementation
from single_runway import *
from multiple_runway import *
import numpy as np
import matplotlib.pyplot as plt

# find upper bound of solution
#1. sort planes on T in ascending order


def heuristic(data_number,R):
    P, A_i, E_i, T_i, L_i, S_ij, g_i, h_i = read_data(data_number)

    # make arrays for the sorting
    A_i = np.array(A_i)
    E_i = np.array(E_i)
    T_i = np.array(T_i)
    L_i = np.array(L_i)
    S_ij = np.array(S_ij)
    g_i = np.array(g_i)
    h_i = np.array(h_i)

    # get the sorted target landing times
    index = np.argsort(T_i)

    #sort all other data in the same way
    A_i = A_i[index]
    E_i = E_i[index]
    T_i = T_i[index]
    L_i = L_i[index]
    S_ij = S_ij[index]
    for i in range(len(S_ij)):
        S_ij[i] = S_ij[i][index]
    g_i = g_i[index]
    h_i = h_i[index]

##### Implement the formula from the paper #####
    # A will be the matrix with which plane lands on which runway
    A = [[]]
    for i in range(R-1):
        A.append([])
    # initialize x_j
    x_j = T_i*1
    #s_ij zet to 0; #todo investigate ivm overestimate
    s_ij = np.zeros((P,P))

    # Sort every plane to the correct runway
    for j in range(P):
        #initialize B the matrix telling the earliest landing time per runway for the plane
        B =[]
        for r in range(R):
            # print(r,j,A)
            # try except because for the first plane, the formula gives an error
            try:
                #calculate the earliest allowed landing time for the next plane based on the seperation from the same runway
                max_A_r = max(x_j[k] + S_ij[k][j] for k in A[r])
            except ValueError:
                max_A_r = 0

            # todo max A_u is redundant
            # try except because for the first plane, the formula gives an error
            try:
                #calculate the earliest allowed landing time for the next plane based on the seperation from other runway(s)
                max_A_u = max(x_j[k] + s_ij[k,j] for u in range(len(A)) if u != r for k in A[u])
            except ValueError:
                max_A_u = 0
            # Compute B_r, the most critical landing time
            B_r = max(T_i[j], max_A_r, max_A_u)
            # save B_r to B
            B.append(B_r)
        # ind is the runway on which the plane will land, because there it can land the earliest
        ind = B.index(min(B))
        # Save plane j to the proper row of the A matrix
        A[ind].append(j)
        # update x_j, the actual landing time
        x_j[j] = min(B)
    # calculate the solution if only this heuristic is used
    solu = (x_j - T_i)*h_i

    return A, P, A_i, E_i, T_i, L_i, S_ij, g_i, h_i, solu
def optimize_multiple_runway_heuristic(A, P, A_i, E_i, T_i, L_i, S_ij, g_i, h_i):

    #read data
    # P, A_i, E_i, T_i, L_i, S_ij, g_i, h_i = read_data(data_number)
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

##### EXTRA CONSTRAINTS FROM HEURISTIC #####
    # planes landing on the same runway actuallly land on the same runway
    for r in range(len(A)):
        for i in A[r]:
            for j in A[r]:
                if i != j:
                    model.addLConstr(z[i,j], GRB.EQUAL, 1,
                             name='h1')
    #planes fixed on a runway are actually fixed on the runway
    for r in range(len(A)):
        for i in A[r]:
            model.addLConstr(y[i,r], GRB.EQUAL, 1,
                     name='h2')

    # # sequence fixed
    # this one is removed because it causes errors related to the sepperation between different runways
    # for i in range(P):
    #     for j in range(P):
    #         if i < j:
    #             model.addLConstr(d[i, j], GRB.EQUAL, 1,
    #                              name='h2')

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
    model.write("testout.sol")

    # get solution: is a list of all decision variables ordered x, alpha, beta ,d. z, y
    solution = []
    for v in model.getVars():
        solution.append(v.x)

    ### Format solution
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

data_number = 6
R = 2
A, P, A_i, E_i, T_i, L_i, S_ij, g_i, h_i, solu = heuristic(data_number,R)
print('the A matrix:',A)
optimize_multiple_runway_heuristic(A, P, A_i, E_i, T_i, L_i, S_ij, g_i, h_i)
print(solu, sum(solu))
