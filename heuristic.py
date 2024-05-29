#heuristic implementation
from single_runway import *
from multiple_runway import *
import numpy as np
import matplotlib.pyplot as plt

# find upper bound of solution
#1. sort planes on T in ascending order
data_number = 4
R = 3
#
P, A_i, E_i, T_i, L_i, S_ij, g_i, h_i = read_data(data_number)
solution = optimize_multiple_runway(data_number,R)[1]

# sorted(T_i)
A_i = np.array(A_i)
E_i = np.array(E_i)
T_i = np.array(T_i)
L_i = np.array(L_i)
S_ij = np.array(S_ij)
g_i = np.array(g_i)
h_i = np.array(h_i)

index = np.argsort(T_i)

A_i = A_i[index]
E_i = E_i[index]
T_i = T_i[index]
L_i = L_i[index]
S_ij = S_ij[index]
for i in range(len(S_ij)):
    S_ij[i] = S_ij[i][index]
g_i = g_i[index]
h_i = h_i[index]


# get upper bound

#2. compute B_r for each runway
# x = solution['x']
#
#
# # for r in range(R):
# #     for j in range(P):
# #         key = str(j)
# #         max(T_i[j],
# #             max(x[k] + S_ij[k,j]), # for k on r
# #             max(x[k]+s_ij)) #for k not on r
#
# print('hello',solution)
# print(solution['x'])

A = [[]]
for i in range(R-1):
    A.append([])
x = solution['x']
s_ij = np.zeros((P,P))

# Compute max[x_k + S_kj | k ∈ A_r]
for j in range(P):
    B =[]
    for r in range(R):
        print(r,j,A)
        try:
            # for k in A[r]:
            #     print('A-r',k,T_i[k], S_ij[k][j])
            max_A_r = max(T_i[k] + S_ij[k][j] for k in A[r])
        except ValueError:
            max_A_r = 0
        # Compute max[x_k + s_kj | u ≠ r ∀ k ∈ A_u]
        try:
            max_A_u = max(T_i[k] + s_ij[k,j] for u in range(len(A)) if u != r for k in A[u])
        except ValueError:
            max_A_u = 0
        # Compute B_r
        # print('B_r=max:',T_i[j], max_A_r, max_A_u)
        B_r = max(T_i[j], max_A_r, max_A_u)
        B.append(B_r)
    ind = B.index(min(B))
    A[ind].append(j)
    x_j = min(B) # todo find out what this does and why x_j

print(T_i)


