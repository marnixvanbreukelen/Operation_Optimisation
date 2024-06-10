import matplotlib.pyplot as plt
import numpy as np
from single_runway import *
from multiple_runway import *
from heuristic import *
import time
start_time = time.time()


# #get data
data_number = 6
R = 1
P, A_i, E_i, T_i, L_i, S_ij, g_i, h_i = read_data(data_number)
if R == 1:
    solution, final_var_dict = optimize_single_runway(P, E_i, T_i, L_i, S_ij, g_i, h_i, data_number)
else:
    solution, final_var_dict = optimize_multiple_runway(P, E_i, T_i, L_i, S_ij, g_i, h_i, R)
#make plot
#https://matplotlib.org/stable/api/markers_api.html
plt.scatter(E_i,np.arange(P),color='red',marker='|',label='E')
plt.scatter(T_i,np.arange(P),color='blue',marker='o',label='T')
plt.scatter(L_i,np.arange(P),color='green',marker='|',label='L')
plt.scatter(solution[:P],np.arange(P),color='orange',marker='x',label='O')
for i in range(P):
    plt.plot((E_i[i],L_i[i]),(i,i),color='black',linestyle='dotted')

#format plot
plt.legend()
plt.xlabel('Time')
plt.ylabel('Aircraft ID')
plt.show()

# DataMatrix = []
heur_DataMatrix = []
for data_number in [1,2,3,4,5,6,7,8]:
    R = 1
    # DataMatrix.append([])
    heur_DataMatrix.append([])

    max_number_runways = None
    while max_number_runways == None:
        #optimizer
        P, A_i, E_i, T_i, L_i, S_ij, g_i, h_i = read_data(data_number)
        solution, final_var_dict = optimize_multiple_runway(P, E_i, T_i, L_i, S_ij, g_i, h_i, R)
        alpha_lists = [alpha for alpha in final_var_dict["alpha"].values()]
        beta_lists = [beta for beta in final_var_dict["beta"].values()]
        objective = sum(np.multiply(alpha_lists,g_i) + np.multiply(beta_lists,h_i))

        #heuristic
        P, A_i, E_i, T_i, L_i, S_ij, g_i, h_i = read_data(data_number)
        A, P, E_i, T_i, L_i, S_ij, g_i, h_i, solu = heuristic(P, E_i, T_i, L_i, S_ij, g_i, h_i,R)
        heur_solution, heur_final_var_dict = optimize_multiple_runway_heuristic(A, P, E_i, T_i, L_i, S_ij, g_i, h_i, R)
        heur_alpha_lists = [alpha for alpha in heur_final_var_dict["alpha"].values()]
        heur_beta_lists = [beta for beta in heur_final_var_dict["beta"].values()]
        heur_objective = sum(np.multiply(heur_alpha_lists, g_i) + np.multiply(heur_beta_lists, h_i))

        #save solutions
        # DataMatrix[data_number-1].append(objective)
        heur_DataMatrix[data_number-1].append(heur_objective)

        #check stop condition
        # print('#########',data_number,R,objective,'############')
        if heur_objective <= 0.1: # not equal to 0 due to numerical rounding errors
            max_number_runways = True
        else:
            R += 1
# print(DataMatrix)
print(heur_DataMatrix)

end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Runtime: {elapsed_time} seconds")