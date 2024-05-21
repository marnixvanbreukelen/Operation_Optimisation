import matplotlib.pyplot as plt
import numpy as np
from single_runway import *
from multiple_runway import *

#get data
data_number = 8
R = 2
P, E_i, T_i, L_i, S_ij, g_i, h_i = read_data(data_number)
if R == 1:
    solution = optimize_single_runway(data_number)
else:
    solution = optimize_multiple_runway(data_number,R)
#make plot
#https://matplotlib.org/stable/api/markers_api.html
plt.scatter(E_i,np.arange(P),color='red',marker='|',label='E')
plt.scatter(T_i,np.arange(P),color='blue',marker='|',label='T')
plt.scatter(L_i,np.arange(P),color='green',marker='|',label='L')
plt.scatter(solution[:P],np.arange(P),color='orange',marker='x',label='O')
for i in range(P):
    plt.plot((E_i[i],L_i[i]),(i,i),color='black',linestyle='--')

#format plot
plt.legend()
plt.xlabel('Time')
plt.ylabel('Aircraft ID')
plt.show()