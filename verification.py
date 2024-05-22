import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from single_runway import *


def verification_plot(data_number):
    P, A_i, E_i, T_i, L_i, S_ij, g_i, h_i = read_data(data_number)
    solution = optimize_single_runway(data_number)
    number_of_planes = len(E_i)

    print(E_i)
    print(L_i)
    print(T_i)
    print(g_i)

    # Make nodes appearance time
    nodes_appearance = []
    for t in A_i:
        nodes_appearance.append([t, 1])


    nodes_landing = []
    for t in solution[0:number_of_planes]:
        nodes_landing.append([t, 0])

    ### Two aircraft close to eachother, expect to spread


    # Plotting
    # Extract coordinates from the lists
    appearance_x = [point[0] for point in nodes_appearance]
    appearance_y = [point[1] for point in nodes_appearance]
    landing_x = [point[0] for point in nodes_landing]
    landing_y = [point[1] for point in nodes_landing]

    # Calculate deviation from target time
    deviations = [landing[0] - target for landing, target in zip(nodes_landing, T_i)]

    # Plot the points and lines
    plt.figure(figsize=(10, 5))

    # Plot nodes_appearance points
    plt.scatter(appearance_x, appearance_y, color='blue', label='Appearance Nodes')
    # Plot nodes_landing points
    plt.scatter(landing_x, landing_y, color='red', label='Landing Nodes')

    # Draw lines between corresponding points with color based on deviation and thickness based on cost
    cmap = plt.cm.get_cmap('coolwarm')
    norm = mcolors.Normalize(vmin=min(deviations), vmax=max(deviations))
    for i in range(len(nodes_appearance)):
        line_width = g_i[i] / 7  # Arbitrary scaling for line width
        line_color = cmap(norm(deviations[i]))
        plt.plot([appearance_x[i], landing_x[i]], [appearance_y[i], landing_y[i]], color=line_color, linewidth=line_width)

    # Add horizontal bars for earliest and latest landing times
    for i in range(len(nodes_landing)):
        plt.hlines(y=nodes_landing[i][1], xmin=E_i[i], xmax=L_i[i], colors='black', linestyles='dashed', label='Landing Time Range')

    # Adding colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Deviation from Target Time')

    # Adding labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Lines Between Appearance and Landing Nodes with Landing Time Range and Deviation from Target Time')
    plt.legend()
    plt.grid(True)
    plt.show()

    return

# One aircraft higher cost passes lower cost aircraft = datanumber 14
# 2 aircraft close to each other expect them to spread = datanumber 15

data_number = 14
verification_plot(data_number)