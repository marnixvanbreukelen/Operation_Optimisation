import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from single_runway import *
from multiple_runway import *
from matplotlib.lines import Line2D
from heuristic import *

# This file contains functions for the verification of the single runway model, multirunway and heuristic model

def verification_plot_single_runway(data_number):
    P, A_i, E_i, T_i, L_i, S_ij, g_i, h_i = read_data(data_number)
    solution, final_var_dict = optimize_single_runway(P, E_i, T_i, L_i, S_ij, g_i, h_i)
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


    # Plotting
    # Extract coordinates from the lists
    appearance_x = [point[0] for point in nodes_appearance]
    appearance_y = [point[1] for point in nodes_appearance]
    landing_x = [point[0] for point in nodes_landing]
    landing_y = [point[1] for point in nodes_landing]

    # Calculate deviation from target time
    deviations = [landing[0] - target for landing, target in zip(nodes_landing, T_i)]

    # # Plot the points and lines
    # plt.figure(figsize=(10, 5))

    # # Plot nodes_appearance points
    # plt.scatter(appearance_x, appearance_y, color='blue', label='Appearance Nodes')
    # # Plot nodes_landing points
    # plt.scatter(landing_x, landing_y, color='red', label='Landing Nodes')

    # Define a custom color map
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'gray', 'red'], N=256)

    # Normalize deviations such that 0 maps to gray
    norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)  # Adjust vmin and vmax based on your range

    # Plot the points and lines
    plt.figure(figsize=(10, 5))

    # Draw lines between corresponding points with color based on deviation and thickness based on cost
    line_widths = []
    for i in range(len(nodes_appearance)):
        line_width = g_i[i] / 7  # Arbitrary scaling for line width
        line_color = cmap(norm(deviations[i]))
        plt.plot([appearance_x[i], landing_x[i]], [appearance_y[i], landing_y[i]], color=line_color, linewidth=line_width)

        # Add small vertical lines for the landing time range
        plt.vlines(x=E_i[i], ymin=landing_y[i] - 0.05, ymax=landing_y[i] + 0.05, colors=line_color, linestyles='solid',
                   linewidth=2)
        plt.vlines(x=L_i[i], ymin=landing_y[i] - 0.05, ymax=landing_y[i] + 0.05, colors=line_color, linestyles='solid',
                   linewidth=2)

    # # Add horizontal bars for earliest and latest landing times
    # for i in range(len(nodes_landing)):
    #     plt.hlines(y=nodes_landing[i][1], xmin=E_i[i], xmax=L_i[i], colors='black', linestyles='dashed', label='Landing Time Range')

    # Adding colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Deviation from Target Time')

    # Customize y-axis ticks
    plt.yticks([0, 1], ['Runway', 'Appearance Time'])

    # Adding labels and title
    plt.xlabel('Time')
    plt.ylabel('Location')
    # plt.title('Single Runway appearance time and landing time visualisation')
    # plt.legend()
    plt.grid(True)

    plt.show()

    return




def verification_plot_multiple_runway(data_number, R):
    P, A_i, E_i, T_i, L_i, S_ij, g_i, h_i = read_data(data_number)
    solution, final_var_dict = optimize_multiple_runway(P, E_i, T_i, L_i, S_ij, g_i, h_i, R)
    number_of_planes = len(E_i)
    # print('SOLUTION', solution)


    # Make nodes appearance time
    nodes_appearance = []
    for t in A_i:  # t is time of appearance
        nodes_appearance.append([t, 0])

    print(final_var_dict['y'])

    nodes_landing = []
    for i in range(number_of_planes):  # for every aircraft landing
        t = final_var_dict['x'][str(i)]  # first number_of_planes values are landing time for aircraft i
        for r in range(R):  # for every runway
            if final_var_dict['y'][str(i)][str(r)] == 1:  # plane lands on runway r, make a new node
                nodes_landing.append([t, -r - 1])
    print(nodes_landing)


    # Plotting
    # Extract coordinates from the lists
    appearance_x = [point[0] for point in nodes_appearance]
    appearance_y = [point[1] for point in nodes_appearance]
    landing_x = [point[0] for point in nodes_landing]
    landing_y = [point[1] for point in nodes_landing]
    print(landing_y)
    print(appearance_y)

    # Calculate deviation from target time
    deviations = [landing[0] - target for landing, target in zip(nodes_landing, T_i)]

    # Plot the points and lines
    plt.figure(figsize=(10, 5))

    # Define a custom color map
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'gray', 'red'], N=256)

    # Normalize deviations such that 0 maps to gray
    norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)  # Adjust vmin and vmax based on your range

    line_widths = []
    for i in range(len(nodes_appearance)):
        line_width = g_i[i] / 7  # Arbitrary scaling for line width
        line_color = cmap(norm(deviations[i]))
        line_widths.append(line_width)
        plt.plot([appearance_x[i], landing_x[i]], [appearance_y[i], landing_y[i]], color=line_color, linewidth=line_width)

        # Add small vertical lines for the landing time range
        plt.vlines(x=E_i[i], ymin=landing_y[i] - 0.05, ymax=landing_y[i] + 0.05, colors=line_color, linestyles='solid',
                   linewidth=line_width)
        plt.vlines(x=L_i[i], ymin=landing_y[i] - 0.05, ymax=landing_y[i] + 0.05, colors=line_color, linestyles='solid',
                   linewidth=line_width)

    # Adding colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Deviation from Target Time')

    # Customize y-axis ticks
    y_values = [0] + list(range(-1, -R-1, -1))
    y_labels = ['Appearance Time'] + [f'Runway {abs(y) - 1}' for y in range(-1, -R-1, -1)]
    plt.yticks(y_values, y_labels)

    # Set x-axis limit to largest x coordinate + 50
    max_x = max(appearance_x + landing_x)
    # plt.xlim(0, max_x + 50)

    # Adding labels and title
    plt.xlabel('Time')
    plt.ylabel('Location')
    # plt.title('Lines Between Appearance and Landing Nodes with Landing Time Range and Deviation from Target Time')
    plt.grid(True)

    # Create a custom legend for line thickness
    unique_line_widths = sorted(set(line_widths))
    legend_lines = [Line2D([0], [0], color='gray', linewidth=lw, label=f'Cost {lw * 7:.1f}') for lw in unique_line_widths]
    plt.legend(handles=legend_lines, title='Line Thickness (Cost)', loc='upper right')

    plt.show()

    return

def verification_heuristic_plot_multiple_runway(data_number, R):
    P, A_i, E_i, T_i, L_i, S_ij, g_i, h_i = read_data(data_number)
    A, P, A_i, T_i, L_i, S_ij, g_i, h_i, solu = heuristic(P, E_i, T_i, L_i, S_ij, g_i, h_i, R)

    solution, final_var_dict = optimize_multiple_runway_heuristic(A, P, E_i, T_i, L_i, S_ij, g_i, h_i, R)
    number_of_planes = len(E_i)
    # print('SOLUTION', solution)


    # Make nodes appearance time
    nodes_appearance = []
    for t in A_i:  # t is time of appearance
        nodes_appearance.append([t, 0])

    print(final_var_dict['y'])

    nodes_landing = []
    for i in range(number_of_planes):  # for every aircraft landing
        t = final_var_dict['x'][str(i)]  # first number_of_planes values are landing time for aircraft i
        for r in range(R):  # for every runway
            if final_var_dict['y'][str(i)][str(r)] == 1:  # plane lands on runway r, make a new node
                nodes_landing.append([t, -r - 1])
    print(nodes_landing)


    # Plotting
    # Extract coordinates from the lists
    appearance_x = [point[0] for point in nodes_appearance]
    appearance_y = [point[1] for point in nodes_appearance]
    landing_x = [point[0] for point in nodes_landing]
    landing_y = [point[1] for point in nodes_landing]
    print(landing_y)
    print(appearance_y)

    # Calculate deviation from target time
    deviations = [landing[0] - target for landing, target in zip(nodes_landing, T_i)]

    # Plot the points and lines
    plt.figure(figsize=(10, 5))

    # Define a custom color map
    cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', ['blue', 'gray', 'red'], N=256)

    # Normalize deviations such that 0 maps to gray
    norm = mcolors.TwoSlopeNorm(vmin=-10, vcenter=0, vmax=10)  # Adjust vmin and vmax based on your range

    line_widths = []
    for i in range(len(nodes_appearance)):
        line_width = g_i[i] / 7  # Arbitrary scaling for line width
        line_color = cmap(norm(deviations[i]))
        line_widths.append(line_width)
        plt.plot([appearance_x[i], landing_x[i]], [appearance_y[i], landing_y[i]], color=line_color, linewidth=line_width)

        # Add small vertical lines for the landing time range
        plt.vlines(x=E_i[i], ymin=landing_y[i] - 0.05, ymax=landing_y[i] + 0.05, colors=line_color, linestyles='solid',
                   linewidth=line_width)
        plt.vlines(x=L_i[i], ymin=landing_y[i] - 0.05, ymax=landing_y[i] + 0.05, colors=line_color, linestyles='solid',
                   linewidth=line_width)

    # Adding colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, label='Deviation from Target Time')

    # Customize y-axis ticks
    y_values = [0] + list(range(-1, -R-1, -1))
    y_labels = ['Appearance Time'] + [f'Runway {abs(y) - 1}' for y in range(-1, -R-1, -1)]
    plt.yticks(y_values, y_labels)

    # Set x-axis limit to largest x coordinate + 50
    max_x = max(appearance_x + landing_x)
    # plt.xlim(0, max_x + 50)

    # Adding labels and title
    plt.xlabel('Time')
    plt.ylabel('Location')
    # plt.title('Lines Between Appearance and Landing Nodes with Landing Time Range and Deviation from Target Time')
    plt.grid(True)

    # Create a custom legend for line thickness
    unique_line_widths = sorted(set(line_widths))
    legend_lines = [Line2D([0], [0], color='gray', linewidth=lw, label=f'Cost {lw * 7:.1f}') for lw in unique_line_widths]
    plt.legend(handles=legend_lines, title='Line Thickness (Cost)', loc='upper right')

    plt.show()

    return

# Verficitation test: Aircraft high cost lands on time other aircrafts with lower do not = datanumber 14
# Verification test: 3 aircraft close to each other (with same cost) expect to spread = datanumber 15
### Explanation .txt files:
# Airland14.txt = 3 aircraft landing on same target time with one aircraft higher cost
# Airland16.txt = 2 aircraft landing on same target time
# Airland 17.txt = 2 aircraft with seperation time larger than timewindow
# Airland 18.txt = 3 aircarft with seperation time larger than timewindow


# Run Verifications
R = 2
data_number = 15
# verification_plot_multiple_runway(data_number, R)
# verification_plot_single_runway(data_number)
# verification_heuristic_plot_multiple_runway(data_number, R)



