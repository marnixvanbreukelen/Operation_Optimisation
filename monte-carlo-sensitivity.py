from single_runway import read_data, optimize_single_runway
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# # number of planes
# P = int(number_of_planes)
#
# # earliest landing time
# E_i = [el[1] for el in data]
# print(E_i)
#
# # target landing time
# T_i = [el[2] for el in data]
# print(T_i)
#
# # latest landing time
# L_i = [el[3] for el in data]
# print(L_i)
#
# # penalty cost too early
# g_i = [el[4] for el in data]
# print(g_i)
#
# # penalty cost too late
# h_i = [el[5] for el in data]
# print(h_i)
#
# # seperation requirement
# S_ij = [el[6:] for el in data]
# print(S_ij)



data_number = 3
P, E_i, T_i, L_i, S_ij, g_i, h_i = read_data(data_number)
# convert to int
S_ij = [[int(s) for s in s_ij] for s_ij in S_ij]

sens_steps = 5





"""Sensitivity"""
######### Separation time
def vary_separation_time_sens(S_ij, set_separation):
    return [[set_separation if s != 99999 else s for s in s_ij] for s_ij in S_ij]


def plot_varied_separation_time():
    # initialising lists
    S_ij_sens_list = []
    # Determine min and max separation time
    min_sep = min([s for s_ij in S_ij for s in s_ij])
    max_sep = max([s for sublist in S_ij for s in sublist if s != 99999])
    set_separation = min_sep
    # Create a loop
    for _ in range(sens_steps):
        S_ij_sens = vary_separation_time_sens(S_ij, set_separation)
        S_ij_sens_list.append(S_ij_sens)
        # Increase set_separation
        # might be that max seperation for all would result in unfeasible solutions?
        set_separation += (max_sep-min_sep)/(sens_steps)

    solutions = []
    # looping for the different combinations of g_i_sens and h_i_sens
    for index, S_ij_sens in enumerate(S_ij_sens_list):
        # solution is a list of all decision variables ordered x, alpha, beta ,d
        # every one of them is length P
        solution = optimize_single_runway(P, E_i, T_i, L_i, S_ij_sens, g_i, h_i,
                                          str(data_number) + "_var_sep_" + str(index))
        solutions.append(solution)

    alpha_lists = [solution[P:2*P] for solution in solutions]
    beta_lists = [solution[2*P:3*P] for solution in solutions]

    percentual_deviation = []
    for alpha_list, beta_list in zip(alpha_lists, beta_lists):
        percent_dev = []

        for i in range(P):
            print(alpha_list[i], beta_list[i])
            deviation = alpha_list[i] + beta_list[i]
            percent_deviation = round(deviation / T_i[i], 2) if T_i[i] != 0 else 0
            percent_dev.append(percent_deviation)

        percentual_deviation.append(percent_dev)


    # Prepare data for seaborn
    data = []
    for i in range(sens_steps):
        for deviation in percentual_deviation[i]:
            data.append({'Deviation': deviation, 'Group': f'{min_sep + i*((max_sep-min_sep)/(sens_steps))}'})


    df = pd.DataFrame(data)

    # Plotting
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Group', y='Deviation', data=df, notch=True)
    plt.title('Deviations for set separation distance')
    plt.ylabel('Deviation')
    plt.xlabel('Separation distance (time)')
    plt.xticks(rotation=45, ha='right')

    plt.grid(True)
    plt.tight_layout()  # Adjust layout to accommodate x-axis labels
    plt.show()

# plot_varied_separation_time()

######### Penalty costs:
# Options:
# we reduce early cost penalties and increase late cost penalties
# we increase early cost penalties and decrease late cost penalties
# half the planes we give higher penalty and we see whether those have a lower deviation
# Using set cost (all the same)

def vary_penalty_costs(g_i, h_i, set_cost_early, set_cost_late):
    g_i_sens = [set_cost_early for g in g_i]
    h_i_sens = [set_cost_late for h in h_i]
    return g_i_sens, h_i_sens

def plot_varying_cost_panalties():
    # we increase early cost penalties and decrease late cost penalties
    # initialising lists
    g_i_sens_list = []
    h_i_sens_list = []
    # Determine max costs
    max_cost_early = max(g_i)
    max_cost_late = max(h_i)
    set_cost_early = int(1/sens_steps*max_cost_early)
    set_cost_late = int(max_cost_early)
    # Create a loop
    for _ in range(sens_steps):
        g_i_sens, h_i_sens = vary_penalty_costs(g_i, h_i, set_cost_early, set_cost_late)
        g_i_sens_list.append(g_i_sens)
        h_i_sens_list.append(h_i_sens)
        # Increase set_cost_early
        set_cost_early += 1/sens_steps *max_cost_early
        set_cost_late -= 1 /sens_steps *max_cost_early

    # we expect avg alpha to decrease and avg beta to increase

    solutions = []
    # looping for the different combinations of g_i_sens and h_i_sens
    for index, g_i_sens in enumerate(g_i_sens_list):
        h_i_sens = h_i_sens_list[index]
        # solution is a list of all decision variables ordered x, alpha, beta ,d
        # every one of them is length P
        solution = optimize_single_runway(P, E_i, T_i, L_i, S_ij, g_i_sens, h_i_sens,
                                          str(data_number) + "_var_cost_" + str(index)
                                          )
        solutions.append(solution)

    alpha_list = [solution[P:2*P] for solution in solutions]
    beta_list = [solution[2*P:3*P] for solution in solutions]

    #  average alpha and beta values
    avg_alpha = [sum(alphas) / len(alphas) for alphas in alpha_list]
    avg_beta = [sum(betas) / len(betas) for betas in beta_list]

    # sensitivity steps and corresponding (g_i, h_i) tuples
    sensitivity_steps = range(1, sens_steps + 1)
    sensitivity_labels = [(g_i_sens_list[i][0], h_i_sens_list[i][0]) for i in range(sens_steps)]

    # Plotting the data
    plt.figure(figsize=(14, 8))
    plt.plot(sensitivity_steps, avg_alpha, label='Average Alpha (Early Deviation)', marker='o')
    plt.plot(sensitivity_steps, avg_beta, label='Average Beta (Late Deviation)', marker='o')

    # Adding titles and labels
    plt.title('Average Alpha and Beta Values Across Sensitivity Steps')
    plt.xlabel('Sensitivity Step (g_i, h_i)')
    plt.ylabel('Average Deviation')
    plt.xticks(sensitivity_steps, sensitivity_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to accommodate x-axis labels
    plt.show()

plot_varying_cost_panalties()

##### half the planes we give higher penalty and we see whether those have a lower deviation
# Function to vary penalty costs for half the planes with a preference factor
def vary_penalty_costs_half(g_i, h_i, set_cost_early, set_cost_late, pref_factor):
    g_i_sens = [set_cost_early // pref_factor if i % 2 == 0 else set_cost_early for i, g in enumerate(g_i)]
    h_i_sens = [set_cost_late // pref_factor if i % 2 == 0 else set_cost_late for i, h in enumerate(h_i)]
    return g_i_sens, h_i_sens

def plot_preferred_ac_penalties():
    # Initialize lists
    g_i_sens_list = []
    h_i_sens_list = []
    # Determine max costs
    set_cost_early = max(g_i)
    set_cost_late = max(h_i)
    # run for various pref_factors
    for pref_factor in range(1, sens_steps + 1):  # start from 1 to avoid division by zero
        g_i_sens, h_i_sens = vary_penalty_costs_half(g_i, h_i, set_cost_early, set_cost_late, pref_factor)
        g_i_sens_list.append(g_i_sens)
        h_i_sens_list.append(h_i_sens)

    # obtain optimal solutions for every pref_factor
    solutions = []
    for index, g_i_sens in enumerate(g_i_sens_list):
        h_i_sens = h_i_sens_list[index]
        solution = optimize_single_runway(P, E_i, T_i, L_i, S_ij, g_i_sens, h_i_sens,
                                          str(data_number) + "_pref_pen_" + str(index))
        solutions.append(solution)

    # alpha and beta deviations and categorising them into preferred and non-preferred groups
    alpha_lists = [solution[P:2*P] for solution in solutions]
    beta_lists = [solution[2*P:3*P] for solution in solutions]

    # alpha_lists_pref = []
    # alpha_lists_non_pref = []
    # beta_lists_pref = []
    # beta_lists_non_pref = []
    percentual_deviation_pref = []
    percentual_deviation_non_pref = []
    for alpha_list, beta_list in zip(alpha_lists, beta_lists):
        percent_dev_pref = []
        percent_dev_non_pref = []

        for i in range(P):
            print(alpha_list[i], beta_list[i])
            deviation = alpha_list[i] + beta_list[i]
            percent_deviation = round(deviation / T_i[i], 2) if T_i[i] != 0 else 0

            if i % 2 == 0:
                percent_dev_non_pref.append(percent_deviation)
            else:
                percent_dev_pref.append(percent_deviation)

        percentual_deviation_pref.append(percent_dev_pref)
        percentual_deviation_non_pref.append(percent_dev_non_pref)

    # Prepare data for seaborn
    data = []
    for i in range(sens_steps):
        for deviation in percentual_deviation_pref[i]:
            data.append({'Deviation': deviation, 'Group': f'Pref (Factor {i+1})'})
        for deviation in percentual_deviation_non_pref[i]:
            data.append({'Deviation': deviation, 'Group': f'Non-Pref (Factor {i+1})'})

    df = pd.DataFrame(data)

    # Plotting
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='Group', y='Deviation', data=df, notch=True)
    plt.title('Deviations for Preferred and Non-Preferred Groups Across Sensitivity Steps')
    plt.ylabel('Deviation')
    plt.xlabel('Groups (Pref and Non-Pref) with Pref Factor')
    plt.xticks(rotation=45, ha='right')

    plt.grid(True)
    plt.tight_layout()  # Adjust layout to accommodate x-axis labels
    plt.show()

#########  Landing times
def vary_landing_times(E_i, L_i, dev_earlier, dev_later):
    E_i_sens = [E - dev_earlier for E in E_i]
    L_i_sens = [L + dev_later for L in L_i]
    return E_i_sens, L_i_sens

def plot_varying_landing_times(E_i, T_i, L_i, P, S_ij, g_i, h_i):
    # Initialize lists to store sensitivity values
    E_i_sens_list = []
    L_i_sens_list = []

    # Determine max deviations such that E_i never exceeds L_i
    max_dev_earliest = min(E_i)
    max_dev_latest = min([L - E for E, L in zip(E_i, L_i)])

    dev_earlier = max_dev_earliest / sens_steps
    dev_later = max_dev_latest / sens_steps

    # Create a loop to vary landing times
    for it in range(sens_steps):
        E_i_sens, L_i_sens = vary_landing_times(E_i, L_i,
                                                max_dev_earliest - dev_earlier * it,
                                                dev_later * it)
        E_i_sens_list.append(E_i_sens)
        L_i_sens_list.append(L_i_sens)

    # Initialize a list to store solutions
    solutions = []

    # Optimize for different combinations of E_i_sens and L_i_sens
    for index, E_i_sens in enumerate(E_i_sens_list):
        L_i_sens = L_i_sens_list[index]
        solution = optimize_single_runway(P, E_i_sens, T_i, L_i_sens, S_ij, g_i, h_i,
                                          str(data_number) + "_var_landing_" + str(index))
        solutions.append(solution)

    # Extract alpha and beta values from solutions
    alpha_list = [solution[P:2*P] for solution in solutions]
    beta_list = [solution[2*P:3*P] for solution in solutions]

    # Calculate average alpha and beta values
    avg_alpha = [sum(alphas) / len(alphas) for alphas in alpha_list]
    avg_beta = [sum(betas) / len(betas) for betas in beta_list]

    # Define sensitivity steps and labels for plotting
    sensitivity_steps = range(1, sens_steps + 1)
    # TODO
    sensitivity_labels = [(round(max_dev_earliest - dev_earlier * i, 2), round(dev_later * i, 2)) for i in range(sens_steps)]

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(sensitivity_steps, avg_alpha, label='Average Alpha (Early Deviation)', marker='o')
    plt.plot(sensitivity_steps, avg_beta, label='Average Beta (Late Deviation)', marker='o')

    # Add titles and labels to the plot
    plt.title('Average Alpha and Beta Values Across Sensitivity Steps (Varying Landing Times)')
    plt.xlabel('Sensitivity Step (E_i, L_i)')
    plt.ylabel('Average Deviation (time)')
    plt.xticks(sensitivity_steps, sensitivity_labels, rotation=45, ha='right')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust layout to accommodate x-axis labels
    plt.show()

plot_varying_landing_times(E_i, T_i, L_i, P, S_ij, g_i, h_i)

# Number of aircraft in a time window


"""Monte-carlo"""
def vary_separation_time_sens(S_ij, random_range):
    S_ij_adj = []
    for index, s in enumerate(S_ij):
        if s < 99999:
            random_separation_deviation = random.randint(-random_range,random_range)
            if s + random_separation_deviation > 0:
                s_adj = s + random_separation_deviation
            else:
                s_adj = s - random_separation_deviation
        else:
            s_adj = 99999
        S_ij_adj.append(s_adj)

    return S_ij_adj












