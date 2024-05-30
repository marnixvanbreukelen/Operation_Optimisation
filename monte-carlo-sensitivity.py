from single_runway import read_data, optimize_single_runway
from multiple_runway import optimize_multiple_runway
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import copy


# Loading Data for the rest of the analysis
data_number = 5
P, E_i, T_i, L_i, S_ij, g_i, h_i = read_data(data_number)
# convert to int
S_ij = [[int(s) for s in s_ij] for s_ij in S_ij]
opt = "multi_runway" # single_runway, multi_runway, or heuristic

"""Sensitivity"""
if opt == "multi_runway":
    R = 2
# How many sensitivity steps 
sens_steps = 5
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
        if opt == "single_runway":
            solution, final_var_dict = optimize_single_runway(P, E_i, T_i, L_i, S_ij_sens, g_i, h_i,
                                              str(data_number) + "_var_sep_" + str(index))
            solutions.append(final_var_dict)
        if opt == "multi_runway":
            solution, final_var_dict = optimize_multiple_runway(P, E_i, T_i, L_i, S_ij_sens, g_i, h_i,
                                             R)
            solutions.append(final_var_dict)

    alpha_lists = [[alpha for alpha in sol_dict["alpha"].values()] for sol_dict in mc_solutions]
    beta_lists = [[beta for beta in sol_dict["beta"].values()] for sol_dict in mc_solutions]

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
        if opt == "single_runway":
            solution, final_var_dict = optimize_single_runway(P, E_i, T_i, L_i, S_ij, g_i_sens, h_i_sens,
                                              str(data_number) + "_var_cost_" + str(index)
                                              )
            solutions.append(final_var_dict)
        if opt == "multi_runway":
            solution, final_var_dict = optimize_multiple_runway(P, E_i, T_i, L_i, S_ij, g_i_sens, h_i_sens,
                                                              R
                                                              )
            solutions.append(final_var_dict)


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

        if opt == "single_runway":
            solution, final_var_dict = optimize_single_runway(P, E_i, T_i, L_i, S_ij, g_i_sens, h_i_sens,
                                              str(data_number) + "_var_cost_" + str(index)
                                              )
            solutions.append(final_var_dict)

        if opt == "multi_runway":
            solution, final_var_dict = optimize_multiple_runway(P, E_i, T_i, L_i, S_ij, g_i_sens, h_i_sens,
                                                              R
                                                              )
            solutions.append(final_var_dict)

    # alpha and beta deviations and categorising them into preferred and non-preferred groups
    alpha_lists = [[alpha for alpha in sol_dict["alpha"].values()] for sol_dict in mc_solutions]
    beta_lists = [[beta for beta in sol_dict["beta"].values()] for sol_dict in mc_solutions]

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

        if opt == "single_runway":
            solution, final_var_dict = optimize_single_runway(P, E_i_sens, T_i, L_i_sens, S_ij, g_i, h_i,
                                                              str(data_number) + "_var_landing_" + str(index))
            solutions.append(final_var_dict)

        if opt == "multi_runway":
            solution, final_var_dict = optimize_multiple_runway(P, E_i_sens, T_i, L_i_sens, S_ij, g_i, h_i,
                                                                R
                                                                )
            solutions.append(final_var_dict)

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

""""PLOT ALL"""
# plot_preferred_ac_penalties()
# plot_varying_cost_panalties()
# plot_varied_separation_time()
# plot_varying_landing_times(E_i, T_i, L_i, P, S_ij, g_i, h_i)

# Number of aircraft in a time window

##############################################################################################################################################
##############################################################################################################################################
##############################################################################################################################################

"""Monte-carlo"""
def vary_separation_time_sens(S_ij, range_S):
    S_ij_adj = []
    for index, s in enumerate(S_ij):
        if s < 99999:
            s_adj = random.sample(range_S, 1)[0]
        else:
            s_adj = 99999
        S_ij_adj.append(s_adj)
    return S_ij_adj


def data_permutation(E_range, T_range, L_range, range_g, range_h, range_S,
                     P, E_i, T_i, L_i, S_ij, g_i, h_i):
    E_i_adj = E_i[:]
    T_i_adj = T_i[:]
    L_i_adj = L_i[:]
    S_ij_adj = [row[:] for row in S_ij]
    g_i_adj = g_i[:]
    h_i_adj = h_i[:]

    # go through every row for every plane
    for i in range(P):
        E_i_adj[i] = max(E_i[i] + random.sample(E_range, 1)[0], 0)
        T_i_adj[i] = E_i_adj[i] + random.sample(T_range, 1)[0]
        L_i_adj[i] = T_i_adj[i] + random.sample(L_range, 1)[0]

        g_i_adj[i] = random.sample(range_g, 1)[0]
        h_i_adj[i] = random.sample(range_h, 1)[0]

        S_ij_adj[i] = vary_separation_time_sens(S_ij[i], range_S)

    return E_i_adj, T_i_adj, L_i_adj, S_ij_adj, g_i_adj, h_i_adj


# Lists with integer values to add/subtract to permute
E_range = [-100, -50, 50, 100]
T_range = [50, 100, 150, 200]
L_range = [50, 100, 150, 200]

# Lists with values to choose from
range_S = [3, 8, 15]
range_g = [10, 30]
range_h = [10, 30]

# Initialize variations
S_ij_variations = []
E_i_variations = []
T_i_variations = []
L_i_variations = []
g_i_variations = []
h_i_variations = []

nr_mc_variations = 10
mc_solutions = []


def run_monte_carlo_variations(nr_mc_variations,
                               S_ij_variations, E_i_variations, T_i_variations,
                               L_i_variations, g_i_variations, h_i_variations, R):
    for _ in range(nr_mc_variations):
        # permutating data
        E_i_adj, T_i_adj, L_i_adj, S_ij_adj, g_i_adj, h_i_adj = data_permutation(
            E_range,
            T_range,
            L_range,
            range_g,
            range_h,
            range_S,
            P, E_i, T_i, L_i, S_ij, g_i, h_i
        )

        # monte_carlo_runs
        if opt == "single_runway":
            try:
                solution, final_var_dict = optimize_single_runway(
                    P, E_i_adj, T_i_adj, L_i_adj, S_ij_adj, g_i_adj, h_i_adj,
                    str(data_number) + "_monte_carlo_" + str(_)
                )  # TODO let's not write results to file
                mc_solutions.append(final_var_dict)
                # append monte-carlo variations
                E_i_variations.append(E_i_adj)
                T_i_variations.append(T_i_adj)
                L_i_variations.append(L_i_adj)
                S_ij_variations.append(S_ij_adj)
                g_i_variations.append(g_i_adj)
                h_i_variations.append(h_i_adj)
            except Exception as e:
                print(f"Optimization failed for iteration {_}: {e}")

        if opt == "multi_runway":
            try:
                solution, final_var_dict = optimize_multiple_runway(
                    P, E_i_adj, T_i_adj, L_i_adj, S_ij_adj, g_i_adj, h_i_adj, R
                )  # TODO let's not write results to file
                mc_solutions.append(final_var_dict)
                # append monte-carlo variations
                E_i_variations.append(E_i_adj)
                T_i_variations.append(T_i_adj)
                L_i_variations.append(L_i_adj)
                S_ij_variations.append(S_ij_adj)
                g_i_variations.append(g_i_adj)
                h_i_variations.append(h_i_adj)
            except Exception as e:
                print(f"Optimization failed for iteration {_}: {e}")

        if opt == "heuristic":
            try:
                solution, final_var_dict = optimize_heuristic_runway(
                    P, E_i_adj, T_i_adj, L_i_adj, S_ij_adj, g_i_adj, h_i_adj,
                    str(data_number) + "_monte_carlo_" + str(_)
                )  # TODO let's not write results to file
                mc_solutions.append(final_var_dict)
                # append monte-carlo variations
                E_i_variations.append(E_i_adj)
                T_i_variations.append(T_i_adj)
                L_i_variations.append(L_i_adj)
                S_ij_variations.append(S_ij_adj)
                g_i_variations.append(g_i_adj)
                h_i_variations.append(h_i_adj)
            except Exception as e:
                print(f"Optimization failed for iteration {_}: {e}")

    return (mc_solutions,
           E_i_variations, T_i_variations, L_i_variations,
           S_ij_variations, g_i_variations, h_i_variations)

def plot_mc_solutions(mc_solutions, T_i_variations):
    alpha_lists = [[alpha for alpha in sol_dict["alpha"].values()] for sol_dict in mc_solutions]
    beta_lists = [[beta for beta in sol_dict["beta"].values()] for sol_dict in mc_solutions]
    # alpha_lists = [solution[P:2*P] for solution in mc_solutions]
    # beta_lists = [solution[2*P:3*P] for solution in mc_solutions]
    percent_devs = []
    for variation, alpha_list in enumerate(alpha_lists):
        beta_list = beta_lists[variation]
        T_i = T_i_variations[variation]
        percent_dev = []

        for i in range(P):
            deviation = alpha_list[i] + beta_list[i]
            percent_deviation = 100*round(deviation / T_i[i], 2) if T_i[i] != 0 else 0
            percent_dev.append(percent_deviation)

        percent_devs.append(percent_dev)

    # Plot the data
    plt.figure(figsize=(12, 6))
    # Create a boxplot
    plt.boxplot(percent_devs)

    # Set title and labels
    plt.title(f'Boxplot of Deviations for Monte-Carlo variations of airland{data_number}.txt')
    plt.xlabel('Monte-Carlo Variation')
    plt.ylabel('Time Deviation (%)')

    # Show plot
    plt.show()


mc_solutions, E_i_variations, T_i_variations, L_i_variations, \
S_ij_variations, g_i_variations, h_i_variations = run_monte_carlo_variations(nr_mc_variations, \
                                S_ij_variations, E_i_variations, T_i_variations,
                               L_i_variations, g_i_variations, h_i_variations, R)
plot_mc_solutions(mc_solutions, T_i_variations)


"""Multiple Runways Changing number of runways"""
# Can play with this
if opt == "multi_runway":
    R = 1
    mc_solutions, E_i_variations, T_i_variations, L_i_variations, \
    S_ij_variations, g_i_variations, h_i_variations = run_monte_carlo_variations(nr_mc_variations, \
                                    S_ij_variations, E_i_variations, T_i_variations,
                                   L_i_variations, g_i_variations, h_i_variations, R)

all_mc_solutions_runways = {}
all_mc_solutions_runways[1] = mc_solutions
runway_list = [2, 3, 4]

for nr_runways in runway_list:
    # optimisation with added planes
    mc_solutions_runways = []
    for variation in range(nr_mc_variations):
        if opt == "multi_runway":
            try:
                solution, final_var_dict = optimize_multiple_runway(
                    P, E_i_variations[variation], T_i_variations[variation],
                    L_i_variations[variation], S_ij_variations[variation],
                    g_i_variations[variation], h_i_variations[variation],
                    nr_runways
                )
                mc_solutions_runways.append(final_var_dict)
            except Exception as e:
                print(f"Optimization failed for iteration {variation} with {nr_runways} planes added: {e}")

    all_mc_solutions_runways[nr_runways] = mc_solutions_runways


def plot_average_deviation_runways(runway_list, all_mc_solutions_runways, T_i_variations, P):
    avg_devs_per_runway_added = {}
    runway_list = [1] + runway_list
    for runway in runway_list:
        mc_solutions_runways = all_mc_solutions_runways[runway]

        alpha_lists = [[alpha for alpha in sol_dict["alpha"].values()] for sol_dict in mc_solutions_runways]
        beta_lists = [[beta for beta in sol_dict["beta"].values()] for sol_dict in mc_solutions_runways]
        avg_devs = []

        for variation, alpha_list in enumerate(alpha_lists):
            beta_list = beta_lists[variation]
            T_i = T_i_variations[variation]
            total_dev = 0

            for i in range(P):
                deviation = alpha_list[i] + beta_list[i]
                percent_deviation = 100 * round(deviation / T_i[i], 2) if T_i[i] != 0 else 0
                total_dev += percent_deviation

            avg_dev = total_dev / (P)
            avg_devs.append(avg_dev)

        avg_devs_per_runway_added[runway] = avg_devs

    plt.figure(figsize=(12, 6))
    plt.boxplot([avg_devs_per_runway_added[runway] for runway in runway_list], positions=runway_list)
    plt.title(f'Boxplot of Average Deviations per Number of Runways, {nr_mc_variations} Monte-Carlo Variations')
    plt.xlabel('Number of Runways')
    plt.ylabel('Average Time Deviation (%)')
    plt.show()

plot_average_deviation_runways(runway_list, all_mc_solutions_runways, T_i_variations, P)

"""Adding rows of planes"""
# Can play with this
if opt == "multi_runway":
    R = 2
    mc_solutions, E_i_variations, T_i_variations, L_i_variations, \
    S_ij_variations, g_i_variations, h_i_variations = run_monte_carlo_variations(nr_mc_variations, \
                                    S_ij_variations, E_i_variations, T_i_variations,
                                   L_i_variations, g_i_variations, h_i_variations, R)

# Genetic algorithm: random mutations in children based on parent
def add_planes(planes_to_add,
               E_range,
               T_range,
               L_range,
               range_g,
               range_h,
               range_S,
               P_var, E_i_var, T_i_var, L_i_var, S_ij_var, g_i_var, h_i_var):
    print(planes_to_add)
    for _ in range(planes_to_add):
        # s select a random existing plane
        i = random.randint(0,P_var-1)
        # Parameters to be permuted:
        # E_i earliest landing time, cannot be before 0
        E_p = max(0, E_i_var[i] + random.sample(E_range, 1)[0])

        # T_i target landing time
        T_p = E_p + random.sample(T_range, 1)[0]

        # L_i latest landing time
        L_p = T_p + random.sample(L_range, 1)[0]

        # g_i penalty cost early
        g_p = random.sample(range_g, 1)[0]

        # h_i penalty cost late
        h_p = random.sample(range_h, 1)[0]

        # S_ij
        # Also need to update all the other lists, when we are adding a plane

        # using a given range of separation times
        S_pj = []
        for j in range(P_var):
            # Add a separation time for plane p to other planes
            S_pj.append(random.sample(range_S, 1)[0])
            # Add a separation time for other plans j to plane p
            S_ij_var[j].append(random.sample(range_S, 1)[0])

        S_pj.append(99999)

        # Other option: adjusting the separation times of plane i:
        #for s in S_ij[i]:

        # Update all lists
        P_var += 1
        E_i_var.append(E_p)
        T_i_var.append(T_p)
        L_i_var.append(L_p)
        S_ij_var.append(S_pj)
        g_i_var.append(g_p)
        h_i_var.append(h_p)

    return E_i_var, T_i_var, L_i_var, S_ij_var, g_i_var, h_i_var


def add_ac_to_mc(nr_mc_variations,
                 planes_to_add,
                E_range,
                T_range,
                L_range,
                range_g,
                range_h,
                range_S,
                P_add,
                E_i_vars, T_i_vars, L_i_vars,
                S_ij_vars, g_i_vars, h_i_vars):

    # Initialising lists to store variations with added planes
    E_i_vars_ac_add = []
    T_i_vars_ac_add = []
    L_i_vars_ac_add = []
    S_ij_vars_ac_add = []
    g_i_vars_ac_add = []
    h_i_vars_ac_add = []

    for variation in range(nr_mc_variations):
        E_i_var = E_i_vars[variation]
        T_i_var = T_i_vars[variation]
        L_i_var = L_i_vars[variation]
        S_ij_var = S_ij_vars[variation]
        g_i_var = g_i_vars[variation]
        h_i_var = h_i_vars[variation]

        # Add planes based on the parameters
        E_i_adj, T_i_adj, L_i_adj, S_ij_adj, g_i_adj, h_i_adj = add_planes(
            planes_to_add, E_range, T_range, L_range, range_g, range_h, range_S,
            P_add, E_i_var, T_i_var, L_i_var, S_ij_var, g_i_var, h_i_var
        )

        # Append variation with planes added
        E_i_vars_ac_add.append(E_i_adj)
        T_i_vars_ac_add.append(T_i_adj)
        L_i_vars_ac_add.append(L_i_adj)
        S_ij_vars_ac_add.append(S_ij_adj)
        g_i_vars_ac_add.append(g_i_adj)
        h_i_vars_ac_add.append(h_i_adj)

    return (E_i_vars_ac_add, T_i_vars_ac_add,
            L_i_vars_ac_add, S_ij_vars_ac_add,
            g_i_vars_ac_add, h_i_vars_ac_add)


# dict to save for added planes
all_E_i_variations_ac_added = {}
all_T_i_variations_ac_added = {}
all_L_i_variations_ac_added = {}
all_S_ij_variations_ac_added = {}
all_g_i_variations_ac_added = {}
all_h_i_variations_ac_added = {}

# original variations
all_E_i_variations_ac_added[P] = copy.deepcopy(E_i_variations)
all_T_i_variations_ac_added[P] = copy.deepcopy(T_i_variations)
all_L_i_variations_ac_added[P] = copy.deepcopy(L_i_variations)
all_S_ij_variations_ac_added[P] = copy.deepcopy(S_ij_variations)
all_g_i_variations_ac_added[P] = copy.deepcopy(g_i_variations)
all_h_i_variations_ac_added[P] = copy.deepcopy(h_i_variations)

planes_to_add_list = [5, 10, 15, 20]
for planes_to_add in planes_to_add_list:
    all_E_i_variations_ac_added[P + planes_to_add] = []
    all_T_i_variations_ac_added[P + planes_to_add] = []
    all_L_i_variations_ac_added[P + planes_to_add] = []
    all_S_ij_variations_ac_added[P + planes_to_add] = []
    all_g_i_variations_ac_added[P + planes_to_add] = []
    all_h_i_variations_ac_added[P + planes_to_add] = []

all_mc_solutions_ac_added = {}
all_mc_solutions_ac_added[P] = mc_solutions

for planes_to_add in planes_to_add_list:
    # Create deep copies of the original variations for each iteration
    E_i_variations_copy = copy.deepcopy(E_i_variations)
    T_i_variations_copy = copy.deepcopy(T_i_variations)
    L_i_variations_copy = copy.deepcopy(L_i_variations)
    S_ij_variations_copy = copy.deepcopy(S_ij_variations)
    g_i_variations_copy = copy.deepcopy(g_i_variations)
    h_i_variations_copy = copy.deepcopy(h_i_variations)

    E_i_variations_ac_added, T_i_variations_ac_added, \
    L_i_variations_ac_added, S_ij_variations_ac_added, \
    g_i_variations_ac_added, h_i_variations_ac_added = add_ac_to_mc(nr_mc_variations,
                                                                    planes_to_add,
                                                                    E_range,
                                                                    T_range,
                                                                    L_range,
                                                                    range_g,
                                                                    range_h,
                                                                    range_S,
                                                                    P,
                                                                    E_i_variations_copy, T_i_variations_copy, L_i_variations_copy,
                                                                    S_ij_variations_copy, g_i_variations_copy, h_i_variations_copy)

    all_E_i_variations_ac_added[P + planes_to_add] = E_i_variations_ac_added
    all_T_i_variations_ac_added[P + planes_to_add] = T_i_variations_ac_added
    all_L_i_variations_ac_added[P + planes_to_add] = L_i_variations_ac_added
    all_S_ij_variations_ac_added[P + planes_to_add] = S_ij_variations_ac_added
    all_g_i_variations_ac_added[P + planes_to_add] = g_i_variations_ac_added
    all_h_i_variations_ac_added[P + planes_to_add] = h_i_variations_ac_added

    # optimisation with added planes
    mc_solutions_ac_added = []
    for variation in range(nr_mc_variations):
        if opt == "single_runway":
            try:
                solution, final_var_dict = optimize_single_runway(
                    P + planes_to_add, E_i_variations_ac_added[variation], T_i_variations_ac_added[variation],
                    L_i_variations_ac_added[variation], S_ij_variations_ac_added[variation],
                    g_i_variations_ac_added[variation], h_i_variations_ac_added[variation],
                    str(data_number) + "_monte_carlo_" + str(variation)
                )
                mc_solutions_ac_added.append(final_var_dict)
            except Exception as e:
                print(f"Optimization failed for iteration {variation} with {planes_to_add} planes added: {e}")

        if opt == "multi_runway":
            try:
                solution, final_var_dict = optimize_multiple_runway(
                    P + planes_to_add, E_i_variations_ac_added[variation], T_i_variations_ac_added[variation],
                    L_i_variations_ac_added[variation], S_ij_variations_ac_added[variation],
                    g_i_variations_ac_added[variation], h_i_variations_ac_added[variation],
                    R
                )
                mc_solutions_ac_added.append(final_var_dict)
            except Exception as e:
                print(f"Optimization failed for iteration {variation} with {planes_to_add} planes added: {e}")

        if opt == "_runway":
            try:
                solution, final_var_dict = optimize_single_runway(
                    P + planes_to_add, E_i_variations_ac_added[variation], T_i_variations_ac_added[variation],
                    L_i_variations_ac_added[variation], S_ij_variations_ac_added[variation],
                    g_i_variations_ac_added[variation], h_i_variations_ac_added[variation],
                    str(data_number) + "_monte_carlo_" + str(variation)
                )
                mc_solutions_ac_added.append(final_var_dict)
            except Exception as e:
                print(f"Optimization failed for iteration {variation} with {planes_to_add} planes added: {e}")

    all_mc_solutions_ac_added[P + planes_to_add] = mc_solutions_ac_added


def plot_average_deviation(planes_to_add_list, all_mc_solutions_ac_added, all_T_i_variations_ac_added):
    avg_devs_per_planes_added = {}
    planes_to_add_list = [0] + planes_to_add_list
    for planes_to_add in planes_to_add_list:
        mc_solutions_ac_added = all_mc_solutions_ac_added[P + planes_to_add]
        T_i_variations_ac_added = all_T_i_variations_ac_added[P + planes_to_add]

        alpha_lists = [[alpha for alpha in sol_dict["alpha"].values()] for sol_dict in mc_solutions_ac_added]
        beta_lists = [[beta for beta in sol_dict["beta"].values()] for sol_dict in mc_solutions_ac_added]
        avg_devs = []

        for variation, alpha_list in enumerate(alpha_lists):
            beta_list = beta_lists[variation]
            T_i = T_i_variations_ac_added[variation]
            total_dev = 0

            for i in range(P + planes_to_add):
                deviation = alpha_list[i] + beta_list[i]
                percent_deviation = 100 * round(deviation / T_i[i], 2) if T_i[i] != 0 else 0
                total_dev += percent_deviation

            avg_dev = total_dev / (P + planes_to_add)
            avg_devs.append(avg_dev)

        avg_devs_per_planes_added[P + planes_to_add] = avg_devs

    plt.figure(figsize=(12, 6))
    plt.boxplot([avg_devs_per_planes_added[P + planes_to_add] for planes_to_add in planes_to_add_list], positions=planes_to_add_list, widths=2)
    plt.title(f'Boxplot of Average Deviations per Number of Planes Added, {nr_mc_variations} Monte-Carlo Variations')
    plt.xticks(planes_to_add_list, [P + planes_to_add for planes_to_add in planes_to_add_list])
    plt.xlabel('Number of Planes')
    plt.ylabel('Average Time Deviation (%)')
    plt.show()

plot_average_deviation(planes_to_add_list, all_mc_solutions_ac_added, all_T_i_variations_ac_added)


