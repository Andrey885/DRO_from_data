import numpy as np
import scipy
import scipy.stats
import math
import networkx
from tqdm import tqdm
import cplex
import cvxpy as cp
import argparse
import graph_utils
import distribution_utils

np.random.seed(4)


def Cnk(n, k):
    f = math.factorial
    return f(n) / (f(k) * f(n-k))


def get_c_worst_hoefding(c_hat, T, m, d, alpha=0.05):
    """
    Get estimation of worst weights c with threshold alpha using Hoefding inequality (CHECKED)
    """
    c_worst = []
    for a in range(m):
        empirical_mean = np.mean(c_hat[a])
        epsilon = (d-1) * math.sqrt(- math.log(alpha/m) / (2 * T[a]))
        c_worst.append(min(empirical_mean + epsilon, d))
    c_worst = np.array(c_worst)
    return c_worst


def solve_cvx_primal_individual(q_hat_a, r_a, d):
    """
    COMMENT: this function does not solve the problem, if q_hat_a[d - 1] != 0
    """
    q = cp.Variable(d)
    constraints = [q >= 0, q <= 1]
    objective_func = 0
    prob_sum = 0
    sum_kl_dist = 0
    for i in range(d):
        prob_sum += q[i]
        if q_hat_a[i] != 0:
            objective_func += (i + 1) * q[i]
            sum_kl_dist += q_hat_a[i] * (np.log(q_hat_a[i]) - cp.log(q[i]))

    constraints.append(prob_sum == 1)
    constraints.append(sum_kl_dist <= r_a)
    objective = cp.Maximize(objective_func)
    prob = cp.Problem(objective, constraints)

    prob.solve(verbose=False)
    q = q.value
    return prob.solution.opt_val, q


def solve_cvx_dual_individual(q_hat_a, r_a):
    """
    This function fails as cvxpy does not recognize the problem as DCP due to numerical issues.
    It is replaced with solve_cvx_primal.
    """

    alpha_a = cp.Variable(1)
    d_a = np.max(np.argwhere(q_hat_a != 0)[:, 0]) + 1  # last non zero value of freq (TO CHECK?)

    multiple_part = 1
    for i in range(d_a):
        multiple_part *= (alpha_a - (i + 1)) ** q_hat_a[i]
    constraints = [alpha_a >= d_a]

    objective_func = alpha_a - np.exp(-r_a) * multiple_part
    objective = cp.Minimize(objective_func)

    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    return prob.solution.opt_val


def count_classic_cnk_ra(T_a, alpha, d, m):
    return -1 / T_a * math.log(alpha / (m * Cnk(T[a] + d - 1, d - 1)))


def count_agrawal_ra(T_a, alpha, d, m):
    """
    Theorem 1.2 from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9097208
    """
    # is always epsilon > (k-1)/n ?
    k = d  # paper notation
    n = T_a  # paper notation

    right_part = alpha / m
    left_part_mulitpier = math.pow(math.e * T_a / (d - 1), d - 1)
    def objective_function(ra):
        """
        Solve transcendent equation by minimizing difference between left and right part
        """
        return abs(left_part_mulitpier * math.pow(ra, d-1) * math.exp(-ra * T_a) - right_part)
    min_value = scipy.optimize.minimize(objective_function, x0=d + 1, method='Nelder-Mead')
    ra = min_value['x'][0]
    return ra

def count_mardia_ra(T_a, alpha, d, m):
    """
    Constraint on Kullback-Leibler distance described in
    https://pdfs.semanticscholar.org/70aa/1ca0395b6282f2183ea9889583ba25352748.pdf
    """
    def get_cj(j):
        if j == 0:
            return math.pi
        elif j == 1:
            return 2
        elif j % 2 == 0:
            nomirator = np.prod(np.arange(1, j, 2)) * math.pi
            denominator = np.prod(np.arange(2, j + 1, 2))
        else:
            nomirator = np.prod(np.arange(2, j, 2)) * 2
            denominator = np.prod(np.arange(1, j, 2))
        assert nomirator > 0 and denominator > 0, f"Value overflow processing c_{j}! Please decrease d"
        return nomirator / denominator

    def get_km(m):
        """
        Equality (14) of paper
        """
        if m % 2 == 0:
            nomirator = math.pi * math.pow(2 * math.pi, m//2)
            denominator = np.prod(np.arange(2, m + 1, 2))
        else:
            nomirator = math.pow(2 * math.pi, (m+1)//2)
            denominator = np.prod(np.arange(1, m + 1, 2))
        assert nomirator > 0 and denominator > 0, f"Value overflow processing k_{m}! Please decrease d"
        return nomirator / denominator

    # is it OK that estimation is not strictly e^(-T_a r_a), but sqrt(T_a) * e^(-T_a r_a) ?
    square_bracket = 0  # equation (15)
    k = d  # paper notation
    K_i_minus1 = 1
    for i in range(k - 1):
        square_bracket += (K_i_minus1 * math.pow(math.e * math.sqrt(T_a) / (2 * math.pi), i))
        K_i_minus1 *= get_cj(i)
    square_bracket *= (3 * get_cj(1) / get_cj(2))
    ra = - 1 / T_a * math.log(alpha / (m * square_bracket))
    return ra


def get_c_worst_DRO(q_hat, alpha, T, ra_choice):
    """
    Solve with different radiuses
    """
    def objective_function(alpha_a):
        multiple_part = 1
        d_a = np.max(np.argwhere(q_hat[a] != 0)[:, 0]) + 1  # last non zero value of freq
        if alpha_a < d_a:
            return 1e5

        for i in range(d_a):
            multiple_part *= (alpha_a - (i + 1)) ** q_hat[a, i]

        objective_func = alpha_a - np.exp(-r_a) * multiple_part
        return objective_func

    d = q_hat.shape[1]
    m = len(q_hat)

    c_worst = []
    for a in range(m):
        if ra_choice == 'classic_cnk':
            r_a = count_classic_cnk_ra(T[a], alpha, d, m)
        elif ra_choice == 'Mardia':
            r_a = count_mardia_ra(T[a], alpha, d, m)
        elif ra_choice == 'Agrawal':
            r_a = count_agrawal_ra(T[a], alpha, d, m)

        # min_value = solve_cvx_dual_individual(q_hat[a], r_a)
        min_value, alpha_a_primal = solve_cvx_primal_individual(q_hat[a], r_a, d)

        # another minimization method for the dual task - just to check
        min_value2 = scipy.optimize.minimize(objective_function, x0=d + 1, method='Nelder-Mead')
        min_value2 = float(min_value2['fun'])
        assert abs(min_value2 - min_value) < 1e-3, (min_value2, min_value)

        if type(min_value) == np.ndarray:
            min_value = min_value[0]

        c_worst.append(min_value)

    c_worst = np.array(c_worst)
    return c_worst


def get_q_distribution(c_hat, d):
    m = len(c_hat)
    q_hat = np.zeros((m, d))
    for a in range(m):
        for i in range(d):
            q_hat[a, i] = np.mean(c_hat[a] == i + 1)
    # print(q_hat.shape)
    assert np.max(np.abs(np.sum(q_hat, axis=1) - 1)) < 1e-3, q_hat.shape
    return q_hat


def run_graph(g, edges_num_dict, args, start_node, finish_node, verbose=False):
    m = len(g.edges)
    if args.mode == 'binomial_with_binomial_T':
        c_hat, c_bar, T, _ = distribution_utils.create_binomial_costs_with_binomial_T(m, d=args.d, T_min=args.T_min,
                                                                                      T_max=args.T_max, verbose=verbose)
    elif args.mode == 'multinomial':
        c_hat, c_bar, T, _ = distribution_utils.create_multinomial_costs(m, d=args.d, T_min=args.T_min,
                                                                         T_max=args.T_max, verbose=verbose)
    elif args.mode == 'binomial':
        c_hat, c_bar, T, _ = distribution_utils.create_binomial_costs(m, d=args.d, T_min=args.T_min,
                                                                      T_max=args.T_max, verbose=verbose)
    elif args.mode == 'normal':
        c_hat, c_bar, T, _ = distribution_utils.create_normal_costs(m, d=args.d, T_min=args.T_min,
                                                                      T_max=args.T_max, std=args.normal_std, verbose=verbose)
    # Nominal
    nominal_expected_loss, path_c_bar = graph_utils.solve_shortest_path(c_bar.astype(float), edges_num_dict, g,
                                                                        start_node, finish_node,
                                                                        verbose=False)  # nominal solution
    # Hoeffding
    c_worst_hoef = get_c_worst_hoefding(c_hat, T, m, d=args.d, alpha=args.alpha)
    _, path_c_worst_hoefding = graph_utils.solve_shortest_path(c_worst_hoef.astype(float), edges_num_dict, g,
                                                               start_node, finish_node, verbose=False)
    expected_loss_hoeffding = np.sum(path_c_worst_hoefding * c_bar)

    # DRO
    q_hat = get_q_distribution(c_hat, args.d)
    c_worst_dro = get_c_worst_DRO(q_hat, args.alpha, T, ra_choice=args.ra_choice)
    _, path_c_worst_dro = graph_utils.solve_shortest_path(c_worst_dro.astype(float), edges_num_dict, g, start_node,
                                                          finish_node, verbose=False)
    expected_loss_dro = np.sum(path_c_worst_dro * c_bar)

    failed = {'hoef': np.mean(c_worst_hoef < c_bar), 'dro': np.mean(c_worst_dro < c_bar)}
    if verbose:
        print("Solution hoef:", expected_loss_hoeffding / nominal_expected_loss)
        print("Solution DRO:", expected_loss_dro / nominal_expected_loss)
    return expected_loss_hoeffding / nominal_expected_loss, expected_loss_dro / nominal_expected_loss, failed


def parse_args():
    parser = argparse.ArgumentParser(description='Experimental part for paper "DRO from data"')
    parser.add_argument('--debug', type=str, default='', help='debug mode', choices=['', 'true'])
    parser.add_argument('--h', type=int, default=5,
                        help='h fully-connected layers + 1 start node + 1 finish node in graph')
    parser.add_argument('--w', type=int, default=5, help='num of nodes in each layer of generated graph')
    parser.add_argument('--d', type=int, default=10, help='num of different possible weights values')
    parser.add_argument('--T_min', type=int, default=10, help='min samples num')
    parser.add_argument('--T_max', type=int, default=100, help='max samples num')
    parser.add_argument('--alpha', type=int, default=1e-6, help='feasible error')
    parser.add_argument('--ra_choice', type=str, default='classic_cnk', help='choose the radius calculation algorithm',
                        choices=['classic_cnk', 'Agrawal', 'Mardia'])
    parser.add_argument('--normal_std', type=int, default=5, help='std for normal data distribution')
    parser.add_argument('--num_exps', type=int, default=10, help='number of runs with different distributions')
    parser.add_argument('--mode', type=str, default='binomial', help='number of runs with different distributions',
                        choices=['binomial_with_binomial_T', 'multinomial', 'binomial', 'normal'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    g = graph_utils.create_fc_graph(args.h, args.w)
    edges_num_dict = graph_utils.numerate_edges(g)
    start_node = 0
    finish_node = list(g.nodes)[-1]
    if args.debug != '':
        run_graph(g, edges_num_dict, args, start_node, finish_node, verbose=True)  # test run
        exit()
    solutions_hoef = []
    solutions_dro = []
    all_failed = []
    for _ in tqdm(range(args.num_exps)):
        solution_hoef, solution_dro, failed = run_graph(g, edges_num_dict, args, start_node, finish_node)
        solutions_hoef.append(solution_hoef)
        solutions_dro.append(solution_dro)
        all_failed.append(failed)
    all_failed = {key: [f[key] for f in all_failed] for key in failed}
    all_failed = {key: np.mean(all_failed[key]) for key in all_failed}
    print("FINAL RESULT (method solution / nominal solution):")
    print("HOEFDING:", np.mean(solutions_hoef), u"\u00B1", np.std(solutions_hoef))
    print(f"DRO METHOD {args.mode}:", np.mean(solutions_dro), u"\u00B1", np.std(solutions_dro))
    print("Failed samples:", all_failed)


if __name__ == '__main__':
    main()
