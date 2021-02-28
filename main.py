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
    return f(n) / f(k) / f(n-k)


def get_c_worst_hoefding(c_hat, T, m, d, alpha=0.05):
    """
    Get estimation of worst weights c with threshold alpha using Hoefding inequality
    """
    c_worst = []
    for a in range(m):
        mean_est = np.mean(c_hat[a])
        epsilon_a = (d-1) * math.sqrt(- math.log(alpha/m) / (2 * T[a]))
        c_worst.append(min(mean_est + epsilon_a, d))
    c_worst = np.array(c_worst)
    return c_worst


def solve_cvx_primal(qai_hat_a, ra):
    d = qai_hat_a.shape[0]
    q = cp.Variable(d)
    constraints = [q >= 0, q <= 1]
    objective_func = 0
    sum_a = 0
    sum_kl_dist = 0
    for i in range(d):
        sum_a += q[i]
        if qai_hat_a[i] == 0:
            # constraints.append(q[i] == 0)
            continue
        objective_func += (i + 1) * q[i]
        sum_kl_dist += qai_hat_a[i] * (np.log(qai_hat_a[i]) - cp.log(q[i]))
    constraints.append(sum_a == 1)
    constraints.append(sum_kl_dist <= ra)
    objective = cp.Maximize(objective_func)
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    q = q.value
    assert abs(np.sum(q) - 1) < 1e-3
    return prob.solution.opt_val


def solve_cvx_dual(qai_hat_a, ra):
    """
    This function fails as cvxpy does not recognize the problem as DCP due to numerical issues.
    It is replaced with solve_cvx_primal.
    """
    assert abs(np.sum(qai_hat_a) - 1) < 1e-3
    alpha_a = cp.Variable(1)
    d_ = np.max(np.argwhere(qai_hat_a != 0)[:, 0]) + 1  # last non zero value of freq
    # total_sum = 0
    # exponential_part = 0
    # for i in range(d_):
    #     total_sum += qai_hat_a[i]
    #     exponential_part += qai_hat_a[i] * cp.log(alpha_a - i - 1)
    #     val = (d_ - i - 1) ** qai_hat_a[i]
    #     assert val == val
    # assert abs(total_sum - 1) < 1e-3
    multiple_part = 1
    total_sum = 0
    for i in range(d_):
        total_sum += qai_hat_a[i]
        multiple_part *= (alpha_a - i - 1) ** qai_hat_a[i]
        val = (d_ - i - 1) ** qai_hat_a[i]
        assert val == val
    assert abs(total_sum - 1) < 1e-3
    # objective_func = alpha_a + np.exp(-ra) * multiple_part
    objective_func = np.exp(-ra) * multiple_part
    objective = cp.Minimize(objective_func)
    constraints = [alpha_a >= d_]
    prob = cp.Problem(objective, constraints)
    prob.solve(verbose=False)
    return alpha_a.value


def solve_part(q_hat, alpha, T_min, T_max, T, mode='same_ra'):
    """
    Solve with equal radiuses
    """
    def objective_function(alpha_a):
        multiple_part = 1
        if alpha_a < d_ + 1:
            return 1e5
        total_sum = 0
        for i in range(1, d_+2):
            multiple_part *= (alpha_a - i) ** q_hat[a, i - 1]
            assert multiple_part == multiple_part
            total_sum += q_hat[a, i-1]
        assert abs(total_sum - 1) < 1e-4, total_sum
        objective_func = alpha_a - np.exp(-ra) * multiple_part
        return objective_func

    d = q_hat.shape[1]
    if mode == 'same_ra':
        ra = -1 / T_min * math.log(alpha / len(q_hat) / math.pow(T_max + 1, d))  # old mode

    c_worst = []
    for a in range(len(q_hat)):
        if mode == 'different_ra':
            ra = -1 / T[a] * math.log(alpha / (len(q_hat) * Cnk(T[a] + d - 1, d-1)))
        d_ = np.max(np.argwhere(q_hat[a] != 0)[:, 0])  # last non zero value of free
        # min_value = solve_cvx_dual(q_hat[a], ra)
        min_value = solve_cvx_primal(q_hat[a], ra)
        # another minimization method for dual task. Gives exact same result:
        min_value2 = scipy.optimize.minimize(objective_function, x0=d+1, method='Nelder-Mead')['fun']
        assert abs(min_value2 - min_value) < 1e-3, (min_value2, min_value)
        if type(min_value) == np.ndarray:
            min_value = min_value[0]
        c_worst.append(min_value)
    c_worst = np.array(c_worst)
    return c_worst


def get_q_distribution(c_hat, d):
    q_hat = np.zeros((len(c_hat), d))
    for i in range(1, d+1):
        for a in range(len(c_hat)):
            q_hat[a, i-1] = np.mean(c_hat[a] == i)
    return q_hat


def run_graph(g, edges_num_dict, args, start_node, finish_node, verbose=False):
    if args.mode == 'binomial_binomial_T':
        c_hat, c_bar, T, p = distribution_utils.create_weights_distribution_with_binomial_T(len(g.edges), d=args.d,
                                                                                            T_min=args.T_min,
                                                                                            T_max=args.T_max,
                                                                                            verbose=verbose)
    elif args.mode == 'multinomial':
        c_hat, c_bar, T, p = distribution_utils.create_weights_distribution_with_multinomial_T(len(g.edges), d=args.d,
                                                                                               T_min=args.T_min,
                                                                                               T_max=args.T_max,
                                                                                               verbose=verbose)
    elif args.mode == 'binomial_uniform_T':
        c_hat, c_bar, T, p = distribution_utils.create_weights_distribution_from_uniform_T(len(g.edges), d=args.d,
                                                                                           T_min=args.T_min,
                                                                                           T_max=args.T_max,
                                                                                           verbose=verbose)
    c_worst_hoef = get_c_worst_hoefding(c_hat, T, len(g.edges), d=args.d, alpha=args.alpha)

    _, path_c_bar = graph_utils.solve_cplex(c_bar, edges_num_dict, g, start_node, finish_node,
                                                 verbose=False)  # nominal solution
    _, path_c_worst_hoefding = graph_utils.solve_cplex(c_worst_hoef, edges_num_dict, g, start_node, finish_node,
                                                 verbose=False)  # nominal solution

    solution_hoef = np.sum(path_c_worst_hoefding * c_worst_hoef)
    nominal_solution_hoef = np.sum(path_c_worst_hoefding * c_bar)
    q_hat = get_q_distribution(c_hat, args.d)
    c_worst_dro = solve_part(q_hat, args.alpha, args.T_min, args.T_max, T, args.ra_choice)

    _, path_c_worst_dro = graph_utils.solve_cplex(c_worst_dro, edges_num_dict, g, start_node, finish_node,
                                                 verbose=False)  # nominal solution

    solution_dro = np.sum(path_c_worst_dro * c_worst_dro)
    nominal_solution_dro = np.sum(path_c_worst_dro * c_bar)
    failed = {'hoef': np.mean(c_worst_hoef < c_bar), 'dro': np.mean(c_worst_dro < c_bar)}
    if verbose:
        print("Solution hoef:", solution_hoef / nominal_solution_hoef)
        print("Solution DRO:", solution_dro / nominal_solution_dro)
    return solution_hoef / nominal_solution_hoef, solution_dro / nominal_solution_dro, failed


def main():
    parser = argparse.ArgumentParser(description='Experimental part for paper "DRO from data"')
    parser.add_argument('--debug', type=str, default='', help='debug mode', choices=['', 'true'])
    parser.add_argument('--h', type=int, default=2,
                        help='h fully-connected layers + 1 start node + 1 finish node in graph')
    parser.add_argument('--w', type=int, default=5, help='num of nodes in each layer of generated graph')
    parser.add_argument('--d', type=int, default=5, help='num of different possible weights values')
    parser.add_argument('--T_min', type=int, default=10, help='min samples num')
    parser.add_argument('--T_max', type=int, default=100, help='max samples num')
    parser.add_argument('--alpha', type=int, default=0.05, help='feasible error')
    parser.add_argument('--num_exps', type=int, default=100, help='number of runs with different distributions')
    parser.add_argument('--ra_choice', type=str, default='different_ra',
                        help='formula for KL-dist constraint ("different_ra" is preferred)',
                        choices=['different_ra', 'same_ra'])
    parser.add_argument('--mode', type=str, default='binomial_uniform_T', help='number of runs with different distributions',
                        choices=['binomial_binomial_T', 'multinomial', 'binomial_uniform_T'])
    args = parser.parse_args()
    g = graph_utils.create_fc_graph(args.h, args.w)
    edges_num_dict = graph_utils.numerate_edges(g)
    start_node = 0
    finish_node = list(g.nodes)[-1]
    run_graph(g, edges_num_dict, args, start_node, finish_node, verbose=True)
    if args.debug != '':
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
