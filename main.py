import numpy as np
import scipy
import scipy.stats
import math
import networkx
from tqdm import tqdm
import time
import cvxpy as cp
import argparse
import functools
from multiprocessing import Pool
import graph_utils
import distribution_utils


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
        objective_func += (i + 1) * q[i]
        if q_hat_a[i] != 0:
            sum_kl_dist += q_hat_a[i] * (np.log(q_hat_a[i]) - cp.log(q[i]))

    constraints.append(prob_sum == 1)
    constraints.append(sum_kl_dist <= r_a)
    objective = cp.Maximize(objective_func)
    prob = cp.Problem(objective, constraints)

    prob.solve(verbose=False)
    return prob.solution.opt_val


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
    # return -1 / T_a * math.log(alpha / (m * math.pow(T_a + 1, d)))
    return - (1 / T_a) * math.log(alpha / m) + (1 / T_a) * math.pow(d, m) * math.log(T_a + 1)
    # return -1 / T_a * math.log(alpha / (m * Cnk(T_a + d - 1, d - 1)))


def count_agrawal_ra(T_a, alpha, d, m):
    """
    Theorem 1.2 from https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9097208
    """
    # is always epsilon > (k-1)/n ?
    # k = d in paper notation
    # n = T_a in paper notation

    right_part = math.log(alpha / m)
    def objective_function(ra):
        """
        Solve transcendent equation by minimizing difference between left and right part
        """
        if ra < (d-1) / T_a:
            return 1e5
        return abs(-ra * T_a + (d - 1) * math.log(math.e * ra * T_a / (d - 1)) - right_part)
    min_value = scipy.optimize.minimize(objective_function, x0=d + 1, method='Nelder-Mead')
    ra = min_value['x'][0]
    solution = min_value['fun']
    if abs(solution) > 1e-3:  # the solution has failed
        return np.inf
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
            # nominator = np.prod(np.arange(1, j, 2)) * math.pi
            # denominator = np.prod(np.arange(2, j + 1, 2))
            result = 1
            for i in range(j // 2):
                result *= ((2*i + 1) / (2 * i + 2))
            result *= math.pi
        else:
            # nominator = np.prod(np.arange(2, j, 2)) * 2
            # denominator = np.prod(np.arange(1, j + 1, 2))
            result = 1
            for i in range((j + 1) // 2):
                result *= ((2*i + 2) / (2 * i + 1))
            result *= (2 / (j + 1))
        # assert nominator > 0 and denominator > 0, f"Value overflow processing c_{j}! Please decrease d"
        return result

    def get_km(m):
        """
        Equality (14) of paper
        """
        if m == -1:
            return 1
        elif m % 2 == 0:
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
    K_i = 1  # k_{-1}
    for i in range(k - 1):
        res = K_i * math.pow(math.e * math.sqrt(T_a) / (2 * math.pi), i)
        if res == 0:
            break
        square_bracket += res
        K_i *= get_cj(i)
    square_bracket *= (3 * get_cj(1) / get_cj(2))
    ra = - 1 / T_a * math.log(alpha / (m * square_bracket))
    return ra


def get_q_distribution_cropped(c_hat):
    # m = c_hat.shape[1]
    T = c_hat.shape[0]
    # q_hat = np.zeros(T)
    q_hat = []
    # maximal_realizations_count = math.pow(m, d)
    # z = np.zeros((T, m))
    z = []
    delete_array = np.zeros(T)
    for n in range(T):
        data_vector = c_hat[n]
        if delete_array[n] == 1:
            continue
        realizations_differences = np.sum((np.abs(c_hat - data_vector)), axis=1)
        # z[n] = c_hat[realizations_differences == 0][0]  # remember realization
        delete_array[realizations_differences == 0] = 1
        z.append(c_hat[realizations_differences == 0][0])
        sum_realizations = np.sum(realizations_differences == 0)
        # q_hat[n] = sum_realizations / T
        q_hat.append(sum_realizations / T)
    q_hat = np.array(q_hat)
    z = np.array(z)
    return q_hat, z


def run_DRO_cropped(c_hat, edges_num_dict, args, all_paths):
    def F(x, alpha):
        product = 1
        z_x_product = np.dot(z, x)
        for n in range(len(z)):
            product *= math.pow(alpha - z_x_product[n], q_hat[n])
        # return alpha - math.pow(math.e, r) * product
        return product

    def find_alpha(x):
        D = args.d * np.ones(m)
        alpha_lower_bound = np.dot(D, x)
        F_alpha = functools.partial(F, x)

        def objective_function(alpha):
            if alpha < alpha_lower_bound:
                return 1e5
            product = F_alpha(alpha)
            return alpha - math.pow(math.e, -r) * product
        solution = scipy.optimize.minimize(objective_function, x0=alpha_lower_bound + 1, method='Nelder-Mead')
        alpha = solution['x'][0]
        sol = solution['fun']
        assert sol < 1e5

        return alpha, sol

    def find_x():
        # x = np.ones(m)  # init

        all_path_encoded = []
        all_path_scores = []
        for path in all_paths:
            x = np.zeros(m)
            for i in range(len(path) - 1):
                source_node = path[i]
                target_node = path[i+1]
                x[edges_num_dict[source_node][target_node]] = 1
            alpha_star, solution_value = find_alpha(x)
            all_path_scores.append(solution_value)
            all_path_encoded.append(x)

        x_optimal = all_path_encoded[np.argmin(all_path_scores)]
        return x_optimal

    c_hat = np.array(c_hat).T
    T = c_hat.shape[0]
    m = c_hat.shape[1]
    q_hat, z = get_q_distribution_cropped(c_hat)
    d = int(math.pow(args.d, m))
    r_1 = count_classic_cnk_ra(T, args.alpha, d, m=1)  # m=1 because one r
    r_3 = count_agrawal_ra(T, args.alpha, d, m=1)
    r_2 = count_mardia_ra(T, args.alpha, d, m=1)
    r = min(r_1, r_2, r_3)
    path_dro = find_x()
    return path_dro


def get_c_worst_DRO(q_hat, alpha, T):
    """
    Solve with different radiuses
    """
    def objective_function(alpha_a):
        multiple_part = 1

        if alpha_a < d:
            return 1e5

        for i in range(d):
            multiple_part *= (alpha_a - (i + 1)) ** q_hat[a, i]

        objective_func = alpha_a - np.exp(-r_a) * multiple_part
        return objective_func

    d = q_hat.shape[1]
    m = len(q_hat)

    c_worst = []
    for a in range(m):
        r_a1 = count_classic_cnk_ra(T[a], alpha, d, m)
        r_a2 = count_mardia_ra(T[a], alpha, d, m)
        r_a3 = count_agrawal_ra(T[a], alpha, d, m)
        r_a = min(r_a1, r_a2, r_a3)

        min_value = scipy.optimize.minimize(objective_function, x0=d + 1, method='BFGS')
        min_value = float(min_value['fun'])

        # another minimization methods - just to check
        # min_value2 = solve_cvx_dual_individual(q_hat[a], r_a)
        # min_value2 = solve_cvx_primal_individual(q_hat[a], r_a, d)

        # assert abs(min_value2 - min_value) < 1e-3, (min_value2, min_value)

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
    assert np.max(np.abs(np.sum(q_hat, axis=1) - 1)) < 1e-3, q_hat.shape
    return q_hat


def run_graph(g, edges_num_dict, args, start_node, finish_node, all_paths=None, verbose=False, fixed_p=None):
    m = len(g.edges)
    if args.mode == 'binomial_with_binomial_T':
        c_hat, c_bar, T, p = distribution_utils.create_binomial_costs_with_binomial_T(m, d=args.d, T_min=args.T_min,
                                                                                      T_max=args.T_max, verbose=verbose,
                                                                                      fixed_p=fixed_p)
    elif args.mode == 'multinomial':
        c_hat, c_bar, T, p = distribution_utils.create_multinomial_costs(m, d=args.d, T_min=args.T_min,
                                                                         T_max=args.T_max, verbose=verbose,
                                                                         fixed_p=fixed_p)
    elif args.mode == 'binomial':
        c_hat, c_bar, T, p = distribution_utils.create_binomial_costs(m, d=args.d, T_min=args.T_min,
                                                                      T_max=args.T_max, verbose=verbose,
                                                                      fixed_p=fixed_p)
    elif args.mode == 'normal':
        c_hat, c_bar, T, p = distribution_utils.create_normal_costs(m, d=args.d, T_min=args.T_min,
                                                                    T_max=args.T_max, std=args.normal_std,
                                                                    verbose=verbose,  fixed_p=fixed_p)
    elif args.mode == 'binomial_with_binomial_T_reverse':
        c_hat, c_bar, T, p = distribution_utils.create_binomial_costs_with_binomial_T_reverse(m, d=args.d,
                                                                                              T_min=args.T_min,
                                                                                              T_max=args.T_max,
                                                                                              verbose=verbose,
                                                                                              fixed_p=fixed_p)

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
    c_worst_dro = get_c_worst_DRO(q_hat, args.alpha, T)
    _, path_c_worst_dro = graph_utils.solve_shortest_path(c_worst_dro.astype(float), edges_num_dict, g, start_node,
                                                          finish_node, verbose=False)

    expected_loss_dro = np.sum(path_c_worst_dro * c_bar)
    if args.use_best_found_path == 'true' and expected_loss_dro > expected_loss_hoeffding:
        path_c_worst_dro = path_c_worst_hoefding
        expected_loss_dro = np.sum(path_c_worst_dro * c_bar)
    # DRO on cropped data (compare with strongly optimal solution)
    min_length = min([c.shape[0] for c in c_hat])
    c_hat_cropped = np.array([c[:min_length] for c in c_hat])
    if args.count_cropped == 'true':
        path_c_worst_dro_cropped = run_DRO_cropped(c_hat_cropped, edges_num_dict, args, all_paths)
        expected_loss_dro_cropped = np.sum(path_c_worst_dro_cropped * c_bar)
    else:
        expected_loss_dro_cropped = 0
    failed = {'hoef': np.mean(c_worst_hoef < c_bar), 'dro': np.mean(c_worst_dro < c_bar)}
    if verbose:
        print("Solution hoef:", expected_loss_hoeffding / nominal_expected_loss)
        print("Solution DRO:", expected_loss_dro / nominal_expected_loss)
        if args.count_cropped == 'true':
            print("Solution DRO cropped:", expected_loss_dro_cropped / nominal_expected_loss)
    return expected_loss_hoeffding / nominal_expected_loss, expected_loss_dro / nominal_expected_loss,\
           expected_loss_dro_cropped / nominal_expected_loss, failed, p


def parse_args():
    parser = argparse.ArgumentParser(description='Experimental part for paper "DRO from data"')
    parser.add_argument('-d', '--debug', type=str, default='', help='debug mode', choices=['', 'true'])
    parser.add_argument('--num_workers', type=int, default=11, help='number of parallel jobs')
    parser.add_argument('--h', type=int, default=3,
                        help='h fully-connected layers + 1 start node + 1 finish node in graph')
    parser.add_argument('--w', type=int, default=3, help='num of nodes in each layer of generated graph')
    parser.add_argument('--d', type=int, default=50, help='num of different possible weights values')
    parser.add_argument('--T_min', type=int, default=30, help='min samples num')
    parser.add_argument('--T_max', type=int, default=30, help='max samples num')
    parser.add_argument('--count_cropped', type=str, default='false',
                        help='True if count cropped baseline method (computationally consuming)')
    parser.add_argument('--alpha', type=int, default=0.05, help='feasible error')
    parser.add_argument('--normal_std', type=int, default=5, help='std for normal data distribution')
    parser.add_argument('--num_exps', type=int, default=11, help='number of runs with different distributions')
    parser.add_argument('--mode', type=str, default='binomial', help='number of runs with different distributions',
                        choices=['binomial_with_binomial_T', 'binomial_with_binomial_T_reverse', 'multinomial',
                                 'binomial', 'normal'])
    parser.add_argument('--percentage_mode', type=str, default='false', help='if return result in binary (best solution or not)',
                        choices=['true', 'false'])
    parser.add_argument('--use_best_found_path', type=str, default='true', help='if use minimal path of dro and Hoeffding',
                        choices=['true', 'false'])
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    g = graph_utils.create_fc_graph(args.h, args.w)
    edges_num_dict = graph_utils.numerate_edges(g)
    start_node = 0
    finish_node = list(g.nodes)[-1]
    all_paths = [x for x in networkx.all_simple_paths(g, start_node, finish_node)]

    if args.debug != '':
        run_graph(g, edges_num_dict, args, start_node, finish_node, verbose=True, all_paths=all_paths)  # test run
        exit()
    solutions_hoef = []
    solutions_dro = []
    solutions_dro_cropped = []
    all_failed = []
    f = functools.partial(run_graph, g, edges_num_dict, args, start_node, finish_node)
    if args.num_workers > 1:
        p = Pool(args.num_workers)
        res = p.map(f, [all_paths] * args.num_exps)
    else:
        res = [f(all_paths) for _ in tqdm(range(args.num_exps))]
    for r in res:
        solution_hoef, solution_dro, solution_dro_cropped, failed, _ = r
        solutions_hoef.append(solution_hoef)
        solutions_dro.append(solution_dro)
        solutions_dro_cropped.append(solution_dro_cropped)
        all_failed.append(failed)
    all_failed = {key: [f[key] for f in all_failed] for key in failed}
    all_failed = {key: np.mean(all_failed[key]) for key in all_failed}
    print("FINAL RESULT (method solution / nominal solution):")
    print("HOEFDING:", np.mean(solutions_hoef), u"\u00B1", np.std(solutions_hoef))
    print(f"DRO METHOD {args.mode}:", np.mean(solutions_dro), u"\u00B1", np.std(solutions_dro))
    if args.count_cropped == 'true':
        print(f"DRO CROPPED METHOD {args.mode}:", np.mean(solutions_dro_cropped), u"\u00B1", np.std(solutions_dro_cropped))
        print("Failed samples:", all_failed)


if __name__ == '__main__':
    main()
