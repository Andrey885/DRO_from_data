import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from multiprocessing import Pool
from main import run_graph, parse_args
import graph_utils


def experiment(param):
    args = parse_args()
    args.count_cropped = False
    args.mode = 'binomial'
    # args.alpha= T_min
    # args.T_min = T_min
    # args.d = T_min
    args.d = 50
    args.w = 5
    args.h = 10
    args.T_min = param
    args.T_max = args.T_min + int(8.5 * np.log(args.T_min))
    # args.T_max = T_min
    args.normal_std = T_min
    print(f"Running with T_min {T_min}, T_max {args.T_max}")
    g = graph_utils.create_fc_graph(args.h, args.w)
    edges_num_dict = graph_utils.numerate_edges(g)
    start_node = 0
    finish_node = list(g.nodes)[-1]
    solutions_hoef = []
    solutions_dro = []
    solutions_dro_cropped = []
    all_failed = []
    for _ in tqdm(range(args.num_exps)):
        solution_hoef, solution_dro, solution_dro_cropped, failed = run_graph(g, edges_num_dict, args, start_node, finish_node)
        solutions_hoef.append(solution_hoef)
        solutions_dro.append(solution_dro)
        solutions_dro_cropped.append(solution_dro_cropped)
        all_failed.append(failed)
    solutions_hoef = np.array(solutions_hoef)
    solutions_dro = np.array(solutions_dro)
    solutions_dro_cropped = np.array(solutions_dro_cropped)
    return solutions_hoef, solutions_dro, solutions_dro_cropped


def main():
    params = [10 + i*5 for i in range(90//5)]
    p = Pool(11)
    results = p.map(experiment, params)
    # results = [experiment(T) for T in T_mins]
    mean_hoef = []
    mean_dro = []
    std_hoef = []
    std_dro = []
    mean_dro_cropped = []
    std_dro_cropped = []
    exp_name = 'exp1'
    for res in results:
        solutions_hoef, solutions_dro, solutions_dro_cropped = res
        mean_hoef.append(np.mean(solutions_hoef))
        std_hoef.append(np.mean(np.abs(solutions_hoef - np.median(solutions_hoef))))
        mean_dro.append(np.mean(solutions_dro))
        std_dro.append(np.mean(np.abs(solutions_dro - np.median(solutions_dro))))
        mean_dro_cropped.append(np.mean(solutions_dro_cropped))
        std_dro_cropped.append(np.mean(np.abs(solutions_dro_cropped - np.median(solutions_dro_cropped))))
    os.makedirs(f'data_{exp_name}', exist_ok=True)
    np.save(f'data_{exp_name}/mean_hoef.npy', mean_hoef)
    np.save(f'data_{exp_name}/std_hoef.npy', std_hoef)
    np.save(f'data_{exp_name}/mean_dro.npy', mean_dro)
    np.save(f'data_{exp_name}/std_dro.npy', std_dro)
    # np.save(f'data_{exp_name}/mean_dro_cropped.npy', mean_dro_cropped)
    # np.save(f'data_{exp_name}/std_dro_cropped.npy', std_dro_cropped)
    np.save(f'data_{exp_name}/params.npy', T_mins)



if __name__ == '__main__':
    main()
