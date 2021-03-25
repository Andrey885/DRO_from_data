import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from multiprocessing import Pool
from functools import partial
import json
from main import run_graph, parse_args
import graph_utils
import plot


def experiment(args, param_name, param):
    print(f"running experiment with {param_name}={param}")
    setattr(args, param_name, param)
    args.T_max = args.T_min + 4*np.log(args.T_min)
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
    exp_name = 'exp1'
    title = "Hoeffding vs DRO, binomial, T_min=10"
    x_name = "T_min"
    params = [10 + i*5 for i in range(90//5)]
    args = parse_args()
    os.makedirs(f'data_{exp_name}', exist_ok=True)
    with open(f'data_{exp_name}/args.json', 'w') as f:
        dict = args.__dict__
        dict["changed_parameter"] = x_name
        dict["changed_parameter_values"] = params
        json.dump(dict, f, indent=4)
    func = partial(experiment, args, x_name)
    p = Pool(11)
    results = [func(p) for p in params]
    # results = p.map(func, params)
    mean_hoef = []
    mean_dro = []
    std_hoef = []
    std_dro = []
    mean_dro_cropped = []
    std_dro_cropped = []
    for res in results:
        solutions_hoef, solutions_dro, solutions_dro_cropped = res
        mean_hoef.append(np.mean(solutions_hoef))
        std_hoef.append(np.mean(np.abs(solutions_hoef - np.median(solutions_hoef))))
        mean_dro.append(np.mean(solutions_dro))
        std_dro.append(np.mean(np.abs(solutions_dro - np.median(solutions_dro))))
        mean_dro_cropped.append(np.mean(solutions_dro_cropped))
        std_dro_cropped.append(np.mean(np.abs(solutions_dro_cropped - np.median(solutions_dro_cropped))))
    np.save(f'data_{exp_name}/mean_hoef.npy', mean_hoef)
    np.save(f'data_{exp_name}/std_hoef.npy', std_hoef)
    np.save(f'data_{exp_name}/mean_dro.npy', mean_dro)
    np.save(f'data_{exp_name}/std_dro.npy', std_dro)
    np.save(f'data_{exp_name}/mean_dro_cropped.npy', mean_dro_cropped)
    np.save(f'data_{exp_name}/std_dro_cropped.npy', std_dro_cropped)
    np.save(f'data_{exp_name}/params.npy', T_mins)
    plot.main(exp_name)


if __name__ == '__main__':
    main()
