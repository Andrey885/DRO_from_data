import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from multiprocessing import Pool
from functools import partial
import networkx
import json
from main import run_graph, parse_args
import graph_utils
import plot


def experiment(args, param_name, g, edges_num_dict, start_node, finish_node, fixed_p, params):
    mean_hoef = []
    mean_dro = []
    std_hoef = []
    std_dro = []
    mean_dro_cropped = []
    std_dro_cropped = []

    all_paths = [x for x in networkx.all_simple_paths(g, start_node, finish_node)]
    all_paths = [p for p in all_paths if np.min(np.diff(p)) > 0]
    # print(f"running experiment with {param_name}={param}")
    solutions_hoef = []
    solutions_dro = []
    solutions_dro_cropped = []
    all_failed = []
    # p = Pool(11)
    for _ in tqdm(range(args.num_exps)):
        solutions_hoef_tmp = []
        solutions_dro_tmp = []
        solutions_dro_cropped_tmp = []
        _, _, _, _, fixed_p = run_graph(g, edges_num_dict, args, start_node, finish_node, all_paths=all_paths)
        for param in params:
            setattr(args, param_name, param)
            args.T_max = int(args.T_min + 3.106*np.log(args.T_min)) + 1
            solution_hoef, solution_dro, solution_dro_cropped, failed, _ = \
                            run_graph(g, edges_num_dict, args, start_node, finish_node, fixed_p=fixed_p, all_paths=all_paths)
            solutions_hoef_tmp.append(solution_hoef)
            solutions_dro_tmp.append(solution_dro)
            solutions_dro_cropped_tmp.append(solution_dro_cropped)
        solutions_hoef.append(solutions_hoef_tmp)
        solutions_dro.append(solutions_dro_tmp)
        solutions_dro_cropped.append(solutions_dro_cropped_tmp)
    solutions_hoef = np.array(solutions_hoef)
    solutions_dro = np.array(solutions_dro)
    solutions_dro_cropped = np.array(solutions_dro_cropped)
    return solutions_hoef, solutions_dro, solutions_dro_cropped


def main():
    exp_name = 'exp1_fixed'
    x_name = "normal_std"
    # x_name = "d"
    # x_name = "normal_std"
    args = parse_args()
    params = [1 + i*3 for i in range(27//3)]
    # params = [10 + i*5 for i in range(45//5)]
    # params = [50 + i*5 for i in range(50//5)]
    title = f"Hoeffding vs DRO, {args.mode}, {x_name}"
    os.makedirs(f'data_{exp_name}', exist_ok=True)
    with open(f'data_{exp_name}/args.json', 'w') as f:
        dict = args.__dict__
        dict["changed_parameter"] = x_name
        dict["changed_parameter_values"] = params
        json.dump(dict, f, indent=4)
    g = graph_utils.create_fc_graph(args.h, args.w)
    edges_num_dict = graph_utils.numerate_edges(g)
    start_node = 0
    finish_node = list(g.nodes)[-1]

    all_paths = [x for x in networkx.all_simple_paths(g, start_node, finish_node)]
    all_paths = [p for p in all_paths if np.min(np.diff(p)) > 0]
    _, _, _, _, fixed_p = run_graph(g, edges_num_dict, args, start_node, finish_node, all_paths=all_paths)
    func = partial(experiment, args, x_name, g, edges_num_dict, start_node, finish_node, fixed_p)
    solutions_hoef, solutions_dro, solutions_dro_cropped = func(params)
    # p = Pool(11)
    # results = p.map(func, params)
    # results = [func(p) for p in params]


    # for res in results:
    # solutions_hoef, solutions_dro, solutions_dro_cropped = res
    print(solutions_hoef.shape)
    mean_hoef = np.mean(solutions_hoef, axis=0)
    mean_dro = np.mean(solutions_dro, axis=0)
    mean_dro_cropped = np.mean(solutions_dro_cropped, axis=0)
    std_dro = np.mean(np.abs(solutions_dro - np.median(solutions_dro)), axis=0)
    std_dro_cropped = np.mean(np.abs(solutions_dro_cropped - np.median(solutions_dro_cropped)), axis=0)
    std_hoef = np.mean(np.abs(solutions_hoef - np.median(solutions_hoef)), axis=0)
    # mean_hoef.append(np.mean(solutions_hoef))
    # std_hoef.append(np.mean(np.abs(solutions_hoef - np.median(solutions_hoef))))
    # mean_dro.append(np.mean(solutions_dro))
    # std_dro.append(np.mean(np.abs(solutions_dro - np.median(solutions_dro))))
    # mean_dro_cropped.append(np.mean(solutions_dro_cropped))
    # std_dro_cropped.append(np.mean(np.abs(solutions_dro_cropped - np.median(solutions_dro_cropped))))
    print(f"Finished exp, {x_name}", mean_hoef)
    np.save(f'data_{exp_name}/mean_hoef.npy', mean_hoef)
    np.save(f'data_{exp_name}/std_hoef.npy', std_hoef)
    np.save(f'data_{exp_name}/mean_dro.npy', mean_dro)
    np.save(f'data_{exp_name}/std_dro.npy', std_dro)
    np.save(f'data_{exp_name}/mean_dro_cropped.npy', mean_dro_cropped)
    np.save(f'data_{exp_name}/std_dro_cropped.npy', std_dro_cropped)
    np.save(f'data_{exp_name}/params.npy', params)
    plot.main(exp_name, x_name, title)


if __name__ == '__main__':
    main()
