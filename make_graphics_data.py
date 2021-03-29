import numpy as np
from tqdm import tqdm
import os
from multiprocessing import Pool
from functools import partial
import networkx
import json
from main import run_graph, parse_args
import graph_utils
import plot


def run_one_exp(g, edges_num_dict, args, start_node, all_paths, x_name, params, finish_node):
    solutions_hoef_tmp = []
    solutions_dro_tmp = []
    solutions_dro_cropped_tmp = []
    _, _, _, _, fixed_p = run_graph(g, edges_num_dict, args, start_node, finish_node, all_paths=all_paths)
    for param in tqdm(params):
        setattr(args, x_name, param)
        if x_name != 'T_max':
            args.T_max = int(args.T_min + 3.106*np.log(args.T_min)) + 1
        solution_hoef, solution_dro, solution_dro_cropped, failed, _ = run_graph(g, edges_num_dict, args, start_node,
                                                                                 finish_node, fixed_p=fixed_p,
                                                                                 all_paths=all_paths)
        solutions_hoef_tmp.append(solution_hoef)
        solutions_dro_tmp.append(solution_dro)
        solutions_dro_cropped_tmp.append(solution_dro_cropped)
    return solutions_hoef_tmp, solutions_dro_tmp, solutions_dro_cropped_tmp


def run_experiments(args, g, edges_num_dict, start_node, finish_node, x_name, params):
    if args.run_DRO_cropped == 'true':
        all_paths = [x for x in networkx.all_simple_paths(g, start_node, finish_node)]
    else:
        all_paths = None
    print(f"running experiment with param_name={x_name}")
    solutions_hoef = []
    solutions_dro = []
    solutions_dro_cropped = []
    # all_failed = []
    experiment_func = partial(run_one_exp, g, edges_num_dict, args, start_node, all_paths, x_name, params)
    if args.num_workers > 1:
        p = Pool(args.num_workers)
        results = p.map(experiment_func, [finish_node] * args.num_exps)
    else:
        results = [experiment_func(finish_node) for _ in range(args.num_exps)]
    for res in results:
        solutions_hoef_tmp, solutions_dro_tmp, solutions_dro_cropped_tmp = res
        solutions_hoef.append(solutions_hoef_tmp)
        solutions_dro.append(solutions_dro_tmp)
        solutions_dro_cropped.append(solutions_dro_cropped_tmp)
    solutions_hoef = np.array(solutions_hoef)
    solutions_dro = np.array(solutions_dro)
    solutions_dro_cropped = np.array(solutions_dro_cropped)
    if args.percentage_mode == 'true':
        solutions_hoef_perc = np.zeros_like(solutions_hoef)
        solutions_dro_perc = np.zeros_like(solutions_dro)
        solutions_eq_perc = np.zeros_like(solutions_dro)
        result_array = np.stack((solutions_hoef, solutions_dro), axis=0)
        solutions_eq_perc[solutions_hoef == solutions_dro] = 1
        result_array[result_array == np.min(result_array, axis=0)] = 1
        result_array[result_array != np.min(result_array, axis=0)] = 0
        result_array[:, solutions_hoef == solutions_dro] = 0
        solutions_hoef, solutions_dro = result_array
        solutions_dro_cropped = solutions_eq_perc
    return solutions_hoef, solutions_dro, solutions_dro_cropped


def main():
    exp_name = 'exp14'
    x_name = "h"
    # x_name = "d"
    # x_name = "normal_std"
    args = parse_args()
    # params = [1 + i*3 for i in range(50//3)]
    params = [1 + i for i in range(9)]
    # params = [1, 2]
    # params = [args.T_min + i * 3 for i in range(14)]
    print(f"Running exp with param {x_name}", params)
    if args.debug != '':
        exit()
    # params = [50 + i*5 for i in range(50//5)]
    title = f"Hoeffding vs DRO vs DRO_truncated, {args.mode}, {x_name}"
    os.makedirs(exp_name, exist_ok=True)
    with open(f'{exp_name}/args.json', 'w') as f:
        dict = args.__dict__
        dict["changed_parameter"] = x_name
        dict["changed_parameter_values"] = params
        json.dump(dict, f, indent=4)
    g = graph_utils.create_fc_graph(args.h, args.w)
    edges_num_dict = graph_utils.numerate_edges(g)
    start_node = 0
    finish_node = list(g.nodes)[-1]

    solutions_hoef, solutions_dro, solutions_dro_cropped = run_experiments(args, g, edges_num_dict, start_node,
                                                                           finish_node, x_name, params)

    mean_hoef = np.mean(solutions_hoef, axis=0)
    mean_dro = np.mean(solutions_dro, axis=0)
    mean_dro_cropped = np.mean(solutions_dro_cropped, axis=0)
    std_dro = np.mean(np.abs(solutions_dro - np.median(solutions_dro)), axis=0)
    std_dro_cropped = np.mean(np.abs(solutions_dro_cropped - np.median(solutions_dro_cropped)), axis=0)
    std_hoef = np.mean(np.abs(solutions_hoef - np.median(solutions_hoef)), axis=0)

    print(f"Finished exp, {x_name}", mean_hoef)
    np.save(f'{exp_name}/mean_hoef.npy', mean_hoef)
    np.save(f'{exp_name}/std_hoef.npy', std_hoef)
    np.save(f'{exp_name}/mean_dro.npy', mean_dro)
    np.save(f'{exp_name}/std_dro.npy', std_dro)
    np.save(f'{exp_name}/mean_dro_cropped.npy', mean_dro_cropped)
    np.save(f'{exp_name}/std_dro_cropped.npy', std_dro_cropped)
    np.save(f'{exp_name}/params.npy', params)
    plot.main(exp_name, x_name, title, args)


if __name__ == '__main__':
    main()
