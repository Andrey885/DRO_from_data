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
    c_worst_dro_tmp = []
    c_worst_hoef_tmp = []
    c_bar_tmp = []
    T_min_default = int(args.m.split('-')[0])
    T_max_default = int(args.m.split('-')[1])
    m = (T_max_default - T_min_default) / np.log(T_min_default)
    if x_name != 'T_max':
        args.T_max = int(args.T_min + m*np.log(args.T_min)) + 1
    _, _, _, _, _, _, _, fixed_p = run_graph(g, edges_num_dict, args, start_node, finish_node, all_paths=all_paths)
    # fixed_p = None
    for param in params:
        setattr(args, x_name, param)
        if x_name != 'T_max':
            args.T_max = int(args.T_min + m*np.log(args.T_min)) + 1
        g = graph_utils.create_fc_graph(args.h, args.w)
        edges_num_dict = graph_utils.numerate_edges(g)
        finish_node = max(g.nodes)
        all_paths = [x for x in networkx.all_simple_paths(g, start_node, finish_node)]

        solution_hoef, solution_dro, solution_dro_cropped, c_worst_dro, c_worst_hoef, c_bar, failed, _ = run_graph(g, edges_num_dict, args, start_node,
                                                                                 finish_node, fixed_p=None,
                                                                                 all_paths=all_paths)
        solutions_hoef_tmp.append(solution_hoef)
        solutions_dro_tmp.append(solution_dro)
        solutions_dro_cropped_tmp.append(solution_dro_cropped)
        c_worst_dro_tmp.append(c_worst_dro)
        c_worst_hoef_tmp.append(c_worst_hoef)
        c_bar_tmp.append(c_bar)
    return solutions_hoef_tmp, solutions_dro_tmp, solutions_dro_cropped_tmp, c_worst_dro_tmp, c_worst_hoef_tmp, c_bar_tmp


def run_experiments(args, g, edges_num_dict, start_node, finish_node, x_name, params):
    if args.count_cropped == 'true':
        all_paths = [x for x in networkx.all_simple_paths(g, start_node, finish_node)]
    else:
        all_paths = None
    print(f"running experiment with param_name={x_name}")
    solutions_hoef = []
    solutions_dro = []
    solutions_dro_cropped = []
    c_worst_dro = []
    c_worst_hoef = []
    c_bar = []
    # all_failed = []
    experiment_func = partial(run_one_exp, g, edges_num_dict, args, start_node, all_paths, x_name, params)
    if args.num_workers > 1:
        p = Pool(args.num_workers)
        results = []
        for res in tqdm(p.imap_unordered(experiment_func, [finish_node] * args.num_exps), total=args.num_exps):
            results.append(res)
    else:
        results = [experiment_func(finish_node) for _ in range(args.num_exps)]
    for res in results:
        solutions_hoef_tmp, solutions_dro_tmp, solutions_dro_cropped_tmp, c_worst_dro_tmp, c_worst_hoef_tmp, c_bar_tmp = res
        solutions_hoef.append(solutions_hoef_tmp)
        solutions_dro.append(solutions_dro_tmp)
        solutions_dro_cropped.append(solutions_dro_cropped_tmp)
        c_worst_dro.append(c_worst_dro_tmp)
        c_worst_hoef.append(c_worst_hoef_tmp)
        c_bar.append(c_bar_tmp)
    solutions_hoef = np.array(solutions_hoef)
    solutions_dro = np.array(solutions_dro)
    solutions_dro_cropped = np.array(solutions_dro_cropped)
    c_worst_dro = np.array(c_worst_dro)
    c_worst_hoef = np.array(c_worst_hoef)
    c_bar = np.array(c_bar)
    return solutions_hoef, solutions_dro, solutions_dro_cropped, c_worst_dro, c_worst_hoef, c_bar


def main():
    exp_name = 'exp1'
    x_name = "T_min"
    # x_name = "d"
    # x_name = "normal_std"
    args = parse_args()
    # params = [10 + i*3 for i in range(50//3)]
    # params = ['true']
    # params = [10 + i*5 for i in range(18)]
    # params = [10]
    # params = [1, 2]
    params = [10 + i for i in range(25)]
    print(f"Running exp with param {x_name}", params)
    if args.debug != '':
        exit()
    run_dro_truncated_title = ' vs DRO_truncated' if args.count_cropped == 'true' else ''
    title = f"Hoeffding vs DRO{run_dro_truncated_title}, {args.mode}, {x_name}"
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

    solutions_hoef, solutions_dro, solutions_dro_cropped, c_worst_dro, c_worst_hoef, c_bar = run_experiments(args, g, edges_num_dict, start_node,
                                                                           finish_node, x_name, params)

    print(f"Finished exp, {x_name}")
    np.save(f'{exp_name}/c_worst_dro.npy', c_worst_dro)
    np.save(f'{exp_name}/c_worst_hoef.npy', c_worst_hoef)
    np.save(f'{exp_name}/c_bar.npy', c_bar)

    np.save(f'{exp_name}/solutions_hoef.npy', solutions_hoef)
    np.save(f'{exp_name}/solutions_dro.npy', solutions_dro)
    np.save(f'{exp_name}/solutions_dro_cropped.npy', solutions_dro_cropped)
    np.save(f'{exp_name}/params.npy', params)
    count_costs = args.costs == 'true'
    count_percentage = args.percentage_mode == 'true'
    plot.main(exp_name, x_name, title, args)
    if count_costs:
        plot.main(exp_name, x_name, title.replace(x_name, "costs"), args, count_costs=count_costs)
    if count_percentage:
        plot.main(exp_name, x_name, title + ' percentage', args, count_percentage=count_percentage)


if __name__ == '__main__':
    main()
