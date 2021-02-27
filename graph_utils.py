import networkx
import cplex
import numpy as np


def make_cplex_nodes_constraints(colnames, num_variables, edges_num_dict, g, start_node, finish_node):
    """
    Make constraints for solving shortest path problem in cplex format
    """
    rows = []
    senses = ''
    rhs = []
    for node in g.nodes:
        adj_edges = np.zeros(num_variables)
        for i in edges_num_dict[node]:
            adj_edges[edges_num_dict[node][i]] = 1  # mark as -1 all edges coming out of the node

        for other_node in edges_num_dict:
            if node in edges_num_dict[other_node] and other_node > node:
                adj_edges[edges_num_dict[other_node][node]] = -1  # mark as 1 all edges coming in the node
        if node == start_node:
            rows.append([colnames, adj_edges])
            rhs.append(-1)  # 1 is the right part of constraint for start and finish nodes
            senses += 'E'
        elif node == finish_node:
            rows.append([colnames, adj_edges])
            rhs.append(1)  # 1 is the right part of constraint for start and finish nodes
            senses += 'E'
        else:
            rows.append([colnames, adj_edges])
            rhs.append(0)  # 0 is the right part of constraint for all nodes except start and finish
            senses += 'E'
    rownames = ['d' + str(i) for i in range(len(rhs))]
    return rows, senses, rhs, rownames


def solve_cplex(weights, edges_num_dict, g, start_node, finish_node, verbose=False, y_star=None, c_bar=None):
    """
    Solve shortest path problem with cplex
    """
    num_edges = len(weights)
    obj = weights  # objective function weights.T @ nodes
    prob = cplex.Cplex()
    prob.set_log_stream(None)  # turn off logs
    prob.set_results_stream(None)
    prob.set_warning_stream(None)
    prob.objective.set_sense(prob.objective.sense.minimize)
    colnames = ['x' + str(i) for i in range(num_edges)]  # names of variables
    prob.variables.add(obj=obj, names=colnames)
    [prob.variables.set_types(i, prob.variables.type.binary) for i in range(num_edges)]  # set binary variables
    # rows1 are for upper bounds
    rhs1 = np.ones(num_edges, dtype=np.int32).tolist()  # right parts of constraints
    senses1 = 'L' * num_edges  # all constraints are '<='
    rownames1 = ['b'+str(i+1) for i in range(num_edges)]  # names of constraints
    rows1 = []
    for i in range(num_edges):
        a = np.zeros(num_edges)
        a[i] = 1
        rows1.append([colnames, a.tolist()])

    # rows2 are for lower bounds
    rhs2 = np.zeros(num_edges, dtype=np.int32).tolist()  # right parts of constraints
    senses2 = 'G' * num_edges  # all constraints are '>='
    rownames2 = ['c'+str(i+1) for i in range(num_edges)]  # names of constraints
    rows2 = []
    for i in range(num_edges):
        a = np.zeros(num_edges)
        a[i] = 1
        rows2.append([colnames, a.tolist()])
    # rows3 are for actual constraints
    rows3, senses3, rhs3, rownames3 = make_cplex_nodes_constraints(colnames, num_edges, edges_num_dict, g,
                                                                   start_node, finish_node)
    rows = rows1 + rows2 + rows3
    senses = senses1 + senses2 + senses3
    rhs = rhs1 + rhs2 + rhs3
    rownames = rownames1 + rownames2 + rownames3

    prob.linear_constraints.add(lin_expr=rows, senses=senses,
                                rhs=rhs, names=rownames)
    prob.solve()
    solution = prob.solution.get_objective_value()
    values = prob.solution.get_values()

    if verbose:
        print("Problem solved:\nSolution:", solution)
        print("Values:", values)
        print("Check worst-case scenario > nominal estimation:", np.mean(weights > c_bar) == 1)
        print(f"Check true nominal solution = {y_star} < our worst-case estimation (method2) = {solution}:"
              f"{solution > y_star}")
        # prob.write("lpex1.lp")
        # prob.solution.write("lpex1_sol.lp")
    return solution, values


def numerate_edges(g):
    edges_num_dict = {i: {} for i in range(len(g.nodes))}  # numerate edges
    for count, i in enumerate(g.edges):
        edges_num_dict[i[0]][i[1]] = count
        edges_num_dict[i[1]][i[0]] = count
    return edges_num_dict


def create_fc_graph(h, w):
    """
    Returns fully connected graph of h layers and w nodes in each layer
    """
    g = networkx.Graph()
    node_num = 0
    g.add_node(node_num)
    last_layer_node = node_num
    node_num += 1
    # form first layer
    current_layer_nodes = []
    for node in range(w):
        g.add_node(node_num)
        g.add_edge(last_layer_node, node_num)
        current_layer_nodes.append(node_num)
        node_num += 1

    # form h layers
    for l in range(h-1):
        last_layer_nodes = current_layer_nodes
        current_layer_nodes = []
        for node in range(w):
            g.add_node(node_num)
            [g.add_edge(last_layer_node, node_num) for last_layer_node in last_layer_nodes]
            current_layer_nodes.append(node_num)
            node_num += 1

    # form finish layer
    g.add_node(node_num)
    [g.add_edge(last_layer_node, node_num) for last_layer_node in current_layer_nodes]

    print("Generated graph: \n", networkx.to_dict_of_dicts(g))
    return g
