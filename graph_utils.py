import networkx
import cplex
import numpy as np


def check_flow_balance_constraints(edges_num_dict, g, start_node, finish_node, x):
    for node in g.nodes:
        adj_edges = np.zeros(len(x))
        for i in edges_num_dict[node]:
            adj_edges[edges_num_dict[node][i]] = 1  # mark as 1 all outgoing arcs
        for other_node in edges_num_dict:
            if node in edges_num_dict[other_node] and other_node > node:
                adj_edges[edges_num_dict[other_node][node]] = -1  # mark as -1 all incoming arcs
        if node == start_node:
            right_part = -1  # -1 is the right part of constraint for the start node
        elif node == finish_node:
            right_part = 1  # 1 is the right part of constraint for the finish node
        else:
            right_part = 0  # 0 is the right part of constraint for all nodes except start and finish
        if np.dot(x, adj_edges) != right_part:
            return False
    return True


def make_flow_balance_constraints(variable_names, num_variables, edges_num_dict, g, start_node, finish_node):
    """
    Make constraints for solving the shortest path problem in cplex
    """
    rows = []
    senses = ''
    rhs = []
    for node in g.nodes:
        adj_edges = np.zeros(num_variables)
        for i in edges_num_dict[node]:
            adj_edges[edges_num_dict[node][i]] = 1  # mark as 1 all incoming arcs

        for other_node in edges_num_dict:
            if node in edges_num_dict[other_node] and other_node > node:
                adj_edges[edges_num_dict[other_node][node]] = -1  # mark as -1 all outgoing arcs
        if node == start_node:
            rows.append([variable_names, adj_edges])
            rhs.append(-1)  # -1 is the right part of constraint for the start node
            senses += 'E'
        elif node == finish_node:
            rows.append([variable_names, adj_edges])
            rhs.append(1)  # 1 is the right part of constraint for the finish node
            senses += 'E'
        else:
            rows.append([variable_names, adj_edges])
            rhs.append(0)  # 0 is the right part of constraint for all nodes except start and finish
            senses += 'E'
    rownames = ['d' + str(i) for i in range(len(rhs))]
    return rows, senses, rhs, rownames


def solve_shortest_path(edges_num_dict, g, start_node, finish_node, weights, verbose=False, y_star=None, c_bar=None):
    """
    Solve the shortest path problem with cplex
    """
    num_edges = len(weights)
    obj_coefficients = weights  # objective function weights.T @ path incidence vector

    prob = cplex.Cplex()
    prob.set_log_stream(None)  # turn off logs
    prob.set_results_stream(None)
    prob.set_warning_stream(None)
    prob.objective.set_sense(prob.objective.sense.minimize)

    variables_names = ['y' + str(i) for i in range(num_edges)]  # names of variables
    prob.variables.add(obj=obj_coefficients, names=variables_names)
    [prob.variables.set_types(i, prob.variables.type.binary) for i in range(num_edges)]  # set binary variables

    # zero-one constrints
    rhs1 = np.ones(num_edges, dtype=np.int32).tolist()  # right parts of constraints
    senses1 = 'L' * num_edges  # all constraints are '<='
    rownames1 = ['b'+str(i+1) for i in range(num_edges)]  # names of constraints
    rows1 = []
    for i in range(num_edges):
        a = np.zeros(num_edges)
        a[i] = 1
        rows1.append([variables_names, a.tolist()])

    rhs2 = np.zeros(num_edges, dtype=np.int32).tolist()  # right parts of constraints
    senses2 = 'G' * num_edges  # all constraints are '>='
    rownames2 = ['c'+str(i+1) for i in range(num_edges)]  # names of constraints
    rows2 = []
    for i in range(num_edges):
        a = np.zeros(num_edges)
        a[i] = 1
        rows2.append([variables_names, a.tolist()])

    # flow-balance constraints
    rows3, senses3, rhs3, rownames3 = make_flow_balance_constraints(variables_names, num_edges, edges_num_dict, g,
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
    Returns a fully connected graph with h layers and w nodes at each layer
    """
    g = networkx.DiGraph()
    node_num = 0

    g.add_node(node_num)
    node_num += 1

    # form first layer
    current_layer_nodes = []
    for node in range(w):
        g.add_node(node_num)
        g.add_edge(0, node_num)
        current_layer_nodes.append(node_num)
        node_num += 1

    # form h layers
    for l in range(h-1):
        last_added_layer_nodes = current_layer_nodes
        current_layer_nodes = []
        for node in range(w):
            g.add_node(node_num)
            [g.add_edge(last_layer_node, node_num) for last_layer_node in last_added_layer_nodes]
            current_layer_nodes.append(node_num)
            node_num += 1

    # form finish layer
    g.add_node(node_num)
    [g.add_edge(last_layer_node, node_num) for last_layer_node in current_layer_nodes]
    return g


def solve_knapsak(W, weights):
    """
    Solve the knapsak problem with cplex: minimize weights * x w.r.t. constraint w * x > W, x={0,1}
    """
    num_edges = len(weights)
    w = np.zeros(len(weights))
    s = 0
    for i in range(len(weights)):
        s += 1
        w[i] = s
        if s == 5:
            s = 0
    obj_coefficients = weights  # objective function weights.T @ path incidence vector

    prob = cplex.Cplex()
    prob.set_log_stream(None)  # turn off logs
    prob.set_results_stream(None)
    prob.set_warning_stream(None)
    prob.objective.set_sense(prob.objective.sense.minimize)

    variables_names = ['y' + str(i) for i in range(num_edges)]  # names of variables
    prob.variables.add(obj=obj_coefficients, names=variables_names)
    [prob.variables.set_types(i, prob.variables.type.binary) for i in range(num_edges)]  # set binary variables

    # zero-one constrints
    rhs1 = np.ones(num_edges, dtype=np.int32).tolist()  # right parts of constraints
    senses1 = 'L' * num_edges  # all constraints are '<='
    rownames1 = ['b'+str(i+1) for i in range(num_edges)]  # names of constraints
    rows1 = []
    for i in range(num_edges):
        a = np.zeros(num_edges)
        a[i] = 1
        rows1.append([variables_names, a.tolist()])

    rhs2 = np.zeros(num_edges, dtype=np.int32).tolist()  # right parts of constraints
    senses2 = 'G' * num_edges  # all constraints are '>='
    rownames2 = ['c'+str(i+1) for i in range(num_edges)]  # names of constraints
    rows2 = []
    for i in range(num_edges):
        a = np.zeros(num_edges)
        a[i] = 1
        rows2.append([variables_names, a.tolist()])

    rhs3 = [W]  # right parts of constraints
    senses3 = 'G'  # all constraints are '>='
    rownames3 = ['d']  # names of constraints
    rows3 = [[variables_names, w]]
    # flow-balance constraints
    rows = rows1 + rows2 + rows3
    senses = senses1 + senses2 + senses3
    rhs = rhs1 + rhs2 + rhs3
    rownames = rownames1 + rownames2 + rownames3

    prob.linear_constraints.add(lin_expr=rows, senses=senses,
                                rhs=rhs, names=rownames)
    prob.solve()
    solution = prob.solution.get_objective_value()
    values = prob.solution.get_values()
    return solution, values
