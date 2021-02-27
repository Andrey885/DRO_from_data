import numpy as np
import scipy
import scipy.stats


def create_weights_distribution_with_binomial_T(m, d=10, T_min=100, T_max=1000, verbose=False):
    _, c_bar, _, _ = create_weights_distribution_from_uniform_T(m, d, T_min, T_max)
    # c_bar is true mathematical expectation of generated data
    alpha_a = (c_bar - np.min(c_bar)) / (np.max(c_bar) - np.min(c_bar))
    E_T = T_min + alpha_a * (T_max - T_min)  # make size of data proportional to its expectation
    p = (E_T - T_min) / (T_max - T_min)
    Ta_binomial = np.zeros(len(c_bar), dtype=np.int32)
    for i in range(len(Ta_binomial)):
        Ta_binomial[i] = scipy.stats.binom.rvs(n=T_max - T_min + 1, p=p[i]) + T_min
    c_hat, c_bar = create_binomial_weights_distribution_from_T(Ta_binomial, p, d, verbose=verbose)
    return c_hat, c_bar, Ta_binomial, p



def create_weights_distribution_with_multinomial_T(m, d=10, T_min=100, T_max=1000, verbose=False):
    T = np.random.randint(T_min, T_max, size=m)  # uniform
    p = np.random.uniform(size=m)
    p /= np.sum(p)

    complete_multinomial_data = scipy.stats.multinomial.rvs(n=d-1, p=p, size=T_max).transpose(1, 0) + 1
    c_hat = []  # incomplete data for every edge
    for a in range(m):
        c_hat.append(complete_multinomial_data[a, :T[a] - T_min + 1])
    c_bar = p * (d - 1) + 1  # true mathematical expectation for each arc
    if verbose:
        print("Generated distribution:")
        print("Check mean estimation:", np.array([np.mean(c) for c in c_hat]), '\nvs \n', c_bar)
    return c_hat, c_bar, T, p


def create_weights_distribution_from_uniform_T(m, d=10, T_min=100, T_max=1000, verbose=False):
    p = np.random.uniform(size=m)  # from uniform dist
    T = np.random.randint(T_min, T_max, size=m)  # uniform
    c_hat, c_bar = create_binomial_weights_distribution_from_T(T, p, d, verbose=verbose)
    return c_hat, c_bar, T, p


def create_binomial_weights_distribution_from_T(T, p, d, verbose=False):
    c_hat = []  # incomplete data for every edge
    for a in range(len(T)):
        c_hat.append(scipy.stats.binom.rvs(n=d-1, p=p[a], size=T[a]) + 1)
    c_bar = (d-1)*p + 1  # true mathematical expectation
    if verbose:
        print("Generated distribution")
        print("Check mean estimation:", np.mean(c_hat[0]), c_bar[0])
    return c_hat, c_bar
