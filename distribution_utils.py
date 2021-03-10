import numpy as np
import scipy
import scipy.stats


def create_binomial_costs_with_binomial_T(m, d=10, T_min=10, T_max=100, verbose=False):
    # c_bar is the nominal mean of the generated data
    _, c_bar, _, _ = create_binomial_costs(m, d, T_min, T_max)

    p = (c_bar - np.min(c_bar)) / (np.max(c_bar) - np.min(c_bar)) # in fact, p = c_bar_normalized

    T_binomial = np.zeros(m, dtype=np.int32)
    for a in range(m):
        T_binomial[a] = scipy.stats.binom.rvs(n=T_max - T_min + 1, p=p[a]) + T_min
    c_hat, c_bar = generate_binomial_samples(T_binomial, p, d, verbose=verbose)
    return c_hat, c_bar, T_binomial, p


def create_multinomial_costs(m, d=10, T_min=10, T_max=100, verbose=False):
    T = np.random.randint(T_min, T_max, size=m)  # uniform
    p = np.random.uniform(size=m)
    p /= np.sum(p)

    complete_multinomial_data = scipy.stats.multinomial.rvs(n=d-1, p=p, size=T_max).transpose(1, 0) + 1
    c_hat = []  # incomplete data for every edge
    for a in range(m):
        c_hat.append(complete_multinomial_data[a, :T[a] - T_min + 1])

    c_bar = p * (d - 1) + 1  # the nominal mean for each arc
    if verbose:
        print("Generated distribution:")
        print("Check mean estimation:", np.array([np.mean(c) for c in c_hat]), '\nvs \n', c_bar)
    return c_hat, c_bar, T, p


def create_binomial_costs(m, d=10, T_min=10, T_max=100, verbose=False):
    p = np.random.uniform(size=m)
    T = np.random.randint(T_min, T_max + 1, size=m)  # uniform T
    c_hat, c_bar = generate_binomial_samples(T, p, d, verbose=verbose)
    return c_hat, c_bar, T, p


def generate_binomial_samples(T, p, d, verbose=False):
    c_hat = []  # incomplete data for every arc
    for a in range(len(T)):
        c_hat.append(scipy.stats.binom.rvs(n=d-1, p=p[a], size=T[a]) + 1)    
    c_bar = (d - 1)*p + 1  # nominal mean
    if verbose:
        print("Generated distribution")
        print("Check mean estimation:", np.mean(c_hat[0]), c_bar[0])

    return c_hat, c_bar
