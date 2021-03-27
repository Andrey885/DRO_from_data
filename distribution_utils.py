import numpy as np
import scipy
import scipy.stats
import scipy.integrate
import math


def get_binomial_T(m, d, T_min, T_max, fixed_p=None):
    # c_bar is the nominal mean of the generated data
    _, c_bar, _, _ = create_binomial_costs(m, d, T_min, T_max, fixed_p=fixed_p)
    p = (c_bar - np.min(c_bar)) / (np.max(c_bar) - np.min(c_bar)) # in fact, p = c_bar_normalized

    T_binomial = np.zeros(m, dtype=np.int32)
    for a in range(m):
        T_binomial[a] = scipy.stats.binom.rvs(n=T_max - T_min + 1, p=p[a]) + T_min
    return T_binomial, p


def get_binomial_T_reversed(m, d, T_min, T_max, fixed_p=None):
    # c_bar is the nominal mean of the generated data
    _, c_bar, _, _ = create_binomial_costs(m, d, T_min, T_max, fixed_p=fixed_p)
    p = 1 - (c_bar - np.min(c_bar)) / (np.max(c_bar) - np.min(c_bar)) # in fact, p = c_bar_normalized
    T_binomial = np.zeros(m, dtype=np.int32)
    for a in range(m):
        T_binomial[a] = T_max - scipy.stats.binom.rvs(n=T_max - T_min + 1, p=p[a])
    return T_binomial, p


def create_binomial_costs_with_binomial_T_reverse(m, d=10, T_min=10, T_max=100, verbose=False,
                                          fixed_p=None):
    T_binomial, p = get_binomial_T_reversed(m, d, T_min, T_max, fixed_p)
    c_hat, c_bar = generate_binomial_samples(T_binomial, p, d, verbose=verbose)
    return c_hat, c_bar, T_binomial, p


def create_binomial_costs_with_binomial_T(m, d=10, T_min=10, T_max=100, verbose=False,
                                          fixed_p=None):
    T_binomial, p = get_binomial_T(m, d, T_min, T_max, fixed_p)

    c_hat, c_bar = generate_binomial_samples(T_binomial, p, d, verbose=verbose)
    return c_hat, c_bar, T_binomial, p


def create_multinomial_costs(m, d=10, T_min=10, T_max=100, verbose=False, fixed_p=None):
    T = np.random.randint(T_min, T_max + 1, size=m)  # uniform
    if fixed_p is not None:
        p = fixed_p
    else:
        p = np.random.uniform(size=m)
        p /= np.sum(p)
    complete_multinomial_data = scipy.stats.multinomial.rvs(n=d-1, p=p, size=T_max).transpose(1, 0) + 1
    c_hat = []  # incomplete data for every edge
    for a in range(m):
        c_hat.append(complete_multinomial_data[a, :T[a]])

    c_bar = p * (d - 1) + 1  # the nominal mean for each arc
    if verbose:
        print("Generated distribution:")
        print("Check mean estimation:", np.array([np.mean(c) for c in c_hat]), '\nvs \n', c_bar)
    return c_hat, c_bar, T, p


def create_binomial_costs(m, d=10, T_min=10, T_max=100, verbose=False, fixed_p=None):
    if fixed_p is not None:
        p = fixed_p
    else:
        p = np.random.uniform(size=m)
    T = np.random.randint(T_min, T_max + 1, size=m)  # uniform T
    c_hat, c_bar = generate_binomial_samples(T, p, d, verbose=verbose)
    return c_hat, c_bar, T, p


def create_normal_costs(m, d=10, T_min=10, T_max=100, std=2, verbose=False, fixed_p=None):
    def count_disrete_gaussian_distribution(mean, std):
        def gaussian(x):
            return math.exp(- ((x - mean) ** 2) / (2 * (std ** 2))) / (math.sqrt(2 * math.pi) * std)

        p = np.zeros(d)
        p[0] = scipy.integrate.quad(gaussian, 0.5, 1.5)[0]
        for i in range(1, d - 1):
            p[i] = scipy.integrate.quad(gaussian, i+0.5, i+1.5)[0]
        p[d-1] = scipy.integrate.quad(gaussian, d-0.5, d+0.5)[0]
        p /= np.sum(p)
        return p

    # T, _ = get_binomial_T(m, d, T_min, T_max)  # binomial T
    T = np.random.randint(T_min, T_max + 1, size=m)  # uniform T
    c_hat = []  # incomplete data for every arc
    c_bar = np.zeros(m)
    full_p = [] if fixed_p is None else fixed_p
    for a in range(m):
        if fixed_p is None:
            mean = np.random.randint(1, d)
        else:
            mean = fixed_p[a]
        p = count_disrete_gaussian_distribution(mean, std)
        expectation = np.sum(np.linspace(1, d, d) * p)
        full_p.append(mean)
        p = np.concatenate((np.array([0]), p))
        p = np.cumsum(p)
        float_uniform_0_1_values = np.random.rand(T[a])
        integer_normal_values = np.zeros(T[a])
        for i in range(1, d+1):
            integer_normal_values[(float_uniform_0_1_values > p[i-1]) & (float_uniform_0_1_values <= p[i])] = i
        c_hat.append(integer_normal_values)
        c_bar[a] = expectation
    if verbose:
        print("Generated distribution")
        print("Check mean estimation:", np.mean(c_hat[0]), c_bar[0])
    return c_hat, c_bar, T, full_p


def create_normal_costs_deprecated(m, d=10, T_min=10, T_max=100, std=2, verbose=False):
    """
    The function is deprecated because the tails of gaussian "spoil" the disrete gaussian distribution.
    """
    def gaussian(x):
        return math.exp(- ((x - mean) ** 2) / (2 * (std ** 2))) / (math.sqrt(2 * math.pi) * std)

    def count_expectation():
        p = np.zeros(d)
        p[0] = scipy.integrate.quad(gaussian, -np.inf, 1.5)[0]
        expectation = 1 * p[0]
        for i in range(1, d - 1):
            p[i] = scipy.integrate.quad(gaussian, i+0.5, i+1.5)[0]
            expectation += (i + 1) * p[i]
        p[d-1] = scipy.integrate.quad(gaussian, d-0.5, np.inf)[0]
        expectation += d * p[d-1]
        assert abs(p.sum() - 1) < 1e-3
        return expectation
    T = np.random.randint(T_min, T_max + 1, size=m)  # uniform T
    c_hat = []  # incomplete data for every arc
    c_bar = np.zeros(m)
    for a in range(len(T)):
        mean = np.random.randint(1, d)
        normal_values = np.random.randn(T[a]) * std + mean
        normal_values[normal_values <= 0.5] = 1
        normal_values[normal_values >= d + 0.5] = d
        normal_values = np.round(normal_values, 0).astype(int)
        c_hat.append(normal_values)
        c_bar[a] = count_expectation()
    if verbose:
        print("Generated distribution")
        print("Check mean estimation:", np.mean(c_hat[0]), c_bar[0])
    return c_hat, c_bar, T, None


def generate_binomial_samples(T, p, d, verbose=False):
    c_hat = []  # incomplete data for every arc
    for a in range(len(T)):
        c_hat.append(scipy.stats.binom.rvs(n=d-1, p=p[a], size=T[a]) + 1)
    c_bar = (d - 1)*p + 1  # nominal mean
    if verbose:
        print("Generated distribution")
        print("Check mean estimation:", np.mean(c_hat[0]), c_bar[0])

    return c_hat, c_bar
