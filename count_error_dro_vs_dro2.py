import numpy as np


def main():
    exp = 'exp6'
    c_bar = np.load(f'{exp}/c_bar.npy')
    c_worst_dro = np.load(f'{exp}/c_worst_dro.npy')
    c_worst_dro2 = np.load(f'{exp}/c_worst_hoef.npy')
    c_worst_dro2 = c_worst_dro2[:, -1]
    c_worst_dro = c_worst_dro[:, -1]
    c_bar = c_bar[:, -1]
    errors_dro = np.zeros(len(c_worst_dro))
    errors_dro2 = np.zeros(len(c_worst_dro))
    for exp in range(c_worst_dro2.shape[0]):
        c_bar_exp = c_bar[exp]
        c_worst_dro2_exp = c_worst_dro2[exp]
        c_worst_dro_exp = c_worst_dro[exp]
        num_edges = c_worst_dro_exp.shape[0]
        for i in range(num_edges):
            for j in range(num_edges):
                if c_bar_exp[i] < c_bar_exp[j] and c_worst_dro2_exp[i] > c_worst_dro2_exp[j]:
                    errors_dro2[exp] += 1
                if c_bar_exp[i] < c_bar_exp[j] and c_worst_dro_exp[i] > c_worst_dro_exp[j]:
                    errors_dro[exp] += 1

    print("Number of errors DRO:", np.mean(errors_dro), np.std(errors_dro))
    print("Number of errors DRO2:", np.mean(errors_dro2), np.std(errors_dro2))

if __name__ == '__main__':
    main()
