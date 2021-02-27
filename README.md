# DRO from data
Numerical experiments for the paper "On the data-driven combinatorial optimization with incomplete information"

## Introduction
We propose the solution technique for the linear combinatorial optimization problem with the cost vectors unknown a priori. The weak optimality of our solution is proved in paper. In these experiments we show that the proposed technique dominates the baseline, which exploits Hoeffding's inequality.

## Experiment description

We compare the proposed technique with the baseline as follows. Let us find the optimal solution of the shortest path problem in fully connected graph, when the edges weights **c** are the random values from some nominal discrete distribution.  

Firstly, we find the optimal shortest path using the true mathematical expectation of the distribution for each weight. The cost of this path with nominal weights is referred to as _nominal expected loss._ 

Secondly, we generate a finite data set from this distribution. For each arc _a_ we sample  <img src="https://render.githubusercontent.com/render/math?math=T_a \in (T_{min}, T_{max})"> samples. Then we make an estimation of each arc's weight using the reviewed algorithms. The solution must me _feasible_, which means if must not fail the _finite sample guarantee_. In other words, the nominal solution must exceed the derived solution only with a small probability <img src="https://render.githubusercontent.com/render/math?math=\alpha = 0.05">.

We aim at deriving the less conservative solution. It means that the solution must be as small as possible, but still be _feasible_.

# Distributions

We hold *3* experiment modes with different dataset distribution construction techniques.

* **'uniform'** mode

For each arc we generate probability _p_ and samples number _T_ from *uniform* distribution. Then we generate _T_ points from *binomial* distribution with parameters (_d_, _p_), where _d_ is the number of possible weight's values. 
* **'binomial'** mode

In this experiment we study the influence of dataset size. We generate fewer samples with low weights. This results in less robust estimations on those arcs which are likely to be optimal. In order to do that we first generate the nominal expectations using _uniform_ mode, and then for each arc _a_ we generate dataset size number _T_ from binomial distribution with parameters (<img src="https://render.githubusercontent.com/render/math?math=T_{max} - T_{min}"> + 1, p). After that the _T_ points for each arc _a_ are generated from _binomial_ distribution with parameters (_d, p_).

* **'multinomial'** mode

In this experiment we study the influence of dependence between different arcs weights. For that purpose we generate probabilies _p_ for each arc _a_ so that all _p_ sum to 1. Then we generate data from multinomial distribution with parameters (_d, p_).

## Results

The results are expressed in the following table. Each number is the ratio of derived solution to nominal solution. Smallest numbers are the best, but they cannot not be less than 1 due to feasibility constraint.

| Distribution        | DRO technique          |  Hoeffding's technique |
| ------------- |:-------------:| -----:|
|   uniform   | **1.429 ± 0.130** | 1.686 ± 0.252 |
| binomial      | **1.440 ± 0.167** | 2.299 ± 0.654 |
| multinomial | **1.546 ± 0.164** | 2.73 ± 0.306 |

The table shows that the proposed solution outperforms the baseline in each experiment.
