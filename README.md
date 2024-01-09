# Granger Causality Estimator

## Granger Causality mathemetical bases

Granger Causality is a common method to express effects of a time series to another one, in this context, cause may be prefered to replaced by predection. Granger Causality is defined as below:

Assume we have two time series x(t) and y(t), and we want to estimate how much does x(t) affect by y(t). We may consider time series x(t) as a Auto-Regressive Process, means that

*x(t) = a_1 * x(t - 1) + a_2 * x(t - 2) + ... a_n * x(t - n)*, 

and if y(t) have some effects on signal x(t), we may consider it as, 

*x(t) = a_1 * x(t - 1) + a_2 * x(t - 2) + ... a_n * x(t - n) + b_1 * y(t - 1) + b_2 * y(t - 2) + ... + b_m * y(t - m)*

so, if we have two signals x and y and once try to estimate x(t) as an AR process by itself with error e_i and then try to estimate it as a combination of itself and signal y(t) with error value e_b, we define Granger Causality as

*GC = log(e_i / e_b)*

## References

[1] [Granger Causaltiy in Wikipedia](https://en.wikipedia.org/wiki/Granger_causality)

** Please note that this project is not completed yet, for more details contact me please! **