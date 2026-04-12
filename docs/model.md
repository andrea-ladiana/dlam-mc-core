# Model Notes

This document summarizes the implementation choices used by `dlam_mc_core`.

## Layered data model

For each layer `l in {1,2,3}` and archetype `mu in {1,...,K}`:

- latent pattern: `xi_l^mu in {-1,+1}^N`,
- noisy examples: `eta_l^{mu,a} = xi_l^mu * chi_l^{mu,a}`,
- with `P(chi=+1) = (1+r_l)/2`.

The empirical mean over examples is:

`U_l^mu = (1/M_l) sum_a eta_l^{mu,a}`.

Dataset entropy is:

`rho_l = (1-r_l^2) / (M_l r_l^2)`.

The normalized empirical archetype is:

`hat_xi_l^mu = U_l^mu / (r_l (1+rho_l))`.

## Dreaming kernel

Given the empirical correlation matrix

`C_l = (hat_xi_l hat_xi_l^T) / N_l`,

the dreaming kernel is

`A_l(t_l) = (1+t_l) (I + t_l C_l)^(-1)`.

The simulator computes this matrix with a linear solve for numerical stability.

## Effective local fields

At every Monte Carlo step, each layer field is computed in factorized form:

- auto term from `A_l`,
- cross-layer terms from couplings `g_lm`.

This reproduces the same algebraic structure as the optimized script in the original project while keeping the code fully readable and testable.

## Dynamics

Spins are updated synchronously with Glauber dynamics:

`P(s_i = +1) = 0.5 * (1 + tanh(beta * h_i))`.

Magnetizations are tracked against the target archetype `mu0` for each layer across the full trajectory.
