"""
A bayesian gaussian mixture model implementation with PyMC.

Note: 
We use MCMC here but it is not an efficient method for mixture model.
A better implementation should consider EM algorithm or Gibbs sampling. 
So don't test it with large dataset, it will take long to converge.
"""

import numpy as np
import pymc as pm


def gmm_model(data, K, mu_0=0.0, alpha_0=0.1, beta_0=0.1, alpha=1.0):
    """
    K: number of component
    n_samples: number of n_samples
    n_features: number of features

    mu_0: prior mean of mu_k 
    alpha_0: alpha of Inverse Gamma tau_k 
    beta_0: beta of Inverse Gamma tau_k
    alpha = prior of dirichlet distribution phi_0

    latent variable:
    phi_0: shape = (K-1, ), dirichlet distribution
    phi: shape = (K, ), add K-th value back to phi_0
    z: shape = (n_samples, ), Categorical distribution, z[k] is component indicator 
    mu_k: shape = (K, n_features), normal distribution, mu_k[k] is mean of k-th component
    tau_k : shape = (K, n_features), inverse-gamma distribution, tau_k[k] is variance of k-th component
    """

    n_samples, n_features = data.shape

    # latent variables
    tau_k = pm.InverseGamma(
        'tau_k', alpha_0 * np.ones((K, n_features)), beta_0 * np.ones((K, n_features)), value=beta_0 * np.ones((K, n_features)))
    mu_k = pm.Normal('mu_k', np.ones((K, n_features)) *
                     mu_0, tau_k, value=np.ones((K, n_features)) * mu_0)
    phi_0 = pm.Dirichlet('phi_0', theta=np.ones(K) * alpha)

    @pm.deterministic(dtype=float)
    def phi(value=np.ones(K) / K, phi_0=phi_0):
        val = np.hstack((phi_0, (1 - np.sum(phi_0))))
        return val

    z = pm.Categorical(
        'z', p=phi, value=pm.rcategorical(np.ones(K) / K, size=n_samples))

    # observed variables
    x = pm.Normal('x', mu=mu_k[z], tau=tau_k[z], value=data, observed=True)

    return pm.Model([mu_k, tau_k, phi_0, phi, z, x])
