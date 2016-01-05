import numpy as np
import pymc as pm

from gmm import gmm_model


def generate_data():

    mu_s = np.array([[0.0, 0.0, 0.0],
                     [10.0, 10.0, 10.0],
                     [-10.0, -10.0, -10.0]])
    tau_s = np.array([1.0, 1.0, 1.0])
    size_s = np.array([10, 20, 10])

    data = []
    for mu, tau, size in zip(mu_s, tau_s, size_s):
        data.append(np.random.multivariate_normal(
            mu, tau * np.eye(len(mu_s)), size))

    data = np.vstack(data)
    # np.random.shuffle(data)

    return data


def gmm_example():
    data = generate_data()

    # set parameters
    K = 3
    mu_0 = 0.0
    alpha_0 = 0.1
    beta_0 = 0.1
    alpha = 1.0

    model = gmm_model(
        data, K, mu_0=mu_0, alpha_0=alpha_0, beta_0=beta_0, alpha=alpha)

    mcmc = pm.MCMC(model)
    mcmc.sample(50000, burn=10000, thin=100)

    print "mu_k:"
    print mcmc.trace('mu_k')[-1]
    print 'z:'
    print mcmc.trace('z')[-1]

if __name__ == '__main__':
    gmm_example()
