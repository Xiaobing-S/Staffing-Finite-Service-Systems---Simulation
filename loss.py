import numpy as np
from Metrics import *
import matplotlib.pyplot as plt
from tqdm import tqdm


def compute_lb_EX(m, n, lambda_, mu):
    assert m / n < lambda_ / (lambda_ + mu)
    return m - 2 * mu / (lambda_ * n / m - (lambda_ + mu))


def compute_ub_EX(m, n, lambda_, mu, nu):
    assert m / n < lambda_ / (lambda_ + mu)
    return m - 1 / (1 + nu * (n - m + 1) / (m * mu))


def compute_lb_alpha(m, n, lambda_, mu, nu):
    if float(m) / n < lambda_ / (lambda_ + mu):
        lb_EX = compute_lb_EX(m, n, lambda_, mu)
        numerator = lb_EX * mu
        denominator = lb_EX * mu + (n - (lambda_ + mu) / lambda_ * lb_EX) * nu
    else:
        tmp = lambda_ * m * n / (lambda_ * n + mu * m)
        numerator = tmp * mu
        denominator = n * nu - (nu + nu / lambda_ * mu - mu) * tmp
    return numerator / denominator


def compute_ub_alpha(m, n, lambda_, mu, nu):
    if float(m) / n < lambda_ / (lambda_ + mu):
        ub_EX = compute_ub_EX(m, n, lambda_, mu, nu)
        numerator = ub_EX * mu
        denominator = ub_EX * mu + (n - (lambda_ + mu) / lambda_ * ub_EX) * nu
    else:
        numerator = mu * m
        denominator = n * nu - (nu + nu / lambda_ * mu - mu) * m
    return numerator / denominator


def plot_gap_of_loss(given_alpha, lambda_, mu, nu, n_lb, n_ub, n_step=1):
    n_list = np.arange(n_lb, n_ub, n_step)
    realized_alphas = np.zeros(len(n_list))
    c = compute_c(lambda_, mu, nu, given_alpha)

    for i, n in tqdm(enumerate(n_list)):
        nominal_load = int(c * n)
        realized_alphas[i] = compute_availability_handler(
            nominal_load,
            n,
            lambda_,
            mu,
            nu,
            True,
        )

    plt.scatter(n_list, realized_alphas, label="reallized_alpha")
    plt.xlabel(r"$n$")
    plt.ylabel(r"$\alpha$")
    plt.axhline(y=given_alpha, label="given_alpha")
    plt.title(rf"$\lambda$={lambda_}, $\mu$={mu}, $\nu$={nu}")
    plt.legend()
    plt.show()
    plt.close()


def plot_bounds_of_alpha(n, lambda_, mu, nu, n_step=1):
    m_list = np.arange(1, n, n_step)
    N = len(m_list)
    alphas = np.zeros(N)
    alphas_lb = np.zeros(N)
    alphas_ub = np.zeros(N)

    for i, m in tqdm(enumerate(m_list)):
        alphas[i] = compute_availability_handler(m, n, lambda_, mu, nu, True)
        alphas_lb[i] = compute_lb_alpha(m, n, lambda_, mu, nu)
        alphas_ub[i] = compute_ub_alpha(m, n, lambda_, mu, nu)

    plt.plot(m_list, alphas, label=r"$\bar \alpha(m, n)$")
    plt.scatter(m_list, alphas_lb, label="lb", marker="*")
    plt.scatter(m_list, alphas_ub, label="ub", marker="o")
    plt.xlabel(r"$m$")
    plt.ylim([0, 1])
    plt.ylabel(r"$\alpha$")
    plt.title(rf"$\lambda$={lambda_}, $\mu$={mu}, $\nu$={nu}, $n$={n}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    plot_gap_of_loss(given_alpha=0.5, lambda_=1, mu=1, nu=2, n_lb=2, n_ub=200, n_step=5)
    plot_bounds_of_alpha(n=60, lambda_=1, mu=1, nu=2, n_step=2)
