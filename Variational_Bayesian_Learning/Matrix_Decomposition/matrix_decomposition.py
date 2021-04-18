import numpy as np
from scipy import linalg


class SVD:
    def __init__(self):
        pass

    def fit(self, v, h):
        self.h = h
        self.wb, self.gamma, self.wa = linalg.svd(v, full_matrices=False)
        self.wa_h = self.wa[:self.h, :]
        self.wb_h = self.wb[:, :self.h]
        self.gamma_diag = np.diag(self.gamma[:self.h])
        return self.wb_h, self.gamma[:self.h], self.wa_h

    def reconstruct(self):
        u = self.wb_h @ self.gamma_diag @ self.wa_h
        return u


class VBMD(SVD):
    def __init__(self, sigma, ca_square, cb_square):
        """Variational_Bayesian_Matirx_Decomposition
        https://jmlr.org/papers/v16/nakajima15a.html
        """
        super(VBMD, self).__init__()
        self.sigma = sigma
        self.ca_square = ca_square
        self.cb_square = cb_square

    def fit(self, v, h):
        self.h = h
        self.wb, self.gamma, self.wa = linalg.svd(v, full_matrices=False)
        self.wa_h = self.wa[:self.h, :]
        self.wb_h = self.wb[:, :self.h]

        L = v.shape[0]
        M = v.shape[1]

        c_ab_square = self.ca_square * self.cb_square
        term_1 = (L + M) / 2 + self.sigma**2 / (2 * c_ab_square)
        gamma_vb_threshold = self.sigma * \
            np.sqrt(term_1 + np.sqrt(term_1**2 - L*M))

        gamma_h = self.gamma[:self.h]
        gamma_vb_hat = gamma_h * (1 - (self.sigma**2 / (2 * (gamma_h**2))) *
                                  (M + L + np.sqrt((M-L)**2 + 4*(gamma_h**2) /
                                                   c_ab_square)))
        self.gamma_vb = np.where(gamma_h > gamma_vb_threshold, gamma_vb_hat, 0)
        self.gamma_diag = np.diag(self.gamma_vb)

        return self.wb_h, self.gamma_vb, self.wa_h


class EVBMD(SVD):
    def __init__(self, approximate_coefficient=2.5129):
        """Empirical_Variational_Bayesian_Matirx_Decomposition
        https://jmlr.org/papers/v16/nakajima15a.html
        """
        super(EVBMD, self).__init__()
        self.approximate_coefficient = approximate_coefficient

    def fit(self, v):
        self.wb, self.gamma, self.wa = linalg.svd(v, full_matrices=False)
        L = v.shape[0]
        M = v.shape[1]
        assert L <= M

        alpha = L / M
        tau_approximate = self.approximate_coefficient * np.sqrt(alpha)

        h = L-2
        x_bar = (1 + tau_approximate) * (1 + alpha / tau_approximate)
        h_bar = min(np.abs(L/(1+alpha))-1, h)
        sigma_h_plus_1_square = (self.gamma[h]**2) / (M*x_bar)

        def tau(x): return (1/2) * (x - (1 + alpha) +
                                    np.sqrt((x - (1 + alpha))**2 - 4 * alpha))

        def psi_0(x): return x - np.log(x)
        def psi_1(x): return np.log(tau(x)+1) * alpha * \
            np.log(tau(x) / alpha + 1) - tau(x)

        def theta(b): return np.array(b, dtype=np.uint)
        def psi(x): return psi_0(x) + theta(x > x_bar) * psi_1(x)

        def omega(sigma, inv=False):
            if inv:
                sigma = 1/sigma
            term_1 = np.sum(psi((self.gamma[:h]**2) / (M*(sigma**2))))
            term_2 = np.sum(psi_0((self.gamma[h:L]**2) / (M*(sigma**2))))
            return (1/L) * (term_1 + term_2)

        def linear_search(lower_bound, upper_bound, f, numpoints=100):
            target = np.linspace(lower_bound, upper_bound, numpoints)
            min_idx = np.argmin(np.vectorize(f)(target))
            return target[min_idx]

        lower_bound = min(sigma_h_plus_1_square,
                          np.sum(self.gamma[h:L] ** 2) / (M * (L - h_bar)))
        upper_bound = (1 / (L*M)) * np.sum(self.gamma[:L]**2)
        sigma_h_inv = (M*x_bar) / (self.gamma**2)
        sigma_estimator = linear_search(lower_bound, upper_bound, omega)
        sigma_estimator_inv = 1 / sigma_estimator
        self.h_evb = np.sum(sigma_h_inv <= sigma_estimator_inv)
        self.wa_h = self.wa[:self.h_evb, :]
        self.wb_h = self.wb[:, :self.h_evb]
        gamma_h = self.gamma[:self.h_evb]

        gamma_evb_threshold = sigma_estimator * \
            np.sqrt(M*(1+tau_approximate)*(1+(alpha/tau_approximate)))
        term = 1 - (M + L) * (sigma_estimator**2) / (gamma_h**2)
        gamma_evb_hat = (gamma_h / 2) *\
                        (term + np.sqrt(term ** 2 -
                                        (4 * L * M * (sigma_estimator**4) /
                                         (tau_approximate**4))))
        self.gamma_evb = np.where(
            gamma_h > gamma_evb_threshold, gamma_evb_hat, 0)
        self.gamma_diag = np.diag(self.gamma_evb)

        return self.wb_h, self.gamma_evb, self.wa_h
