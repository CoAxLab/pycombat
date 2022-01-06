from __future__ import absolute_import
import numpy as np
from sklearn.utils import check_consistent_length
from sklearn.utils.validation import column_or_1d


def _compute_lambda(del_hat_sq):
    """Estimation of hyper-parameter lambda."""
    v = np.mean(del_hat_sq)
    s2 = np.var(del_hat_sq, ddof=1)
    # In Johnson 2007  there's a typo
    # in the suppl. material as it
    # should be with v^2 and not v
    return (2*s2 + v**2)/float(s2)


def _compute_theta(del_hat_sq):
    """Estimation of hyper-parameter theta."""
    v = del_hat_sq.mean()
    s2 = np.var(del_hat_sq, ddof=1)
    return (v*s2+v**3)/s2


def _post_gamma(x, gam_hat, gam_bar, tau_bar_sq, n):
    # x is delta_star
    num = tau_bar_sq*n*gam_hat + x * gam_bar
    den = tau_bar_sq*n + x
    return num/den


def _post_delta(x, Z, lam_bar, the_bar, n):
    num = the_bar + 0.5*np.sum((Z - x[np.newaxis, :])**2, axis=0)
    den = n/2.0 + lam_bar - 1
    return num/den


def _inverse_gamma_moments(del_hat_sq):
    """Compute the inverse moments of the inverse gamma distribution."""
    lam_bar = np.apply_along_axis(_compute_lambda,
                                  arr=del_hat_sq,
                                  axis=1)
    the_bar = np.apply_along_axis(_compute_theta,
                                  arr=del_hat_sq,
                                  axis=1)

    return (lam_bar, the_bar)


def _it_eb_param(Z_batch,
                 gam_hat_batch,
                 del_hat_sq_batch,
                 gam_bar_batch,
                 tau_sq_batch,
                 lam_bar_batch,
                 the_bar_batch,
                 conv):
    """Parametric EB estimation of location and scale paramaters."""
    # Number of non nan samples within the batch for each variable
    n = np.sum(1 - np.isnan(Z_batch), axis=0)
    gam_prior = gam_hat_batch.copy()
    del_sq_prior = del_hat_sq_batch.copy()

    change = 1
    count = 0
    while change > conv:
        gam_post = _post_gamma(del_sq_prior,
                               gam_hat_batch,
                               gam_bar_batch,
                               tau_sq_batch,
                               n)

        del_sq_post = _post_delta(gam_post,
                                  Z_batch,
                                  lam_bar_batch,
                                  the_bar_batch,
                                  n)

        change = max((abs(gam_post - gam_prior) / gam_prior).max(),
                     (abs(del_sq_post - del_sq_prior) / del_sq_prior).max())
        gam_prior = gam_post
        del_sq_prior = del_sq_post
        count = count + 1

    # TODO: Make namedtuple?
    return (gam_post, del_sq_post)


def _it_eb_non_param():
    # TODO
    return


class Combat(object):
    """Combat Class."""

    def __init__(self,
                 mode='p',
                 conv=0.0001):

        if (mode == 'p') | (mode == 'np'):
            self.mode = mode
        else:
            raise ValueError("mode can only be 'p' o 'np'")

        self.conv = conv

    def fit(self, Y, b, X=None, C=None):
        """
        Fit method, where Combat parameters are estimated

        Parameters
        ----------
        Y : array like of shape (n_samples, n_features)
            Dataset to be harmonised.
        b : array like of shape (n_samples, )
            Vector of batch ids.
        X : array like of shape (n_samples, mx), optional
            Matrix with mx columns whose effects we want to preserve.
            The default is None.
        C : array like of shape (n_samples, mc), optional
            Matrix with mc columns whose effects we want to remove.
            The default is None.

        Returns
        -------
        self:  object
            Fitted Estimator

        """
        # Check length of all input arrays and ensure b is a vector
        check_consistent_length(Y, b, X, C)
        b = column_or_1d(b)

        # extract unique batch categories
        batches = np.unique(b)
        self.batches_ = batches

        # Construct one-hot-encoding matrix for batches
        B = np.column_stack([(b == b_name).astype(int)
                             for b_name in self.batches_])

        n_samples, n_features = Y.shape
        n_batch = B.shape[1]

        if n_batch == 1:
            raise ValueError('The number of batches should be at least 2')

        sample_per_batch = B.sum(axis=0)

        if np.any(sample_per_batch == 1):
            raise ValueError('Each batch should have at least 2 observations'
                             'In the future, when this does not happens,'
                             'only mean adjustment will take place')

        # Construct design matrix
        M = B.copy()
        if isinstance(X, np.ndarray):
            M = np.column_stack((M, X))
            end_x = n_batch + X.shape[1]
        else:
            end_x = n_batch

        if isinstance(C, np.ndarray):
            M = np.column_stack((M, C))
            end_c = end_x + C.shape[1]
        else:
            end_c = end_x

        # Fixed effects
        beta_hat = np.matmul(np.linalg.inv(np.matmul(M.T, M)),
                             np.matmul(M.T, Y))

        # Find intercepts
        alpha_hat = np.matmul(sample_per_batch/float(n_samples),
                              beta_hat[:n_batch, :])
        self.intercept_ = alpha_hat
        # Find slopes for covariates/effects
        beta_x = beta_hat[n_batch:end_x, :]
        self.coefs_x_ = beta_x

        beta_c = beta_hat[end_x:end_c, :]
        self.coefs_c_ = beta_c

        Y_hat = np.matmul(M, beta_hat)
        sigma = np.mean(((Y - Y_hat)**2), axis=0)
        self.epsilon_ = sigma

        # Standarise data
        Z = Y.copy()
        Z -= alpha_hat[np.newaxis, :]
        Z -= np.matmul(M[:, n_batch:end_x], beta_x)
        Z -= np.matmul(M[:, end_x:end_c], beta_c)
        Z /= np.sqrt(sigma)

        # Find gamma fitted to standard data
        gam_hat = np.matmul(np.linalg.inv(np.matmul(B.T, B)),
                            np.matmul(B.T, Z)
                            )
        # Mean across response variable
        gam_bar = np.mean(gam_hat, axis=1)
        # Variance across response variable
        tau_bar_sq = np.var(gam_hat, axis=1, ddof=1)

        # Variance per batch and gen
        del_hat_sq = [np.var(Z[B[:, ii] == 1, :], axis=0, ddof=1)
                      for ii in range(B.shape[1])]
        del_hat_sq = np.array(del_hat_sq)

        lam_bar, the_bar = _inverse_gamma_moments(del_hat_sq)

        if self.mode == 'p':
            it_eb = _it_eb_param
        else:
            it_eb = _it_eb_non_param

        gam_star, del_sq_star = [], []
        for ii in range(B.shape[1]):
            g, d_sq = it_eb(Z[B[:, ii] == 1, :],
                            gam_hat[ii, :],
                            del_hat_sq[ii, :],
                            gam_bar[ii],
                            tau_bar_sq[ii],
                            lam_bar[ii],
                            the_bar[ii],
                            self.conv)

            gam_star.append(g)
            del_sq_star.append(d_sq)

        gam_star = np.array(gam_star)
        del_sq_star = np.array(del_sq_star)

        self.gamma_ = gam_star
        self.delta_sq_ = del_sq_star

        return self

    def transform(self, Y, b, X=None, C=None):
        """
        Transform method that effectively harmonises the input data.

        Parameters
        ----------
        Y : array like of shape (n_samples, n_features)
            Dataset to be harmonised.
        b : array like of shape (n_samples, )
            Vector of batch ids.
        X : array like of shape (n_samples, mx), optional
            Matrix with mx columns whose effects we want to preserve.
            The default is None.
        C : array like of shape (n_samples, mc), optional
            Matrix with mc columns whose effects we want to remove.
            The default is None.

        Returns
        -------
        Y_trans : array like of shape (n_samples, n_features)
            Harmonised dataset.
        """
        Y, b, X, C = self._validate_for_transform(Y, b, X, C)

        test_batches = np.unique(b)

        # First standarise again the data
        Y_trans = Y - self.intercept_[np.newaxis, :]

        if self.coefs_x_.size > 0:
            Y_trans -= np.matmul(X, self.coefs_x_)

        if self.coefs_c_.size > 0:
            Y_trans -= np.matmul(C, self.coefs_c_)

        Y_trans /= np.sqrt(self.epsilon_)

        for batch in test_batches:

            ix_batch = np.where(self.batches_ == batch)[0]

            Y_trans[b == batch, :] -= self.gamma_[ix_batch]
            Y_trans[b == batch, :] /= np.sqrt(self.delta_sq_[ix_batch, :])
        Y_trans *= np.sqrt(self.epsilon_)

        # Add intercept
        Y_trans += self.intercept_[np.newaxis, :]

        if self.coefs_x_.size > 0:
            Y_trans += np.matmul(X, self.coefs_x_)

        return Y_trans

    def fit_transform(self, Y, b, X=None, C=None):
        """
        Fit transform method, that first estimates the Combat parameters
        and then harmonises the data.

        Y : array like of shape (n_samples, n_features)
            Dataset to be harmonised.
        b : array like of shape (n_samples, )
            Vector of batch ids.
        X : array like of shape (n_samples, mx), optional
            Matrix with mx columns whose effects we want to preserve.
            The default is None.
        C : array like of shape (n_samples, mc), optional
            Matrix with mc columns whose effects we want to remove.
            The default is None.

        Returns
        -------
        Y_trans : array like of shape (n_samples, n_features)
            Harmonised dataset.

        """
        return self.fit(Y, b, X, C).transform(Y, b, X, C)

    def _validate_for_transform(self, Y, b, X, C):

        check_consistent_length(Y, b, X, C)
        b = column_or_1d(b)

        # check if fitted
        attributes = ['intercept_', 'coefs_x_',
                      'coefs_c_', 'epsilon_',
                      'gamma_', 'delta_sq_']

        attrs = all([hasattr(self, attr) for attr in attributes])
        if not attrs:
            raise AttributeError("Combat was not fitted?")

        if Y.shape[1] != len(self.intercept_):
            raise ValueError("Wrong number of features for Y")
        if len(np.unique(b)) != self.gamma_.shape[0]:
            raise ValueError("Wrong number of categories for b")
        if isinstance(X, np.ndarray):
            if X.shape[1] != self.coefs_x_.shape[0]:
                raise ValueError("Dimensions of fitted beta "
                                 "and input X matrix do not match")
        if isinstance(C, np.ndarray):
            if C.shape[1] != self.coefs_c_.shape[0]:
                raise ValueError("Dimensions of fitted beta "
                                 "and input C matrix do not match")

        return Y, b, X, C
