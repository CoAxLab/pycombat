"""

"""

from __future__ import absolute_import
import pandas as pd
import numpy as np


def _compute_lambda(del_hat_sq):
    """
    
    Estimation of hyper-parameter lambda
    
    Parameters
    ----------
    del_hat_sq : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    v = np.mean(del_hat_sq)
    s2 = np.var(del_hat_sq, ddof=1)
    # In Johnson 2007  there's a typo
    # in the suppl. material as it
    # should be with v^2 and not v
    return (2*s2 +v**2) / float(s2)

def _compute_theta(del_hat_sq):
    """
    
    Estimation of hyper-parameter theta

    Parameters
    ----------
    del_hat_sq : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    v = del_hat_sq.mean()
    s2 = np.var(del_hat_sq, ddof=1)
    return (v*s2+v**3)/s2


def _post_gamma(x, gam_hat, gam_bar, tau_bar_sq, n):
    # x is delta_star
    num = tau_bar_sq*n*gam_hat + x * gam_bar
    den = tau_bar_sq*n + x
    return num/den

def _post_delta(x, Z, lam_bar, the_bar, n):
    num = the_bar + 0.5*np.sum((Z - x[np.newaxis,:])**2, axis=0)
    den = n/2.0 + lam_bar - 1 
    return num/den

def _inverse_gamma_moments(del_hat_sq):
    """
    Function to compute the inverse moments of 
    the inverse gamma distribution.

    Parameters
    ----------
    delta_hat_sq : TYPE
        DESCRIPTION.

    Returns
    -------
    lam_bar : TYPE
        DESCRIPTION.
    the_bar : TYPE
        DESCRIPTION.

    """
    
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
    """
    
    Parametric EB estimation of location
    and scale paramaters.

    Parameters
    ----------
    Z_batch : TYPE
        DESCRIPTION.
    gam_hat_batch : TYPE
        DESCRIPTION.
    del_hat_sq_batch : TYPE
        DESCRIPTION.
    gam_bar_batch : TYPE
        DESCRIPTION.
    tau_sq_batch : TYPE
        DESCRIPTION.
    lam_bar_batch : TYPE
        DESCRIPTION.
    the_bar_batch : TYPE
        DESCRIPTION.
    conv : TYPE
        DESCRIPTION.

    Returns
    -------
    gam_post : TYPE
        DESCRIPTION.
    del_sq_post : TYPE
        DESCRIPTION.

    """
    
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
            
    #TODO: Make namedtuple?
    return (gam_post, del_sq_post)

def _it_eb_non_param():
    #TODO
    return 

class Combat(object):
    
    def __init__(self, 
                 mode='p', 
                 conv=0.0001):
        
        
        if (mode == 'p') | (mode == 'np'):
            self.mode = mode
        else:
            raise IOError("mode can only be 'p' o 'np'")
        
        self.conv = conv
        
    
    def fit(self, 
            Y,  
            b, 
            X = None, 
            C = None):
       
       n_sample = Y.shape[0]         
       B = pd.get_dummies(b).values
       n_batch = B.shape[1]
       sample_per_batch = B.sum(axis=0)

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
       
       beta_hat = np.matmul(np.linalg.inv(np.matmul(M.T, M)), 
                            np.matmul(M.T, Y))
       
       # Find intercepts
       alpha_hat = np.matmul(sample_per_batch/ float(n_sample), 
                             beta_hat[:n_batch,:])
       self.intercept_ = alpha_hat
       # Find slopes for covariates/effects
       beta_x = beta_hat[n_batch:end_x, :]
       self.beta_x_ = beta_x
       
       beta_c = beta_hat[end_x:end_c, :]
       self.beta_c_ = beta_c
       
       
       Y_hat =  np.matmul(M, beta_hat)
       sigma = np.mean(((Y - Y_hat)**2), axis=0)
       self.epsilon_ = sigma 
       
       # Standarise data
       Z = Y.copy()
       Z -= alpha_hat[np.newaxis,:]
       Z -= np.matmul(M[:, n_batch:end_x], beta_x) 
       Z -= np.matmul(M[:, end_x:end_c], beta_c)
       Z/=np.sqrt(sigma)    
       
       #Find gamma fitted to standard data 
       gam_hat = np.matmul(np.linalg.inv(np.matmul(B.T, B)), 
                             np.matmul(B.T, Z)
                             )
       # Mean across genes
       gam_bar = np.mean(gam_hat, axis=1) 
       # Variance across genes
       tau_bar_sq = np.var(gam_hat, axis=1, ddof=1)
       
       # Variance per batch and gen
       del_hat_sq = [np.var(Z[B[:, ii]==1,:], axis=0, ddof=1)\
                    for ii in range(B.shape[1])]    
       del_hat_sq = np.array(del_hat_sq)

       lam_bar, the_bar = _inverse_gamma_moments(del_hat_sq)
           
       if self.mode == 'p':
           it_eb = _it_eb_param
       else:
           it_eb = _it_eb_non_param
           
        
       gam_star, del_sq_star = [], []
       for ii in range(B.shape[1]):
           g, d_sq = it_eb(Z[B[:, ii]==1,:], 
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
       self.delta2_ = del_sq_star
       
       return self
   
    def transform(self, Y, b, X = None, C = None):
        
        Y, b, X, C = self._validate_for_transform(Y, b, X, C)

        B = pd.get_dummies(b).values
        n_batch = B.shape[1]
        
        #First standarise again the data
        Y_trans = Y - self.intercept_[np.newaxis,:]
        
        if self.beta_x_.size > 0:
            Y_trans -= np.matmul(X, self.beta_x_)
            
        if self.beta_c_.size > 0:
            Y_trans -= np.matmul(C, self.beta_c_)
            
        Y_trans/= np.sqrt(self.epsilon_)    
        
        for ii in range(n_batch):
            batch_idxs = (B[:, ii]==1)
            Y_trans[batch_idxs,:] -= self.gamma_[ii, :][np.newaxis, :]
            Y_trans[batch_idxs,:] /= np.sqrt(self.delta2_[ii, :]) 

        
        Y_trans *= np.sqrt(self.epsilon_)
        Y_trans += self.intercept_[np.newaxis, :] 
        
        if self.beta_x_.size > 0:
            Y_trans += np.matmul(X, self.beta_x_)
        
        return Y_trans
    
    
    def fit_transform(self, Y, b, X = None, C = None):
        return self.fit(Y, b, X, C).transform(Y, b, X, C)
    
    def _validate_for_transform(self, Y, b, X, C):
        
        # check if fitted 
        attributes = ['intercept_', 'beta_x_', 
                      'beta_c_', 'epsilon_',
                      'gamma_', 'delta2_']
        
        attrs = all([hasattr(self, attr) for attr in attributes])
        if not attrs:
            raise AttributeError("Combat was not fitted?")
    
        if Y.shape[1] != len(self.intercept_):
            raise ValueError("Wrong number of features for Y")
        if len(np.unique(b)) != self.gamma_.shape[0]:
            raise ValueError("Wrong number of categories for b")
        if isinstance(X, np.ndarray):
            if X.shape[1] != self.beta_x_.shape[0]:
                raise ValueError("Dimensions of fitted beta "
                                 "and input X matrix do not match")        
        if isinstance(C, np.ndarray):
            if C.shape[1] != self.beta_c_.shape[0]:
                raise ValueError("Dimensions of fitted beta "
                                 "and input C matrix do not match")
                    
        return Y, b, X, C
        
        
         