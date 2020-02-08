import pandas as pd
import numpy as np
from scipy import optimize


def sub_stamp(stamp, dim=15, flatten=False):
    if stamp.ndim == 1:
        # original dimension of the thumbnail
        original_dim = int(np.sqrt(stamp.shape[0]))
        # reshape the original thumbnail from 1D to 2D
        stamp = stamp.reshape((original_dim, original_dim))
    else:
        # original dimension of the thumbnail
        original_dim = int(stamp.shape[0])

    # central pixel index of the original thumbnail
    original_c_idx = original_dim//2
    # new central pixel index after cutting out with given dimension
    c_idx = dim//2
    # cut out thumbnail
    substamps = stamp[original_c_idx-c_idx:original_c_idx+c_idx+1, original_c_idx-c_idx:original_c_idx+c_idx+1]
    if flatten:
        return substamps.ravel()
    else:
        return substamps


def gaussian(amp, sig_x, sig_y, ang):
    """
    Returns a 2Dgaussian function with the given parameters
    
    Parameters
    ----------
    amp: float
        Amplitude of the 2D Gaussian function.
    sig_x: float
        Sigma along x-direction.
    sig_y: float
        Sigama along y-direction.
    ang: float
        Inclination angle in the unit of radians.
        
    Return
    ------
    out: func
        2D Gaussian function taking x,y as an input.
    """
    dim = 15
    c_idx = dim//2
    a = 0.5*(np.cos(ang)/sig_x)**2 + 0.5*(np.sin(ang)/sig_y)**2
    b = -np.sin(2*ang)/(4*sig_x**2) + np.sin(2*ang)/(4*sig_y**2)
    c = 0.5*(np.sin(ang)/sig_x)**2 + 0.5*(np.cos(ang)/sig_y)**2
    return lambda x,y: amp*np.exp(-(a*(x-c_idx)**2+2*b*(x-c_idx)*(y-c_idx)+c*(y-c_idx)**2))

def moments(stamp, dim=15):
    """
    Returns initial guess of (amp, sig_x, sig_y, ang), 
    the gaussian parameters of a 2D distribution by calculating its
    moments 
    
    Parameters
    ----------
    data: np.array
        Image matrix.
    
    """
    # cut out sub stamp
    substamp = sub_stamp(stamp, dim=dim, flatten=False)
    # new central pixel index after cutting out with given dimension
    c_idx = dim//2
    
    total = substamp.sum()
    X, Y = np.indices(substamp.shape)
    col = substamp[:, int(c_idx)]
    sig_x = np.sqrt(np.abs((np.arange(col.size)-c_idx)**2*col).sum()/col.sum())
    row = substamp[int(c_idx), :]
    sig_y = np.sqrt(np.abs((np.arange(row.size)-c_idx)**2*row).sum()/row.sum())
    amp = sub_stamp(substamp.ravel(), dim=3).max()
    ang = 0
    return amp, sig_x, sig_y, ang

def fitgaussian(stamp, dim=15):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    substamp = sub_stamp(stamp, dim=dim)
    params = moments(stamp, dim=dim)
    errorfunction = lambda p: np.ravel(gaussian(*p)(*np.indices(substamp.shape)) -
                                 substamp)
    p, success = optimize.leastsq(errorfunction, params)
    p = np.nan_to_num(p)

    if success == 1:
        Xin, Yin = np.mgrid[0:dim, 0:dim]
        fit_residual = substamp - gaussian(*p)(Xin, Yin)
        fit_Rsquared = 1 - np.var(fit_residual)/np.var(substamp)
    else:
        fit_Rsquared = 0
    return p, fit_Rsquared

def chunk_fit(ind, stamp_chuck, amp_dict, r_dict):
    gauss_amp, gauss_R = [], []
    for s in stamp_chuck:
        try:
            p, r = fitgaussian(s, dim=15)
            gauss_amp.append(p[0])
            gauss_R.append(r)
        except RuntimeError:
            gauss_amp.append(0)
            gauss_R.append(0)
    amp_dict[ind] = gauss_amp
    r_dict[ind] = gauss_R

    
    


