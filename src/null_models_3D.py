import numpy as np


def bernoulli(n, p):
    return (np.random.uniform(0, 1, n) < p).astype(int)


#########################################
# Discrete models
#########################################

def _add_discr_noise(n, x, y, z, pX=0.5, pY=0.5, pZ=0.5):
    aX = bernoulli(n, pX)
    aY = bernoulli(n, pY)
    aZ = bernoulli(n, pZ)
    xNew = (1 - aX) * x + aX * bernoulli(n, 0.5)
    yNew = (1 - aY) * y + aY * bernoulli(n, 0.5)
    zNew = (1 - aZ) * z + aZ * bernoulli(n, 0.5)
    return xNew, yNew, zNew


def discr_red_noisy(nSample, pX=0.5, pY=0.5, pZ=0.5):
    t = bernoulli(nSample, 0.5)
    return _add_discr_noise(nSample, t, t, t, pX=pX, pY=pY, pZ=pZ)


def discr_unq_noisy(nSample, pX=0.5, pY=0.5, pZ=0.5):
    tx = bernoulli(nSample, 0.5)
    ty = bernoulli(nSample, 0.5)
    return _add_discr_noise(nSample, tx, ty, tx, pX=pX, pY=pY, pZ=pZ)


def discr_syn_noisy(nSample, pX=0.5, pY=0.5, pZ=0.5):
    x = bernoulli(nSample, 0.5)
    y = bernoulli(nSample, 0.5)
    z = np.logical_xor(x, y)
    return _add_discr_noise(nSample, x, y, z, pX=pX, pY=pY, pZ=pZ)


def discr_method_dict():
    return {
        'discr_red': discr_red_noisy,
        'discr_unq': discr_unq_noisy,
        'discr_syn': discr_syn_noisy,
    }


#########################################
# Continuous models
#########################################

def _add_cont_noise(n, x, y, z, pX=0.5, pY=0.5, pZ=0.5):
    xNew = (1 - pX) * x + pX * np.random.normal(0, 1, n)
    yNew = (1 - pY) * y + pY * np.random.normal(0, 1, n)
    zNew = (1 - pZ) * z + pZ * np.random.normal(0, 1, n)
    return xNew, yNew, zNew


def cont_red_noisy(n=1000, pX=0.5, pY=0.5, pZ=0.5):
    t = np.random.normal(0, 1, n)
    return _add_cont_noise(n, t, t, t, pX=pX, pY=pY, pZ=pZ)


def cont_unq_noisy(n=1000, pX=0.5, pY=0.5, pZ=0.5):
    tx = np.random.normal(0, 1, n)
    ty = np.random.normal(0, 1, n)
    return _add_cont_noise(n, tx, ty, tx, pX=pX, pY=pY, pZ=pZ)


# NOTE: Basic multiplication has redundancy, need to do continuous XOR:
#  where magnitudes are uncorrelated, but signs are XOR
def cont_xor_noisy(n=1000, pX=0.5, pY=0.5, pZ=0.5):
    x0 = np.random.normal(0, 1, n)
    y0 = np.random.normal(0, 1, n)
    z0 = np.abs(np.random.normal(0, 1, n))
    z0 *= np.sign(x0) * np.sign(y0)
    return _add_cont_noise(n, x0, y0, z0, pX=pX, pY=pY, pZ=pZ)


def cont_sum_noisy(n=1000, pX=0.5, pY=0.5, pZ=0.5):
    x0 = np.random.normal(0, 1, n)
    y0 = np.random.normal(0, 1, n)
    return _add_cont_noise(n, x0, y0, x0+y0, pX=pX, pY=pY, pZ=pZ)


def cont_method_dict():
    return {
        'cont_red': cont_red_noisy,
        'cont_unq': cont_unq_noisy,
        'cont_syn': cont_xor_noisy,
        'cont_sum': cont_sum_noisy,
    }
