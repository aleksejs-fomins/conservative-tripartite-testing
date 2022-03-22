import numpy as np

from mesostat.utils.signals.resample import bin_data_1D
from mesostat.metric.dim3d.r2 import pr2_quadratic_triplet_decomp_1D
from mesostat.metric.dim3d.idtxl_pid import bivariate_pid_3D

from lib.nullmodels.null_test import run_1D_scan_bare
import lib.nullmodels.null_models_3D as null3D


def _bin_data_3D(x,y,z, nBins=None):
    if nBins is None:
        return np.array([x, y, z])
    else:
        return np.array([
            bin_data_1D(x, nBins),
            bin_data_1D(y, nBins),
            bin_data_1D(z, nBins)
        ])


def metric_bin(x, y, z, metricName, nBins=None):
    if metricName == 'PID':
        dataPR = _bin_data_3D(x,y,z, nBins=None)
        dataRPS = dataPR.T[..., None]
        print(dataRPS.shape)

        settings_estimator = {'pid_estimator': 'TartuPID', 'lags_pid': [0, 0]}
        return bivariate_pid_3D(dataRPS, settings={'src': [0, 1], 'trg': 2, 'settings_estimator': settings_estimator})
    elif metricName == 'PR2':
        assert nBins is None
        return pr2_quadratic_triplet_decomp_1D(x, y, z)
    else:
        raise ValueError('Unexpected metric', metricName)


def metric_adversarial_distribution(nData, pidType, nBins=None):
    if pidType == 'red':
        f_data_1D = lambda nSample, alpha: null3D.cont_unq_noisy(nSample, alpha, alpha, alpha)
    else:
        f_data_1D = lambda nSample, alpha: null3D.cont_red_noisy(nSample, alpha, alpha, alpha)

    f_metric_cont = lambda x, y, z: metric_bin(x, y, z, nBins)
    return run_1D_scan_bare(f_data_1D, f_metric_cont, pidType,
                            varLimits=(0, 1), nData=nData, nStep=100, nTest=100, nTestResample=10000)[1]
