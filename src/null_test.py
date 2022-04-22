from copy import copy
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
import matplotlib.pyplot as plt
# from collections import defaultdict

from mesostat.utils.arrays import unique_ordered
from mesostat.utils.pandas_helper import merge_df_from_dict
# from mesostat.stat.testing.htests import tolerance_interval
from mesostat.visualization.mpl_axis_scale import nonlinear_xaxis
from mesostat.visualization.mpl_1D import prettify_plot_1D


# from mesostat.visualization.mpl_matrix import imshow


##############################
# Auxiliary Functions
##############################

def shuffle(x):
    x1 = x.copy()
    np.random.shuffle(x1)
    return x1


def fraction_significant(df, dfRand, pVal, valThrDict=None):
    sigDict = {}
    for method in unique_ordered(df['Method']):
        dataTrueMethod = df[df['Method'] == method]
        dataRandMethod = dfRand[dfRand['Method'] == method]

        # print(method,
        #       np.min(dataTrueMethod['Value']),
        #       np.max(dataTrueMethod['Value']),
        #       np.min(dataRandMethod['Value']),
        #       np.max(dataRandMethod['Value'])
        #       )

        # Compute threshold based on shuffled data
        thr = np.quantile(dataRandMethod['Value'], 1 - pVal)

        # If available, also apply constant threshold to data magnitude
        # Choose bigger of the two thresholds
        if valThrDict is not None and valThrDict[method] is not None:
            thr = max(thr, valThrDict[method])

        sigDict[method] = [np.mean(dataTrueMethod['Value'] - thr > 1.0E-6)]

    return sigDict


def effect_size_by_method(df, dfRand):
    dfEffSize = pd.DataFrame()
    for method in unique_ordered(df['Method']):
        dfMethodTrue = df[df['Method'] == method]
        dfMethodRand = dfRand[dfRand['Method'] == method]

        muRand = np.mean(dfMethodRand['Value'])
        stdRand = np.std(dfMethodRand['Value'])

        dfMethodEff = dfMethodTrue.copy()
        dfMethodEff['Value'] = (dfMethodEff['Value'] - muRand) / stdRand

        dfEffSize = dfEffSize.append(dfMethodEff)
    return dfEffSize


##############################
# Functions
##############################


def sample_decomp(datagen_func_noparam, decomp_func, trgLabel, nData=10000, nSample=10000, haveShuffle=False):
    rezLst = []
    for i in range(nSample):
        x, y, z = datagen_func_noparam(nData)
        if haveShuffle:
            z = shuffle(z)
        rez = decomp_func(x, y, z)
        rezLst += [rez[trgLabel]]
    return rezLst


def run_tests(datagen_func, decomp_func, decompLabels, nTest=100, haveShuffle=False):
    rezDict = {k: [] for k in decompLabels}
    for iTest in range(nTest):
        x, y, z = datagen_func()
        zEff = z if not haveShuffle else shuffle(z)

        rez = decomp_func(x, y, zEff)

        for k in rezDict.keys():
            rezDict[k] += [rez[k]]

    rezDF = pd.DataFrame()

    for iLabel, label in enumerate(decompLabels):
        rezTmp = pd.DataFrame({'Method': [label] * nTest, 'Value': rezDict[label]})
        rezDF = rezDF.append(rezTmp)

    return rezDF


def plot_test_summary(df, dfRand, suptitle=None, haveEff=True, logTrue=True, logEff=False, valThrDict=None):
    nFig = 3 if haveEff else 2
    fig, ax = plt.subplots(ncols=nFig, figsize=(4*nFig, 4), tight_layout=True)
    if suptitle is not None:
        fig.suptitle(suptitle)

    # # Clip data
    # df['Value'] = np.clip(df['Value'], 1.0E-6, None)
    # dfRand['Value'] = np.clip(dfRand['Value'], 1.0E-6, None)

    # Plot 1: True vs Random
    dfMerged = merge_df_from_dict({'True': df, 'Random': dfRand}, columnNames=['Kind'])
    sns.violinplot(ax=ax[0], x="Method", y="Value", hue="Kind", data=dfMerged, scale='width', cut=0)
    if logTrue:
        ax[0].set_yscale('log')
    ax[0].set_xlabel('')
    ax[0].set_ylabel('Metric Value')

    # Calculate effect sizes
    dfEffSize = effect_size_by_method(df, dfRand)

    # Plot 2: Effect Sizes
    if haveEff:
        sns.violinplot(ax=ax[1], x="Method", y="Value", data=dfEffSize, scale='width', cut=0)
        if logEff:
            ax[1].set_yscale('log')
        # ax[1].axhline(y='2', color='pink', linestyle='--')
        ax[1].set_xlabel('')
        ax[1].set_ylabel('Effect Size')

    # Calculate fraction significant
    sigDict = fraction_significant(df, dfRand, 0.01, valThrDict=valThrDict)

    # Plot 3: Fraction significant
    idx3 = 2 if haveEff else 1
    sns.barplot(ax=ax[idx3], data=pd.DataFrame(sigDict))
    ax[idx3].set_ylim([0, 1])
    ax[idx3].set_xlabel('')
    ax[idx3].set_ylabel('Fraction Significant')


def _stratify_range(x, eta=1):
    return (1 - np.exp(eta*x)) / (1 - np.exp(eta))


def run_plot_param_effect(datagen_func, decomp_func, decompLabels,
                          nStep=100, nSkipTest=10, nTest=10000, pVal=0.01, alphaRange=(0, 1), valThr=1.0E-7,
                          avgRand=False, thrMetricDict=None, plotAlphaSq=False, fontsize=20):

    alphaLst = np.linspace(*alphaRange, nStep)
    # alphaLst = _stratify_range(alphaLst, eta=2)
    # alphaLst = alphaRange[0] + (alphaRange[1] - alphaRange[0]) * alphaLst
    alphaTestLst = []

    rezTrueLst = []
    rezRandLst = []
    fracSignShuffleLst = []
    fracSignAdjustedLst = []
    for iStep in range(nStep):
        # alpha = np.random.uniform(*alphaRange)
        x, y, z = datagen_func(alphaLst[iStep])
        rezTrue = decomp_func(x, y, z)
        rezTrue = [rezTrue[k] for k in decompLabels]
        rezTrueLst += [copy(rezTrue)]

        if (iStep < 7) or (iStep % nSkipTest) == 0:
            rezRandTmpLst = []
            rezTrueTmpLst = []
            for iTest in range(nTest):
                x, y, z = datagen_func(alphaLst[iStep])
                rezTrue = decomp_func(x, y, z)
                rezRand = decomp_func(x, y, shuffle(z))
                rezTrueTmpLst += [[rezTrue[k] for k in decompLabels]]
                rezRandTmpLst += [[rezRand[l] for l in decompLabels]]

            rezRandTmpLst = np.quantile(rezRandTmpLst, 1 - pVal, axis=0)
            rezTrueTmpLst = np.array(rezTrueTmpLst)

            alphaTestLst += [alphaLst[iStep]]
            rezRandLst += [rezRandTmpLst]

            fracSignShuffleLst += [[np.mean(rezTrueTmpLst[:, i] > rezRandTmpLst[i]) for i in range(len(decompLabels))]]

            if thrMetricDict is not None:
                tmpLst = []
                for iKind, kindName in enumerate(decompLabels):
                    if thrMetricDict[kindName] is None:
                        tmpLst += [None]
                    else:
                        tmpLst += [np.mean(rezTrueTmpLst[:, iKind] > thrMetricDict[kindName])]
                fracSignAdjustedLst += [tmpLst]

    rezTrueLst = np.clip(np.array(rezTrueLst), valThr, None)
    rezRandLst = np.clip(np.array(rezRandLst), valThr, None)
    fracSignShuffleLst = np.array(fracSignShuffleLst)
    fracSignAdjustedLst = np.array(fracSignAdjustedLst)

    alphaLstPlot     = alphaLst     if not plotAlphaSq else alphaLst ** 2
    alphaTestLstPlot = alphaTestLst if not plotAlphaSq else np.array(alphaTestLst) ** 2

    nMethods = len(decompLabels)
    fig, ax = plt.subplots(nrows=2, ncols=nMethods, figsize=(4*nMethods, 8), tight_layout=True)
    for iKind, kindLabel in enumerate(decompLabels):
        # Metric Values
        ax[0, iKind].set_title(kindLabel)
        ax[0, iKind].semilogy(alphaLstPlot, rezTrueLst[:, iKind], '.', label='Data', color='black')
        if avgRand:
            ax[0, iKind].axhline(y=np.mean(rezRandLst[:, iKind]), label='thrShuffle', color='red')
        else:
            ax[0, iKind].semilogy(alphaTestLstPlot, rezRandLst[:, iKind], label='thrShuffle', color='red')
        if (thrMetricDict is not None) and (thrMetricDict[kindLabel] is not None):
            ax[0, iKind].axhline(y=thrMetricDict[kindLabel], label='thrAdjusted', color='purple')
        ax[0, iKind].set_ylim([valThr / 2, 10])
        ax[0, iKind].legend()

        # Fraction of significant items
        ax[1, iKind].plot(alphaTestLstPlot, fracSignShuffleLst[:, iKind], label='shuffle-test', color='red')
        if (thrMetricDict is not None) and (thrMetricDict[kindLabel] is not None):
            ax[1, iKind].plot(alphaTestLstPlot, fracSignAdjustedLst[:, iKind], label='adjusted-test', color='purple')
        ax[1, iKind].set_ylim([-0.1, 1.1])
        ax[1, iKind].legend()

        # Set nonlinearity to axes
        nonlinear_xaxis(ax[0, iKind], scale=0.001)
        nonlinear_xaxis(ax[1, iKind], scale=0.001)

        xTicksNew = [0, 0.001, 0.01, 0.1, 0.5, 1]
        prettify_plot_1D(ax[0, iKind], haveTopRightBox=False, margins=0.05, xTicks=xTicksNew, xRotation=90,
                         xFontSize=fontsize, yFontSize=fontsize)
        prettify_plot_1D(ax[1, iKind], haveTopRightBox=False, margins=0.05, xTicks=xTicksNew, xRotation=90,
                         xFontSize=fontsize, yFontSize=fontsize)


    ax[0, 0].set_ylabel('Metric Value')
    ax[1, 0].set_ylabel('Fraction Significant')


def run_plot_param_effect_test(datagen_func, decomp_func, decompLabels,
                               nStep=10, nTest=1000, alphaRange=(0, 1), valThrDict=None, fontsize=10):
    # alphaLst = np.linspace(*alphaRange, nStep)
    alphaLst = np.linspace(0, 1, nStep)
    alphaLst = _stratify_range(alphaLst, eta=5)
    alphaLst = alphaRange[0] + (alphaRange[1] - alphaRange[0]) * alphaLst

    dfTrueDict = {}
    dfRandDict = {}
    dfEffDict = {}
    for alpha in alphaLst:
        gen_data_eff = lambda: datagen_func(alpha)

        rezDF   = run_tests(gen_data_eff, decomp_func, decompLabels, nTest=nTest)
        rezDFsh = run_tests(gen_data_eff, decomp_func, decompLabels, nTest=nTest, haveShuffle=True)
        dfEffSize = effect_size_by_method(rezDF, rezDFsh)

        dfTrueDict[(np.round(alpha, 2), )] = rezDF
        dfRandDict[(np.round(alpha, 2), )] = rezDFsh
        dfEffDict[(np.round(alpha, 2), )] = dfEffSize

    dfRez = merge_df_from_dict(dfEffDict, ['alpha'])

    nMethods = len(decompLabels)
    fig, ax = plt.subplots(nrows=2, ncols=nMethods, figsize=(4*nMethods, 8), tight_layout=True)

    for iMethod, methodName in enumerate(decompLabels):
        # Compute plot effect sizes
        dfRezMethod = dfRez[dfRez['Method'] == methodName]

        sns.violinplot(ax=ax[0, iMethod], x="alpha", y="Value", data=dfRezMethod, scale='width', color='lightgray', label=methodName)
        ax[0, iMethod].set_xticklabels(ax[0, iMethod].get_xticklabels(), rotation = 90)
        ax[0, iMethod].set_xlabel('')
        ax[0, iMethod].set_title(methodName)

        # # Compute plot thresholded effect sizes
        # sigDict = {}
        # for alpha, dfTrue in dfTrueDict.items():
        #     dfRand = dfRandDict[alpha]
        #     dfTrueMethod = dfTrue[dfTrue['Method'] == methodName]
        #     dfRandMethod = dfRand[dfRand['Method'] == methodName]
        #
        #     thr = np.quantile(dfRandMethod['Value'], 0.99)
        #     sigDict[alpha[0]] = [np.mean(dfTrueMethod['Value'] > thr)]

        # Compute plot thresholded effect sizes
        sigDict = {}
        for alpha, dfTrue in dfTrueDict.items():
            dfRand = dfRandDict[alpha]
            sigDict[alpha[0]] = fraction_significant(dfTrue, dfRand, 0.01, valThrDict=valThrDict)[methodName]

        valDF = pd.DataFrame(sigDict)
        sns.barplot(ax=ax[1, iMethod], data=valDF, color='lightgray')
        ax[1, iMethod].set_xticklabels(ax[1, iMethod].get_xticklabels(), rotation=90)
        ax[1, iMethod].set_ylim(0, 1.05)
        ax[0, iMethod].set_xlabel('$\sigma$')

    ax[0, 0].set_ylabel('Effect Size')
    ax[1, 0].set_ylabel('Fraction Significant')


def run_plot_param_effect_test_single(datagen_func, decomp_func, decompLabels, alpha, nTest=1000, valThrDict=None):
    gen_data_eff = lambda: datagen_func(alpha)

    rezDF   = run_tests(gen_data_eff, decomp_func, decompLabels, nTest=nTest)
    rezDFsh = run_tests(gen_data_eff, decomp_func, decompLabels, nTest=nTest, haveShuffle=True)
    dfEffSize = effect_size_by_method(rezDF, rezDFsh)

    print(fraction_significant(rezDF, rezDFsh, 0.01, valThrDict=valThrDict))

    fig, ax = plt.subplots(ncols=3, figsize=(12,4))
    sns.violinplot(ax=ax[0], data=rezDF, x="Method", y="Value", scale='width', cut=0)
    sns.violinplot(ax=ax[1], data=rezDFsh, x="Method", y="Value", scale='width', cut=0)
    sns.violinplot(ax=ax[2], data=dfEffSize, x="Method", y="Value", scale='width', cut=0)
    ax[0].set_yscale('log')
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')
    # ax[0].set_ylim([1.0E-7, 10])
    # ax[1].set_ylim([1.0E-7, 10])
    plt.show()


def run_plot_data_effect(datagen_func, decomp_func, decompLabels,
                          nStep=100, nSkipTest=10, nTest=200, pVal=0.01,
                          thrMetricDict=None, fontsize=10):
    # iSkipStep = 0
    nDataLst = (10 ** np.linspace(2, 4, nStep)).astype(int)
    nDataTestLst = []

    rezTrueLst = []
    rezRandLst = []
    fracSignShuffleLst = []
    fracSignAdjustedLst = []
    for iStep, nData in enumerate(nDataLst):
        # alpha = np.random.uniform(*alphaRange)
        x, y, z = datagen_func(nData)
        rezTrue = decomp_func(x, y, z)
        rezTrue = [rezTrue[k] for k in decompLabels]
        rezTrueLst += [copy(rezTrue)]

        if iStep % nSkipTest == 0:
            rezRandTmpLst = []
            rezTrueTmpLst = []
            for iTest in range(nTest):
                x, y, z = datagen_func(nData)
                rezTrue = decomp_func(x, y, z)
                rezRand = decomp_func(x, y, shuffle(z))
                rezTrueTmpLst += [[rezTrue[k] for k in decompLabels]]
                rezRandTmpLst += [[rezRand[l] for l in decompLabels]]

            rezRandTmpLst = np.quantile(rezRandTmpLst, 1 - pVal, axis=0)
            rezTrueTmpLst = np.array(rezTrueTmpLst)

            nDataTestLst += [nData]
            rezRandLst += [rezRandTmpLst]

            fracSignShuffleLst += [[np.mean(rezTrueTmpLst[:, i] > rezRandTmpLst[i]) for i in range(len(decompLabels))]]

            if thrMetricDict is not None:
                tmpLst = []
                for iKind, kindName in enumerate(decompLabels):
                    if thrMetricDict[kindName] is None:
                        tmpLst += [None]
                    else:
                        if isinstance(thrMetricDict[kindName], dict):
                            tmpLst += [np.mean(rezTrueTmpLst[:, iKind] > np.mean(list(thrMetricDict[kindName].values())))]
                        else:
                            tmpLst += [np.mean(rezTrueTmpLst[:, iKind] > thrMetricDict[kindName])]
                fracSignAdjustedLst += [tmpLst]

            # if thrMetricDict is not None:
            #     tmpLst = []
            #     for iKind, kindName in enumerate(decompLabels):
            #         if thrMetricDict[kindName] is None:
            #             tmpLst += [None]
            #         else:
            #             tmpLst += [np.mean(rezTrueTmpLst[:, iKind] > thrMetricDict[kindName][iSkipStep])]
            #     fracSignAdjustedLst += [tmpLst]
            #
            # iSkipStep += 1


    rezTrueLst = np.array(rezTrueLst)
    rezRandLst = np.array(rezRandLst)
    fracSignShuffleLst = np.array(fracSignShuffleLst)
    fracSignAdjustedLst = np.array(fracSignAdjustedLst)

    nMethods = len(decompLabels)
    fig, ax = plt.subplots(nrows=2, ncols=nMethods, figsize=(4*nMethods, 8), tight_layout=True)
    for iKind, kindLabel in enumerate(decompLabels):
        ax[0, iKind].set_title(kindLabel)
        ax[0, iKind].loglog(nDataLst, rezTrueLst[:, iKind], '.', label='Data', color='black')
        ax[0, iKind].loglog(nDataTestLst, rezRandLst[:, iKind], label='thrShuffle', color='red')
        if (thrMetricDict is not None) and (thrMetricDict[kindLabel] is not None):
            if isinstance(thrMetricDict[kindLabel], dict):
                ax[0, iKind].loglog(list(thrMetricDict[kindLabel].keys()), list(thrMetricDict[kindLabel].values()), label='thrAdjusted', color='purple')
            else:
                ax[0, iKind].axhline(y = thrMetricDict[kindLabel], label='thrAdjusted', color='purple')

        ax[0, iKind].set_ylim([1.0E-7, 10])
        ax[0, iKind].legend()

        ax[1, iKind].semilogx(nDataTestLst, fracSignShuffleLst[:, iKind], label='shuffle-test', color='red')
        if (thrMetricDict is not None) and (thrMetricDict[kindLabel] is not None):
            ax[1, iKind].semilogx(nDataTestLst, fracSignAdjustedLst[:, iKind], label='adjusted-test', color='purple')
        ax[1, iKind].set_ylim([-0.1, 1.1])
        ax[1, iKind].legend()

        prettify_plot_1D(ax[0, iKind], haveTopRightBox=False, margins=0.05, xFontSize=fontsize, yFontSize=fontsize)
        prettify_plot_1D(ax[1, iKind], haveTopRightBox=False, margins=0.05, xFontSize=fontsize, yFontSize=fontsize)

    ax[0, 0].set_ylabel('Metric Value')
    ax[1, 0].set_ylabel('Fraction Significant')


def run_plot_data_effect_test(datagen_func, decomp_func, decompLabels, nStep=10, nTest=1000, valThrDict=None):
    nDataLst = (10 ** np.linspace(2, 5, nStep)).astype(int)

    dfTrueDict = {}
    dfRandDict = {}
    dfEffDict = {}
    for nData in nDataLst:
        gen_data_eff = lambda: datagen_func(nData)

        rezDF   = run_tests(gen_data_eff, decomp_func, decompLabels, nTest=nTest)
        rezDFsh = run_tests(gen_data_eff, decomp_func, decompLabels, nTest=nTest, haveShuffle=True)
        dfEffSize = effect_size_by_method(rezDF, rezDFsh)

        dfTrueDict[(nData, )] = rezDF
        dfRandDict[(nData, )] = rezDFsh
        dfEffDict[(nData, )] = dfEffSize


    dfRez = merge_df_from_dict(dfEffDict, ['nData'])

    nMethods = len(decompLabels)
    fig, ax = plt.subplots(nrows=2, ncols=nMethods, figsize=(4*nMethods, 8), tight_layout=True)
    for iMethod, methodName in enumerate(decompLabels):
        ax[0, iMethod].set_title(methodName)

        # Compute plot effect sizes
        dfRezMethod = dfRez[dfRez['Method'] == methodName]
        sns.violinplot(ax=ax[0, iMethod], x="nData", y="Value", data=dfRezMethod, scale='width', color='lightgray')
        ax[0, iMethod].set_xticklabels(ax[0, iMethod].get_xticklabels(), rotation = 90)
        ax[0, iMethod].set_xlabel('')

        # Compute plot thresholded effect sizes
        sigDict = {}
        for nDataTuple, dfTrue in dfTrueDict.items():
            dfRand = dfRandDict[nDataTuple]
            sigDict[nDataTuple[0]] = fraction_significant(dfTrue, dfRand, 0.01, valThrDict=valThrDict)[methodName]

            # dfTrueMethod = dfTrue[dfTrue['Method'] == methodName]
            # dfRandMethod = dfRand[dfRand['Method'] == methodName]
            #
            # thr = np.quantile(dfRandMethod['Value'], 0.99)
            # sigDict[nDataTuple[0]] = [np.mean(dfTrueMethod['Value'] > thr)]

        valDF = pd.DataFrame(sigDict)
        sns.barplot(ax=ax[1, iMethod], data=valDF, color='lightgray')
        ax[1, iMethod].set_xticklabels(ax[1, iMethod].get_xticklabels(), rotation=90)
        ax[1, iMethod].set_ylim(0, 1.05)
        ax[0, iMethod].set_xlabel('$\sigma$')

    ax[0, 0].set_ylabel('Effect Size')
    ax[1, 0].set_ylabel('Fraction Significant')


##############################
# Max-Synergy-Parameter Search
##############################

def _in_limits(x, varLim):
    return np.all(x >= varLim[0]) and np.all(x <= varLim[1]) and (x[1] <= x[0])


# Resampling procedure working on a grid of arbitrary dimension
# Discretizes each dimension into nStep steps, samples nTest samples over outer product
# Returns sampled coordinates and all results requested in the atomNames, shape (nGridPoint, nTest, nAtom)
# Use cases:
#   * natural 1D, 2D, 3D over noise parameters
#   * Transformed 2D (e.g. noiseS1 = noiseS2, but noiseTrg separately)
def run_scan_bare(datagen_func, decomp_func, nVars, atomNames, varLimits=(0, 1),
                  nData=1000, nStep=100, nTest=20):
    rezLst = []

    x1 = np.linspace(*varLimits, nStep)
    paramProdLst = list(itertools.product(*[x1] * nVars))

    for vars in paramProdLst:
        rezLstTmp = []
        for iTest in range(nTest):
            x, y, z = datagen_func(nData, *vars)
            rez = decomp_func(x, y, z)
            rezLstTmp += [[rez[k] for k in atomNames]]

        rezLst += [rezLstTmp]
    return np.array(paramProdLst), np.array(rezLst)


# Sample nTestResample from a model given noise parameter values `param`
def resample_model(datagen_func, decomp_func, atomLabel, param, nData=1000, nTestResample=1000, haveShuffle=False):
    datagen_func_noparam = lambda nData: datagen_func(nData, *param)
    return sample_decomp(datagen_func_noparam, decomp_func, atomLabel,
                         nData=nData, nSample=nTestResample, haveShuffle=haveShuffle)


def resample_get_thr(datagen_func, decomp_func, atomLabel, atomLabels, paramArr, dataArr,
                     nData=1000, nTestResample=1000, pVal=0.01, haveShuffle=False):
    atomIdx = atomLabels.index(atomLabel)
    rezMu = np.mean(dataArr[:, :, atomIdx], axis=1)
    param = paramArr[np.argmax(rezMu)]

    atomDistr = resample_model(datagen_func, decomp_func, atomLabels[atomIdx], param,
                               nData=nData, nTestResample=nTestResample, haveShuffle=haveShuffle)
    return np.quantile(atomDistr, 1-pVal)


# Print top 10 maximal points
def print_scan_max(paramArr, dataArr, atomLabel, atomLabels, nMax=10):
    atomIdx = atomLabels.index(atomLabel)
    dataAtom1D = np.mean(dataArr[:,:,atomIdx], axis=1)
    idxsMax = np.argsort(dataAtom1D)[-nMax:][::-1]

    rezArr = np.append(paramArr[idxsMax], dataAtom1D[idxsMax, None], axis=1)
    print(rezArr)


def plot_scan_1D(paramArr, dataArr, atomLabelsPlot, trgAtomLabel, atomLabels, savename=None, maxThr=None,
                 colorDict=None, fontsize=16, xlabel='Parameter values', ylabel='Function values'):
    plt.rcParams.update({'font.size': fontsize})
    plt.figure()
    for atomLabel in atomLabelsPlot:
        atomIdx = atomLabels.index(atomLabel)

        rezMu = np.mean(dataArr[:,:,atomIdx], axis=1)
        rezStd = np.std(dataArr[:,:,atomIdx], axis=1)

        color = None if colorDict is None else colorDict[atomLabel]

        if atomLabel == trgAtomLabel:
            plt.errorbar(paramArr, rezMu, rezStd, label=atomLabel, color=color)

            paramMax = paramArr[np.argmax(rezMu)]
            plt.axvline(paramMax, color='red', alpha=0.3, linestyle='--')
            if maxThr is not None:
                plt.axhline(maxThr, color='red', alpha=0.3, linestyle='--')
        else:
            plt.errorbar(paramArr, rezMu, rezStd, label=atomLabel, color=color, linestyle='--')

    plt.yscale('log')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if atomLabels is not None:
        plt.legend()
    if savename is not None:
        plt.savefig(savename)


def plot_scan_2D(dataArr, atomLabel, atomLabels, nStep, varLimits, fontsize=16, haveColorbar=False):
    plt.rcParams.update({'font.size': fontsize})

    atomIdx = atomLabels.index(atomLabel)
    dataAtom1D = np.mean(dataArr[:, :, atomIdx], axis=1)

    dataAtom2D = dataAtom1D.reshape((nStep, nStep))
    plt.figure()
    plt.imshow(dataAtom2D)
    if haveColorbar:
        plt.colorbar()


def plot_scan_3D_2D_bytrg(paramArr, dataArr, atomLabel, atomLabels, zIdx, nStep, varLimits, fontsize=16, haveColorbar=False):
    atomIdx = atomLabels.index(atomLabel)
    valZ = np.linspace(*varLimits, nStep)[zIdx]
    mask = paramArr[:, 2] == valZ
    # paramArr2D = paramArr[mask, :2]
    dataArr2D = dataArr[mask]
    plot_scan_2D(dataArr2D, atomLabel, atomLabels, nStep, varLimits, fontsize=fontsize, haveColorbar=haveColorbar)


# def run_1D_scan_bare(datagen_func_1D, decomp_func, atomLabel, varLimits=(0, 1),
#                      nData=1000, nStep=100, nTest=20, nTestResample=1000):
#     rezMuLst = []
#
#     alphaLst = np.linspace(*varLimits, nStep)
#     for alpha in alphaLst:
#         rezTmpLst = []
#         for iTest in range(nTest):
#             x, y, z = datagen_func_1D(nData, alpha)
#             rez = decomp_func(x, y, z)
#
#             rezTmpLst += [rez[atomLabel]]
#
#         rezMuLst += [np.mean(rezTmpLst)]
#
#     # Find and report maximal synergy point
#     iAlphaMax = np.argmax(rezMuLst)
#     alphaMax = alphaLst[iAlphaMax]
#
#     # Find distribution at maximal synergy point
#     atomDistr = []
#     for iTest in range(nTestResample):
#         x, y, z = datagen_func_1D(nData, alphaMax)
#         rez = decomp_func(x, y, z)
#         atomDistr += [rez[atomLabel]]
#
#     return alphaMax, atomDistr


# def run_plot_1D_scan(datagen_func_1D, decomp_func, labelA, labelB, varLimits=(0, 1),
#                      nData=1000, nStep=100, nTest=20, nTestResample=1000,
#                      havePlot=True, colorA=None, colorB=None):
#     rezAMuLst = []
#     rezBMuLst = []
#     rezAStdLst = []
#     rezBStdLst = []
#
#     alphaLst = np.linspace(*varLimits, nStep)
#     for alpha in alphaLst:
#         aTmp = []
#         bTmp = []
#         for iTest in range(nTest):
#             x, y, z = datagen_func_1D(nData, alpha)
#             rez = decomp_func(x, y, z)
#
#             aTmp += [rez[labelA]]
#             bTmp += [rez[labelB]]
#
#         rezAMuLst += [np.mean(aTmp)]
#         rezBMuLst += [np.mean(bTmp)]
#         rezAStdLst += [np.std(aTmp)]
#         rezBStdLst += [np.std(bTmp)]
#
#     # Find and report maximal synergy point
#     iAlphaMax = np.argmax(rezBMuLst)
#     alphaMax = alphaLst[iAlphaMax]
#
#     # Find distribution at maximal synergy point
#     atomDistr = []
#     for iTest in range(nTestResample):
#         x, y, z = datagen_func_1D(nData, alphaMax)
#         rez = decomp_func(x, y, z)
#         atomDistr += [rez[labelB]]
#
#     atomThrMax = np.quantile(atomDistr, 0.99)
#     print('alpha', alphaMax, 'thr', atomThrMax)
#
#     if havePlot:
#         plt.figure()
#         plt.errorbar(alphaLst, rezAMuLst, rezAStdLst, label=labelA, color=colorA)
#         plt.errorbar(alphaLst, rezBMuLst, rezBStdLst, label=labelB, color=colorB)
#         plt.axhline(atomThrMax, color='red', alpha=0.3, linestyle='--')
#         plt.axvline(alphaMax, color='red', alpha=0.3, linestyle='--')
#
#         plt.yscale('log')
#         plt.xlabel('Parameter values')
#         plt.ylabel('Function values')
#         # plt.title('Synergy-Redundancy relationship for noisy redundant model')
#         plt.legend()
#
#     return alphaMax, atomThrMax





##############################
# Relation between two parameters
##############################


def run_plot_scatter_explore(datagen_func, decomp_func, labelA, labelB, nVars, varLimits=(0, 1), nData=1000, nTestDim=10):
    rezALst = []
    rezBLst = []

    sTmp = 0
    sVars = 0

    x1 = np.linspace(*varLimits, nTestDim)
    prodIt = itertools.product(*[x1]*nVars)

    for vars in prodIt:
        # vars = np.random.uniform(*varLimits, nVars)
        x, y, z = datagen_func(nData, *vars)
        rez = decomp_func(x, y, z)

        rezALst += [rez[labelA]]
        rezBLst += [rez[labelB]]

        # if rez[labelA] >= 1:
        #     print(vars)

        if rez[labelB] > sTmp:
            sTmp = rez[labelB]
            sVars = vars

    print('maxSyn', sTmp, sVars)

    plt.figure()
    plt.plot(rezALst, rezBLst, '.')
    plt.xlabel(labelA)
    plt.ylabel(labelB)
    # plt.title('Synergy-Redundancy relationship for noisy redundant model')
    plt.show()


def run_plot_scatter_exact(datagen_func, decomp_func, labelA, labelB, vars, nData=1000, nTest=1000):
    rezALst = []
    rezBLst = []

    for iTest in range(nTest):
        x, y, z = datagen_func(nData, *vars)
        rez = decomp_func(x, y, z)

        rezALst += [rez[labelA]]
        rezBLst += [rez[labelB]]

    plt.figure()
    plt.plot(rezALst, rezBLst, '.')
    plt.xlabel(labelA)
    plt.ylabel(labelB)
    # plt.title('Synergy-Redundancy relationship for noisy redundant model')
    plt.show()

#
# def run_plot_2D_scan(datagen_func, decomp_func, labelA, labelB, varLimits=(0, 1), nData=1000, nStep=10, nTest=20):
#     rezAMat = np.zeros((nStep, nStep))
#     rezBMat = np.zeros((nStep, nStep))
#
#     alphaLst = np.linspace(*varLimits, nStep)
#
#     for iAlpha, alphaX in enumerate(alphaLst):
#         for jAlpha, alphaY in enumerate(alphaLst):
#
#             tmpA = []
#             tmpB = []
#             for iTest in range(nTest):
#                 x, y, z = datagen_func(nData, alphaX, alphaY, 0)
#                 rez = decomp_func(x, y, z)
#
#                 tmpA += [rez[labelA]]
#                 tmpB += [rez[labelB]]
#
#             rezAMat[iAlpha][jAlpha] = np.mean(tmpA)
#             rezBMat[iAlpha][jAlpha] = np.mean(tmpB)
#
#     # Find and report maximal synergy point
#     iAlphaMax, jAlphaMax = np.unravel_index(np.argmax(rezBMat), rezBMat.shape)
#     print('maxSyn', np.max(rezBMat), 'red', rezAMat[iAlphaMax][jAlphaMax], 'alpha', alphaLst[iAlphaMax], alphaLst[jAlphaMax])
#
#     # Find distribution at maximal synergy point
#     rezDict = {labelA: [], labelB: []}
#     for iTest in range(1000):
#         x, y, z = datagen_func(nData, alphaLst[iAlphaMax], alphaLst[jAlphaMax], 0)
#         rez = decomp_func(x, y, z)
#         rezDict[labelA] += [rez[labelA]]
#         rezDict[labelB] += [rez[labelB]]
#     dfMax = pd.DataFrame(rezDict)
#
#     print('1% quantile max synergy', np.quantile(rezDict[labelB], 0.99))
#
#     fig, ax = plt.subplots(ncols=3, figsize=(12,4), tight_layout=True)
#     imshow(fig, ax[0], rezAMat, title=labelA, haveColorBar=True)
#     imshow(fig, ax[1], rezBMat, title=labelB, haveColorBar=True)
#     sns.violinplot(ax=ax[2], data=dfMax)
#     plt.show()






