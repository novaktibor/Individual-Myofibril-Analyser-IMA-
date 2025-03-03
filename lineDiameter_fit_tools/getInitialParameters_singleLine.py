import numpy as np

def getInitialParameters_singleLine(histogramStruct, modelFuncType, modelFunctionSettings, fittingParameterList):
    """
        This function initializes the initial parameters and constraints vector for fitting a single line model function to the histogram data.

        INPUT VARIABLES:
        - histogramStruct: structure containing histogram data
        - modelFuncType: type of the model function
        - modelFunctionSettings: settings for the model function
        - fittingParameterList: list of parameters to fit

        OUTPUT VARIABLES:
        - initialParameters: initial parameters for fitting
        - constrains: constraints on the fitting parameters
        """

    histCounts = histogramStruct['values']
    histEdges = histogramStruct['coordinates']
    histN = len(histCounts)
    histX = (histEdges[0:-1] + histEdges[1:]) / 2

    # initialize the initial parameters and the constraints vector
    initialParameters = np.zeros(len(fittingParameterList))
    constrains = np.zeros((len(fittingParameterList), 2))

    # set the background before the other parameters
    backgroundBoolVect = [param == 'background' for param in fittingParameterList]
    if 'background' in fittingParameterList:
        histN_forBG = histN // 8
        initialParameters[backgroundBoolVect] = np.mean(
            [np.mean(histCounts[:histN_forBG]), np.mean(histCounts[-histN_forBG:])])
        constrains[backgroundBoolVect, :] = [0, np.inf]
        background = initialParameters[backgroundBoolVect]
    else:
        if 'background' in modelFunctionSettings:
            background = modelFunctionSettings['background']
        else:
            raise ValueError('The background for the model function (single line) has to be set or to be fitted.')

    # set the central position before the other parameters
    positionBoolVect = [param == 'position' for param in fittingParameterList]
    if 'position' in fittingParameterList:

        initialParameters[positionBoolVect] = np.sum(
            (histCounts - background) * histX / np.sum(histCounts - background))
        constrains[positionBoolVect, :] = [-np.inf, np.inf]
        position = initialParameters[positionBoolVect]
    else:
        if 'position' in modelFunctionSettings:
            position = modelFunctionSettings['position']
        else:
            raise ValueError('The position for the model function (single line) has to be set or to be fitted.')

    for idxParam, param in enumerate(fittingParameterList):
        if param == 'width':

            valami1 = (histCounts - background)
            valmi2 = (histX - position) ** 2

            initialParameters[idxParam] = 2.0 * np.sqrt(
                np.sum(((histCounts - background) * (histX - position) ** 2)) / ((np.sum(histCounts - background) - 1)))
            constrains[idxParam, :] = [0, np.inf]
        elif param == 'area':
            initialParameters[idxParam] = np.sum(histCounts) * (histEdges[1] - histEdges[0])
            constrains[idxParam, :] = [0, np.inf]
        elif param == 'height':
            initialParameters[idxParam] = np.max(histCounts)
            constrains[idxParam, :] = [0, np.inf]
        elif param == 'position' or param == 'background':
            pass
        elif param == 'GaussianConvSigma':
            initialParameters[idxParam] = abs(histEdges[1] - histEdges[0]) * 0.01
            constrains[idxParam, :] = [0, np.inf]
        else:
            raise ValueError('Invalid parameter was given to be fit (single line fitting).')

    return initialParameters, constrains