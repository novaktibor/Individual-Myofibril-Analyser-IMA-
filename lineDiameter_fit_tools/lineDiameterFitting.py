import numpy as np
from scipy.optimize import minimize
from lineDiameter_fit_tools.calculate_density_distribution import calculate_density_distribution as calculateDensityDistribution


def lineDiameterFitting(histogramStruct, modelFunctionStruct, algorithm, fittingParameterList, initialParameters, constrains, options):
    """
        Fit the model function to the histogram data of the measured distances.

        INPUT VARIABLES:
        - histogramStruct: structure containing histogram data
        - modelFunctionStruct: structure containing model function settings
        - algorithm: fitting algorithm to use ("fmincon", "fminsearch", "fminunc")
        - fittingParameterList: list of parameters to fit
        - initialParameters: initial parameters for the fitting
        - constrains: constraints on the fitting parameters
        - options: options parameter structure for the fitting algorithm (default=None)

        OUTPUT VARIABLES:
        - fittedParameters: updated model function parameters with the fitted ones
        - FVAL: function value at the minimum (residual)
        """

    # Set default options if not provided
    if options is None:
        options = {'MaxIter': 1000, 'TolFun': np.finfo(float).eps, 'TolX': np.finfo(float).eps}

    if algorithm == "fminsearch":
        def objective_function(x):
            return residuum(x, histogramStruct, modelFunctionStruct, fittingParameterList)

        # Perform optimization using scipy.optimize.minimize
        result = minimize(objective_function, initialParameters, method='nelder-mead', options=options) ### method='nelder-mead' ez hozott a Matlab kóddal megegyező eredményt ##

        # Extract the fitted parameters and the function value at the minimum
        fittedParamsVect = result.x
        FVAL = result.fun

    else:
        raise ValueError("Invalid fitting algorithm.")

    # Update the model function parameters with the fitted ones
    fittedParameters = modelFunctionStruct['modelFunctionSettings']
    for idxParam, param in enumerate(fittingParameterList):
        fittedParameters[param] = fittedParamsVect[idxParam]

    return fittedParameters, FVAL


def residuum(x, histogramStruct, modelFunctionStruct, fittingParameterList):
    """
    Calculate the residuum used for fitting the model function to the histogram original data.

    INPUT VARIABLES:
    - x: parameters to fit
    - histogramStruct: structure containing histogram data
    - modelFunctionStruct: structure containing model function settings
    - fittingParameterList: list of parameters to fit

    OUTPUT VARIABLES:
    - res: residual of the fitted model function
    """

    # Passing the settings of the model function
    modelFunctionSettings = modelFunctionStruct['modelFunctionSettings']

    modelFunctionStruct['backgroundFlag'] = False

    # Update parameters to fit
    for idxParameter, param in enumerate(fittingParameterList):
        modelFunctionSettings[param] = x[idxParameter]

    # Values of the model function
    modelFunc_Y_convolved_binned, _ = calculateDensityDistribution(x, modelFunctionSettings, modelFunctionStruct)

    # Passing the histogram counts of the original data
    histCounts = histogramStruct['histCounts']

    # Residual of the fitted model function
    res = np.sum((modelFunc_Y_convolved_binned - histCounts) ** 2) / np.sum(histCounts ** 2)

    return res