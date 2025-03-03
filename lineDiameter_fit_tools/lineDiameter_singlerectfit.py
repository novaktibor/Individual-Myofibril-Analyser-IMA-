import numpy as np

from lineDiameter_fit_tools.model_function_sampling import settings as modelFunctionSampling_settings
from lineDiameter_fit_tools.model_function_sampling import dx as modelFunctionSampling_dX
from lineDiameter_fit_tools.calculate_convolution_function import calculate_convolution_function as calculateConvolutionFunction
from lineDiameter_fit_tools.getInitialParameters_singleLine import getInitialParameters_singleLine
from lineDiameter_fit_tools.lineDiameterFitting import lineDiameterFitting

from lineDiameter_fit_tools.lineDiameter_fit_visualization_single_line import single_line as lineDiameter_fit_visualization_single_line

def lineDiameter_singlerectfit(locPrecision, histHandler, linkerType, modelFuncType, linkerRad):
    """
    Fits a single rectangular model function to histogram data to estimate a line's width.

    Parameters:
    - locPrecision: The localization precision (typically a measure of error or noise in data).
    - histHandler: Dictionary containing histogram data with the following keys:
        - 'BinCounts': Array of bin counts from the histogram.
        - 'BinEdges': Array of edges of the histogram bins.
    - linkerType: Type of linker distribution ('sphere' or 'Gaussian').
    - modelFuncType: Type of the model function to use ('Disk', etc.).
    - linkerRad: Radius (or characteristic length) of the linker.

    Returns:
    - densFuncWidth_fitted: Fitted width of the line based on the histogram data and model.
    - FigData: Dictionary containing visualization and fitting details (histogram struct, fitted parameters, etc.).
    """

    # Determines if iterative convolution with a Gaussian function should be performed
    iterGaussianConvBool = False
    # Specifies the sample type (a single line is assumed here)
    sampleType = 'single line'

    # Extract histogram counts and edges from the input data and structure them
    histCounts = np.transpose(histHandler['BinCounts'])
    histEdges = np.transpose(histHandler['BinEdges'])

    histogramStruct = {
        'values': histCounts,
        'coordinates': histEdges
    }

    # Configure parameters for further sampling of the model function
    binRefinement = 10  # Refinement factor for finer sampling
    binningRegion, sampling_N = modelFunctionSampling_settings(histEdges, binRefinement)

    samplingSettings = {
        'binningRegion': binningRegion,  # The region over which the model is sampled
        'sampling_N': sampling_N,  # Number of points for sampling
        'binRefinement': binRefinement,  # Refinement factor
        'convRad_N': [],  # Placeholder for convolution radius
    }

    # Initial parameters for the model function
    modelFunctionParameters = {
        'GaussianConvSigma': 0,  # Sigma for Gaussian convolution (if applied)
        'ringThickness': 250  # Thickness of the ring in the model function
    }

    # Configure convolution settings based on whether iterative Gaussian convolution is used
    if not iterGaussianConvBool:
        modelFunc_sampling_dX = modelFunctionSampling_dX(binningRegion, sampling_N)
        convFunc, convRad_N = calculateConvolutionFunction(
            modelFunc_sampling_dX,
            linkerRad,
            locPrecision,
            linkerType,
            modelFunctionParameters['GaussianConvSigma']
        )
    else:
        convFunc = []  # No convolution applied initially
        convRad_N = []

    convolutionSettings = {
        'convFunc': convFunc,  # Convolution function
        'locPrecision': locPrecision,  # Localization precision
        'iterGaussianConvBool': iterGaussianConvBool,  # Whether iterative convolution is used
        'linkerType': linkerType,  # Type of linker
        'linkerRad': linkerRad  # Linker radius (or length)
    }

    # Update sampling settings with convolution radius
    samplingSettings['convRad_N'] = convRad_N

    # Prepare the model function structure with settings and sampling information
    modelFunctionStruct = {
        'sampleType': sampleType,
        'modelFuncType': modelFuncType,
        'modelFunctionSettings': modelFunctionParameters,
        'samplingSettings': samplingSettings,
        'convolutionSettings': convolutionSettings
    }

    # Prepare for fitting: select algorithm and parameters to be optimized
    algorithm = 'fminsearch'  # Optimization algorithm (Nelder-Mead used here)
    fittingParameterList = ['width', 'height', 'position', 'background']

    # Add Gaussian convolution sigma as a fitting parameter if iterative convolution is enabled
    if iterGaussianConvBool:
        fittingParameterList.append('GaussianConvSigma')

    # Initialize fitting parameters and constraints
    initialParameters, constrains = getInitialParameters_singleLine(
        histogramStruct, modelFuncType, modelFunctionParameters, fittingParameterList
    )

    # Configure optimization options (bounds and tolerances)
    options = {'maxiter': 1000, 'ftol': np.finfo(float).eps, 'xtol': np.finfo(float).eps}

    # Perform the fitting process to find the best parameters
    fittedParameters, FVAL = lineDiameterFitting(
        histogramStruct,
        modelFunctionStruct,
        algorithm,
        fittingParameterList,
        initialParameters,
        constrains,
        options
    )

    # Extract key fitted parameters for further use or visualization
    densFuncWidth_fitted = fittedParameters['width']  # Fitted width parameter
    densFuncHeight_fitted = fittedParameters['height']  # Fitted height parameter
    densFuncBG_fitted = fittedParameters['background']  # Fitted background level

    # Handle Gaussian convolution sigma depending on iterative convolution configuration
    if iterGaussianConvBool:
        GaussianConvSigma_fitted = fittedParameters['GaussianConvSigma']
    else:
        GaussianConvSigma_fitted = modelFunctionParameters['GaussianConvSigma']

    # Update model function settings with the fitted parameters
    fittedParameterNames = fittedParameters.keys()
    for param in fittedParameterNames:
        modelFunctionParameters[param] = fittedParameters[param]

    modelFunctionStruct['modelFunctionSettings'] = modelFunctionParameters

    # Optionally recalculate the convolution function, if Gaussian convolution is used
    if iterGaussianConvBool:
        convFunc, convRad_N = calculateConvolutionFunction(
            modelFunc_sampling_dX,
            linkerRad,
            locPrecision,
            linkerType,
            GaussianConvSigma_fitted
        )
        convolutionSettings['convFunc'] = convFunc
        convolutionSettings['convRad_N'] = convRad_N
        modelFunctionStruct['convolutionSettings'] = convolutionSettings

    # Package result and additional data for visualization or further analysis
    FigData = {
        'histogramStruct': histogramStruct,
        'modelFunctionStruct': modelFunctionStruct,
        'densFuncWidth_fitted': densFuncWidth_fitted,  # Fitted line width
        'densFuncBG_fitted': densFuncBG_fitted,  # Fitted background level
        'FVAL': FVAL,  # Residual of the fitting
        'xystartendall': [],  # Additional visualization data
        'xall': [],  # Additional visualization data
        'perpendicular_lines': []  # Additional visualization data
    }

    lineDiameter_fit_visualization_single_line(histogramStruct, fittedParameters, modelFunctionStruct, densFuncWidth_fitted, densFuncBG_fitted, FVAL)

    # Return the fitted line width and additional fitting data
    return densFuncWidth_fitted, FigData