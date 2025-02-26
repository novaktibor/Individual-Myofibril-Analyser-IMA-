import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from scipy.special import erf
import cmath






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
        'histCounts': histCounts,
        'histEdges': histEdges
    }

    # Configure parameters for further sampling of the model function
    binRefinement = 10  # Refinement factor for finer sampling
    histBinN, binningRegion, sampling_N = modelFunctionSampling_settings(histEdges, binRefinement)

    samplingSettings = {
        'binningRegion': binningRegion,  # The region over which the model is sampled
        'sampling_N': sampling_N,  # Number of points for sampling
        'binRefinement': binRefinement,  # Refinement factor
        'convRad_N': [],  # Placeholder for convolution radius
        'histBinN': histBinN  # Original histogram bin count
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

    # Return the fitted line width and additional fitting data
    return densFuncWidth_fitted, FigData


def modelFunctionSampling_dX(binningRegion, sampling_N):
    """
    Calculate the sampling bin sizes for numerical integration.

    Parameters:
    - binningRegion: The region over which the sampling is done
    - sampling_N: Number of samples

    Returns:
    - dX: Sampling bin sizes
    """

    sampling_dX = (binningRegion[1] - binningRegion[0]) / sampling_N
    return sampling_dX
    pass








def modelFunctionSampling_settings(histEdges, binRefinement):
    """
        This function determines how the model function should be
        sampled based on the histogram bin coordinates of the
        original data on the bin refinement. It only determines the
        region that should be binned later (larger region should be
        sampled because of the convolution).

        Parameters:
        - histEdges: The histogram bin coordinates of the original data
        - binRefinement: The level of refinement for the bins

        Returns:
        - histBinN: Number of bins in the original data
        - binningRegion: The region that should be binned later
        - sampling_N: Number of bins after refinement
        """

    # Number of bins in the original data:
    histBinN = len(histEdges) - 1

    # Number of bins after the refinement
    sampling_N = histBinN * binRefinement

    # Size of this region:
    binningRegion = [histEdges[0], histEdges[-1]]

    return histBinN, binningRegion, sampling_N


def calculateConvolutionFunction(modelFunc_sampling_dX, linkerRad, locPrecision, linkerType, GaussianConvSigma):
    """
        This function calculates the function with which the dye density should be
        convolved in order to compare it with the measured histogram data. This
        function considers two terms: the localization precision following
        Gaussian distribution and the inaccuracy persisting because of the linker
        length. The latter one follows the projection of a sphere surface and
        convolved with a line ('linkerType=sphere') which results in a top-hat
        function or can also be set to be a Gaussian with FWHM equaling to the
        linker length ('linkerType=Gaussian'). The resultant function is the
        convolution of the aforementioned two terms. If the effect of the linker
        into account as a Gaussian function then it calculates the convolution
        analytically instead of a numerical convolution.

        INPUT VARIABLES:
        - dX: distance of the sampling points for the convolution
        - linkerRad: length of the linkers
        - locPrecision: standard deviation of the localization precision, can be a vector containing values belonging to each localization
        - linkerType: type of distribution associated with the linker length, can be 'sphere' or 'Gaussian'
        - GaussianConvSigma: additional Gaussian smearing beside that of the localization precision and linker length that accounts for the uncertainties of the sample binding sites

        OUTPUT VARIABLES:
        - f: the convolution function with which the dye density should be convolved
        - convRad_N: half the points of the convolution function minus 0.5
        """

    GaussianVariance = locPrecision ** 2 + GaussianConvSigma ** 2

    STD_max = np.sqrt(np.max(GaussianVariance))

    linkerSTD = linkerRad / (2 * np.sqrt(2 * np.log(2)))

    if linkerType == 'sphere':
        t_max = linkerRad + 3 * STD_max
    elif linkerType == 'Gaussian':
        t_max = 3 * (linkerSTD + STD_max)
    else:
        raise ValueError('Unknown linker type')

    t_min = -t_max

    convRad_N = np.floor(np.ceil((t_max - t_min) / modelFunc_sampling_dX) / 2)
    conv_N = int(2 * convRad_N + 1)

    t = np.linspace(-convRad_N * modelFunc_sampling_dX, convRad_N * modelFunc_sampling_dX, conv_N)

    if linkerType == 'sphere':
        if linkerRad != 0:
            f_array = (-erf((-linkerRad + t) / np.sqrt(2 * GaussianVariance)) + erf(
                (linkerRad + t) / np.sqrt(2 * GaussianVariance))) / (2 * (2 * linkerRad))
        else:
            f_array = 1 / np.sqrt(2 * np.pi * GaussianVariance) * np.exp(-(t) ** 2 / (2 * GaussianVariance))
    elif linkerType == 'Gaussian':
        f_array = 1 / np.sqrt(2 * np.pi * (GaussianVariance + linkerSTD ** 2)) * np.exp(
            -(t) ** 2 / (2 * (GaussianVariance + linkerSTD ** 2)))
    else:
        raise ValueError('Unknown linker type')

    # Reshape f_array to a column vector
    f_array = f_array.reshape(-1, 1)

    # Calculate the mean along the second dimension (columns)
    f = np.mean(f_array, axis=1)

    return f, convRad_N


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

    histCounts = histogramStruct['histCounts']
    histEdges = histogramStruct['histEdges']
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
        result = minimize(objective_function, initialParameters, method='nelder-mead', options=options) ### method='nelder-mead' ez hozott a matalbb kóddal megegyező eredményt ##

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
    sampleType = modelFunctionStruct['sampleType']
    modelFuncType = modelFunctionStruct['modelFuncType']
    modelFunctionSettings = modelFunctionStruct['modelFunctionSettings']
    samplingSettings = modelFunctionStruct['samplingSettings']
    convolutionSettings = modelFunctionStruct['convolutionSettings']

    # Update parameters to fit
    for idxParameter, param in enumerate(fittingParameterList):
        modelFunctionSettings[param] = x[idxParameter]

    # Values of the model function
    modelFunc_Y_convolved_binned = calculateDensityDistribution(sampleType, modelFuncType, modelFunctionSettings,
                                                                samplingSettings, convolutionSettings)

    # Passing the histogram counts of the original data
    histCounts = histogramStruct['histCounts']

    # Residual of the fitted model function
    res = np.sum((modelFunc_Y_convolved_binned - histCounts) ** 2) / np.sum(histCounts ** 2)

    return res

def boundaries(binningRegion, sampling_N, convRad_N):
    """
    This function returns the left and the right boundaries of the sampling bins (the
    regions belonging to each sampling points) of the model function.
    They are determined from the histogram of the original data
    and from the radius of the convolution function (in number of sampling points) that the
    model function shall be convolved.

    Parameters:
    - binningRegion: Tuple containing the region for binning.
    - sampling_N: Number of sampling points.
    - convRad_N: Radius of the convolution function in number of sampling points.

    Returns:
    - samplingBoundaries_left: Left boundaries of the sampling bins.
    - samplingBoundaries_right: Right boundaries of the sampling bins.
    - sampling_dX: Size of the sampling bins.
    """

    # Calculate the size of the sampling bins
    sampling_dX = (binningRegion[1] - binningRegion[0]) / sampling_N

    # Calculate the number of boundaries
    samplingBoundaries_N = sampling_N + 2 * convRad_N + 1

    # Determine the sampling region
    samplingRegion = [binningRegion[0] - convRad_N * sampling_dX, binningRegion[1] + convRad_N * sampling_dX]

    # Generate the boundaries

    samplingBoundaries_N = int(samplingBoundaries_N)  # Convert to integer if it's a floating-point value
    samplingBoundaries = np.linspace(samplingRegion[0], samplingRegion[1], samplingBoundaries_N)

    # Calculate the left and right boundaries of the bins
    samplingBoundaries_left = samplingBoundaries[:-1]
    samplingBoundaries_right = samplingBoundaries[1:]

    return samplingBoundaries_left, samplingBoundaries_right, sampling_dX

def modelFunctionSampling_points(binningRegion, sampling_N, convRad_N):
    """
    This function returns the centers, the numbers, and the sizes of the sampling bins of the model function.
    They are determined from the histogram of the original data and from the radius of the convolution
    (in number of sampling points) function that the model function shall be convolved.

    Parameters:
    - binningRegion: Tuple containing the region for binning.
    - sampling_N: Number of sampling points.
    - convRad_N: Radius of the convolution function in number of sampling points.

    Returns:
    - sampling_X: Centers of the refined bins.
    - sampling_dX: Sizes of the sampling bins.
    - sampling_extended_N: Number of sampling points.
    """

    # Calculate the boundaries of the sampling bins
    samplingBoundaries_left, samplingBoundaries_right, sampling_dX = boundaries(binningRegion, sampling_N, convRad_N)

    # Calculate the sampling coordinates (centers of the refined bins)
    sampling_X = (samplingBoundaries_left + samplingBoundaries_right) / 2

    # Number of sampling points
    sampling_extended_N = len(sampling_X)

    return sampling_X, sampling_dX, sampling_extended_N


def lineDiameter_fit_visualization_plotDistribution(histogramStruct, modelFunctionStruct):
    # Passing the settings:
    histCounts = histogramStruct['histCounts']
    histEdges = histogramStruct['histEdges']
    hist_X = (histEdges[:-1] + histEdges[1:]) / 2
    sampleType = modelFunctionStruct['sampleType']
    modelFuncType = modelFunctionStruct['modelFuncType']
    modelFunctionSettings = modelFunctionStruct['modelFunctionSettings']
    samplingSettings = modelFunctionStruct['samplingSettings']
    convolutionSettings = modelFunctionStruct['convolutionSettings']
    linkerType = convolutionSettings['linkerType']

    # Calculate the values of the fitted distribution (convolved model function)
    sample_Y_convolved_fitted = calculateDensityDistribution(sampleType, modelFuncType, modelFunctionSettings,
                                                              samplingSettings, convolutionSettings)

    # Do not extend the region for the unconvolved model function
    samplingSettings['convRad_N'] = 0

    # Sampling without the extended region (because of the convolution)
    binningRegion = samplingSettings['binningRegion']
    sampling_N = samplingSettings['sampling_N']
    modelFunc_sampling_X_binningRegion = modelFunctionSampling_points(binningRegion, sampling_N,
                                                                      samplingSettings['convRad_N'])

    # Set convolution function to 1 and other settings
    convolutionSettings['convFunc'] = 1
    samplingSettings['binRefinement'] = 1
    samplingSettings['histBinN'] = len(modelFunc_sampling_X_binningRegion)
    sample_Y_fitted = calculateDensityDistribution(sampleType, modelFuncType, modelFunctionSettings,
                                                   samplingSettings, convolutionSettings)

    # Plot the fitted model function and the histogram data of the measured distances
    plt.figure(figsize=(8, 6))
    plt.hist(histEdges, bins=histEdges, weights=histCounts, alpha=0.7, color='blue', edgecolor='black')
    plt.plot(modelFunc_sampling_X_binningRegion, sample_Y_fitted, linewidth=1.5, color='red',
             label='Fitted Model Function')
    plt.plot(hist_X, sample_Y_convolved_fitted, color=[0.9290, 0.6940, 0.1250], linewidth=1.5,
             label='Fitted Convolved Function')

    plt.title(f"Hist of distances, {modelFuncType} fit, {linkerType} linker")
    plt.xlabel('distance [nm]')
    plt.ylabel('N of localizations')
    plt.legend()
    plt.grid(True)
    #plt.show()
    plt.clf()

# You would need to define the following functions to match the MATLAB code:
# calculateDensityDistribution()
# modelFunctionSampling_points()

def circlePreparation(circleRad, modelFunc_sampBoundaries_left, modelFunc_sampBoundaries_right, position):
    """
    This function examines which sampling points are affected by the disk of circle pattern.

    Parameters:
    - circleRad: Radius of the circle.
    - modelFunc_sampBoundaries_left: Left boundaries of the sampling bins.
    - modelFunc_sampBoundaries_right: Right boundaries of the sampling bins.
    - position: Position of the circle.

    Returns:
    - theta_left: Polar angle for the left boundary points.
    - theta_right: Polar angle for the right boundary points.
    - sampXWithinBool: Boolean vector for the sampling points within the circle.
    - boundIdx_left: Indices of the left sampling points of the circle where the sampling region is only partly within the circle.
    - boundIdx_right: Indices of the right sampling points of the circle where the sampling region is only partly within the circle.
    """

    # Angle value for the boundary points outside the circle, should be the same for the "left" and "right" boundary points
    thetaOutsideValue = 0  # Let it be

    # Finding which left boundaries are within the circle
    sampXWithinBool_left = np.logical_and(modelFunc_sampBoundaries_left > -circleRad + position, modelFunc_sampBoundaries_left < circleRad + position)

    # Polar angle for the left boundary points
    theta_left = np.zeros_like(modelFunc_sampBoundaries_left)
    theta_left[~sampXWithinBool_left] = thetaOutsideValue
    theta_left[sampXWithinBool_left] = np.arccos((modelFunc_sampBoundaries_left[sampXWithinBool_left] - position) / circleRad)

    # Finding which right boundaries are within the circle
    sampXWithinBool_right = np.logical_and(modelFunc_sampBoundaries_right > -circleRad + position, modelFunc_sampBoundaries_right < circleRad + position)

    # Polar angle for the right boundary points
    theta_right = np.zeros_like(modelFunc_sampBoundaries_right)
    theta_right[~sampXWithinBool_right] = thetaOutsideValue
    theta_right[sampXWithinBool_right] = np.arccos((modelFunc_sampBoundaries_right[sampXWithinBool_right] - position) / circleRad)

    # Boolean vector for the sampling points within the circle
    sampXWithinBool = np.logical_and(sampXWithinBool_left, sampXWithinBool_right)

    # Indices of the left and right sampling points of the circle where the sampling region is only partly within the circle
    boundIdx_left = np.where(np.logical_and(~sampXWithinBool_left, sampXWithinBool_right))[0]
    boundIdx_right = np.where(np.logical_and(sampXWithinBool_left, ~sampXWithinBool_right))[0]

    return theta_left, theta_right, sampXWithinBool, boundIdx_left, boundIdx_right

def model_functions_disk(modelFunctionSettings, samplingSettings):
    """
    This function describes the projection of a disk-like structure (solid cylinder).
    Samples the model function by integrating it over the sampling bins.

    Parameters:
    - modelFunctionSettings: Settings of the model function.
    - samplingSettings: Settings for sampling.

    Returns:
    - modelFunc_Y: Model function values.
    """

    # Parameters of the model function
    # Converting the FWHM (of projection of the disk) to the disk radius
    radius = modelFunctionSettings['width'] / np.sqrt(3)
    # Center of the disk
    position = modelFunctionSettings['position']

    # Get the sampling points
    modelFunc_sampBoundaries_left, modelFunc_sampBoundaries_right, _ = boundaries(samplingSettings['binningRegion'], samplingSettings['sampling_N'], samplingSettings['convRad_N'])

    # Check which boundary points are affected by the model function
    # and calculate the polar angle of the boundary points
    theta_left, theta_right, sampXWithinBool, boundIdx_left, boundIdx_right = circlePreparation(radius, modelFunc_sampBoundaries_left, modelFunc_sampBoundaries_right, position)

    # Initialize the model function
    modelFunc_Y = np.zeros_like(modelFunc_sampBoundaries_left)

    # Model function sampling values where the sampling region is fully within the rectangle
    modelFunc_Y[sampXWithinBool] = radius**2 * ((theta_left[sampXWithinBool] - np.sin(2 * theta_left[sampXWithinBool]) / 2) - (theta_right[sampXWithinBool] - np.sin(2 * theta_right[sampXWithinBool]) / 2))

    # Model function sampling values where the sampling region is only partially within the rectangle
    if boundIdx_left != boundIdx_right:
        modelFunc_Y[boundIdx_left] = radius**2 * (np.pi - (theta_right[boundIdx_left] - np.sin(2 * theta_right[boundIdx_left]) / 2))
        modelFunc_Y[boundIdx_right] = radius**2 * (theta_left[boundIdx_right] - np.sin(2 * theta_left[boundIdx_right]) / 2)
    else:
        modelFunc_Y[boundIdx_right] = radius**2 * np.pi

    # Normalize the model function
    modelFunc_Y = modelFunc_Y / (radius**2 * np.pi)

    return modelFunc_Y


def calculateModelFunction(sample_type, model_func_type, model_function_settings, sampling_settings):
    """
    This function calculates the model function to be fit on the histogram data
    of the measurement. This model function shall be convolved and binned
    before comparing it with the histogram of the original data.

    Parameters:
    - sample_type: Type of the sample ('single line', 'double lines', 'gap').
    - model_func_type: Type of the model function.
    - model_function_settings: Settings for the model function.
    - sampling_settings: Settings for sampling.

    Returns:
    - modelFunc_Y: Calculated model function.
    """

    # Model function prior to the convolution
    if sample_type == 'single line':
        if model_func_type == 'Disk':
            modelFunc_Y = model_functions_disk(model_function_settings, sampling_settings)

        else:
            raise ValueError('Invalid model function was given.')
    else:
        raise ValueError('Unknown sample type')

    return modelFunc_Y

def convolveModelFunction\
                (modelFunc_Y, convFunc, modelFunc_dX):
    """
    This function performs numerical convolution on the model function with a convolution function.

    Parameters:
    - modelFunc_Y: Model function.
    - convFunc: Convolution function.
    - modelFunc_dX: Sampling size of the model function.

    Returns:
    - modelFunc_Y_convolved: Convolved model function.
    """

    # Radius of the convolution function
    convRad_N = int((np.size(convFunc) - 1) / 2)

    # Initialization of the convolved model function
    modelFunc_Y_convolved = np.zeros(len(modelFunc_Y) - 2 * convRad_N)

    # Numerical calculation of the convolution
    for idxSamp in range(convRad_N + 1, len(modelFunc_Y_convolved) + convRad_N + 1):
        modelFunc_Y_convolved[idxSamp - convRad_N - 1] = np.sum(
            modelFunc_Y[idxSamp - convRad_N - 1:idxSamp + convRad_N] * convFunc) * modelFunc_dX

    return modelFunc_Y_convolved


def setArea(modelFunc_Y, modelFunc_sampling_dX, area, background):
    """
    Changes the area under the model function to the desired value.
    The area corresponds to the area under sampled model function plus the background.
    It does not add the background to the model function, only scales it.

    Parameters:
    - modelFunc_Y: Model function values.
    - modelFunc_sampling_dX: Size of the sampling bins.
    - area: Desired area under the model function.
    - background: Background values.

    Returns:
    - modelFunc_Y: Scaled model function.
    """

    # Area under the background
    if isinstance(background, (int, float)):
        area_background = background * len(modelFunc_Y) * modelFunc_sampling_dX
    elif len(background) == len(modelFunc_Y):
        area_background = np.sum(background) * modelFunc_sampling_dX
    else:
        raise ValueError('Invalid number of data for the background')

    # Area under the model function
    area_modelFunc = np.sum(modelFunc_Y) * modelFunc_sampling_dX

    # Scaling factor for the model function
    scaling = (area - area_background) / area_modelFunc

    # Changed model function
    modelFunc_Y = scaling * modelFunc_Y

    return modelFunc_Y

def setHeight(modelFunc_Y, height):
    """
    Changes the height of the model function to the desired value.
    The height corresponds to the sampled model function maximum value
    without background.

    Parameters:
    - modelFunc_Y: Model function values.
    - height: Desired height of the model function.

    Returns:
    - modelFunc_Y: Model function with adjusted height.
    """

    # Changed model function
    modelFunc_Y = height / np.max(modelFunc_Y) * modelFunc_Y

    return modelFunc_Y


def binModelFunction(modelFunc_Y_convolved, histBinN, binRefinement):
    """
    This function numerically integrates the model function over the
    histogram bins. The histogram bins must be equally sized.
    The integration is a simple average of the model function sampling points
    belonging to the different bins.

    Parameters:
    - modelFunc_Y_convolved: Model function values after convolution.
    - histBinN: Number of histogram bins.
    - binRefinement: Refinement factor for binning.

    Returns:
    - modelFunc_Y_convolved_binned: Model function integrated over the bins.
    """

    # Indices of the model function values belonging to the first bin
    intIndices_firstBin = np.arange(binRefinement)

    # For each bin, these values need to be added to the first bin indices
    # to get the indices of the model function values belonging to the bins
    intIndices_binIncrement = ((np.arange(histBinN)[:, np.newaxis] - 1) * binRefinement)

    # Indices of the model function values belonging to the bins
    linIndices_forPixelization = intIndices_firstBin + intIndices_binIncrement

    # Constants for the numerical integration
    numIntConstant = np.ones_like(linIndices_forPixelization) / binRefinement

    # Model function values integrated over the bins
    modelFunc_Y_convolved_binned = np.sum(numIntConstant * modelFunc_Y_convolved[linIndices_forPixelization], axis=1)

    return modelFunc_Y_convolved_binned

def calculateDensityDistribution(sampleType, modelFuncType, modelFunctionSettings, samplingSettings, convolutionSettings):
    # Calculate the convolution function
    modelFunc_sampling_dX = modelFunctionSampling_dX(samplingSettings['binningRegion'], samplingSettings['sampling_N'])

    if not convolutionSettings['iterGaussianConvBool']:
        # Convolution function already calculated prior to iteration
        convFunc = convolutionSettings['convFunc']
        convRad_N = samplingSettings['convRad_N']
    else:
        # Calculate convolution function within the iteration
        GaussianConvSigma = modelFunctionSettings['GaussianConvSigma']
        convFunc, convRad_N = calculateConvolutionFunction(modelFunc_sampling_dX, convolutionSettings['linkerRad'],
                                                           convolutionSettings['locPrecision'],
                                                           convolutionSettings['linkerType'], GaussianConvSigma)
        samplingSettings['convRad_N'] = convRad_N

    # Model function prior to convolution
    modelFunc_Y = calculateModelFunction(sampleType, modelFuncType, modelFunctionSettings, samplingSettings)

    # Convolution
    modelFunc_Y_convolved = convolveModelFunction(modelFunc_Y, convFunc, modelFunc_sampling_dX)

    # Convolve the background
    background = modelFunctionSettings['background']
    if np.size(background) == 1:  # If background is a scalar
        background_convolved = background
    elif np.size(background) == len(modelFunc_Y):  # If background has the same number of elements as modelFunc_Y
        background_convolved = convolveModelFunction(background, convFunc, modelFunc_sampling_dX)
    else:
        raise ValueError('Invalid number of data points of the background.')

    # Set the model function height or its area under the curve
    if 'height' in modelFunctionSettings and 'area' in modelFunctionSettings:
        raise ValueError('Please give only the desired height or the desired area of the model function, not both.')

    if 'height' in modelFunctionSettings:
        modelFunc_Y_convolved = setHeight(modelFunc_Y_convolved, modelFunctionSettings['height'])
    elif 'height1' in modelFunctionSettings:
        modelFunc_Y_convolved = setHeight(modelFunc_Y_convolved,
                                                         max(modelFunctionSettings['height1'],
                                                             modelFunctionSettings['height2']))
    elif 'area' in modelFunctionSettings:
        modelFunc_Y_convolved = setArea(modelFunc_Y_convolved, modelFunc_sampling_dX,
                                                       modelFunctionSettings['area'], background_convolved)
    elif 'gapDepth' in modelFunctionSettings:
        modelFunc_Y_convolved = setHeight(modelFunc_Y_convolved,
                                                         modelFunctionSettings['background'] +
                                                         modelFunctionSettings['gapDepth'] +
                                                         modelFunctionSettings['rimHeight'])
    else:
        raise ValueError('Either height or area value has to be given for the model function.')

    # Add the background
    modelFunc_Y_convolved += background_convolved

    # Numerical integration of the convolved model function for comparison with the measured histogram data
    modelFunc_Y_convolved_binned = binModelFunction(modelFunc_Y_convolved, samplingSettings['histBinN'],
                                                    samplingSettings['binRefinement'])

    return modelFunc_Y_convolved_binned

# You would need to define the following functions to match the MATLAB code:
# modelFunctionSampling_dX()
# calculateConvolutionFunction()
# calculateModelFunction()
# convolveModelFunction()
# modelFunctions_setHeight()
# modelFunctions_setArea()
# binModelFunction()





# Main function to process input data and calculate line length
def processStart(inputdata, user_definedPSF):
    """
    Main entry point for the program. Processes input histogram data, performs line fitting,
    and calculates line length based on modeled data.

    Parameters:
    - inputdata: List or array-like object containing histogram data (bin centers, bin counts).

    Returns:
    - lineLength: The computed line length based on model fitting.
    - FigData: A dictionary containing fitted and processed histogram/model data.
    """
    # Convert input data to Pandas DataFrame for easier processing
    array = pd.DataFrame(inputdata)
    array = array.apply(pd.to_numeric, errors='coerce')
    array = array.to_numpy()

    # Validate and process data by calculating histogram properties
    BinWidth = array[1, 0] - array[0, 0]  # Bin width of the histogram
    NumBins = len(array)  # Total number of bins in the histogram
    start = array[0, 0] - BinWidth / 2  # Start edge of the first bin
    stop = array[-1, 0] + BinWidth / 2  # End edge of the last bin

    # Generate histogram bin edges and counts
    BinCounts = array[:, 1]  # Extract bin counts
    BinCountslen = np.size(BinCounts)  # Number of bins
    BinEdges = np.linspace(start, stop, num=BinCountslen + 1)

    # Create the histogram data structure
    Histogram = {
        'BinWidth': BinWidth,
        'NumBins': NumBins,
        'BinEdges': BinEdges,
        'BinCounts': BinCounts
    }

    # Perform line diameter fitting using the single rectangular fit function
    densFuncWidth_fitted, FigData = lineDiameter_singlerectfit(
        locPrecision=0,
        histHandler=Histogram,
        linkerType='Gaussian',
        modelFuncType='Disk',
        linkerRad=user_definedPSF
    )

    # Derive line length from the fitted width (FWHM)
    # Conversion considers specific geometric proportions
    lineLength = densFuncWidth_fitted / 1000 / 0.866025403784438

    return lineLength, FigData