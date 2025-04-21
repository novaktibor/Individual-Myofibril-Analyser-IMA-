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

    return binningRegion, sampling_N

def modelFunctionSampling_boundaries(binningRegion, sampling_N, convRad_N):
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


def sampling_centers(sampling_settings):
    # Returns the centers of the sampling bins without the padding used for the convolution.
    
    binningRegion = sampling_settings['binningRegion']
    sampling_N = sampling_settings['sampling_N']
    # discard the padding
    convRad_N = 0
    
    model_func_samp_boundaries_left, model_func_samp_boundaries_right, sampling_dx = modelFunctionSampling_boundaries(
        binningRegion, sampling_N, convRad_N
    )
    
    model_func_x = (model_func_samp_boundaries_left + model_func_samp_boundaries_right) / 2
    
    return model_func_x


def calculateConvolutionFunction(dX, linkerRad, locPrecision, linkerType, GaussianConvSigma):
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

    convRad_N = np.floor(np.ceil((t_max - t_min) / dX) / 2)
    conv_N = int(2 * convRad_N + 1)

    t = np.linspace(-convRad_N * dX, convRad_N * dX, conv_N)

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


    # Beware!!
    # the python version will work with single "locPrecision" value, and not a vector of "locPrecisions" like the Matlab version
    # Reshape f_array to a column vector
    # f_array = f_array.reshape(-1, 1)
    #f = np.mean(f_array, axis=1)
    f = f_array

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

def circleSlicing(circle_rad, model_func_samp_boundaries_left, model_func_samp_boundaries_right, position):
    """
    This function "slices" the circle by the sampling bins, i.e. it finds
    angles belonging to the sampling boundaries that falls within a circle.
    It is later used for calculating the areas of segments.
    
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
    theta_outside_value = 0  # Let it be
    
    # Finding which left boundaries are within the circle
    samp_x_within_bool_left = np.zeros_like(model_func_samp_boundaries_left, dtype=bool)
    samp_x_within_bool_left[(model_func_samp_boundaries_left > -circle_rad + position) & (model_func_samp_boundaries_left < circle_rad + position)] = True
    
    # Polar angle for the left boundary points
    theta_left = np.full_like(model_func_samp_boundaries_left, theta_outside_value, dtype=float)
    theta_left[samp_x_within_bool_left] = np.arccos((model_func_samp_boundaries_left[samp_x_within_bool_left] - position) / circle_rad)
    
    # Finding which right boundaries are within the circle
    samp_x_within_bool_right = np.zeros_like(model_func_samp_boundaries_right, dtype=bool)
    samp_x_within_bool_right[(model_func_samp_boundaries_right > -circle_rad + position) & (model_func_samp_boundaries_right < circle_rad + position)] = True
    
    # Polar angle for the right boundary points
    theta_right = np.full_like(model_func_samp_boundaries_right, theta_outside_value, dtype=float)
    theta_right[samp_x_within_bool_right] = np.arccos((model_func_samp_boundaries_right[samp_x_within_bool_right] - position) / circle_rad)
    
    # Boolean vector for the sampling points within the circle
    samp_x_within_bool = samp_x_within_bool_left & samp_x_within_bool_right
    
    # Indices of the left and right sampling points of the circle where the sampling region is only partly within the circle
    bound_idx_left = np.where((model_func_samp_boundaries_left <= -circle_rad + position) & (model_func_samp_boundaries_right > -circle_rad + position))[0]
    bound_idx_right = np.where((model_func_samp_boundaries_right >= circle_rad + position) & (model_func_samp_boundaries_left < circle_rad + position))[0]
    
    return theta_left, theta_right, samp_x_within_bool, bound_idx_left, bound_idx_right

def model_functions_disk(model_function_settings, sampling_settings):
    """
    This function describes the projection of a disk-like structure (solid cylinder).
    It samples the density model function with sampling bins and
    returns an upscaled (whose height is increased to a requested value)
    density function.

    Parameters:
    - modelFunctionSettings: Settings of the model function.
    - samplingSettings: Settings for sampling.

    Returns:
    - modelFunc_Y: Model function values.
    """

    # Parameters of the model function
    # Converting the FWHM (of projection of the disk) to the disk radius
    radius = model_function_settings['width'] / np.sqrt(3)
    # Center of the disk
    position = model_function_settings['position']
    
    # Get the sampling points
    model_func_samp_boundaries_left, model_func_samp_boundaries_right, sampling_dx = modelFunctionSampling_boundaries(
        sampling_settings['binningRegion'], sampling_settings['sampling_N'], sampling_settings['convRad_N']
    )
    
    # Check which boundary points are affected by the model function
    # and calculate the polar angle of the boundary points
    theta_left, theta_right, samp_x_within_bool, bound_idx_left, bound_idx_right = circleSlicing(
        radius, model_func_samp_boundaries_left, model_func_samp_boundaries_right, position
    )
    
    # Initialize the model function
    model_func_y = np.zeros_like(model_func_samp_boundaries_left)
    
    # Model function sampling values where the sampling region is fully within the rectangle
    model_func_y[samp_x_within_bool] = radius**2 * (
        (theta_left[samp_x_within_bool] - np.sin(2 * theta_left[samp_x_within_bool]) / 2) -
        (theta_right[samp_x_within_bool] - np.sin(2 * theta_right[samp_x_within_bool]) / 2)
    )
    
    # Model function sampling values where the sampling region is only partially within the rectangle
    if bound_idx_left != bound_idx_right:
        # leftmost bin (segment angles of pi and theta_right[bound_idx_left])
        model_func_y[bound_idx_left] = radius**2 * (np.pi - (theta_right[bound_idx_left] - np.sin(2 * theta_right[bound_idx_left]) / 2))
        # rightmost bin (segment angles of theta_left[bound_idx_right] and 0)
        model_func_y[bound_idx_right] = radius**2 * (theta_left[bound_idx_right] - np.sin(2 * theta_left[bound_idx_right]) / 2)
    else:
        model_func_y[bound_idx_right] = radius**2 * np.pi
        
    # normalize the model function to unit area
    model_func_y /= (radius**2 * np.pi)
    
    # calculate the density
    model_func_y /= sampling_dx
    
    # the "height" should scale up the model function by its central bin density as it were placed exactly in the middle
    if 2 * radius > sampling_dx:
        # average density of the central bin
        fi_central = np.arcsin(sampling_dx / 2 / radius)
        central_bin_value = (2 * fi_central * radius**2 + sampling_dx * radius * np.cos(fi_central))/(radius**2*np.pi)/sampling_dx
    else:
        # if model function width is smaller the sampling bins, take the whole normalized model function for the density calculation
        central_bin_value = 1 / sampling_dx
    
    # check: before the upscaling, the model function should be normalized:
    # np.sum(model_func_y) * sampling_dx == 1
    
    area = model_function_settings['height'] / central_bin_value
    
    model_func_y *= area
    
    return model_func_y, area


def calculateModelFunction(sample_type, model_func_type, model_function_settings, sampling_settings, background_flag):
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
            modelFunc_Y, _ = model_functions_disk(model_function_settings, sampling_settings)

        else:
            raise ValueError('Invalid model function was given.')
    else:
        raise ValueError('Unknown sample type')

    if background_flag:
        background_level = model_function_settings['background']
        modelFunc_Y += background_level

    return modelFunc_Y

def convolveModelFunction(modelFunc_Y, convFunc, modelFunc_dX):
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


def binModelFunction(model_func_y_convolved, sampling_n, bin_refinement):
    """
    This function numerically averages the localization density model
    function over the histogram bins. The histogram bins must be equally sized.
    The integration is a simple average of the model function sampling points
    belonging to the different bins.
    
    Parameters:
    - model_func_y_convolved: Model function values after convolution.
    - sampling_n: Number of refined sampling points.
    - bin_refinement: Refinement factor for binning.

    Returns:
    - modelFunc_Y_convolved_binned: Model function integrated over the bins.
    
    """
    
    # Number of the original histogram bins
    hist_bin_n = sampling_n // bin_refinement
    
    if hist_bin_n * bin_refinement != sampling_n:
        raise ValueError('The model function sampling number and the histogram bin refinement are inconsistent, check them.')
    
    # Indices of the model function values belonging to the first bin
    int_indices_first_bin = np.arange(bin_refinement)
    
    # For each bin, these values need to be added to the aforementioned indices
    # to get the indices of the model function values belonging to the bins
    int_indices_bin_increment = (np.arange(hist_bin_n) * bin_refinement).reshape(-1, 1)
    
    # Indices of the model function values belonging to the bins
    lin_indices_for_pixelization = int_indices_first_bin + int_indices_bin_increment
    
    # Model function values integrated over the bins
    model_func_y_convolved_binned = np.mean(model_func_y_convolved[lin_indices_for_pixelization], axis=1)
    
    return model_func_y_convolved_binned

def calculateDensityDistribution(data_coordinates, model_function_parameters, model_settings):
    """
    This function calculates the model function to be fit on the histogram
    data of the measurement. First, calls a function that calculates the
    function values of the supposed binding site density, then convolves it
    with the inaccuracies arising from the localization precision and the
    linker length, finally numerically integrates it over the histogram bins.
    """
    
    # Unfolding the model settings
    sample_type = model_settings['sampleType']
    model_func_type = model_settings['modelFuncType']
    sampling_settings = model_settings['samplingSettings']
    convolution_settings = model_settings['convolutionSettings']
    background_flag = model_settings['backgroundFlag']

    # Calculate the convolution function
    gaussian_conv_sigma = model_function_parameters['GaussianConvSigma']
    sampling_settings, convolution_settings = update_convolution_settings(sampling_settings, convolution_settings, gaussian_conv_sigma)

    # Model function prior to the convolution
    model_func_y = calculateModelFunction(sample_type, model_func_type, model_function_parameters, sampling_settings, background_flag)

    # Convolution
    model_func_sampling_dx = modelFunctionSampling_dX(sampling_settings['binningRegion'], sampling_settings['sampling_N'])
    conv_func = convolution_settings['convFunc']
    model_func_y_convolved = convolveModelFunction(model_func_y, conv_func, model_func_sampling_dx)

    # Numerical Averaging of the convolved model density function, for comparison with the measured histogram density data
    model_func_y_convolved_binned = binModelFunction(model_func_y_convolved, sampling_settings['sampling_N'], sampling_settings['binRefinement'])

    # Set the model function height or its area under the curve
    if 'area' in model_function_parameters and model_function_parameters['area']:
        model_func_y_convolved_binned, scaling = set_area(model_func_y_convolved_binned, model_function_parameters['area'])
    elif 'convolvedHeight' in model_function_parameters and model_function_parameters['convolvedHeight']:
        model_func_y_convolved_binned, scaling = set_maximum(model_func_y_convolved_binned, model_function_parameters['convolvedHeight'])
    else:
        scaling = 1

    return model_func_y_convolved_binned, scaling

def update_convolution_settings(sampling_settings, convolution_settings, gaussian_conv_sigma):
    if not convolution_settings['iterGaussianConvBool']:
        # In this case, the convolution function should already be calculated, prior to the iteration
        pass
    else:
        # For calculating the convolution function within the iteration
        model_func_sampling_dx = modelFunctionSampling_dX(sampling_settings['binningRegion'], sampling_settings['sampling_N'])
        conv_func, conv_rad_n = calculateConvolutionFunction(model_func_sampling_dx, convolution_settings['linkerRad'], convolution_settings['locPrecision'], convolution_settings['linkerType'], gaussian_conv_sigma)
        
        convolution_settings['convFunc'] = conv_func
        sampling_settings['convRad_N'] = conv_rad_n
    
    return sampling_settings, convolution_settings

def set_area(model_func_y, area):
    """
    Changes the area under the model function to the desired value.
    The area corresponds to the area under sampled model function plus the background.
    It does not add the background to the model function, only scales it.
    """
    area_model_func = np.sum(model_func_y)
    scaling = area / area_model_func
    model_func_y = scaling * model_func_y
    return model_func_y, scaling

def set_maximum(model_func_y, height):
    """
    Changes the height of the model function to the desired value.
    The height corresponds to the sampled model function maximum value without background.
    """
    scaling = height / np.max(model_func_y)
    model_func_y = scaling * model_func_y
    return model_func_y, scaling


def visualization_calculate(histogram_struct, model_function_parameters, model_settings):

    # Passing the settings
    hist_counts = histogram_struct['histCounts']
    hist_edges = histogram_struct['histEdges']
    hist_x = (hist_edges[:-1] + hist_edges[1:]) / 2
    model_func_type = model_settings['modelFuncType']
    sampling_settings = model_settings['samplingSettings']
    convolution_settings = model_settings['convolutionSettings']
    linker_type = convolution_settings['linkerType']

    bin_refinement = sampling_settings['binRefinement']
    binning_region = sampling_settings['binningRegion']
    sampling_n = sampling_settings['sampling_N']

    # The sampling points for the calculated unbinned model functions
    sample_x = sampling_centers(model_settings['samplingSettings'])

    # Convolved model function
    model_settings_convolved = model_settings.copy()
    model_settings_convolved['samplingSettings']['binRefinement'] = 1

    # Calculate the values of the fitted distribution (convolved model function)
    sample_y_fitted_convolved, _ = calculateDensityDistribution(hist_edges, model_function_parameters, model_settings_convolved)

    # Unconvolved model function
    sampling_dx = modelFunctionSampling_dX(binning_region, sampling_n)
    conv_rad_n = int(sampling_settings['convRad_N'])
    conv_dirac_delta = convolution_function_dirac_delta(conv_rad_n, sampling_dx)
    convolution_settings_dirac_delta = convolution_settings.copy()
    convolution_settings_dirac_delta['convFunc'] = conv_dirac_delta
    model_settings_unconvolved = model_settings_convolved.copy()
    model_settings_unconvolved['convolutionSettings'] = convolution_settings_dirac_delta

    # Change the height of the (unbinned) model function to that of the original histogram
    sample_y_fitted_unconvolved, _ = calculateDensityDistribution(hist_edges, model_function_parameters, model_settings_unconvolved)

    return sample_x, sample_y_fitted_unconvolved, sample_y_fitted_convolved


def visualization_plotDistribution(histogram_struct, model_function_parameters, model_settings):
    """
    This function plots the distribution of the histogram data and the fitted model function.
    """
    
    sample_x, sample_y_fitted_unconvolved, sample_y_fitted_convolved = visualization_calculate(histogram_struct, model_function_parameters, model_settings)
    
    # Visualization
    fitted_hist_figure = plt.figure()
    plt.hist(hist_x, bins=hist_edges, weights=hist_counts, alpha=0.7, label='Histogram Data')
    plt.plot(sample_x, sample_y_fitted_unconvolved, linewidth=1.5, color='red', label='Unconvolved Model Function')
    plt.plot(sample_x, sample_y_fitted_convolved, color=[0.9290, 0.6940, 0.1250], linewidth=2.5, label='Convolved Model Function')
    
    plt.title(f'Hist of distances, {model_func_type} fit, {linker_type} linker')
    plt.xlabel('Distance [nm]')
    plt.ylabel('Intensity [Counts/nm^2???]')
    plt.legend()
    plt.grid(True)
    
    return fitted_hist_figure

def convolution_function_dirac_delta(conv_rad_n, model_func_sampling_dx):
    """
    Returns a Dirac delta function for numerical convolution with the given kernel size.
    """
    
    # Number of sampling points for the convolved function, an odd number
    conv_n = 2 * conv_rad_n + 1
    
    # Sampling points for the convolved function
    f = np.zeros(conv_n)
    f[conv_rad_n] = 1 / model_func_sampling_dx
    
    return f



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
