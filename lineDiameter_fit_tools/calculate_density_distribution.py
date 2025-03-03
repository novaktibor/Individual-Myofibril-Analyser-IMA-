import numpy as np

from lineDiameter_fit_tools.calculate_model_function import calculate_model_function
from lineDiameter_fit_tools.model_function_sampling import dx as model_function_sampling_dx
from lineDiameter_fit_tools.calculate_convolution_function import calculate_convolution_function
from lineDiameter_fit_tools.convolveModelFunction import convolveModelFunction as convolve_model_function
from lineDiameter_fit_tools.bin_model_function import bin_model_function


def calculate_density_distribution(data_coordinates, model_function_parameters, model_settings):
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
    model_func_y = calculate_model_function(sample_type, model_func_type, model_function_parameters, sampling_settings, background_flag)

    # Convolution
    model_func_sampling_dx = model_function_sampling_dx(sampling_settings['binningRegion'], sampling_settings['sampling_N'])
    conv_func = convolution_settings['convFunc']
    model_func_y_convolved = convolve_model_function(model_func_y, conv_func, model_func_sampling_dx)

    # Numerical Averaging of the convolved model density function, for comparison with the measured histogram density data
    model_func_y_convolved_binned = bin_model_function(model_func_y_convolved, sampling_settings['sampling_N'], sampling_settings['binRefinement'])

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
        model_func_sampling_dx = model_function_sampling_dx(sampling_settings['binningRegion'], sampling_settings['sampling_N'])
        conv_func, conv_rad_n = calculate_convolution_function(model_func_sampling_dx, convolution_settings['linkerRad'], convolution_settings['locPrecision'], convolution_settings['linkerType'], gaussian_conv_sigma)
        
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