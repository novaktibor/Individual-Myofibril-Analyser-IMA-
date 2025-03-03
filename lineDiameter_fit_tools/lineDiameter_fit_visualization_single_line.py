import numpy as np
import matplotlib.pyplot as plt

import lineDiameter_fit_tools.simpleModelFunctions as simple_model_functions
from lineDiameter_fit_tools.model_function_sampling import dx as model_function_sampling_dx
from lineDiameter_fit_tools.calculate_density_distribution import calculate_density_distribution

def plot_distribution(histogram_struct, model_function_parameters, model_settings):
    """
    This function plots the distribution of the histogram data and the fitted model function.
    """
    
    # Passing the settings
    hist_counts = histogram_struct['values']
    hist_edges = histogram_struct['coordinates']
    hist_x = (hist_edges[:-1] + hist_edges[1:]) / 2
    model_func_type = model_settings['modelFuncType']
    sampling_settings = model_settings['samplingSettings']
    convolution_settings = model_settings['convolutionSettings']
    linker_type = convolution_settings['linkerType']
    
    bin_refinement = sampling_settings['binRefinement']
    binning_region = sampling_settings['binningRegion']
    sampling_n = sampling_settings['sampling_N']
    
    # The sampling points for the calculated unbinned model functions
    sample_x = simple_model_functions.sampling_centers(model_settings['samplingSettings'])
    
    # Convolved model function
    model_settings_convolved = model_settings.copy()
    model_settings_convolved['samplingSettings']['binRefinement'] = 1
    
    # Calculate the values of the fitted distribution (convolved model function)
    sample_y_fitted_convolved, _ = calculate_density_distribution(hist_edges, model_function_parameters, model_settings_convolved)
    
    # Unconvolved model function
    sampling_dx = model_function_sampling_dx(binning_region, sampling_n)
    conv_rad_n = sampling_settings['convRad_N']
    conv_dirac_delta = convolution_function_dirac_delta(conv_rad_n, sampling_dx)
    convolution_settings_dirac_delta = convolution_settings.copy()
    convolution_settings_dirac_delta['convFunc'] = conv_dirac_delta
    model_settings_unconvolved = model_settings_convolved.copy()
    model_settings_unconvolved['convolutionSettings'] = convolution_settings_dirac_delta
    
    # Change the height of the (unbinned) model function to that of the original histogram
    sample_y_fitted_unconvolved, _ = calculate_density_distribution(hist_edges, model_function_parameters, model_settings_unconvolved)
    
    # Visualization
    fitted_hist_figure = plt.figure()
    plt.hist(hist_x, bins=hist_edges, weights=hist_counts, alpha=0.7, label='Histogram Data')
    plt.plot(sample_x, sample_y_fitted_unconvolved, linewidth=1.5, color='red', label='Unconvolved Model Function')
    plt.plot(sample_x, sample_y_fitted_convolved, color=[0.9290, 0.6940, 0.1250], linewidth=2.5, label='Convolved Model Function')
    
    plt.title(f'Hist of distances, {model_func_type} fit, {linker_type} linker')
    plt.xlabel('Distance [nm]')
    plt.ylabel('Localization density [1/nm]')
    plt.legend()
    
    # TODO!!!
    plt.show()
    #plt.close(fitted_hist_figure)
    return fitted_hist_figure

def single_line(histogram_struct, model_function_parameters, model_settings, dens_func_width_fitted, dens_func_bg_fitted, FVAL):
    fitted_hist_figure = plot_distribution(histogram_struct, model_function_parameters, model_settings)
    
    # Passing the settings
    hist_counts = histogram_struct['values']
    hist_edges = histogram_struct['coordinates']
    
    # The parameters that determine the standard deviation of the convolution function
    loc_precision = model_settings['convolutionSettings']['locPrecision']
    linker_rad = model_settings['convolutionSettings']['linkerRad']
    gaussian_conv_sigma = model_settings['modelFunctionSettings']['GaussianConvSigma']
    
    # Variance of the convolution function
    if model_settings['convolutionSettings']['linkerType'] == 'sphere':
        linker_std = linker_rad / np.sqrt(3)
        conv_func_var = np.mean(loc_precision)**2 + linker_std**2 + gaussian_conv_sigma**2
    elif model_settings['convolutionSettings']['linkerType'] == 'Gaussian':
        linker_std = linker_rad / (2 * np.sqrt(2 * np.log(2)))
        conv_func_var = np.mean(loc_precision)**2 + linker_std**2 + gaussian_conv_sigma**2
    else:
        raise ValueError('Unknown linker type')
    
    # Variance of the localization-central line distances
    hist_x = (hist_edges[:-1] + hist_edges[1:]) / 2
    loc_var = np.var(hist_x * hist_counts)
    
    # Correcting the variance by subtracting the variance of the background, linker distribution, and the localization precision
    bg_var = ((hist_edges[-1] - hist_edges[0]) / (2 * np.sqrt(3)))**2
    bg_n = (len(hist_edges) - 1) * dens_func_bg_fitted
    loc_n = np.sum(hist_counts)
    loc_var_bg_var_subs = (loc_var * loc_n - bg_var * bg_n) / (loc_n - bg_n)  # Subtracting the variance of the background
    line_var = loc_var_bg_var_subs - conv_func_var
    line_std = np.sqrt(line_var)
    
    # Calculating the line diameter from the variance of corrected variance of the distances
    FWHM_from_var_Gauss = 2 * np.sqrt(2 * np.log(2)) * line_std
    FWHM_from_var_disk = 2 * np.sqrt(3) * line_std
    
    label_x_pos = [0.58]
    label_y_pos = [0.91]
    
    # Make the "normalizedSquareDifference" fitting error bin width independent
    bin_width = abs(hist_edges[-1] - hist_edges[0]) / (len(hist_edges) - 1)
    FVAL = FVAL / bin_width
    
    label_text = [
        f'FWHM fit: {dens_func_width_fitted:.4g} nm',
        f'Fit error: {FVAL:.4g}'
    ]
    
    fitted_hist_figure.text(label_x_pos[0], label_y_pos[0], '\n'.join(label_text), transform=fitted_hist_figure.transFigure)
    
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