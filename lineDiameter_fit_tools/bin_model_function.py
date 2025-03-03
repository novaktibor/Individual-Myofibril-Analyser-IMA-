import numpy as np

def bin_model_function(model_func_y_convolved, sampling_n, bin_refinement):
    """
    This function numerically averages the localization density model
    function over the histogram bins. The histogram bins must be equally sized.
    The integration is a simple average of the model function sampling points
    belonging to the different bins.
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