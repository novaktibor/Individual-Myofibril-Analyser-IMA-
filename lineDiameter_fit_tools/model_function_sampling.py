import numpy as np

def settings(hist_edges, bin_refinement):
    """
    This function determines how the model function should be sampled based on the histogram bin coordinates of the
    original data on the bin refinement. It only determines the region that should be binned later (larger region should be
    sampled because of the convolution).
    """
    # Number of bins in the original data
    hist_bin_n = len(hist_edges) - 1
    
    # Number of bins after the refinement
    sampling_n = hist_bin_n * bin_refinement
    
    # Size of this region
    binning_region = [hist_edges[0], hist_edges[-1]]
    
    return binning_region, sampling_n

def dx(binning_region, sampling_n):
    """
    This function calculates the refined bins sizes belonging to the sampling points.
    """
    sampling_dx = (binning_region[1] - binning_region[0]) / sampling_n
    return sampling_dx

def points(binning_region, sampling_n, conv_rad_n):
    """
    This function returns the centers, the numbers, and the sizes of the sampling bins of the model function.
    They are determined from the histogram of the original data and from the radius of the convolution (in number of sampling points)
    function that the model function shall be convolved.
    """
    # Get the centers and the sizes of the sampling bins
    sampling_boundaries_left, sampling_boundaries_right, sampling_dx = boundaries(binning_region, sampling_n, conv_rad_n)
    
    # The sampling coordinates (centers of the refined bins)
    sampling_x = (sampling_boundaries_left + sampling_boundaries_right) / 2
    
    # Number of the sampling points
    sampling_extended_n = len(sampling_x)
    
    return sampling_x, sampling_dx, sampling_extended_n

def boundaries(binning_region, sampling_n, conv_rad_n):
    """
    This function returns the left and the right boundaries of the sampling bins (the regions belonging to each sampling points) of the model function.
    They are determined from the histogram of the original data and from the radius of the convolution function (in number of sampling points) that the
    model function shall be convolved.
    """
    # Prepare the sampling
    sampling_dx = dx(binning_region, sampling_n)
    sampling_boundaries_n = sampling_n + 2 * conv_rad_n + 1
    sampling_region = [binning_region[0] - conv_rad_n * sampling_dx, binning_region[1] + conv_rad_n * sampling_dx]
    sampling_boundaries = np.linspace(sampling_region[0], sampling_region[1], sampling_boundaries_n)
    
    # Calculate the bins' boundaries
    sampling_boundaries_left = sampling_boundaries[:-1]
    sampling_boundaries_right = sampling_boundaries[1:]
    
    return sampling_boundaries_left, sampling_boundaries_right, sampling_dx