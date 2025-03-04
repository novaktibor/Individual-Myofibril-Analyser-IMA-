import numpy as np
from scipy.special import erf

def calculate_convolution_function(dX, linkerRad, locPrecision, linkerType, GaussianConvSigma):
    """
    This function calculates the function with which the dye density should be
    convolved in order to compare it with the measured histogram data. This
    function consists of two terms, the localization precision which follows
    Gaussian distribution and the inaccuracy persisting because of the linker
    length. The latter one follows the projection of a sphere surface and
    convolved with a line ('linkerType=sphere') which results in a top-hat
    function or can also be set to be a Gaussian with FWHM equalling to the
    linker length ('linkerType=Gaussian'). The resultant function is the
    convolution of the aforementioned two terms. If the effect of the linker 
    into account as a Gaussian function then it calculates the convolution
    analytically instead of a numerical convolution.
    
    INPUT VARIABLES
    dX: distance of the sampling points for the convolution
    linkerRad: length of the linkers
    locPrecision: standard deviation of the localization precision, can be a vector containing values belonging to each localization
    linkerType: type of distribution associated with the linker length, can be 'sphere' or 'Gaussian'
    GaussianConvSigma: additional Gaussian smearing beside that of the localisation precision and the linker length that accounts for the uncertainties of the sample binding sites 
    
    OUTPUT VARIABLES:
    f: the convolution function with which the dye density should be convolved
    convRad_N: half the points of the convolution function minus 0.5
    """

    GaussianVariance = locPrecision**2 + GaussianConvSigma**2
    STD_max = np.sqrt(np.max(GaussianVariance))
    linkerSTD = linkerRad / (2 * np.sqrt(2 * np.log(2)))

    if linkerType == 'sphere':
        t_max = linkerRad + 3 * STD_max
    elif linkerType == 'Gaussian':
        t_max = 3 * (linkerSTD + STD_max)
    else:
        raise ValueError('Unknown linker type')
    
    t_min = -t_max

    convRad_N = int(np.floor(np.ceil((t_max - t_min) / dX) / 2))
    conv_N = 2 * convRad_N + 1

    t = np.linspace(-convRad_N * dX, convRad_N * dX, conv_N)

    if linkerType == 'sphere':
        if linkerRad != 0:
            f_array = (-erf((-linkerRad + t) / np.sqrt(2 * GaussianVariance)) + 
                       erf((linkerRad + t) / np.sqrt(2 * GaussianVariance))) / (2 * (2 * linkerRad))
        else:
            f_array = 1 / np.sqrt(2 * np.pi * GaussianVariance) * np.exp(-(t)**2 / (2 * GaussianVariance))
    elif linkerType == 'Gaussian':
        f_array = 1 / np.sqrt(2 * np.pi * (GaussianVariance + linkerSTD**2)) * np.exp(-(t)**2 / (2 * (GaussianVariance + linkerSTD**2)))
    else:
        raise ValueError('Unknown linker type')

    # Beware!!
    # the python version will work with single "locPrecision" value, and not a vector of "locPrecisions" like the Matlab version
    #f = np.mean(f_array, axis=1)
    f = f_array

    return f, convRad_N
