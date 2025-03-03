import numpy as np

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