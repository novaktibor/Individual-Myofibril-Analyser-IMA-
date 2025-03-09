import numpy as np

#from lineDiameter_fit_tools.lineDiameter_singlerectfit import lineDiameter_singlerectfit

#from lineDiameter_fit_tools.lineDiameter_fit_visualization_single_line import single_line as lineDiameter_fit_visualization_single_line


from WidthCalculation import lineDiameter_singlerectfit

from WidthCalculation import lineDiameter_fit_visualization_plotDistribution

def rainSTORM_extras_lineDiameter_fit(model):

    lineDiam_histData_input = 1
    
    initFitVals = {
        'gap': {
        'densFuncSize_init': model['densFuncSize_init'],
        'rimWidth': model['rimWidth']
        }
        }
    precBounds = [-float('inf'), float('inf')]
    

    locPrecision = model['locPrecision'];
    
    N = 50
    edges = np.linspace(-500, 500, N+1)
    x = (edges[0:-1] + edges[1:])/2
    histHandler = {'BinCounts': np.exp(-x**2/(2*50**2)),
                   'BinEdges': edges}
    linkerType = model['linkerType']
    modelFuncType = model['modelFuncType']
    linkerRad = model['linkerRad']
    densFuncWidth_fitted, FigData = lineDiameter_singlerectfit(locPrecision, histHandler, linkerType, modelFuncType, linkerRad)
    
    histogramStruct = FigData['histogramStruct']
    fittedParameters = FigData['modelFunctionStruct']['modelFunctionSettings']
    modelFunctionStruct = FigData['modelFunctionStruct']
    densFuncWidth_fitted = FigData['densFuncWidth_fitted']
    densFuncBG_fitted = FigData['densFuncBG_fitted']
    FVAL = FigData['FVAL']
    
    #lineDiameter_fit_visualization_single_line(histogramStruct, fittedParameters, modelFunctionStruct, densFuncWidth_fitted, densFuncBG_fitted, FVAL)
    lineDiameter_fit_visualization_plotDistribution(histogramStruct, fittedParameters, modelFunctionStruct)
    
    
# Parameters for the convolution to calculate the model function to fit
model = {
    'linkerRad': 10,                    # Radius of the linker, dyes can only be positioned on sphere surfaces around the epitopes
    'multiPrecBool': True,              # Calculate the average of the Gaussian distributions derived from the individual localizations precision values
    'locPrecision': 40,                 # Given standard deviation of the localization precision, if empty or zero, it will take the average of the STDs of the localizations
    'iterGaussianConvBool': False,      # Whether the algorithm should apply additional Gaussian smearing beside the localization precision, can be used for the "gap" structure, the STD of this one is iterated
    'modelFuncType': 'Disk',        # Dye density following Gaussian distribution
    'sampleType': 'single line',        # String variable for whether the curve fitting should be executed on a double line or on a single line or on a gap
    'linkerType': 'Gaussian',           # Type of the dye distribution caused by the linker, "sphere" or "Gaussian"
    'densFuncSize_init': 130,
    'rimWidth': 50,
    'backgroundFlag': True
}

rainSTORM_extras_lineDiameter_fit(model)