import numpy as np
import lineDiameter_fit_tools.simpleModelFunctions as simple_model_functions
#import compound_model_functions

def calculate_model_function(sample_type, model_func_type, model_function_parameters, sampling_settings, background_flag):
    """
    This function calculates the model function (in localization density 
    unit) to be fit on the histogram (localization density) data of the
    measurement. This model function shall be convolved and binned before 
    comparing it with the histogram of the original data.
    """

    ## Model function prior to the convolution
    # Calculating the values of the selected model function prior to the convolution
    if sample_type == 'single line':
        # Single line structures with different model functions
        if model_func_type == 'Circle':
            model_func_y, _ = simple_model_functions.circle(model_function_parameters, sampling_settings)
        elif model_func_type == 'Disk':
            model_func_y, _ = simple_model_functions.disk(model_function_parameters, sampling_settings)
        elif model_func_type == 'Ring':
            #model_func_y, _ = compound_model_functions.ring(model_function_parameters, sampling_settings)
            pass
        elif model_func_type == 'Gaussian':
            model_func_y, _ = simple_model_functions.gaussian(model_function_parameters, sampling_settings)
        elif model_func_type == 'Lorentzian':
            model_func_y, _ = simple_model_functions.lorentzian(model_function_parameters, sampling_settings)
        elif model_func_type == 'Rectangular':
            model_func_y, _ = simple_model_functions.rectangle(model_function_parameters, sampling_settings)
        else:
            raise ValueError('Invalid model function was given.')
    elif sample_type == 'double lines':
        # Double line structure
        parameter_names = model_function_parameters.keys()

        model_function_parameters_1 = {}
        model_function_parameters_2 = {}

        for param_name in parameter_names:
            if len(model_function_parameters[param_name]) == 1:
                model_function_parameters_1[param_name] = model_function_parameters[param_name]
                model_function_parameters_2[param_name] = model_function_parameters[param_name]
            elif len(model_function_parameters[param_name]) == 2:
                model_function_parameters_1[param_name] = model_function_parameters[param_name][0]
                model_function_parameters_2[param_name] = model_function_parameters[param_name][1]
            else:
                raise ValueError(f'Invalid element number vector was given for the double line parameter {param_name}.')

        model_function_parameters_1['background'] = 0
        model_function_parameters_2['background'] = 0

        model_func_y_1 = calculate_model_function('single line', model_func_type, model_function_parameters_1, sampling_settings, False)
        model_func_y_2 = calculate_model_function('single line', model_func_type, model_function_parameters_2, sampling_settings, False)

        # The resultant double line structure
        model_func_y = model_func_y_1 + model_func_y_2
    elif sample_type == 'gap':
        # Rimmed gap structure
        if model_func_type == 'Gap':  # Rectangular Gap
            #model_func_y = compound_model_functions.rectangular_gap(model_function_parameters, sampling_settings)
            pass
        else:
            raise ValueError('Invalid model function was given.')
    else:
        raise ValueError('Unknown sample type')

    if background_flag:
        background_level = model_function_parameters['background']
        model_func_y += background_level

    return model_func_y