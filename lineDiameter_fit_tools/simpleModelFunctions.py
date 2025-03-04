import numpy as np

from lineDiameter_fit_tools.model_function_sampling import boundaries as boundaries
from lineDiameter_fit_tools.model_function_sampling import points as points

def sampling_centers(sampling_settings):
    binningRegion = sampling_settings['binningRegion']
    sampling_N = sampling_settings['sampling_N']
    convRad_N= 0
    model_func_samp_boundaries_left, model_func_samp_boundaries_right, sampling_dx = boundaries(
        binningRegion, sampling_N, convRad_N
    )
    model_func_x = (model_func_samp_boundaries_left + model_func_samp_boundaries_right) / 2
    return model_func_x

def circle(model_function_settings, sampling_settings):
    radius = model_function_settings['width'] / 2
    position = model_function_settings['position']
    model_func_samp_boundaries_left, model_func_samp_boundaries_right, sampling_dx = boundaries(
        sampling_settings['binningRegion'], sampling_settings['sampling_N'], sampling_settings['convRad_N']
    )
    theta_left, theta_right, samp_x_within_bool, bound_idx_left, bound_idx_right = circle_preparation(
        radius, model_func_samp_boundaries_left, model_func_samp_boundaries_right, position
    )
    model_func_y = np.zeros_like(model_func_samp_boundaries_left)
    model_func_y[samp_x_within_bool] = radius * 2 * (theta_left[samp_x_within_bool] - theta_right[samp_x_within_bool])
    if bound_idx_left != bound_idx_right:
        model_func_y[bound_idx_left] = radius * 2 * abs(np.pi - theta_right[bound_idx_left])
        model_func_y[bound_idx_right] = radius * 2 * abs(theta_left[bound_idx_right] - 0)
    else:
        model_func_y[bound_idx_right] = radius * 2 * np.pi
    model_func_y /= (2 * radius * np.pi)
    model_func_y /= sampling_dx
    if sampling_dx * sampling_settings['binRefinement'] / 2 < radius:
        fi_central = np.arcsin(sampling_dx * sampling_settings['binRefinement'] / 2 / radius)
        central_bin_value = 2 * fi_central / np.pi
    else:
        central_bin_value = 1
    area = model_function_settings['centralValue'] / central_bin_value
    model_func_y *= area
    return model_func_y, area

def disk(model_function_settings, sampling_settings):
    radius = model_function_settings['width'] / np.sqrt(3)
    position = model_function_settings['position']
    model_func_samp_boundaries_left, model_func_samp_boundaries_right, sampling_dx = boundaries(
        sampling_settings['binningRegion'], sampling_settings['sampling_N'], sampling_settings['convRad_N']
    )
    theta_left, theta_right, samp_x_within_bool, bound_idx_left, bound_idx_right = circle_preparation(
        radius, model_func_samp_boundaries_left, model_func_samp_boundaries_right, position
    )
    model_func_y = np.zeros_like(model_func_samp_boundaries_left)
    model_func_y[samp_x_within_bool] = radius**2 * (
        (theta_left[samp_x_within_bool] - np.sin(2 * theta_left[samp_x_within_bool]) / 2) -
        (theta_right[samp_x_within_bool] - np.sin(2 * theta_right[samp_x_within_bool]) / 2)
    )
    if bound_idx_left != bound_idx_right:
        model_func_y[bound_idx_left] = radius**2 * (np.pi - (theta_right[bound_idx_left] - np.sin(2 * theta_right[bound_idx_left]) / 2))
        model_func_y[bound_idx_right] = radius**2 * (theta_left[bound_idx_right] - np.sin(2 * theta_left[bound_idx_right]) / 2)
    else:
        model_func_y[bound_idx_right] = radius**2 * np.pi
    model_func_y /= (radius**2 * np.pi)
    model_func_y /= sampling_dx
    if sampling_dx * sampling_settings['sampling_N'] / 2 < radius:
        fi_central = np.arcsin(sampling_dx * sampling_settings['sampling_N'] / 2 / radius)
        central_bin_value = 2 * fi_central / np.pi + sampling_dx * sampling_settings['sampling_N'] * radius * np.cos(fi_central)
    else:
        central_bin_value = 1
    area = model_function_settings['height'] / central_bin_value
    model_func_y *= area
    return model_func_y, area

def gaussian(model_function_settings, sampling_settings):
    position = model_function_settings['position']
    gauss_std = model_function_settings['width'] / 2.3548
    model_func_sampling_x, sampling_dx, _ = points(
        sampling_settings['binningRegion'], sampling_settings['sampling_N'], sampling_settings['convRad_N']
    )
    model_func_y = 1 / (gauss_std * np.sqrt(2 * np.pi)) * np.exp(-(model_func_sampling_x - position) ** 2 / (2 * gauss_std ** 2))
    central_bin_value = 1 / (gauss_std * np.sqrt(2 * np.pi))
    area = model_function_settings['height'] / central_bin_value
    model_func_y *= area
    return model_func_y, area

def lorentzian(model_function_settings, sampling_settings):
    position = model_function_settings['position']
    gamma = model_function_settings['width'] / 2
    model_func_sampling_x, sampling_dx, _ = points(
        sampling_settings['binningRegion'], sampling_settings['sampling_N'], sampling_settings['convRad_N']
    )
    model_func_y = 1 / (np.pi * gamma * (1 + ((model_func_sampling_x - position) / gamma) ** 2))
    central_bin_value = 1 / (np.pi * gamma) * sampling_dx * sampling_settings['binRefinement']
    area = model_function_settings['height'] / central_bin_value
    model_func_y *= area
    return model_func_y, area

def rectangle(model_function_settings, sampling_settings):
    width = model_function_settings['width']
    height = model_function_settings['height']
    position = model_function_settings['position']
    model_func_samp_boundaries_left, model_func_samp_boundaries_right, sampling_dx = boundaries(
        sampling_settings['binningRegion'], sampling_settings['sampling_N'], sampling_settings['convRad_N']
    )
    model_func_y = np.zeros_like(model_func_samp_boundaries_left)
    bound_left = -width / 2 + position
    bound_right = width / 2 + position
    model_func_y[(model_func_samp_boundaries_left >= bound_left) & (model_func_samp_boundaries_right <= bound_right)] = height
    r_bound_idx_left = np.where((model_func_samp_boundaries_left <= bound_left) & (model_func_samp_boundaries_right >= bound_left))[0]
    r_bound_idx_right = np.where((model_func_samp_boundaries_left <= bound_right) & (model_func_samp_boundaries_right >= bound_right))[0]
    if r_bound_idx_left.size > 0 and r_bound_idx_right.size > 0:
        if r_bound_idx_left[0] != r_bound_idx_right[0]:
            model_func_y[r_bound_idx_left[0]] = (
                (bound_left - model_func_samp_boundaries_right[r_bound_idx_left[0]]) /
                (model_func_samp_boundaries_left[r_bound_idx_left[0]] - model_func_samp_boundaries_right[r_bound_idx_left[0]]) * height
            )
            model_func_y[r_bound_idx_right[0]] = (
                (model_func_samp_boundaries_left[r_bound_idx_right[0]] - bound_right) /
                (model_func_samp_boundaries_left[r_bound_idx_right[0]] - model_func_samp_boundaries_right[r_bound_idx_right[0]]) * height
            )
        else:
            model_func_y[r_bound_idx_right[0]] = (
                (bound_left - bound_right) /
                (model_func_samp_boundaries_left[r_bound_idx_right[0]] - model_func_samp_boundaries_right[r_bound_idx_right[0]]) * height
            )
    function_sum = width * height
    model_func_y /= function_sum
    central_bin_value = min(sampling_dx * sampling_settings['binRefinement'], width) * height
    area = model_function_settings['height'] / central_bin_value
    model_func_y *= area
    return model_func_y, area

def circle_preparation(circle_rad, model_func_samp_boundaries_left, model_func_samp_boundaries_right, position):
    """
    This function examines which sampling points are affected by the disk or circle pattern.
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
    bound_idx_left = np.where(~samp_x_within_bool_left & samp_x_within_bool_right)[0]
    bound_idx_right = np.where(samp_x_within_bool_left & ~samp_x_within_bool_right)[0]
    
    return theta_left, theta_right, samp_x_within_bool, bound_idx_left, bound_idx_right
