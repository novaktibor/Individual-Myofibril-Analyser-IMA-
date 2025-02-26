from scipy.interpolate import splprep, splev
from scipy.stats import linregress
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import WidthCalculation
from scipy.ndimage import map_coordinates

def perpendicular_histogram_creator(selected_x, selected_y, i, spline_points, image, perpendicular_lines,y_startall,y_endall,x_startall, x_endall, line_length):
    centroid_x = selected_x[i]
    centroid_y = selected_y[i]

    # Find the index of the selected centroid in the spline_points array
    # Find the index of the selected centroid in the spline_points array
    closest_idx = np.argmin(
        np.sqrt((spline_points[1] - centroid_x) ** 2 + (spline_points[0] - centroid_y) ** 2))

    # Find the previous and next points in the spline_points
    prev_point_idx = closest_idx - 1
    next_point_idx = closest_idx + 1

    prev_point = spline_points[:, prev_point_idx]
    next_point = spline_points[:, next_point_idx]

    # Calculate the slope of the line passing through the previous and next points
    line_slope = (prev_point[0] - next_point[0]) / (prev_point[1] - next_point[1])

    # Calculate the perpendicular slope
    perpendicular_slope = -1 / line_slope


    # Calculate angle in radians
    angle_radians = math.atan(perpendicular_slope)

    x1 = centroid_x + line_length * math.cos(angle_radians)
    y1 = centroid_y + line_length * math.sin(angle_radians)

    x2 = centroid_x - line_length * math.cos(angle_radians)
    y2 = centroid_y - line_length * math.sin(angle_radians)

    # Plot the image or whatever you have
    # plt.imshow(image, cmap='gray')  # Replace your_image with your actual image data

    # Plot the line segment
    # plt.plot([prev_point[1], next_point[1]], [prev_point[0], next_point[0]], color='green')
    # plt.plot([x1, x2], [y1, y2], color='red')

    # Plot the centroid point
    # plt.plot(centroid_x, centroid_y, marker='o', markersize=10, color='blue')

    # Show the plot
    # plt.show()

    # Fit a line through the points (x1, y1) and (x2, y2) using linear regression
    line_points = np.array([[x1, y1], [x2, y2]])
    slope, intercept, _, _, _ = linregress(line_points)

    # Calculate the length of the line
    line_length2 = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Generate coordinates along the line
    num_points = int(np.ceil(line_length2))  # Round up to ensure at least one point per pixel
    x_coords = np.linspace(x1, x2, num_points)
    y_coords = np.linspace(y1, y2, num_points)

    # Get image dimensions
    image_height, image_width = image.shape

    valid_x_coords = []
    valid_y_coords = []

    # Validate points within image boundaries
    for i in range(len(x_coords)):

        x = x_coords[i]  # Extract x-coordinate
        y = y_coords[i]  # Extract y-coordinate

        # Check if the point is within the image boundaries
        if 0 <= x < image_width - 1 and 0 <= y < image_height - 1:
            # Append the valid point to the list of valid spline points
            valid_y_coords.append(y)  # Append y-coordinate
            valid_x_coords.append(x)  # Append x-coordinate

    x_coords = valid_x_coords
    y_coords = valid_y_coords
    num_points = len(x_coords)

    # Empty
    line_histogram = []
    xall = []
    iall = []

    # Iterate over spline points
    for i in range((num_points) - 1):
        # Current point
        y = y_coords[i]
        x = x_coords[i]

        # Previous and next points for direction
        y_prev = y_coords[i - 1]
        x_prev = x_coords[i - 1]
        y_next = y_coords[i + 1]
        x_next = x_coords[i + 1]

        # Direction vector (average of previous and next segment)
        dy = (y_next - y_prev) / 2.0
        dx = (x_next - x_prev) / 2.0

        # Perpendicular direction vector (normalized)
        length = np.sqrt(dx ** 2 + dy ** 2)
        pdx = -dy / length
        pdy = dx / length

        # Calculate the perpendicular line coordinates
        y_start = y - 5 * pdy
        y_end = y + 5 * pdy
        x_start = x - 5 * pdx
        x_end = x + 5 * pdx

        # Ensure coordinates are within image boundaries (using floor and ceiling to account for floating points)
        y_start = np.clip(y_start, 0, image.shape[0] - 1)
        y_end = np.clip(y_end, 0, image.shape[0] - 1)
        x_start = np.clip(x_start, 0, image.shape[1] - 1)
        x_end = np.clip(x_end, 0, image.shape[1] - 1)

        # Get points along the perpendicular line
        line_points = bresenham_line(int(round(x_start)), int(round(y_start)), int(round(x_end)), int(round(y_end)))
        perpendicular_lines.append(line_points)

        # Extract intensity values along the line
        intensity_values = [image[point[1], point[0]] for point in line_points]

        # Calculate the average intensity
        intensity_average = np.mean(intensity_values)

        # Append the average intensity to the list
        line_histogram.append(intensity_average)


        if line_length != 75:

            y_startall.append(y_start)
            y_endall.append(y_end)
            x_startall.append(x_start)
            x_endall.append(x_end)

            iall.append(i)
            xall.append(x)



    return y_startall, y_endall, x_startall,x_endall,iall, xall, line_histogram, perpendicular_lines

# extract histogram for width calculation
def extract_width_histogram_along_line(image, Centroidyall, Centroidxall, selected_centroids_Indexes, pixel_size, allWidthData, spline_points, user_definedPSF):
    allFigData = []
    sallWidthData = []

    # Convert spline_points to a NumPy array
    spline_points = np.array(spline_points)

    # Assuming you have selected_centroids_Indexes, xall, yall, and spline_points
    # Extract the x and y coordinates of the selected centroids
    selected_x = [Centroidxall[i] for i in selected_centroids_Indexes]
    selected_y = [Centroidyall[i] for i in selected_centroids_Indexes]

    y_startall = []
    y_endall = []
    x_startall = []
    x_endall = []
    perpendicular_lines = []



    # Loop through each selected centroid
    for i in range(len(selected_centroids_Indexes)):

        line_length = 75
        #Initial perpendicular line creation
        y_startall, y_endall, x_startall, x_endall, iall, xall, line_histogram, perpendicular_lines = perpendicular_histogram_creator(selected_x, selected_y,
                                                                                                 i, spline_points,
                                                                                                 image,
                                                                                                 perpendicular_lines,
                                                                                                 y_startall, y_endall,
                                                                                                 x_startall, x_endall,
                                                                                                 line_length)

        #Estimating the width of sarcomeres in pixels
        max_value = max(line_histogram)
        threshold = 0.3 * max_value
        count_higher_than_threshold = sum(1 for value in line_histogram if value > threshold)

        line_length = count_higher_than_threshold*2

        #Recrating the perpendicular line with the recalculated line length
        y_startall, y_endall, x_startall, x_endall, iall, xall, line_histogram, perpendicular_lines = perpendicular_histogram_creator(selected_x, selected_y,
                                                                                                 i, spline_points,
                                                                                                 image,
                                                                                                 perpendicular_lines,
                                                                                                 y_startall, y_endall,
                                                                                                 x_startall, x_endall,
                                                                                                 line_length)

        # Calculate iall using pixel size
        iall = [x * pixel_size for x in iall]

        # Convert the list to a NumPy array
        iall = np.array(iall)
        # Perform the division
        iallµm = iall / 1000

        # creating a data to calculate the width of the chosen sarcomeres
        df_intensity2 = np.column_stack((iall, line_histogram))
        histogramµm = np.column_stack((iallµm, line_histogram))

        # Calling the width calculation function
        lineLength, FigData = WidthCalculation.processStart(df_intensity2, user_definedPSF)

        xystartendall = {
            'y_startall': y_startall,
            'y_endall': y_endall,
            'x_startall': x_startall,
            'x_endall': x_endall,
            'perpendicular_lines': perpendicular_lines
        }



        # Saving some data to creat fugires for saving
        FigData['xystartendall'] = xystartendall
        FigData['xall'] = xall
        FigData['perpendicular_lines']=perpendicular_lines

        allFigData.append(FigData)  # Collecting the datas for creating figures later

        WidthData = {
            'histogramsWidth': histogramµm,
            'CalculatedWidth': lineLength
        }
        sallWidthData.append(WidthData)



    allWidthData.append(sallWidthData)

    return allWidthData, allFigData

##### This part fits the spline to the manually or automaticly chosen centroids than extract the hist profile as a 10 pixel average ##########
def bresenham_line(x0, y0, x1, y1):
    """Bresenham's line algorithm to generate points on a line."""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = err * 2
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points

# Function to interpolate intensity along a line
def interpolate_intensity_along_line(image, x0, y0, x1, y1, num_points=10):
    x_values = np.linspace(x0, x1, num_points)
    y_values = np.linspace(y0, y1, num_points)
    intensities = map_coordinates(image, [y_values, x_values], order=1)
    return x_values, y_values, intensities
def splineFitting_histogram(yall, xall, max_aactinin, interpolate, pixel_size):
    # Let's say you have yall and xall containing the y and x coordinates respectively
    image_with_lines = max_aactinin.copy

    # Connect the centroid coordinates to form a spline
    tck, u = splprep([yall, xall], s=0)

    # Evaluate the spline to get points along the curve
    u_new = np.linspace(0, 1, num=1000)
    spline_points = splev(u_new, tck)

    # Calculate the length of the spline in pixels
    def calculate_spline_length(spline_points):
        x_points, y_points = spline_points
        distances = []
        for i in range(1, len(x_points)):
            distance = np.sqrt((x_points[i] - x_points[i - 1]) ** 2 + (y_points[i] - y_points[i - 1]) ** 2)
            distances.append(distance)
        return sum(distances)

    spline_length = calculate_spline_length(spline_points)
    num_pixels = int(np.ceil(spline_length)) # Round up to ensure at least one point per pixel

    #calculate spline length difference
    diff_length = num_pixels/spline_length


    # Evaluate the spline with one point to every pixel
    u_new = np.linspace(0, 1, num=num_pixels)
    spline_points = splev(u_new, tck)





    # Get image dimensions
    image_height, image_width = max_aactinin.shape

    valid_spline_points = [[], []]

    # Validate spline points within image boundaries
    for i  in range(len(spline_points[0])):

        x = spline_points[1][i] # Extract x-coordinate
        y = spline_points[0][i] # Extract y-coordinate

        # Check if the point is within the image boundaries
        if 0 <= x < image_width-1 and 0 <= y < image_height-1:
            # Append the valid point to the list of valid spline points
            valid_spline_points[0].append(y)  # Append y-coordinate
            valid_spline_points[1].append(x)  # Append x-coordinate

    spline_points = valid_spline_points


    # Create empty list to store intensity profiles
    intensity_profiles = []
    y_startall = []
    y_endall = []
    x_startall = []
    x_endall = []
    xall = []
    perpendicular_lines = []

    intensity_profiles_original = []
    intensity_profiles_interpolated = []
    all_original_points = []
    all_interpolated_points = []


    # Iterate over spline points
    for i in range(1, len(spline_points[0]) - 1):
        # Current point
        y = spline_points[0][i]
        x = spline_points[1][i]

        # Previous and next points for direction
        y_prev = spline_points[0][i - 1]
        x_prev = spline_points[1][i - 1]
        y_next = spline_points[0][i + 1]
        x_next = spline_points[1][i + 1]

        # Direction vector (average of previous and next segment)
        dy = (y_next - y_prev) / 2.0
        dx = (x_next - x_prev) / 2.0

        # Perpendicular direction vector (normalized)
        length = np.sqrt(dx ** 2 + dy ** 2)
        pdx = -dy / length
        pdy = dx / length


        # Calculate the perpendicular line coordinates
        y_start = y - 5 * pdy
        y_end = y + 5 * pdy
        x_start = x - 5 * pdx
        x_end = x + 5 * pdx


        # Ensure coordinates are within image boundaries (using floor and ceiling to account for floating points)
        y_start = np.clip(y_start, 0, max_aactinin.shape[0] - 1)
        y_end = np.clip(y_end, 0, max_aactinin.shape[0] - 1)
        x_start = np.clip(x_start, 0, max_aactinin.shape[1] - 1)
        x_end = np.clip(x_end, 0, max_aactinin.shape[1] - 1)

        if interpolate == 0:
            # Original intensity values along the line using Bresenham's line algorithm
            line_points = bresenham_line(int(x_start), int(y_start), int(x_end), int(y_end))
            intensity_values_original = [max_aactinin[point[1], point[0]] for point in line_points]
            intensity_average_original = np.mean(intensity_values_original)

            intensity_profiles_original.append(intensity_average_original)
            intensity_profiles.append(intensity_average_original)



            print("noInterpolate")

        else:
            # Interpolated intensity values along the line
            x_interp, y_interp, intensity_values_interpolated = interpolate_intensity_along_line(max_aactinin, x_start,
                                                                                             y_start, x_end, y_end)
            intensity_average_interpolated = np.mean(intensity_values_interpolated)

            intensity_profiles_interpolated.append(intensity_average_interpolated)
            intensity_profiles.append(intensity_average_interpolated)
            print("Interpolate")

            # Visualization of Bresenham lines with alternating colors
        color = 'r' if i % 2 == 0 else 'g'  # Red for even, Green for odd
        line_x, line_y = zip(*line_points)
        plt.plot(line_x, line_y, color + '-')



        perpendicular_lines.append((x_start, y_start, x_end, y_end))


        # Append some data into lists
        y_startall.append(y_start)
        y_endall.append(y_end)
        x_startall.append(x_start)
        x_endall.append(x_end)
        xall.append(x)




    xystartendall = {
        'y_startall': y_startall,
        'y_endall': y_endall,
        'x_startall': x_startall,
        'x_endall': x_endall,
        'perpendicular_lines': perpendicular_lines
    }


    # Create a Pandas DataFrame to store the intensity data
    df_intensity = pd.DataFrame({'Spline_Point': range(len(spline_points[0])-2), 'Intensity': intensity_profiles})

    return df_intensity, spline_points, xystartendall