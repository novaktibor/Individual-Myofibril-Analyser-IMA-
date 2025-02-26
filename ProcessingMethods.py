from skimage.measure import label, regionprops, regionprops_table
from scipy import ndimage
from scipy.interpolate import splprep, splev
from scipy.spatial import distance
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
import tkinter as tk

####Automatic centroid detection function (MODE 0) ####

def centroid_detection(thresh_im, random_manual, mode, newStartEnd):
    """
    Function to detect objects in binary images

    :param thresh_im: the binary image where we try to found objects end centroids
    :param random_manual: Could be 'random', 'manual' or 'not' (when it is not needed)
    :param mode: it contains the processing method (Automatic or Semi_manual)
    :param newStartEnd: Could be True or False. Needed during the length points findend to creat a new start end point so the date wont start and end in the middle of a signal

    :return: ordered indexism y and x coordinates of all the found objects' ordered centroids and also the other important propertis of the obejcts in the propsall
    """


    thresh_im = morphology.binary_erosion(thresh_im)
    thresh_im = morphology.binary_dilation(thresh_im)


    # Define a margin distance from the image boundary
    margin_distance = 40  # Adjust this value as needed

    # Creating empty frames for later use
    yall = []
    xall = []
    propsall = []

    ### Centroid detecten in the binary Aactinin chanel
    image = ndimage.binary_dilation(thresh_im).astype(np.uint8)
    label_img = label(image)
    regions = regionprops(label_img) #finding objects

    # Finding obejcti n the binary image
    for props in regions:
        y0, x0 = props.centroid

        # Check if the centroid is within the margin distance from the image boundary
        if (margin_distance < x0 < image.shape[1] - margin_distance) and \
                (margin_distance < y0 < image.shape[0] - margin_distance):
            # Save the coordinates if the centroid is not close to the edge
            yall.append(y0)
            xall.append(x0)

        #Saving all the props data for later use
        propsall.append(props)


    # Choosing the sorting method to be 'Automatic' or 'Manual' (Semi_manual mode). The width mode is for displaying the width data to chose which one to be evaluated
    if mode == "Automatic":

        # Sorting the points
        ordered_Indexes = nearest_neighbor(xall, yall) #Sorting the centroid points by the nearest_neighbour and finding the shortast route to connect them
        xall_sorted = [xall[i] for i in ordered_Indexes]
        yall_sorted = [yall[i] for i in ordered_Indexes]

        xall_sorted = list(xall_sorted)
        yall_sorted = list(yall_sorted)

        Change_to_Manual = False


    elif mode == "Semi_Manual":

        ordered_Indexes, Change_to_Manual = display_and_select_centroids(thresh_im, yall, xall, random_manual, mode) #Display the found centroids and let the user the chose the order of these points



        xall_sorted = [xall[i] for i in ordered_Indexes]
        yall_sorted = [yall[i] for i in ordered_Indexes]

        xall_sorted = list(xall_sorted)
        yall_sorted = list(yall_sorted)

        if Change_to_Manual == True:  # Call the manual centroid selector which
            newStartEnd = False


    #Creatin new start and end points in the odered length points to ensure that tho histogram doesn't start or end in the middle of a peak
    if newStartEnd == True:

        # creating a new start and new last pont
        # Extract the first two points
        x0, y0 = xall_sorted[0], yall_sorted[0]
        x1, y1 = xall_sorted[1], yall_sorted[1]

        # Calculate direction vector from the first two points
        direction_vector = np.array([x1 - x0, y1 - y0])

        # Calculate the new start point
        new_start_point = np.array([x0, y0]) - direction_vector / 2

        # Extract the last two points
        x_end_1, y_end_1 = xall_sorted[-1], yall_sorted[-1]
        x_end_2, y_end_2 = xall_sorted[-2], yall_sorted[-2]

        # Calculate direction vector from the last two points
        direction_vector_end = np.array([x_end_2 - x_end_1, y_end_2 - y_end_1])

        # Calculate the new end point
        new_end_point = np.array([x_end_1, y_end_1]) - direction_vector_end / 2

        xall_sorted = [new_start_point[0]] + xall_sorted
        yall_sorted = [new_start_point[1]] + yall_sorted
        xall_sorted.append(new_end_point[0])
        yall_sorted.append(new_end_point[1])



    return yall_sorted, xall_sorted, ordered_Indexes, propsall, Change_to_Manual


# Sorting algorithm with nearest_neighbour. Getting the order of the centroids by calculating the sortest rout to connect them (With this I can find the start and end of a myofibri)
def nearest_neighbor(xall, yall):

    num_points = len(xall)
    min_total_distance = float('inf')
    best_ordered_indices = None
    for start_idx in range(num_points):
        unvisited = set(range(num_points))
        current_idx = start_idx
        ordered_indices = [current_idx]
        unvisited.remove(current_idx)
        total_distance = 0
        while unvisited:
            nearest_dist = float('inf')
            nearest_idx = None
            for idx in unvisited:
                dist = distance.euclidean((xall[current_idx], yall[current_idx]), (xall[idx], yall[idx]))
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest_idx = idx
            ordered_indices.append(nearest_idx)
            unvisited.remove(nearest_idx)
            total_distance += nearest_dist
            current_idx = nearest_idx
        if total_distance < min_total_distance:
            min_total_distance = total_distance
            best_ordered_indices = ordered_indices
    return best_ordered_indices




#### OpenCV manual selector works perfectly and fast. Only bug is during selecten after the 4 selection will you see the spline ######
def select_centroids(image):
    # Make a copy of the image

    image_uint8 = (image / np.max(image) * 255).astype(np.uint8)

    # Convert the uint8 image to BGR format (for compatibility with OpenCV)
    image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)

    # Copy the original image for overlay
    image_copy = image_bgr.copy()

    # Lists to store centroid coordinates
    xall = []
    yall = []

    # Create a window to display the image
    cv2.namedWindow('Select Centroids')
    cv2.imshow('Select Centroids', image_copy)

    # Function to draw the spline curve connecting centroids
    def draw_splineancentroid(image_copy, xall, yall):
        # Create a copy of the original image to draw the overlay
        overlay = image_copy.copy()

        # Draw centroids
        if len(xall) > 0:
            for i in range(len(xall)):
                cv2.circle(overlay, (xall[i], yall[i]), 5, (0, 255, 0), -1)

            # Draw a line or spline curve if more than two centroids are selected
            if len(xall) == 2:
                start_point = (xall[0], yall[0])
                end_point = (xall[1], yall[1])
                cv2.line(overlay, start_point, end_point, (255, 0, 0), 2)
            elif len(xall) > 2:
                tck, _ = splprep([xall, yall], s=0, k=2)
                u_new = np.linspace(0, 1, 100)
                spline_points = splev(u_new, tck)
                for i in range(len(spline_points[0]) - 1):
                    cv2.line(overlay, (int(spline_points[0][i]), int(spline_points[1][i])),
                             (int(spline_points[0][i + 1]), int(spline_points[1][i + 1])), (255, 0, 0), 2)

        cv2.imshow('Select Centroids', overlay)

    # Flag to indicate if dragging mode is active and the index of the centroid being dragged
    dragging = False
    dragging_index = -1
    # Mouse callback function to handle mouse events
    def mouse_callback(event, x, y, flags, param):
        nonlocal image_copy, xall, yall, dragging, dragging_index

        # Left mouse button click event to add centroid
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add centroid coordinates to the lists
            xall.append(x)
            yall.append(y)

            draw_splineancentroid(image_copy, xall, yall)
        # Right mouse button click event to delete centroid
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Find the closest centroid to the right-clicked point
            distances = np.sqrt((np.array(xall) - x) ** 2 + (np.array(yall) - y) ** 2)
            closest_index = np.argmin(distances)
            if distances[closest_index] < 10:
                # Delete the closest centroid
                xall.pop(closest_index)
                yall.pop(closest_index)


            draw_splineancentroid(image_copy, xall, yall)
            # Middle mouse button press event to start dragging the nearest centroid
        elif event == cv2.EVENT_MBUTTONDOWN:
            distances = np.sqrt((np.array(xall) - x) ** 2 + (np.array(yall) - y) ** 2)
            closest_index = np.argmin(distances)
            if distances[closest_index] < 10:
                dragging = True
                dragging_index = closest_index

            # Mouse movement event to drag the centroid if dragging is active
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            # Update the position of the centroid being dragged
            xall[dragging_index] = x
            yall[dragging_index] = y
            draw_splineancentroid(image_copy, xall, yall)

            # Middle mouse button release event to stop dragging
        elif event == cv2.EVENT_MBUTTONUP:
            dragging = False
            dragging_index = -1


            draw_splineancentroid(image_copy, xall, yall)


    # Function to handle "Stop Selection" button click
    def stop_selection():
        nonlocal selection_stopped
        selection_stopped = True
        root.quit()  # Close the Tkinter window


    cv2.setMouseCallback('Select Centroids', mouse_callback)

    # Initialize Tkinter for the Stop Selection button
    root = tk.Tk()
    root.title("Control Panel")

    # Create and pack the buttons
    stop_button = tk.Button(root, text="Stop Selection", command=stop_selection)
    stop_button.pack()

    # Flag to indicate if selection has stopped
    selection_stopped = False

    # Start a Tkinter loop in a separate thread
    root.update()

    # Main loop for OpenCV window
    while cv2.getWindowProperty('Select Centroids', cv2.WND_PROP_VISIBLE) > 0:
        key = cv2.waitKey(1) & 0xFF
        root.update()  # Update the Tkinter GUI

        if key == 27 or selection_stopped:  # Press Esc or stop button to exit
            break

    cv2.destroyAllWindows()
    root.quit()  # Stop Tkinter loop
    root.destroy()


    # Return centroid coordinates
    return yall, xall

def display_and_select_centroids(image, yall, xall, random_manual, mode):
    """
    Display centroids overlaying an image and allow selection of centroids.

    :param image: Original image
    :param xall: List of x coordinates of centroids
    :param yall: List of y coordinates of centroids
    :param random_manual: 'random' or 'manual'
    :return: List of selected centroid coordinates
    """

    # Random mode will choose 3 random points to be used in the width calculations
    if random_manual == 'random':
        num_points = len(xall)
        # Exclude the first and last indices
        indices = list(range(1, num_points - 1))
        # Randomly select three indices
        selected_centroids_Indexes = random.sample(indices, 3)
        return selected_centroids_Indexes, False

    # In manual mode, the user will choose as many centroids as they want
    if random_manual == 'manual':

        image_uint8 = (image / np.max(image) * 255).astype(np.uint8)

        # Convert the uint8 image to BGR format (for compatibility with OpenCV)
        image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_GRAY2BGR)

        # Copy the original image for overlay
        overlay = image_bgr.copy()

        # Draw centroids on the overlay
        for x, y in zip(xall, yall):
            cv2.circle(overlay, (int(x), int(y)), 5, (0, 0, 255), -1)

        # Create a window for display
        cv2.namedWindow('Select centroids')
        cv2.imshow('Select centroids', overlay)

        # List to store selected centroid coordinates
        selected_centroids = []
        selected_centroids_Indexes = []

        # Function to draw selected centroids
        def draw_selected_centroids(centroids):
            overlay_copy = overlay.copy()
            for x, y in centroids:
                # Draw a circle around the centroid (change color to blue)
                cv2.circle(overlay_copy, (int(x), int(y)), 5, (255, 0, 0), -1)
            cv2.imshow('Select centroids', overlay_copy)

        # Function to find the nearest centroid to a point so the user doesn't have to click precisely on the centroid
        def find_nearest_centroid(x, y):
            distances = np.sqrt((np.array(xall) - x) ** 2 + (np.array(yall) - y) ** 2)
            nearest_index = np.argmin(distances)
            return xall[nearest_index], yall[nearest_index], nearest_index

        # Function to delete the last selected point
        def delete_last_selection():
            if selected_centroids:
                selected_centroids.pop()
                selected_centroids_Indexes.pop()
                draw_selected_centroids(selected_centroids)

        # Set mouse callback function
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                # Find nearest centroid
                nearest_x, nearest_y, nearest_index = find_nearest_centroid(x, y)
                selected_centroid = (nearest_x, nearest_y)

                # Check if the selected centroid is already in the list
                if selected_centroid not in selected_centroids:
                    selected_centroids.append(selected_centroid)
                    selected_centroids_Indexes.append(nearest_index)
                    draw_selected_centroids(selected_centroids)
            elif event == cv2.EVENT_RBUTTONDOWN:  # Right mouse button clicked
                delete_last_selection()

        cv2.setMouseCallback('Select centroids', mouse_callback)

        # Initialize flags
        selection_stopped = False
        Change_to_Manual = False

        # Function to handle "Stop Selection" button click
        def stop_selection():
            nonlocal selection_stopped
            selection_stopped = True
            root.quit()  # Close the Tkinter window

        # Function to handle "Change to Manual" button click
        def change_to_manual():
            nonlocal Change_to_Manual
            Change_to_Manual = True
            root.quit()  # Close the Tkinter window

        # Initialize Tkinter for the Stop Selection button
        root = tk.Tk()
        root.title("Control Panel")

        # Create and pack the buttons
        stop_button = tk.Button(root, text="Stop Selection", command=stop_selection)
        stop_button.pack()

        if mode == "Semi_Manual":
            Change_to_Manual_button = tk.Button(root, text="Change to Manual", command=change_to_manual)
            Change_to_Manual_button.pack()

        # Start Tkinter loop in a separate thread
        root.update()

        # Wait for user to select centroids (press ESC to finish)
        while cv2.getWindowProperty('Select centroids', cv2.WND_PROP_VISIBLE) >= 1:
            key = cv2.waitKey(1) & 0xFF
            root.update()  # Update the Tkinter GUI

            if key == 27 or selection_stopped or Change_to_Manual:  # ESC key
                break

        # Close the OpenCV window
        cv2.destroyAllWindows()
        root.quit()  # Stop Tkinter loop
        root.destroy()  # Close Tkinter window

    return selected_centroids_Indexes, Change_to_Manual


# Function to test if any point is too close or too farawy from the others
def fail_safe_test (yall, xall):

    distances = []

    for i in range(1, len(yall) - 1):  # Iterate over points except the first and last one (as it is created by the program)
        distance = ((yall[i + 1] - yall[i]) ** 2 + (xall[i + 1] - xall[i]) ** 2) ** 0.5
        distances.append(distance)

    #Getting some data
    mean_distance = np.mean(distances)
    std_distance = np.std(distances)

    # Calculating the threshold to find any outlier. The threshold is the mean+- 2 times the STD of the distance
    threshold_low = mean_distance - 2 * std_distance
    threshold_high = mean_distance + 2 * std_distance

    #Finding the outliers in the data
    outliers = []
    for i in range(len(distances) - 1):  # Iterate over indices except the last one
        if distances[i] < threshold_low or distances[i] > threshold_high:
            outliers.append(i)


    return outliers
