
import tkinter as tk
from tkinter import Tk, Label, Entry, Button
from tkinter import filedialog
import aicsimageio
import matplotlib.pyplot as plt
from aicsimageio.readers import CziReader

import numpy as np
import LengthCalculation
import ProcessingMethods
import HistogramExtractors
import DataSaving
import os
import pathlib
import cv2





####Function to normalize the data######
def normalize_im(im):
    """
    Normalizes a given image such that the values range between 0 and 1.

    Parameters
    ----------
    im : 2d-array
        Image to be normalized.

    Returns
    -------
    im_norm: 2d-array
        Normalized image ranging from 0.0 to 1.0. Note that this is now
        a floating point image and not an unsigned integer array.
    """
    im_norm = (im - im.min()) / (im.max() - im.min())
    return im_norm

###Binary image crator for object finding##########
def dynamic_binaryimage_creator(first_ch, secound_ch ):

    if first_ch.dtype != np.uint8 and secound_ch.dtype != np.uint8:
        # Convert to 8-bit (using the may value as our scaling factor so it can convert bot 16 and 32bit to 8bit as this part is only used for obejct finding in the binary image)
        scaling_factor1 = np.max(first_ch)
        scaling_factor2 = np.max(secound_ch)

        first_ch = first_ch * (255.0 / scaling_factor1)
        secound_ch = secound_ch * (255.0 / scaling_factor2)

        # Convert the arrays to uint8
        first_ch = first_ch.astype(np.uint8)
        secound_ch = secound_ch.astype(np.uint8)

    # Apply Otsu's Thresholding with OpenCv
    _, otsu_threshold1 = cv2.threshold(first_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, otsu_threshold2 = cv2.threshold(secound_ch, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Display the results
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].imshow(first_ch, cmap='gray')
    axs[1, 0].imshow(secound_ch, cmap='gray')
    axs[0, 1].imshow(otsu_threshold1, cmap='gray')
    axs[1, 1].imshow(otsu_threshold2, cmap='gray')


    # Optionally, you can add titles to each subplot
    axs[0, 0].set_title('Original Image')
    axs[1, 0].set_title('Original Image')

    axs[0, 1].set_title("Otsu's Thresholding")
    axs[1, 1].set_title("Otsu's Thresholding")

    #plt.show()  #apply it so you can visualize the effect of the adaptive otsu thresholding
    #plt.show()
    plt.clf()


    return otsu_threshold1, otsu_threshold2


############################################################################## THE MAIN PART OF THE GUI WHICH WILL START THE PROCESS   ##########################################################


#### File selector for both *Czi, *tiff, *lif ####
def select_files():
    file_paths = filedialog.askopenfilename(multiple=True,
                                            title='Choose a file',
                                            filetypes=[("All Files", "*.*"),
                                                       ("CZI, TIFF, Lif files", "*.czi, *.tiff; *.tif; *.lif"),
                                                       ("Excel files", "*.xlsx;*.xls")])

    for file_path in file_paths:
        # Getting the path for the working folder
        file_listbox.insert(tk.END, file_path)



#### This function is called when the process button is pressed this contains the initial steps and the other function calls ####
def initial_process(filepathz, files_to_process, processing_mode_var, what_to_process, filecount, allWidthData, AllLengthData, Allfilename, readingmode, user_definedPSF):


    if filecount < 1:
        print("Selection failed, nothing to process")
    elif filecount >= 1:

        #Selecting the reader for Czi and *.tiff, *.lif
        if readingmode == "CZI":
            # reading the Czi file into multidimensional numpy array
            img = CziReader(filepathz)
        else:
            # reading LIF, TIFF, OME-TIFF files into multidimensional numpy array
            img = aicsimageio.AICSImage(filepathz)


        # checking if the pixel has the same size in both direction otherwise the evaluation is impossible
        if img.physical_pixel_sizes.Y == img.physical_pixel_sizes.X:

            # getting the pixel site from the metadate to use it later
            pixel_size = img.physical_pixel_sizes.Y # returns the Y dimension pixel size as found in the metadata
            pixel_size = pixel_size * 1000 #Convert it to nm

        else:
            raise ValueError('Invalid pixel size. Pixel is not a square')


        # Manual or automatic CH selection based on the checkbox adn input
        if manual_CH_Selection_var.get() == 1:

            lengthch = int(length_CH_var.get())
            widthch = int(width_CH_var.get())

            aactinin_ch = img.get_image_data("YXZ", C=lengthch, S=0, T=0)
            max_aactinin = np.amax(aactinin_ch, axis=2)
            sumaactinin = np.sum(aactinin_ch, axis=2)

            phallch = img.get_image_data("YXZ", C=widthch, S=0, T=0)
            max_phall = np.amax(phallch, axis=2)
            sumphall = np.sum(phallch, axis=2)


            # Creating a dnyamic threshold and apply it to create a binary image
            thresh_aactinin, thresh_Phall = dynamic_binaryimage_creator(sumaactinin, sumphall)

        # The automatic process is optimized for aActinin and phalloidin CH and it is determined by the covered areas in the binary CH
        else:

            # Determine which channel is the Aactinin
            first_channel_data = img.get_image_data("YXZ", C=0, S=0, T=0)
            secound_channel_data = img.get_image_data("YXZ", C=1, S=0, T=0)

            max_first_chanel = np.amax(first_channel_data, axis=2)  # max projection of the first channel
            sum_first_chanel = np.sum(first_channel_data, axis=2)
            max_secound_chanel = np.amax(secound_channel_data, axis=2)  # max projection of the second channel
            sum_secound_chanel = np.sum(secound_channel_data, axis=2)

            # Creating a dynamic threshold and apply it to create a binary image
            thresh_im1, thresh_im2= dynamic_binaryimage_creator(sum_first_chanel, sum_secound_chanel)

            # Calculating the area covered in each CH
            areacovered_first_ch = np.size(thresh_im1) - np.count_nonzero(thresh_im1)
            areacovered_secound_ch = np.size(thresh_im2) - np.count_nonzero(thresh_im2)

            # The phalloidin covers more and the Aactninn covers less area
            if areacovered_first_ch < areacovered_secound_ch:
                aactinin_ch = img.get_image_data("YXZ", C=1, S=0, T=0)
                max_aactinin = np.amax(aactinin_ch, axis=2)
                phallch = img.get_image_data("YXZ", C=0, S=0, T=0)
                sumphall = np.sum(phallch, axis=2)
                thresh_aactinin = thresh_im2
                thresh_Phall = thresh_im1
            else:
                aactinin_ch = img.get_image_data("YXZ", C=0, S=0, T=0)
                max_aactinin = np.amax(aactinin_ch, axis=2)
                phallch = img.get_image_data("YXZ", C=1, S=0, T=0)
                sumphall = np.sum(phallch, axis=2)
                thresh_aactinin = thresh_im1
                thresh_Phall = thresh_im2

        ################################################################### LENGTH PROCESSING ######################################################


        if  what_to_process in ["Length and Width", "Length"]:

            Change_to_SemiManual = False
            Change_to_Manual = False

            # Checking the selected processing method, Automatic (default), Semi_manual or Manual
            if processing_mode_var == "Automatic": # Automatic process

                # Checking for random or manual centroid selection for width calculation
                if random_3_width_var.get() == 1:
                    random_manual = 'random'

                else:
                    random_manual = 'manual'

                # This detects the "objects" in the binary image which is created by the otsu thresholding.
                yall, xall, selected_centroids_Indexes, propsallLength, Change_to_Manual = ProcessingMethods.centroid_detection(thresh_aactinin, random_manual = 'not', mode = "Automatic", newStartEnd = True)


                # Checking the neighbour points distances. If there is an outlier point itt will automatically go to semi-manual mode, so the user can manually correct the mistake
                outliers = ProcessingMethods.fail_safe_test(yall, xall)


                if outliers: # If there are some outlier the program should change to semi manual mode and than the width selection is always in manual mode
                    Change_to_SemiManual = True
                    random_manual = 'manual'


            if processing_mode_var == "Semi_Manual" or Change_to_SemiManual == True: #SemiManual process

                yall, xall, selected_centroids_Indexes, propsallLength, Change_to_Manual = ProcessingMethods.centroid_detection(thresh_aactinin,  random_manual='manual', mode = "Semi_Manual", newStartEnd = True)

                # Checking for random centroid selection or manual
                if random_3_width_var.get() == 1:
                    random_manual = 'random'

                else:
                    random_manual = 'manual'


            if processing_mode_var == "Manual" or Change_to_Manual == True: #Manual process
                # Call the manual centroid selector which
                yall, xall = ProcessingMethods.select_centroids(max_aactinin)


                random_manual = 'manual' # In manual the width selection should always be manual



            # Extractin the length histogram
            histogramLength, spline_points, xystartendall = HistogramExtractors.splineFitting_histogram(yall, xall, max_aactinin, interpolate_var.get(), pixel_size)

            # processing the length of the sarcomeres
            gauspeaks, peakdistances, histparameterx, histparametery, peakx, peaky, allactivepeakx, allgausscord = LengthCalculation.length_processing(
                histogramLength)

            # Collecting essential datas for later use
            LengthData = {
                'spline_points': spline_points,
                'histparameterx': histparameterx,
                'histparametery': histparametery,
                'peakx': peakx,
                'peaky': peaky,
                'allactivepeakx': allactivepeakx,
                'allgausscord': allgausscord,
                'propsallLength': propsallLength if processing_mode_var in ["Automatic", "Semi_Manual"] else None,
                'gauspeaks': gauspeaks,
                'peakdistances': peakdistances,
                'histogramLength': histogramLength,
                'xystartendall': xystartendall,
                'pixel_size': pixel_size
            }

            # Collecting all the data into one library for later use during the excel saving
            AllLengthData.append(LengthData)

            # Getting the path for the working folder to know where to save the images
            my_path = pathlib.Path(filepathz).parent.resolve()
            file_name = os.path.basename(filepathz)  # filename with extension
            filename = os.path.splitext(file_name)[0]  # filename without extension

            Allfilename.append(filename) #collecting all the filenames for the excel saving

            directory = "FittedHistAndResults" #the name of the new folder
            path = os.path.join(my_path, directory)

            if not os.path.exists(path): #Checking if the folder already exist
                os.mkdir(path) #Creating the new foldar for saving images and excel

            # Creating a library to stor the filename in path for later use in the data saving
            filepath = {
                'filename': filename,
                'path': path,
                'Allfilename': Allfilename
            }

            # Saving the length image
            DataSaving.imagesave(LengthData, max_aactinin, filepath, processing_mode_var, mode='length')

            # Creat an empty dataframe to don't get an error during the excel saving if only length is processed
            if what_to_process == "Length":
                alllWidthData = []

            ###################################################### Sarcomere selection for width processing and Width calculation #######################################

        if what_to_process == "Length and Width":


            selected_centroids_Indexes, Change_to_Manual = ProcessingMethods.display_and_select_centroids(sumphall, yall, xall, random_manual, mode = "Width")  # Display the found centroids so the user can chose which one should be evaluated in the width calculations

            alllWidthData, allFigData = HistogramExtractors.extract_width_histogram_along_line(sumphall, yall, xall,
                                                                                               selected_centroids_Indexes,
                                                                                               pixel_size, allWidthData,
                                                                                               spline_points, user_definedPSF)

            # Saving the image
            DataSaving.imagesave(allFigData, sumphall, filepath, processing_mode_var, mode='width')

            ############################################################################# SAVING DATAS INTO AN EXCEL FILE #######################################################


            DataSaving.excelsaving(AllLengthData, alllWidthData, filepath, what_to_process)

        try:
            # Successful processing
            return "Processed successfully"
        except Exception as e:
            # Processing failed
            return f"Processing failed: {str(e)}"


########################################################## THE GUI #######################################################


##### Process file buton function #####
def process_files():# Get the selected files from the listbox

    # Get the indices of the selected items
    selected_indices = file_listbox.curselection()

    # Get all values in the listbox
    all_values = file_listbox.get(0, tk.END)

    if not selected_indices:
        # No items selected, process all files
        files_to_process = all_values
    else:
        # Process only the selected files
        files_to_process = [all_values[idx] for idx in selected_indices]

    # Retrieve user-defined linker radius
    try:
        user_definedPSF = float(linkerRad_entry.get().strip()) if linkerRad_entry.get().strip() else 160
    except ValueError:
        user_definedPSF = 160  # Default value if invalid input is provided
        print("Invalid input for linker radius, defaulting to 160.")


    # Update progress label
    filecount = len(files_to_process)
    progress_label.config(text="Processing...")
    # Creating empty frame for later use
    AllLengthData = []
    allWidthData = []
    Allfilename = []
    # Iterate over each file and process it
    for file_path in files_to_process:
        # Call the initial_process function for each file
        print("Processing file: " + file_path)
        result = initial_process(file_path, files_to_process, processing_mode_var.get(), what_to_process_var.get(), filecount, allWidthData, AllLengthData, Allfilename, reading_mode_var.get(), user_definedPSF)
        # Update progress label with processing status
        if result is not None and result.startswith("Processed"):
            # Update progress label
            progress_label.config(text=result)
            # Display the file name in the listbox with color indicator
            file_index = files_to_process.index(file_path)
            file_listbox.itemconfig(file_index, {'fg': 'green', 'bg': 'white'})
        else:
            # Handle processing failure
            progress_label.config(text=f"Processing failed for file: {file_path}")
            # Display the file name in the listbox with color indicator
            file_index = files_to_process.index(file_path)
            file_listbox.itemconfig(file_index, {'fg': 'red', 'bg': 'white'})
        # Update GUI to refresh the display
        root.update()
    # Update progress label after processing all files
    progress_label.config(text="Processing finished.")


### Delete file button function #####
def delete_files():
    # Get the selected files from the listbox
    selected_indices = file_listbox.curselection()
    if not selected_indices:
        # No files are selected, delete all files from the listbox
        file_listbox.delete(0, tk.END)
    else:
        # Remove the selected files from the listbox in reverse order
        for index in reversed(selected_indices):
            file_listbox.delete(index)



# Create a Tkinter window
root = tk.Tk()
# Set the background color
root.configure(bg='lightgray') #'#FFDAB9'

root.title("Individual Myofibril Analyser")


# Create a frame for listbox
file_frame = tk.Frame(root)
file_frame.configure(bg='lightgray')
file_frame.grid(row=0, column=0, columnspan=3, padx=10, pady=5)

# Create the label and place it at the top left corner
file_label = tk.Label(file_frame, text="File List Box")
file_label.pack(side=tk.TOP, anchor='w', padx=10)

# Create a listbox to display selected files
file_listbox = tk.Listbox(file_frame, selectmode=tk.MULTIPLE, width=150)
file_listbox.pack(side=tk.TOP, padx=10, pady=5)



# Create a frame for the select, process and delete buttons
button_frame = tk.Frame(root)
button_frame.configure(bg='lightgray')
button_frame.grid(row=1, column=0, padx=10, pady=5)

# Create buttons for select, process, and delete files
select_button = tk.Button(button_frame, text="Select Files", command=select_files)
#select_button.configure(bg='#F0EAD6')
select_button.pack(side=tk.TOP, padx=10, pady=5)

process_button = tk.Button(button_frame, text="Process Files", command=process_files)
#process_button.configure(bg='#F0EAD6')
process_button.pack(side=tk.TOP, padx=10, pady=5)

delete_button = tk.Button(button_frame, text="Delete Files", command=delete_files)
#delete_button.configure(bg='#F0EAD6')
delete_button.pack(side=tk.TOP, padx=10, pady=5)

# Create a label to show processing progress
progress_label = tk.Label(button_frame, text="")
#progress_label.configure(bg='#F0EAD6')
progress_label.pack(side=tk.TOP, padx=10, pady=5)




# Create a frame for the manual Ch selector buttons
Ch_selection_frame = tk.Frame(root)
Ch_selection_frame.configure(bg='lightgray')
Ch_selection_frame.grid(row=1, column=1, padx=10, pady=5)

# Create a checkbox widget
manual_CH_Selection_var = tk.IntVar()  # Variable to store the checkbox state (0 for unchecked, 1 for checked)
manual_CH_Selection_checkbox = tk.Checkbutton(Ch_selection_frame, text="Manual Channel Selection", variable=manual_CH_Selection_var)
#manual_CH_Selection_checkbox.configure(bg='#F0EAD6')
manual_CH_Selection_checkbox.pack(side=tk.TOP, padx=10, pady=5)

# Create a label and option menu for processing mode selection
length_CH_label = tk.Label(Ch_selection_frame, text="Select Length Channel:")
#length_CH_label.configure(bg='#F0EAD6')
length_CH_label.pack(side=tk.TOP, padx=10, pady=5)

length_CH_var = tk.StringVar(root)
length_CH_var.set("Automatic")  # Default value

length_CH_option_menu = tk.OptionMenu(Ch_selection_frame, length_CH_var, 0, 1, 2)
#length_CH_option_menu.configure(bg='#F0EAD6')
length_CH_option_menu.pack(side=tk.TOP, padx=10, pady=5)

# Create a label and option menu for processing mode selection
width_CH_label = tk.Label(Ch_selection_frame, text="Select width Channel:")
#width_CH_label.configure(bg='#F0EAD6')
width_CH_label.pack(side=tk.TOP, padx=10, pady=5)

width_CH_var = tk.StringVar(root)
width_CH_var.set("Automatic")  # Default value

width_CH_option_menu = tk.OptionMenu(Ch_selection_frame, width_CH_var, 0, 1, 2)
#width_CH_option_menu.configure(bg='#F0EAD6')
width_CH_option_menu.pack(side=tk.TOP, padx=10, pady=5)


# Add label and entry for linkerRad
linkerRad_entry_Label = tk.Label(Ch_selection_frame, text="PSF:")
linkerRad_entry_Label.pack(side=tk.TOP, padx=10, pady=5)  # Pack the label

linkerRad_entry = tk.Entry(Ch_selection_frame)
linkerRad_entry.pack(side=tk.TOP, padx=10, pady=5)  # Pack the entry
linkerRad_entry.insert(0, "160")  # Set default value to 160



# Create a frame for the options (reading mode, processing mode, and checkbox)
options_frame = tk.Frame(root)
options_frame.configure(bg='lightgray')
options_frame.grid(row=1, column=2, padx=10, pady=5)

# Create a label and option menu for reading mode selection
reading_mode_label = tk.Label(options_frame, text="Reading file type:")
#reading_mode_label.configure(bg='#F0EAD6')
reading_mode_label.pack(side=tk.TOP, padx=10, pady=5)

reading_mode_var = tk.StringVar(root)
reading_mode_var.set("CZI")  # Default value

reading_mode_option_menu = tk.OptionMenu(options_frame, reading_mode_var, "CZI", "TIFF_LIF")
#reading_mode_option_menu.configure(bg='#F0EAD6')
reading_mode_option_menu.pack(side=tk.TOP, padx=10, pady=5)

# Create a label and option menu for processing mode selection
what_to_process_label = tk.Label(options_frame, text="Select what to process :")
#what_to_process_label.configure(bg='#F0EAD6')
what_to_process_label.pack(side=tk.TOP, padx=10, pady=5)

what_to_process_var = tk.StringVar(root)
what_to_process_var.set("Length and Width")  # Default value

what_to_process_menu = tk.OptionMenu(options_frame, what_to_process_var, "Length and Width", "Length")
#what_to_process_menu.configure(bg='#F0EAD6')
what_to_process_menu.pack(side=tk.TOP, padx=10, pady=5)

# Create a label and option menu for processing mode selection
processing_mode_label = tk.Label(options_frame, text="Select Processing Mode:")
#processing_mode_label.configure(bg='#F0EAD6')
processing_mode_label.pack(side=tk.TOP, padx=10, pady=5)

processing_mode_var = tk.StringVar(root)
processing_mode_var.set("Automatic")  # Default value

processing_mode_option_menu = tk.OptionMenu(options_frame, processing_mode_var, "Automatic", "Semi_Manual", "Manual")
#processing_mode_option_menu.configure(bg='#F0EAD6')
processing_mode_option_menu.pack(side=tk.TOP, padx=10, pady=5)

# Create a checkbox widget
random_3_width_var = tk.IntVar()  # Variable to store the checkbox state (0 for unchecked, 1 for checked)
random_3_width_checkbox = tk.Checkbutton(options_frame, text="Randomly 3 Width", variable=random_3_width_var)
#random_3_width_checkbox.configure(bg='#F0EAD6')
random_3_width_checkbox.pack(side=tk.TOP, padx=10, pady=5)

# Create a checkbox widget
interpolate_var = tk.IntVar()  # Variable to store the checkbox state (0 for unchecked, 1 for checked)
interpolate_checkbox = tk.Checkbutton(options_frame, text="Interpoalte during averaging", variable=interpolate_var)
#interpolate_checkbox.configure(bg='#F0EAD6')
#interpolate_checkbox.pack(side=tk.TOP, padx=10, pady=5)

# Run the Tkinter event loop
root.mainloop()
