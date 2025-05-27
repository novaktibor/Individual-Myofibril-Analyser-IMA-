from tkinter import *
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import WidthCalculation
import cmath





### function to save the data as an excel file in tha same folder where the images was ###
def excelsaving(AllLengthData, AllWidthData, filepath, what_to_process, number):

    path = filepath['path']
    Allfilename = filepath['Allfilename']
    num_files = len(AllLengthData)

    # Excel file path
    if number == 0:
        output_file = os.path.join(path, 'Results.xlsx')
    else:
        output_file = os.path.join(path, 'Results' + str(number) + '.xlsx')

    allpeakdistancesNM = []
    FirstSheetData = pd.DataFrame()
    SecondSheetData = pd.DataFrame()
    ThirdSheetData = pd.DataFrame()
    FourthSheetData = pd.DataFrame()

    if not os.path.exists(path):
        os.mkdir(path)

    for i, (LengthData, filename) in enumerate(zip(AllLengthData, Allfilename)):

        #################### Creating the first sheet data which contains all the length histograms with the filanmes as the indexes

        #Getting the histogram from the LengthData library
        histogramLength = LengthData['histogramLength']

        #Getting the pixel size from the library
        pixel_size = LengthData['pixel_size']

        # Create a DataFrame with the histogram data
        histogramLengthDF = pd.DataFrame(histogramLength)

        ## Separate the data to calculat the spline points into µm and raname it colums
        spline_points = histogramLengthDF['Spline_Point'] * pixel_size/1000  # Multiply by pixel size
        intensity = histogramLengthDF['Intensity']

        # Create DataFrame with modified spline points
        histogramLengthDF_modified = pd.DataFrame({'spline points': spline_points, 'Intensity': intensity})

        # Rename the columns in histogramLengthDF_modified
        histogramLengthDF_modified.columns = [filename, 'Intensity']
        # Reset the index
        histogramLengthDF_modified.reset_index(drop=True, inplace=True)

        # Concatenate the DataFrame with the existing data
        if FirstSheetData is None:
            FirstSheetData = histogramLengthDF_modified
        else:
            FirstSheetData = pd.concat([FirstSheetData, histogramLengthDF_modified], axis=1)

        #################### Creating the secound sheet data which is the peakcords with the peak distances and the mean peakdistance in NM

        gauspeaks = LengthData['gauspeaks']
        peakdistances = LengthData['peakdistances']

        # Converting the information into pandas dataframe to write excel file
        peakcordsdf = pd.DataFrame(gauspeaks)
        #print(peakdistances.shape)
        peakdistancesdf = pd.DataFrame(peakdistances)

        #Calculating the distance into NM from the pixel
        pixel_sizeµm = pixel_size/1000
        peakdistancesDiv = peakdistancesdf.mul(pixel_sizeµm)
        peakdistancesNM = pd.DataFrame(peakdistancesDiv)

        allpeakdistancesNM.append(peakdistancesNM)

        peakcordsDiv = peakcordsdf.mul(pixel_sizeµm)
        peakcordsNM = pd.DataFrame(peakcordsDiv)

        # Rename the columns
        peakdistancesNM.columns = ['Length']
        # Reset the index
        peakdistancesNM.reset_index(drop=True, inplace=True)

        # Rename the columns in histogramLengthDF_modified
        peakcordsNM.columns = [filename]
        # Reset the index
        peakcordsNM.reset_index(drop=True, inplace=True)

        SinglefileData = pd.concat([peakcordsNM, peakdistancesNM], axis=1)

        # Concatenate the DataFrame with the existing data
        if SecondSheetData is None:
            SecondSheetData = SinglefileData
        else:
            SecondSheetData = pd.concat([SecondSheetData, SinglefileData], axis=1)

    #for WidthData, filename in zip(AllWidthData, Allfilename):

    if what_to_process == "Individual L+W" or what_to_process == "Multiple Myofbiril L+w":

        for i, (WidthData, filename) in enumerate(zip(AllWidthData, Allfilename)):
            SingleSheetData = pd.DataFrame()
            SingleSheetData2 = pd.DataFrame()

            #################### Creating the third sheet data which contains all the width histograms with the filnames as indexes
            for j, (widthData) in enumerate(WidthData):

                # Getting the histogram from the LengthData library
                histogramsWidth = widthData['histogramsWidth']

                # Create a DataFrame with the histogram data
                histogramsWidthDF = pd.DataFrame(histogramsWidth)

                # Rename the columns in histogramLengthDF_modified
                histogramsWidthDF.columns = [filename + ' ' + str(j) , 'Intensity']
                # Reset the index
                histogramsWidthDF.reset_index(drop=True, inplace=True)

                # Concatenate the DataFrame with the existing data
                if SingleSheetData is None:
                    SingleSheetData = histogramsWidthDF
                else:
                    SingleSheetData = pd.concat([SingleSheetData, histogramsWidthDF], axis=1)

            # Concatenate the DataFrame with the existing data
            if ThirdSheetData is None:
                ThirdSheetData = SingleSheetData
            else:
                ThirdSheetData = pd.concat([ThirdSheetData, SingleSheetData], axis=1)


            ##################### Creating the Fourth sheet data which contains all the calculated width and the mean width

            for k, (widthData) in enumerate(WidthData):

                # Getting the histogram from the LengthData library
                CalculatedWidth = widthData['CalculatedWidth']

                # Create a DataFrame with the histogram data
                CalculatedWidthDF = pd.DataFrame({'CalculatedWidth': [CalculatedWidth]})

                # Rename the columns in histogramLengthDF_modified
                CalculatedWidthDF['Datas'] = [filename + ' ' + str(k + 1)]

                # Swap the columns by reordering them
                CalculatedWidthDF = CalculatedWidthDF[['Datas', 'CalculatedWidth']]

                # Concatenate the DataFrame with the existing data
                if SingleSheetData2 is None:
                    SingleSheetData2 = CalculatedWidthDF
                else:
                    SingleSheetData2 = pd.concat([SingleSheetData2, CalculatedWidthDF], axis=0)

            # Concatenate the DataFrame with the existing data
            if FourthSheetData is None:
                FourthSheetData = SingleSheetData2
            else:
                FourthSheetData = pd.concat([FourthSheetData, SingleSheetData2], axis=0)

    # Final export to Excel — safe batch-wise write
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
        FirstSheetData.to_excel(writer, sheet_name='LengthHistogram', index=False)
        SecondSheetData.to_excel(writer, sheet_name='LengthResults', index=False)

        if what_to_process in ["Individual L+W", "Multiple Myofbiril L+w"]:
            ThirdSheetData.to_excel(writer, sheet_name='WidthHistogram', index=False)
            FourthSheetData.to_excel(writer, sheet_name='WidthResults', index=False)


### Function to save the fitted spline, the histogram with the found peaks, gaussian fitting as an image
def imagesave(Data, image, filepath,processing_mode_var, number, mode):

    filename = filepath['filename']
    path = filepath['path']

    if mode == 'length':

        ## initializa the data to use

        max_aactinin = image
        spline_points = Data['spline_points']
        histparameterx = Data['histparameterx']
        histparametery = Data['histparametery']
        peakx = Data['peakx']
        peaky = Data['peaky']
        allactivepeakx = Data['allactivepeakx']
        allgausscord = Data['allgausscord']
        xystartendall = Data['xystartendall']

        y_startall = xystartendall['y_startall'],
        y_endall = xystartendall['y_endall'],
        x_startall = xystartendall['x_startall'],
        x_endall = xystartendall['x_endall'],
        perpendicular_lines = xystartendall['perpendicular_lines'],


        # Create a figure and two subplot areas
        fig, axs = plt.subplots(2, 2, figsize=(20, 10))

        if  processing_mode_var in ["Automatic", "Semi_Manual"]:
            propsallLength = Data['propsallLength']

            # PLot the binary image and the centroids in the first
            axs[0, 0].imshow(image, cmap=plt.cm.gray)
            for props in propsallLength:
                y0, x0 = props.centroid
                orientation = props.orientation
                minr, minc, maxr, maxc = props.bbox
                bx = (minc, maxc, maxc, minc, minc)
                by = (minr, minr, maxr, maxr, minr)

                x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
                y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
                x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
                y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length



                axs[0, 0].plot((x0, x1), (y0, y1), '-g', linewidth=0.5)
                axs[0, 0].plot((x0, x2), (y0, y2), '-r', linewidth=0.5)
                axs[0, 0].plot(x0, y0, '.g', markersize=1)
                axs[0, 0].plot(bx, by, '-b', linewidth=0.5)
                axs[0, 0].set_title('Found objects in the Length channel')

        else:
            axs[0, 0].imshow(image, cmap=plt.cm.gray)

        # Plot the original image on the Secound subplot
        axs[0, 1].imshow(max_aactinin, cmap='gray')
        axs[0, 1].axis('off')
        axs[0, 1].plot(spline_points[1][1], spline_points[0][1], 'b+', linewidth=1, label='Start point')
        axs[0, 1].plot(spline_points[1], spline_points[0], color='red', linewidth=1, alpha=.5, label='Fitted spline')

        for i in range(len(y_startall)):
            axs[0, 1].plot([x_startall[i], x_endall[i]], [y_startall[i], y_endall[i]], 'r', linewidth=1, alpha=.5)

        axs[0, 1].set_title('Fitted Spline in the Length channel')
        axs[0, 1].legend()


        # Plot the histogram on the Third subplot
        axs[1, 0].plot(histparameterx, histparametery, 'b--', label='Histogram')
        axs[1, 0].plot(peakx, peaky, 'r+', label='Peaks to process')
        axs[1, 0].set_title('Histogram and found peaks')
        axs[1, 0].legend()

        # Plot the gaussian fittings into the histogram on the Fourth subplot
        axs[1, 1].plot(histparameterx, histparametery, 'b--', label='Histogram')
        axs[1, 1].plot(allactivepeakx, allgausscord, 'r-',alpha=.5, label='Gausian curve')
        axs[1, 1].set_title('Histogram and fitted Gaussian curve')
        axs[1, 1].legend()


        # Adjust layout to prevent overlap
        plt.tight_layout()

        if number == 0:
            plt.savefig(path + '/foundpeaks' + filename + 'CentroidfindingandFittingData' + '.png', dpi=300)
        else:
            plt.savefig(path + '/foundpeaks' + filename + 'CentroidfindingandFittingData' + str(number) + '.png', dpi=300)
        # Show the plot
        #plt.show()
        plt.close('all')


    elif mode == 'width':

        ## initializa the data to use

        filename = filepath['filename']
        path = filepath['path']
        sumphall = image
        ystartall = []
        yendall = []
        xalll = []

        #  setting up the parameters
        subplotrowcount = 1  # number of rows
        b = len(Data)  # number figure data
        subplotcolumcount = b + 1

        fig, axes = plt.subplots(subplotrowcount, subplotcolumcount, figsize=(20, 10))
        fig.suptitle('Width measuring results')

        # loop through to make subplot form the individual fittings
        for i in range(b):

            Data2 = Data[i]

            subplotcolumncount = b + 1

            histogramStruct = Data2['histogramStruct']
            modelFunctionStruct = Data2['modelFunctionStruct']
            densFuncWidth_fitted = Data2['densFuncWidth_fitted']
            densFuncBG_fitted = Data2['densFuncBG_fitted']
            FVAL = Data2['FVAL']
            xystartendall = Data2['xystartendall']
            xall = Data2['xall']

            y_startall = xystartendall['y_startall']
            y_endall = xystartendall['y_endall']
            x_startall = xystartendall['x_startall']
            x_endall = xystartendall['x_endall']
            perpendicular_lines = xystartendall['perpendicular_lines']

            # Collecting all the y_startall, y_endall, xall to plot in the end the selected sarcomeres with the rectangel overlaying it
            ystartall.append(y_startall)
            yendall.append(y_endall)
            xalll.append(xall)

            # Calculating data for the plotting
            # TODO: finer sampling for the unconvolved??
            model_function_parameters = modelFunctionStruct['modelFunctionSettings']
            model_settings = {}
            model_settings['sampleType'] = modelFunctionStruct['sampleType']
            model_settings['modelFuncType'] = modelFunctionStruct['modelFuncType']
            model_settings['samplingSettings'] = modelFunctionStruct['samplingSettings']
            model_settings['convolutionSettings'] = modelFunctionStruct['convolutionSettings']
            model_settings['backgroundFlag'] = modelFunctionStruct['backgroundFlag']

            sample_x, sample_y_fitted_unconvolved, sample_y_fitted_convolved = WidthCalculation.visualization_calculate(
                histogramStruct, model_function_parameters, model_settings)

            # Passing the settings
            histCounts = histogramStruct['histCounts']
            histEdges = histogramStruct['histEdges']

            # Parameters determining the standard deviation of the convolution function
            locPrecision = modelFunctionStruct['convolutionSettings']['locPrecision']
            linkerRad = modelFunctionStruct['convolutionSettings']['linkerRad']
            GaussianConvSigma = modelFunctionStruct['modelFunctionSettings']['GaussianConvSigma']

            # Variance of the convolution function
            linkerType = modelFunctionStruct['convolutionSettings']['linkerType']
            if linkerType == 'sphere':
                linker_STD = linkerRad / np.sqrt(3)
                convFuncVar = np.mean(locPrecision) ** 2 + linker_STD ** 2 + GaussianConvSigma ** 2
            elif linkerType == 'Gaussian':
                linker_STD = linkerRad / (2 * np.sqrt(2 * np.log(2)))
                convFuncVar = np.mean(locPrecision) ** 2 + linker_STD ** 2 + GaussianConvSigma ** 2
            else:
                raise ValueError('Unknown linker type')

            # Variance of the localization-central line distances
            histX = (histEdges[0] + histEdges[1]) / 2
            locVar = np.var(histX * histCounts)

            # Calculating variance of the distances, correcting the value by subtracting the variance of the background, linker distribution and the localization precision
            bgVar = ((histEdges[-1] - histEdges[0]) / (2 * np.sqrt(3))) ** 2
            bgN = (len(histEdges) - 1) * densFuncBG_fitted
            locN = np.sum(histCounts)
            locVar_bgVar_subs = (locVar * locN - bgVar * bgN) / (
                    locN - bgN)  # subtracting the variance of the background
            lineVar = locVar_bgVar_subs - convFuncVar
            lineSTD = cmath.sqrt(lineVar)

            # Calculating the line diameter from the variance of corrected variance of the distances
            FWHM_from_var_Gauss = 2 * np.sqrt(2 * np.log(2)) * lineSTD
            FWHM_from_var_disk = 2 * np.sqrt(3) * lineSTD
            lineLength = densFuncWidth_fitted / 1000 / 0.866025403784438

            labelText = ['Width fit: {:.4g} µm'.format(lineLength),
                         # 'Width var: {:.4g}/{:.4g} nm'.format(FWHM_from_var_Gauss, FWHM_from_var_disk),
                         # 'residual: {:.4g}'.format(FVAL)
                         ]

            # Plotting the data
            modelFuncType = modelFunctionStruct['modelFuncType']
            ax = axes[i]

            ax.hist(histEdges[:-1], bins=histEdges, weights=histCounts, edgecolor='black', alpha=0.5)
            ax.plot(sample_x, sample_y_fitted_unconvolved, linewidth=1.5, color='red',
                    label='Fitted Model Function')
            ax.plot(sample_x, sample_y_fitted_convolved, color=[0.9290, 0.6940, 0.1250], linewidth=1.5,
                    label='Fitted Convolved Function')

            # Conditionally overlay extended fit if number != 0
            if number != 0:
                extended_data = Data2['extendedModelFunction']
                model_center = Data2['modelFunctionStruct']['modelFunctionSettings']['position']

                # Shift x_range to match the model's peak position
                shifted_x_range = extended_data['x_range'] + model_center

                ax.plot(shifted_x_range, extended_data['intensity'],
                        label='Extended (Same Peak)', lw=2, linestyle='--')


            ax.set_title(f'Hist of distances, {modelFuncType} fit, {linkerType} linker, {b}')
            ax.set_xlabel('distance [nm]')
            ax.set_ylabel('Pixel intensity')
            ax.text(0.02, 0.98, '\n'.join(labelText), ha='left', va='top',
                    transform=ax.transAxes)  # Fixed coordinates
            ax.legend(fontsize='small')

            if i == (b - 1):
                # Add the last subplot for image display
                ax_img = axes[i + 1]

                plt.imshow(sumphall, cmap='gray')
                for i in range(len(y_startall)):
                    ax_img.plot([x_startall[i], x_endall[i]], [y_startall[i], y_endall[i]], 'r', alpha=.5)
                ax_img.axis("off")
                if number == 0:
                    plt.savefig(path + '/' + filename + 'ChosenSarcomeresAndWidth' + '.png', dpi=300)
                else:
                    plt.savefig(path + '/' + filename + 'ChosenSarcomeresAndWidth' + str(number) + '.png', dpi=300)

                # plt.show()
                plt.close('all')


def save_data_to_multiple_sheets(output_file, sheet_name, data):
    """
    Appends or creates a sheet in an Excel workbook and writes the DataFrame to it.

    Parameters:
    - output_file: Path to the Excel file.
    - sheet_name: Name of the sheet to create or overwrite.
    - data: DataFrame to write.
    """
    try:
        if not os.path.exists(output_file):
            # If file does not exist, create and write directly
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                data.to_excel(writer, sheet_name=sheet_name, index=False)
        else:
            # Open the file in append mode without directly manipulating the book
            with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                data.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        print(f"Error while saving sheet {sheet_name}: {e}")
