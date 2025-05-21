from scipy.signal import savgol_filter
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import pandas as pd
import numpy as np



def length_processing(histparameters):

    ##### Gauss function for fitting ####
    # Gaussian + baseline
    def Gauss(x, a, mean, sigma, b):
        return a * np.exp(-((x - mean) ** 2) / (2 * sigma ** 2)) + b

    # Empty DataFrames to collect data for later use during the image creation
    allactivepeakx = pd.DataFrame()
    allgausscord = pd.DataFrame()

    # convert the x and y parameters to 1D array and getting rid of the Nan datas
    histparameterx = histparameters.iloc[:, [0, ]].values.flatten()
    histparameterx = histparameterx[~np.isnan(histparameterx)]
    histparametery = histparameters.iloc[:, [1, ]].values.flatten()
    histparametery = histparametery[~np.isnan(histparametery)]

    # Filtering the data and searcing for the max value (later used for height of the peak) to make an easier peak finding

    histparametery_max = max(histparametery)
    histparametery_min = min(histparametery)
    height_treshold = histparametery_max * 0.3 + histparametery_min

    # Finding peaks based on the given data
    peaks, _ = find_peaks(histparametery, distance=30, height=height_treshold)  # finding peaks
    peaklast = peaks[-1:]
    peakx = histparameters.iloc[peaks, [0, ]].values.flatten()
    peaky = histparameters.iloc[peaks, [1, ]].values.flatten()
    peakcount = len(peaks)
    activepeak = 0

    # Looping through all the peaks in the data
    while (activepeak < peakcount):

        # Calculating a rough peak distance in index to determine the starting and ending index of the peak
        activepeak_1 = activepeak + 1

        if activepeak < peakcount - 1:
            halfpeakdistance = (peaks[activepeak_1] - peaks[activepeak]) / 2
        elif activepeak < peakcount:  # The ending index of the last peak in every column is the last index
            halfpeakdistance = len(histparametery) - peaks[activepeak]

        # Setting the starting histogram index. The first peak starting index is always a 0
        if halfpeakdistance <= peaks[activepeak]:
            peakstart = peaks[activepeak] - halfpeakdistance
            peakstart = round(peakstart)
        elif halfpeakdistance > peaks[activepeak]:
            peakstart = 0

        # Setting the ending histogram index. The last peak ending index is the last index
        if activepeak == peakcount:
            peakend = len(histparametery)
        elif activepeak < peakcount:
            peakend = peaks[activepeak] + halfpeakdistance
            peakend = round(peakend)

        # Separating the peaks to make individual gauss fits

        peakindexes = range(peakstart, peakend, +1)
        activepeakx = histparameters.iloc[peakindexes, [0, ]].values.flatten()
        activepeaky = histparameters.iloc[peakindexes, [1, ]].values.flatten()

        # Gaussin fitting
        # Initial guesses
        amplitude_guess = max(activepeaky) - np.min(activepeaky)
        mean_guess = activepeakx[np.argmax(activepeaky)]
        sigma_guess = (max(activepeakx) - min(activepeakx)) / 4
        baseline_guess = np.min(activepeaky)

        p0 = [amplitude_guess, mean_guess, sigma_guess, baseline_guess]

        # Bounds for fitting: ([lower bounds], [upper bounds])
        bounds = (
            [0, min(activepeakx), 0, 0],
            # Lower bounds: positive amplitude, reasonable mean, positive sigma, non-negative baseline
            [np.inf, max(activepeakx), np.inf, np.max(activepeaky)]  # Upper bounds
        )

        try:
            # Fit the Gaussian with baseline
            popt, pcov = curve_fit(Gauss, activepeakx, activepeaky, p0=p0, bounds=bounds, maxfev=5000)

        except RuntimeError:
            print("Error - curve_fit failed")

        # Saving the peaks and peak distances (sarcomere length) in pandas dataframe and calculating the peak distances

        gauspeak = np.array(popt[1])

        if (activepeak == 0):
            gauspeaks = np.array(popt[1])

            # Saving the first peak cordinates
            peak1 = gauspeak
        elif (activepeak == 1):
            gauspeaks = np.concatenate([gauspeaks.reshape(1), gauspeak.reshape(1)], axis=0)

            # calculating the peak distance and saving it in an array
            peak2 = gauspeak
            peakdistance = peak2 - peak1
            peak1 = gauspeak
            peakdistances = np.array(peakdistance)
        else:
            gauspeaks = np.concatenate([gauspeaks, gauspeak.reshape(1)], axis=0)

            # calculating the peak distances and saving it in an array
            peak2 = gauspeak
            peakdistance = peak2 - peak1
            peakdistance = np.array(peakdistance)
            peak1 = gauspeak
            if (activepeak == 2):
                peakdistances = np.concatenate([peakdistances.reshape(1), peakdistance.reshape(1)], axis=0)
            else:
                peakdistances = np.concatenate([peakdistances, peakdistance.reshape(1)], axis=0)

        activepeakx = pd.DataFrame(activepeakx)
        gausscord = pd.DataFrame(Gauss(activepeakx, *popt))

        allactivepeakx = pd.concat([allactivepeakx, activepeakx], ignore_index=True)
        allgausscord = pd.concat([allgausscord, gausscord], ignore_index=True)

        # Setting up the next loop active peak
        activepeak = activepeak + 1


    return gauspeaks, peakdistances, histparameterx, histparametery, peakx, peaky, allactivepeakx, allgausscord