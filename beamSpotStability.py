# I24 Lancelot Beam Spot Stability Analysis
# v2.0
# DAWN v2.3.1, Python 2.7.12 :: Anaconda 1.7.0 (64-bit)
# Hassan Chagani 2017-02-13 10:40 DLS
# Incorporated spatial calibration: set calFlag = True to use
# Fixed bug with determining position of centroid
# v1.0
# DAWN v2.3.1, Python 2.7.12 :: Anaconda 1.7.0 (64-bit)
# Hassan Chagani 2017-02-02 12:15 DLS
# Plot total counts, and mean x- and y-positions as functions of time from Lancelot images stored in
# hdf5 files. Allow user input to look at images in detail (image as 2D histogram, distribution of mean
# x- and y-positions, and x- and y-profiles at their opposing means) by specifying rough time frame.

import h5py
import numpy as np
import scisoftpy as dnp
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.optimize import curve_fit
from decimal import Decimal

# Definition of Gaussian function to fit mean x- and y-profiles
def GaussFunc(x, maxC, mu, sigma):
    return maxC*np.exp(-((x-mu)*(x-mu))/(2*sigma*sigma))

# Edit lines from here...
nImages = 100   # Number of images (hdf5 files)
expTime = 0.2   # Exposure time
acqTime = 10.   # Acquisition time
minReadOutTime = 0.00041   # Do not edit: minimum readout time for 6-bit depth
firstFileIndex = 120321   # First file index

# File path and name handling
filePath = "/dls/detectors/data/2016/lancelot/i24/lancelot_data/170110/beamExp0-2s/"
filePrefix = "i24Lancelot_"
fileSuffix = ".h5"

# Calibration
calFlag = True   # set to True to convert pixels to um using resolutions below
pixelXRes = 3.09   # x-position resolution in um/pixel
pixelYRes = 1.59   # y position resolution in um/pixel
# ... to here

dataSetName = "entry/instrument/detector/data"   # Data stored in this folder in hdf5 file

# If acquisition time is lower than exposure time + minimum readout time, correct for this
if acqTime < (expTime+minReadOutTime):
    acqTime = expTime + minReadOutTime

# Empty numpy arrays for data handling
beamIntensity = np.empty([nImages])
xMean = np.empty([nImages])
yMean = np.empty([nImages])
timeFrame = np.empty([nImages])

# Loop over all hdf5 files
for image in range(nImages):
    
    # Filename handling
    fileName = filePath + filePrefix + str(firstFileIndex+image) + fileSuffix
    dataFile = h5py.File(fileName,"r")   # Open in read-only mode
    
    # Read data from file and write to arrays
    dataSet = dataFile.get(dataSetName)
    yOut = dataSet.shape[0]
    xOut = dataSet.shape[1]
    dataOut = np.array(dataSet[:,:])
    beamIntensity[image] = dataOut.sum()
    timeFrame[image] = image * acqTime
    
    # x- and y-Profiles: fit to data
    xProfile = np.average(dataOut, axis=0)
    xBins = np.arange(xOut)
    if calFlag == True:
        xBins = xBins.astype(float)
        for pixel in range(len(xBins)):
            xBins[pixel] = (xBins[pixel] - (xOut / 2)) * pixelXRes
    popt, pconv = curve_fit(GaussFunc, xBins, xProfile, p0=[xProfile.max(),xProfile.argmax(),10.])
    xMean[image] = popt[1]
    yProfile = np.average(dataOut, axis=1)
    yBins = np.arange(yOut)
    if calFlag == True:
        yBins = yBins.astype(float)
        for pixel in range(len(yBins)):
            yBins[pixel] = (yBins[pixel] - (yOut / 2)) * pixelYRes
    popt, pconv = curve_fit(GaussFunc, yBins, yProfile, p0=[yProfile.max(),yProfile.argmax(),10.])
    yMean[image] = popt[1]

# Plot data in dawn in plot window named 'Overlay'
plotName = "Overlay"
dnp.plot.clear(plotName)
if calFlag == True:
    dnp.plot.line({"Time [s]":timeFrame},[{"Total Counts":(beamIntensity,"Total Counts")},
                                          {("x-Position [um]","right"):(xMean,"Mean x-Position")},
                                          {("y-Position [um]","right"):(yMean,"Mean y-Position")}],
                  title="Total Counts, and Mean x- and y-Positions as functions of Time",name=plotName)
else:
    dnp.plot.line({"Time [s]":timeFrame},[{"Total Counts":(beamIntensity,"Total Counts")},
                                          {("x-Position [pixel]","right"):(xMean,"Mean x-Position")},
                                          {("y-Position [pixel]","right"):(yMean,"Mean y-Position")}],
                  title="Total Counts, and Mean x- and y-Positions as functions of Time",name=plotName)

# Function using matplotlib to show images, mean positions and centroids
# Takes input from user via function ImageViewerMenu()
def PlotFigure(val,fileIndex,filePath,filePrefix,fileSuffix,firstFileIndex,dataSetName,calFlag):
    
    # Filename string handling
    fileName = filePath + filePrefix + str(fileIndex) + fileSuffix
    dataFile = h5py.File(fileName,"r")   # Open in read-only mode
    
    # Read data from file and write to arrays
    dataSet = dataFile.get(dataSetName)
    yOut = dataSet.shape[0]
    xOut = dataSet.shape[1]
    dataOut = np.array(dataSet[:,:])
    xBins = np.arange(xOut)
    if calFlag == True:
        xBins = xBins.astype(float)
        for pixel in range(len(xBins)):
            xBins[pixel] = (xBins[pixel] - (xOut / 2.)) * pixelXRes
    yBins = np.arange(yOut)
    if calFlag == True:
        yBins = yBins.astype(float)
        for pixel in range(len(yBins)):
            yBins[pixel] = (yBins[pixel] - (yOut / 2)) * pixelYRes
    xProfile = np.average(dataOut, axis=0)
    yProfile = np.average(dataOut, axis=1)
    
    # Define figure and grid spacing
    fig = plt.figure(figsize=(4.*4,4.*2))
    gs = gridspec.GridSpec(2,4)
    
    # 2D histogram of image
    ax1 = fig.add_subplot(gs[:,:2])
    im1 = ax1.imshow(dataOut,extent=[xBins[0],xBins[-1],yBins[-1],yBins[0]])
    fig.colorbar(im1)
    ax1.set_title(str(val)+" s\n"+filePrefix+str(fileIndex))
    if calFlag == True:
        ax1.set_xlabel("X [um]")
        ax1.set_ylabel("Y [um]")
    else:
        ax1.set_xlabel("X [pixel]")
        ax1.set_ylabel("Y [pixel]")
    
    # Mean x-Position distribution and fit
    ax2 = fig.add_subplot(gs[0,2])
    poptX, pconvX = curve_fit(GaussFunc, xBins, xProfile, p0=[xProfile.max(),xProfile.argmax(),10.])
    perrX = np.absolute(pconvX.diagonal()**0.5)
    ax2.hist(xBins,bins=xOut,weights=xProfile,histtype="step")
    gaussX = np.linspace(xBins[0], xBins[-1], len(xBins)*5)
    ax2.plot(gaussX, GaussFunc(gaussX, *poptX))
    ax2.set_title("Mean x-Position")
    if calFlag == True:
        ax2.set_xlabel("X [um]")
    else:
        ax2.set_xlabel("X [pixel]")
    ax2.set_ylabel("Mean Counts")
    ax2.text(0.95,0.95,"{:}{:.2f}".format("Mean = ",poptX[1]),verticalalignment="top",
             horizontalalignment="right",transform=ax2.transAxes,color="green",fontsize=12)
    ax2.text(0.95,0.9,"{:}{:.2f}".format("+/- ",perrX[1]),verticalalignment="top",
             horizontalalignment="right",transform=ax2.transAxes,color="green",fontsize=12)
    ax2.text(0.95,0.82,"{:}{:.2f}".format("Sigma = ",poptX[2]),verticalalignment="top",
             horizontalalignment="right",transform=ax2.transAxes,color="green",fontsize=12)
    ax2.text(0.95,0.77,"{:}{:.2f}".format("+/- ",perrX[2]),verticalalignment="top",
             horizontalalignment="right",transform=ax2.transAxes,color="green",fontsize=12)
    
    # Mean y-Position distribution and fit
    ax3 = fig.add_subplot(gs[0,3])
    poptY, pconvY = curve_fit(GaussFunc, yBins, yProfile, p0=[yProfile.max(),yProfile.argmax(),10.])
    perrY = np.absolute(pconvY.diagonal()**0.5)
    ax3.hist(yBins,bins=yOut,weights=yProfile,histtype="step")
    gaussY = np.linspace(yBins[0], yBins[-1], len(yBins)*5)
    ax3.plot(gaussY, GaussFunc(gaussY, *poptY))
    ax3.set_title("Mean y-Position")
    if calFlag == True:
        ax3.set_xlabel("Y [um]")
    else:
        ax3.set_xlabel("Y [pixel]")
    ax3.set_ylabel("Mean Counts")
    ax3.text(0.95,0.95,"{:}{:.2f}".format("Mean = ",poptY[1]),verticalalignment="top",
             horizontalalignment="right",transform=ax3.transAxes,color="green",fontsize=12)
    ax3.text(0.95,0.9,"{:}{:.2f}".format("+/- ",perrY[1]),verticalalignment="top",
             horizontalalignment="right",transform=ax3.transAxes,color="green",fontsize=12)
    ax3.text(0.95,0.82,"{:}{:.2f}".format("Sigma = ",poptY[2]),verticalalignment="top",
             horizontalalignment="right",transform=ax3.transAxes,color="green",fontsize=12)
    ax3.text(0.95,0.77,"{:}{:.2f}".format("+/- ",perrY[2]),verticalalignment="top",
             horizontalalignment="right",transform=ax3.transAxes,color="green",fontsize=12)
    
    # y-Profile at mean x-Position
    ax4 = fig.add_subplot(gs[1,2])
    # Determine closest bin to mean
    xCentre = filter(lambda x: x < poptX[1] and x+abs(xBins[1]-xBins[0]) > poptX[1], xBins)
    if poptX[1]-xCentre[0] >= abs(xBins[1]-xBins[0]):
        xCentre[0] += (xBins[1]-xBins[0])
    centroidX = dataOut[:,xBins.tolist().index(xCentre[0])]
    ax4.hist(yBins,bins=yOut,weights=centroidX,histtype="step")
    ax4.set_title("y-Profile at Mean x-Position")
    if calFlag == True:
        ax4.set_xlabel("X [um]")
    else:
        ax4.set_xlabel("X [pixel]")
    ax4.set_ylabel("Counts")
    
    # x-Profile at mean y-Position
    ax5 = fig.add_subplot(gs[1,3])
    # Determine closest bin to mean
    yCentre = filter(lambda y: y < poptY[1] and y+abs(yBins[1]-yBins[0]) > poptY[1], yBins)
    if poptY[1]-yCentre[0] >= abs(yBins[1]-yBins[0]):
        yCentre[0] += (yBins[1]-yBins[0])
    centroidY = dataOut[yBins.tolist().index(yCentre[0]),:]
    ax5.hist(xBins,bins=xOut,weights=centroidY,histtype="step")
    ax5.set_title("x-Profile at Mean y-Position")
    if calFlag == True:
        ax5.set_xlabel("Y [um]")
    else:
        ax5.set_xlabel("Y [pixel]")
    ax5.set_ylabel("Counts")
    
    fig.tight_layout()
    plt.show()

# User input to plot images using function PlotFigure()
# Error checking included to reject non-floats, times greater than range and negative numbers
# Time entered by user does not have to be exact as function will find closest frame
# Enter 'q' to exit program
def ImageViewerMenu():
    while True:
        print "-----------------"
        print "Image Viewer Menu"
        print "-----------------"
        uInput = raw_input("Please enter a time in seconds (q to quit): ")
        
        try:
            val = float(uInput)
            # Determine closest time frame
            if Decimal(uInput)%Decimal(str(acqTime)) < acqTime/2.:
                val = val - float(Decimal(uInput)%Decimal(str(acqTime)))
            else:
                val = val + acqTime - float(Decimal(uInput)%Decimal(str(acqTime)))
            if (val/acqTime) > nImages-1:   # Time entered greater than maximum
                print "Time exceeds maximum of " + str(acqTime*(nImages-1)) + " s, please try again..."
            elif val < 0:   # Negative time entered
                print "Negative number entered, please try again..."
            else:
                print "Selected time frame: " + str(val) + " s"
                fileIndex = int(val/acqTime) + firstFileIndex
                PlotFigure(val,fileIndex,filePath,filePrefix,fileSuffix,firstFileIndex,dataSetName,calFlag)
        except ValueError:
            if uInput == "q":
                print "Exiting..."
                break
            else:
                print "Invalid choice, please try again..."
        print

ImageViewerMenu()   # Display menu for user input