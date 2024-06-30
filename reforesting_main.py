#----------------Reforesting Entrepreneurs------------------
#  Identifying locations that may be afforested.

#  Members:
#  Alfonso Medina Marrero
#  Paula Catalan Santana
#  Alfonso Santana Samper

from picamera import PiCamera
from sense_hat import SenseHat
import csv
import os
import cv2 as cv
from logzero import logger, logfile
import ephem
from datetime import datetime, timedelta
import numpy as np
import time

#  Initialising files

dir_path = os.path.dirname(os.path.realpath(__file__))

logfile(dir_path + "/ReforestingEntrepreneurs.log")

altitude_file = dir_path + "/Altitude_File.txt"

moisture_file = dir_path + "/SoilMoisture_File.txt"

data_file = dir_path + "/data.csv"


#  Preparing ISS Telemetry Data

name = "ISS (ZARYA)"
line1 = "1 25544U 98067A   20029.69572272  .00004768  00000-0  94250-4 0  9992"
line2 = "2 25544  51.6452 318.6562 0005196 207.4446 220.3378 15.49124691210396"
iss = ephem.readtle(name, line1, line2)

#---------------------Initialising Sense-Hat Display---------------------

sh = SenseHat()

#  Defining colours
y = [255,255,0]  # Yellow
aq = [0,128,128]  # Teal (Ocean-Blue)
m = [139,69,19]  # Sadle Brown
g = [0,255,0]  # Lime green
b = [0,0,128]  # Blue navy
o = [0,0,0]  # Black
r = [255,0,0]  # Red

#  Defining images to be shown in the LED display
#  No image is shown when taking the photo:
no_image = [
o,o,o,o,o,o,o,o,
o,o,o,o,o,o,o,o,
o,o,o,o,o,o,o,o,
o,o,o,o,o,o,o,o,
o,o,o,o,o,o,o,o,
o,o,o,o,o,o,o,o,
o,o,o,o,o,o,o,o,
o,o,o,o,o,o,o,o,
]

#  Loading images
loading_1 = [
o,o,o,o,o,o,o,o,
o,o,o,o,o,o,o,o,
b,b,b,b,b,b,b,b,
aq,aq,o,o,o,o,o,o,
aq,aq,o,o,o,o,o,o,
b,b,b,b,b,b,b,b,
o,o,o,o,o,o,o,o,
o,o,o,o,o,o,o,o,
]

loading_2 = [
o,o,o,o,o,o,o,o,
o,o,o,o,o,o,o,o,
b,b,b,b,b,b,b,b,
o,o,aq,aq,o,o,o,o,
o,o,aq,aq,o,o,o,o,
b,b,b,b,b,b,b,b,
o,o,o,o,o,o,o,o,
o,o,o,o,o,o,o,o,
]

loading_3 = [
o,o,o,o,o,o,o,o,
o,o,o,o,o,o,o,o,
b,b,b,b,b,b,b,b,
o,o,o,o,aq,aq,o,o,
o,o,o,o,aq,aq,o,o,
b,b,b,b,b,b,b,b,
o,o,o,o,o,o,o,o,
o,o,o,o,o,o,o,o,
]

loading_4 = [
o,o,o,o,o,o,o,o,
o,o,o,o,o,o,o,o,
b,b,b,b,b,b,b,b,
o,o,o,o,o,o,aq,aq,
o,o,o,o,o,o,aq,aq,
b,b,b,b,b,b,b,b,
o,o,o,o,o,o,o,o,
o,o,o,o,o,o,o,o,
]

# Storing all loading images into a variable.
loading = [loading_1, loading_2, loading_3, loading_4]

#  Altitude image (mountain with clouds)
Altitude_image = [
o,b,b,o,o,o,b,o,
b,b,b,b,o,b,b,b,
o,b,b,o,o,o,b,o,
o,o,o,m,m,o,o,o,
o,o,m,m,m,m,o,o,
o,m,m,m,m,m,m,o,
m,m,m,m,m,m,m,m,
m,m,m,m,m,m,m,m,
]

#  Slope image (45º slope)
Slope_image = [
o,o,o,o,o,o,o,y,
o,o,o,o,o,o,y,y,
o,o,o,o,o,y,y,y,
o,o,o,o,y,y,y,y,
o,o,o,y,y,y,y,y,
o,o,y,y,y,y,y,y,
o,y,y,y,y,y,y,y,
y,y,y,y,y,y,y,y,
]

#  Soil Moisture image (water drop)
Soil_Moisture_image = [
o,o,o,b,b,o,o,o,
o,o,o,b,b,o,o,o,
o,o,o,b,b,o,o,o,
o,o,b,b,b,b,o,o,
o,b,b,b,b,b,b,o,
o,b,b,b,b,b,b,o,
o,b,b,b,b,b,b,o,
o,o,b,b,b,b,o,o,
]

#  NDVI image (leaf)
NDVI_image = [
o,o,o,o,o,o,g,o,
o,o,o,o,o,g,g,o,
o,o,o,g,g,g,g,o,
o,o,g,g,g,g,g,o,
o,o,g,g,g,g,g,o,
o,o,g,g,g,g,o,o,
o,g,o,g,g,o,o,o,
g,o,o,o,o,o,o,o,
]

#  Afforestable image (Tree)
Afforestable_image = [
o,o,o,o,o,o,o,o,
o,o,o,g,g,o,o,o,
o,o,g,g,g,g,o,o,
o,g,g,g,g,g,g,o,
g,g,g,g,g,g,g,g,
g,g,g,g,g,g,g,g,
o,o,o,m,m,o,o,o,
o,o,o,m,m,o,o,o,
]

#---------------------------------------------------------------


def Night_Detector(img, ImgHeight, ImgWidth):
    #  Function to identify a night photo.

    #  Finding the average pixel BGR value. Instead of reading each pixel value,
    #  which takes an average of 176 seconds, we are reading every four pixels
    #  for an improved efficiency, reducing analysis time to 11 seconds (16 times less).
    #  The average BGR value results more or less the same as previously obtained values.
    #  Note: OpenCV reads pixels in the format BGR.

    TotalB = 0
    TotalR = 0
    TotalG = 0

    count = 0  # Counting variable to output loading images

    #  Totalling BGR values of every pixel
    try:
        for row in range(0, ImgHeight, 4):

            sh.set_pixels(loading[count % len(loading)])  # Output loading images
            count += 1

            for column in range(0, ImgWidth, 4):
                RGB_Value = list(img[row, column])
                TotalB = TotalB + RGB_Value[0]
                TotalG = TotalG + RGB_Value[1]
                TotalR = TotalR + RGB_Value[2]
    except Exception as e:
        logger.error(("Night_Detector function error: {}: {}").format(e.__class__.__name__, e))

    AverageB = TotalB / ((ImgWidth * ImgHeight) / 16)
    AverageG = TotalG / ((ImgWidth * ImgHeight) / 16)
    AverageR = TotalR / ((ImgWidth * ImgHeight) / 16)

    # Translating the individual BGR values to a unified greyscale value

    Average_GreyScale = (AverageB + AverageG + AverageR) / 3

    Night_Value = 40  # When experimenting, we have considered night a value below 40 in greyscale

    if Average_GreyScale < Night_Value:
        Night = True
    elif Average_GreyScale == 0:
        logger.error("Night_Detector function error: Average_GreyScale cannot be equal to zero")
    else:
        Night = False

    return(Night)


def NDVI_Filter_Image(img, ImgHeight, ImgWidth, photo_name):
    #  Function to process the image with NDVI principles and output the average
    #  NDVI value. This allows us to find out if the area under the photo requires
    #  afforestation efforts, due to the chlorophyll concentration in the area.

    sh.set_pixels(NDVI_image)  # Output NDVI image

    #  Initialising the arrays to store pixel data.

    Red_Values = np.zeros((ImgHeight, ImgWidth))
    Blue_Values = np.zeros((ImgHeight, ImgWidth))
    NDVI_Values = np.zeros((ImgHeight, ImgWidth))

    #  Storing all of the red and blue values for each pixel in separate arrays.
    #  Note: OpenCV reads pixels in the format BGR.

    Red_Values = (img[:, :, 2]).astype('float')
    Blue_Values = (img[:, :, 0]).astype('float')

    #  To carry out the NDVI operation, we must discard the values of the denominator which
    #  are equal to zero to avoid division by zero errors.

    Denominator = Blue_Values + Red_Values
    Denominator = np.where(Denominator != 0, Denominator, 0.001)

    #  NDVI operation: (Blue-Red)/(Blue+Red)

    try:
        NDVI_Values = np.divide((Blue_Values - Red_Values), Denominator)
    except Exception as e:
        logger.error(("NDVI calculation: {}: {}").format(e.__class__.__name__, e))

    #  Calculating average NDVI value

    Mean_NDVI = round(np.mean(NDVI_Values), 2)

    #  We have considered appropriate an NDVI value between 0 and 0.4 as it corresponds to
    #  soil areas/meadows/shrubs according to Earth Observing System: https://eos.com/ndvi/
    #  Inferior values correspond to water, snow and clouds. Superior values correspond to areas already
    #  concentrated in chlorophyll (ie. forests), and therefore do not require afforestation/reforestation efforts.

    if Mean_NDVI > 0 and Mean_NDVI <= 0.4:
        Good_NDVI = "Yes"
    else:
        Good_NDVI = "No"

    #  This is an extra piece of code that creates an image with the NDVI results previously
    #  obtained. This part of the code causes the programm to run extremely slow, so to
    #  be able to take more photos in the 3 hour window, we have decided to not execute it
    #  in the ISS. Later on we could do back on Earth this image processing.

    #try:
    #    List_Value = list(np.arange(-1, 1.01, 0.01))
    #    List_Value = [round(elem, 2) for elem in List_Value]
    #
    #    Green = np.linspace(0, 255, num=len(List_Value))
    #    Blue = np.linspace(255, 0, num=len(List_Value))
    #
        #  Saving the image as another variable to change its pixel values
    #    NDVI_Image = img

        #  Changing the value of every pixel in the original image to the green and blue value
        #  that corresponds to the pixel's NDVI.
    #    for x in range(ImgWidth):
    #        for y in range(ImgHeight):
    #            NDVI_Value = round(NDVI_Values[y, x], 2)
    #            pos = List_Value.index(NDVI_Value)
    #            Blue_Value = round(Blue[pos])
    #            Green_Value = round(Green[pos])
    #            NDVI_Image[y, x] = [Blue_Value, Green_Value, 0]

        #  Saving the image with "_NDVI" sufix
    #    cv.imwrite(dir_path + photo_name + "_NDVI.jpg", NDVI_Image)
    #except Exception as e:
    #    logger.error(("NDVI photo creator error: {}: {}").format(e.__class__.__name__, e))

    time.sleep(2)  # Time to show in the LED matrix the NDVI image
    return(Mean_NDVI, Good_NDVI)


def Get_LatLon():
    # Function to identify the real-time location of ISS
    iss.compute()

    Coord_DMS_Lat = [float(i) for i in str(iss.sublat).split(":")]  # Latitude Coordinate in DMS form
    Coord_DMS_Long = [float(i) for i in str(iss.sublong).split(":")]  # Longitude Coordinate in DMS form

    # DMS (Degrees, Minutes, Seconds) to Decimal Degrees converter, rounded to 6 decimal places
    Coord_Dec_Deg_Lat = round(Coord_DMS_Lat[0] + (Coord_DMS_Lat[1] / 60) + (Coord_DMS_Lat[2] / 3600), 6)  # Latitude Coordinate in decimal degrees form
    Coord_Dec_Deg_Long = round(Coord_DMS_Long[0] + (Coord_DMS_Long[1] / 60) + (Coord_DMS_Long[2] / 3600), 6)  # Longitude Coordinate in decimal degrees form

    return(Coord_Dec_Deg_Lat, Coord_Dec_Deg_Long)


def Altitudes_Array_Creator(Lat, Long):
    #  Function to create an array of altitudes. This enables us to calculate the average height and
    #  slope. Such array will contain 121 altitude values, obtained from a NASA topographic
    #  database: https://neo.sci.gsfc.nasa.gov/view.php?datasetId=SRTM_RAMP2_TOPO that correspond
    #  to 121 locations under the scope of Izzy's image

    #  First of all, we must approximate the latitude and longitude coordinates to
    #  the coordinates available in the altitude file: .125 / .375 / .625 / .875
    #   * The method is difficult to understand, but works!

    #----Processing latitude coordinate------
    Lat_Value_3DP = round(Lat, 3)  # This is the latitude value in decimal degrees rounded to 3 decimal places
    if (round(Lat_Value_3DP / 0.125) % 2) == 0:
        if ((Lat_Value_3DP / 0.125) / 2) >= (round(Lat_Value_3DP / 0.125) / 2):
            Main_Lat_Coord_Height = ((round(Lat_Value_3DP / 0.125) / 2 + 0.5) * 2 * 0.125)
        elif ((Lat_Value_3DP / 0.125) / 2) < (round(Lat_Value_3DP / 0.125) / 2):
            Main_Lat_Coord_Height = ((round(Lat_Value_3DP / 0.125) / 2 - 0.5) * 2 * 0.125)
    else:
        Main_Lat_Coord_Height = (round(Lat_Value_3DP / 0.125)) * 0.125

    #----Processing longitude coordinate------
    Long_Value_3DP = round(Long, 3)  # This is the longitude value in decimal degrees rounded to 3 decimal places
    if (round(Long_Value_3DP / 0.125) % 2) == 0:
        if ((Long_Value_3DP / 0.125) / 2) >= (round(Long_Value_3DP / 0.125) / 2):
            Main_Long_Coord_Height = ((round(Long_Value_3DP / 0.125) / 2 + 0.5) * 2 * 0.125)
        elif ((Long_Value_3DP / 0.125) / 2) < (round(Long_Value_3DP / 0.125) / 2):
            Main_Long_Coord_Height = ((round(Long_Value_3DP / 0.125) / 2 - 0.5) * 2 * 0.125)
    else:
        Main_Long_Coord_Height = (round(Long_Value_3DP / 0.125)) * 0.125

    #  Creating two lists: one contianing all the latitude coordinates, and the other
    #  containing all the longitude coordinates that represent all the possible coordinates
    #  under the scope of Izzy's image.

    First_Lat_Coord_Height = Main_Lat_Coord_Height - (0.25 * 5)
    First_Long_Coord_Height = Main_Long_Coord_Height - (0.25 * 5)

    List_Lat_Coord = []
    List_Long_Coord = []

    #  Each list will have a length of 11 elements (coordinates). Since in the altitude file
    #  the distance between samples is 0.25º, which accounts to around 27.75 Km, we have calculated
    #  that eleven coordinates in the latitude and longitude direction cover a major portion of the
    #  image taken. All of the 121 height samples obtained later will be used to calculate the
    #  average slope (between each sample) and average height in the image taken.

    for i in range(11):
        List_Lat_Coord.append(First_Lat_Coord_Height + (0.25 * i))
        List_Long_Coord.append(First_Long_Coord_Height + (0.25 * i))

    #  Re-writing impossible coordinates: longitudes above 180º or below -180º
    #  and latitudes above 90º or below -90º (these are represented as 0º, or no data)

    for i in range(len(List_Long_Coord)):
        if List_Long_Coord[i] > 180:
            List_Long_Coord[i] = List_Long_Coord[i] - 360
        elif List_Long_Coord[i] < -180:
            List_Long_Coord[i] = List_Long_Coord[i] + 360

    for i in range(len(List_Lat_Coord)):
        if List_Lat_Coord[i] > 90:
            List_Lat_Coord[i] = 0
        elif List_Lat_Coord[i] < -90:
            List_Lat_Coord[i] = 0

    #  Preparing the array where the all the altitudes (in the possible coordinates occupied
    #  by the Astro-Pi's image) will be stored for later processing/filtering.

    Heights_Array = np.zeros((11, 11))

    #  Searching in the altitude file for the altitude value in each of the coordinates stored in the
    #  lists. Results will be then stored in the array.

    count = 0  # Counting variable for loading images

    for List_Long_Pos in range(len(List_Long_Coord)):
        for List_Lat_Pos in range(len(List_Lat_Coord)):

            sh.set_pixels(loading[count % len(loading)])  # Output loading images
            count += 1

            Read_Altitude_File = open(altitude_file, "r")

            LineOfText = Read_Altitude_File.readline()
            All_Long_Coord = LineOfText.split(",")
            All_Long_Coord[-1] = All_Long_Coord[-1].rstrip('\n')

            Found_Long = False
            Pos_Long = 0

            while not Found_Long:
                try:
                    Pos_Long = Pos_Long + 1
                    if float(All_Long_Coord[Pos_Long]) == List_Long_Coord[List_Long_Pos]:
                        Found_Long = True
                except Exception as e:
                        logger.error(("Altitude_Array Error - search long value: {}: {}").format(e.__class__.__name__, e))

            Found_Lat = False
            if List_Lat_Coord[List_Lat_Pos] != 0:
                while LineOfText != "" and not Found_Lat:
                    try:
                        LineOfText = Read_Altitude_File.readline()
                        Lat_Values = LineOfText.split(",")
                        if float(Lat_Values[0]) == List_Lat_Coord[List_Lat_Pos]:
                            Found_Lat = True
                    except Exception as e:
                        logger.error(("Altitude_Array Error - search lat value: {}: {}").format(e.__class__.__name__, e))

                Read_Altitude_File.close()

                Lat_Values[-1] = Lat_Values[-1].rstrip('\n')

                Height = float(Lat_Values[Pos_Long])
            else:
                Height = 0  # Discarding the latitude coordinates whose values are above 90º and below -90º and storing them as 0 metres

            if Height == 99999.0:  # Changing the default no-data value in the altitude file to 0 metres
                Height = 0

            Heights_Array[List_Lat_Pos, List_Long_Pos] = Height  # Saving the altitude value in the corresponding location of the array

    return(Heights_Array)


def Soil_Moisture_Array_Creator(Lat, Long):
    #  Function to create an array of soil moistures. This enables us to then calculate the average soil
    #  moisutre. Such array will contain 121 soil moisture values, obtained from ESA's databases:
    #  https://www.esa-soilmoisture-cci.org/  that correspond to 121 locations under the scope of Izzy's image.

    # Like previously done with the heights array, we must approximate the latitude
    # and longitude coordinates to the coordinates available in the soil moisture
    # file: .125 / .375 / .625 / .875
    #  * The method is difficult to understand, but works!
    #  ** SM stands for Soil Moisture

    #----Processing latitude coordinate------
    Lat_Value_3DP = round(Lat, 3)  # This is the latitude value in decimal degrees rounded to 3 decimal places
    if (round(Lat_Value_3DP / 0.125) % 2) == 0:
        if ((Lat_Value_3DP / 0.125) / 2) >= (round(Lat_Value_3DP / 0.125) / 2):
            Main_Lat_Coord_SM = ((round(Lat_Value_3DP / 0.125) / 2 + 0.5) * 2 * 0.125)
        elif ((Lat_Value_3DP / 0.125) / 2) < (round(Lat_Value_3DP / 0.125) / 2):
            Main_Lat_Coord_SM = ((round(Lat_Value_3DP / 0.125) / 2 - 0.5) * 2 * 0.125)
    else:
        Main_Lat_Coord_SM = (round(Lat_Value_3DP / 0.125)) * 0.125

    #----Processing longitude coordinate------
    Long_Value_3DP = round(Long, 3)  # This is the longitude value in decimal degrees rounded to 3 decimal places
    if (round(Long_Value_3DP / 0.125) % 2) == 0:
        if ((Long_Value_3DP / 0.125) / 2) >= (round(Long_Value_3DP / 0.125) / 2):
            Main_Long_Coord_SM = ((round(Long_Value_3DP / 0.125) / 2 + 0.5) * 2 * 0.125)
        elif ((Long_Value_3DP / 0.125) / 2) < (round(Long_Value_3DP / 0.125) / 2):
            Main_Long_Coord_SM = ((round(Long_Value_3DP / 0.125) / 2 - 0.5) * 2 * 0.125)
    else:
        Main_Long_Coord_SM = (round(Long_Value_3DP / 0.125)) * 0.125

    #  Creating two lists: one contianing all the latitude coordinates, and the other
    #  containing all the longitude coordinates that represent all the possible coordinates
    #  under the scope of Izzy's image.

    First_Lat_Coord_SM = Main_Lat_Coord_SM - (0.25 * 5)
    First_Long_Coord_SM = Main_Long_Coord_SM - (0.25 * 5)

    List_Lat_Coord = []
    List_Long_Coord = []

    #  Each list will have a length of 11 elements (coordinates). Since in the soil moisture file
    #  the distance between samples is 0.25º, which accounts to around 27.75 Km, we have calculated
    #  that eleven of these in the latitude and longitude direction cover a major portion of the
    #  image taken. All of the 121 soil moisture samples obtained later will be used to calculate the
    #  average soil moisture in the area taken by the camera.

    for i in range(11):
        List_Lat_Coord.append(First_Lat_Coord_SM + (0.25 * i))
        List_Long_Coord.append(First_Long_Coord_SM + (0.25 * i))

    #  Re-writing impossible coordinates: longitudes above 180º or below -180º
    #  and latitudes above 90º or below -90º (these are represented as 0º, or no data)

    for i in range(len(List_Long_Coord)):
        if List_Long_Coord[i] > 180:
            List_Long_Coord[i] = List_Long_Coord[i] - 360
        elif List_Long_Coord[i] < -180:
            List_Long_Coord[i] = List_Long_Coord[i] + 360

    for i in range(len(List_Lat_Coord)):
        if List_Lat_Coord[i] > 90:
            List_Lat_Coord[i] = 0
        elif List_Lat_Coord[i] < -90:
            List_Lat_Coord[i] = 0

    #  Preparing the array where the all the soil moistures (in the possible coordinates occupied
    #  by the Astro-Pi's image) will be stored for later processing/filtering.

    Soil_Moisture_Array = np.zeros((11, 11))

    #  Searching in the soil moisture file for the soil moisture value in each of the coordinates stored in the
    #  lists. Results will be then stored in the array.

    Read_SM_File = open(moisture_file, "r")
    LineOfText = Read_SM_File.readline()

    count = 0  # Counting variable for loading images

    for List_Lat_Pos in range(len(List_Lat_Coord)):

        sh.set_pixels(loading[count % len(loading)])  # Output loading images
        count += 1

        for List_Long_Pos in range(len(List_Long_Coord)):
            Found = False
            if List_Lat_Coord[List_Lat_Pos] != 0:
                while LineOfText != "" and not Found:
                    try:
                        LineOfText = Read_SM_File.readline()
                        SM_Coord_List = LineOfText.split(", ")
                        SM_Coord_List[-1] = SM_Coord_List[-1].rstrip('\n')
                        if float(SM_Coord_List[0]) == List_Lat_Coord[-(List_Lat_Pos + 1)] and float(SM_Coord_List[1]) == List_Long_Coord[List_Long_Pos]:
                            Found = True
                    except Exception as e:
                        logger.error(("Soil_Moisture_Array Error - search lat/long value: {}: {}").format(e.__class__.__name__, e))
                if not Found:
                    Soil_Moisture = 0
                else:
                    Soil_Moisture = SM_Coord_List[2]

                if Soil_Moisture == "NaN":
                    Soil_Moisture = 0  # Changing the default no-data value in the soil moisture file to 0 units
                else:
                    Soil_Moisture = float(Soil_Moisture)
            else:
                Soil_Moisture = 0  # Discarding the latitude coordinates whose values are above 90º and below -90º and storing them as 0 metres

            Soil_Moisture_Array[-(List_Lat_Pos + 1), List_Long_Pos] = Soil_Moisture  # Saving the soil moisture value in the corresponding location of the array

    Read_SM_File.close()

    return(Soil_Moisture_Array)


def Altitude_Filter(Alt_Array):
    #  Function to find average height and output if it corresponds to a good average for afforestation

    sh.set_pixels(Altitude_image)  # Output altitude image

    #  Calculating the average altitude from the 11x11 matrix
    Average_Altitude = round(np.mean(Alt_Array), 2)

    #  The altitude value must fall between sea level and 3500 metres. As it whould be unrealistic to afforest,
    #  according to the following paper section 2.2 : Zomer, Robert & Trabucco, Antonio & Verchot, Louis & Muys,
    #  Bart. (2008). Land Area Eligible for Afforestation and Reforestation within the Clean Development Mechanism:
    #  A Global Analysis of the Impact of Forest Definition. Mitigation and Adaptation Strategies for Global Change.
    #  13. 219-239. DOI : 10.1007/s11027-007-9087-4.

    if Average_Altitude > 0 and Average_Altitude < 3500:
        Good_Altitude = "Yes"
    else:
        Good_Altitude = "No"

    time.sleep(2)  # Time to show in the LED matrix the altitude image
    return(Average_Altitude, Good_Altitude)


def Slopes_Filter(Alt_Array):
    #  Function to find the average slope between samples and output if it corresponds to a good average for afforestation

    sh.set_pixels(Slope_image)  # Output slope image

    #  Calculaing the mean change in altitude (Δy) between each element in the array. All values are positive
    Average_Difference = (np.mean(abs(np.diff(Alt_Array, axis=0))) + np.mean(abs(np.diff(Alt_Array, axis=1)))) / 2

    #  Calculating slope (Δy/Δx) in percentage. Both distances are in metres. Δx = 27750 metres as the distance between
    #  samples in the altitude file is 27.75 Km

    Average_Slope = round((((Average_Difference) / 27750) * 100), 2)

    #  We have considered steep a slope above 19.4% (11º) Reference: Wondie,
    #  Menale & Teketay, Demel & Melesse, Assefa & Schneider, Werner. (2011).
    #  Relationship between Topographic Variables and Land Cover in the Simen
    #  Mountains National Park, a World Heritage Site in Northern Ethiopia.
    #  International Journal of Remote Sensing.

    if Average_Slope < 19.4:
        Good_Slopes = "Yes"
    else:
        Good_Slopes = "No"

    time.sleep(2)  # Time to show in the LED matrix the slopes image
    return(Average_Slope, Good_Slopes)


def Soil_Moisture_Filter(SM_Array):
    #  Function to find average soil moisture and output if it corresponds to a good average for afforestation

    sh.set_pixels(Soil_Moisture_image)  # Output soil moisture image

    #  The mean is multiplied by 100 to obtain the percentage

    Average_SM = round((100 * np.mean(SM_Array)), 2)

    #  The soil moisture value must be higher than 12.5% . It is best plant in loam soil, as it has many advantages
    #  when compared to the extremes of sand and clay: according to the book Lockhart & Wiseman’s Crop Husbandry
    #  Including Grassland (Ninth Edition), 2014 by H.J.S. Finch, A.M. Samuel and G.P.F. in chapter 3: LandSoils and soil management
    #  According to Spruce Irrigation: What is my target moisture level?:
    #  https://support.spruceirrigation.com/knowledge-base/what-is-my-target-moisture-level/,
    #  the ideal soil moisture for loam is above 10%, but since all the soil on earth isn't loam type, we raised the value to 12.5%.

    if Average_SM > 12.5:
        Good_SM = "Yes"
    else:
        Good_SM = "No"

    time.sleep(2)  # Time to show in the LED matrix the soil moisture image
    return(Average_SM, Good_SM)


def Add_csv_Data(data_file, data):
    #  Function to add data into the csv. Function courtesy: ESA/Raspberry
    with open(data_file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(data)
    f.close()


def Elapsed(time1, time2):
    #  Function to calculate the elapsed time between loops
    Difference = (time2 - time1)
    Seconds = round(Difference.total_seconds(), 2)
    return(Seconds)


with open(data_file, 'w') as f:
    writer = csv.writer(f)
    header = ("Date/time", "Photo Name", "Lat", "Long", "Night", "NDVI", "Y/N", "Altitude", "Y/N", "Slopes", "Y/N", "Soil Moisture", "Y/N", "Afforestable", "Elapsed Time")
    writer.writerow(header)
    f.close()

#  Initialise in variables image height and width
ImgHeight = 1944
ImgWidth = 2592

#  Initialise camera
cam = PiCamera()
cam.resolution = (ImgWidth, ImgHeight)

photo_counter = 1

#  Store the start time
start_time = datetime.now()
now_time = datetime.now()

#  Start of the loop
while (now_time < start_time + timedelta(minutes=165)):
    try:
        time1 = datetime.now()

        sh.set_pixels(no_image)  # Ensuring the LED display is completely off before taking the photo to avoid any reflection

        #  Establishing the photo name
        photo_name = "/photo_" + (str(photo_counter).zfill(3))

        #  Taking the picture
        cam.capture(dir_path + photo_name + ".jpg")

        #  Changing the counter value
        photo_counter = photo_counter + 1

        #  Obtaining the latitude and longitude decimal degree value.
        Lat, Long = Get_LatLon()

        #  Reading the image with the OpenCV module
        img = cv.imread(dir_path + photo_name + ".jpg")

        #  Detect night-photo
        Night = Night_Detector(img, ImgHeight, ImgWidth)

        if not Night:  # Only continue processing if the photo isn't a night-time-photo
            Afforestable = False

            NDVI, Good_NDVI = NDVI_Filter_Image(img, ImgHeight, ImgWidth, photo_name)

            Heights_Array = Altitudes_Array_Creator(Lat, Long)
            SM_Array = Soil_Moisture_Array_Creator(Lat, Long)

            Average_Altitude, Good_Altitude = Altitude_Filter(Heights_Array)
            Average_Slopes, Good_Slopes = Slopes_Filter(Heights_Array)
            Average_SM, Good_SM = Soil_Moisture_Filter(SM_Array)

            time2 = datetime.now()
            Elapsed_Time = Elapsed(time1, time2)

            if Good_NDVI == "Yes" and Good_Altitude == "Yes" and Good_Slopes == "Yes" and Good_SM == "Yes":
                Afforestable = True

                sh.set_pixels(Afforestable_image)  # Output afforestable image
                time.sleep(2)

            #  Storing all the data under the variable "data"
            data = (time1, photo_name, Lat, Long, Night, NDVI, Good_NDVI, Average_Altitude, Good_Altitude, Average_Slopes, Good_Slopes, Average_SM, Good_SM, Afforestable, Elapsed_Time)
        else:
            os.remove(dir_path + photo_name + ".jpg")  # Removing night images
            time2 = datetime.now()
            Elapsed_Time = Elapsed(time1, time2)

            data = (time1, photo_name, Lat, Long, Night, Elapsed_Time)

            time.sleep(15)  # Wait 15 seconds before taking another photo if it was night.

        #  Adding data into the csv
        Add_csv_Data(data_file, data)

        now_time = datetime.now()
    except Exception as e:
        logger.error('Error in loop : {}: {})'.format(e.__class__.__name__, e))

sh.show_message("Finished!! - Reforesting Entrepreneurs", scroll_speed=0.05)
sh.clear()