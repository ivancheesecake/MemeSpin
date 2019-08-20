import csv
from sklearn import metrics
import numpy as np
import math


def calculateJMD(partOfClass,notPartOfClass):

    meanVegIndex = np.mean(partOfClass)
    meanNonvegIndex = np.mean(notPartOfClass)

    stdevVegIndex = np.var(partOfClass)
    stdevNonvegIndex = np.var(notPartOfClass)


    BIndex = (float(1/8) * (((meanVegIndex - meanNonvegIndex)**2)/(((stdevVegIndex**2) + (stdevNonvegIndex**2))/2))) + (0.5*(math.log(     (((stdevVegIndex**2 + stdevNonvegIndex**2)/2)   /(stdevVegIndex*stdevNonvegIndex))   ,math.e)))
    jmd = 2*(1 - (math.e ** - BIndex ))

    return jmd


bands = ["B1","B2","B3","B4","B5","B6","B7"]


labels = []
trial4 = []
luma = []
built2 = []
ndbi = []
trial4_partOfClass = []
trial4_notPartOfClass = []
ndvi_partOfClass = []
ndvi_notPartOfClass = []

ndvi = []
veg1 = []
veg2 = []
veg3 = []
veg4 = []
veg5 = []

ndviPart = []
veg1Part = []
veg2Part = []
veg3Part = []
veg4Part = []
veg5Part = []

ndviNotPart = []
veg1NotPart = []
veg2NotPart = []
veg3NotPart = []
veg4NotPart = []
veg5NotPart = []




with open("input/ValidationPoints_0517.csv") as f:

    points = list(csv.reader(f))

for point in points[1:]:
    labels.append(point[0])
    ndvi.append([point[1]])
    veg1.append([point[2]])
    veg2.append([point[3]])
    veg3.append([point[4]])
    veg4.append([point[5]])
    veg5.append([point[6]])

    if point[0] == '2':
    	ndviPart.append(float(point[1]))
    	veg1Part.append(float(point[2]))
    	veg2Part.append(float(point[3]))
    	veg3Part.append(float(point[4]))
    	veg4Part.append(float(point[5]))
    	veg5Part.append(float(point[6]))
    	
    else:
    	ndviNotPart.append(float(point[1]))
    	veg1NotPart.append(float(point[2]))
    	veg2NotPart.append(float(point[3]))
    	veg3NotPart.append(float(point[4]))
    	veg4NotPart.append(float(point[5]))
    	veg5NotPart.append(float(point[6]))




score_ndvi = metrics.silhouette_score(np.array(ndvi),np.array(labels))
score_veg1 = metrics.silhouette_score(np.array(veg1),np.array(labels))
score_veg2 = metrics.silhouette_score(np.array(veg2),np.array(labels))
score_veg3 = metrics.silhouette_score(np.array(veg3),np.array(labels))
score_veg4 = metrics.silhouette_score(np.array(veg4),np.array(labels))
score_veg5 = metrics.silhouette_score(np.array(veg5),np.array(labels))



jmd_ndvi = calculateJMD(ndviPart,ndviNotPart)
jmd_veg1 = calculateJMD(veg1Part,veg1NotPart)
jmd_veg2 = calculateJMD(veg2Part,veg2NotPart)
jmd_veg3 = calculateJMD(veg3Part,veg3NotPart)
jmd_veg4 = calculateJMD(veg4Part,veg4NotPart)
jmd_veg5 = calculateJMD(veg5Part,veg5NotPart)


print("Silhouette: ")
print("NDVI",score_ndvi)
print("VEG1",score_veg1)
print("VEG2",score_veg2)
print("VEG3",score_veg3)
print("VEG4",score_veg4)
print("VEG5",score_veg5)

print("JMD: ")
print("NDVI",jmd_ndvi)
print("VEG1",jmd_veg1)
print("VEG2",jmd_veg2)
print("VEG3",jmd_veg3)
print("VEG4",jmd_veg4)
print("VEG5",jmd_veg5)



