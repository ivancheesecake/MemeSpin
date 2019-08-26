from __future__ import print_function

import SpectralIndex
import random
import math
import copy
import csv
import numpy as np
import operator
import pickle

from sklearn import metrics


def basicCrossover(tree1, tree2, CROSSOVER_RATE = 0.95):

    if(random.random() > CROSSOVER_RATE):
        # print("NO CROSSOVER")
        return copy.deepcopy(tree1),copy.deepcopy(tree2)

    tree3 = copy.deepcopy(tree1)
    tree4 = copy.deepcopy(tree2)

    temp = tree3.index.left

    tree3.index.left = tree4.index.left
    tree3.index.left.parent = tree3.index


    tree4.index.left = temp
    tree4.index.left.parent = tree4.index

    # tree3.length = tree3.count(tree3.index)
    # tree4.length = tree4.count(tree4.index)


    return tree3,tree4

def calculateJMD(partOfClass,notPartOfClass):

    meanVegIndex = np.mean(partOfClass)
    meanNonvegIndex = np.mean(notPartOfClass)

    stdevVegIndex = np.var(partOfClass)
    stdevNonvegIndex = np.var(notPartOfClass)


    BIndex = (float(1/8) * (((meanVegIndex - meanNonvegIndex)**2)/(((stdevVegIndex**2) + (stdevNonvegIndex**2))/2))) + (0.5*(math.log(     (((stdevVegIndex**2 + stdevNonvegIndex**2)/2)   /(stdevVegIndex*stdevNonvegIndex))   ,math.e)))
    jmd = 2*(1 - (math.e ** - BIndex ))

    return jmd


def calculateFitness(index,bands,pointSet):


    # Class Kodigo
    # 1 - Vegetation
    # 2 - Builtup
    # 3 - Open
    # 4 - Sediment
    # 5 - Water

    classes = ['1','2','3','4','5']
    classes2 = ['1','2','3','4','5']
    selectedClass = '1'

    allPoints = []

    partOfClass = []
    notPartOfClass = []

    feature = []
    labels = []

    for c in classes:
        allPoints += pointSet[c]

    # print(allPoints[0])

    for i in range(len(allPoints)):

        idx = evaluate(index,allPoints[i])
        
        feature.append([idx])
        labels.append(allPoints[i]['class'])

        if allPoints[i]['class'] == selectedClass:
            partOfClass.append(idx)
        else:
            notPartOfClass.append(idx)



    classMean = np.mean(partOfClass)
    nonclassMean = np.mean(notPartOfClass)

    # Calculate silhouette score

    score = metrics.silhouette_score(feature,labels)

    # Calculate Jeffries-Matusita Distance

    jmd = calculateJMD(partOfClass,notPartOfClass)

    # print(score)

    isLarger = 999

    # Mean of selected class is larger than 0.5 

    if classMean > 0.5:
        isPointFive = 1
    else:
        isPointFive = 0

    # Mean of nonselected class is smaller than 0.5 

    if nonclassMean < 0.5:
        isNotPointFive = 1
    else:
        isNotPointFive = 0    

    # Less than 1 constraint

    if classMean <= 1 and nonclassMean <= 1 and classMean >= -1 and nonclassMean >= -1:
        lessThanTen = 1
    else:
        lessThanTen = 0

    # print(ipvMean,ipnMean,isLarger)

    # print(isLarger)
    return ((0.35 * score) + (0.35 * (jmd/2)) + (0.1 * isPointFive) + (0.1 * isNotPointFive) + (0.1 * lessThanTen)),score,isLarger,isPointFive,isNotPointFive,lessThanTen,jmd
    # return silhouette


def expandForStep(index,index2,isLeft):

    if index.left == None and index.right == None:

        if isLeft:
            index.parent.left = index2
            index2.parent = index
        else:
            index.parent.right = index2
            index2.parent = index

        # return index

    else:

        if random.random() < 0.5:
            expandForStep(index.left,index2,True)
        else:
            expandForStep(index.right,index2,False)

def mutateForStep(index,bands,mutation_rate = 1.0):

    operators = ["+","-","/","*"]
    unary_operators = ["-","sqrt","log"]


    if index == None:
        return

    if random.random() < mutation_rate:

        if index.operation:
            index.value = random.choice(operators)
        else:
            if random.random() < 0.5:
                index.unary_op = random.choice(unary_operators)
            if random.random() < 0.5:
                index.value = random.choice(bands)
            if random.random() < 0.5:
                index.coefficient = random.random()

            index.update_display_coefficient()

    mutateForStep(index.left,bands,mutation_rate)
    mutateForStep(index.right,bands,mutation_rate)

def step(index,bands,length,maxlength,mutation_rate = 0.05):


    if(random.random() < mutation_rate):

        if random.random() < 0.95:
            # print("BASIC")
            mutateForStep(index,bands, mutation_rate = mutation_rate)
        
        else:
            # print("EXPAND")
            if length < maxlength:
                bools = [True,False]
                index2 = SpectralIndex.SpectralIndex(bands)
                expandForStep(index,index2.index,random.choice(bools))
            else:
                mutateForStep(index,bands, mutation_rate = mutation_rate)


            # index.length = index.count(index.index)


def evaluate(root,values):

    # empty SpectralIndexNode
    if root is None:
        return 0

    # leaf node
    if root.left is None and root.right is None:

        # multiply value with coefficient

        temp = float(values[root.value]) * root.coefficient
        # evaluate unary operation

        if root.unary_op == "sqrt":
            return math.sqrt(abs(temp))
        elif root.unary_op == "log":
            if temp == 0:
                return 0
            else:
                return math.log(abs(temp),10)
        elif root.unary_op == "-":
            return temp*-1

        return temp


    left_sum = evaluate(root.left,values)
    right_sum = evaluate(root.right,values)

    if root.value == '+':
        return left_sum + right_sum

    elif root.value == '-':
        return left_sum - right_sum

    elif root.value == '*':
        return left_sum * right_sum

    else:
        if right_sum == 0:
            return 1
        else:
            return left_sum / right_sum


def simulatedAnnealing(index, bands, pointSet,tMax = 5000,tMin = 1,cooling = 0.3, maxIterations = 10, MAXLENGTH = 10):

    currentIndex = copy.deepcopy(index)
    nextIndex = copy.deepcopy(index)
    bestIndex = copy.deepcopy(index)

    T = tMax;
    while(T > tMin):

        # print(T)
        for i in range(maxIterations):
            currentIndex.fitness,currentIndex.score,currentIndex.isLarger,currentIndex.isPointFive,currentIndex.isNotPointFive,currentIndex.lessThanTen,currentIndex.jmd = calculateFitness(currentIndex.index,bands,pointSet)
            currentFitness = currentIndex.fitness
            # print("Current",currentFitness)

            nextIndex = copy.deepcopy(currentIndex)

            step(nextIndex.index,bands,nextIndex.length,MAXLENGTH,mutation_rate = T/tMax)
            # step(nextIndex.index,bands,nextIndex.length,MAXLENGTH,mutation_rate = 1.0)
            nextIndex.length = nextIndex.count(nextIndex.index)
            nextIndex.fitness,nextIndex.score,nextIndex.isLarger,nextIndex.isPointFive,nextIndex.isNotPointFive,nextIndex.lessThanTen,nextIndex.jmd = calculateFitness(nextIndex.index,bands,pointSet)
            nextFitness = nextIndex.fitness

            # print("Next",nextFitness)

            deltaFitness = nextFitness - currentFitness
            # print("Delta",deltaFitness)

            if deltaFitness > 0:
                currentIndex = copy.deepcopy(nextIndex)
                if bestIndex.fitness < currentIndex.fitness:
                    bestIndex = copy.deepcopy(currentIndex)            

            elif (math.exp(deltaFitness/T)) > random.random():
                # print("BLAG")
                currentIndex = copy.deepcopy(nextIndex)


        T *= 1 - cooling

    # print("Current Index")
    # currentIndex.display()
    # print(currentIndex.fitness)

    # print("Best Index")
    # bestIndex.display()
    # print(bestIndex.fitness)

    return bestIndex





# Start of Main program

MAXLENGTH = 10


# Initialize the bands
bands = ["B1","B2","B3","B4","B5","B6","B7"]


with open("../MemeSpinInput/Points_1123.csv") as f:

    points = list(csv.reader(f))

vegetationPoints = []
nonvegetationPoints = []

pointSet = {'1':[],'2':[],'3':[],'4':[],'5':[]}

for point in points[1:]:
 
    pointSet[point[0]].append({"class": point[0] ,"B1":float(point[1]),"B2":float(point[2]),"B3":float(point[3]),"B4":float(point[4]),"B5":float(point[5]),"B6":float(point[6]),"B7":float(point[7])})

index = SpectralIndex.SpectralIndex(bands);
index.fitness,index.score,index.isLarger,index.isPointFive,index.isNotPointFive,index.lessThanTen,index.jmd = calculateFitness(index.index,bands,pointSet)

print("Initial Index")
index.display()
print(index.fitness)


improvedIndex = simulatedAnnealing(index,bands,pointSet)

print("Improved Index")
improvedIndex.display()
print(improvedIndex.fitness)





# DON'T DELETE

# index = SpectralIndex.SpectralIndex(bands);
# index.fitness,index.score,index.isLarger,index.isPointFive,index.isNotPointFive,index.lessThanTen,index.jmd = calculateFitness(index.index,bands,pointSet)
# index.display()
# print(index.fitness)

# tMax = 5000
# tMin = 1
# cooling = 0.3
# maxIterations = 10
# T = tMax;

# currentIndex = copy.deepcopy(index)
# nextIndex = copy.deepcopy(index)
# bestIndex = copy.deepcopy(index)

# while(T > tMin):

#     # print(T)
#     for i in range(maxIterations):
#         currentIndex.fitness,currentIndex.score,currentIndex.isLarger,currentIndex.isPointFive,currentIndex.isNotPointFive,currentIndex.lessThanTen,currentIndex.jmd = calculateFitness(currentIndex.index,bands,pointSet)
#         currentFitness = currentIndex.fitness
#         # print("Current",currentFitness)

#         nextIndex = copy.deepcopy(currentIndex)

#         # mutate(nextIndex.index,bands,nextIndex.length,MAXLENGTH,mutation_rate = T/tMax)
#         step(nextIndex.index,bands,nextIndex.length,MAXLENGTH,mutation_rate = 1.0)
#         nextIndex.length = nextIndex.count(nextIndex.index)
#         nextIndex.fitness,nextIndex.score,nextIndex.isLarger,nextIndex.isPointFive,nextIndex.isNotPointFive,nextIndex.lessThanTen,nextIndex.jmd = calculateFitness(nextIndex.index,bands,pointSet)
#         nextFitness = nextIndex.fitness

#         # print("Next",nextFitness)

#         deltaFitness = nextFitness - currentFitness
#         # print("Delta",deltaFitness)

#         if deltaFitness > 0:
#             currentIndex = copy.deepcopy(nextIndex)
#             if bestIndex.fitness < currentIndex.fitness:
#                 bestIndex = copy.deepcopy(currentIndex)            

#         elif (math.exp(deltaFitness/T)) > random.random():
#             # print("BLAG")
#             currentIndex = copy.deepcopy(nextIndex)


#     T *= 1 - cooling

# print("Current Index")
# currentIndex.display()
# print(currentIndex.fitness)

# print("Best Index")
# bestIndex.display()
# print(bestIndex.fitness)