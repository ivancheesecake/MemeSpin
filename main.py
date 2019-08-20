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


def expandIndex(index,index2,isLeft):

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
            expandIndex(index.left,index2,True)
        else:
            expandIndex(index.right,index2,False)

def mutateBasic(index,bands,mutation_rate = 1.0):

    operators = ["+","-","/","*"]
    unary_operators = ["-","sqrt","log"]


    if index == None:
        return

    if random.random() < mutation_rate:

        if index.operation:
            index.value = random.choice(operators)
        else:
            index.unary_op = random.choice(unary_operators)
            index.value = random.choice(bands)
            index.coefficient = random.random()
            index.update_display_coefficient()

    mutateBasic(index.left,bands,mutation_rate)
    mutateBasic(index.right,bands,mutation_rate)

def mutate(index,bands,length,maxlength,mutation_rate = 0.05):


    if(random.random() < mutation_rate):

        if random.random() < 0.5:
            # print("BASIC")
            mutateBasic(index,bands,mutation_rate = 0.5)
        
        else:
            # print("EXPAND")
            if length < maxlength:
                bools = [True,False]
                index2 = SpectralIndex.SpectralIndex(bands)
                expandIndex(index,index2.index,random.choice(bools))

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
    selectedClass = '5'

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
    return ((0.25 * score) + (0.25 * (jmd/2)) + (0.1 * isPointFive) + (0.1 * isNotPointFive) + (0.3 * lessThanTen)),score,isLarger,isPointFive,isNotPointFive,lessThanTen,jmd
    # return silhouette




def bamratatata(population):

    fitness = [individual.fitness for individual in population]
    # print(fitness)
    index, value = max(enumerate(fitness), key=operator.itemgetter(1))  

    return population[index],population[index].fitness


def selectRoulette(population):

    population.sort(key=lambda x: x.fitness, reverse=True)
    totalFitness = 0
    
    # Normalize fitness
    # Calculate the sum of fitnesses
    for p in population:
        # print(p.fitness)
        totalFitness += p.fitness

    # Actual normalization 
    # print("Normalized")
    for i in range(len(population)):
        population[i].normalizedFitness += population[i].fitness/totalFitness
        # totalNormalizedFitness +=  population[i].fitness/totalFitness

        # print(population[i].normalizedFitness)
    roulette = random.random()
    # print("roulette: "+str(roulette))


    totalNormalizedFitness = 0

    selected = 0
    while totalNormalizedFitness<roulette:
        totalNormalizedFitness += population[selected].normalizedFitness
        selected +=1

    # print(selected)
    if(selected >=POPSIZE):
        selected = POPSIZE - 1
    return population[selected]


# Start of Main program

# Initialize the bands
bands = ["B1","B2","B3","B4","B5","B6","B7"]


with open("input/Points_1123.csv") as f:

    points = list(csv.reader(f))

vegetationPoints = []
nonvegetationPoints = []

pointSet = {'1':[],'2':[],'3':[],'4':[],'5':[]}


for point in points[1:]:
    # if point[0] == '1':
        # vegetationPoints.append({"B1":float(point[1]),"B2":float(point[2]),"B3":float(point[3]),"B4":float(point[4]),"B5":float(point[5]),"B6":float(point[6]),"B7":float(point[7])})
    # else:    
        # nonvegetationPoints.append({"B1":float(point[1]),"B2":float(point[2]),"B3":float(point[3]),"B4":float(point[4]),"B5":float(point[5]),"B6":float(point[6]),"B7":float(point[7])})
    pointSet[point[0]].append({"class": point[0] ,"B1":float(point[1]),"B2":float(point[2]),"B3":float(point[3]),"B4":float(point[4]),"B5":float(point[5]),"B6":float(point[6]),"B7":float(point[7])})


# print(pointSet['1'])


index = SpectralIndex.SpectralIndex(bands);
# print(calculateFitnessSilhouette(index.index,bands,pointSet))
# index.fitness = calculateFitnessSilhouette2(index.index,bands,pointSet)

MAXLENGTH = 10

# INITIALIZE GA PARAMETERS

GIRLSGENERATIONS = 2000
POPSIZE = 50
CROSSOVER_RATE = 0.90
MUTATION_RATE = 0.90


bestEver = ""

# INITIALIZE POPULATION

population = []

for p in range(POPSIZE):
    # print(p)
    index = SpectralIndex.SpectralIndex(bands);
    index.fitness,index.score,index.isLarger,index.isPointFive,index.isNotPointFive,index.lessThanTen,index.jmd = calculateFitness(index.index,bands,pointSet)
    index.normalizedFitness = 0
    population.append(index)
    
bestEver = population[0]

# bestEver.pretty(bestEver.index,0);
# print(bestEver.count(bestEver.index))


# START OF GENETIC ALGORITHM


for generation in range(GIRLSGENERATIONS):

    print("Generation",generation)
    newPopulation = []

    # Humayo kayo at magpakarami

    for i in range(int(POPSIZE/2)):

        mommy = selectRoulette(population)
        daddy = selectRoulette(population)

        # Insert length constraint here

        child1,child2 = basicCrossover(mommy,daddy,CROSSOVER_RATE)
        mutate(child1.index,bands,child1.length,MAXLENGTH,mutation_rate = MUTATION_RATE)
        mutate(child2.index,bands,child2.length,MAXLENGTH,mutation_rate = MUTATION_RATE)
        
        child1.length = child1.count(child1.index)
        child2.length = child2.count(child2.index)

        child1.fitness,child1.score,child1.isLarger,child1.isPointFive,child1.isNotPointFive,child1.lessThanTen,child1.jmd = calculateFitness(child1.index,bands,pointSet)
        child2.fitness,child2.score,child2.isLarger,child2.isPointFive,child2.isNotPointFive,child2.lessThanTen,child2.jmd = calculateFitness(child2.index,bands,pointSet)
        child1.normalizedFitness = 0
        child2.normalizedFitness = 0


        newPopulation.append(child1)
        newPopulation.append(child2) 


    population = copy.deepcopy(newPopulation)
    # print(len(population))
    
    currentBest,currentBestFitness =  bamratatata(population)
    # bamratatata(population)[0].display()
    # print(bamratatata(population)[1])
    print("Current Best:")
    currentBest.display()
    print(currentBestFitness)
    print("==================================")

    if currentBest.fitness > bestEver.fitness:

        bestEver = currentBest

    print("Best Ever:")
    bestEver.display()
    print(bestEver.fitness)
    print("---------------------")
    print("Silhouette Score:",bestEver.score)
    print("Jeffries-Matusita Distance:",bestEver.jmd)
    print("Is class mean larger?:",bestEver.isLarger)
    print("Is class mean > 0.5?:",bestEver.isPointFive)
    print("Is non class mean < 0.5:",bestEver.isNotPointFive)
    print("Are all calculated values within [-1,1]?",bestEver.lessThanTen)
    print("Index Length",bestEver.length)
    print("==================================")    


# bestIndex,fitness = bamratatata(population)

print("Best Ever Talaga:")

bestIndex = bestEver

bestIndex.display()
print(bestIndex.fitness)

pickle.dump(bestIndex,open("Open_0819_Test2.index","wb"))


