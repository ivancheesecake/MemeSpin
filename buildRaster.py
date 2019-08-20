import SpectralIndex
import random
import math
import copy
import csv
import numpy as np
import operator
import pickle
from skimage import io

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


index = pickle.load(open("Open_0819_Test2.index","rb"))
print("VEG!")

B1 = io.imread("input/B1.tif")
B2 = io.imread("input/B2.tif")
B3 = io.imread("input/B3.tif")
B4 = io.imread("input/B4.tif")
B5 = io.imread("input/B5.tif")
B6 = io.imread("input/B6.tif")
B7 = io.imread("input/B7.tif")

output = copy.deepcopy(B1)


for y in range(len(B1)):
	for x in range(len(B1[0])):
		# print(y,x)
		# print(evaluate(index.index, 
		# 	{"B1":float(B1[y][x]),
		# 	"B2":float(B2[y][x]),
		# 	"B3":float(B3[y][x]),
		# 	"B4":float(B4[y][x]),
		# 	"B5":float(B5[y][x]),
		# 	"B6":float(B6[y][x]),
		# 	"B7":float(B7[y][x])}))

		b1 = B1[y][x]
		b2 = B2[y][x]
		b3 = B3[y][x]
		b4 = B4[y][x]
		b5 = B5[y][x]
		b6 = B6[y][x]
		b7 = B7[y][x]

		# if b1 < 0.01:
		# 	b1 = 0
		# if b2 < 0.01:
		# 	b2 = 0
		# if b3 < 0.01:
		# 	b3 = 0
		# if b4 < 0.01:
		# 	b4 = 0	
		# if b5 < 0.01:
		# 	b5 = 0
		# if b6 < 0.01:
		# 	b6 = 0
		# if b7 < 0.01:
		# 	b7 = 0

		value = evaluate(index.index, 
			{"B1":float(b1),
			"B2":float(b2),
			"B3":float(b3),
			"B4":float(b4),
			"B5":float(b5),
			"B6":float(b6),
			"B7":float(b7)})

		# print(value)

		if value > 1:
			value = 1

		if value < -1:
			value = -1	

		output[y][x] = value



io.imsave("Open_0819_Test2.tif",output)

print(index)

