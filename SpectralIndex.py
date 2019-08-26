from __future__ import print_function
import random
import SpectralIndexNode

class SpectralIndex:

	# Initialize Random Spectral Index

	def __init__(self, bands, coefficient_probability = 0.1, unary_operator_probability = 0.1):

		self.operators = ["+","-","/","*"]
		self.unary_operators = ["-","sqrt","log"]
		self.bands = bands

		# Assumptions:
		# Basic spectral index is one binary operation with two (unique) bands
		# Coefficients of 1 are favored

		b0,b1 = random.sample(bands,2)
		op = random.choice(self.operators)

		# Prepare root node
		self.index = SpectralIndexNode.SpectralIndexNode(op)

		# Prepare left node


		coefficient = 1.0

		if random.random() < coefficient_probability:
			coefficient = random.random()

		unary_operator = ""

		if random.random() < unary_operator_probability:
			unary_operator = random.choice(self.unary_operators)

		self.index.left = SpectralIndexNode.SpectralIndexNode(b0,coefficient=coefficient,unary_op = unary_operator)
		self.index.left.parent = self.index

		# Prepare right node

		coefficient = 1.0

		if random.random() < coefficient_probability:
			coefficient = random.random()

		unary_operator = ""

		if random.random() < unary_operator_probability:
			unary_operator = random.choice(self.unary_operators)


		self.index.right = SpectralIndexNode.SpectralIndexNode(b1,coefficient=coefficient,unary_op = unary_operator)
		self.index.right.parent = self.index

		self.length = 3

		# print(self.index.right.parent)
		# print(self.index.left.parent)

	def inorderTraversal(self, root):

	    if root == None:
	        return

	    if root.left != None:
	        print("(", end='')
	    self.inorderTraversal(root.left)
	    print(root, end='')
	    self.inorderTraversal(root.right)

	    if root.right != None:
	        print(")", end='')


	def preorderTraversal(self, root):

	    if root == None:
	        return
	    print(root)     
	    self.preorderTraversal(root.left)
	    self.preorderTraversal(root.right)

	# https://stackoverflow.com/questions/33228660/how-to-count-the-total-number-of-nodes-in-binary-tree
	def count(self, root):
    
	    if root == None:
	        return 0
	    
	    else:
	        return 1 + self.count(root.left) + self.count(root.right)


	def pretty(self, root,depth):

	    if root == None:
	        return
	    
	    for i in range(depth):
	    	print("  ",end="")
	    print(root)     
	    
	    
	    self.pretty(root.left,depth+1)
	    self.pretty(root.right,depth+1)    

	    

	def display(self):

		self.inorderTraversal(self.index)
		print("")








