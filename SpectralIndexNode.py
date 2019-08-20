class SpectralIndexNode:

	def __init__(self,value,coefficient=1.0,unary_op=""):

		self.parent = None
		self.left = None
		self.right = None
		self.value = value
		self.coefficient = coefficient
		if coefficient == 1.0:
			self.display_coefficient = ""
		else:
			self.display_coefficient = str(round(self.coefficient, 2))

		self.unary_op = unary_op

		if self.value in "+-/*":
			self.operation = True
		else:
			self.operation = False

	# def insertChildren(self,child1,child2):

	def update_display_coefficient(self):

		if self.coefficient == 1.0:
			self.display_coefficient = ""
		else:
			self.display_coefficient = str(round(self.coefficient, 2))



	def __repr__(self):

		if self.operation:
			return self.value
		
		return self.unary_op + "" + self.display_coefficient + self.value


