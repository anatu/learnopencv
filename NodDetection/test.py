class Node():
	def __init__(self, media, yChild, nChild):
		self.media = media
		self.yChild = yChild
		self.nChild = nChild
		# if (self.yChild == None or self.nChild == None) and (self.yChild != self.nChild):
		# 	raise AssertionError("Can't have only one None child")

	def __repr__(self):
		result = "Media: %s, Yes_Child: %s, No_Child: %s" % (self.media, self.yChild, self.nChild)
		return result

def build_decision_tree():
	tree = Node(1, "B", "C")
	return tree


if __name__ == "__main__":
	node = Node(1,2,3)
	print(node)