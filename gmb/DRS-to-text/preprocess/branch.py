import sys
import json
import re


thematic_relations = [
	"Agent", "Experiencer", "Stimulus", "Theme", "Patient", 
	"Instrument", "Force", "Location", "Direction", "Goal",
	"Recipient", "Source", "Origin", "Time", "Beneficiary",
	"Manner", "Purpose", "Cause", "Rel"
	]

def recursive_bracket(bracket):
	stack = [[]]
	for tok in bracket:
		if tok[-1] == "(":
			stack.append([tok])
		elif tok == ")":
			back = stack.pop()
			stack[-1].append(back)
		else:
			stack[-1].append(tok)
	return stack[0][0]

def recursive_string(bracket):

	string = []
	def travel(b):
		if isinstance(b, str):
			string.append(b)
		else:
			string.append(b[0])
			for c in b[1:]:
				if c[0] == "OR(":
					c[0] = "DIS("
				travel(c)
			string.append(")")

	travel(bracket)
	return " ".join(string)

def right_branch(line, branch):
	tree = recursive_bracket(line.split())


	def traversal(parent):
		if isinstance(parent, list):
			if len(parent) - 1 > branch:
				new_child = parent[branch:]
				new_child.insert(0, "*"+ parent[0].replace("*",""))
				while len(parent) > branch:
					parent.pop()
				parent.append(new_child)
			for child in parent:
				traversal(child)

	traversal(tree)

	return recursive_string(tree)

def left_branch(line, branch):
	tree = recursive_bracket(line.split())


	def traversal(parent):
		if isinstance(parent, list):
			if len(parent) - 1 > branch:
				p = parent[0]
				new_child = parent[1:-branch+1]
				new_child.insert(0, "*"+ p.replace("*",""))
				parent.pop(0)
				while len(parent) >= branch:
					parent.pop(0)
				parent.insert(0, new_child)
				parent.insert(0, p)
			for child in parent:
				traversal(child)

	traversal(tree)

	return recursive_string(tree)


if __name__ == "__main__":

	branch = int(sys.argv[2])
	for line in open(sys.argv[1]):
		line = line.strip()
		# line = right_branch(line, branch)
		line = left_branch(line, branch)
		print(line)
		exit(-1)

		
