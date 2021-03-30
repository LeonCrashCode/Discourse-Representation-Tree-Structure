
import sys
import types
import logging
from utils import *
import json

def get_all_variable(root):
	variables = set()
	def travel(item):
		if is_cond_ref_node(item):
			pass
		elif is_drel_node(item):
			pass
		elif is_cond_name_node(item) or is_cond_pred_node(item) or is_cond_card_node(item) or is_cond_time_node(item):
			variables.add(item[2])
		elif is_prop_scope_node(item):
			variables.add(item[0][:-1])
		elif is_cond_other_node(item):
			# print(item)
			assert len(item[2:]) == 2
			variables.add(item[2])
			variables.add(item[3])
		for child in item:
			if type(child) == types.ListType:
				travel(child)
	travel(root)
	return list(variables)

def get_ids(vs, nodes_variables):
	ids = []
	for i, variable in enumerate(nodes_variables):
		t, variable = variable[0], variable[1:]
		if t == 0 and variable[0] in vs:
			ids.append(i)
	return ids

graph = []
def create_graph(root):
	nodes = []
	nodes_variables = []
	for item in root:
		if is_cond_ref_node(item):
			continue
		elif is_cond_name_node(item) or is_cond_pred_node(item) or is_cond_card_node(item) or is_cond_time_node(item):
			nodes.append(item[0]+" "+" ".join(item[3:]))
			nodes_variables.append([0, item[2]])
		elif is_prop_scope_node(item):
			nodes.append(item[0])
			nodes_variables.append([0, item[0][:-1]])
		elif is_unary_scope_node(item) or is_binary_scope_node(item):
			nodes.append(item[0])
			nodes_variables.append([1] + get_all_variable(item))
		elif is_cond_other_node(item):
			nodes.append(item[0])
			assert len(item[2:]) == 2
			nodes_variables.append([2] + item[2:])
		else:
			continue
	# print("====")
	# print(nodes)
	# print(nodes_variables)


	# 0: name or pred or proposition, 1: scope, 2: relation
	extended_nodes = []
	extended_nodes_variables = []

	# for variable in nodes_variables:
	# 	t, variable = variable[0], variable[1:]
	# 	if t == 2:
	# 		ids = get_ids([variable[0]], nodes_variables + extended_nodes_variables)
	# 		if len(ids) == 0:
	# 			extended_nodes.append("DummyNode")
	# 			extended_nodes_variables.append([0, variable[0]])
	# 		ids = get_ids([variable[1]], nodes_variables + extended_nodes_variables)
	# 		if len(ids) == 0:
	# 			extended_nodes.append("DummyNode")
	# 			extended_nodes_variables.append([0, variable[1]])

	# for variable in nodes_variables:
	# 	t, varibale = variable[0], variable[1:]
	# 	if t == 1:
	# 		ids = get_ids(variable, nodes_variables + extended_nodes_variables)
	# 		assert len(ids) != 0, "scope should be there"

	nodes += extended_nodes
	nodes_variables += extended_nodes_variables
	# print("====")
	# print(nodes)
	# print(nodes_variables)
	
	d = {"nodes": nodes, "nodes_variables": nodes_variables}

	edges = []

	for i, variable in enumerate(nodes_variables):
		t, variable = variable[0], variable[1:]
		if t == 1:
			ids = get_ids(variable[0], nodes_variables)
			for id in ids:
				edges.append(" ".join([str(id), str(i), "EDGE_related"]))
		elif t == 2:
			ids = get_ids([variable[0]], nodes_variables)
			for id in ids:
				edges.append(" ".join([str(i), str(id), "EDGE_arg0"]))
			ids = get_ids([variable[1]], nodes_variables)
			for id in ids:
				edges.append(" ".join([str(i), str(id), "EDGE_arg1"]))
		elif t == 0:
			ids = get_ids([variable[0]], nodes_variables)
			for id in ids:
				if id == i:
					continue
				edges.append(" ".join([str(i), str(id), "EDGE_equ"]))
		else:
			print(t)			
			assert False, "no other types except 0, 1, 2"
	# print("====")
	# print(edges)
	
	edges = []
	for i in range(len(nodes)):
		for j in range(len(nodes)):
			if i == j:
				continue
			edges.append(" ".join([str(i), str(j), "EDGE_highway"]))
	graph.append({"nodes": " ||| ".join(nodes), "nodes_variables": " ||| ".join([ " ".join([str(item[0])] + item[1:]) for item in nodes_variables]), "edges": " ||| ".join(edges)})

def travel(root):
	if is_drs_node(root):
		# print(root)
		create_graph(root)
	for item in root:
		if type(item) == types.ListType:
			travel(item)

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
data = json.load(open(sys.argv[1]))
i = 0
for inst in data["data"]:
	bracket = inst["basic"].encode("utf-8").split()
	root = recursive_bracket(bracket)
	travel(root)
	i += 1
	if i % 500 == 0:
		logging.info("processing -" + str(i))
logging.info("total case - " + str(len(graph)))
json.dump({"data": graph}, sys.stdout, indent=4)




