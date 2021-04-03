import sys
import types
import logging
from utils import *
import json
def get_docs(filename):
	docs = [[]]
	for line in open(filename):
		line = line.strip()
		if line == "":
			docs.append([])
		else:
			docs[-1].append(line)
	if len(docs[-1]) == 0:
		docs.pop()
	return docs

def get_src_var_tree(doc):
	src = []
	i = 0
	while i < len(doc):
		if doc[i].startswith("###"):
			i += 1
			continue
		if doc[i] == "VARIABLE":
			break
		src.append(doc[i])
		i += 1
	i += 1
	assert i == len(doc)-3

	var_l = [item.split(":") for item in doc[i].split()]
	tree = doc[i+2].split()

	var = {}
	for v in var_l:
		var[v[0]] = v[1]
	return src, var, tree


def simplify_SDRS(bracket):

	def travel(b):
		if is_sdrs_node(b):
			#check children of SDRS
			idx = 0
			mp = {}
			for c in b[1:]:
				if is_drs_node(c):
					mp[c[1]] = str(idx)
					idx += 1
			# relative to K
			for c in b[1:]:
				if is_drel_node(c):
					#print c[1], c[2], mp
					assert c[1] in mp and c[2] in mp
					c[1] = "K"+mp[c[1]]
					c[2] = "K"+mp[c[2]]
			# delete K
			i = 1
			while i < len(b):
				if is_constituent_node(b[i]):
					b[i] = b[i][1]
				i += 1

		for c in b:
			if type(c) == types.ListType:
				travel(c)
	travel(bracket)

def add_conditionXEST(bracket ,var):
	exist = []
	def travel(b):
		if is_drs_node(b):
			i = 2
			while i < len(b):
				if is_cond_node(b[i]):
					add_v = []
					for v in b[i][2:]:
						if is_v(v, "XEST") and v not in exist:
							assert v in var
							add_v.append(v)
							exist.append(v)
					for v in add_v:
						b.insert(i, ["Ref(", var[v], v])
						i += 1
				elif type(b[i]) == types.ListType:
					travel(b[i])
				i += 1
		else:
			for c in b:
				if type(c) == types.ListType:
					travel(c)
	travel(bracket)

def add_conditionP(bracket ,var):
	exist = []
	def travel(b):
		if is_drs_node(b):
			i = 2
			while i < len(b):
				if is_cond_node(b[i]):
					add_v = []
					for v in b[i][2:]:
						if is_v(v, "P") and v not in exist:
							assert v in var
							add_v.append(v)
							exist.append(v)
					for v in add_v:
						b.insert(i, ["Ref(", var[v], v])
						i += 1
				elif type(b[i]) == types.ListType:
					if is_prop_scope_node(b[i]) and b[i][0][:-1] not in exist:
						assert b[i][0][:-1] in var
						exist.append(b[i][0][:-1])
						b.insert(i, ["Ref(", var[b[i][0][:-1]], b[i][0][:-1]])
						i += 1
					travel(b[i])
				i += 1
		else:
			for c in b:
				if type(c) == types.ListType:
					travel(c)
	travel(bracket)

def relative_DRS(bracket):

	
	mp_l = []
	mp = {}
	def travel1(b):
		if is_drs_node(b):
			mp_l.append(b[1])
		for c in b:
			if type(c) == types.ListType:
				travel1(c)

	travel1(bracket)
	for i, v in enumerate(mp_l):
		mp[v] = i

	outside_box = []
	def travel(b):
		if is_drs_node(b):
			p = b[1]
			for c in b[2:]:
				# current box
				if c[1] == p:
					c[1] = "B0"
				#outside box
				elif c[1] not in mp:
					if c[1] in outside_box:
						c[1] = "O" + str(outside_box.index(c[1]) + 1)
					else:
						outside_box.append(c[1])
						c[1] = "O" + str(len(outside_box))
				#relative to inside box
				else:
					rel_idx = mp[p] - mp[c[1]]
					c[1] = "B"+str(rel_idx)
			del b[1]
		for c in b:
			if type(c) == types.ListType:
				travel(c)
	travel(bracket)

def relative_varXEST(string):
	exist_ref = [[], [], [], []] # XEST

	tokens = string.split()
	i = 0
	while i < len(tokens):
		if tokens[i] == "Ref(":
			for j, v in enumerate(["X", "E", "S", "T"]):
				if is_v(tokens[i+2], v):
					assert tokens[i+2] not in exist_ref[j]
					exist_ref[j].append(tokens[i+2])
					tokens[i+2] = tokens[i+2][0]
			i += 4
		else:
			for j, v in enumerate(["X", "E", "S", "T"]):
				if is_v(tokens[i], v):
					assert tokens[i] in exist_ref[j]
					tokens[i] = v + str(len(exist_ref[j]) - exist_ref[j].index(tokens[i]) - 1)
			i += 1
	return " ".join(tokens)

def relative_varP(string):

	exist_ref = [] # XEST

	tokens = string.split()
	i = 0
	while i < len(tokens):
		if tokens[i] == "Ref(":
			if is_v(tokens[i+2], "P"):
				assert tokens[i+2] not in exist_ref
				exist_ref.append(tokens[i+2])
				tokens[i+2] = tokens[i+2][0]
			i += 4
		else:
			if is_v(tokens[i], "P"):
				assert tokens[i] in exist_ref
				tokens[i] = "P" + str(len(exist_ref) - exist_ref.index(tokens[i]) - 1)
			elif is_v(tokens[i][:-1], "P"):
				assert tokens[i][:-1] in exist_ref
				tokens[i] = "P" + str(len(exist_ref) - exist_ref.index(tokens[i][:-1]) - 1) + "("
			i += 1
	return " ".join(tokens)

def relative_varO(string):

	exist_ref = [] # XEST

	tokens = string.split()
	i = 0
	while i < len(tokens):
		if is_v(tokens[i], "O"):
			if tokens[i] not in exist_ref:
				exist_ref.append(tokens[i])
				tokens[i] = tokens[i][0]
			else:
				tokens[i] = "O" + str(len(exist_ref) - exist_ref.index(tokens[i]) - 1)
		i += 1
	return " ".join(tokens)


def recursive_string(bracket):

	string = []
	def travel(b):
		if type(b) == types.StringType or type(b) == types.UnicodeType:
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

def align(string, op=None):
	tokens = string.split()
	n_tokens = []
	assert op in ["align", None]

	for tok in tokens:
		if tok[0] == "[" and tok[-1] == "]":
			if op == None:
				n_tokens.append(tok[1:-1])
			else:
				pass
		elif is_sense(tok):
			n_tokens.append(tok)
		elif len(tok) >= 2 and tok[0] == '"' and tok[-1] == '"':
			if op == None:
				pass
			else:
				n_tokens.append(tok)
		elif is_align_var(tok):
			if op == None:
				pass
			else:
				n_tokens.append(tok)
		elif is_align_cond(tok):
			idx = tok.find("[")
			if op == None:
				n_tokens.append(tok[idx+1:-2]+"(")
			else:
				n_tokens.append(tok[:idx]+"(")
		else:
			n_tokens.append(tok)

	i = 0
	while i < len(n_tokens):
		n_tokens[i] = deQuation(n_tokens[i])
		i += 1

	# i = 0
	# while i < len(n_tokens):
	# 	n_tokens[i] = deDRS(n_tokens[i])
	# 	i += 1
	return " ".join(n_tokens)

def deQuation(v):
	if len(v) >= 3 and v[0] == "\"" and v[-1] == "\"":
		return v[1:-1]
	return v

def deDRS(v):
	if re.match("^DRS-[0-9]+\($", v):
		return "DRS("
	return v

def combine_sense(string):
	tokens = string.split()
	newtokens = []
	i = 0
	while i < len(tokens):
		if tokens[i] == "Pred(":
			newtokens.append(tokens[i])
			i += 1
			newtokens.append(tokens[i])
			i += 1
			newtokens.append(tokens[i])
			i += 1
			assert is_sense(tokens[i+1])
			newtokens.append(".".join([tokens[i], tokens[i+1]]))
			i += 2
		else:
			newtokens.append(tokens[i])
			i += 1
	return " ".join(newtokens)

if __name__ == "__main__":
	logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
	data = json.load(open(sys.argv[1], "r"))
	logging.info("document size - "+str(len(data["data"])))

	cnt = 0
	for inst in data["data"]:
		
		src = [t.encode("utf-8") for t in inst["tokens"]]

		tmp = inst["variables"].encode("utf-8").split()
		var = {}
		for t in tmp:
			t = t.split(":")
			var[t[0]] = t[1]

		tree = inst["tree"].encode("utf-8").split()
		
		bracket = recursive_bracket(tree)
		simplify_SDRS(bracket)
		add_conditionXEST(bracket, var)
		add_conditionP(bracket, var)

		basic = recursive_string(bracket)

		relative_DRS(bracket)
		string = recursive_string(bracket)
		string = relative_varXEST(string)
		string = relative_varP(string)
		string = relative_varO(string)

		relative = string

		inst["basic_align"] = align(basic, "align")
		inst["basic_sents"] = align(basic, None)

		inst["relative_align"] = align(relative, "align")
		inst["relative_sents"] = align(relative, None)

		inst["basic"] = " ".join([deDRS(v) for v in inst["basic_sents"].split()])
		inst["relative"] = " ".join([deDRS(v) for v in inst["relative_sents"].split()])

		inst["basic_sense"] = combine_sense(inst["basic"])
		inst["relative_sense"] = combine_sense(inst["relative"])
		inst["basic_sents_sense"] = combine_sense(inst["basic_sents"])
		inst["relative_sents_sense"] = combine_sense(inst["relative_sents"])

		cnt += 1
		if cnt % 500 == 0:
			logging.info("processing -" + str(cnt))

	json.dump(data, sys.stdout, indent=4)
	

