#!- encoding: utf-8 -
import os
import sys
import xml.etree.ElementTree as ET
import re
import editdistance

from utils import *
#from correction import solve_multiple_label
import json

def argument(arg, child):
	if re.match("^[xestpb][0-9]+$",arg):
		return arg.upper()
	elif arg in ["speaker", "hearer", "now"]:
		return "\""+arg+"\" [\""+arg+"\"]"
	else:
		f = int(child.attrib["from"])
		t = int(child.attrib["to"])
		if f >= t:
			return "\""+arg+"\" [\""+arg+"\"]"
		for item in sents:
			if item[0] == f and item[1] == t: #and arg == tok["sym"]:
					return item[3]+" [\""+arg+"\"]"
		assert False, "not legal aligns"

def logic_relations(parent):
	assert parent.tag == "relations", "function relations errors"

	for child in parent:
		logic.append(child.attrib["sym"].upper()+"( "+argument(child.attrib["arg1"],None)+" "+argument(child.attrib["arg2"],None)+" )")

def logic_constituent(parent):
	assert parent.tag == "constituent"

	#logic.append(parent.attrib["label"].upper()+"(")
	for child in parent:
		if child.tag == "sdrs":
			assert False, "error should no sdrs for sdrs"
			logic_sdrs(child)
		elif child.tag == "drs":
			logic_drs(child)
		else:
			assert False, "constituent confused"

	#logic.append(")")

def logic_sub(parent):
	assert parent.tag == "sub"

	for child in parent:
		if child.tag == "constituent":
			logic_constituent(child)
		else:
			assert False, "sub confused"

def logic_constituents(parent):
	assert parent.tag == "constituents"

	for child in parent:
		if child.tag == "sub":
			logic_sub(child)
		elif child.tag == "constituent":
			logic_constituent(child)
		else:
			assert False, "constituents confused"

def logic_sdrs(parent):
	assert parent.tag == "sdrs"
	assert len(parent) == 2

	logic.append("SDRS(")
	logic_constituents(parent[0])
	logic_relations(parent[1])
	logic.append(")")

def logic_prev(parent, tag, p):
	if tag == "prop":
		logic.append(parent.attrib["argument"].upper()+"(")
	elif tag == "not" or tag == "nec" or tag == "pos" or tag == "imp":
		logic.append(tag.upper()+"(")
	elif tag == "or":
		logic.append("DIS"+"(")
	elif tag == "duplex":
		logic.append("DUP"+"(")
	else:
		assert False, "prev errors"
	logic.append(p[:-1])
	for child in parent:
		if child.tag == "sdrs":
			logic_sdrs(child)
		elif child.tag == "drs":
			logic_drs(child)
		elif child.tag == "indexlist":
			pass
		else:
			assert False, "prev confused"

	logic.append(")")


def getName(parent):
	f = int(parent.attrib["from"])
	t = int(parent.attrib["to"])

	if f < 0 or t < 0 or f > t:
		return ""
	align = False
	for token in tokens:
		if int(token["from"]) == f and int(token["to"]) == t and editdistance.eval(token["lemma"], parent.attrib["symbol"]) * 1.0 / len(token["lemma"]) <= 0.3:
			align = True

	if align == False:
		return ""
	for item in sents:
		if item[0] == f and item[1] == t:
			return item[3]
	assert False, "error"

def getPred(parent):
	f = int(parent.attrib["from"])
	t = int(parent.attrib["to"])

	if f < 0 or t < 0 or f > t:
		return ""

	align = False
	for token in tokens:
		if int(token["from"]) == f and int(token["to"]) == t and editdistance.eval(token["lemma"], parent.attrib["symbol"]) * 1.0 / len(token["lemma"]) <= 0.3:
			align = True

	if align == False:
		return ""

	for item in sents:
		if item[0] == f and item[1] == t:
			return item[3]
	assert False, "error"
def getRel(parent):
	f = int(parent.attrib["from"])
	t = int(parent.attrib["to"])

	if f < 0 or t < 0 or f > t:
		return ""

	align = False
	for token in tokens:
		if int(token["from"]) == f and int(token["to"]) == t and editdistance.eval(token["lemma"], parent.attrib["symbol"]) * 1.0 / len(token["lemma"]) <= 0.3:
			align = True

	if align == False:
		return ""

	for item in sents:
		if item[0] == f and item[1] == t:
			return item[3]
	assert False, "error"

def getPointer(pointer):
	#print box_stack
	return pointer.upper()+" "
	if pointer != box_stack[-1]:
		return pointer.upper()+" "
	else:
		return "B0 "
def logic_cond(parent):
	assert parent.tag == "cond"

	child = parent[0]
	p = getPointer(parent.attrib["label"])
	if child.tag == "named":
		index = getName(child)
		assert index != ""
		tag = "Named("
		logic.append(tag+" "+p+" "+argument(child.attrib["arg"], child)+" "+index+" [\""+child.attrib["symbol"]+"\"]"+" )")
	elif child.tag == "pred":
		sym, typ, sen = child.attrib["synset"].split(".")
		assert len(sen) == 2
		index = getPred(child)
		if index != "":
			logic.append("Pred"+"( "+p+argument(child.attrib["arg"], child)+" "+index+" [\""+sym+"\"] \""+typ+"."+sen+"\" )")
		else:
			logic.append("Pred"+"( "+p+argument(child.attrib["arg"], child)+" \""+sym+"\" [\""+sym+"\"] \""+typ+"."+sen+"\" )")
	elif child.tag == "comp":
		logic.append("comp_"+child.attrib["symbol"]+"( "+p+argument(child.attrib["arg1"], child)+" "+argument(child.attrib["arg2"], child)+" )")
	
	elif child.tag == "rel":
		rel = child.attrib["symbol"]
		index = getRel(child)
		if index == "":
			if rel == "ClockTime":
				assert re.match("[0-9]{2}:[0-9]{2}",child.attrib["arg2"])
				if int(child.attrib["arg2"][:2]) < 12:
					rel = rel + "Am"
				else:
					rel = rel + "Pm"
			logic.append(rel+"( "+p+argument(child.attrib["arg1"], child)+ " "+ argument(child.attrib["arg2"], child)+" )")
		else:
			index = index.replace(" ","-")
			logic.append(index+"["+rel+"]"+"( "+p+argument(child.attrib["arg1"], child)+ " "+ argument(child.attrib["arg2"], child)+" )")

	elif child.tag == "prop":
		#print child.attrib["argument"]
		#assert p[:-1] == "B0"
		logic_prev(child, "prop", p)
	elif child.tag == "not":
		#assert p[:-1] == "B0"
		logic_prev(child, "not", p)
	elif child.tag == "pos":
		#assert p[:-1] == "B0"
		logic_prev(child, "pos", p)
	elif child.tag == "nec":
		#assert p[:-1] == "B0"
		logic_prev(child, "nec", p)
	elif child.tag == "duplex":
		#assert p[:-1] == "B0"
		logic_prev(child, "duplex", p)
	elif child.tag == "or":
		#assert p[:-1] == "B0"
		logic_prev(child, "or", p)
	elif child.tag == "imp":
		#assert p[:-1] == "B0"
		logic_prev(child, "imp", p)

	else:
		assert False, "cond confused"

def logic_conds(parent):
	assert parent.tag == "conds"

	for child in parent:
		if child.tag == "cond":
			logic_cond(child)
		else:
			assert False, "conds confused"
ext = []
def see_domain(parent):
	assert parent.tag == "domain"
	for child in parent:
		n = child.attrib["name"].upper()
		l = child.attrib["label"].upper()
		if n not in domain:
			domain[n] = l
		else:
			if domain[n] != l:
				domain[n] = l
				assert n[0] == "T", "error"

def get_sentidx(f, t):
	i = 1
	while i < len(boundary):
		if t <= boundary[i] and f > boundary[i-1]:
			return i
		i += 1
	assert False, "cross sentences"

box_stack = []
def logic_drs(parent):
	assert parent.tag == "drs"


	indexs = []
	def aligns(parent):
		if "from" in parent.attrib and "to" in parent.attrib:
			if int(parent.attrib["from"]) < int(parent.attrib["to"]):
				indexs.append(get_sentidx(int(parent.attrib["from"]), int(parent.attrib["to"])))
		for child in parent:
			aligns(child)
	aligns(parent)

	global cur_index, prev_index
	for i in range(len(indexs)-1):
		if indexs[i] != indexs[i+1]:
			print indexs
		assert indexs[i] == indexs[i+1]
		i += 1	

	if len(indexs) == 0:
		logic.append("DRS-"+str(prev_index-1)+"(")
		cur_index = prev_index - 1
	else:
		if prev_index+1 == indexs[0]:
			logic.append("DRS-"+str(prev_index)+"(")
			cur_index = prev_index
		elif prev_index == indexs[0]:
			logic.append("DRS-"+str(prev_index-1)+"(")
			cur_index = prev_index - 1
		else:
			assert False, "indexs are not continued"
		prev_index = indexs[0]
		#logic.append("DRS-"+indexs[0][0:-3]+"(")

	box_stack.append(parent.attrib["label"])
	logic.append(parent.attrib["label"].upper())
	#solve_multiple_label(parent, domain)
	for child in parent:
		if child.tag == "domain":
			see_domain(child)
		elif child.tag == "conds":
			logic_conds(child)
		elif child.tag == "tokens":
			pass
		elif child.tag == "taggedtokens":
			pass
		else:
			assert False, "drs confused"

	logic.append(")")
	box_stack.pop()

def out_sents(parent, filename):
	tokens = []
	for child in parent:
		for cc in child:
			tmp = {}
			for ccc in cc:
				tmp[ccc.attrib["type"]] = ccc.text

			if tmp["tok"].encode("utf-8") == "ø":
				pass
			else:
				tokens.append(tmp)

	idx = 0
	sents = []
	prev_sentid = -1
	for line in open(filename):
		line = line.strip()
		if line == "":
			continue
		line = line.split()
		f = int(line[0])
		t = int(line[1])
		n = len(line[3:])

		if line[2][0:-3] != prev_sentid:
			prev_sentid = line[2][0:-3]
			idx = 0
		m = ""
		i = 0
		while i < n:
			m += "$"+str(idx)+" "
			i += 1
			idx += 1
		# from, to, sentid, $1, words
		sents.append([f, t, int(line[2][0:-3]), m.strip(), " ".join(line[3:])])

	boundary = [-1]
	i = 1
	while i < len(sents):
		if sents[i][2] > sents[i-1][2]:
			boundary.append(sents[i-1][1])
		i += 1
	boundary.append(sents[-1][1])
	return tokens, sents, boundary

def normalized_p(tree):
	p = 1000
	v = []
	for i in range(len(tree)):
		if re.match("^[XEST][0-9]+\($", tree[i]):
			if tree[i][:-1] not in v:
				v.append(tree[i][:-1])

	for i in range(len(tree)):
		if re.match("^[XEST][0-9]+\($", tree[i]) and tree[i][:-1] in v:
			idx = v.index(tree[i][:-1])
			tree[i] = "P"+str(p+idx)+"("
		elif re.match("^[XEST][0-9]+$", tree[i]) and tree[i] in v:
			idx = v.index(tree[i])
			tree[i] = "P"+str(p+idx)

	keys = domain.keys()

	for key in keys:
		if key in v:
			idx = v.index(key)
			domain["P"+str(p+idx)] = domain[key]
			del domain[key]

def xmlReader(tokfilename, filename):
	tree = ET.parse(filename)
	root = tree.getroot()[0]
	
	assert len(root) == 2
	assert root[0].tag == "taggedtokens"
	assert (root[1].tag == "sdrs" or root[1].tag == "drs")

	global tokens
	global sents
	global boundary
	tokens, sents, boundary= out_sents(root[0], tokfilename)

	#print tokens
	#print sents
	#print boundary
	global cur_index, prev_index
	cur_index = -1
	prev_index = 1

	data_tokens = [[]]
	idx = 1
	for item in sents:
		assert item[2] == idx or item[2] == idx + 1
		if item[2] == idx:
			data_tokens[-1].append(item[-1])
		else:
			idx = item[2]
			data_tokens.append([item[-1]])
	

	global logic
	logic = []
	global domain
	domain = {}
	if root[1].tag == "sdrs":
		logic_sdrs(root[1])
	else:
		logic_drs(root[1])
	tree = " ".join(logic).encode("UTF-8")
	tree = tree.split()
	normalized_p(tree)

	# print "VARIABLE"
	data_variables = []
	for key in domain.keys():
	# 	print key.upper()+":"+domain[key].upper(),
		data_variables.append(key.upper()+":"+domain[key].upper())
	# print 
	# print "TREE"
	data_tree = " ".join(tree)
	# print " ".join(tree)
	# print 

	data = {"tokens":[" ".join(t) for t in data_tokens], "variables": " ".join(data_variables), "tree": data_tree}

	return data
if __name__ == "__main__":
	# print "###", " ".join(sys.argv)
	data = xmlReader(sys.argv[1], sys.argv[2])
	data["command"] = " ".join(sys.argv)

	json.dump(data, sys.stdout, indent=4)
#if __name__ == "__main__":
#	for (path, dirs, files) in os.walk(sys.argv[1]):
#		if len(files) != 0:
#			print path
#			tokens = []
#			newpath = "data/"+path.split("/")[-2]+"/"+path.split("/")[-1]
#			if not os.path.exists(newpath):
#				os.makedirs(newpath)
#			out = open(newpath+"/en.logic", "w")
#			xmlReader(path+"/en.drs.xml.notime.index_normalized",out)
#			out.close()
