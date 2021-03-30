import os
import sys
import xml.etree.ElementTree as ET
import re
from utils import *
import json

def logic_relations(parent):
	assert parent.tag == "relations", "function relations errors"

	for child in parent:
		logic.append(child.attrib["sym"].upper()+"( "+argument(child.attrib["arg1"].upper())+" "+argument(child.attrib["arg2"].upper())+" )")

def logic_constituent(parent):
	assert parent.tag == "constituent"

	logic.append(parent.attrib["label"].upper()+"(")
	for child in parent:
		if child.tag == "sdrs":
			logic_sdrs(child)
		elif child.tag == "drs":
			logic_drs(child)
		else:
			assert False, "constituent confused"

	logic.append(")")

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

lemmas = []
multiple_lemmas = []
def getCard(parent):
	child = parent[0]
	l = []
	for cc in child:
		pos1 = int(cc.text[1:-3])
		pos2 = int(cc.text[-3:])
		assert pos1-1 == cur_index
		l.append(pos2-1)
	assert len(l) > 0
	return  " ".join(["$"+str(index) for index in l])

def getTime(parent):
	child = parent[0]

	l = []
	pos1 = 999
	for cc in child:
		pos1 = int(cc.text[1:-3]) - 1
		pos2 = int(cc.text[-3:])
		assert pos1 == cur_index
		l.append(pos2-1)
	assert len(l) > 0
	
	
	date = parent[1].text
	
	assert len(date) == 9
	Year = date[1:5]
	Month = date[5:7]
	Day = date[7:9]
	
	Year_ex = Month_ex = Day_ex = 1
	
	if Year == "XXXX":
		Year_ex = 0
	if Month == "XX":
		Month_ex = 0
	if Day == "XX":
		Day_ex = 0
	
	ex = Year_ex + Month_ex + Day_ex 
	assert ex !=0

	s = "T"
	if Year_ex == 1:
		s += "y"
	else:
		s += "x"

	if Month_ex == 1:
		s += "m"
	else:
		s += "x"
	
	if Day_ex == 1:
		s += "d"
	else:
		s += "x"
	return s+"(", " ".join(["$"+str(index) for index in l])
	"""
	if Year_ex == 1:
		for pos2 in l:
			if isYear(words[pos1][pos2]):
				timex.append([pos2, Year, pos2])
				break
		assert len(timex) == 0
	else:
		timex.append([])
	if Month_ex == 1:
		for pos2 in l:
			if isMonth(words[pos1][pos2]):
				timex.append([pos2, Year, pos2])
				break
		assert len(timex) == 1
	else:
		timex.append([])

	if day_ex == 1:
		for pos2 in l:
			if isMonth(words[pos1][pos2]):
				timex.append([pos2, Year, pos2])
				break
			assert len(timex) == 2
	else:
		timex.append([])
	return  " ".join(["$"+str(index) for index in l])
	"""
def getName(parent):
	child = parent[0]
	l = []
	for cc in child:
		pos1 = int(cc.text[1:-3])
		pos2 = int(cc.text[-3:])
		assert pos1-1 == cur_index
		l.append((lemmas[pos1-1][pos2-1],pos2-1))
	if len(l) == 1:
		assert parent.attrib["symbol"] == l[0][0]
		return [l[0][1]]
	elif len(l) == 0:
		assert parent.attrib["symbol"] in lemmas[cur_index]
		d = multiple_lemmas[cur_index]
		come = d[parent.attrib["symbol"]]
		index = -1
		for i, v in enumerate(lemmas[cur_index]):
			if parent.attrib["symbol"] == v:
				index = i
				if come == 0:
					break
				come -= 1
		assert index != -1
		d[parent.attrib["symbol"]] += 1
		return [index]
	else:
		# assert False, "more than two indexs" 
		# print "out of", sys.argv[1]
		cand = "_".join([lem[0] for lem in l])
		#assert parent.attrib["symbol"] == cand
		return [lem[1] for lem in l]
	return  " ".join(l)

def getPred(parent):
	child = parent[0]
	l = []
	for cc in child:
		pos1 = int(cc.text[1:-3])
		pos2 = int(cc.text[-3:])
		assert pos1-1 == cur_index
		l.append((lemmas[pos1-1][pos2-1],pos2-1))

	if len(l) == 1:
		if parent.attrib["symbol"] == l[0][0]:
			return l[0][1]
		else:
			return -1
	elif len(l) == 0:
		if parent.attrib["symbol"] not in lemmas[cur_index]:
			return -1
		assert parent.attrib["symbol"] in lemmas[cur_index]
		d = multiple_lemmas[cur_index]
		come = d[parent.attrib["symbol"]]
		index = -1
		for i, v in enumerate(lemmas[cur_index]):
			if parent.attrib["symbol"] == v:
				index = i
				if come == 0:
					break
				come -= 1
		d[parent.attrib["symbol"]] += 1
		return index
	else:
		assert False, "more than two indexs" 
		print("out of", sys.argv[1])
def getRel(parent):
	child = parent[0]
	l = []
	for cc in child:
		pos1 = int(cc.text[1:-3])
		pos2 = int(cc.text[-3:])
		assert pos1-1 == cur_index
		l.append((lemmas[pos1-1][pos2-1], pos2-1))
	if len(l) == 1:
		if parent.attrib["symbol"] == l[0][0]:
			return l[0][1]
		else:
			return -1
	elif len(l) == 0:
		return -1
	else:
		assert False, "more than two indexs" 
		print("out of", sys.argv[1])
def argument(v):
	if vk.match(v):
		return v
	assert v in domain
	return v
	return v+" "+domain[v]
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
		#assert index != -1
		assert len(index) != 0
		tag = "Named_"+child.attrib["class"].upper()+"_"+child.attrib["type"].upper()+"(" 
		#logic.append(tag+" "+p+" "+argument(child.attrib["arg"].upper())+" "+"$"+str(index)+" [\""+child.attrib["symbol"]+"\"]"+" )")
		logic.append(tag+" "+p+" "+argument(child.attrib["arg"].upper())+" "+" ".join(["$"+str(i) for i in index])+" [\""+child.attrib["symbol"]+"\"]"+" )")
	elif child.tag == "pred":
		sense = child.attrib["sense"]
		if len(sense) == 1:
			sense = "0"+sense
		assert len(sense) == 2
		index = getPred(child)
		if index != -1:
			logic.append("Pred"+"( "+p+argument(child.attrib["arg"].upper())+" "+"$"+str(index)+" [\""+child.attrib["symbol"]+"\"] \""+child.attrib["type"]+"."+sense+"\" )")
		else:
			logic.append("Pred"+"( "+p+argument(child.attrib["arg"].upper())+" \""+child.attrib["symbol"]+"\" [\""+child.attrib["symbol"]+"\"] \""+child.attrib["type"]+"."+sense+"\" )")
	elif child.tag == "card":
		#logic.append("CARD( "+argument(child.attrib["arg"].upper()) + " " + child.attrib["value"]+" )")
		value = getCard(child)
		logic.append("Card( "+p+argument(child.attrib["arg"].upper()) + " " + value+" [\""+child.attrib["value"]+"\"]"+" )")
	elif child.tag == "timex":
		#logic.append("TIMEX( "+argument(child.attrib["arg"].upper()) + " " + child[1].text+" )")
		tag, value = getTime(child)
		logic.append(tag+" "+p+argument(child.attrib["arg"].upper()) + " " + value+" [\""+child[1].text+"\"]"+ " )")

	elif child.tag == "eq":
		logic.append("Equ( "+p+argument(child.attrib["arg1"].upper()) + " "+argument(child.attrib["arg2"].upper())+" )")
	elif child.tag == "rel":
		rel = child.attrib["symbol"]
		index = getRel(child)
		if index == -1:
			logic.append(rel[0].upper()+rel[1:]+"( "+p+argument(child.attrib["arg1"].upper()) + " "+ argument(child.attrib["arg2"].upper())+" )")
		else:
			logic.append("$"+str(index)+"["+rel+"]"+"( "+p+argument(child.attrib["arg1"].upper()) + " "+ argument(child.attrib["arg2"].upper())+" )")
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

def see_domain(parent):
	assert parent.tag == "domain"
	for child in parent:
		n = child.attrib["name"].upper()
		l = child.attrib["label"].upper()
		if n not in domain:
			domain[n] = l
		else:
			if domain[n] != l:
				print("error", n)

box_stack = []
prev_index = 1
cur_index = -1
def logic_drs(parent):
	assert parent.tag == "drs"
	indexs = []
	def aligns(parent):
		if parent.tag == "index":
			indexs.append(parent.text)
		for child in parent:
			aligns(child)

	aligns(parent)

	#print indexs
	for i in range(len(indexs)-1):
		if indexs[i][0:-3] != indexs[i+1][0:-3]:
			print(indexs)
		assert indexs[i][0:-3] == indexs[i+1][0:-3]

	global prev_index
	global cur_index
	if len(indexs) == 0:
		logic.append("DRS-"+str(prev_index-1)+"(")
		cur_index = prev_index - 1
	else:
		if prev_index+1 == int(indexs[0][1:-3]):
			logic.append("DRS-"+str(prev_index)+"(")
			cur_index = prev_index
		elif prev_index == int(indexs[0][1:-3]):
			logic.append("DRS-"+str(prev_index-1)+"(")
			cur_index = prev_index - 1
		else:
			assert False, "indexs are not continued"
		prev_index = int(indexs[0][1:-3])
		#logic.append("DRS-"+indexs[0][0:-3]+"(")

	box_stack.append(parent.attrib["label"])
	logic.append(parent.attrib["label"].upper())
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

def out_sents(parent):
	sents = []
	tokens = []
	lemmas = []
	senses = []
	prev_id = 1
	for child in parent:
		current_id = int(child.attrib["{http://www.w3.org/XML/1998/namespace}id"][1])
		if len(child.attrib["{http://www.w3.org/XML/1998/namespace}id"]) == 6:
			current_id = int(child.attrib["{http://www.w3.org/XML/1998/namespace}id"][1:3])
		if current_id != prev_id:
			assert current_id == prev_id + 1
			sents.append((tokens,lemmas,senses))
			tokens = []
			lemmas = []
			senses = []
			prev_id = current_id
		tmp = {}
		for cc in child[0]:
			if cc.attrib["type"] == "tok":
				tmp["tok"] = cc.text
			elif cc.attrib["type"] == "lemma":
				tmp["lemma"] = cc.text
			elif cc.attrib["type"] == "senseid":
				tmp["senseid"] = cc.text
		if "tok" in tmp:
			tokens.append(tmp["tok"])
		else:
			assert False, "no tok"
		if "lemma" in tmp:
			lemmas.append(tmp["lemma"])
		else:
			assert False, "no lemma"
		if "senseid" in tmp:
			senses.append(tmp["senseid"])
		else:
			senses.append("O")

	if len(tokens) != 0 and len(lemmas) != 0:
		sents.append((tokens, lemmas, senses))
	return sents

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


def normalized_index(tree):
	v = ["X", "S", "E", "P", "K", "T", "B"]
	c = [ [] for i in range(7)]

	#preprocess pointer for scope, because it will be predicted in the first stage.
	for i in range(len(tree)):
		if re.match("^P[0-9]+\($", tree[i]) or (tree[i] in ["NOT(", "NEC(", "POS(", "DUP(", "IMP(", "OR("]):
			if (tree[i+1] not in c[6]) and (tree[i+1] != "B0"):
				c[6].append(tree[i+1])
	for i in range(len(tree)):
		if tree[i] == "B0":
			continue
		if re.match("^[XESTPKB][0-9]+$",tree[i]):
			idx = v.index(tree[i][0])
			if tree[i] not in c[idx]:
				c[idx].append(tree[i])
		if re.match("^[PK][0-9]+\($", tree[i]):
			idx = v.index(tree[i][0])
			if tree[i][:-1] not in c[idx]:
				c[idx].append(tree[i][:-1])

	for i in range(len(tree)):
		if tree[i] == "B0":
			continue
		if re.match("^[XESTPKB][0-9]+$",tree[i]):
			idx = v.index(tree[i][0])
			assert tree[i] in c[idx]
			iidx = c[idx].index(tree[i])
			tree[i] = v[idx] + str(iidx+1)
		if re.match("^[XESTPKB][0-9]+\($",tree[i]):
			idx = v.index(tree[i][0])
			assert tree[i][:-1] in c[idx]
			iidx = c[idx].index(tree[i][:-1])
			tree[i] = v[idx] + str(iidx+1) + "("

def xmlReader(filename):
	tree = ET.parse(filename)
	root = tree.getroot()[0]
	data = {"tokens":[], "lemmas":[], "senses":[]}
	assert len(root) == 2
	assert root[0].tag == "taggedtokens"
	assert (root[1].tag == "sdrs" or root[1].tag == "drs")

	global words
	words = []
	global lemmas
	global multiple_lemmas
	sents = out_sents(root[0])
	for sent in sents:
		data["tokens"].append(" ".join(sent[0]).encode("UTF-8"))
		words.append(sent[0])
		data["lemmas"].append(" ".join(sent[1]).encode("UTF-8"))
		lemmas.append(sent[1])
		data["senses"].append(sent[2])
		# construct multiple lemmas for each sent
		d = {}
		for item in sent[1]:
			if item not in d:
				d[item] = 0
		multiple_lemmas.append(d)
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
	#normalized_index(tree)
	#print "VARIABLE"
	variables = []
	for key in domain.keys():
		variables.append(key.upper()+":"+domain[key].upper())
	data["variables"] = " ".join(variables).encode("UTF-8")
	data["tree"] = " ".join(tree)


	# we found that some senses are not consistent with the DRS, 
	# we modify the senses according to the DRS.

	i = 0
	tree = data["tree"].split()
	idx = 0
	while idx < len(tree):
		if re.match("^DRS-[0-9]+\($",tree[idx]):
			i = int(tree[idx][4:-1])
		if tree[idx] == "Pred(":
			if re.match("^\$[0-9]+$", tree[idx+3]):
				j = int(tree[idx+3][1:])
				sense = tree[idx+4][2:-2]+"."+tree[idx+5][1:-1]
				assert is_sense(tree[idx+5][1:-1])
				data["senses"][i][j] = sense
		idx += 1

	i = 0
	tree = data["tree"].split()
	idx = 0
	while idx < len(tree):
		if re.match("^DRS-[0-9]+\($",tree[idx]):
			i = int(tree[idx][4:-1])

		if tree[idx].startswith("Named") or (tree[idx] in ["Card(", "Txxx(", "Txxd(", "Tyxx(", "Txmx(", "Tymx(", "Tyxd(", "Txmd(", "Tymd("]):
			if re.match("^\$[0-9]+$", tree[idx+3]):
				j = int(tree[idx+3][1:])
				data["senses"][i][j] = "O"
		idx += 1

	data["senses"] = [" ".join(item)for item in data["senses"]]
	return data


if __name__ == "__main__":
	data = xmlReader(sys.argv[1])
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
