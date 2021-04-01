
import re
import types

drs_n = re.compile("^DRS\-[0-9]+\($")
vb = re.compile("^B[0-9]+$")
vp = re.compile("^P[0-9]+$")
vk = re.compile("^K[0-9]+$")
bp = re.compile("^P[0-9]+\($")
bk = re.compile("^K[0-9]+\($")
vall = re.compile("^[XESTPK][0-9]+$")

def is_align_var(v):
	if type(v) == types.StringType and \
		re.match("^\$[0-9]+$", v):
		return True
	return False
def is_align_cond(v):
	if type(v) == types.StringType and \
		re.match("^\$[0-9]+\[.+\]\($", v):
		return True
	return False

def is_sense(v):
	if type(v) == types.StringType and \
		re.match("^\"?[anvr]\.[0-9]+\"?$", v): 
		return True
	return False

def is_v(v, ss, op="general"):
	if type(v) != types.StringType:
		return False
	assert op in ["general", "leastone", "noone"]
	for s in list(ss):
		if op == "leastone" and re.match("^"+s+"-?[0-9]+$", v):
			return True
		elif op == "noone" and re.match("^"+s+"$", v):
			return True
		elif op == "general" and re.match("^"+s+"-?[0-9]*$", v):
			return True
	return False

# FOR DRS NODE
############################
def is_drs_node(tup):
	if type(tup) == types.ListType and \
		(re.match("^DRS-[0-9]+\($", tup[0]) or \
			re.match("^DRS\($", tup[0]) ):
		return True
	return False
def drs_wellform(tup):
	if len(tup) < 1:
		return False
	for tu in tup[1:]:
		if is_cond_name_node(tu) or \
			is_cond_pred_node(tu) or \
			is_cond_ref_node(tu) or \
			is_cond_other_node(tu) or \
			is_not_scope_node(tu) or \
			is_nec_scope_node(tu) or \
			is_pos_scope_node(tu) or \
			is_imp_scope_node(tu) or \
			is_dis_scope_node(tu) or \
			is_dup_scope_node(tu) or \
			is_prop_scope_node(tu):
			pass
		else:
			return False
	return True
############################################

#FOR SEGMENTED DRS NODE
############################################
def is_sdrs_node(tup):
	if type(tup) == types.ListType and \
		tup[0] == "SDRS(":
		return True
	return False
def sdrs_wellform(tup):
	if len(tup) < 4:
		return False
	for tu in tup[1:]:
		if is_drs_node(tu) or \
			is_sdrs_node(tu) or \
			is_drel_node(tu):
			pass
		else:
			return False
	return True
#############################################

#FOR CONSTITUENT NODE
#############################################
def is_constituent_node(tup):
	if type(tup) == types.ListType and \
		re.match("^K[0-9]+\($", tup[0]):
		return True
	return False
def constituent_wellform(tup):
	if len(tup) < 4:
		return False
	for tu in tup[1:]:
		if is_drs_node(tu) or \
			is_sdrs_node(tu) or \
			is_drel_node(tu):
			pass
		else:
			return False
	return True
#############################################

#FOR DISCOURSE RELATION NODE
#############################################
def is_drel_node(tup):
	if type(tup) == types.ListType and \
		len(tup) == 3 and \
		tup[0].isupper() and \
		all([type(tu) == types.StringType for tu in tup[1:]]):
		return True
	return False
def drel_wellform(tup):
	if len(tup) != 3:
		return False
	for tu in tup[1:]:
		if is_v(tu, "BK", "leastone"):
			pass
		else:
			return False
	return True
#############################################

#FOR UNUARY SCOPE NODE
#############################################
def is_unary_scope_node(tup):
	if is_not_scope_node(tup) or \
		is_nec_scope_node(tup) or \
		is_pos_scope_node(tup):
		return True
	return False
def unary_scope_wellform(tup):
	if len(tup) != 3:
		return False
	if is_v(tup[1], "B", "leastone") or is_v(tup[1], "O"):
		pass
	else:
		return False
	if is_drs_node(tup[2]) or is_sdrs_node(tup[2]):
		pass
	else:
		return False
	return True
#############################################

#FOR BINARY SCOPE NODE
#############################################
def is_binary_scope_node(tup):
	if is_imp_scope_node(tup) or \
		is_dis_scope_node(tup) or \
		is_dup_scope_node(tup):
		return True
	return False
def binary_scope_wellform(tup):
	if len(tup) != 4:
		return False
	if is_v(tup[1], "B", "leastone") or is_v(tup[1], "O"):
		pass
	else:
		return False
	if is_drs_node(tup[2]) or is_sdrs_node(tup[2]):
		pass
	else:
		return False
	if is_drs_node(tup[3]) or is_sdrs_node(tup[3]):
		pass
	else:
		return False
	return True
#############################################

#FOR NOT SCOPE NODE
#############################################
def is_not_scope_node(tup):
	if type(tup) == types.ListType and \
		tup[0] == "NOT(":
		return True
	return False
def not_scope_wellform(tup):
	return unary_scope_wellform(tup)
##############################################

#FOR POS SCOPE NODE
#############################################
def is_pos_scope_node(tup):
	if type(tup) == types.ListType and \
		tup[0] == "POS(":
		return True
	return False
def pos_scope_wellform(tup):
	return unary_scope_wellform(tup)
##############################################

#FOR NEC SCOPE NODE
#############################################
def is_nec_scope_node(tup):
	if type(tup) == types.ListType and \
		tup[0] == "NEC(":
		return True
	return False
def nec_scope_wellform(tup):
	return unary_scope_wellform(tup)
##############################################

#FOR IMPLICATION SCOPE NODE
#############################################
def is_imp_scope_node(tup):
	if type(tup) == types.ListType and \
		tup[0] == "IMP(":
		return True
	return False
def imp_scope_wellform(tup):
	return binary_scope_wellform(tup)
##############################################

#FOR DISJUNCTION SCOPE NODE
#############################################
def is_dis_scope_node(tup):
	if type(tup) == types.ListType and \
		tup[0] in ["DIS(", "OR("]:
		return True
	return False
def dis_scope_wellform(tup):
	return binary_scope_wellform(tup)
##############################################

#FOR DUPLEX SCOPE NODE
#############################################
def is_dup_scope_node(tup):
	if type(tup) == types.ListType and \
		tup[0] == "DUP(":
		return True
	return False
def dup_scope_wellform(tup):
	return binary_scope_wellform(tup)
##############################################

#FOR PROPOSITION SCOPE NODE
##############################################
def is_prop_scope_node(tup):
	if type(tup) == types.ListType and \
		re.match("^P[0-9]+\($", tup[0]):
		return True
	return False
def prop_scope_wellform(tup):
	return unary_scope_wellform(tup)
##############################################

#FOR CONDITION NAME NODE
##############################################
def is_cond_name_node(tup):
	if type(tup) == types.ListType and \
		tup[0].startswith("Named"):
		return True
	return False
def cond_name_wellform(tup):
	if len(tup) < 4:
		return False
	if is_v(tup[1], "B", "leastone") or is_v(tup[1], "O"):
		pass
	else:
		return False
	if is_v(tup[2], "XESTP", "leastone"):
		pass
	else:
		return False
	return True
##############################################

#FOR CONDITION PERD NODE
##############################################
def is_cond_pred_node(tup):
	if type(tup) == types.ListType and \
		tup[0] == "Pred(":
		return True
	return False
def cond_pred_wellform(tup):
	if len(tup) < 5:
		return False
	if is_v(tup[1], "B", "leastone") or is_v(tup[1], "O"):
		pass
	else:
		return False
	if is_v(tup[2], "XESTP", "leastone"):
		pass
	else:
		return False
	if is_sense(tup[-1]):
		pass
	else:
		return False
	return True
##############################################

#FOR CONDITION REFERENCE NODE
##############################################
def is_cond_ref_node(tup):
	if type(tup) == types.ListType and \
		tup[0] == "Ref(":
		return True
	return False
def cond_ref_wellform(tup):
	if len(tup) != 3:
		return False
	if is_v(tup[1], "B", "leastone") or is_v(tup[1], "O"):
		pass
	else:
		return False
	if is_v(tup[2], "XESTP", "noone"):
		pass
	else:
		return False
	return True
##############################################

#FOR CONDITION OTHER NODE
##############################################
def is_cond_other_node(tup):
	if type(tup) == types.ListType and \
		all([type(tu) == types.StringType for tu in tup[1:]]):
		return True
	return False
def cond_other_wellform(tup):
	if len(tup) < 4:
		return False
	if is_v(tup[1], "B", "leastone") or is_v(tup[1], "O"):
		pass
	else:
		return False
	
	if len(tup) == 4:
		if is_v(tup[2], "XESTP", "leastone"):
			pass
		elif is_v(tup[3], "XESTP", "leastone"):
			pass
		else:
			return False
	else:
		if is_v(tup[2], "XESTP", "leastone"):
			pass
		else:
			return False
	return True
def is_cond_node(tup):
	if is_cond_name_node(tup) or \
		is_cond_pred_node(tup) or \
		is_cond_ref_node(tup) or \
		is_cond_other_node(tup):
		return True
	return False
##############################################

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

def add_b_sb(bracket):

	index = [1]
	def travel(b):
		if is_drs_node(b) or is_sdrs_node(b):
			b.insert(1, "B"+str(index[0]))
			index[0] += 1
		for c in b[1:]:
			if type(c) == types.ListType:
				travel(c)
	travel(bracket)
	return index[0]

def get_drs_pointer(bracket):
	mp_l = []
	def travel(b):
		if is_drs_node(b):
			mp_l.append(b[1])
		for c in b:
			if type(c) == types.ListType:
				travel(c)
	travel(bracket)
	return mp_l
