import torch
from copy import deepcopy
import re
class sequence_state(object):
	def __init__(self):
		self.stack = []
		self.var = {'X':0, 'E':0, 'S':0, 'T':0, 'P':0, 'O':0}
		self.first = True
	def print(self):
		print(self.stack)
		print(self.var)
		print(self.first)

class sequence_mask(object):
	def __init__(self, itos, stoi, eos, device):

		self.itos = itos

		vs = list("XESTPKOB")
		self.XESTPKOB_set = [ [] for i in range(8)] #X1, E1, O1 ...
		for act, act_name in enumerate(itos):
			for i, v in enumerate(vs):
				if re.match("^"+v+"-?[0-9]+$", act_name):
					self.XESTPKOB_set[i].append(act)

		self.P_set = [] # P1(, ...
		for act, act_name in enumerate(itos):
			if re.match("^P[0-9]+\($", act_name):
					self.P_set.append(act)

		XESTPKOB_idx = [ -1 for i in range(8)] # X, E, S, T ....
		for i, v in enumerate(vs):
			XESTPKOB_idx[i] = stoi[v]

		#the token ending with character "("
		endBracket = []
		for act, act_name in enumerate(self.itos):
			if act_name[-1] == "(":
				endBracket.append(act)

		DRS_idx = stoi["DRS("] if "DRS(" in stoi else -1
		SDRS_idx = stoi["SDRS("] if "SDRS(" in stoi else -1
		close_idx = stoi[")"] if ")" in stoi else -1
		unk_idx = stoi["<unk>"] if "<unk>" in stoi else -1

		OR_idx = stoi["OR("] if "OR(" in stoi else -1
		DIS_idx = stoi["DIS("] if "DIS(" in stoi else -1
		DUP_idx = stoi["DUP("] if "DUP(" in stoi else -1
		IMP_idx = stoi["IMP("] if "IMP(" in stoi else -1

		NEC_idx = stoi["NEC("] if "NEC(" in stoi else -1
		POS_idx = stoi["POS("] if "POS(" in stoi else -1
		NOT_idx = stoi["NOT("] if "NOT(" in stoi else -1

		self.allow_drs = torch.zeros(len(self.itos), dtype=torch.uint8, device=device)
		self.set(self.allow_drs, [DRS_idx], 1)

		self.allow_sdrs = torch.zeros(len(self.itos), dtype=torch.uint8, device=device)
		self.set(self.allow_sdrs, [SDRS_idx], 1)

		self.allow_end = torch.zeros(len(self.itos), dtype=torch.uint8, device=device)
		self.set(self.allow_end, [eos], 1)

		self.allow_cond = torch.zeros(len(self.itos), dtype=torch.uint8, device=device)
		self.set(self.allow_cond, endBracket, 1)
		self.set(self.allow_cond, [DRS_idx, SDRS_idx], 0)

		self.allow_close = torch.zeros(len(self.itos), dtype=torch.uint8, device=device)
		self.set(self.allow_close, [close_idx], 1)

		self.allow_unk = torch.zeros(len(self.itos), dtype=torch.uint8, device=device)
		self.set(self.allow_unk, [unk_idx], 1)

		self.allow_normal_cond = torch.zeros(len(self.itos), dtype=torch.uint8, device=device)
		self.set(self.allow_normal_cond, endBracket, 1)
		self.set(self.allow_normal_cond, [DRS_idx, SDRS_idx, OR_idx, DIS_idx, DUP_idx, IMP_idx, NEC_idx, POS_idx, NOT_idx], 0)

		self.allow_ref0 = torch.zeros(len(self.itos), dtype=torch.uint8, device=device)
		self.set(self.allow_ref0, self.XESTPKOB_set[-1] + self.XESTPKOB_set[-2] + XESTPKOB_idx[-2:], 1)


		self.allow_ref1 = torch.zeros(len(self.itos), dtype=torch.uint8, device=device)
		self.set(self.allow_ref1, XESTPKOB_idx[0:-3], 1)

		self.allow_dv = torch.zeros(len(self.itos), dtype=torch.uint8, device=device)
		self.set(self.allow_dv, self.XESTPKOB_set[-3], 1)

		self.allow_anyv = torch.ones(len(self.itos), dtype=torch.uint8, device=device)
		self.set(self.allow_anyv, range(4), 0)
		self.set(self.allow_anyv, [close_idx] + endBracket + XESTPKOB_idx + self.XESTPKOB_set[-1] + self.XESTPKOB_set[-2], 0)

	def set(self, tensor, indices, value):
		for index in indices:
			if index < 0:
				continue
			tensor[index] = value

	def set_p_cond(self, mask, pmax):
		mask = deepcopy(mask)
		for p in self.P_set:
			if int(self.itos[p][1:-1]) >= pmax:
				mask[p] = 0
		return mask
	def set_xestpkob_var(self, mask, i, vmax):
		mask = deepcopy(mask)
		for v in self.XESTPKOB_set[i]:
			if int(self.itos[v][1:]) >= vmax:
				mask[v] = 0
		return mask

class BB_sequence_state(object):
	def __init__(self, itos, stoi, mb_device, batch_size, beam_size, eos=3):
		self.states = [sequence_state() for i in range(batch_size*beam_size)] # empty
		self.itos = itos
		self.stoi = stoi
		self.device = mb_device
		self.batch_size = batch_size
		self.beam_size = beam_size
		self.eos = eos

		self.maskset = sequence_mask(itos, stoi, eos, mb_device)
	def get_mask(self):
		masks = []
		expanded_masks = []
		for state in self.states:
			mask, expanded_mask = self.get_mask_one(state)
			masks.append(mask.unsqueeze(0))
			expanded_masks.append(expanded_mask.unsqueeze(0))
		return torch.cat(masks, 0), torch.cat(expanded_masks, 0)

	def get_mask_one(self, state):
		stack = state.stack
		mask = None
		expaned_v = 0
		if len(stack) == 0: # empty
			if state.first:
				mask = self.maskset.allow_drs | self.maskset.allow_sdrs
			else:
				mask = self.maskset.allow_end

		elif stack[-1][0] == self.stoi["DRS("]: 
			mask = self.maskset.allow_cond
			mask = self.maskset.set_p_cond(mask, state.var["P"])
			if stack[-1][1] > 0:
				mask |= self.maskset.allow_close

		elif stack[-1][0] == self.stoi["SDRS("]:
			if stack[-1][1] < 2:
				mask = self.maskset.allow_drs | self.maskset.allow_sdrs
			else:
				mask = self.maskset.allow_normal_cond
				if stack[-1][1] >= 1000: # start predict discourse relations
					mask |= self.maskset.allow_close
				else:
					mask |= self.maskset.allow_drs
					mask |= self.maskset.allow_sdrs

		elif self.itos[stack[-1][0]] in ["OR(", "DIS(", "DUP(", "IMP("]:
			if stack[-1][1] == 0:
				mask = self.maskset.allow_ref0
				mask = self.maskset.set_xestpkob_var(mask, -2, state.var["O"]) # O
			elif stack[-1][1] < 3:
				mask = self.maskset.allow_drs | self.maskset.allow_sdrs
			else:
				mask = self.maskset.allow_close

		elif self.itos[stack[-1][0]]in ["NEC(", "POS(", "NOT("]:
			if stack[-1][1] == 0:
				mask = self.maskset.allow_ref0
				mask = self.maskset.set_xestpkob_var(mask, -2, state.var["O"]) # O
			elif stack[-1][1] < 2:
				mask = self.maskset.allow_drs | self.maskset.allow_sdrs
			else:
				mask = self.maskset.allow_close

		elif re.match("^P[0-9]+\($", self.itos[stack[-1][0]]):
			if stack[-1][1] == 0:
				mask = self.maskset.allow_ref0
				mask = self.maskset.set_xestpkob_var(mask, -2, state.var["O"]) # O
			elif stack[-1][1] < 2:
				mask = self.maskset.allow_drs | self.maskset.allow_sdrs
			else:
				mask = self.maskset.allow_close

		elif stack[-1][0] == self.stoi["Ref("]:
			if stack[-1][1] == 0:
				mask = self.maskset.allow_ref0
				mask = self.maskset.set_xestpkob_var(mask, -2, state.var["O"]) # O
			elif stack[-1][1] == 1:
				mask = self.maskset.allow_ref1
			else:
				mask = self.maskset.allow_close

		#elif stack[-1][0] == self.stoi["Pred("]:
		else:
			if stack[-2][0] == self.stoi["SDRS("]:
				if stack[-1][1] < 2:
					mask = self.maskset.allow_dv
					mask = self.maskset.set_xestpkob_var(mask, -3, stack[-2][1]) #K
				else:
					mask = self.maskset.allow_close
			else:
				if stack[-1][1] == 0:
					mask = self.maskset.allow_ref0
					mask = self.maskset.set_xestpkob_var(mask, -2, state.var["O"]) # O
				elif stack[-1][1] == 1000:
					mask = self.maskset.allow_close
				else:
					mask = self.maskset.allow_anyv
					for i, v in enumerate(list("XESTP")):
						mask = self.maskset.set_xestpkob_var(mask, i, state.var[v])
					mask |= self.maskset.allow_unk
					mask |= self.maskset.allow_close
					expaned_v = 1
		#print("expanded_mask", v)
		expanded_mask = torch.full([1], expaned_v, dtype=torch.uint8, device=self.device)
		return mask, expanded_mask
	def index_select(self, select_indices):
		states = []
		for index in select_indices:
			states.append(deepcopy(self.states[index]))
		self.states = states

	def update(self, actions):
		for i, act in enumerate(actions):
			self.states[i] = self.update_one(self.states[i], act)

	def update_beam(self, actions, selects, scores):
		"""
		selects = selects.data.tolist()
		actions = actions.data.tolist()
		scores = scores.exp().data.tolist()
		assert len(selects) == len(actions) == len(self.states) == len(scores)

		states = []
		for i, j, k in zip(selects, actions, scores):
			print(i,j,k)
			self.states[i].print()
			if k <= 0.0:
				states.append(deepcopy(self.states[i]))
			else:
				states.append(self.update_one(self.states[i], j))
		self.states = states
		"""
		assert len(actions) == len(selects) == len(scores)

		states = []
		for i, (sel, act) in enumerate(zip(selects, actions)):
			if scores[i] <= 0:
				states.append(deepcopy(self.states[sel]))
			else:
				states.append(self.update_one(self.states[sel], act))
			#self.states[i].print()
		self.states = states

	def update_one(self, state, act):
		nstate = deepcopy(state)
		nstate.first = False
		stack = nstate.stack
		if act == self.eos:
			pass
		elif act == self.stoi[")"]:
			if (self.itos[stack[-1][0]] not in ["DRS(", "SDRS("]) and stack[-2][0] == self.stoi["SDRS("]:
				stack.pop()
				stack[-1][1] = 1000
			else:
				stack.pop()
				if len(stack) != 0:
					stack[-1][1] += 1
		elif act < len(self.itos) and self.itos[act][-1] == "(" :
			stack.append([act, 0])
		else:
			if act < len(self.itos) and re.match("^[anvr]\.[0-9][0-9]$", self.itos[act]):
				stack[-1][1] = 1000
			else:
				stack[-1][1] += 1

			if stack[-1][0] == self.stoi["Ref("]:
				act_name = self.itos[act]
				if act_name in list("XESTPO"):
					nstate.var[act_name] += 1
		return nstate
	"""
	def allow_drs(self, mask):
		mask[self.stoi["DRS("]] = self.unmask
	def allow_sdrs(self, mask):
		mask[self.stoi["SDRS("]] = self.unmask
	def allow_end(self, mask):
		mask[self.eos] = self.unmask
	def allow_cond(self, mask, d):
		for act, act_name in enumerate(self.itos):
			if act_name[-1] == "(" and act_name not in ["DRS(", "SDRS("]:
				mask[act] = self.unmask
			if re.match("^P[0-9]\($", act_name) and int(act_name[1:-1]) < d["P"]:
				mask[act] = self.unmask
	def allow_close(self, mask):
		mask[self.stoi[")"]] = self.unmask
	def allow_unk(self, mask):
		mask[self.stoi["<unk>"]] = self.unmask
	def allow_normal_cond(self, mask):
		for act, act_name in enumerate(self.itos):
			if act_name[-1] == "(" and act_name not in ["DRS(", "SDRS(", "OR(", "DIS(", "DUP(", "IMP(","NEC(", "POS(", "NOT("]:
				mask[act] = self.unmask
	def allow_ref0(self, mask, d):
		for act, act_name in enumerate(self.itos):
			if re.match("^B[0-9]*$", act_name):
				mask[act] = self.unmask
			if act_name == "O":
				mask[act] = self.unmask
			if re.match("^O[0-9]+$", act_name) and int(act_name[1:]) < d["O"]:
				mask[act] = self.unmask
	def allow_ref1(self, mask):
		for act_name in list("XESTP"):
			mask[self.stoi[act_name]] = self.unmask
	def allow_dv(self, mask, ndrs):
		for act, act_name in enumerate(self.itos):
			if re.match("^K[0-9]+$", act_name) and int(act_name[1:]) < ndrs:
				mask[act] = self.unmask
	def allow_anyv(self, mask, d):
		for act, act_name in enumerate(self.itos):
			if act < 4:
				continue
			if act_name == ")":
				continue
			if act_name[-1] == "(":
				continue
			if act_name in list("XESTPBO"):
				continue
			if re.match("^[BO][0-9]*$",act_name):
				continue
			if re.match("^[XESTP][0-9]+$",act_name):
				if int(act_name[1:]) < d[act_name[0]]:
					mask[act] = self.unmask
				continue
			mask[act] = self.unmask
	"""








