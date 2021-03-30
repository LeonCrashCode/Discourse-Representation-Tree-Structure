from collections import Counter
import json
import sys
from torchtext.vocab import Vocab
import torch
data = json.load(open(sys.argv[1],'r'))

cnt = Counter()
for inst in data["data"]:
	nodes = inst["nodes"].strip()
	if nodes == "":
		continue
	nodes = nodes.split(" ||| ")
	for node in nodes:
		for n in node.split():
			cnt[n] += 1

	nodes = inst["edges"].strip()
	if nodes == "":
		continue
	nodes = nodes.split(" ||| ")
	for node in nodes:
		for n in node.split():
			cnt[n] += 1



vocab = Vocab(cnt,specials=['<unk>','<pad>'])
torch.save(vocab, "trn.vocab")
