import sys
import json

data = []
for filename in open(sys.argv[1]):
	filename = filename.strip()
	d = json.load(open(filename,"r"))
	data.append(d)

json.dump({"data":data}, sys.stdout, indent=4)
	


