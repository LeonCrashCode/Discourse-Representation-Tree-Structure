import sys
import json

with open(sys.argv[1], "r") as f:
        data = json.load(f)

        for inst in data["data"]:
                src = inst["relative"]
                tree = []
                for s in src.strip().split():
                        if s == ")":
                                continue
                        tree.append(s)
                print(" ".join(tree))
        f.close()
