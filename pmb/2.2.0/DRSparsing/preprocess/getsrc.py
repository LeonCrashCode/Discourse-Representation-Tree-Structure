import sys
import json

with open(sys.argv[1], "r") as f:
        data = json.load(f)

        for inst in data["data"]:
                tgt = " ".join(inst["tokens"])

                print(tgt.lower())
        f.close()
