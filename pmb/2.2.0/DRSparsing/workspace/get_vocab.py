import sys

d = set()
for filename in sys.argv[1:]:
    for line in open(filename):
        line = line.strip()
        if not line:
            break
        line = line.split()
        d.update(line)

d = list(d)
d.sort()
for item in d:
    print(item)
