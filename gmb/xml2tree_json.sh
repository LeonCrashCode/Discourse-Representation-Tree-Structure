

for f in `ls -d data/*/*`
do
		echo "processing $f"
		python xml2tree_json.py $f/en.drs.xml >$f/en.tree.json 
done	
