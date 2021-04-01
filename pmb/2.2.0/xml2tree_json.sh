
lang=${1}
part=${2}
for d in `ls -d data/${part}/*`
do 
	echo ${d}
	for f in `ls -d ${d}/*`
	do
		if [ -f ${f}/${lang}.drs.xml ]; then
			echo "processing $f"
			python xml2tree_json.py $f/${lang}.tok.off $f/${lang}.drs.xml >$f/${lang}.tree.json 
		fi
	done
done

