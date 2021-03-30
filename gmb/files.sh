
	for f in `ls -d data/p1*/*`
	do
		if [ -s ${f}/en.tree.json ]; then
			echo ${f}/en.tree.json
		fi
	done

