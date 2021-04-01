
lang=$1
part=$2

if [ ${lang} = en ]; then

	if [ ${part} = gold ]; then
		for d in `ls -d data/${part}/p[0-9][2-9]` 
		do 
			for f in `ls -d ${d}/*`
			do
				if [ -s ${f}/${lang}.tree.json ]; then
					echo ${f}/${lang}.tree.json
				fi
			done
		done
	elif [ ${part} = bronze ] || [ ${part} = silver ]; then
		for d in `ls -d data/${part}/p*`
		do
			for f in `ls -d ${d}/*`
			do
				if [ -s ${f}/${lang}.tree.json ]; then
					echo ${f}/${lang}.tree.json
				fi
			done
		done
	elif [ ${part} = dev ]; then
		for d in `ls -d data/gold/p[0-9]0`
		do
			for f in `ls -d ${d}/*`
			do
				if [ -s ${f}/${lang}.tree.json ]; then
					echo ${f}/${lang}.tree.json
				fi
			done
		done
	elif [ ${part} = test ]; then
		for d in `ls -d data/gold/p[0-9]1`
		do
			for f in `ls -d ${d}/*`
			do
				if [ -s ${f}/${lang}.tree.json ]; then
					echo ${f}/${lang}.tree.json
				fi
			done
		done
	fi

fi

if [ ${lang} = de ] || [ ${lang} = it ] || [ ${lang} = nl ]; then

	if [ ${part} = bronze ] || [ ${part} = silver ]; then
		for d in `ls -d data/${part}/p*`
		do
			for f in `ls -d ${d}/*`
			do
				if [ -s ${f}/${lang}.tree.json ]; then
					echo ${f}/${lang}.tree.json
				fi
			done
		done
	elif [ ${part} = dev ]; then
		for d in `ls -d data/gold/p[0-9]1 data/gold/p[0-9]3 data/gold/p[0-9]5 data/gold/p[0-9]7 data/gold/p[0-9]9`
		do
			for f in `ls -d ${d}/*`
			do
				if [ -s ${f}/${lang}.tree.json ]; then
					echo ${f}/${lang}.tree.json
				fi
			done
		done
	elif [ ${part} = test ]; then
		for d in `ls -d data/gold/p[0-9]0 data/gold/p[0-9]2 data/gold/p[0-9]4 data/gold/p[0-9]6 data/gold/p[0-9]8`
		do
			for f in `ls -d ${d}/*`
			do
				if [ -s ${f}/${lang}.tree.json ]; then
					echo ${f}/${lang}.tree.json
				fi
			done
		done
	fi



fi
