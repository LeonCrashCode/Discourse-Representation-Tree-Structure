echo "please check if the implementation is in the correct directory"

onmt_translate \
        --model ${1} \
        --beam_size ${5} \
        --max_length 400 \
        --replace_unk \
	--batch_size 1 \
        --batch_type sents \
        --gpu 0 \
        --data_type text \
        --src ${2} \
        --tree ${3} \
        --output ${4} &

