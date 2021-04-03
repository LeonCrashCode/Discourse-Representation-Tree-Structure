echo "please check if the implementation is in the correct directory"
python ../../tools/translate.py \
        --model $3 \
        --beam_size 10 \
        --max_length 400 \
        --replace_unk \
        --batch_size 30 \
        --gpu 0 \
        --data_type text \
        --src $1 \
        --output $2
