
python -u main.py \
        --vocab trn.vocab \
	--load_from ${1} \
	--mode test \
        --writetrans ${3} \
	--test ${2} \
        --loss 0 \
        --batch_size 32 \
        --beam_size 64 \
        --seed 123456789 \
        --labeldim 100 \
        --d_emb 100 \
        --d_rnn 300 \
        --d_mlp 300 \
	--gpu
