mkdir models
python -u main.py \
        --model_path models \
        --vocab trn.vocab \
        --corpus trn.graph \
        --valid dev.graph \
        --test tst.graph \
        --report 1000 \
        --valid_steps 10000 \
        --maximum_steps 400000 \
        --batch_size 32 \
        --beam_size 64 \
        --input_drop_ratio 0.5 \
        --drop_ratio 0.5 \
        --gnndp 0.5 \
        --lr 0.001 \
        --optim adam \
        --seed 123456789 \
        --labeldim 100 \
        --d_emb 100 \
        --d_rnn 300 \
        --d_mlp 300 \
        --learning_rate_decay 0.3 \
        --start_decay_steps 200000 \
        --decay_steps 20000 \
	--gpu

