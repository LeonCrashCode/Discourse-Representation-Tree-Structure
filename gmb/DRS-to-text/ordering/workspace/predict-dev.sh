
mkdir outputs

for i in `seq 1 40`
do
python -u main.py \
        --vocab trn.vocab \
	--load_from models/${i}0000.pt \
	--mode test \
        --writetrans outputs/dev.${i}0000 \
	--test dev.graph \
        --loss 0 \
        --batch_size 32 \
        --beam_size 64 \
        --seed 123456789 \
        --labeldim 100 \
        --d_emb 100 \
        --d_rnn 300 \
        --d_mlp 300 \
	--gpu > outputs/eval.${i}0000

done
