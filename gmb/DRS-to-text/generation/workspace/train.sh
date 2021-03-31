echo "please check if the implementation is in the correct directory"
mkdir models
onmt_train \
        --src_word_vec_size 300 \
        --tgt_word_vec_size 300 \
        --model_type text \
        --encoder_type tree \
        --tree_type neighbor \
        --bidirectional \
        --decoder_type rnn \
        --enc_layers 2 \
        --dec_layers 2 \
        --enc_rnn_size 512 \
        --dec_rnn_size 512 \
        --no_init \
        --input_feed 1 \
        --rnn_type LSTM \
        --global_attention general \
        --global_attention_function softmax \
        --generator_function softmax \
        --apex_opt_level O1 \
        --data data-bin/basic \
        --save_model models/basic \
        --save_checkpoint_steps 1000 \
        --gpu_ranks 0 \
        --world_size 1 \
        --seed 123456789 \
        --param_init 0.1 \
        --batch_size 30000 \
        --batch_type tokens \
        --normalization sents \
        --accum_count 1 \
        --valid_steps 1000 \
        --valid_batch_size 10 \
        --max_generator_batches 0 \
        --train_steps 30000 \
        --optim adam \
        --max_grad_norm 5 \
        --dropout 0.5 \
        --attention_dropout 0.1 \
        --dropout_steps 0 \
        --truncated_decoder 0 \
        --adam_beta1 0.9 \
        --adam_beta2 0.999 \
        --label_smoothing 0.0 \
        --learning_rate 0.001 \
        --learning_rate_decay 0.5 \
        --start_decay_steps 8000 \
        --decay_steps 1000 \
        --decay_method none \
        --report_every 10 



