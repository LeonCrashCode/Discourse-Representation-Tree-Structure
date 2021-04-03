

source config-bsg

program="../../tools/train.py"

python $program -data ${data} \
	-src_word_vec_size $src_word_vec_size \
	-tgt_word_vec_size $tgt_word_vec_size \
	-layers $layers \
	-rnn_size $rnn_size \
	-transformer_ff $transformer_ff \
	-heads $heads \
	-save_model $save_model \
	-log_file $log_file \
	-dropout $dropout \
	$bridge \
	-encoder_type $encoder_type \
	-decoder_type $decoder_type \
	$position_encoding \
	$copy_attn -copy_attn_type $copy_attn_type \
	-max_relative_positions $max_relative_positions \
	-global_attention $global_attention \
	-report_every $report_every \
	-train_steps $train_steps \
	-batch_size $batch_size \
	-batch_type $batch_type \
	-optim $optim \
	-learning_rate $learning_rate \
	-learning_rate_decay ${learning_rate_decay} \
	-start_decay_steps ${start_decay_steps} \
	-decay_steps ${decay_steps} \
	-max_grad_norm $max_grad_norm \
	-adam_beta2 $adam_beta2 \
	-pre_word_vecs_enc $pre_word_vecs_enc \
	-save_checkpoint_steps $save_checkpoint_steps \
	-valid_steps $valid_steps \
	-param_init $param_init \
	$param_init_glorot \
	-normalization $normalization \
	-accum_count $accum_count \
	-seed $random_seed \
	-world_size 1 -gpu_ranks 0   
