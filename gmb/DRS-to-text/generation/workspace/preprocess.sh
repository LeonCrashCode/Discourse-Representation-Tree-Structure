mkdir data-bin
echo "please check if the implementation is in the correct directory"
onmt_preprocess \
        --train_src data/trn.src \
        --train_tgt data/trn.tgt \
        --train_tree data/trn.tree \
        --valid_src data/dev.src \
        --valid_tgt data/dev.tgt \
        --valid_tree data/dev.tree \
        --with_tree \
        --tgt_words_min_frequency 2 \
        --src_seq_length 100000 \
        --tgt_seq_length 100000 \
        --save_data data-bin/basic
