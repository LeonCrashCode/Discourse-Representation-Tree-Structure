program="../.../tools/preprocess.py"

train_src=data/bsg.src
train_tgt=data/bsg.tgt

valid_src=data/dev.src
valid_tgt=data/dev.tgt

save_data=data-bsg-bin/data
mkdir data-bsg-bin
python $program --train_src $train_src \
	--train_tgt $train_tgt \
	--valid_src $valid_src \
	--valid_tgt $valid_tgt \
	--src_vocab data/vocab.src \
	--src_words_min_frequency 1 \
	--tgt_words_min_frequency 3 \
	--src_seq_length 1000 \
	--tgt_seq_length 1000 \
	--dynamic_dict \
	-save_data $save_data


train_src=data/sg.src
train_tgt=data/sg.tgt

valid_src=data/dev.src
valid_tgt=data/dev.tgt

save_data=data-sg-bin/data
mkdir data-sg-bin
python $program --train_src $train_src \
        --train_tgt $train_tgt \
        --valid_src $valid_src \
        --valid_tgt $valid_tgt \
	--src_vocab data/vocab.src \
        --src_words_min_frequency 1 \
        --tgt_words_min_frequency 3 \
        --src_seq_length 1000 \
        --tgt_seq_length 1000 \
        --dynamic_dict \
        -save_data $save_data


train_src=data/g.src
train_tgt=data/g.tgt

valid_src=data/dev.src
valid_tgt=data/dev.tgt

save_data=data-g-bin/data
mkdir data-g-bin
python $program --train_src $train_src \
        --train_tgt $train_tgt \
        --valid_src $valid_src \
        --valid_tgt $valid_tgt \
	--src_vocab data/vocab.src \
        --src_words_min_frequency 1 \
        --tgt_words_min_frequency 3 \
        --src_seq_length 1000 \
        --tgt_seq_length 1000 \
        --dynamic_dict \
        -save_data $save_data


