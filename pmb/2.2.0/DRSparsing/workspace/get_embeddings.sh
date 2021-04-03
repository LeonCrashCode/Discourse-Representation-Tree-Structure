lang=$1
program=../OpenDRS-py/tools/embeddings_to_torch.py

for part in bsg sg g
do
python $program \
        -emb_file_enc ../embeddings/${lang}.vec \
	-emb_file_dec ../embeddings/en.vec \
	-dict_file data-${part}-bin/data.vocab.pt \
        -output_file data-${part}-bin/embeddings-300
done
