#for lang in de it 
for lang in en
do
#for part in bronze silver dev tst
for part in bronze silver dev tst gold
do
nohup srun --gres gpu:0 --mem 20G -p ILCC_CPU python oracle.py ${lang}.${part}.json > ${lang}.${part}.oracle.json 2> ${lang}.${part}.out &
done
done
