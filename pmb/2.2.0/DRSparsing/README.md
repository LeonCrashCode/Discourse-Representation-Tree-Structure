

# Instruction

## Requirements
```
Pytorch >= 1.0
torchtext == 0.4.0
```
## Data prepare

Go to the preprocess directory to prepare the data for DRS parsing, and construct director for each language.
```
cd preprocess
mkdir en de it nl
```

Build data for English (en)
```
cd en
ln -s ../oracle.py
ln -s ../util.py
ln -s ../../../en.[bronze|silver|gold|dev|tst].json
python2 oracle.py en.[bronze|silver|gold|dev|tst].json > en.[bronze|silver|gold|dev|tst].oracle.json

ln -s ../getsrc.py
ln -s ../gettgt.py
python2 getsrc.py en.[bronze|silver|gold|dev|tst].oracle.json > en.[bronze|silver|gold|dev|tst].src
python2 gettgt.py en.[bronze|silver|gold|dev|tst].oracle.json > en.[bronze|silver|gold|dev|tst].tgt
cd ..
```
Build data for low-resource language, including German (de), Italian (it) and Dutch (nl).
```
cd [lang] # de, it or nl
ln -s ../oracle.py
ln -s ../util.py
ln -s ../../../[lang].[bronze|silver|dev|tst].json
python2 oracle.py [lang].[bronze|silver|dev|tst].json > [lang].[bronze|silver|dev|tst].oracle.json

ln -s ../getsrc.py
ln -s ../gettgt.py
python2 getsrc.py [lang].[bronze|silver|dev|tst].oracle.json > [lang].[bronze|silver|dev|tst].src
python2 gettgt.py [lang].[bronze|silver|dev|tst].oracle.json > [lang].[bronze|silver|dev|tst].tgt
cd ..
```
Build data from Enlgish for low-resource language, including German (de), Italian (it) and Dutch (nl).
```
cd [lang] # de, it or nl
ln -s ../get_rest.py
ln -s ../../../en.[gold|dev|tst].json
python2 get_rest.py --create en.gold.json en.dev.json en.tst.json --exclude [lang].dev.json [lang].tst.json > [lang].gold.json

python2 oracle.py [lang].gold.json > [lang].gold.oracle.json
python2 getsrc.py [lang].gold.oracle.json > en.gold.src
python2 gettgt.py [lang].gold.oracle.json > [lang].gold.tgt
```
`en.gold.src` has to be translated to `[lang]` language, and the traslated texts are saved in `[lang].gold.src` that will be pairing with `[lang].gold.tgt` for training. The translated training data for German (de), Italian (it) and Dutch (nl) are [available](https://drive.google.com/drive/folders/1IaNRpMEDEzhE0CZz9sGq-tPX0giexdOt?usp=sharing), using Google Translate.

## Train

Go to the `workspace` directory to prepare the data for DRS parsing, and construct director for each language.
```
cd workspace
mkdir en de it nl
```
Train model for each langauge. Codes are based on the [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py), ad you can install the local onmt. Download the [embeddings]() or you can use your own, and put it onto the director `workspace`
```
cd [lang]

mkdir data
cd data

ln -s ../../../preprocess/[lang]/[lang].bronze.[src|tgt]
ln -s ../../../preprocess/[lang]/[lang].silver.[src|tgt]
ln -s ../../../preprocess/[lang]/[lang].gold.[src|tgt]
ln -s ../../../preprocess/[lang]/[lang].dev.[src|tgt]
ln -s ../../../preprocess/[lang]/[lang].tst.[src|tgt]

cat [lang].bronze.[src|tgt] [lang].silver.[src|tgt] [lang].gold.[src|tgt] > bsg.[src|tgt]
cat [lang].silver.[src|tgt] [lang].gold.[src|tgt] > sg.[src|tgt]
cat [lang].gold.[src|tgt] > g.[src|tgt]
cat [lang].dev.[src|tgt] > dev.[src|tgt]
cat [lang].tst.[src|tgt] > tst.[src|tgt]

ln -s ../../get_vocab.py
python get_vocab.py bsg.src > vocab.src
cd ..
```
Only construct `*.bs.*` and `*.s.*` training data if having no `gold` training data

Prepare data for onmt. 
```
ln -s ../preprocess.sh
bash preprocess.sh
ln -s ../get_embeddings.sh
bash get_embeddings.sh [lang]
```
It will automatically generate three directors `data-bsg-bin`, `data-sg-bin`, and `data-g-bin`
```
ln -s ../config-bsg
ln -s ../config-sg
ln -s ../config-g

ln -s ../train-bsg.sh
ln -s ../train-sg.sh
ln -s ../train-g.sh

bash train-bsg.sh
```
The `checkpoints` are saved in `checkpoints-bsg/`, choose the best checkpoint and then
```
bash train-sg.sh [best_ckpt]
```
The `checkpoints` are saved in `checkpoints-sg/`, choose the best checkpoint and then
```
bash train-g.sh [best_ckpt]
```
The `checkpoints` are saved in `checkpoints-g/`.

## Predict
```
ln -s ../predict.sh
bash predict.sh [input] [output] [ckpt]
```
