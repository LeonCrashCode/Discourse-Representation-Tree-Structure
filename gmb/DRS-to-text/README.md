

# Instruction

## Requirements
```
Pytorch >= 1.0
torchtext == 0.4.0
```
## Data prepare

Go to the preprocess directory to prepare the data for DRS-to-text generation. 

```
cd preprocess

ln -s ../../trn.json
ln -s ../../dev.json
ln -s ../../tst.json

python3 create_graph_bi.py trn.json > trn.graph_bi
python3 create_graph_bi.py dev.json > dev.graph_bi
python3 create_graph_bi.py tst.json > tst.graph_bi

python3 create_vocab.py trn.graph
```
The vocabulary file `trn.vocab` is automatically generated.

## Condition ordering

```
cd ordering
cd workspace

ln -s ../../preprocess/trn.graph_bi trn.graph
ln -s ../../preprocess/dev.graph_bi dev.graph
ln -s ../../preprocess/tst.graph_bi tst.graph

ln -s ../../preprocess/trn.vocab
```

### Train

```
mkdir models
bash train.sh
```
The `checkpoints` are saved in the `models/` directory

### Predict

```
bash predict.sh [model_path] tst.graph tst.order
```
Given the `tst.graph`, the trained model predict the orders of conditions `tst.order`.

```
ln -s ../../../tst.json
ln -s ../../preprocess/oracle_reorder.py
python3 oracle_reorder.py tst.json tst.order > tst_reorder.json
```

2) Go to the ordering directory to buid condition ordering model.
3) Go to the generation directory to build generation model.
