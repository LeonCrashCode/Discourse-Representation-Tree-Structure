

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

python3 oracle.py trn.json > trn_oracle.json
python3 oracle.py dev.json > dev_oracle.json
python3 oracle.py tst.json > tst_oracle.json

python3 create_graph_bi.py trn_oracle.json > trn.graph_bi
python3 create_graph_bi.py dev_oracle.json > dev.graph_bi
python3 create_graph_bi.py tst_oracle.json > tst.graph_bi

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
The `checkpoints` are saved in the `models/` directory. The best model can be downloaded [here](https://drive.google.com/file/d/1wE7Ul-br5isw26ycXHLuXPLnxdCXho7f/view?usp=sharing)

### Predict

```
bash predict.sh [model_path] tst.graph tst.order_bi
```
Given the `tst.graph`, the trained model predict the orders of conditions `tst.order_bi`.

```
ln -s ../../../tst.json
ln -s ../../preprocess/oracle_reorder.py
python3 oracle_reorder.py tst.json tst.order_bi > tst_oracle_reorder_bi.json
```

## Generation


