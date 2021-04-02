

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
The `checkpoints` are saved in the `models/` directory. The model can be downloaded [here](https://drive.google.com/file/d/1wE7Ul-br5isw26ycXHLuXPLnxdCXho7f/view?usp=sharing)

### Predict

```
bash predict.sh [model_path] dev.graph dev.order_bi
bash predict.sh [model_path] tst.graph tst.order_bi
```
Given the `tst.graph`, the trained model predict the orders of conditions `tst.order_bi`.

```
ln -s ../../../tst.json
ln -s ../../preprocess/oracle_reorder.py

python3 oracle_reorder.py dev.json dev.order_bi > dev_oracle_reorder_bi.json
python3 oracle_reorder.py tst.json tst.order_bi > tst_oracle_reorder_bi.json
```

## Generation

Generation codes are modified based on the [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py), ad you have to install the local onmt

```
cd generation
cd onmt
pip install .
```

Prepare the data for onmt for ideal world generation
```
cd generation/workspace/data
```
```
ln -s ../../../preprocess/trn.oracle.json trn
ln -s ../../../preprocess/dev.oracle.json dev
ln -s ../../../preprocess/tst.oracle.json tst
```
or real world generation
```
ln -s ../../../preprocess/trn.oracle.json trn
ln -s ../../../ordering/workspace/dev_oracle_reorder_bi.json dev
ln -s ../../../ordering/workspace/tst_oracle_reorder_bi.json tst
```
```
python3 getsrc.py [trn|dev|tst] > [trn|dev|tst].src
python3 gettgt.py [trn|dev|tst] > [trn|dev|tst].tgt
python3 gettree.py [trn|dev|tst] > [trn|dev|tst].tree

cd ..
bash preprocess.sh
```

### Train
```
bash train.sh
```
The `checkpoints` are saved in the `models` directory. The model can be downloaded [here](https://drive.google.com/file/d/1i3XunGzOecsBpKnHFoTf3ZZUXcACOgsm/view?usp=sharing)

### Predict
```
bash predict.sh [model_path] data/tst.src data/tst.tree data/tst.output [beam_size]
```

