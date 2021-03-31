

# Data Prepare

```
ln -s ../../trn.json
ln -s ../../dev.json
ln -s ../../tst.json

python3 create_graph_bi.py trn.json > trn.graph_bi
python3 create_graph_bi.py dev.json > dev.graph_bi
python3 create_graph_bi.py tst.json > tst.graph_bi
```
