# Data Prepare

```
python == 2.7
editdistance >= 0.3.1
```
The original dataset Parallel Meaning Bank (PMB) can be downloaded in [here](https://pmb.let.rug.nl/data.php). Unzip the `pmb-3.0.0.zip` and move the `pmb-3.0.0/data` to the current directory `pmb/data`.

Prepare for English (en)
```
bash xml2tree_json.sh en bronze
bash xml2tree_json.sh en silver
bash xml2tree_json.sh en gold

bash files.sh en gold > en_gold_files
bash files.sh en bronze > en_bronze_files
bash files.sh en silver > en_silver_files
bash files.sh en test > en_tst_files
bash files.sh en dev > en_dev_files

python2 create.py en_gold_file > en.gold.json
python2 create.py en_bronze_file > en.bronze.json
python2 create.py en_silver_file > en.silver.json
python2 create.py en_tst_file > en.tst.json
python2 create.py en_dev_file > en.dev.json
```

Prepare for low-resource languages (de|it|nl)

```
bash xml2tree_json.sh [de|it|nl] bronze
bash xml2tree_json.sh [de|it|nl] silver
bash xml2tree_json.sh [de|it|nl] gold

bash files.sh [de|it|nl] bronze > [de|it|nl]_bronze_files
bash files.sh [de|it|nl] silver > [de|it|nl]_silver_files
bash files.sh [de|it|nl] test > [de|it|nl]_tst_files
bash files.sh [de|it|nl] dev > [de|it|nl]_dev_files

python2 create.py [de|it|nl]_bronze_file > [de|it|nl].bronze.json
python2 create.py [de|it|nl]_silver_file > [de|it|nl].silver.json
python2 create.py [de|it|nl]_tst_file > [de|it|nl].tst.json
python2 create.py [de|it|nl]_dev_file > [de|it|nl].dev.json
```

Or you can directly download the processed `json` files [3.0.0](https://drive.google.com/drive/folders/1sDCs8f-bZUf1SvDIllzxZoMikpxtwH6c?usp=sharing)
