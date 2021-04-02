# Data Prepare

```
python == 2.7
editdistance >= 0.3.1
```
The original dataset Parallel Meaning Bank (PMB) can be downloaded in [here](https://pmb.let.rug.nl/data.php). Unzip the `pmb-3.0.0.zip` and move the `pmb-3.0.0/data` to the current directory `pmb/data`.

Prepare for English (en) and German (de)
```
bash xml2tree_json.sh [en|de] bronze
bash xml2tree_json.sh [en|de] silver
bash xml2tree_json.sh [en|de] gold

bash files.sh [en|de] gold > [en|de]_gold_files
bash files.sh [en|de] bronze > [en|de]_bronze_files
bash files.sh [en|de] silver > [en|de]_silver_files
bash files.sh [en|de] test > [en|de]_tst_files
bash files.sh [en|de] dev > [en|de]_dev_files

python2 create.py [en|de]_gold_file > [en|de].gold.json
python2 create.py [en|de]_bronze_file > [en|de].bronze.json
python2 create.py [en|de]_silver_file > [en|de].silver.json
python2 create.py [en|de]_tst_file > [en|de].tst.json
python2 create.py [en|de]_dev_file > [en|de].dev.json
```

Prepare for low-resource languages, including Italian (it) and Dutch (nl)

```
bash xml2tree_json.sh [it|nl] bronze
bash xml2tree_json.sh [it|nl] silver
bash xml2tree_json.sh [it|nl] gold

bash files.sh [it|nl] bronze > [it|nl]_bronze_files
bash files.sh [it|nl] silver > [it|nl]_silver_files
bash files.sh [it|nl] test > [it|nl]_tst_files
bash files.sh [it|nl] dev > [it|nl]_dev_files

python2 create.py [it|nl]_bronze_file > [it|nl].bronze.json
python2 create.py [it|nl]_silver_file > [it|nl].silver.json
python2 create.py [it|nl]_tst_file > [it|nl].tst.json
python2 create.py [it|nl]_dev_file > [it|nl].dev.json
```

Or you can directly download the processed `json` files [3.0.0](https://drive.google.com/drive/folders/1-sZjis2SLnZ6hYuq0giguE2HOU36NAIH?usp=sharing)
