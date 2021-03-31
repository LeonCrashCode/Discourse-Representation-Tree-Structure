

# Data Prepare

The original dataset Groningen Meaning Bank (GMB) v2.2.0 can be downloaded in [here](https://gmb.let.rug.nl/data.php). Unzip the `gmb-2.2.0.zip` and move the `gmb-2.2.0/data` to the current directory `gmb/data`.

```
bash xml2tree_json.sh
python2 create.py dev_files > dev.json
python2 create.py tst_files > tst.json
python2 create.py trn_files > trn.json
```

Or you can directly download the processed files [trn.json](https://drive.google.com/file/d/14_7bjTliGVH-MKGRWuPL5NpeLgHnq3Qk/view?usp=sharing), [dev.json](https://drive.google.com/file/d/1Z-PwdwwUmua_acqYj_U6kJTH5pzB6u6h/view?usp=sharing), and [tst.json](https://drive.google.com/file/d/1QoExYcZPjnBgNc0ybwMYsjPCcEExA6Kp/view?usp=sharing).
