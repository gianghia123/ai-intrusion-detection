#!/bin/bash

python3 src/data_preprocessing/data_preprocessing.py
python3 src/models/label/lgbm_fl/train.py
python3 src/models/label/lgbm_fl/eval.py
python3 src/models/cat/lgbm_gbdt/train.py
python3 src/models/cat/balanced_rf/train.py
python3 src/models/cat/xgboost/train.py
python3 src/models/cat/meta/train.py
python3 src/eval.py
