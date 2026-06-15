#!/bin/bash

../../bin/python3 src/data_preprocessing/data_preprocessing.py
../../bin/python3 src/models/label/lgbm_fl/train.py
../../bin/python3 src/models/cat/lgbm_gbdt/train.py
../../bin/ython3 src/models/cat/balanced_rf/train.py
../../bin/python3 src/models/cat/xgboost/train.py
../../bin/python3 src/models/cat/meta/train.py
