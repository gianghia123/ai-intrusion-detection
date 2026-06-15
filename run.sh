#!/bin/bash

# Setting up the environment
if [ -f "./bin/python3" ]; then
    python3 -m venv ./
    ./bin/python3 -m pip install -r ./requirement.txt
fi

echo "Choose your option: "
echo "  1) Training all model"
echo "  2) Evaluate all model"

read -r -p "Your option [1, 2]: " choice

if [[ $choice == "1" ]]; then
    ./bin/python3 ./src/lightgbm/train.py
    ./bin/python3 ./src/lstm/train.py
    ./bin/python3 ./src/svm/train.py
    ./bin/python3 ./src/xgboost/gradient_boosting.py
    ./src/ensemble/train.sh
elif [[ $choice == "2" ]]; then
    ./bin/python3 ./src/lightgbm/eval.py
    ./bin/python3 ./src/lstm/predict.py
    ./bin/python3 ./src/svm/evaluate.py
    ./bin/python3 ./src/xgboost/test_gradient_boosting_tree.py
    ./src/ensemble/eval.sh
fi
