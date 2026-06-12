import pickle
from pathlib import Path

import pandas as pd
import cupy as cp
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.preprocessing import LabelEncoder
from sklearnex import patch_sklearn
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from tqdm import tqdm

patch_sklearn()


def split_x_y(path):
    df = pd.read_parquet(path)
    return df.drop(['label', 'attack_cat'], axis=1), df['attack_cat']


def gen_fold_training(fold_index):
    result_path = Path('./results')
    train_set = pd.read_parquet(
        result_path / f"fold_{fold_index}_train.parquet")
    test_set = pd.read_parquet(result_path / f"fold_{fold_index}_test.parquet")
    X_train_fold = train_set.drop('attack_cat', axis=1)
    y_train_fold = train_set['attack_cat']
    X_test_fold = test_set.drop('attack_cat', axis=1)
    y_test_fold = test_set['attack_cat']
    sample_weight = compute_sample_weight(
        class_weight='balanced', y=y_train_fold.get())
    return X_train_fold, y_train_fold, X_test_fold, y_test_fold, sample_weight


def save_folds():
    X_train, y_train = split_x_y("../dataset/train.parquet")
    result_path = Path("./results")
    result_path.mkdir(exist_ok=True)
    rskf = RepeatedStratifiedKFold(n_repeats=1, n_splits=10, random_state=42)
    for i, (train_index, test_index) in tqdm(enumerate(rskf.split(X_train, y_train))):
        smote = SMOTE(random_state=42)
        X_train_fold = X_train.iloc[train_index]
        y_train_fold = y_train.iloc[train_index]
        X_test_fold = X_train.iloc[test_index]
        y_test_fold = y_train.iloc[test_index]
        X_train_fold_oversamp, y_train_fold_oversamp = smote.fit_resample(
            X_train_fold, y_train_fold)
        train = pd.concat(
            [X_train_fold_oversamp, y_train_fold_oversamp], axis=1)
        test = pd.concat([X_test_fold, y_test_fold], axis=1)
        train.to_parquet(result_path / f"fold_{i}_train.parquet")
        test.to_parquet(result_path / f"fold_{i}_test.parquet")


def load_dataset():
    X_train, y_train = split_x_y("../dataset/train.parquet")
    X_val, y_val = split_x_y("../dataset/test.parquet")
    sample_weight = compute_sample_weight(class_weight='balanced', y=y_train)
    return X_train, y_train, X_val, y_val, sample_weight


def dataset_with_smote():
    X_train, y_train, X_val, y_val, sample_weight = load_dataset()
    smote = SMOTE(
        sampling_strategy={9: 500, 8: 1200,
                           0: 2000, 1: 2000},
        random_state=42
    )
    smotetomek = SMOTETomek(smote=smote, random_state=42, n_jobs=-1)
    X_train_oversamp, y_train_oversamp = smotetomek.fit_resample(
        X_train, y_train)
    print(y_train_oversamp.unique())
    return X_train_oversamp, y_train_oversamp, X_val, y_val, sample_weight


if __name__ == '__main__':
    cat_map = Path("cat_map.pickle")
    if not cat_map.exists():
        train = pd.read_parquet("../dataset/train.parquet")
        test = pd.read_parquet("../dataset/test.parquet")
        enc = LabelEncoder()
        train['attack_cat'] = enc.fit_transform(train['attack_cat'])
        test['attack_cat'] = enc.transform(test['attack_cat'])
        train.to_parquet("../dataset/train.parquet")
        test.to_parquet("../dataset/test.parquet")
        with open('cat_map.pickle', 'wb') as f:
            pickle.dump(enc.classes_, f)
    save_folds()
