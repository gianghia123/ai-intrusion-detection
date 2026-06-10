import data
import pickle
from sklearn.metrics import classification_report

from gradient_boosting import TreeWrapper

with open("best_params_tree.pickle", 'rb') as f:
    hyperparams = pickle.load(f)

hyperparams['early_stopping_rounds'] = 0
tree = TreeWrapper(**hyperparams)

tree.load_model('gradient_boost_tree.json')

X_train, y_train, X_val, y_val, sample_weight = data.load_dataset()

with open("./cat_map.pickle", 'rb') as f:
    cat_map = pickle.load(f)

print(classification_report(y_val.get(), tree.predict(X_val), target_names=cat_map))
