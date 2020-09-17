from sklearn.impute import SimpleImputer
import pandas as pd
from mlxtend.feature_selection import SequentialFeatureSelector, ExhaustiveFeatureSelector
import constant

def replace_result_values_and_drop_features(hyper_dataset, hypo_dataset):

    hyper_dataset = hyper_dataset.replace(
        {'hyperthyroid.': 'hyperthyroid.', 'T3 toxic.': 'negative.', 'goitre.': 'negative.',
         'secondary toxic.': 'negative.'})

    hypo_dataset = hypo_dataset.replace({'hypothyroid.': 'hypothyroid.', 'primary hypothyroid.': 'hypothyroid.',
                                         'compensated hypothyroid.': 'hypothyroid.',
                                         'secondary hypothyroid.': 'hypothyroid.'})

    hyper_dataset = hyper_dataset.drop(constant.FEATURE_NAMES_TO_DROP, axis=1)
    hypo_dataset = hypo_dataset.drop(constant.FEATURE_NAMES_TO_DROP, axis=1)

    return hyper_dataset, hypo_dataset

def create_dataset_for_training(hyper_dataset, hypo_dataset):

    hyper_dataset, hypo_dataset = replace_result_values_and_drop_features(hyper_dataset,hypo_dataset)


    #reduced_hyper = hyper_dataset.dropna()
    #reduced_hypo = hypo_dataset.dropna()

    #y_hypo = reduced_hypo['result'].copy()
    #y_hyper = reduced_hyper['result'].copy()
    #X = reduced_hyper.drop(['result'], axis=1)

    y_hypo = hypo_dataset['result'].copy()
    y_hyper = hyper_dataset['result'].copy()
    X = hyper_dataset.drop(['result'], axis=1)

    y = y_hyper.combine(y_hypo, lambda x, y: constant.HYPERTHYROID if 'hyperthyroid.' in x else
    (constant.HYPOTHYROID if 'hypothyroid.' in y else constant.NEGATIVE))

    return X, y

def create_dataset_for_evaluation(hyper_dataset, hypo_dataset):

    hyper_dataset, hypo_dataset = replace_result_values_and_drop_features(hyper_dataset, hypo_dataset)

    reduced_hyper = hyper_dataset.dropna()
    reduced_hypo = hypo_dataset.dropna()

    y_hypo = reduced_hypo['result'].copy()
    y_hyper = reduced_hyper['result'].copy()
    X = reduced_hyper.drop(['result'], axis=1)

    y = y_hyper.combine(y_hypo, lambda x, y: 'hyperthyroid' if 'hyperthyroid.' in x else ('hypothyroid' if 'hypothyroid.' in y else 'negative'))

    return X, y

def one_hot_encoding(X):
    for feature_name in X.columns:
        if X[feature_name].dtype == 'object':
            df = pd.get_dummies(X[feature_name]).rename(columns=lambda x: feature_name + ' ' + str(x))
            X = pd.concat([X, df], axis=1)

    # Get list of categorical variables
    s = (X.dtypes == 'object')
    object_cols = list(s[s].index)
    X = X.drop(object_cols, axis=1)
    return X

def imput_missing_data(X):
    imp = SimpleImputer(strategy="mean")
    features_names = X.columns
    X = pd.DataFrame(imp.fit_transform(X))  # imputer usuwa nazwy kolumn
    X.columns = features_names
    return X

def feature_selection(X, y,  model):

    correlated_features = set()
    correlation_matrix = X.corr()
    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > 0.8:
                colname = correlation_matrix.columns[i]
                correlated_features.add(colname)

    X.drop(labels=correlated_features, axis=1, inplace=True)
    feature_selector = SequentialFeatureSelector(model, k_features=11, forward=False, verbose=2, scoring='balanced_accuracy', cv=5)
    #feature_selector = ExhaustiveFeatureSelector(model, min_features=5, max_features=10, scoring='balanced_accuracy', print_progress=True, cv=3)

    features = feature_selector.fit(X, y)
    filtered_features = X.columns[list(features.k_feature_idx_)]

    return filtered_features


def preprocess_the_data(X):
    X = one_hot_encoding(X)
    X = imput_missing_data(X)
    return X