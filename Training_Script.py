from DBReader import DbReader
from Model import RandomForestModel
from Preprocessing import preprocess_the_data, create_dataset_for_training, feature_selection
import pickle

def train_script():

    db_reader = DbReader()
    hyper_dataset = db_reader.load_csv("../allhyper.data")
    hypo_dataset = db_reader.load_csv("../allhypo.data")
    X, y = create_dataset_for_training(hyper_dataset, hypo_dataset)
    X = preprocess_the_data(X)

    rf_model = RandomForestModel()
    filtered_features = feature_selection(X, y, rf_model.internal_model)

    with open('selected_best_features.data', 'wb') as filehandle:
        pickle.dump(filtered_features,filehandle)

    model_to_fit = rf_model.gridsearchCV()
    model_to_fit.fit(X[filtered_features], y)
    print(model_to_fit.best_score_)
    print(model_to_fit.best_params_)
    print(filtered_features)
    rf_model.set_internal_model(model_to_fit.best_estimator_)
    rf_model.save_model()

if __name__ == "__main__":
    train_script()
