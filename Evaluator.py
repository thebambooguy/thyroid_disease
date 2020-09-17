from sklearn.metrics import classification_report
from Model import RandomForestModel
from DBReader import DbReader
from Preprocessing import create_dataset_for_evaluation, preprocess_the_data, create_dataset_for_training
import pickle

def evaluating_script():

    db_reader = DbReader()
    hyper_dataset = db_reader.load_csv("../allhyper.test")
    hypo_dataset = db_reader.load_csv("../allhypo.test")
    X, y = create_dataset_for_evaluation(hyper_dataset, hypo_dataset)
    X = preprocess_the_data(X)

    rf_model = RandomForestModel()
    load_model = rf_model.load_model()

    with open('selected_best_features.data','rb') as filehandle:
        filtered_features = pickle.load(filehandle)

    predicted_values = load_model.predict(X[filtered_features])
    print(rf_model.__class__.__name__)
    print(classification_report(y, predicted_values))

if __name__ == "__main__":
    evaluating_script()