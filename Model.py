from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from abc import ABC

class Model(ABC):
    def __init__(self):
        self.model_filename = None
        self.internal_model = None
        self.parameters = None
        self.cv = None
        self.scoring = None

    def fit(self, X_train , y_train):
        self.internal_model.fit(X_train, y_train)

    def predict(self, X_train):
        temp = self.internal_model.predict(X_train)
        return temp

    def gridsearchCV(self):
        best_model = GridSearchCV(self.internal_model,self.parameters,cv= self.cv, scoring=self.scoring)
        return best_model

    def save_model(self):
        dump(self, self.model_filename)

    def load_model(self):
        return load(self.model_filename)

    def set_internal_model(self, external_model):
        self.internal_model = external_model

class RandomForestModel(Model):
    def __init__(self, n_estimators=5, max_depth=None, max_features="auto", max_leaf_nodes=None, random_state=1):
        self.internal_model = RandomForestClassifier(n_estimators = n_estimators,max_depth=max_depth,max_features=max_features,max_leaf_nodes=max_leaf_nodes,random_state=random_state,class_weight='balanced')
        self.model_filename = "RandomForestModelparameters.joblib"

        self.parameters = {'n_estimators': [10, 25, 50, 100, 150, 200], 'max_depth': [None, 2, 5, 10, 15], 'max_leaf_nodes': [None, 2, 5, 10, 15]}
        self.cv = 10
        self.scoring = 'balanced_accuracy'

