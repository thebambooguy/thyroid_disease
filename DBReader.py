import pandas as pd
import regex as re

class DbReader:
    def __init__(self):
        self.data_from_csv = None

    def load_csv(self,filename):

        features_names = []
        with open("../allhyper.names", 'r') as filehandle:
            for line in filehandle:
                searchObj = re.search("(.*):", line)
                if searchObj:
                    features_names.append(searchObj.group(0).replace(":", ""))

        features_names.append('result')

        self.data_from_csv = pd.read_csv(filename, sep=',|\|', engine='python', names=features_names, index_col=-1, na_values = '?')

        return self.data_from_csv