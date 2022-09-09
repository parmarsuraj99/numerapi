import pandas as pd
from numerapi.compute.model_wrapper import ModelWrapper


class CustomModel(ModelWrapper):

    def __init__(self, model, model_id):
        super().__init__(model, model_id)

    def pickle(self, pickle_local_path):
        print('in custom pickle')
        pd.to_pickle(self.model, pickle_local_path)

    def unpickle(self, pickle_local_path):
        print('in custom unpickle')
        self.model = pd.read_pickle(pickle_local_path)

    def predict(self, data):
        print('in custom predict')
        return self.model.predict(data)

    def post_predict(self, predictions, round_number):
        print('in custom post_predict')
        return predictions.rank(pct=True)