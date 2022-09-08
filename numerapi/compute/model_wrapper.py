import pandas as pd


# This is the default model wrapper that can be extended for
# adding custom code to your submission pipeline
# TODO: type hints
class ModelWrapper:

    def __init__(self, model, model_id: str):
        self.model = model
        self.model_id = model_id

    def pickle(self, pickle_local_path):
        pd.to_pickle(self.model, pickle_local_path)

    def unpickle(self, pickle_local_path: str):
        self.model = pd.read_pickle(pickle_local_path)

    def pre_predict(self, data):
        pass

    def predict(self, data):
        return self.model.predict(data)

    def post_predict(self, predictions):
        return predictions.rank(pct=True)

