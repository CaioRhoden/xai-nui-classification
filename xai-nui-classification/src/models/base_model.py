class BaseModel:
    def __init__(self, **kwargs):
        self.model = kwargs.get('model', None)
        self.config = kwargs.get('config', None)

    def train(self, train_data):
        raise NotImplementedError("Train method must be implemented in child classes.")

    def evaluate(self, eval_data):
        raise NotImplementedError("Evaluate method must be implemented in child classes.")

    def predict(self, input_data):
        return self.model.predict(input_data)

    def predict_proba(self, input_data):
        return self.model.predict_proba(input_data)