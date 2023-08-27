class BaseModel:
    def __init__(self, config):
        self.config = config
        self.model = None  # Initialize the model attribute

    def train(self, train_data):
        raise NotImplementedError("Train method must be implemented in child classes.")

    def evaluate(self, eval_data):
        raise NotImplementedError("Evaluate method must be implemented in child classes.")

    def predict(self, input_data):
        raise NotImplementedError("Predict method must be implemented in child classes.")