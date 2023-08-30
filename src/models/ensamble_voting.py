class EnsambleVoting:
    def __init__(self, config, estimators_address):
        self.config = config
        self.estimators = None
        self.model = None
        self.estimators_address = estimators_address
    

    def _get_estimators(self):

        


    def train(self, train_data):
        raise NotImplementedError("Train method must be implemented in child classes.")

    def evaluate(self, eval_data):
        raise NotImplementedError("Evaluate method must be implemented in child classes.")

    def predict(self, input_data):
        raise NotImplementedError("Predict method must be implemented in child classes.")