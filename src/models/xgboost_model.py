import sys
sys.path.append('../../src')
import pickle

from models.base_model import BaseModel

class XGBoostModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.threshold = kwargs.get('threshold', 0.5)

    def load_model(self, path: str) -> None:
        # Open the model file in read binary mode
        with open(path, "rb") as file:
            # Load the model from the file
            self.model = pickle.load(file)
            self.config = self.model.best_params_
    




    
    