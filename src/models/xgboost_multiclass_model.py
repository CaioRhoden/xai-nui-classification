import sys
sys.path.append('../../src')
import pickle
import optuna
from sklearn.metrics import f1_score, roc_auc_score
import numpy as np

from models.base_model import BaseModel

class XGBoostMulticlassModel(BaseModel):

    def __init__(self, **kwargs):
        super().__init__()
        self.thresholds = None

    def load_model(self, path: str) -> None:
        # Open the model file in read binary mode
        with open(path, "rb") as file:
            # Load the model from the file
            self.model = pickle.load(file)
            self.config = self.model.best_params_
        
        return self.model
        
        
    def predict(self, input_data):

        return self.model.predict(input_data)

        




    
    