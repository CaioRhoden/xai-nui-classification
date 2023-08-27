import sys
sys.path.append('../../src')
from xgboost import XGBClassifier


from models.base_model import BaseModel

class XGBoostModel(BaseModel):
    
    def __init__(self, config):
        self.config = config
        self.model = XGBClassifier(**config)
        self._train = None
        self._val = None
        self._test = None
    

    def set_datasets(self, train, val, test):
        """
        Set the datasets for training, validation, and testing.

        Parameters:
            train (type): The training dataset.
            val (type): The validation dataset.
            test (type): The testing dataset.
        """
        self._train, self._val, self._test = train, val, test
    
    def train(self):
        


    